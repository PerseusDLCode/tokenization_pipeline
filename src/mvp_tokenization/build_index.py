#!/usr/bin/env python3
"""Build the corpus search index from token sidecars.

The index is a single SQLite file meant to be queried from the browser over
HTTP range requests (sql.js-httpvfs, the same way pdl-morph-server serves
morph.db), so its layout is chosen for range reads rather than for SQLite
convenience:

- 4096-byte pages (sql.js-httpvfs's requestChunkSize must match), rollback
  journal (not WAL), VACUUMed.
- Every posting table is WITHOUT ROWID and clustered on the key it's queried
  by, so a query's rows sit in contiguous pages instead of being scattered
  across the file -- the browser fetches a few small ranges per page of
  hits, never the file.
- Postings hold only integers. A token's citable URN is rebuilt from them as
  ``{base_urn}:{ref}@{form}[{occ}]``, which is exactly the URN the tokenizer
  wrote into the sidecar and that MinimumViablePerseus renders as
  ``data-token-urn`` -- both read the same sidecars.
- chunk ids are assigned in document order, then citation order (from the
  copied index.json), so ``documents.first_chunk``/``last_chunk`` bound each
  document's postings and hits come back in reading order.
- Context for a hit (keyword-in-context display) comes from
  ``chunk_context``: each chunk's token texts in windows of CONTEXT_PART
  tokens, so showing the words around a hit reads one or two small rows --
  not a lookup per neighboring word, and not a whole chunk (some run to
  megabytes). Within a window, tokens are separated by TOKEN_SEP, and a
  token followed by whitespace ends with a space; window ``part`` holds
  tokens ``part * CONTEXT_PART`` up to the next window's first.
- The key indexes on ``forms`` and ``lemmas`` carry every column the
  browser reads, so resolving a key never needs a second seek into the
  table.
- Dictionary-headword search ("every form of λύω") is precomputed from
  pdl-morph-server's morph.db (``--morph-db``): ``morph_lemmas`` holds the
  Morpheus headwords with at least one form in the corpus, and
  ``morph_lemma_forms`` maps each to the corpus forms whose match key equals
  a form Morpheus generates for it. A Morpheus lemma is identified by its
  natural key, (headword, sequence_number, lang), never by morph.db's row
  ids, which change on every rebuild of that database. Without
  ``--morph-db`` these tables are empty and the page falls back to tagged
  lemmas. ``morph_analyses`` holds Morpheus's analyses of each of those
  forms under each of its headwords, clustered by form, so a search can show
  what else a form might be.
- Morpheus generates forms that are far likelier to belong to some other
  headword (μανίας for μῆνις, by way of Doric μᾶνις), so where a form has
  more than one Morpheus headword, the tagger picks between them:
  ``form_postings.morph_id`` and ``form_doc_counts.morph_id`` are the
  headword the tagger's lemma names, or, failing that, the only one whose
  part of speech its UPOS fits (noun-like vs verb-like; see
  ``_resolve_morph``). 0 means the tagger didn't settle it. A headword search
  hides hits whose ``morph_id`` is some other headword.

Only the default citation scheme is indexed (sidecars directly in a version
dir, not in its scheme subdirectories), since alternate schemes re-chunk the
same text and would double every hit.

    uv run mvp-index --tokens-dir ../tokens --morph-db ../morph.db --out ./search-index

The output dir gets ``search-<sha12>.db`` (content-addressed, so it can be
served as immutable) and ``manifest.json`` pointing at it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import zstandard

from mvp_tokenization.normalize import is_apostrophe, loose_key, match_key

SCHEMA_VERSION = 3
PAGE_SIZE = 4096
DEFAULT_LANGS = ("grc", "la")
SIDECAR_SUFFIX = ".tokens.json.zst"
TOKEN_SEP = "\x1f"
CONTEXT_PART = 32

_IDENTIFIER_OCC = re.compile(r"\[(\d+)\]$")
_LANG_ALIASES = {"lat": "la"}
# The index uses MVP's language codes; morph.db uses ISO 639-2 for Latin.
_MORPH_LANGS = {"grc": "grc", "la": "lat"}

# The only part-of-speech distinction trusted to pick between Morpheus
# headwords: noun-like vs verb-like. Finer ones (adverb vs particle vs
# conjunction, say) are labelled too differently by Morpheus and UD to
# compare. A participle is both.
_UPOS_CLASSES = {
    **dict.fromkeys(("NOUN", "PROPN", "ADJ", "PRON", "DET", "NUM"), "nominal"),
    **dict.fromkeys(("VERB", "AUX"), "verbal"),
}
_MORPH_POS_CLASSES = {
    **dict.fromkeys(("noun", "adj", "pron", "article", "numeral"), {"nominal"}),
    "verb": {"verbal"},
    "part": {"nominal", "verbal"},
}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID;

CREATE TABLE documents (
  doc_id INTEGER PRIMARY KEY,
  base_urn TEXT NOT NULL,
  lang TEXT NOT NULL,
  title TEXT,
  author TEXT,
  corpus TEXT,
  first_chunk INTEGER NOT NULL,
  last_chunk INTEGER NOT NULL,
  n_words INTEGER NOT NULL
);

CREATE TABLE chunks (
  chunk_id INTEGER PRIMARY KEY,
  doc_id INTEGER NOT NULL,
  ref TEXT NOT NULL
);

CREATE TABLE chunk_context (
  chunk_id INTEGER NOT NULL,
  part INTEGER NOT NULL,
  tokens TEXT NOT NULL,
  PRIMARY KEY (chunk_id, part)
) WITHOUT ROWID;

-- A form is a distinct (surface text, match key) pair: an elided form like
-- the "μυρί" of "μυρί’" keys as "μυρί'" and so is a different form from an
-- unelided "μυρί".
CREATE TABLE forms (
  form_id INTEGER PRIMARY KEY,
  form TEXT NOT NULL,
  key TEXT NOT NULL,
  loose TEXT NOT NULL,
  lang TEXT NOT NULL,
  n INTEGER NOT NULL
);

CREATE TABLE lemmas (
  lemma_id INTEGER PRIMARY KEY,
  lemma TEXT NOT NULL,
  key TEXT NOT NULL,
  loose TEXT NOT NULL,
  lang TEXT NOT NULL,
  n INTEGER NOT NULL
);

CREATE TABLE feats (
  feat_id INTEGER PRIMARY KEY,
  upos TEXT,
  feats TEXT
);

-- Words only (punctuation has no URN and isn't searchable).
CREATE TABLE staging (
  doc_id INTEGER NOT NULL,
  chunk_id INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  form_id INTEGER NOT NULL,
  occ INTEGER NOT NULL,
  lemma_id INTEGER,
  feat_id INTEGER,
  morph_id INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX forms_key ON forms (key, lang, form_id, n, form);
"""

CLUSTERED = """
CREATE TABLE form_postings (
  form_id INTEGER NOT NULL,
  chunk_id INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  occ INTEGER NOT NULL,
  lemma_id INTEGER,
  feat_id INTEGER,
  morph_id INTEGER NOT NULL,
  PRIMARY KEY (form_id, chunk_id, seq)
) WITHOUT ROWID;
INSERT INTO form_postings
  SELECT form_id, chunk_id, seq, occ, lemma_id, feat_id, morph_id FROM staging
  ORDER BY form_id, chunk_id, seq;

CREATE TABLE lemma_postings (
  lemma_id INTEGER NOT NULL,
  chunk_id INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  form_id INTEGER NOT NULL,
  occ INTEGER NOT NULL,
  feat_id INTEGER,
  PRIMARY KEY (lemma_id, chunk_id, seq)
) WITHOUT ROWID;
INSERT INTO lemma_postings
  SELECT lemma_id, chunk_id, seq, form_id, occ, feat_id FROM staging
  WHERE lemma_id IS NOT NULL
  ORDER BY lemma_id, chunk_id, seq;

-- One row per headword the tagger chose for the form in the document
-- (usually just morph_id 0), so a headword search can count what it shows.
CREATE TABLE form_doc_counts (
  form_id INTEGER NOT NULL,
  doc_id INTEGER NOT NULL,
  morph_id INTEGER NOT NULL,
  n INTEGER NOT NULL,
  PRIMARY KEY (form_id, doc_id, morph_id)
) WITHOUT ROWID;
INSERT INTO form_doc_counts
  SELECT form_id, doc_id, morph_id, COUNT(*) FROM staging
  GROUP BY form_id, doc_id, morph_id ORDER BY form_id, doc_id, morph_id;

CREATE TABLE lemma_doc_counts (
  lemma_id INTEGER NOT NULL,
  doc_id INTEGER NOT NULL,
  n INTEGER NOT NULL,
  PRIMARY KEY (lemma_id, doc_id)
) WITHOUT ROWID;
INSERT INTO lemma_doc_counts
  SELECT lemma_id, doc_id, COUNT(*) FROM staging
  WHERE lemma_id IS NOT NULL
  GROUP BY lemma_id, doc_id ORDER BY lemma_id, doc_id;

DROP TABLE staging;

CREATE INDEX forms_loose ON forms (loose, lang, form_id, n, form);
CREATE INDEX lemmas_key ON lemmas (key, lang, lemma_id, n, lemma);
CREATE INDEX lemmas_loose ON lemmas (loose, lang, lemma_id, n, lemma);
CREATE INDEX documents_base_urn ON documents (base_urn);
"""

MORPH_SCHEMA = """
CREATE TABLE morph_lemmas (
  morph_id INTEGER PRIMARY KEY,
  headword TEXT NOT NULL,
  sequence_number INTEGER NOT NULL,
  loose TEXT NOT NULL,
  lang TEXT NOT NULL,
  n INTEGER NOT NULL
);

CREATE TABLE morph_lemma_forms (
  morph_id INTEGER NOT NULL,
  form_id INTEGER NOT NULL,
  PRIMARY KEY (morph_id, form_id)
) WITHOUT ROWID;

-- Every headword Morpheus gives a corpus form, with its analyses of the
-- form under that headword ("noun acc pl fem (doric); ..."), so a search
-- can show what else a form could be. Clustered by form.
CREATE TABLE morph_analyses (
  form_id INTEGER NOT NULL,
  morph_id INTEGER NOT NULL,
  analyses TEXT NOT NULL,
  PRIMARY KEY (form_id, morph_id)
) WITHOUT ROWID;
"""

# Runs with morph.db attached as `morph` and temp.morph_langs mapping its
# language codes to ours. match_key, loose_key and morph_analysis are
# registered as SQL functions.
MORPH_FILL = """
CREATE TEMP TABLE morph_keys AS
  SELECT DISTINCT
    p.lemma_id AS src_id, match_key(p.form, ml.lang) AS key, ml.lang,
    p.part_of_speech AS pos,
    morph_analysis(
      p.part_of_speech, p.person, p.number, p.tense, p.mood, p.voice,
      p.grammatical_case, p.gender, p.degree, p.dialect
    ) AS analysis
  FROM morph.parses p
  JOIN morph.lemmas l ON l.id = p.lemma_id
  JOIN temp.morph_langs ml ON ml.code = l.language_code;

-- One row per analysis Morpheus gives the form under the lemma.
CREATE TEMP TABLE morph_hit_parses AS
  SELECT DISTINCT k.src_id, f.form_id, k.pos, k.analysis
  FROM temp.morph_keys k JOIN forms f ON f.key = k.key AND f.lang = k.lang;

CREATE TEMP TABLE morph_hits AS
  SELECT DISTINCT src_id, form_id FROM temp.morph_hit_parses;

-- n: corpus words with any of the lemma's forms (ambiguous forms count
-- toward every lemma that can generate them).
CREATE TEMP TABLE morph_found AS
  SELECT
    ROW_NUMBER() OVER (ORDER BY ml.lang, l.headword, l.sequence_number) AS morph_id,
    s.src_id, l.headword, l.sequence_number, ml.lang, s.n
  FROM (
    SELECT h.src_id, SUM(f.n) AS n
    FROM temp.morph_hits h JOIN forms f ON f.form_id = h.form_id
    GROUP BY h.src_id
  ) s
  JOIN morph.lemmas l ON l.id = s.src_id
  JOIN temp.morph_langs ml ON ml.code = l.language_code;

INSERT INTO morph_lemmas
  SELECT morph_id, headword, sequence_number, loose_key(headword, lang), lang, n
  FROM temp.morph_found ORDER BY morph_id;

INSERT INTO morph_lemma_forms
  SELECT m.morph_id, h.form_id
  FROM temp.morph_found m JOIN temp.morph_hits h ON h.src_id = m.src_id
  ORDER BY m.morph_id, h.form_id;

INSERT INTO morph_analyses
  SELECT form_id, morph_id, group_concat(analysis, '; ')
  FROM (
    SELECT DISTINCT h.form_id, m.morph_id, h.analysis
    FROM temp.morph_found m JOIN temp.morph_hit_parses h ON h.src_id = m.src_id
    ORDER BY h.form_id, m.morph_id, h.analysis
  )
  GROUP BY form_id, morph_id ORDER BY form_id, morph_id;

-- The candidates the tagger has to pick between: forms with more than one
-- headword, each headword with its parts of speech for the form.
CREATE TEMP TABLE morph_candidates AS
  SELECT m.morph_id, h.form_id, m.headword, m.lang, group_concat(DISTINCT h.pos) AS pos
  FROM temp.morph_found m JOIN temp.morph_hit_parses h ON h.src_id = m.src_id
  WHERE h.form_id IN (
    SELECT form_id FROM temp.morph_hits GROUP BY form_id HAVING COUNT(*) > 1
  )
  GROUP BY m.morph_id, h.form_id;

DROP TABLE temp.morph_keys;
DROP TABLE temp.morph_hit_parses;
DROP TABLE temp.morph_hits;
DROP TABLE temp.morph_found;
"""

MORPH_INDEXES = """
CREATE INDEX morph_lemmas_loose
  ON morph_lemmas (loose, lang, morph_id, headword, sequence_number, n);
CREATE INDEX morph_lemmas_headword
  ON morph_lemmas (headword, lang, sequence_number, morph_id, n);
"""


def _natural_key(name: str) -> list:
    stem = name.removesuffix(SIDECAR_SUFFIX)
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in stem.split(".")]


def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _ordered_sidecars(version_dir: Path) -> list[Path]:
    """Sidecars in citation order: index.json's chunk order when the tokens
    tree carries a copy of it, else a natural sort of the chunk filenames."""
    sidecars = {
        p.name: p
        for p in version_dir.glob(f"*{SIDECAR_SUFFIX}")
        if not p.name.startswith(".")
    }
    index = _read_json(version_dir / "index.json")
    if index:
        ordered = []
        for entry in index.get("chunks", []):
            name = Path(entry["file"]).with_suffix("").name + SIDECAR_SUFFIX
            if name in sidecars:
                ordered.append(sidecars.pop(name))
        # Anything index.json doesn't list goes last rather than vanishing.
        return ordered + sorted(sidecars.values(), key=lambda p: _natural_key(p.name))
    return sorted(sidecars.values(), key=lambda p: _natural_key(p.name))


def _iter_version_dirs(tokens_root: Path):
    """Default-scheme version dirs: <namespace>/<textgroup>/<work>/<version>."""
    for version_dir in sorted(tokens_root.glob("*/*/*/*")):
        if version_dir.is_dir() and any(version_dir.glob(f"*{SIDECAR_SUFFIX}")):
            yield version_dir


class _Interner:
    def __init__(self):
        self.ids: dict[tuple, int] = {}

    def get(self, key: tuple) -> int:
        id_ = self.ids.get(key)
        if id_ is None:
            id_ = self.ids[key] = len(self.ids) + 1
        return id_


def _morph_analysis(pos, *features_and_dialect) -> str:
    """One Morpheus parse as text, in Morpheus's own order: "verb 3rd sg
    imperf ind act (homeric ionic)"."""
    *features, dialect = features_and_dialect
    text = " ".join(f for f in (pos, *features) if f)
    return f"{text} ({dialect})" if dialect else text


def _add_morph_lemmas(
    conn: sqlite3.Connection, morph_db: Path | None, langs: tuple[str, ...]
) -> int:
    """Fill morph_lemmas, morph_lemma_forms and morph_analyses from morph.db
    (left empty without one), and resolve ambiguous forms in staging. Needs
    forms and its key index. Returns the lemma count."""
    conn.executescript(MORPH_SCHEMA)
    if morph_db is not None:
        if not morph_db.is_file():
            raise FileNotFoundError(morph_db)
        conn.create_function("match_key", 2, match_key, deterministic=True)
        conn.create_function("loose_key", 2, loose_key, deterministic=True)
        conn.create_function("morph_analysis", 10, _morph_analysis, deterministic=True)
        conn.execute("ATTACH DATABASE ? AS morph", (str(morph_db),))
        conn.execute("CREATE TEMP TABLE morph_langs (code TEXT PRIMARY KEY, lang TEXT)")
        conn.executemany(
            "INSERT INTO temp.morph_langs VALUES (?,?)",
            [(_MORPH_LANGS[lang], lang) for lang in langs if lang in _MORPH_LANGS],
        )
        conn.executescript(MORPH_FILL)
        conn.execute("DROP TABLE temp.morph_langs")
        conn.commit()
        conn.execute("DETACH DATABASE morph")
        _resolve_morph(conn)
    conn.executescript(MORPH_INDEXES)
    return conn.execute("SELECT COUNT(*) FROM morph_lemmas").fetchone()[0]


def _resolve_morph(conn: sqlite3.Connection) -> None:
    """Set staging.morph_id for words whose form has more than one Morpheus
    headword: the one the tagger chose, or 0 if it didn't settle it.

    The tagger's lemma decides when it names one of the candidates (matched
    by key, else ignoring accents, so its μήνις still names μῆνις). When it
    names none -- its lemmas are often just the surface form -- its UPOS
    decides if exactly one candidate fits it, so a VERB μήνιε goes to μηνίω,
    not to the noun μῆνις. Anything else stays 0 and shows under every
    candidate."""
    candidates: dict[int, list[tuple[int, str, str, set[str] | None]]] = {}
    for morph_id, form_id, headword, lang, pos in conn.execute(
        "SELECT morph_id, form_id, headword, lang, pos FROM temp.morph_candidates"
    ):
        classes: set[str] | None = set()
        for p in (pos or "").split(","):
            if p not in _MORPH_POS_CLASSES:
                classes = None  # a part of speech that fits any UPOS
                break
            classes |= _MORPH_POS_CLASSES[p]
        candidates.setdefault(form_id, []).append(
            (morph_id, match_key(headword, lang), loose_key(headword, lang), classes)
        )
    lemma_keys = {
        lemma_id: (key, loose)
        for lemma_id, key, loose in conn.execute("SELECT lemma_id, key, loose FROM lemmas")
    }
    upos_classes = {
        feat_id: _UPOS_CLASSES.get(upos)
        for feat_id, upos in conn.execute("SELECT feat_id, upos FROM feats")
    }

    def fits(candidate, upos_class):
        classes = candidate[3]
        return upos_class is None or classes is None or upos_class in classes

    def resolve(form_id, lemma_id, feat_id):
        cands = candidates[form_id]
        upos_class = upos_classes.get(feat_id)
        if lemma_id in lemma_keys:
            key, loose = lemma_keys[lemma_id]
            named = [c for c in cands if c[1] == key] or [c for c in cands if c[2] == loose]
            if named:
                # Homographs (two headwords, one spelling): the UPOS may still
                # tell them apart.
                cands = [c for c in named if fits(c, upos_class)] or named
                return cands[0][0] if len(cands) == 1 else 0
        cands = [c for c in cands if fits(c, upos_class)]
        return cands[0][0] if len(cands) == 1 else 0

    conn.execute(
        "CREATE TEMP TABLE morph_resolved ("
        " form_id INTEGER, lemma_id INTEGER, feat_id INTEGER, morph_id INTEGER,"
        " PRIMARY KEY (form_id, lemma_id, feat_id)) WITHOUT ROWID"
    )
    combos = conn.execute(
        """
        SELECT DISTINCT form_id, IFNULL(lemma_id, 0), IFNULL(feat_id, 0) FROM staging
        WHERE form_id IN (SELECT form_id FROM temp.morph_candidates)
        """
    ).fetchall()
    conn.executemany(
        "INSERT INTO temp.morph_resolved VALUES (?,?,?,?)",
        (
            (form_id, lemma_id, feat_id, morph_id)
            for form_id, lemma_id, feat_id in combos
            if (morph_id := resolve(form_id, lemma_id, feat_id))
        ),
    )
    # A correlated lookup rather than UPDATE ... FROM, so that staging (which
    # has no indexes) is scanned once.
    conn.execute(
        """
        UPDATE staging SET morph_id = IFNULL((
          SELECT r.morph_id FROM temp.morph_resolved r
          WHERE r.form_id = staging.form_id
            AND r.lemma_id = IFNULL(staging.lemma_id, 0)
            AND r.feat_id = IFNULL(staging.feat_id, 0)
        ), 0)
        WHERE form_id IN (SELECT form_id FROM temp.morph_resolved)
        """
    )
    conn.execute("DROP TABLE temp.morph_resolved")
    conn.execute("DROP TABLE temp.morph_candidates")
    conn.commit()


def build_index(
    tokens_roots: list[Path],
    out_path: Path,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    morph_db: Path | None = None,
) -> dict:
    """Build the index at out_path. Returns summary stats."""
    out_path.unlink(missing_ok=True)
    conn = sqlite3.connect(out_path)
    conn.execute(f"PRAGMA page_size = {PAGE_SIZE}")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -1000000")
    conn.executescript(SCHEMA)

    decompressor = zstandard.ZstdDecompressor()
    forms = _Interner()  # (form, key, loose, lang)
    lemmas = _Interner()  # (lemma, key, loose, lang)
    feats = _Interner()  # (upos, feats)
    form_counts: Counter[int] = Counter()
    lemma_counts: Counter[int] = Counter()

    doc_id = chunk_id = 0
    n_words = 0
    skipped_chunks = 0
    started = time.time()

    for tokens_root in tokens_roots:
        for version_dir in _iter_version_dirs(tokens_root):
            metadata = (_read_json(version_dir / "metadata.json") or {}).get(
                "document", {}
            )
            doc_rows: list[tuple] = []
            chunk_rows: list[tuple] = []
            text_rows: list[tuple] = []
            doc_words = 0
            base_urn = metadata.get("base_urn")
            doc_lang = None
            next_doc_id = doc_id + 1

            for sidecar in _ordered_sidecars(version_dir):
                data = json.loads(decompressor.decompress(sidecar.read_bytes()))
                lang = _LANG_ALIASES.get(data.get("lang", ""), data.get("lang", ""))
                if lang not in langs:
                    skipped_chunks += 1
                    continue
                urn = data["urn"]
                chunk_base, _, ref = urn.rpartition(":")
                base_urn = base_urn or chunk_base
                if chunk_base != base_urn:
                    print(
                        f"  skipping {sidecar}: {urn} not in {base_urn}",
                        file=sys.stderr,
                    )
                    skipped_chunks += 1
                    continue
                doc_lang = doc_lang or lang
                chunk_id += 1
                chunk_rows.append((chunk_id, next_doc_id, ref))

                tokens = data.get("tokens", [])
                texts = [
                    t["text"] + (" " if t.get("whitespace") else "") for t in tokens
                ]
                text_rows.extend(
                    (
                        chunk_id,
                        i // CONTEXT_PART,
                        TOKEN_SEP.join(texts[i : i + CONTEXT_PART]),
                    )
                    for i in range(0, len(texts), CONTEXT_PART)
                )
                for seq, token in enumerate(tokens):
                    text = token["text"]
                    token_urn = token.get("urn")
                    if not token_urn:
                        continue
                    key_text = text
                    if (
                        not token.get("whitespace")
                        and seq + 1 < len(tokens)
                        and is_apostrophe(tokens[seq + 1]["text"])
                    ):
                        key_text = text + "'"
                    form_id = forms.get(
                        (
                            text,
                            match_key(key_text, lang),
                            loose_key(key_text, lang),
                            lang,
                        )
                    )

                    lemma_id = feat_id = None
                    m = _IDENTIFIER_OCC.search(token.get("identifier", ""))
                    occ = int(m.group(1)) if m else 1
                    word = (token.get("words") or [{}])[0]
                    lemma = word.get("lemma")
                    if lemma and not is_apostrophe(lemma):
                        lemma_id = lemmas.get(
                            (
                                lemma,
                                match_key(lemma, lang),
                                loose_key(lemma, lang),
                                lang,
                            )
                        )
                        lemma_counts[lemma_id] += 1
                    if word.get("upos") or word.get("feats"):
                        feat_id = feats.get((word.get("upos"), word.get("feats")))
                    form_counts[form_id] += 1
                    doc_words += 1
                    doc_rows.append(
                        (next_doc_id, chunk_id, seq, form_id, occ, lemma_id, feat_id)
                    )

            if not chunk_rows:
                continue
            doc_id = next_doc_id
            n_words += doc_words
            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    base_urn,
                    doc_lang,
                    metadata.get("title"),
                    metadata.get("author"),
                    metadata.get("source_repo") or version_dir.parts[-4],
                    chunk_rows[0][0],
                    chunk_rows[-1][0],
                    doc_words,
                ),
            )
            conn.executemany("INSERT INTO chunks VALUES (?,?,?)", chunk_rows)
            conn.executemany("INSERT INTO chunk_context VALUES (?,?,?)", text_rows)
            conn.executemany(
                "INSERT INTO staging (doc_id, chunk_id, seq, form_id, occ, lemma_id, feat_id)"
                " VALUES (?,?,?,?,?,?,?)",
                doc_rows,
            )
            if doc_id % 200 == 0:
                print(
                    f"  {doc_id} documents, {n_words:,} words "
                    f"({time.time() - started:.0f}s)"
                )

    conn.executemany(
        "INSERT INTO forms VALUES (?,?,?,?,?,?)",
        ((i, *k, form_counts[i]) for k, i in forms.ids.items()),
    )
    conn.executemany(
        "INSERT INTO lemmas VALUES (?,?,?,?,?,?)",
        ((i, *k, lemma_counts[i]) for k, i in lemmas.ids.items()),
    )
    conn.executemany(
        "INSERT INTO feats VALUES (?,?,?)", ((i, *k) for k, i in feats.ids.items())
    )
    conn.commit()

    # Before clustering: resolving headwords sets staging.morph_id.
    if morph_db is not None:
        print(f"Adding Morpheus headwords ({time.time() - started:.0f}s) ...")
    morph_lemmas = _add_morph_lemmas(conn, morph_db, langs)
    print(f"Clustering postings ({time.time() - started:.0f}s) ...")
    conn.executescript(CLUSTERED)
    built_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    conn.executemany(
        "INSERT INTO meta VALUES (?,?)",
        [
            ("schema_version", str(SCHEMA_VERSION)),
            ("built_at", built_at),
            ("langs", ",".join(langs)),
        ],
    )
    conn.commit()
    conn.execute("PRAGMA analysis_limit = 1000")
    conn.execute("ANALYZE")
    conn.commit()
    conn.execute("PRAGMA journal_mode = DELETE")
    print(f"Vacuuming ({time.time() - started:.0f}s) ...")
    conn.execute("VACUUM")
    conn.close()

    return {
        "documents": doc_id,
        "chunks": chunk_id,
        "words": n_words,
        "forms": len(forms.ids),
        "lemmas": len(lemmas.ids),
        "morph_lemmas": morph_lemmas,
        "skipped_chunks": skipped_chunks,
        "built_at": built_at,
        "seconds": round(time.time() - started),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_index(
    tokens_roots: list[Path],
    out_dir: Path,
    langs: tuple[str, ...] = DEFAULT_LANGS,
    sources: dict[str, str] | None = None,
    morph_db: Path | None = None,
) -> dict:
    """Build the index into out_dir as search-<sha12>.db plus manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "search.db.tmp"
    stats = build_index(tokens_roots, tmp_path, langs, morph_db)
    digest = _sha256(tmp_path)
    db_name = f"search-{digest[:12]}.db"
    tmp_path.replace(out_dir / db_name)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "db": db_name,
        "size": (out_dir / db_name).stat().st_size,
        "sha256": digest,
        "page_size": PAGE_SIZE,
        "langs": list(langs),
        "sources": sources or {},
        **stats,
    }
    manifest_tmp = out_dir / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    manifest_tmp.replace(out_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the range-queryable corpus search index from token sidecars."
    )
    parser.add_argument(
        "--tokens-dir",
        required=True,
        action="append",
        type=Path,
        help="Root of a tokens tree (<namespace>/<textgroup>/<work>/<version>/). "
        "Repeatable.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--morph-db",
        type=Path,
        help="pdl-morph-server's morph.db, for dictionary-headword search. "
        "Without it, the index has no Morpheus headwords.",
    )
    parser.add_argument(
        "--langs",
        default=",".join(DEFAULT_LANGS),
        help=f"Comma-separated sidecar languages to index (default: {','.join(DEFAULT_LANGS)})",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Provenance recorded in manifest.json, e.g. greeklit=<artifact digest>. "
        "Repeatable.",
    )
    args = parser.parse_args()

    sources = dict(s.split("=", 1) for s in args.source)
    manifest = write_index(
        args.tokens_dir, args.out, tuple(args.langs.split(",")), sources, args.morph_db
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
