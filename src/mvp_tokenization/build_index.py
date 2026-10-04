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

Only the default citation scheme is indexed (sidecars directly in a version
dir, not in its scheme subdirectories), since alternate schemes re-chunk the
same text and would double every hit.

    uv run mvp-index --tokens-dir ../tokens --out ./search-index

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

SCHEMA_VERSION = 1
PAGE_SIZE = 4096
DEFAULT_LANGS = ("grc", "la")
SIDECAR_SUFFIX = ".tokens.json.zst"
TOKEN_SEP = "\x1f"
CONTEXT_PART = 32

_IDENTIFIER_OCC = re.compile(r"\[(\d+)\]$")
_LANG_ALIASES = {"lat": "la"}

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
  feat_id INTEGER
);
"""

CLUSTERED = """
CREATE TABLE form_postings (
  form_id INTEGER NOT NULL,
  chunk_id INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  occ INTEGER NOT NULL,
  lemma_id INTEGER,
  feat_id INTEGER,
  PRIMARY KEY (form_id, chunk_id, seq)
) WITHOUT ROWID;
INSERT INTO form_postings
  SELECT form_id, chunk_id, seq, occ, lemma_id, feat_id FROM staging
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

CREATE TABLE form_doc_counts (
  form_id INTEGER NOT NULL,
  doc_id INTEGER NOT NULL,
  n INTEGER NOT NULL,
  PRIMARY KEY (form_id, doc_id)
) WITHOUT ROWID;
INSERT INTO form_doc_counts
  SELECT form_id, doc_id, COUNT(*) FROM staging
  GROUP BY form_id, doc_id ORDER BY form_id, doc_id;

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

CREATE INDEX forms_key ON forms (key, lang, form_id, n, form);
CREATE INDEX forms_loose ON forms (loose, lang, form_id, n, form);
CREATE INDEX lemmas_key ON lemmas (key, lang, lemma_id, n, lemma);
CREATE INDEX lemmas_loose ON lemmas (loose, lang, lemma_id, n, lemma);
CREATE INDEX documents_base_urn ON documents (base_urn);
"""


def _natural_key(name: str) -> list:
    stem = name[: -len(SIDECAR_SUFFIX)] if name.endswith(SIDECAR_SUFFIX) else name
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


def build_index(
    tokens_roots: list[Path],
    out_path: Path,
    langs: tuple[str, ...] = DEFAULT_LANGS,
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
                    print(f"  skipping {sidecar}: {urn} not in {base_urn}", file=sys.stderr)
                    skipped_chunks += 1
                    continue
                doc_lang = doc_lang or lang
                chunk_id += 1
                chunk_rows.append((chunk_id, next_doc_id, ref))

                tokens = data.get("tokens", [])
                texts = [t["text"] + (" " if t.get("whitespace") else "") for t in tokens]
                text_rows.extend(
                    (chunk_id, i // CONTEXT_PART, TOKEN_SEP.join(texts[i : i + CONTEXT_PART]))
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
                        (text, match_key(key_text, lang), loose_key(key_text, lang), lang)
                    )

                    lemma_id = feat_id = None
                    m = _IDENTIFIER_OCC.search(token.get("identifier", ""))
                    occ = int(m.group(1)) if m else 1
                    word = (token.get("words") or [{}])[0]
                    lemma = word.get("lemma")
                    if lemma and not is_apostrophe(lemma):
                        lemma_id = lemmas.get(
                            (lemma, match_key(lemma, lang), loose_key(lemma, lang), lang)
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
            conn.executemany("INSERT INTO staging VALUES (?,?,?,?,?,?,?)", doc_rows)
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
) -> dict:
    """Build the index into out_dir as search-<sha12>.db plus manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "search.db.tmp"
    stats = build_index(tokens_roots, tmp_path, langs)
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
        args.tokens_dir, args.out, tuple(args.langs.split(",")), sources
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
