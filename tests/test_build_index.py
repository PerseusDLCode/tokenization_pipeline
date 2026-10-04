import json
import sqlite3
from pathlib import Path

import zstandard

from mvp_tokenization import build_index
from mvp_tokenization.build_index import PAGE_SIZE, TOKEN_SEP, write_index

BASE = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2"


def _token(text, n, lemma=None, upos=None, ws=True, chunk=f"{BASE}:1.1"):
    punct = not any(ch.isalpha() for ch in text)
    return {
        "text": text,
        "whitespace": ws,
        "identifier": f"{text}[{n}]",
        "urn": None if punct else f"{chunk}@{text}[{n}]",
        "words": [{"lemma": lemma, "upos": upos, "feats": None}],
    }


def _write_sidecar(path: Path, urn: str, lang: str, tokens: list[dict]) -> None:
    payload = json.dumps({"urn": urn, "lang": lang, "tokens": tokens}, ensure_ascii=False)
    path.write_bytes(zstandard.ZstdCompressor().compress(payload.encode()))


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "tokens"
    version = root / "greekLit" / "tlg0012" / "tlg001" / "perseus-grc2"
    version.mkdir(parents=True)
    (version / "metadata.json").write_text(
        json.dumps({"document": {"base_urn": BASE, "title": "Iliad", "author": "Homer"}})
    )
    # Deliberately listed out of lexical order: 1.10 comes after 1.2.
    (version / "index.json").write_text(
        json.dumps({"chunks": [{"file": "1.2.xml"}, {"file": "1.10.xml"}]})
    )
    _write_sidecar(
        version / "1.2.tokens.json.zst",
        f"{BASE}:1.2",
        "grc",
        [
            _token("μῆνιν", 1, "μῆνις", "NOUN", chunk=f"{BASE}:1.2"),
            _token("ἄειδε", 1, "ἀείδω", "VERB", ws=False, chunk=f"{BASE}:1.2"),
            _token(",", 1, chunk=f"{BASE}:1.2"),
            _token("δ", 1, "δέ", "CCONJ", ws=False, chunk=f"{BASE}:1.2"),
            _token("’", 1, chunk=f"{BASE}:1.2"),
            _token("μῆνιν", 2, "μῆνις", "NOUN", chunk=f"{BASE}:1.2"),
        ],
    )
    _write_sidecar(
        version / "1.10.tokens.json.zst",
        f"{BASE}:1.10",
        "grc",
        [_token("Μῆνιν", 1, "μῆνις", "NOUN", chunk=f"{BASE}:1.10")],
    )
    # An alternate-scheme subdirectory must not be indexed (it would double hits).
    (version / "line").mkdir()
    _write_sidecar(
        version / "line" / "1.tokens.json.zst",
        f"{BASE}:1",
        "grc",
        [_token("μῆνιν", 1, "μῆνις", "NOUN", chunk=f"{BASE}:1")],
    )
    # An English translation: not an indexed language.
    eng = root / "greekLit" / "tlg0012" / "tlg001" / "perseus-eng3"
    eng.mkdir(parents=True)
    _write_sidecar(
        eng / "1.tokens.json.zst",
        "urn:cts:greekLit:tlg0012.tlg001.perseus-eng3:1",
        "en",
        [_token("wrath", 1, "wrath", "NOUN")],
    )
    return root


def _connect(tmp_path):
    out = tmp_path / "out"
    manifest = write_index([_corpus(tmp_path)], out, sources={"greeklit": "sha256:abc"})
    return manifest, sqlite3.connect(out / manifest["db"])


def test_manifest_and_pragmas(tmp_path):
    manifest, conn = _connect(tmp_path)
    assert manifest["db"].startswith("search-") and manifest["db"].endswith(".db")
    assert manifest["documents"] == 1
    assert manifest["chunks"] == 2
    assert manifest["words"] == 5
    assert manifest["sources"] == {"greeklit": "sha256:abc"}
    assert conn.execute("PRAGMA page_size").fetchone()[0] == PAGE_SIZE
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_chunks_in_citation_order(tmp_path):
    _, conn = _connect(tmp_path)
    refs = [r for (r,) in conn.execute("SELECT ref FROM chunks ORDER BY chunk_id")]
    assert refs == ["1.2", "1.10"]
    first, last = conn.execute("SELECT first_chunk, last_chunk FROM documents").fetchone()
    assert (first, last) == (1, 2)


def test_lemma_hits_rebuild_sidecar_urns(tmp_path):
    _, conn = _connect(tmp_path)
    rows = conn.execute(
        """
        SELECT d.base_urn || ':' || c.ref || '@' || f.form || '[' || p.occ || ']'
        FROM lemmas l
        JOIN lemma_postings p ON p.lemma_id = l.lemma_id
        JOIN chunks c ON c.chunk_id = p.chunk_id
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN forms f ON f.form_id = p.form_id
        WHERE l.key = 'μῆνισ'
        ORDER BY p.chunk_id, p.seq
        """
    ).fetchall()
    assert [r for (r,) in rows] == [
        f"{BASE}:1.2@μῆνιν[1]",
        f"{BASE}:1.2@μῆνιν[2]",
        f"{BASE}:1.10@Μῆνιν[1]",
    ]


def test_form_key_folds_case_and_counts_by_document(tmp_path):
    _, conn = _connect(tmp_path)
    total = conn.execute(
        """
        SELECT SUM(c.n) FROM forms f JOIN form_doc_counts c ON c.form_id = f.form_id
        WHERE f.key = 'μῆνιν' AND f.lang = 'grc'
        """
    ).fetchone()[0]
    assert total == 3


def test_elided_form_keys_with_apostrophe(tmp_path):
    _, conn = _connect(tmp_path)
    assert conn.execute("SELECT form, key FROM forms WHERE form = 'δ'").fetchone() == (
        "δ",
        "δ'",
    )


def test_chunk_context_is_indexed_by_seq(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "CONTEXT_PART", 4)
    _, conn = _connect(tmp_path)
    parts = conn.execute(
        "SELECT part, tokens FROM chunk_context WHERE chunk_id = 1 ORDER BY part"
    ).fetchall()
    assert [p for p, _ in parts] == [0, 1]
    tokens = [t for _, text in parts for t in text.split(TOKEN_SEP)]
    assert tokens == ["μῆνιν ", "ἄειδε", ", ", "δ", "’ ", "μῆνιν "]
    (seq,) = conn.execute(
        "SELECT seq FROM form_postings WHERE chunk_id = 1 ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert tokens[seq].strip() == "μῆνιν"
