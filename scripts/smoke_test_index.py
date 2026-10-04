"""Sanity-check a freshly built search index before it's published.

    uv run python scripts/smoke_test_index.py <out dir>
"""

import json
import sqlite3
import sys
from pathlib import Path

from mvp_tokenization.build_index import PAGE_SIZE

ILIAD = "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2"


def main(out_dir: Path) -> None:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    db = out_dir / manifest["db"]
    assert db.stat().st_size == manifest["size"], "manifest size mismatch"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA page_size").fetchone()[0] == PAGE_SIZE
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert manifest["documents"] > 0 and manifest["words"] > 0

    lemmatized = conn.execute("SELECT COUNT(*) FROM lemma_postings").fetchone()[0]
    assert lemmatized > 0.5 * manifest["words"], (
        f"only {lemmatized} of {manifest['words']} words have a lemma -- "
        "were the sidecars produced by the tokenize-only pass?"
    )

    iliad = conn.execute(
        "SELECT first_chunk, last_chunk FROM documents WHERE base_urn = ?", (ILIAD,)
    ).fetchone()
    if iliad:
        hits = conn.execute(
            """
            SELECT COUNT(*) FROM lemmas l JOIN lemma_postings p USING (lemma_id)
            WHERE l.key = 'μῆνισ' AND l.lang = 'grc' AND p.chunk_id BETWEEN ? AND ?
            """,
            iliad,
        ).fetchone()[0]
        assert hits > 0, "no hits for μῆνις in the Iliad"

    print(
        f"OK: {manifest['db']} -- {manifest['documents']} documents, "
        f"{manifest['words']:,} words, {lemmatized:,} lemmatized"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
