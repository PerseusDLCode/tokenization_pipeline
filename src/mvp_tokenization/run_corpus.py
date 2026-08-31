#!/usr/bin/env python3
"""Chain citeStructure compilation and fast-pass tokenization for a whole corpus.

Single, resumable entry point: for a corpus repo (e.g. corpora/canonical-latinLit),
walks its data/ tree for TEI files that have a citeStructure refsDecl, compiles each
into citation chunks via perseus-cts's Chunker (skipping files that don't have one --
about half the corpus isn't migrated to citeStructure yet, that's expected, not an
error), then runs the fast tokenize-only pass (mvp_tokenization.run_tokenizer) against
everything that got compiled. Writes a JSON run manifest so a run's outcome is
auditable rather than scrollback-only.

Both stages are idempotent (skip-if-exists, override with --force), so interrupting
and re-running only does the remaining work.

    uv run mvp-pipeline --corpus-root ../corpora/canonical-latinLit \\
        --proto-dir ./proto-pages --tokens-dir ./tokenized-pages
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lxml import etree
from perseus_cts import Chunker, ConfigurationError, LenientTEIDocument

from mvp_tokenization.run_tokenizer import (
    NLP_URL,
    TOKENS_DIR,
    _AbortTokenization,
    _iter_chunk_files,
    _process_chunk,
)

_CTS_REFSDECL_XPATH = (
    "/*[local-name()='TEI']/*[local-name()='teiHeader']"
    "/*[local-name()='encodingDesc']"
    "/*[local-name()='refsDecl'][@xml:id='CTS']"
    "/*[local-name()='citeStructure']"
)


def _has_cite_structure(xml_path: Path) -> bool:
    """Cheap pre-check so we don't spend a full TEIDocument parse on files that
    can't possibly have a citeStructure refsDecl (~half the corpus, currently)."""
    try:
        for _event, elem in etree.iterparse(
            str(xml_path), events=("end",), tag="{*}citeStructure", recover=True
        ):
            elem.clear()
            return True
    except etree.XMLSyntaxError:
        return False
    return False


def _work_output_dir(proto_root: Path, base_urn: str) -> Path:
    """urn:cts:latinLit:phi0959.phi002.perseus-lat2 -> proto_root/latinLit/phi0959/phi002/perseus-lat2

    Mirrors the layout MinimumViablePerseus's proto-pages/tokenized-pages trees
    already use, so this output is a drop-in for the existing consumer.
    """
    _urn, _cts, namespace, work = base_urn.split(":", 3)
    textgroup, work_id, exemplar = (work.split(".") + ["", ""])[:3]
    return proto_root / namespace / textgroup / work_id / exemplar


def _iter_tei_files(corpus_root: Path):
    data_dir = corpus_root / "data"
    for xml_path in sorted(data_dir.glob("**/*.xml")):
        if xml_path.name == "__cts__.xml":
            continue
        yield xml_path


def compile_corpus(
    corpus_root: Path, proto_dir: Path, force: bool
) -> list[dict]:
    """Compile every citeStructure-bearing work in corpus_root into proto_dir.

    Returns a list of per-file manifest entries."""
    entries: list[dict] = []
    for xml_path in _iter_tei_files(corpus_root):
        rel = xml_path.relative_to(corpus_root)
        if not _has_cite_structure(xml_path):
            entries.append({"path": str(rel), "status": "no-citestructure"})
            continue

        try:
            tei_doc = LenientTEIDocument(xml_path)
            chunker = Chunker(tei_doc)
            base_urn = chunker.cts_resolver.base_urn
        except ConfigurationError as exc:
            entries.append(
                {"path": str(rel), "status": "no-citestructure", "reason": str(exc)}
            )
            continue
        except Exception as exc:
            entries.append({"path": str(rel), "status": "failed", "reason": str(exc)})
            print(f"  FAILED (compile): {rel}: {exc}", file=sys.stderr)
            continue

        output_path = _work_output_dir(proto_dir, base_urn)
        index_file = output_path / "index.json"
        if index_file.exists() and not force:
            mtime_ok = index_file.stat().st_mtime >= xml_path.stat().st_mtime
            if mtime_ok:
                entries.append(
                    {"path": str(rel), "status": "skipped", "urn_base": base_urn}
                )
                continue

        try:
            chunker.compile(output_path)
        except Exception as exc:
            entries.append({"path": str(rel), "status": "failed", "reason": str(exc)})
            print(f"  FAILED (compile): {rel}: {exc}", file=sys.stderr)
            continue

        entries.append(
            {
                "path": str(rel),
                "status": "compiled",
                "urn_base": base_urn,
                "chunk_count": len(chunker.citation_chunks),
            }
        )
    return entries


def tokenize_corpus(
    proto_dir: Path, tokens_dir: Path | None, nlp_url: str, workers: int, force: bool
) -> dict:
    proto_dir = proto_dir.resolve()
    generated = skipped = failed = 0
    chunk_files = list(_iter_chunk_files(proto_dir))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_chunk, chunk_file, proto_dir, tokens_dir, nlp_url, force
            ): chunk_file
            for chunk_file in chunk_files
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                if result == "generated":
                    generated += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    failed += 1
                if (generated + skipped + failed) % 500 == 0:
                    print(
                        f"  tokenize: {generated} generated, {skipped} skipped, "
                        f"{failed} failed so far"
                    )
        except _AbortTokenization as exc:
            print(f"ABORTED: {exc}", file=sys.stderr)
            pool.shutdown(wait=False, cancel_futures=True)
            return {
                "generated": generated,
                "skipped": skipped,
                "failed": failed,
                "aborted": str(exc),
            }

    return {"generated": generated, "skipped": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile citeStructure-bearing works into citation chunks, "
        "then tokenize them (fast pass) -- a single, resumable, idempotent run "
        "over a whole corpus repo."
    )
    parser.add_argument(
        "--corpus-root", required=True, type=Path, help="e.g. corpora/canonical-latinLit"
    )
    parser.add_argument("--proto-dir", required=True, type=Path)
    parser.add_argument("--tokens-dir", type=Path, default=TOKENS_DIR)
    parser.add_argument("--nlp-url", default=NLP_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--force", action="store_true", help="Recompile/re-tokenize even if outputs exist"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Where to write the JSON run manifest (default: <proto-dir>/_run-manifest.json)",
    )
    args = parser.parse_args()

    corpus_root = args.corpus_root.resolve()
    proto_dir = args.proto_dir.resolve()
    manifest_path = args.manifest or (proto_dir / "_run-manifest.json")

    started = time.time()
    print(f"Compiling citeStructure works from {corpus_root} ...")
    works = compile_corpus(corpus_root, proto_dir, args.force)
    compiled = sum(1 for w in works if w["status"] == "compiled")
    skipped_compile = sum(1 for w in works if w["status"] == "skipped")
    no_cs = sum(1 for w in works if w["status"] == "no-citestructure")
    failed_compile = sum(1 for w in works if w["status"] == "failed")
    print(
        f"Compile: {compiled} compiled, {skipped_compile} already up to date, "
        f"{no_cs} skipped (no citeStructure), {failed_compile} failed."
    )

    print(f"Tokenizing chunks under {proto_dir} via {args.nlp_url} ...")
    tokenize_result = tokenize_corpus(
        proto_dir, args.tokens_dir, args.nlp_url, args.workers, args.force
    )
    print(f"Tokenize: {tokenize_result}")

    manifest = {
        "corpus_root": str(corpus_root),
        "proto_dir": str(proto_dir),
        "tokens_dir": str(args.tokens_dir) if args.tokens_dir else None,
        "started": started,
        "finished": time.time(),
        "works": works,
        "tokenize": tokenize_result,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")

    if failed_compile or tokenize_result.get("failed") or tokenize_result.get("aborted"):
        sys.exit(1)


if __name__ == "__main__":
    main()
