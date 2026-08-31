#!/usr/bin/env python3
"""Tokenize compiled chunk XML files in-process (fast pass: tokenize-only).

Standalone counterpart to run_analyzer.py: loads only the tokenize-only
stanza pipeline (processors="tokenize") or spaCy's bare tokenizer
(la_core_web_lg/grc_dep_web_lg's `.tokenizer`, not the full pipeline) --
no POS/lemma/depparse, and no NLP server or nlp_pipeline dependency at all.
Sidecar schema matches run_analyzer.py's exactly (mostly-null morphological
fields here); run --force with mvp-analyze later to upgrade a sidecar in
place with full analysis, whenever that's actually wanted.

Sequential by design, same reasoning as run_analyzer.py: the cached
per-language pipeline objects below aren't meant to be driven concurrently.
Tokenize-only inference is far cheaper per call than full analysis, so this
is still fast in practice without needing threads (and without the earlier
HTTP path's server-liveness/readiness problem in CI).
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

import spacy
import stanza
import zstandard
from kodon_py.tei_parser import TEIParser, TEIParserError
from lxml import etree
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc as SpacyDoc
from stanza.models.common.doc import Document as StanzaDocument
from stanza.pipeline.core import Pipeline as StanzaPipeline

TOKENS_DIR = os.getenv("MVP_TOKENS_DIR", "tokenized-pages")

_ZSTD_LEVEL = 19

# Latin and Ancient Greek are handled by LatinCy's spaCy models rather than a
# stanza package -- see nlp_pipeline/src/nlp_pipeline/pipeline.py, which
# this script otherwise mirrors (tokenize-only, not the full pipeline).
SPACY_MODELS = {
    "la": "la_core_web_lg",
    "grc": "grc_dep_web_lg",
}
STANZA_LANGS = ["ar", "de", "en", "es", "fa", "fr", "he", "it", "pt"]

_langid: StanzaPipeline | None = None
_stanza_tokenize: dict[str, StanzaPipeline] = {}
_spacy_pipelines: dict[str, SpacyLanguage] = {}


def token_sidecar_name(chunk_path: Path) -> str:
    """Return the token sidecar filename for a compiled chunk XML file.

    e.g. ``10.xml`` -> ``10.tokens.json.zst``. Shared between the tokenizer
    (this module, which writes these) and the reading-view render path
    (mvp.site.app, which reads them), so both agree on the per-chunk,
    individually-compressed naming that makes lazy per-chunk decompression
    possible.
    """
    return chunk_path.with_suffix("").name + ".tokens.json.zst"


def _is_punct(text: str) -> bool:
    return bool(text) and all(unicodedata.category(c)[0] in ("P", "S") for c in text)


def _get_langid(model_dir: str) -> StanzaPipeline:
    global _langid
    if _langid is None:
        _langid = stanza.Pipeline(
            dir=model_dir,
            lang="multilingual",
            processors="langid",
            langid_lang_subset=[*STANZA_LANGS, *SPACY_MODELS],
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        )
    return _langid


def _identify_lang(text: str, model_dir: str) -> str:
    doc = StanzaDocument([], text=text)
    _get_langid(model_dir)(doc)
    return doc.lang


def _get_stanza_pipeline(lang: str, model_dir: str) -> StanzaPipeline:
    if lang not in _stanza_tokenize:
        _stanza_tokenize[lang] = stanza.Pipeline(
            dir=model_dir,
            lang=lang,
            processors="tokenize",
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        )
    return _stanza_tokenize[lang]


def _get_spacy_pipeline(lang: str) -> SpacyLanguage:
    if lang not in _spacy_pipelines:
        _spacy_pipelines[lang] = spacy.load(SPACY_MODELS[lang])
    return _spacy_pipelines[lang]


def _stanza_token_dicts(doc: StanzaDocument) -> list[dict]:
    return [
        {
            "end_char": token.end_char,
            "id": token.id,
            "misc": None,
            "ner": None,
            "start_char": token.start_char,
            "text": token.text,
            "whitespace": len(token.spaces_after) > 0,
            "words": [
                {
                    "id": word.id,
                    "deprel": word.deprel,
                    "deps": word.deps,
                    "feats": word.feats,
                    "head": word.head,
                    "lemma": word.lemma,
                    "misc": word.misc,
                    "text": word.text,
                    "upos": word.upos,
                    "xpos": word.xpos,
                }
                for word in token.words
            ],
        }
        for token in doc.iter_tokens()
    ]


def _spacy_token_dicts(doc: SpacyDoc) -> list[dict]:
    return [
        {
            "end_char": token.idx + len(token.text),
            "id": [token.i],
            "misc": None,
            "ner": token.ent_type_ or None,
            "start_char": token.idx,
            "text": token.text,
            "whitespace": bool(token.whitespace_),
            "words": [
                {
                    "id": token.i,
                    "deprel": token.dep_ or None,
                    "deps": None,
                    "feats": str(token.morph) or None,
                    "head": (
                        token.head.i if token.dep_ and token.dep_ != "ROOT" else None
                    ),
                    "lemma": token.lemma_ or None,
                    "misc": None,
                    "text": token.text,
                    "upos": token.pos_ or None,
                    "xpos": token.tag_ or None,
                }
            ],
        }
        for token in doc
    ]


def _tokenize_only(
    chunk_urn: str, primary_text: str, model_dir: str
) -> tuple[str, list[dict]]:
    if not primary_text.strip():
        return "", []

    lang = _identify_lang(primary_text, model_dir)

    if lang in SPACY_MODELS:
        # Bare tokenizer, not the full pipeline call -- no tagger/parser/
        # lemmatizer inference, even though the model is loaded (same
        # tradeoff nlp_pipeline's NLPPipeline.tokenize() makes).
        raw_tokens = _spacy_token_dicts(_get_spacy_pipeline(lang).tokenizer(primary_text))
    else:
        raw_tokens = _stanza_token_dicts(_get_stanza_pipeline(lang, model_dir)(primary_text))

    token_counts: dict[str, int] = {}
    tokens = []
    for token in raw_tokens:
        text = token["text"].strip()
        if not text:
            continue
        count = token_counts.get(text, 0) + 1
        token_counts[text] = count
        identifier = f"{text}[{count}]"
        urn = None if _is_punct(text) else f"{chunk_urn}@{identifier}"
        tokens.append({**token, "identifier": identifier, "urn": urn, "text": text})

    return lang, tokens


def _primary_text(chunk_file: Path) -> tuple[str, str]:
    """Return (cts_urn, primary_text) for a compiled chunk XML file."""
    root = etree.parse(chunk_file).getroot()
    cts_urn = root.get("cts_urn", "")
    base_urn = root.get("base_urn", "") or cts_urn.rsplit(":", 1)[0]
    chunk_unit = root.get("unit", "")

    content_el = root.find("elements")
    if content_el is None:
        raise TEIParserError(f"No <elements> in {chunk_file}")

    parser = TEIParser(content_el, base_urn, chunk_unit)
    return cts_urn, parser.primary_text


def _iter_chunk_files(proto_dir: Path):
    for index_file in sorted(proto_dir.glob("**/index.json")):
        version_dir = index_file.parent
        with open(index_file) as f:
            chunks = json.load(f).get("chunks", [])
        for entry in chunks:
            chunk_file = version_dir / entry["file"]
            if chunk_file.exists():
                yield chunk_file


def _process_chunk(
    chunk_file: Path,
    proto_dir: Path,
    tokens_dir: Path | None,
    model_dir: str,
    force: bool,
) -> str:
    """Tokenize one chunk and write its sidecar. Returns "generated", "skipped", or "failed"."""
    if tokens_dir is not None:
        rel_dir = chunk_file.parent.resolve().relative_to(proto_dir)
        sidecar_dir = tokens_dir / rel_dir
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    else:
        sidecar_dir = chunk_file.parent
    sidecar = sidecar_dir / token_sidecar_name(chunk_file)
    if sidecar.exists() and not force:
        return "skipped"

    try:
        cts_urn, primary_text = _primary_text(chunk_file)
        lang, tokens = _tokenize_only(cts_urn, primary_text, model_dir)
    except Exception as exc:
        print(f"  FAILED: {chunk_file.name}: {exc}", file=sys.stderr)
        return "failed"

    payload = json.dumps(
        {"urn": cts_urn, "lang": lang, "tokens": tokens}, ensure_ascii=False
    ).encode("utf-8")
    compressor = zstandard.ZstdCompressor(level=_ZSTD_LEVEL)
    sidecar.write_bytes(compressor.compress(payload))
    return "generated"


def tokenize_dir(
    proto_dir: Path,
    tokens_dir: Path | None,
    model_dir: str = "./stanza_models",
    force: bool = False,
) -> dict:
    """Tokenize every compiled chunk under proto_dir. Returns generated/skipped/failed counts."""
    proto_dir = proto_dir.resolve()
    generated = skipped = failed = 0

    for chunk_file in _iter_chunk_files(proto_dir):
        result = _process_chunk(chunk_file, proto_dir, tokens_dir, model_dir, force)
        if result == "generated":
            generated += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1

        if (generated + skipped + failed) % 500 == 0:
            print(f"So far: {generated} generated, {skipped} skipped, {failed} failed.")

    return {"generated": generated, "skipped": skipped, "failed": failed}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Tokenize compiled chunk XML files in-process (fast pass: "
        "tokenize-only, no POS/lemma/deps, no NLP server needed). Re-run "
        "mvp-analyze later to upgrade sidecars in place with full analysis."
    )
    parser.add_argument(
        "--proto-dir",
        required=True,
        type=Path,
        help="Root directory of compiled chunk XML files (output of Chunker)",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=TOKENS_DIR,
        help=(
            "Root directory to write sidecars into, mirroring --proto-dir's "
            "layout (default: MVP_TOKENS_DIR env var, else 'tokenized-pages'). "
            "If unset, sidecars are written alongside each chunk XML."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default="./stanza_models",
        help="Directory for stanza model downloads/cache (default: ./stanza_models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tokenize even if a sidecar already exists",
    )
    args = parser.parse_args()

    result = tokenize_dir(
        args.proto_dir, args.tokens_dir, args.model_dir, args.force
    )
    print(
        f"Tokenization: {result['generated']} generated, "
        f"{result['skipped']} skipped, {result['failed']} failed."
    )


if __name__ == "__main__":
    main()
