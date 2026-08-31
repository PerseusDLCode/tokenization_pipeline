#!/usr/bin/env python3
"""Analyze compiled chunk XML files in-process and (re)write their token sidecars.

This is the standalone-script counterpart to run_tokenizer.py: instead of
POSTing to a running NLP server's /tokenize endpoint, it loads the
tokenize+POS+lemma+depparse models directly in this process (stanza for
most languages, LatinCy's spaCy models for `la`/`grc`) and writes the full
morphological analysis straight into the same .tokens.json.zst sidecar
format run_tokenizer.py already produces -- run_tokenizer.py's docstring
anticipated exactly this: the sidecar's per-word fields are "mostly null
morphological fields" only because that script does tokenize-only. This
script fills them in, in place, so no separate artifact or consumer change
is needed.

No NLP server, and no dependency on the nlp_pipeline package, is required:
the routing/model logic here is a self-contained port of
nlp_pipeline's NLPPipeline.analyze. Install the `nlp` dependency group
(spacy, stanza, and the LatinCy wheels) into this repo's venv to run it:

    uv sync --group nlp
    python src/mvp_tokenization/run_analyzer.py --proto-dir ./proto-pages --tokens-dir ./tokenized-pages

Sequential by design: stanza/spacy inference is CPU/GPU-bound and the
per-language pipeline objects cached below aren't meant to be driven
concurrently, unlike run_tokenizer.py's network calls. Re-running is safe:
chunks whose sidecar already exists are skipped unless --force is given --
pass --force on the first run over sidecars produced by the old
tokenize-only pass, to upgrade them in place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

import spacy
import stanza
import zstandard
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc as SpacyDoc
from stanza.models.common.doc import Document as StanzaDocument
from stanza.pipeline.core import Pipeline as StanzaPipeline

from mvp_tokenization.run_tokenizer import _iter_chunk_files, _primary_text

_ZSTD_LEVEL = 19

# Latin and Ancient Greek are handled by LatinCy's spaCy models rather
# than a stanza package -- see nlp_pipeline/src/nlp_pipeline/pipeline.py,
# which this script otherwise mirrors.
SPACY_MODELS = {
    "la": "la_core_web_lg",
    "grc": "grc_dep_web_lg",
}
STANZA_LANGS = ["ar", "de", "en", "es", "fa", "fr", "he", "it", "pt"]
STANZA_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"

TOKENS_DIR = os.getenv("MVP_TOKENS_DIR", "tokenized-pages")

_langid: StanzaPipeline | None = None
_stanza_pipelines: dict[str, StanzaPipeline] = {}
_spacy_pipelines: dict[str, SpacyLanguage] = {}


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
    if lang not in _stanza_pipelines:
        _stanza_pipelines[lang] = stanza.Pipeline(
            dir=model_dir,
            lang=lang,
            processors=STANZA_PROCESSORS,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        )
    return _stanza_pipelines[lang]


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


def _analyze(
    chunk_urn: str, primary_text: str, model_dir: str
) -> tuple[str, list[dict]]:
    if not primary_text.strip():
        return "", []

    lang = _identify_lang(primary_text, model_dir)

    if lang in SPACY_MODELS:
        raw_tokens = _spacy_token_dicts(_get_spacy_pipeline(lang)(primary_text))
    else:
        raw_tokens = _stanza_token_dicts(
            _get_stanza_pipeline(lang, model_dir)(primary_text)
        )

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


def _process_chunk(
    chunk_file: Path,
    proto_dir: Path,
    tokens_dir: Path | None,
    model_dir: str,
    force: bool,
) -> str:
    """Analyze one chunk and write its sidecar. Returns "generated", "skipped", or "failed"."""
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
        lang, tokens = _analyze(cts_urn, primary_text, model_dir)
    except Exception as exc:
        print(f"  FAILED: {chunk_file.name}: {exc}", file=sys.stderr)
        return "failed"

    payload = json.dumps(
        {"urn": cts_urn, "lang": lang, "tokens": tokens}, ensure_ascii=False
    ).encode("utf-8")
    compressor = zstandard.ZstdCompressor(level=_ZSTD_LEVEL)
    sidecar.write_bytes(compressor.compress(payload))
    return "generated"


def token_sidecar_name(chunk_path: Path) -> str:
    """Return the token sidecar filename for a compiled chunk XML file.

    e.g. ``10.xml`` -> ``10.tokens.json.zst``. Shared between the tokenizer
    (src/tools/run_tokenizer.py, which writes these) and the reading-view
    render path (mvp.site.app, which reads them), so both agree on the
    per-chunk, individually-compressed naming that makes lazy per-chunk
    decompression possible.
    """
    return chunk_path.with_suffix("").name + ".tokens.json.zst"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze compiled chunk XML files and write token sidecars"
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
            "layout (default: mvp.site.config.TOKENS_DIR / MVP_TOKENS_DIR env "
            "var). If unset, sidecars are written alongside each chunk XML."
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
        help="Re-analyze even if a sidecar already exists (needed to upgrade "
        "sidecars written by run_tokenizer.py's tokenize-only pass)",
    )
    args = parser.parse_args()

    proto_dir = args.proto_dir.resolve()
    generated = skipped = failed = 0

    for chunk_file in _iter_chunk_files(args.proto_dir):
        result = _process_chunk(
            chunk_file, proto_dir, args.tokens_dir, args.model_dir, args.force
        )
        if result == "generated":
            generated += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1

        if (generated + skipped + failed) % 500 == 0:
            print(f"So far: {generated} generated, {skipped} skipped, {failed} failed.")

    print(f"Analysis: {generated} generated, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
