#!/usr/bin/env python3
"""Tokenize compiled chunk XML files in-process, lemmatizing Greek and Latin.

Greek and Latin chunks go through LatinCy's spaCy pipeline -- tagger,
morphologizer and lemmatizers, minus the parser and NER, which nothing
downstream uses -- because the corpus search index (build_index.py) needs a
lemma and morphology for every token. Every other language gets stanza's
tokenize-only pipeline (processors="tokenize"); mvp-analyze can still
upgrade those sidecars in place with full analysis. No NLP server or
nlp_pipeline dependency at all.

Incremental: each sidecar directory carries a tokens-manifest.json recording,
per chunk, a hash of the chunk XML and the pipeline version that produced
the sidecar. A chunk is reprocessed only when either changes (or with
--force), so restoring the previous run's output and re-running only pays
for new and edited texts.

Sequential by design, same reasoning as run_analyzer.py: the cached
per-language pipeline objects below aren't meant to be driven concurrently.
Tokenize-only inference is far cheaper per call than full analysis, so this
is still fast in practice without needing threads (and without the earlier
HTTP path's server-liveness/readiness problem in CI).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import unicodedata
from collections import Counter
from importlib.metadata import version as package_version
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
# Chunks per model call (and per sidecar-writing round). Bounds memory for
# works with thousands of chunks while keeping batches large enough to
# amortize per-call overhead.
_BATCH_SIZE = 256

# Bump when the sidecar contents change for reasons the model versions don't
# capture (e.g. how token text or URNs are derived), to force a rebuild.
SIDECAR_SCHEMA = 2

TOKENS_MANIFEST = "tokens-manifest.json"
# Copied verbatim from each proto version dir so the tokens artifact alone
# carries document metadata and chunk order (build_index.py reads them).
PROTO_METADATA_FILES = ("metadata.json", "index.json")

# Languages that get the full (lemmatizing) spaCy pipeline. Must be a subset
# of SPACY_MODELS.
ANALYZE_LANGS = frozenset({"la", "grc"})
# Unused by search and the most expensive components to run.
SPACY_EXCLUDE = ["parser", "ner"]

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
        _spacy_pipelines[lang] = spacy.load(SPACY_MODELS[lang], exclude=SPACY_EXCLUDE)
    return _spacy_pipelines[lang]


def pipeline_version(lang: str) -> str:
    """Identify everything that determines a sidecar's contents for `lang`."""
    if lang in SPACY_MODELS:
        model = SPACY_MODELS[lang]
        mode = "analyze" if lang in ANALYZE_LANGS else "tokenize"
        model_version = package_version(model.replace("_", "-"))
        return f"{SIDECAR_SCHEMA}:spacy-{spacy.__version__}:{model}-{model_version}:{mode}"
    if lang:
        return f"{SIDECAR_SCHEMA}:stanza-{stanza.__version__}:tokenize"
    return f"{SIDECAR_SCHEMA}:empty"


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


def _finalize_tokens(
    chunk_urn: str, primary_text: str, raw_tokens: list[dict]
) -> list[dict]:
    """Assign each token its source text, identifier, and URN."""
    token_counts: dict[str, int] = {}
    tokens = []
    for token in raw_tokens:
        # Take the text from the source, not the model: LatinCy's Latin
        # tokenizer normalizes u/v (so "virumque" comes back as "uirumque"),
        # and the reading view renders token text verbatim.
        text = (
            primary_text[token["start_char"] : token["end_char"]].strip()
            or token["text"].strip()
        )
        if not text:
            continue
        count = token_counts.get(text, 0) + 1
        token_counts[text] = count
        identifier = f"{text}[{count}]"
        urn = None if _is_punct(text) else f"{chunk_urn}@{identifier}"
        tokens.append({**token, "identifier": identifier, "urn": urn, "text": text})
    return tokens


# Model input is capped at this many characters. Some chunks are whole
# books (First1KGreek has chunks of 200,000+ words), and running a spaCy
# pipeline over a document that size holds activations for every token at
# once -- more memory than a hosted CI runner has. Longer texts are cut into
# segments at sentence (or at least word) boundaries, run separately, and
# their tokens shifted back into the whole text's coordinates.
#
# Measured on a 600k-character First1KGreek chunk with grc_dep_web_lg:
# 2,000-char segments in batches of 8 peak at ~2 GB (model included), the
# same throughput as larger settings, which peak at 3-6 GB.
SEGMENT_CHARS = 2_000
_SPACY_BATCH_SIZE = 8
_SENTENCE_BREAKS = (". ", "\u00b7 ", "\u0387 ", "; ", "\n")


# stanza's langid model runs over its whole input and needs roughly 90 MB
# per thousand characters, so a book-length chunk would take tens of GB.
# A chunk's language is identified from a sample instead: the whole text if
# it's short, else slices from its beginning, middle and end (so front
# matter in another language doesn't decide it alone).
LANGID_SAMPLE_CHARS = 3_000


def _langid_sample(text: str) -> str:
    if len(text) <= LANGID_SAMPLE_CHARS:
        return text
    width = LANGID_SAMPLE_CHARS // 3
    starts = (0, (len(text) - width) // 2, len(text) - width)
    return " ".join(text[start : start + width] for start in starts)


def _segments(text: str, limit: int | None = None) -> list[tuple[int, str]]:
    """Split text into (offset, segment) pieces of at most `limit` chars
    (default SEGMENT_CHARS).

    Cuts go after the whitespace following a sentence break in the second
    half of the window, else after the last space there, else at `limit`;
    keeping the whitespace with the earlier segment preserves its last
    token's trailing-whitespace flag."""
    limit = limit or SEGMENT_CHARS
    segments = []
    start = 0
    while len(text) - start > limit:
        lo, hi = start + limit // 2, start + limit
        cut = max(text.rfind(b, lo, hi) + len(b) for b in _SENTENCE_BREAKS)
        if cut < lo + 1:
            cut = text.rfind(" ", lo, hi) + 1
        if cut < lo + 1:
            cut = hi
        segments.append((start, text[start:cut]))
        start = cut
    segments.append((start, text[start:]))
    return segments


def _shift(tokens: list[dict], chars: int, index: int) -> list[dict]:
    """Move a segment's tokens into the whole text's coordinates: character
    offsets by `chars`, and spaCy's doc-relative token indices by `index`
    (stanza's ids are sentence-relative, so they're left alone)."""
    for token in tokens:
        token["start_char"] += chars
        token["end_char"] += chars
        if isinstance(token["id"], list):
            token["id"] = [i + index for i in token["id"]]
            for word in token["words"]:
                word["id"] += index
                if word["head"] is not None:
                    word["head"] += index
    return tokens


def _raw_tokens(lang: str, texts: list[str], model_dir: str) -> list[list[dict]]:
    pieces = [(i, offset, seg) for i, text in enumerate(texts) for offset, seg in _segments(text)]
    segs = [seg for _i, _offset, seg in pieces]
    if lang in ANALYZE_LANGS:
        # Batched: per-call overhead dominates for short chunks.
        docs = _get_spacy_pipeline(lang).pipe(segs, batch_size=_SPACY_BATCH_SIZE)
        seg_tokens = (_spacy_token_dicts(doc) for doc in docs)
    elif lang in SPACY_MODELS:
        # Bare tokenizer, not the full pipeline -- no tagger/lemmatizer
        # inference, even though the model is loaded.
        docs = _get_spacy_pipeline(lang).tokenizer.pipe(segs)
        seg_tokens = (_spacy_token_dicts(doc) for doc in docs)
    else:
        pipeline = _get_stanza_pipeline(lang, model_dir)
        seg_tokens = (_stanza_token_dicts(pipeline(seg)) for seg in segs)

    results: list[list[dict]] = [[] for _ in texts]
    for (i, offset, _seg), tokens in zip(pieces, seg_tokens):
        results[i].extend(_shift(tokens, offset, len(results[i])))
    return results


def _tokenize_batch(
    items: list[tuple[str, str]], model_dir: str
) -> list[tuple[str, list[dict]] | Exception]:
    """Tokenize (chunk_urn, primary_text) pairs, batching language
    identification and, per language, the models themselves.

    Returns one (lang, tokens) pair per item, or the exception that item
    raised: a batch that fails is retried an item at a time, so one bad
    chunk can't take down its neighbors."""
    results: list[tuple[str, list[dict]] | Exception] = [("", [])] * len(items)
    nonempty = [i for i, (_urn, text) in enumerate(items) if text.strip()]
    if not nonempty:
        return results

    docs = [StanzaDocument([], text=_langid_sample(items[i][1])) for i in nonempty]
    _get_langid(model_dir)(docs)
    by_lang: dict[str, list[int]] = {}
    for i, doc in zip(nonempty, docs):
        by_lang.setdefault(doc.lang, []).append(i)

    for lang, indices in by_lang.items():
        try:
            raw = _raw_tokens(lang, [items[i][1] for i in indices], model_dir)
        except Exception:
            raw = []
            for i in indices:
                try:
                    raw.extend(_raw_tokens(lang, [items[i][1]], model_dir))
                except Exception as exc:
                    raw.append(exc)
        for i, tokens in zip(indices, raw):
            urn, text = items[i]
            results[i] = (
                tokens if isinstance(tokens, Exception)
                else (lang, _finalize_tokens(urn, text, tokens))
            )
    return results


def _tokenize(
    chunk_urn: str, primary_text: str, model_dir: str
) -> tuple[str, list[dict]]:
    result = _tokenize_batch([(chunk_urn, primary_text)], model_dir)[0]
    if isinstance(result, Exception):
        raise result
    return result


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


def _load_tokens_manifest(sidecar_dir: Path) -> dict[str, dict]:
    try:
        with open(sidecar_dir / TOKENS_MANIFEST) as f:
            return json.load(f).get("chunks", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_tokens_manifest(sidecar_dir: Path, chunks: dict[str, dict]) -> None:
    path = sidecar_dir / TOKENS_MANIFEST
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False, indent=1))
    tmp.replace(path)


def _is_up_to_date(entry: dict | None, source_sha: str, sidecar: Path) -> bool:
    return (
        entry is not None
        and entry.get("sha") == source_sha
        and entry.get("pipeline") == pipeline_version(entry.get("lang", ""))
        and sidecar.exists()
    )


def _past(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _process_chunks(
    chunk_files: list[Path],
    sidecar_dir: Path,
    manifest: dict[str, dict],
    model_dir: str,
    force: bool,
    deadline: float | None = None,
) -> Counter[str]:
    """Tokenize one version dir's out-of-date chunks as a batch, writing
    their sidecars and updating `manifest` in place.

    Out-of-date chunks reached after `deadline` (a time.monotonic() value)
    are left alone and counted as "deferred": they get no manifest entry, so
    the next run picks them up.

    Returns generated/skipped/failed/deferred counts."""
    counts: Counter[str] = Counter()
    pending: list[tuple[Path, str, str, str]] = []  # (chunk, sha, urn, text)
    for chunk_file in chunk_files:
        source_sha = hashlib.sha256(chunk_file.read_bytes()).hexdigest()
        sidecar = sidecar_dir / token_sidecar_name(chunk_file)
        if not force and _is_up_to_date(manifest.get(chunk_file.name), source_sha, sidecar):
            counts["skipped"] += 1
            continue
        if _past(deadline):
            counts["deferred"] += 1
            continue
        try:
            cts_urn, primary_text = _primary_text(chunk_file)
        except Exception as exc:
            print(f"  FAILED: {chunk_file}: {exc}", file=sys.stderr)
            counts["failed"] += 1
            continue
        pending.append((chunk_file, source_sha, cts_urn, primary_text))

    for start in range(0, len(pending), _BATCH_SIZE):
        if _past(deadline):
            counts["deferred"] += len(pending) - start
            break
        batch = pending[start : start + _BATCH_SIZE]
        results = _tokenize_batch([(urn, text) for _f, _s, urn, text in batch], model_dir)
        compressor = zstandard.ZstdCompressor(level=_ZSTD_LEVEL)
        for (chunk_file, source_sha, cts_urn, _text), result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"  FAILED: {chunk_file}: {result}", file=sys.stderr)
                counts["failed"] += 1
                continue
            lang, tokens = result
            payload = json.dumps(
                {"urn": cts_urn, "lang": lang, "tokens": tokens}, ensure_ascii=False
            ).encode("utf-8")
            (sidecar_dir / token_sidecar_name(chunk_file)).write_bytes(
                compressor.compress(payload)
            )
            manifest[chunk_file.name] = {
                "sha": source_sha,
                "lang": lang,
                "pipeline": pipeline_version(lang),
            }
            counts["generated"] += 1
    return counts


def _iter_chunk_dirs(proto_dir: Path):
    """Yield (version_dir, [chunk files]) for every index.json under proto_dir."""
    for index_file in sorted(proto_dir.glob("**/index.json")):
        version_dir = index_file.parent
        with open(index_file) as f:
            chunks = json.load(f).get("chunks", [])
        chunk_files = [version_dir / entry["file"] for entry in chunks]
        yield version_dir, [c for c in chunk_files if c.exists()]


def _prune_stale_sidecars(sidecar_dir: Path, chunk_files: list[Path]) -> int:
    """Delete sidecars in sidecar_dir whose chunk no longer exists."""
    expected = {token_sidecar_name(c) for c in chunk_files}
    removed = 0
    for sidecar in sidecar_dir.glob("*.tokens.json.zst"):
        if sidecar.name not in expected:
            sidecar.unlink()
            removed += 1
    return removed


def _prune_missing_works(proto_dir: Path, tokens_dir: Path) -> int:
    """Delete sidecar trees whose proto version dir no longer exists."""
    removed = 0
    for manifest in sorted(tokens_dir.glob(f"**/{TOKENS_MANIFEST}")):
        sidecar_dir = manifest.parent
        rel_dir = sidecar_dir.relative_to(tokens_dir)
        if not (proto_dir / rel_dir / "index.json").exists():
            shutil.rmtree(sidecar_dir)
            removed += 1
    return removed


def tokenize_dir(
    proto_dir: Path,
    tokens_dir: Path | None,
    model_dir: str = "./stanza_models",
    force: bool = False,
    prune: bool = False,
    time_budget: float | None = None,
) -> dict:
    """Tokenize every compiled chunk under proto_dir.

    With `prune` (and a separate `tokens_dir`), also deletes sidecars for
    chunks and works that are no longer in proto_dir -- use it when
    proto_dir holds a complete compile of the corpus, as it does in CI.

    With `time_budget` (seconds), stops tokenizing once it's spent, between
    batches, and reports the chunks it didn't get to as "deferred"; output
    so far is complete and consistent, so rerunning continues where this
    left off. CI uses this to finish inside a job's time limit, and pushes
    what it has rather than losing it.

    Returns generated/skipped/failed/deferred (and pruned) counts."""
    proto_dir = proto_dir.resolve()
    generated = skipped = failed = deferred = pruned = 0
    deadline = time.monotonic() + time_budget if time_budget is not None else None

    for version_dir, chunk_files in _iter_chunk_dirs(proto_dir):
        if tokens_dir is not None:
            sidecar_dir = tokens_dir / version_dir.relative_to(proto_dir)
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            for name in PROTO_METADATA_FILES:
                if (version_dir / name).exists():
                    shutil.copyfile(version_dir / name, sidecar_dir / name)
        else:
            sidecar_dir = version_dir

        manifest = _load_tokens_manifest(sidecar_dir)
        counts = _process_chunks(
            chunk_files, sidecar_dir, manifest, model_dir, force, deadline
        )
        generated += counts["generated"]
        skipped += counts["skipped"]
        failed += counts["failed"]
        deferred += counts["deferred"]
        if counts["generated"] or counts["failed"]:
            print(
                f"{version_dir.relative_to(proto_dir)}: {counts['generated']} generated, "
                f"{counts['failed']} failed (so far: {generated} generated, "
                f"{skipped} skipped, {failed} failed)",
                flush=True,
            )

        live = {c.name for c in chunk_files}
        _write_tokens_manifest(
            sidecar_dir, {k: v for k, v in manifest.items() if k in live}
        )
        if prune and tokens_dir is not None:
            pruned += _prune_stale_sidecars(sidecar_dir, chunk_files)

    if prune and tokens_dir is not None:
        pruned += _prune_missing_works(proto_dir, tokens_dir)

    if deferred:
        print(f"Time budget spent: {deferred} chunks deferred to the next run.", flush=True)

    return {
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "deferred": deferred,
        "pruned": pruned,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Tokenize compiled chunk XML files in-process, lemmatizing "
        "Greek and Latin (no NLP server needed). Only chunks whose XML or "
        "pipeline version changed since the last run are reprocessed."
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
        help="Re-tokenize even if a sidecar is up to date",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete sidecars for chunks/works no longer in --proto-dir",
    )
    parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=None,
        help="Stop tokenizing after this long, leaving the rest for the next run",
    )
    args = parser.parse_args()

    result = tokenize_dir(
        args.proto_dir,
        args.tokens_dir,
        args.model_dir,
        args.force,
        args.prune,
        args.time_budget_minutes * 60 if args.time_budget_minutes is not None else None,
    )
    print(
        f"Tokenization: {result['generated']} generated, "
        f"{result['skipped']} skipped, {result['failed']} failed, "
        f"{result['deferred']} deferred, {result['pruned']} pruned."
    )


if __name__ == "__main__":
    main()
