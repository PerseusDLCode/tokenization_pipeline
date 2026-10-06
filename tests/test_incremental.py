import json
from pathlib import Path

import pytest

from mvp_tokenization import run_tokenizer


@pytest.fixture
def calls(monkeypatch):
    """Stub out the models: record which chunks actually get tokenized."""
    seen: list[str] = []

    def fake_primary_text(chunk_file: Path):
        seen.append(chunk_file.name)
        return f"urn:cts:latinLit:phi0690.phi003.perseus-lat2:{chunk_file.stem}", "arma"

    def fake_tokenize_batch(items, model_dir):
        return [
            ("la", [{"text": "arma", "urn": f"{urn}@arma[1]", "identifier": "arma[1]"}])
            for urn, _text in items
        ]

    monkeypatch.setattr(run_tokenizer, "_primary_text", fake_primary_text)
    monkeypatch.setattr(run_tokenizer, "_tokenize_batch", fake_tokenize_batch)
    return seen


def _proto(tmp_path: Path, chunks: dict[str, str]) -> Path:
    version = tmp_path / "proto" / "latinLit" / "phi0690" / "phi003" / "perseus-lat2"
    version.mkdir(parents=True, exist_ok=True)
    for old in version.glob("*.xml"):
        old.unlink()
    for name, body in chunks.items():
        (version / name).write_text(body)
    (version / "index.json").write_text(
        json.dumps({"chunks": [{"file": n} for n in chunks]})
    )
    (version / "metadata.json").write_text(json.dumps({"document": {"title": "Aeneid"}}))
    return tmp_path / "proto"


def test_only_changed_chunks_are_reprocessed(tmp_path, calls):
    proto = _proto(tmp_path, {"1.xml": "<a/>", "2.xml": "<b/>"})
    tokens = tmp_path / "tokens"

    assert run_tokenizer.tokenize_dir(proto, tokens)["generated"] == 2
    calls.clear()
    assert run_tokenizer.tokenize_dir(proto, tokens)["skipped"] == 2
    assert calls == []

    _proto(tmp_path, {"1.xml": "<a/>", "2.xml": "<b>edited</b>"})
    result = run_tokenizer.tokenize_dir(proto, tokens)
    assert (result["generated"], result["skipped"]) == (1, 1)
    assert calls == ["2.xml"]


def test_metadata_is_copied(tmp_path, calls):
    proto = _proto(tmp_path, {"1.xml": "<a/>"})
    tokens = tmp_path / "tokens"
    run_tokenizer.tokenize_dir(proto, tokens)
    version = tokens / "latinLit" / "phi0690" / "phi003" / "perseus-lat2"
    assert (version / "metadata.json").exists()
    assert (version / "index.json").exists()


def test_pipeline_version_change_forces_rebuild(tmp_path, calls, monkeypatch):
    proto = _proto(tmp_path, {"1.xml": "<a/>"})
    tokens = tmp_path / "tokens"
    run_tokenizer.tokenize_dir(proto, tokens)
    monkeypatch.setattr(run_tokenizer, "SIDECAR_SCHEMA", run_tokenizer.SIDECAR_SCHEMA + 1)
    assert run_tokenizer.tokenize_dir(proto, tokens)["generated"] == 1


def test_prune_removes_deleted_chunks_and_works(tmp_path, calls):
    proto = _proto(tmp_path, {"1.xml": "<a/>", "2.xml": "<b/>"})
    tokens = tmp_path / "tokens"
    run_tokenizer.tokenize_dir(proto, tokens)
    version = tokens / "latinLit" / "phi0690" / "phi003" / "perseus-lat2"

    _proto(tmp_path, {"1.xml": "<a/>"})
    assert run_tokenizer.tokenize_dir(proto, tokens, prune=True)["pruned"] == 1
    assert sorted(p.name for p in version.glob("*.zst")) == ["1.tokens.json.zst"]

    gone = tokens / "latinLit" / "phi0959" / "phi001" / "perseus-lat2"
    gone.mkdir(parents=True)
    (gone / run_tokenizer.TOKENS_MANIFEST).write_text("{}")
    run_tokenizer.tokenize_dir(proto, tokens, prune=True)
    assert not gone.exists()
    assert version.exists()


def test_time_budget_defers_and_next_run_resumes(tmp_path, calls):
    proto = _proto(tmp_path, {"1.xml": "<a/>", "2.xml": "<b/>"})
    tokens = tmp_path / "tokens"

    result = run_tokenizer.tokenize_dir(proto, tokens, time_budget=0)
    assert (result["generated"], result["deferred"]) == (0, 2)
    assert calls == []
    version = tokens / "latinLit" / "phi0690" / "phi003" / "perseus-lat2"
    assert (version / "metadata.json").exists()

    result = run_tokenizer.tokenize_dir(proto, tokens)
    assert (result["generated"], result["deferred"]) == (2, 0)


def test_segments_cover_text_and_respect_limit():
    text = ("Μῆνιν ἄειδε θεά. " * 40) + ("x" * 300) + (" ἄλγε’ ἔθηκε" * 30)
    segments = run_tokenizer._segments(text, limit=100)
    assert "".join(seg for _off, seg in segments) == text
    assert all(len(seg) <= 100 for _off, seg in segments)
    assert all(text[off : off + len(seg)] == seg for off, seg in segments)
    # Sentence breaks are preferred, with the space kept on the left.
    assert segments[0][1].endswith(". ")


def test_segmented_tokens_map_back_to_source(monkeypatch):
    text = "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος οὐλομένην, ἣ μυρία Ἀχαιοῖς ἄλγε ἔθηκε. " * 6
    (whole,) = run_tokenizer._raw_tokens("grc", [text], "./stanza_models")
    monkeypatch.setattr(run_tokenizer, "SEGMENT_CHARS", 120)
    assert len(run_tokenizer._segments(text)) > 1
    (segmented,) = run_tokenizer._raw_tokens("grc", [text], "./stanza_models")
    assert [(t["start_char"], t["end_char"], t["whitespace"]) for t in segmented] == [
        (t["start_char"], t["end_char"], t["whitespace"]) for t in whole
    ]
    assert [t["id"] for t in segmented] == [[i] for i in range(len(segmented))]
    assert all(text[t["start_char"] : t["end_char"]] == t["text"] for t in segmented)


def test_langid_sample_is_bounded():
    short = "μῆνιν ἄειδε θεά"
    assert run_tokenizer._langid_sample(short) == short
    long = "a" * 50_000 + "b" * 50_000 + "c" * 50_000
    sample = run_tokenizer._langid_sample(long)
    assert len(sample) <= run_tokenizer.LANGID_SAMPLE_CHARS + 2
    assert sample.startswith("a") and "b" in sample and sample.endswith("c")
