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
