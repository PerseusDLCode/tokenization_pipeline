from pathlib import Path

from mvp_tokenization import run_corpus


class _FakeResolver:
    def __init__(self, base_urn):
        self.base_urn = base_urn


class _FakeChunker:
    def __init__(self, base_urn):
        self.cts_resolver = _FakeResolver(base_urn)
        self.citation_chunks = []
        self.compiled_to = None

    def compile(self, output_path):
        self.compiled_to = output_path


def test_bad_base_urn_fails_that_file_only(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    for name in ("phi0474.phi042.perseus-lat2.xml", "phi0474.phi041.perseus-lat2.xml"):
        path = corpus / "data" / "phi0474" / name.split(".")[1] / name
        path.parent.mkdir(parents=True)
        path.write_text("<TEI/>")
    base_urns = {
        # <body xml:base> holding the filename instead of a URN.
        "phi0474.phi042.perseus-lat2.xml": "phi0474.phi042.perseus-lat2.xml",
        "phi0474.phi041.perseus-lat2.xml": "urn:cts:latinLit:phi0474.phi041.perseus-lat2",
    }
    monkeypatch.setattr(run_corpus, "_has_cite_structure", lambda path: True)
    monkeypatch.setattr(run_corpus, "LenientTEIDocument", lambda path: Path(path).name)
    monkeypatch.setattr(run_corpus, "Chunker", lambda name: _FakeChunker(base_urns[name]))

    entries = run_corpus.compile_corpus(corpus, tmp_path / "proto", force=False)
    status = {Path(e["path"]).name: e["status"] for e in entries}
    assert status == {
        "phi0474.phi041.perseus-lat2.xml": "compiled",
        "phi0474.phi042.perseus-lat2.xml": "failed",
    }
