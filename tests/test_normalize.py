import json
from pathlib import Path

import pytest

from mvp_tokenization.normalize import loose_key, match_key

FIXTURE = Path(__file__).parent / "fixtures" / "normalization.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['lang']}:{c['text']}" for c in CASES])
def test_shared_fixture(case):
    assert match_key(case["text"], case["lang"]) == case["match_key"]
    assert loose_key(case["text"], case["lang"]) == case["loose_key"]


def test_grave_matches_acute():
    assert match_key("δὲ", "grc") == match_key("δέ", "grc")


def test_accents_still_distinguish_greek_forms():
    assert match_key("μάνης", "grc") != match_key("μανῆς", "grc")
    assert loose_key("μάνης", "grc") == loose_key("μανῆς", "grc")
