"""Match keys for the corpus search index.

The search index (build_index.py) stores two keys per form and lemma, and
MinimumViablePerseus's /search page recomputes the same keys in the browser
(src/mvp/site/static/js/search/normalize.js) -- both for what the user types
and for the Morpheus forms it reads out of pdl-morph-server's morph.db. The
two implementations must agree exactly; tests/fixtures/normalization.json is
the shared contract, and both test suites run against it. Change one side,
change the other, and add a fixture case.

- ``match_key``: accent-preserving, but erases the differences that don't
  distinguish words -- case, grave vs. acute (a grave is just an acute
  before another word), final vs. medial sigma, the various elision
  apostrophes, and Latin u/v and i/j. This is what makes a Morpheus form
  like ``δέ`` match the corpus's ``δὲ``. Latin marks (macrons, the odd
  accent) never distinguish forms, so for Latin this is the same as
  ``loose_key``.
- ``loose_key``: ``match_key`` with every combining mark stripped too, for
  "ignore accents" searches.
"""

from __future__ import annotations

import unicodedata

# Elision marks as they actually occur in the corpora and in Morpheus's
# forms, all folded to a plain ASCII apostrophe.
APOSTROPHES = frozenset("'’ʼ᾽′΄")

_COMBINING_GRAVE = "̀"
_COMBINING_ACUTE = "́"
_COMBINING_TILDE = "̃"
_COMBINING_PERISPOMENI = "͂"

LATIN_LANGS = frozenset({"la", "lat"})


def _fold(text: str, lang: str, strip_marks: bool) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if ch in APOSTROPHES:
            out.append("'")
        elif unicodedata.category(ch).startswith("M"):
            if strip_marks:
                continue
            if ch == _COMBINING_GRAVE:
                out.append(_COMBINING_ACUTE)
            elif ch == _COMBINING_TILDE:
                out.append(_COMBINING_PERISPOMENI)
            else:
                out.append(ch)
        else:
            out.append(ch)
    folded = "".join(out).lower().replace("ς", "σ")
    if lang in LATIN_LANGS:
        folded = folded.replace("j", "i").replace("v", "u")
    return unicodedata.normalize("NFC", folded)


def match_key(text: str, lang: str) -> str:
    return _fold(text, lang, strip_marks=lang in LATIN_LANGS)


def loose_key(text: str, lang: str) -> str:
    return _fold(text, lang, strip_marks=True)


def is_apostrophe(text: str) -> bool:
    return bool(text) and all(ch in APOSTROPHES for ch in text)
