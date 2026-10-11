"""Notice a user asking the same question again: a sign the current model isn't answering well."""

from __future__ import annotations

import re

from .features import RequestFeatures

SIMILAR = 0.5
MIN_WORDS = 3  # "thanks" or "ok" sent again is not a repeated question

RETRY_PHRASES = (
    "try again", "that's wrong", "that is wrong", "thats wrong", "not what i asked", "still wrong",
    "still not working", "doesn't work", "does not work", "doesnt work", "wrong answer", "you didn't answer",
)

_STOP = frozenset(
    "the and for are but not you your with this that what how why when where which can could would should "
    "does did has have had was were will from into about please just like then than them they there their "
    "its also any all our out get got use using some again".split()
)


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))
                     if len(w) >= 3 and w not in _STOP)


def _similar(a: frozenset[str], b: frozenset[str]) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= SIMILAR


def repeat_count(features: RequestFeatures) -> int:
    """How many earlier user messages the latest one repeats (rephrasings and "that's wrong, try again")."""
    earlier = features.earlier_user
    if not earlier:
        return 0
    latest = _words(features.last_user)
    count = sum(1 for text in earlier if _similar(latest, _words(text))) if len(latest) >= MIN_WORDS else 0
    text = features.last_user.lower().replace("’", "'")
    if any(phrase in text for phrase in RETRY_PHRASES):
        count += 1
    return min(count, len(earlier))
