"""Redaction of personal data and secrets before anything leaves the call path.

Used on every string levrouter writes (log lines, the JSONL file, span attributes, error
text) and on the classifier state sent to a lev-http server. It is pattern based, so it
catches structured identifiers (emails, card numbers, keys ...), not names or free-text
descriptions of people. Treat it as defense in depth, not as an anonymizer.
"""

from __future__ import annotations

import re


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


# Order matters: secrets first (they can contain digit runs), then identifiers.
_PATTERNS: list[tuple[str, re.Pattern, object]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[PRIVATE_KEY]"),
    ("bearer", re.compile(r"(?i)\bbearer\s+[a-z0-9._\-~+/=]{16,}"), "Bearer [SECRET]"),
    ("jwt", re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}"), "[JWT]"),
    ("api_key", re.compile(
        r"\b(?:sk-ant-[\w-]{16,}|sk-(?:proj-)?[\w-]{20,}|gsk_[\w]{20,}|hf_[\w]{20,}|xai-[\w]{20,}"
        r"|AIza[\w-]{30,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[\w]{30,}|github_pat_[\w]{30,}"
        r"|xox[baprs]-[\w-]{10,}|sk_live_[\w]{16,}|rk_live_[\w]{16,})"
    ), "[SECRET]"),
    ("secret_assignment", re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b(\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}\3"
    ), r"\1\2[SECRET]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[EMAIL]"),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b"), "[IBAN]"),
    ("card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _card),
    ("us_ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    ("phone", re.compile(r"(?<![\w.])\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"), "[PHONE]"),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), "[IP]"),
]

KINDS = tuple(name for name, _, _ in _PATTERNS)


def redact(text: str | None, enabled: bool = True) -> str | None:
    if not text or not enabled:
        return text
    for _, pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def find(text: str) -> list[str]:
    """Kinds of PII / secrets present in text (for tests and audits)."""
    return [name for name, pattern, _ in _PATTERNS if pattern.search(text or "")]
