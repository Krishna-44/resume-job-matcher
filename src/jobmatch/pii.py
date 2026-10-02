"""Blind screening: remove personal information before anything is scored.

Names, contact details and personal attributes (date of birth, age, gender, marital status,
nationality, religion, photo references) say nothing about whether someone can do a job, and are a
known source of bias. They are stripped first, so no score can depend on them
(``tests/test_pii.py`` checks that changing them leaves the score identical).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github|gitlab|twitter|x)\.com/\S*", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,5}\)[\s.-]?)?\d[\d\s.-]{5,16}\d(?!\w)")
_PERSONAL = re.compile(
    r"^\s*(?:date of birth|d\.?o\.?b\.?|birth ?date|age|gender|sex|marital status|nationality|"
    r"religion|caste|father'?s name|mother'?s name|photo)\s*[:\-–]\s*.+$",
    re.IGNORECASE | re.MULTILINE,
)
_NAME_LINE = re.compile(r"^[A-Z][a-zA-Z'’.-]+(?:\s+[A-Z][a-zA-Z'’.-]+){1,3}$")
_NOT_A_NAME = re.compile(
    r"\b(resume|curriculum|vitae|summary|profile|engineer|developer|scientist|analyst|manager|"
    r"intern|student|experience|education|skills)\b",
    re.IGNORECASE,
)


@dataclass
class Redaction:
    text: str
    removed: Counter[str] = field(default_factory=Counter)


def _phone_like(match: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", match.group())
    return 10 <= len(digits) <= 15


def redact(text: str) -> Redaction:
    removed: Counter[str] = Counter()

    def sub(pattern: re.Pattern[str], label: str, value: str, keep=lambda m: True) -> None:
        nonlocal text

        def repl(m: re.Match[str]) -> str:
            if not keep(m):
                return m.group()
            removed[label] += 1
            return value

        text = pattern.sub(repl, text)

    sub(_EMAIL, "email", "[EMAIL]")
    sub(_URL, "url", "[URL]")
    sub(_PHONE, "phone", "[PHONE]", keep=_phone_like)
    sub(_PERSONAL, "personal detail", "[PERSONAL DETAIL]")

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            candidate = line.strip()
            if _NAME_LINE.match(candidate) and not _NOT_A_NAME.search(candidate):
                lines[i] = "[NAME]"
                removed["name"] += 1
            break
    return Redaction("\n".join(lines), removed)
