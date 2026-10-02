"""Structured information from resumes and job descriptions.

* Resumes: sections, skills (with the line they appear on as evidence), years of experience from
  employment date ranges (overlaps merged), highest degree.
* Job descriptions: required vs. preferred skills (by section), alternatives ("Tableau or Power
  BI"), minimum years of experience, degree requirement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from jobmatch.taxonomy import find_skill_mentions

RESUME_SECTIONS = {
    "summary": ("summary", "profile", "objective", "about me", "professional summary"),
    "experience": ("experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "internships", "internship"),
    "education": ("education", "academics", "academic background"),
    "skills": ("skills", "technical skills", "core skills", "tools", "technologies", "tech stack"),
    "projects": ("projects", "personal projects", "academic projects", "selected projects"),
    "certifications": ("certifications", "certificates", "courses", "licenses"),
}
JOB_SECTIONS = {
    "required": ("requirements", "qualifications", "required qualifications", "must have", "must-have",
                 "what you bring", "what we're looking for", "you have", "minimum qualifications",
                 "required skills", "who you are"),
    "preferred": ("preferred", "preferred qualifications", "nice to have", "nice-to-have", "bonus",
                  "bonus points", "pluses", "good to have", "preferred skills"),
    "about": ("about", "about us", "about the role", "overview", "responsibilities",
              "what you'll do", "what you will do", "the role", "benefits", "perks"),
}

DEGREE_LEVELS = {0: "none found", 1: "diploma", 2: "bachelor's", 3: "master's", 4: "doctorate"}
_DEGREES = [
    (4, r"\b(?:ph\.?\s?d|doctorate|doctor of philosophy)\b"),
    (3, r"\b(?:master'?s|master of|m\.?\s?tech|m\.?\s?sc|m\.?\s?eng\b|mba|m\.e\.|m\.a\.|m\.?s\.?(?=\s+(?:in|of)\b))"),
    (2, r"\b(?:bachelor'?s|bachelor of|b\.?\s?tech|b\.?\s?sc|b\.?\s?eng\b|b\.e\.|b\.a\.|b\.?\s?com|bca"
        r"|undergraduate degree|b\.?s\.?(?=\s+(?:in|of)\b))"),
    (1, r"\b(?:diploma|associate'?s degree)\b"),
]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"


def _date(p: str) -> str:
    """A date like "Jan 2021", "06/2020" or "2019"; groups are prefixed with ``p``."""
    return (rf"(?:(?P<{p}m>{_MONTH_RE})\s+(?P<{p}y>\d{{4}})"
            rf"|(?P<{p}n>\d{{1,2}})/(?P<{p}ny>\d{{4}})|(?P<{p}yo>\d{{4}}))")


_RANGE = re.compile(
    _date("a") + r"\s*(?:-|–|—|to|until)\s*(?:" + _date("b")
    + r"|(?P<present>present|current|now|today|date))",
    re.IGNORECASE,
)
_MIN_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)\b", re.IGNORECASE)


def split_sections(text: str, known: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Group lines under the most recent recognised heading (``"_"`` before the first one)."""
    sections: dict[str, list[str]] = {"_": []}
    current = "_"
    for line in text.splitlines():
        label = line.strip().strip("#*:•-– ").strip().lower()
        is_heading = 0 < len(label) <= 40 and not re.search(r"[.!?]$", line.strip())
        matched = next((k for k, names in known.items() if label in names), None) if is_heading else None
        if matched:
            current = matched
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def _month_index(m: re.Match[str], p: str) -> int | None:
    if m.group(f"{p}m"):
        return int(m.group(f"{p}y")) * 12 + _MONTHS[m.group(f"{p}m")[:3].lower()] - 1
    if m.group(f"{p}n"):
        month = int(m.group(f"{p}n"))
        return int(m.group(f"{p}ny")) * 12 + month - 1 if 1 <= month <= 12 else None
    if m.group(f"{p}yo"):
        return int(m.group(f"{p}yo")) * 12  # year only: assume January
    return None


def experience_years(text: str, today: date | None = None) -> float:
    """Total years covered by employment date ranges, with overlapping roles merged."""
    today = today or date.today()
    now = today.year * 12 + today.month - 1
    intervals = []
    for m in _RANGE.finditer(text):
        start = _month_index(m, "a")
        end = now if m.group("present") else _month_index(m, "b")
        if start is None or end is None or end < start or start > now:
            continue
        intervals.append((start, min(end, now) + 1))
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return round(sum(e - s for s, e in merged) / 12, 1)


def degree_levels(text: str) -> set[int]:
    """Every degree level mentioned in ``text``."""
    lowered = text.lower()
    return {level for level, pattern in _DEGREES if re.search(pattern, lowered)}


def degree_level(text: str) -> int:
    """Highest degree mentioned (for resumes)."""
    return max(degree_levels(text), default=0)


def _evidence(text: str, span: tuple[int, int]) -> str:
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    line = text[start : end if end != -1 else len(text)].strip(" •-*\t")
    return line if len(line) <= 140 else line[:137] + "…"


def _best_evidence(text: str, spans: list[tuple[int, int]]) -> str:
    """Prefer a line that shows the skill in use over a comma-separated skills list."""
    lines = [_evidence(text, span) for span in spans]
    return next((line for line in lines if line.count(",") < 3), lines[0])


@dataclass
class ResumeProfile:
    skills: dict[str, str]  # skill → evidence line
    years: float
    degree: int
    sections: list[str]


def parse_resume(text: str, today: date | None = None) -> ResumeProfile:
    sections = split_sections(text, RESUME_SECTIONS)
    mentions = find_skill_mentions(text)
    skills = {name: _best_evidence(text, spans) for name, spans in sorted(mentions.items())}
    # Only employment dates count as experience; study dates in Education must not.
    experience_text = sections.get("experience") or "\n".join(
        v for k, v in sections.items() if k not in ("education", "certifications")
    )
    return ResumeProfile(
        skills=skills,
        years=experience_years(experience_text, today),
        degree=degree_level(sections.get("education", text)),
        sections=[k for k in sections if k != "_"],
    )


Requirement = tuple[str, ...]  # one skill, or alternatives that each satisfy it


def _requirements(text: str) -> list[Requirement]:
    """Skills mentioned in ``text``; "X or Y" and "X/Y" become one requirement with alternatives."""
    groups: list[list[str]] = []
    for line in text.splitlines():
        mentions = sorted(
            ((span, name) for name, spans in find_skill_mentions(line).items() for span in spans)
        )
        previous_end = None
        for (start, end), name in mentions:
            joined = previous_end is not None and re.fullmatch(
                r"\s*(?:,\s*)?(?:or|/|\|)\s*", line[previous_end:start], re.IGNORECASE
            )
            if joined and name not in groups[-1]:
                groups[-1].append(name)
            elif not any(name in g for g in groups):
                groups.append([name])
            previous_end = end
    return [tuple(g) for g in groups]


@dataclass
class JobProfile:
    required: list[Requirement]
    preferred: list[Requirement] = field(default_factory=list)
    min_years: float = 0.0
    degree: int = 0


def parse_job(text: str) -> JobProfile:
    sections = split_sections(text, JOB_SECTIONS)
    preferred = _requirements(sections.get("preferred", ""))
    if "required" in sections:
        required = _requirements(sections["required"])
        # Skills named only in the description of the role still count, as preferred.
        seen = {s for group in required + preferred for s in group}
        preferred += [g for g in _requirements(sections.get("about", "") + "\n" + sections.get("_", ""))
                      if not seen & set(g)]
    else:
        required = [g for g in _requirements(text) if not {s for p in preferred for s in p} & set(g)]
    years = [int(m.group(1)) for m in _MIN_YEARS.finditer(sections.get("required", text))]
    # "Bachelor's or Master's degree" means a bachelor's is enough: take the lowest level named.
    degree = min(degree_levels(sections.get("required", text)), default=0)
    return JobProfile(required, preferred, float(min(years)) if years else 0.0, degree)
