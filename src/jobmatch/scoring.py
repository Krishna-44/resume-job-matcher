"""Explainable resume ↔ job scoring.

The score (0-100) is a weighted mix of four parts, each between 0 and 1:

=================  ======  ==============================================================
Component          Weight  Meaning
=================  ======  ==============================================================
required skills    0.45    share of required skills the resume shows (half credit for a
                           related skill, e.g. MySQL for PostgreSQL)
preferred skills   0.15    the same for nice-to-haves
text similarity    0.25    TF-IDF cosine similarity of the two documents
experience         0.15    dated experience ÷ years asked for (capped at 1), scaled by
                           required-skill coverage so only relevant experience counts
=================  ======  ==============================================================

Components that don't apply (no preferred skills listed, no years asked for) are dropped and the
remaining weights rescaled, so a posting is never penalised for being short.

Every number comes with its reasons: which skills matched (and the resume line that shows it),
which were partial or missing, and concrete suggestions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from jobmatch import samples
from jobmatch.extract import DEGREE_LEVELS, JobProfile, Requirement, ResumeProfile, parse_job, parse_resume
from jobmatch.pii import redact
from jobmatch.taxonomy import related

WEIGHTS = {"required_skills": 0.45, "preferred_skills": 0.15, "text_similarity": 0.25, "experience": 0.15}
FULL_SIMILARITY = 0.35  # cosine at which the text-similarity component maxes out
BANDS = ((75, "Strong match"), (55, "Good match"), (35, "Partial match"), (0, "Weak match"))


@dataclass
class SkillMatch:
    requirement: str  # e.g. "PostgreSQL" or "Tableau or Power BI"
    status: Literal["matched", "partial", "missing"]
    via: str | None = None  # resume skill that satisfied it, fully or partly
    evidence: str | None = None  # resume line that shows it


@dataclass
class MatchResult:
    score: float
    band: str
    components: dict[str, float]
    weights: dict[str, float]
    required: list[SkillMatch]
    preferred: list[SkillMatch]
    candidate_years: float
    required_years: float
    candidate_degree: str
    required_degree: str
    shared_terms: list[str]
    suggestions: list[str]
    redacted: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def assess_requirement(req: Requirement, skills: dict[str, str]) -> SkillMatch:
    label = " or ".join(req)
    for skill in req:
        if skill in skills:
            return SkillMatch(label, "matched", skill, skills[skill])
    for skill in req:
        for have, evidence in skills.items():
            if related(skill, have):
                return SkillMatch(label, "partial", have, evidence)
    return SkillMatch(label, "missing")


def coverage(matches: Iterable[SkillMatch]) -> float | None:
    matches = list(matches)
    if not matches:
        return None
    credit = {"matched": 1.0, "partial": 0.5, "missing": 0.0}
    return sum(credit[m.status] for m in matches) / len(matches)


class Matcher:
    def __init__(self, background: Iterable[str] | None = None, *, today: date | None = None) -> None:
        if background is None:
            background = [*samples.resumes().values(), *samples.jobs().values()]
        self.background = [redact(t).text for t in background]
        self.today = today

    def similarity(self, resume: str, job: str, top: int = 8) -> tuple[float, list[str]]:
        """TF-IDF cosine similarity, with IDF estimated on a background corpus plus the pair."""
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
        vectorizer.fit([*self.background, resume, job])
        a, b = vectorizer.transform([resume, job]).toarray()
        cosine = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
        contribution = a * b
        order = np.argsort(-contribution)[:top]
        terms = vectorizer.get_feature_names_out()
        return cosine, [str(terms[i]) for i in order if contribution[i] > 0]

    def match(self, resume_text: str, job_text: str) -> MatchResult:
        redaction = redact(resume_text)
        resume = parse_resume(redaction.text, self.today)
        job = parse_job(job_text)
        return self._score(resume, job, redaction.text, job_text, dict(redaction.removed))

    def _score(self, resume: ResumeProfile, job: JobProfile, resume_text: str, job_text: str,
               redacted: dict[str, int]) -> MatchResult:
        required = [assess_requirement(r, resume.skills) for r in job.required]
        preferred = [assess_requirement(r, resume.skills) for r in job.preferred]
        cosine, shared = self.similarity(resume_text, job_text)

        required_cov = coverage(required)
        # Years only count as far as they're relevant: five years as a mechanical engineer don't
        # meet "3+ years of DevOps". Scale tenure by how much of the required skill set is present.
        relevance = required_cov if required_cov is not None else 1.0
        components: dict[str, float | None] = {
            "required_skills": required_cov,
            "preferred_skills": coverage(preferred),
            "text_similarity": min(1.0, cosine / FULL_SIMILARITY),
            "experience": min(1.0, resume.years / job.min_years) * relevance if job.min_years else None,
        }
        active = {k: v for k, v in components.items() if v is not None}
        total_weight = sum(WEIGHTS[k] for k in active)
        weights = {k: WEIGHTS[k] / total_weight for k in active}
        score = round(100 * sum(weights[k] * active[k] for k in active), 1)
        band = next(label for cutoff, label in BANDS if score >= cutoff)

        return MatchResult(
            score=score,
            band=band,
            components={k: round(v, 3) for k, v in active.items()},
            weights={k: round(v, 3) for k, v in weights.items()},
            required=required,
            preferred=preferred,
            candidate_years=resume.years,
            required_years=job.min_years,
            candidate_degree=DEGREE_LEVELS[resume.degree],
            required_degree=DEGREE_LEVELS[job.degree] if job.degree else "not specified",
            shared_terms=shared,
            suggestions=_suggestions(required, preferred, resume, job),
            redacted=redacted,
        )


def _suggestions(required: list[SkillMatch], preferred: list[SkillMatch], resume: ResumeProfile,
                 job: JobProfile, limit: int = 5) -> list[str]:
    tips = []
    for m in required:
        if m.status == "missing":
            tips.append(f"Required: {m.requirement}. If you've used it, add a bullet that shows it "
                        "(a project, what you built, the result).")
        elif m.status == "partial":
            tips.append(f"Required: {m.requirement}. You list {m.via}, which is closely related — "
                        f"mention any {m.requirement} exposure and make the overlap explicit.")
    if job.min_years and resume.years < job.min_years:
        tips.append(f"The role asks for {job.min_years:g}+ years; the dated experience on your resume adds up "
                    f"to {resume.years:g}. Make sure internships and freelance work have dates.")
    for m in preferred:
        if m.status == "missing":
            tips.append(f"Nice to have: {m.requirement} — worth highlighting if you have it.")
    if job.degree and resume.degree < job.degree:
        tips.append(f"The posting mentions a {DEGREE_LEVELS[job.degree]} degree; equivalent experience is "
                    "often accepted, so lead with projects and results.")
    return tips[:limit]
