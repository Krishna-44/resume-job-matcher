"""Many-to-one use cases: rank candidates for a job, rank jobs for a candidate, find skill gaps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from jobmatch.scoring import Matcher, MatchResult


def rank_resumes(matcher: Matcher, job: str, resumes: dict[str, str]) -> list[tuple[str, MatchResult]]:
    """Recruiter view: candidates for one job, best first."""
    results = [(rid, matcher.match(text, job)) for rid, text in resumes.items()]
    return sorted(results, key=lambda r: -r[1].score)


def rank_jobs(matcher: Matcher, resume: str, jobs: dict[str, str]) -> list[tuple[str, MatchResult]]:
    """Candidate view: jobs for one resume, best fit first."""
    results = [(jid, matcher.match(resume, text)) for jid, text in jobs.items()]
    return sorted(results, key=lambda r: -r[1].score)


@dataclass
class SkillGap:
    skill: str
    missing_in: list[str] = field(default_factory=list)  # jobs that require it and the resume lacks
    partial_in: list[str] = field(default_factory=list)  # jobs where a related skill gives partial credit
    related_skill: str | None = None

    @property
    def jobs(self) -> int:
        return len(self.missing_in) + len(self.partial_in)


def skill_gaps(matcher: Matcher, resume: str, jobs: dict[str, str]) -> list[SkillGap]:
    """Required skills the resume lacks across a set of target jobs, most common first.

    This turns "I'm applying to these five roles" into a learning plan: the skill missing from four
    of them matters more than the one missing from one.
    """
    gaps: dict[str, SkillGap] = defaultdict(lambda: SkillGap(""))
    for jid, text in jobs.items():
        for m in matcher.match(resume, text).required:
            if m.status == "matched":
                continue
            gap = gaps[m.requirement]
            gap.skill = m.requirement
            if m.status == "missing":
                gap.missing_in.append(jid)
            else:
                gap.partial_in.append(jid)
                gap.related_skill = m.via
    return sorted(gaps.values(), key=lambda g: (-g.jobs, -len(g.missing_in), g.skill))
