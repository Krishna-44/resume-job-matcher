"""jobmatch — explainable resume ↔ job matching with blind screening."""

from jobmatch.extract import parse_job, parse_resume
from jobmatch.pii import redact
from jobmatch.ranking import rank_jobs, rank_resumes, skill_gaps
from jobmatch.scoring import Matcher, MatchResult

__all__ = ["MatchResult", "Matcher", "parse_job", "parse_resume", "rank_jobs", "rank_resumes", "redact", "skill_gaps"]

__version__ = "0.1.0"
