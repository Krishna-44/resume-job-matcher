"""Ranking quality on the labelled sample set, against two simple baselines.

For each job, the 8 sample resumes are ranked and compared with relevance labels (0 = not a fit,
1 = partial, 2 = strong) that were written before the scorer existed. Reported:

* **NDCG@3** — are the right people at the top of each job's shortlist?
* **Top-1 hit rate** — is the first-ranked resume a strong fit?
* **Spearman ρ** — rank correlation between score and label across all 48 pairs.

Baselines: TF-IDF cosine similarity alone, and Jaccard overlap of the extracted skill sets.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np

from jobmatch import samples
from jobmatch.extract import parse_job, parse_resume
from jobmatch.pii import redact
from jobmatch.scoring import Matcher

EVAL_DATE = date(2026, 10, 1)  # fixed "today" so dated experience, and results, are reproducible

Scorer = Callable[[str, str], float]


def ndcg_at_k(relevances_in_ranked_order: list[int], k: int) -> float:
    def dcg(rels: list[int]) -> float:
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels[:k]))

    ideal = dcg(sorted(relevances_in_ranked_order, reverse=True))
    return dcg(relevances_in_ranked_order) / ideal if ideal else 0.0


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (as Spearman requires)."""
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values), dtype=float)
    for v in np.unique(values):
        tie = values == v
        ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])


def scorers(matcher: Matcher) -> dict[str, Scorer]:
    def full(resume: str, job: str) -> float:
        return matcher.match(resume, job).score

    def tfidf_only(resume: str, job: str) -> float:
        return matcher.similarity(redact(resume).text, job)[0]

    def skill_jaccard(resume: str, job: str) -> float:
        have = set(parse_resume(redact(resume).text, EVAL_DATE).skills)
        profile = parse_job(job)
        want = {s for group in profile.required + profile.preferred for s in group}
        return len(have & want) / len(have | want) if have | want else 0.0

    return {
        "jobmatch (full score)": full,
        "TF-IDF similarity only": tfidf_only,
        "skill overlap (Jaccard)": skill_jaccard,
    }


def evaluate(k: int = 3) -> dict[str, dict[str, float]]:
    matcher = Matcher(today=EVAL_DATE)
    resumes, jobs, labels = samples.resumes(), samples.jobs(), samples.labels()
    results = {}
    for name, score in scorers(matcher).items():
        ndcgs, hits, all_scores, all_labels = [], [], [], []
        for jid, job in jobs.items():
            scored = sorted(((score(text, job), labels[(rid, jid)]) for rid, text in resumes.items()),
                            key=lambda pair: -pair[0])
            rels = [rel for _, rel in scored]
            ndcgs.append(ndcg_at_k(rels, k))
            hits.append(rels[0] == 2)
            all_scores += [s for s, _ in scored]
            all_labels += rels
        results[name] = {
            f"ndcg@{k}": float(np.mean(ndcgs)),
            "top1_hit_rate": float(np.mean(hits)),
            "spearman": spearman(all_scores, all_labels),
        }
    return results


def format_table(results: dict[str, dict[str, float]]) -> str:
    lines = ["| Method | NDCG@3 | Top-1 hit rate | Spearman ρ |", "|---|---:|---:|---:|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['ndcg@3']:.3f} | {m['top1_hit_rate']:.2f} | {m['spearman']:.3f} |")
    return "\n".join(lines)
