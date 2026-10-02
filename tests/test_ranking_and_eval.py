from jobmatch import samples
from jobmatch.evaluation import evaluate, ndcg_at_k, spearman
from jobmatch.ranking import rank_jobs, rank_resumes, skill_gaps


def test_rank_resumes_puts_the_right_candidate_first(matcher):
    ranked = rank_resumes(matcher, samples.jobs()["devops_engineer"], samples.resumes())
    assert ranked[0][0] == "devops_engineer"
    scores = [r.score for _, r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_jobs_for_a_candidate(matcher):
    assert rank_jobs(matcher, samples.resumes()["frontend_dev"], samples.jobs())[0][0] == "frontend_developer"


def test_skill_gaps_are_sorted_by_how_many_jobs_need_them(matcher):
    wanted = ("backend_engineer", "devops_engineer", "fullstack_developer")
    jobs = {k: v for k, v in samples.jobs().items() if k in wanted}
    gaps = skill_gaps(matcher, samples.resumes()["ml_student"], jobs)
    counts = [g.jobs for g in gaps]
    assert counts == sorted(counts, reverse=True)
    kubernetes = next(g for g in gaps if g.skill.startswith("Kubernetes"))
    assert set(kubernetes.missing_in) | set(kubernetes.partial_in) <= set(jobs)


def test_metric_math():
    assert ndcg_at_k([2, 1, 0], 3) == 1.0
    assert ndcg_at_k([0, 0, 2], 3) < ndcg_at_k([2, 0, 0], 3)
    assert spearman([1, 2, 3], [10, 20, 30]) == 1.0
    assert round(spearman([1, 2, 3], [3, 2, 1]), 6) == -1.0


def test_evaluation_quality_floor():
    """Guard rail on the labelled sample set (see README for the numbers and their caveats)."""
    full = evaluate()["jobmatch (full score)"]
    assert full["top1_hit_rate"] == 1.0
    assert full["ndcg@3"] >= 0.9
    assert full["spearman"] >= 0.7
