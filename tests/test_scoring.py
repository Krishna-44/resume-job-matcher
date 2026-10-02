from jobmatch import samples
from jobmatch.scoring import BANDS, assess_requirement, coverage


def test_requirement_assessment():
    skills = {"MySQL": "MySQL at X", "Python": "Python at Y"}
    assert assess_requirement(("Python",), skills).status == "matched"
    partial = assess_requirement(("PostgreSQL",), skills)
    assert (partial.status, partial.via) == ("partial", "MySQL")
    assert assess_requirement(("Python", "R"), {"R": "R"}).status == "matched"
    assert assess_requirement(("Kubernetes",), skills).status == "missing"
    assert coverage([assess_requirement(("Python",), skills), partial]) == 0.75
    assert coverage([]) is None


def test_every_resume_scores_highest_on_its_own_role(matcher):
    pairs = {"ml_student": "ml_engineer_intern", "frontend_dev": "frontend_developer",
             "data_analyst": "data_analyst", "backend_java": "backend_engineer",
             "devops_engineer": "devops_engineer", "fullstack_mern": "fullstack_developer"}
    jobs = samples.jobs()
    for rid, best in pairs.items():
        scores = {jid: matcher.match(samples.resumes()[rid], text).score for jid, text in jobs.items()}
        assert max(scores, key=scores.get) == best, (rid, scores)


def test_result_explains_itself(matcher):
    r = matcher.match(samples.resumes()["data_scientist"], samples.jobs()["ml_engineer_intern"])
    statuses = {m.requirement: m.status for m in r.required}
    assert statuses["Python"] == "matched" and statuses["PyTorch"] == "partial"
    assert statuses["Git"] == "missing"
    via = next(m.via for m in r.required if m.requirement == "PyTorch")
    assert via in {"TensorFlow", "Keras"}
    assert any("PyTorch" in tip and via in tip for tip in r.suggestions)
    assert abs(sum(r.weights.values()) - 1) < 0.01  # weights are rounded for display
    assert r.band in {label for _, label in BANDS}


def test_unrelated_experience_does_not_count(matcher):
    r = matcher.match(samples.resumes()["mechanical_engineer"], samples.jobs()["devops_engineer"])
    assert r.candidate_years >= 3  # plenty of years...
    assert r.components["experience"] < 0.2  # ...but not in anything the role needs
    assert r.band == "Weak match"


def test_components_that_do_not_apply_are_dropped(matcher):
    r = matcher.match(samples.resumes()["ml_student"], "Requirements\n- Python\n- SQL")
    assert set(r.components) == {"required_skills", "text_similarity"}
