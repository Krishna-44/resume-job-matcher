import pytest
from fastapi.testclient import TestClient

from jobmatch import samples
from jobmatch.api import create_app


@pytest.fixture(scope="module")
def client(matcher):
    return TestClient(create_app(matcher))


def test_match(client):
    payload = {"resume": samples.resumes()["frontend_dev"], "job": samples.jobs()["frontend_developer"]}
    res = client.post("/api/match", json=payload)
    body = res.json()
    assert res.status_code == 200 and body["band"] == "Strong match"
    assert {m["requirement"] for m in body["required"]} >= {"React", "TypeScript"}
    assert body["redacted"]["email"] == 1


def test_rank_and_gaps(client):
    resumes = {k: samples.resumes()[k] for k in ("frontend_dev", "backend_java", "data_analyst")}
    payload = {"job": samples.jobs()["backend_engineer"], "resumes": resumes}
    ranked = client.post("/api/rank-resumes", json=payload).json()
    assert ranked[0]["id"] == "backend_java"
    payload = {"resume": samples.resumes()["ml_student"], "jobs": samples.jobs()}
    gaps = client.post("/api/skill-gaps", json=payload).json()
    assert gaps and {"skill", "jobs", "missing_in"} <= set(gaps[0])


def test_extract(client):
    job = client.post("/api/extract", json={"text": samples.jobs()["data_analyst"], "kind": "job"}).json()
    assert "Python or R" in job["required"] and job["min_years"] == 2
    resume = client.post("/api/extract", json={"text": samples.resumes()["devops_engineer"]}).json()
    assert "Terraform" in resume["skills"] and resume["redacted"]["personal detail"] == 1


def test_validation_and_uploads(client):
    assert client.post("/api/match", json={"resume": "", "job": "x"}).status_code == 422
    assert client.post("/api/extract", json={"text": "x", "kind": "poem"}).status_code == 422
    txt = client.post("/api/resume-text", files={"file": ("cv.txt", b"Python developer", "text/plain")})
    assert txt.json() == {"text": "Python developer"}
    assert client.post("/api/resume-text", files={"file": ("cv.docx", b"PK", "application/zip")}).status_code == 415


def test_ui_and_samples(client):
    assert "Resume Matcher" in client.get("/").text
    assert set(client.get("/api/samples").json()) == {"resumes", "jobs"}
