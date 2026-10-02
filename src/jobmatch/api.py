"""FastAPI service and web UI. Run with ``jobmatch serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from jobmatch import samples
from jobmatch.extract import DEGREE_LEVELS, parse_job, parse_resume
from jobmatch.pii import redact
from jobmatch.ranking import rank_jobs, rank_resumes, skill_gaps
from jobmatch.scoring import Matcher

STATIC_DIR = Path(__file__).parent / "static"
MAX_TEXT = 50_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT)]


class MatchRequest(BaseModel):
    resume: Text
    job: Text


class RankRequest(BaseModel):
    job: Text
    resumes: dict[str, Text] = Field(min_length=1, max_length=200)


class JobsRequest(BaseModel):
    resume: Text
    jobs: dict[str, Text] = Field(min_length=1, max_length=50)


class ExtractRequest(BaseModel):
    text: Text
    kind: str = Field(default="resume", pattern="^(resume|job)$")


def create_app(matcher: Matcher | None = None) -> FastAPI:
    matcher = matcher or Matcher()
    app = FastAPI(
        title="Resume–Job Matcher",
        version="0.1.0",
        description="Explainable resume-to-job matching with blind screening and skill-gap analysis.",
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/samples")
    def sample_data() -> dict:
        return {"resumes": samples.resumes(), "jobs": samples.jobs()}

    @app.post("/api/match")
    def match(req: MatchRequest) -> dict:
        return matcher.match(req.resume, req.job).to_dict()

    @app.post("/api/rank-resumes")
    def rank_candidates(req: RankRequest) -> list[dict]:
        return [{"id": rid, **r.to_dict()} for rid, r in rank_resumes(matcher, req.job, req.resumes)]

    @app.post("/api/rank-jobs")
    def rank_openings(req: JobsRequest) -> list[dict]:
        return [{"id": jid, **r.to_dict()} for jid, r in rank_jobs(matcher, req.resume, req.jobs)]

    @app.post("/api/skill-gaps")
    def gaps(req: JobsRequest) -> list[dict]:
        return [
            {"skill": g.skill, "jobs": g.jobs, "missing_in": g.missing_in, "partial_in": g.partial_in,
             "related_skill": g.related_skill}
            for g in skill_gaps(matcher, req.resume, req.jobs)
        ]

    @app.post("/api/extract")
    def extract(req: ExtractRequest) -> dict:
        if req.kind == "job":
            job = parse_job(req.text)
            return {"required": [" or ".join(g) for g in job.required],
                    "preferred": [" or ".join(g) for g in job.preferred],
                    "min_years": job.min_years, "degree": DEGREE_LEVELS[job.degree]}
        redaction = redact(req.text)
        profile = parse_resume(redaction.text)
        return {"skills": profile.skills, "years": profile.years, "degree": DEGREE_LEVELS[profile.degree],
                "sections": profile.sections, "redacted": dict(redaction.removed)}

    @app.post("/api/resume-text")
    async def resume_text(file: Annotated[UploadFile, File()]) -> dict:
        """Extract plain text from an uploaded .txt or .pdf resume."""
        name = (file.filename or "").lower()
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 5 MB)")
        if name.endswith(".txt"):
            return {"text": data.decode("utf-8", errors="replace")}
        if name.endswith(".pdf"):
            try:
                import io

                from pypdf import PdfReader
            except ImportError as exc:  # pragma: no cover
                raise HTTPException(501, "PDF support needs: pip install 'jobmatch[pdf]'") from exc
            try:
                pages = PdfReader(io.BytesIO(data)).pages
            except Exception as exc:
                raise HTTPException(400, "Could not read that PDF") from exc
            return {"text": "\n".join(page.extract_text() or "" for page in pages)}
        raise HTTPException(415, "Upload a .txt or .pdf file")

    return app
