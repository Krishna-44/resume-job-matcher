"""Bundled sample resumes, job descriptions and relevance labels (all fictional)."""

from __future__ import annotations

import csv
from functools import cache
from importlib.resources import files

_ROOT = files("jobmatch") / "samples"


def _load(folder: str) -> dict[str, str]:
    return {
        p.name.removesuffix(".txt"): p.read_text(encoding="utf-8")
        for p in sorted((_ROOT / folder).iterdir(), key=lambda p: p.name)
        if p.name.endswith(".txt")
    }


@cache
def resumes() -> dict[str, str]:
    return _load("resumes")


@cache
def jobs() -> dict[str, str]:
    return _load("jobs")


@cache
def labels() -> dict[tuple[str, str], int]:
    """``{(resume_id, job_id): relevance}`` with 0 = not a fit, 1 = partial, 2 = strong."""
    rows = csv.DictReader((_ROOT / "labels.csv").read_text(encoding="utf-8").splitlines())
    return {(r["resume"], r["job"]): int(r["label"]) for r in rows}
