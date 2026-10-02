"""Command-line interface: ``jobmatch match | rank | gaps | eval | serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jobmatch.evaluation import evaluate, format_table
from jobmatch.ranking import rank_jobs, rank_resumes, skill_gaps
from jobmatch.scoring import Matcher, MatchResult

ICONS = {"matched": "✓", "partial": "~", "missing": "✗"}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _folder(path: str) -> dict[str, str]:
    files = sorted(Path(path).glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files in {path}")
    return {f.stem: f.read_text(encoding="utf-8") for f in files}


def print_match(r: MatchResult) -> None:
    print(f"\n{r.score:.0f}/100 — {r.band}\n")
    for name, value in r.components.items():
        bar = "█" * round(value * 20)
        print(f"  {name.replace('_', ' '):<17} {bar:<20} {value:.2f}  (weight {r.weights[name]:.2f})")
    for title, items in (("Required skills", r.required), ("Preferred skills", r.preferred)):
        if items:
            print(f"\n{title}")
            for m in items:
                via = f" (via {m.via})" if m.status == "partial" else ""
                evidence = f"  — “{m.evidence}”" if m.status == "matched" and m.evidence else ""
                print(f"  {ICONS[m.status]} {m.requirement}{via}{evidence}")
    years = f"{r.candidate_years:g} years" + (f" (asked: {r.required_years:g}+)" if r.required_years else "")
    print(f"\nExperience: {years}   Degree: {r.candidate_degree} (asked: {r.required_degree})")
    if r.shared_terms:
        print(f"Shared terms: {', '.join(r.shared_terms)}")
    if r.redacted:
        print("Removed before scoring: " + ", ".join(f"{n} {k}" for k, n in r.redacted.items()))
    if r.suggestions:
        print("\nSuggestions")
        for tip in r.suggestions:
            print(f"  • {tip}")


def cmd_match(args: argparse.Namespace) -> int:
    result = Matcher().match(_read(args.resume), _read(args.job))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_match(result)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    matcher = Matcher()
    if args.job:
        ranked = rank_resumes(matcher, _read(args.job), _folder(args.folder))
    else:
        ranked = rank_jobs(matcher, _read(args.resume), _folder(args.folder))
    for i, (name, r) in enumerate(ranked, start=1):
        missing = [m.requirement for m in r.required if m.status == "missing"]
        print(f"{i:>2}. {r.score:5.1f}  {r.band:<14} {name}" + (f"   missing: {', '.join(missing)}" if missing else ""))
    return 0


def cmd_gaps(args: argparse.Namespace) -> int:
    jobs = _folder(args.jobs)
    gaps = skill_gaps(Matcher(), _read(args.resume), jobs)
    if not gaps:
        print("No missing required skills across these jobs.")
    for g in gaps:
        note = f" (you have {g.related_skill}, related)" if g.related_skill else ""
        print(f"{g.skill:<28} needed by {g.jobs}/{len(jobs)} jobs{note}")
    return 0


def cmd_eval(_: argparse.Namespace) -> int:
    print(format_table(evaluate()))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("jobmatch.api:create_app", factory=True, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jobmatch", description="Explainable resume–job matching.")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("match", help="score one resume against one job description")
    m.add_argument("resume")
    m.add_argument("job")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_match)

    r = sub.add_parser("rank", help="rank a folder of resumes for a job, or of jobs for a resume")
    target = r.add_mutually_exclusive_group(required=True)
    target.add_argument("--job", help="rank the resumes in FOLDER for this job")
    target.add_argument("--resume", help="rank the jobs in FOLDER for this resume")
    r.add_argument("folder")
    r.set_defaults(func=cmd_rank)

    g = sub.add_parser("gaps", help="skills a resume is missing across a folder of target jobs")
    g.add_argument("resume")
    g.add_argument("jobs")
    g.set_defaults(func=cmd_gaps)

    e = sub.add_parser("eval", help="ranking quality on the labelled sample set")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("serve", help="run the API and web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
