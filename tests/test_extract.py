from datetime import date

import pytest

from jobmatch import samples
from jobmatch.extract import RESUME_SECTIONS, degree_level, experience_years, parse_job, parse_resume, split_sections

TODAY = date(2026, 10, 1)


def test_experience_merges_overlapping_roles():
    text = "Analyst, Mar 2019 to Feb 2020\nAnalyst II, Jan 2020 – Dec 2021"
    assert experience_years(text, TODAY) == 2.8  # Mar 2019 → Dec 2021 = 34 months


def test_experience_handles_present_and_numeric_dates():
    text = "Engineer — Jan 2025 – Present\nIntern — 06/2024 - 08/2024"
    assert experience_years(text, TODAY) == round((22 + 3) / 12, 1)  # Jan 2025–Oct 2026 + 3 months


def test_education_dates_are_not_counted_as_experience():
    profile = parse_resume(samples.resumes()["ml_student"], TODAY)
    assert profile.years == 0.2  # the 3-month internship, not the 4-year degree


@pytest.mark.parametrize(
    "text, level",
    [("B.Tech in CS", 2), ("B.Eng. in Mechanical Engineering", 2), ("MS in Data Science", 3),
     ("M.Sc. Statistics", 3), ("Ph.D. in Physics", 4), ("Proficient in MS Office", 0), ("MBA", 3)],
)
def test_degree_levels(text, level):
    assert degree_level(text) == level


def test_sections():
    sections = split_sections("Jane\n\nSkills:\nPython\n\nEXPERIENCE\nDev at X\n", RESUME_SECTIONS)
    assert sections == {"_": "Jane", "skills": "Python", "experience": "Dev at X"}


def test_job_requirements_preferences_and_alternatives():
    job = parse_job(samples.jobs()["data_analyst"])
    assert job.required == [("SQL",), ("Excel",), ("Python", "R"), ("Tableau", "Power BI"), ("Statistics",)]
    assert job.preferred == [("A/B Testing",), ("dbt",)]
    assert job.min_years == 2


def test_lowest_degree_named_is_the_requirement():
    assert parse_job(samples.jobs()["ml_engineer_intern"]).degree == 2  # "Bachelor's or Master's"


def test_resume_skills_carry_evidence():
    skills = parse_resume(samples.resumes()["backend_java"], TODAY).skills
    assert "Kafka" in skills["Kafka"] and "Event-driven order pipeline" in skills["Kafka"]
