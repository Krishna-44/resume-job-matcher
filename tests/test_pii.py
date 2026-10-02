from jobmatch import samples
from jobmatch.pii import redact

RESUME = samples.resumes()["data_analyst"]


def test_contact_details_and_personal_attributes_are_removed():
    result = redact(RESUME)
    for secret in ("Priya Nair", "priya.nair@example.com", "98200 11223", "Female", "Married"):
        assert secret not in result.text
    assert result.removed == {"name": 1, "email": 1, "phone": 1, "personal detail": 1}


def test_dates_and_numbers_in_content_survive():
    text = redact(RESUME).text
    assert "Sep 2023 – Present" in text and "+4.1% conversion" in text and "2019 – 2022" in text


def test_score_does_not_depend_on_personal_details(matcher):
    job = samples.jobs()["data_analyst"]
    variant = (
        RESUME.replace("Priya Nair", "John Smith")
        .replace("priya.nair@example.com", "j.smith@example.org")
        .replace("98200 11223", "+1 212 555 0100")
        .replace("Gender: Female | Marital Status: Married", "Gender: Male | Marital Status: Single")
    )
    assert matcher.match(RESUME, job).to_dict() == matcher.match(variant, job).to_dict()


def test_section_headings_are_not_mistaken_for_names():
    assert redact("Professional Summary\nBuilt things.").removed == {}


def test_every_sample_resume_loses_its_contact_details():
    for rid, text in samples.resumes().items():
        removed = redact(text).removed
        assert removed["email"] == 1 and removed["phone"] == 1 and removed["name"] == 1, rid
