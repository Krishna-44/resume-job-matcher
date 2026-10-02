import pytest

from jobmatch.taxonomy import SKILLS, find_skill_mentions, related


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Python, C++, C#, Go and R", {"Python", "C++", "C#", "Go", "R"}),
        ("Node.js, React Native and SQL Server", {"Node.js", "React Native", "SQL Server"}),
        (
            "Deployed on AWS (EC2, S3) with k8s; CI/CD with GitHub Actions",
            {"AWS", "Kubernetes", "CI/CD", "GitHub Actions"},
        ),
        ("scikit-learn / sklearn, Power BI, A/B testing", {"scikit-learn", "Power BI", "A/B Testing"}),
    ],
)
def test_finds_skills_and_aliases(text, expected):
    assert set(find_skill_mentions(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I react quickly and excel at teamwork; the rest of the team expressed interest.",
        "Spring 2023 semester; R&D group; my go-to approach; NoSQL; a C-level audience",
    ],
)
def test_ordinary_words_are_not_skills(text):
    assert find_skill_mentions(text) == {}


def test_related_skills_share_a_family():
    assert related("PostgreSQL", "MySQL")
    assert related("PyTorch", "TensorFlow")
    assert not related("PostgreSQL", "MongoDB")
    assert not related("Python", "Python")


def test_taxonomy_names_are_unique():
    names = [s.name for s in SKILLS]
    assert len(names) == len(set(names))
