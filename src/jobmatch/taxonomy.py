"""A curated skill taxonomy: canonical names, aliases, categories and related-skill families.

Aliases are matched case-insensitively on word boundaries. An alias starting with ``=`` is matched
case-sensitively, for short or ambiguous names ("Go", "R", "Node") that would otherwise match
ordinary words.

Skills in the same *family* are interchangeable enough to earn partial credit: a candidate with
MySQL partly satisfies a PostgreSQL requirement, but not a React one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...]
    family: str | None = None


_AMBIGUOUS = {"React", "Excel", "Express", "Spark", "Swift", "Dart", "Rust", "CAD", "RAG", "Go", "C", "R"}


def _s(name: str, category: str, *aliases: str, family: str | None = None) -> Skill:
    # Ambiguous names ("react quickly", "excel at", "go to") match only via their explicit,
    # usually case-sensitive aliases.
    base = () if name in _AMBIGUOUS else (name.lower(),)
    return Skill(name, category, (*base, *aliases), family)


SKILLS: tuple[Skill, ...] = (
    # Programming languages
    _s("Python", "language", "python3"),
    _s("Java", "language", family="jvm"),
    _s("Kotlin", "language", family="jvm"),
    _s("Scala", "language", family="jvm"),
    _s("JavaScript", "language", "js", "es6", "ecmascript", family="js"),
    _s("TypeScript", "language", family="js"),
    _s("C++", "language", "cpp", family="systems"),
    _s("C", "language", "=C", family="systems"),
    _s("C#", "language", "csharp", family="dotnet"),
    _s("Go", "language", "golang", "=Go", family="systems"),
    _s("Rust", "language", "=Rust", family="systems"),
    _s("R", "language", "=R"),
    _s("SQL", "language"),
    _s("Bash", "language", "shell scripting"),
    _s("MATLAB", "language"),
    _s("Swift", "language", "=Swift", family="mobile"),
    _s("Dart", "language", "=Dart", family="mobile"),
    _s("PHP", "language"),
    _s("Ruby", "language"),
    # Front end
    _s("React", "frontend", "=React", "react.js", "reactjs", family="frontend_framework"),
    _s("Next.js", "frontend", "nextjs", family="frontend_framework"),
    _s("Vue", "frontend", "vue.js", "vuejs", family="frontend_framework"),
    _s("Angular", "frontend", "angularjs", family="frontend_framework"),
    _s("Svelte", "frontend", family="frontend_framework"),
    _s("HTML", "frontend", "html5"),
    _s("CSS", "frontend", "css3", "scss", "sass"),
    _s("Tailwind CSS", "frontend", "tailwind"),
    _s("Redux", "frontend"),
    _s("Flutter", "frontend", family="mobile"),
    _s("React Native", "frontend", family="mobile"),
    _s("Figma", "design"),
    # Back end
    _s("Node.js", "backend", "nodejs", "node js", "=Node", family="backend_framework"),
    _s("Express", "backend", "=Express", "express.js", "expressjs", family="backend_framework"),
    _s("Django", "backend", family="backend_framework"),
    _s("Flask", "backend", family="backend_framework"),
    _s("FastAPI", "backend", family="backend_framework"),
    _s("Spring Boot", "backend", "spring framework", family="backend_framework"),
    _s(".NET", "backend", "dotnet", "asp.net", family="dotnet"),
    _s("REST APIs", "backend", "=REST", "restful", "rest api", "rest apis"),
    _s("GraphQL", "backend"),
    _s("gRPC", "backend", "grpc"),
    _s("Microservices", "backend", "microservice"),
    _s("Kafka", "backend", "apache kafka", family="messaging"),
    _s("RabbitMQ", "backend", family="messaging"),
    # Data stores
    _s("PostgreSQL", "database", "postgres", "postgresql", family="sql_database"),
    _s("MySQL", "database", family="sql_database"),
    _s("SQL Server", "database", "mssql", "microsoft sql server", family="sql_database"),
    _s("SQLite", "database", family="sql_database"),
    _s("Oracle Database", "database", "oracle db", family="sql_database"),
    _s("MongoDB", "database", "mongo", family="nosql_database"),
    _s("Redis", "database", family="nosql_database"),
    _s("Cassandra", "database", family="nosql_database"),
    _s("DynamoDB", "database", family="nosql_database"),
    _s("Elasticsearch", "database", "elastic search", "opensearch"),
    _s("Snowflake", "database", family="warehouse"),
    _s("BigQuery", "database", "big query", family="warehouse"),
    # Cloud & DevOps
    _s("AWS", "cloud", "amazon web services", "ec2", "s3", "aws lambda", family="cloud"),
    _s("GCP", "cloud", "google cloud", "google cloud platform", family="cloud"),
    _s("Azure", "cloud", "microsoft azure", family="cloud"),
    _s("Docker", "devops", "containerization", "containerized", family="containers"),
    _s("Kubernetes", "devops", "k8s", "eks", "gke", "aks", family="containers"),
    _s("Terraform", "devops", family="iac"),
    _s("Ansible", "devops", family="iac"),
    _s("CloudFormation", "devops", family="iac"),
    _s("Linux", "devops", "unix", "ubuntu"),
    _s("CI/CD", "devops", "ci/cd", "continuous integration", "continuous delivery", "cicd"),
    _s("Jenkins", "devops", family="ci_tool"),
    _s("GitHub Actions", "devops", family="ci_tool"),
    _s("GitLab CI", "devops", family="ci_tool"),
    _s("Prometheus", "devops", family="monitoring"),
    _s("Grafana", "devops", family="monitoring"),
    _s("Datadog", "devops", family="monitoring"),
    _s("Nginx", "devops"),
    _s("Git", "tool", "github", "gitlab", "version control"),
    # Data & ML
    _s("Machine Learning", "ml", "ml", "machine-learning"),
    _s("Deep Learning", "ml", "neural networks", "neural network"),
    _s("NLP", "ml", "natural language processing"),
    _s("Computer Vision", "ml", "image recognition", "object detection"),
    _s("LLMs", "ml", "llm", "large language models", "large language model", "generative ai", "genai"),
    _s("RAG", "ml", "=RAG", "retrieval-augmented generation", "retrieval augmented generation"),
    _s("PyTorch", "ml", "torch", family="dl_framework"),
    _s("TensorFlow", "ml", "tf2", family="dl_framework"),
    _s("Keras", "ml", family="dl_framework"),
    _s("JAX", "ml", family="dl_framework"),
    _s("scikit-learn", "ml", "sklearn", "scikit learn"),
    _s("XGBoost", "ml", "lightgbm", "gradient boosting"),
    _s("Hugging Face", "ml", "huggingface", "transformers library"),
    _s("OpenCV", "ml"),
    _s("MLOps", "ml", "mlflow", "model deployment"),
    _s("pandas", "data"),
    _s("NumPy", "data", "numpy"),
    _s("Spark", "data", "=Spark", "pyspark", "apache spark"),
    _s("Airflow", "data", "apache airflow"),
    _s("dbt", "data"),
    _s("ETL", "data", "elt", "data pipelines", "data pipeline"),
    _s("Statistics", "data", "statistical analysis", "statistical modeling", "hypothesis testing"),
    _s("A/B Testing", "data", "a/b testing", "ab testing", "a/b tests", "experimentation"),
    _s("Data Visualization", "data", "data visualisation", "dashboards", "dashboarding"),
    _s("Tableau", "data", family="bi_tool"),
    _s("Power BI", "data", "powerbi", family="bi_tool"),
    _s("Looker", "data", family="bi_tool"),
    _s("Excel", "data", "=Excel", "microsoft excel", "spreadsheets", "vlookup", "pivot tables"),
    # Testing
    _s("Unit Testing", "testing", "unit tests", "test-driven development", "tdd"),
    _s("Jest", "testing", family="js_testing"),
    _s("Cypress", "testing", family="js_testing"),
    _s("Playwright", "testing", family="js_testing"),
    _s("pytest", "testing"),
    _s("JUnit", "testing"),
    # Engineering & design
    _s("CAD", "engineering", "=CAD", "computer-aided design"),
    _s("SolidWorks", "engineering"),
    _s("AutoCAD", "engineering"),
    _s("Agile", "practice", "scrum", "kanban"),
    _s("System Design", "practice", "distributed systems", "scalability"),
    _s("Data Structures", "practice", "algorithms", "data structures and algorithms", "dsa"),
    _s("Project Management", "practice", "jira"),
)

BY_NAME: dict[str, Skill] = {s.name: s for s in SKILLS}


@cache
def _pattern(alias: str) -> re.Pattern[str]:
    case_sensitive = alias.startswith("=")
    text = alias[1:] if case_sensitive else alias
    # Word boundaries that also respect skill punctuation (C++, C#, .NET, Node.js, CI/CD).
    body = re.escape(text).replace(r"\ ", r"[\s\-]+")
    # Short case-sensitive names also must not be followed by "-" ("Go-to", "C-level").
    after = r"(?![\w+#&-]|\.\w)" if case_sensitive else r"(?![\w+#&]|\.\w)"
    return re.compile(rf"(?<![\w+#&./-]){body}{after}", 0 if case_sensitive else re.IGNORECASE)


def find_skill_mentions(text: str) -> dict[str, list[tuple[int, int]]]:
    """Return ``{canonical skill: [(start, end), ...]}`` for every skill mentioned in ``text``."""
    found: dict[str, list[tuple[int, int]]] = {}
    for skill in SKILLS:
        spans: list[tuple[int, int]] = []
        for alias in skill.aliases:
            spans += [m.span() for m in _pattern(alias).finditer(text)]
        if spans:
            found[skill.name] = sorted(set(spans))
    # A longer skill name contains a shorter one ("React Native" ⊃ "React", "SQL Server" ⊃ "SQL"):
    # drop mentions of the shorter skill that sit inside a longer skill's mention.
    for name, spans in list(found.items()):
        others = [s for other, ss in found.items() if other != name for s in ss]
        kept = [sp for sp in spans if not any(o[0] <= sp[0] and sp[1] <= o[1] and o != sp for o in others)]
        if kept:
            found[name] = kept
        else:
            del found[name]
    return found


def related(a: str, b: str) -> bool:
    """True when two different skills belong to the same family."""
    fa, fb = BY_NAME[a].family, BY_NAME[b].family
    return a != b and fa is not None and fa == fb
