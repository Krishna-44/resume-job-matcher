from datetime import date

import pytest

from jobmatch.scoring import Matcher

TODAY = date(2026, 10, 1)


@pytest.fixture(scope="session")
def matcher() -> Matcher:
    return Matcher(today=TODAY)
