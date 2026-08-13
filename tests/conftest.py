from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_screenplay() -> str:
    return (FIXTURES / "sample.fountain").read_text()


@pytest.fixture
def sample_srt() -> str:
    return (FIXTURES / "sample.srt").read_text()
