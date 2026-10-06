from pathlib import Path

import pytest

from audience_predictor.coverage import reset_backend

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _heuristic_openjev(monkeypatch):
    monkeypatch.setenv("OPENJEV_BACKEND", "heuristic")
    reset_backend()
    yield
    reset_backend()


@pytest.fixture
def sample_screenplay() -> str:
    return (FIXTURES / "sample.fountain").read_text()


@pytest.fixture
def sample_srt() -> str:
    return (FIXTURES / "sample.srt").read_text()
