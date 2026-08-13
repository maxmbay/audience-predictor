from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_SCREENPLAY = (FIXTURES / "sample.fountain").read_text()
SAMPLE_SRT = (FIXTURES / "sample.srt").read_text()
