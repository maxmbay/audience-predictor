from audience_predictor.data.document import classify_document, make_record
from tests.conftest import SAMPLE_SCREENPLAY, SAMPLE_SRT


def test_classifier_separates_screenplay_and_subtitle():
    assert classify_document(SAMPLE_SCREENPLAY) == "screenplay"
    assert classify_document(SAMPLE_SRT) == "subtitle"


def test_transcript_heuristic():
    text = "\n".join(f"NARRATOR: line {i} happens now." for i in range(12))
    assert classify_document(text) == "transcript"


def test_provenance_record_never_collapses_type():
    rec = make_record(
        document_id="doc-1",
        text=SAMPLE_SCREENPLAY,
        source="fixture",
        scene_count=3,
        parse_quality=1.0,
    )
    assert rec.document_type == "screenplay"
    assert rec.word_count > 20
    assert rec.source == "fixture"
