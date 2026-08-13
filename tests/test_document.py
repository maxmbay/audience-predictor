from audience_predictor.data.document import classify_document, make_record


def test_classifier_separates_screenplay_and_subtitle(sample_screenplay, sample_srt):
    assert classify_document(sample_screenplay) == "screenplay"
    assert classify_document(sample_srt) == "subtitle"


def test_transcript_heuristic():
    text = "\n".join(f"NARRATOR: line {i} happens now." for i in range(12))
    assert classify_document(text) == "transcript"


def test_provenance_record_never_collapses_type(sample_screenplay):
    rec = make_record(
        document_id="doc-1",
        text=sample_screenplay,
        source="fixture",
        scene_count=3,
        parse_quality=1.0,
    )
    assert rec.document_type == "screenplay"
    assert rec.word_count > 20
    assert rec.source == "fixture"
