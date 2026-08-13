from audience_predictor.data.encoder import HashEncoder
from audience_predictor.data.infer import blocks_from_document


def test_blocks_from_screenplay_document(sample_screenplay):
    d, a, mask, struct, kind = blocks_from_document(
        sample_screenplay, max_blocks=8, encoder=HashEncoder(dim=16)
    )
    assert kind in {"screenplay", "shooting_script"}
    assert d.shape == (1, 8, 16)
    assert int(mask.sum()) >= 1


def test_blocks_from_srt_document(sample_srt):
    d, a, mask, struct, kind = blocks_from_document(
        sample_srt, max_blocks=8, encoder=HashEncoder(dim=16)
    )
    assert kind == "subtitle"
    assert d.shape[0] == 1
    assert int(mask.sum()) >= 1
