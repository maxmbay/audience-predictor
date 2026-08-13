import math

from audience_predictor.data.screenplay_parser import parse_screenplay, structural_features


def test_parse_scene_headings_and_characters(sample_screenplay):
    parsed = parse_screenplay(sample_screenplay)
    assert parsed.n_scenes == 3
    assert parsed.scenes[0].interior is True
    assert parsed.scenes[1].interior is False
    assert "KIRA" in parsed.characters
    assert "JON" in parsed.characters
    assert any("don't have long" in s.dialogue_text for s in parsed.scenes)
    assert any("Steam ghosts" in s.action_text for s in parsed.scenes)
    assert "CUT TO:" in parsed.transitions


def test_structural_features_are_finite(sample_screenplay):
    parsed = parse_screenplay(sample_screenplay)
    stats = structural_features(parsed)
    assert parsed.n_scenes == 3
    assert stats["n_scenes"] == math.log1p(3)
    assert 0 < stats["dialogue_fraction"] < 1
    assert stats["n_characters"] == math.log1p(2)
    assert stats["graph_n_edges"] >= math.log1p(1)
    assert stats["character_entropy"] > 0
    assert all(math.isfinite(v) for v in stats.values())
