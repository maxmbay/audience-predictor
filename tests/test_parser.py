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


def test_fountain_title_page_is_not_a_scene(sample_screenplay):
    titled = "Title: THE TEST\nCredit: written by\nAuthor: someone\n\n" + sample_screenplay
    parsed = parse_screenplay(titled)
    assert parsed.n_scenes == 3
    assert all(scene.heading != "UNSLUGGED" for scene in parsed.scenes)


def test_shouted_notes_and_extensions_are_not_new_characters():
    text = """INT. KITCHEN - NIGHT

A note under a magnet:

    DON'T CALL DOWN.

MARA (CONT'D)
Who wrote this?

ELLIS (O.S.)
Don't.

CUT TO BLACK.
"""
    parsed = parse_screenplay(text)
    assert parsed.characters == ["MARA", "ELLIS"]
    assert "CUT TO BLACK." in parsed.transitions
    assert any("DON'T CALL DOWN." in line for line in parsed.scenes[0].action)


def test_play_cues_with_periods_and_act_markers():
    text = """ACT I

EXT. ELSINORE - NIGHT

BARNARDO.
Who's there?

FRANCISCO.
Nay, answer me.

BARNARDO.
Long live the King!
"""
    parsed = parse_screenplay(text)
    assert parsed.n_scenes == 1
    assert parsed.characters == ["BARNARDO", "FRANCISCO"]
    assert len(parsed.scenes[0].dialogue) == 3


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
