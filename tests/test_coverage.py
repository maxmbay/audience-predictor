from audience_predictor.coverage import cover_pages
from audience_predictor.coverage.judge import select_scene_indices, stamp_for
from audience_predictor.coverage.nli import HeuristicNLI
from audience_predictor.coverage.schema import SCENE_QUESTIONS
from audience_predictor.web.demos import load_demo

ON_THE_NOSE = """
INT. OFFICE - DAY

JOHN
I am very sad because you don't love me. I feel angry.

MARY
I know you are sad. I don't love you. You always make me unhappy.
"""

DEAD_AIR = """
INT. HALLWAY - DAY

A hallway.
"""


def test_sample_pages_get_a_real_stamp(sample_screenplay):
    card = cover_pages(sample_screenplay)
    assert card["judge"]["kind"] == "heuristic"
    assert card["stamp"] in {"consider", "recommend"}
    assert card["n_judged"] == 3
    means = card["scene_means"]
    assert means["want"] > 0.25
    assert means["specific"] > 0.35
    assert means["dead_air"] < 0.55
    assert card["stats"]["n_characters"] == 2
    assert len(card["timeline"]["x"]) == 3
    for row in card["scenes"]:
        mass = sum(row["softmax"].values())
        assert abs(mass - 1.0) < 1e-3
        assert 0.0 <= row["craft"] <= 1.0


def test_on_the_nose_dialogue_is_flagged():
    card = cover_pages(ON_THE_NOSE)
    assert card["scene_means"]["on_the_nose"] > 0.35
    assert card["scene_means"]["specific"] < card["scene_means"]["on_the_nose"]


def test_empty_hallway_is_dead_air():
    card = cover_pages(DEAD_AIR)
    assert card["scene_means"]["dead_air"] > 0.5
    assert card["stamp"] == "pass"


def test_well_leans_horror_tickets_lean_romance():
    well = cover_pages(load_demo("the_well")[1])
    tickets = cover_pages(load_demo("two_tickets")[1])
    assert well["genre_probs"]["horror"] > tickets["genre_probs"]["horror"]
    assert tickets["genre_probs"]["romance"] > well["genre_probs"]["romance"]


def test_heuristic_nli_answers_typed_questions(sample_screenplay):
    from audience_predictor.coverage.craft import scene_premise
    from audience_predictor.data.screenplay_parser import parse_screenplay

    scene = parse_screenplay(sample_screenplay).scenes[0]
    premise = scene_premise(scene)
    triples = HeuristicNLI().predict([(premise, q.hypothesis) for q in SCENE_QUESTIONS])
    assert len(triples) == len(SCENE_QUESTIONS)
    for triple in triples:
        mass = triple.contradiction + triple.entailment + triple.neutral
        assert abs(mass - 1.0) < 1e-6
        assert 0.0 <= triple.entailment <= 1.0


def test_scene_sampler_spreads():
    assert select_scene_indices(3, 12) == [0, 1, 2]
    picked = select_scene_indices(20, 5)
    assert picked[0] == 0
    assert picked[-1] == 19
    assert len(picked) == 5


def test_stamp_thresholds():
    assert stamp_for(0.2) == "pass"
    assert stamp_for(0.45) == "consider"
    assert stamp_for(0.7) == "recommend"
