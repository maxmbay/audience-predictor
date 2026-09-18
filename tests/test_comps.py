import numpy as np

from audience_predictor.eval.comps import neighbor_recall, product_metrics
from audience_predictor.eval.heads import format_predict_card, predict_head_for_document


def test_neighbor_recall_perfect_when_pred_equals_catalog():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20, 4)).astype(np.float32)
    popular = np.arange(20)
    hold = np.array([0, 1, 2, 3])
    out = neighbor_recall(z, z, hold, popular, k=5)
    assert out["neighbor_recall_at_k"] == 1.0
    assert out["n"] == 4.0


def test_neighbor_recall_drops_when_pred_is_noise():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(30, 8)).astype(np.float32)
    noise = rng.normal(size=z.shape).astype(np.float32)
    popular = np.arange(30)
    hold = np.arange(8)
    perfect = neighbor_recall(z, z, hold, popular, k=5)["neighbor_recall_at_k"]
    bad = neighbor_recall(z, noise, hold, popular, k=5)["neighbor_recall_at_k"]
    assert perfect > bad


def test_neighbor_recall_ignores_unpopular_titles():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(12, 4)).astype(np.float32)
    popular = np.array([0, 1, 2, 3, 4])
    hold = np.array([0, 1])
    out = neighbor_recall(z, z, hold, popular, k=3)
    assert out["neighbor_recall_at_k"] == 1.0
    assert out["neighbor_recall_chance"] == 3.0 / 4.0


def test_product_metrics_splits_obscure_and_famous():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(40, 6)).astype(np.float32)
    hold = np.arange(20)
    n_ratings = np.concatenate([np.full(10, 100.0), np.full(30, 5000.0)])
    popular = np.arange(40)
    out = product_metrics(z, z, hold, n_ratings, popular, k=5)
    assert out["neighbor_recall_at_5"] == 1.0
    assert "neighbor_recall_at_5_obscure" in out
    assert "cosine_famous" in out
    assert out["n_obscure"] + out["n_famous"] == 20.0


def test_screenplay_routes_to_dialogue_head_when_trained():
    have = frozenset({"subtitle_mean", "dialogue_mean", "genome"})
    assert predict_head_for_document("screenplay", have) == "dialogue_mean"
    assert predict_head_for_document("shooting_script", have) == "dialogue_mean"
    assert predict_head_for_document("subtitle", have) == "subtitle_mean"
    assert predict_head_for_document("synopsis", have) == "subtitle_mean"


def test_format_predict_card_is_a_guess_not_json():
    text = format_predict_card(
        {
            "path": "dialogue_mean",
            "document_type": "screenplay",
            "n_blocks": 12,
            "sigma": 0.81,
            "polarization": {
                "mean_rating": 3.4,
                "rating_std": 1.1,
                "p_love": 0.5,
                "p_hate": 0.1,
            },
            "comps": [{"title": "Halloween (1978)", "cosine": 0.41}],
        }
    )
    assert "this is a guess" in text
    assert "dialogue_mean" in text
    assert "Halloween" in text
    assert not text.strip().startswith("{")


def test_screenplay_falls_back_to_subtitle_baseline():
    have = frozenset({"subtitle_mean"})
    assert predict_head_for_document("screenplay", have) == "subtitle_mean"
    assert predict_head_for_document("unknown", have) == "subtitle_mean"
