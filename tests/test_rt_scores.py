from pathlib import Path

import numpy as np
import polars as pl

from audience_predictor.data.rt_scores import (
    normalize_imdb,
    parse_hetrec_movies,
    parse_percent,
    rt_target_matrices,
)
from audience_predictor.eval.heads import format_predict_card
from audience_predictor.eval.metrics import score_metrics
from audience_predictor.models.scores import ScoreMLP
from audience_predictor.train.loop import predict_scores, train_score_mlp

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_imdb_strips_prefix_and_zeros():
    assert normalize_imdb("0114709") == "114709"
    assert normalize_imdb("tt0114709") == "114709"
    assert normalize_imdb("114709") == "114709"
    assert normalize_imdb("") == ""


def test_parse_percent_accepts_0_100_and_fractions():
    assert parse_percent("81") == 0.81
    assert parse_percent("100") == 1.0
    assert parse_percent("0.64") == 0.64
    assert parse_percent("\\N") is None
    assert parse_percent("N/A") is None


def test_parse_hetrec_movies_reads_critic_and_audience():
    text = (FIXTURES / "hetrec_movies.dat").read_text()
    frame = parse_hetrec_movies(text)
    toy = frame.filter(pl.col("imdb_norm") == "114709")
    assert toy["rt_critics"][0] == 1.0
    assert abs(float(toy["rt_audience"][0]) - 0.81) < 1e-6
    missing = frame.filter(pl.col("imdb_norm") == "1")
    assert missing["rt_critics"][0] is None


def test_rt_target_matrices_mask_missing():
    table = pl.DataFrame(
        {
            "movie_row": [0, 2],
            "rt_critics": [0.9, None],
            "rt_audience": [0.7, 0.4],
        }
    )
    y, mask = rt_target_matrices(4, table)
    assert y.shape == (4, 2)
    assert mask[0, 0] and mask[0, 1]
    assert not mask[1].any()
    assert not mask[2, 0] and mask[2, 1]
    np.testing.assert_allclose(y[0], [0.9, 0.7])


def test_score_metrics_perfect_and_gap():
    true = np.array([[0.2, 0.8], [0.5, 0.4], [0.9, 0.1], [0.3, 0.6]] * 3)
    mask = np.ones_like(true, dtype=bool)
    out = score_metrics(true, true, mask)
    assert out["spearman_critics"] == 1.0
    assert out["spearman_audience"] == 1.0
    assert out["spearman_gap"] == 1.0


def test_score_mlp_fits_a_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 8)).astype(np.float32)
    w = rng.normal(size=(8, 2)).astype(np.float32)
    y = 1.0 / (1.0 + np.exp(-(x @ w)))
    mask = np.ones_like(y, dtype=bool)
    model, history = train_score_mlp(
        x, y, mask, np.arange(48),
        hidden=16, epochs=8, lr=0.05, weight_decay=0.0, seed=0,
    )
    assert history[-1] < history[0]
    pred = predict_scores(model, x)
    assert pred.shape == (64, 2)
    assert ScoreMLP(8, 16, __import__("jax").random.key(0)).in_dim == 8


def test_predict_card_includes_rt_estimates():
    text = format_predict_card(
        {
            "path": "subtitle_mean",
            "document_type": "subtitle",
            "n_blocks": 8,
            "sigma": 0.5,
            "polarization": {
                "mean_rating": 3.4,
                "rating_std": 1.0,
                "p_love": 0.5,
                "p_hate": 0.1,
            },
            "rt": {"critics": 72.0, "audience": 81.0},
            "comps": [],
        }
    )
    assert "critics" in text
    assert "81" in text
    assert "if this were released" in text
