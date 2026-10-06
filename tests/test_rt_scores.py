from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from audience_predictor.data.rt_scores import (
    normalize_imdb,
    parse_count,
    parse_hetrec_movies,
    parse_optional_count,
    parse_percent,
    rt_count_matrices,
    rt_target_matrices,
)
from audience_predictor.eval.heads import format_predict_card
from audience_predictor.eval.metrics import score_metrics
from audience_predictor.models.losses import beta_binomial_nll
from audience_predictor.models.scores import ScoreBeta, beta_interval, beta_mean
from audience_predictor.train.loop import predict_score_beta, train_score_beta

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
    assert toy["k_critics"][0] == 73
    assert toy["n_critics"][0] == 73
    missing = frame.filter(pl.col("imdb_norm") == "1")
    assert missing["rt_critics"][0] is None


def test_parse_hetrec_treats_zero_score_with_zero_reviews_as_missing():
    """hetrec encodes 'never rated' as score 0 with 0 reviews, not as \\N."""
    text = (FIXTURES / "hetrec_movies.dat").read_text()
    unrated = parse_hetrec_movies(text).filter(pl.col("imdb_norm") == "2")
    assert unrated["rt_critics"][0] is None
    assert unrated["rt_audience"][0] is None
    assert unrated["n_critics"][0] is None


def test_one_review_hundred_keeps_the_count():
    """A real 100% from one review is (k=1, n=1), not a missing label."""
    thin = parse_hetrec_movies((FIXTURES / "hetrec_movies.dat").read_text()).filter(
        pl.col("imdb_norm") == "3"
    )
    assert thin["rt_critics"][0] == 1.0
    assert thin["k_critics"][0] == 1
    assert thin["n_critics"][0] == 1


def test_parse_count_handles_missing_markers():
    assert parse_count("73") == 73
    assert parse_count("\\N") == 0
    assert parse_count("") == 0
    assert parse_optional_count("\\N") is None
    assert parse_optional_count("0") == 0


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


def test_rt_count_matrices_use_k_and_n():
    table = pl.DataFrame(
        {
            "movie_row": [0, 1],
            "k_critics": [1, 73],
            "n_critics": [1, 73],
            "k_audience": [None, 800],
            "n_audience": [None, 1000],
        }
    )
    k, n, mask = rt_count_matrices(3, table)
    assert mask[0, 0] and not mask[0, 1]
    assert k[0, 0] == 1 and n[0, 0] == 1
    assert k[1, 0] == 73 and n[1, 1] == 1000
    assert not mask[2].any()


def test_score_metrics_perfect_and_gap():
    true = np.array([[0.2, 0.8], [0.5, 0.4], [0.9, 0.1], [0.3, 0.6]] * 3)
    mask = np.ones_like(true, dtype=bool)
    out = score_metrics(true, true, mask)
    assert out["spearman_critics"] == 1.0
    assert out["spearman_audience"] == 1.0
    assert out["spearman_gap"] == 1.0


def test_beta_binomial_prefers_matching_counts():
    import jax.numpy as jnp

    k = jnp.array([[8.0, 8.0]])
    n = jnp.array([[10.0, 10.0]])
    mask = jnp.ones_like(k)
    matched = beta_binomial_nll(jnp.array([[8.0, 8.0]]), jnp.array([[2.0, 2.0]]), k, n, mask)
    wrong = beta_binomial_nll(jnp.array([[2.0, 2.0]]), jnp.array([[8.0, 8.0]]), k, n, mask)
    assert float(matched) < float(wrong)


def test_one_review_hundred_does_not_pin_the_posterior():
    """Beta(1,1) updated by (k=1, n=1) is Beta(2,1); mean 2/3, not 1."""
    mean = beta_mean(np.array([2.0]), np.array([1.0]))[0]
    lo, hi = beta_interval(np.array([2.0]), np.array([1.0]))
    assert 0.6 < mean < 0.75
    assert lo[0] < 0.3
    assert hi[0] > 0.95


def test_score_beta_fits_a_smooth_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 8)).astype(np.float32)
    w = rng.normal(size=(8, 2)).astype(np.float32)
    pi = 1.0 / (1.0 + np.exp(-(x @ w)))
    n = np.full((80, 2), 40, dtype=np.int32)
    k = rng.binomial(40, pi).astype(np.int32)
    mask = np.ones_like(k, dtype=bool)
    model, history = train_score_beta(
        x, k, n, mask, np.arange(64),
        hidden=16, epochs=80, lr=0.05, weight_decay=0.0, seed=0, val_frac=0.2,
    )
    assert history[-1] < history[0]
    post = predict_score_beta(model, x)
    mean = beta_mean(post["alpha"], post["beta"])
    assert mean.shape == (80, 2)
    assert ((mean > 0.0) & (mean < 1.0)).all()
    assert not np.isclose(mean, 1.0).any()
    assert ScoreBeta(8, 16, __import__("jax").random.key(0)).in_dim == 8
    for col in range(2):
        assert spearmanr(mean[64:, col], pi[64:, col]).statistic > 0.3


def test_predict_card_includes_rt_interval():
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
            "rt": {
                "critics": 72.0,
                "audience": 81.0,
                "critics_lo": 54.0,
                "critics_hi": 86.0,
                "audience_lo": 70.0,
                "audience_hi": 88.0,
            },
            "comps": [],
        }
    )
    assert "critics" in text
    assert "81" in text
    assert "90%" in text
    assert "54" in text
    assert "if this were released" in text
