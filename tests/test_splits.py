import numpy as np
import pytest

from audience_predictor.data.splits import movie_split, temporal_holdout


def test_movie_split_is_disjoint():
    n_ratings = np.array([10, 500, 800, 50, 1200, 300, 900, 40], dtype=np.int64)
    split = movie_split(n_ratings, min_train_ratings=200, holdout_frac=0.4, seed=0)
    assert np.intersect1d(split.train_idx, split.holdout_idx).size == 0
    assert split.holdout_mask[split.holdout_idx].all()
    assert not split.holdout_mask[split.train_idx].any()
    assert set(split.train_idx).isdisjoint({0, 3, 7})


def test_temporal_split_respects_cutoff():
    years = np.array([1990, 2000, 2010, 2018, 1995, 2015])
    n_ratings = np.array([500, 500, 500, 500, 500, 500])
    split = temporal_holdout(years, n_ratings, cutoff_year=2010, min_train_ratings=100)
    assert set(years[split.train_idx]) <= {1990, 2000, 1995}
    assert set(years[split.holdout_idx]) <= {2010, 2018, 2015}


def test_empty_train_raises():
    with pytest.raises(ValueError):
        movie_split(np.array([5, 5, 5]), min_train_ratings=100, holdout_frac=0.2, seed=0)
