"""Movie-level cold-start splits. Never split ratings of the same title."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MovieSplit:
    train_idx: NDArray[np.int64]
    holdout_idx: NDArray[np.int64]
    holdout_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train_idx, self.holdout_idx)
        if overlap.size:
            raise ValueError(f"train/holdout share {overlap.size} movie rows")


def movie_split(
    n_ratings: NDArray[np.integer],
    *,
    min_train_ratings: int,
    holdout_frac: float,
    seed: int,
) -> MovieSplit:
    n = int(n_ratings.shape[0])
    eligible = np.flatnonzero(np.asarray(n_ratings) >= min_train_ratings).astype(np.int64)
    rng = np.random.default_rng(seed)
    shuffled = eligible.copy()
    rng.shuffle(shuffled)
    n_hold = max(1, int(len(shuffled) * holdout_frac))
    holdout_idx = np.sort(shuffled[:n_hold])
    train_idx = np.sort(shuffled[n_hold:])
    if train_idx.size == 0:
        raise ValueError("movie split produced an empty train set")
    mask = np.zeros(n, dtype=bool)
    mask[holdout_idx] = True
    return MovieSplit(train_idx=train_idx, holdout_idx=holdout_idx, holdout_mask=mask)


def temporal_holdout(
    years: NDArray[np.integer],
    n_ratings: NDArray[np.integer],
    *,
    cutoff_year: int,
    min_train_ratings: int,
) -> MovieSplit:
    """Train on films released before cutoff_year; test after, among well-rated titles."""
    n = int(n_ratings.shape[0])
    rated = np.asarray(n_ratings) >= min_train_ratings
    train_idx = np.flatnonzero(rated & (np.asarray(years) < cutoff_year) & (np.asarray(years) > 0)).astype(
        np.int64
    )
    holdout_idx = np.flatnonzero(
        rated & (np.asarray(years) >= cutoff_year) & (np.asarray(years) > 0)
    ).astype(np.int64)
    if train_idx.size == 0 or holdout_idx.size == 0:
        raise ValueError("temporal split is empty on one side")
    mask = np.zeros(n, dtype=bool)
    mask[holdout_idx] = True
    return MovieSplit(train_idx=train_idx, holdout_idx=holdout_idx, holdout_mask=mask)
