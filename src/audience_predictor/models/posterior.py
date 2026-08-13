"""Precision-weighted shrink of collaborative embeddings toward a content prior."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from audience_predictor.models.types import TitlePosterior

FloatArray = NDArray[np.float64]


def residual_variance(z_collab: FloatArray, z_content: FloatArray, train_mask: np.ndarray) -> float:
    delta = z_collab[train_mask] - z_content[train_mask]
    return float(np.mean(delta**2))


def shrink_posterior(
    z_collab: FloatArray,
    z_content: FloatArray,
    bias_collab: FloatArray,
    bias_content: FloatArray,
    n_ratings: NDArray[np.int64] | NDArray[np.float64],
    movie_row: NDArray[np.int64],
    *,
    n0: float,
    content_var: float,
    collab_mask: np.ndarray | None = None,
) -> TitlePosterior:
    n = np.asarray(n_ratings, dtype=np.float64)
    if collab_mask is not None:
        n = np.where(collab_mask, n, 0.0)
    weight = n / (n + n0)
    mu = weight[:, None] * z_collab + (1.0 - weight)[:, None] * z_content
    bias = weight * bias_collab + (1.0 - weight) * bias_content
    var = np.maximum(content_var, 1e-8) * (1.0 - weight)
    return TitlePosterior(
        mu=mu.astype(np.float64),
        var=var.astype(np.float64),
        bias=bias.astype(np.float64),
        movie_row=np.asarray(movie_row, dtype=np.int64),
    )
