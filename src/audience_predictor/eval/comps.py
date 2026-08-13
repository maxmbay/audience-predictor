"""Nearest-neighbor comps in collaborative vs predicted audience space."""

from __future__ import annotations

import numpy as np
import polars as pl


def neighbor_table(
    movies: pl.DataFrame,
    z: np.ndarray,
    title: str,
    *,
    k: int = 6,
    min_ratings: int = 1500,
) -> pl.DataFrame | None:
    hit = movies.filter(pl.col("title") == title)
    if hit.is_empty():
        return None
    idx = int(hit["movie_row"][0])
    unit = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-8, None)
    scores = unit @ unit[idx]
    scores[idx] = -np.inf
    popular = set(int(r) for r in movies.filter(pl.col("n_ratings") >= min_ratings)["movie_row"].to_list())
    ranked = [int(i) for i in np.argsort(scores)[::-1] if int(i) in popular][:k]
    titles = dict(zip(movies["movie_row"].to_list(), movies["title"].to_list()))
    genres = dict(zip(movies["movie_row"].to_list(), movies["genres"].to_list()))
    return pl.DataFrame(
        {
            "title": [titles[i] for i in ranked],
            "genres": [genres[i] for i in ranked],
            "cosine": [float(scores[i]) for i in ranked],
        }
    )
