"""Secondary audience outcomes computed from ratings (not collaborative z)."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def polarization_table(ratings: pl.DataFrame, n_movies: int) -> pl.DataFrame:
    stats = ratings.group_by("movie_row").agg(
        pl.col("rating").mean().alias("mean_rating"),
        pl.col("rating").var().alias("var_rating"),
        (pl.col("rating") >= 4.0).mean().alias("p_love"),
        (pl.col("rating") <= 2.0).mean().alias("p_hate"),
        pl.len().alias("n"),
    )
    frame = pl.DataFrame({"movie_row": np.arange(n_movies, dtype=np.int64)}).join(
        stats, on="movie_row", how="left"
    )
    return frame.with_columns(
        pl.col("mean_rating").fill_null(3.5),
        pl.col("var_rating").fill_null(1.0),
        pl.col("p_love").fill_null(0.0),
        pl.col("p_hate").fill_null(0.0),
        pl.col("n").fill_null(0),
    ).sort("movie_row")


def polarization_matrix(table: pl.DataFrame) -> FloatArray:
    return np.column_stack(
        [
            table["mean_rating"].to_numpy().astype(np.float32),
            np.sqrt(np.maximum(table["var_rating"].to_numpy(), 0.0)).astype(np.float32),
            table["p_love"].to_numpy().astype(np.float32),
            table["p_hate"].to_numpy().astype(np.float32),
        ]
    )
