"""MovieLens genome tags + year + genres as a fixed content matrix."""

from __future__ import annotations

import re

import numpy as np
import polars as pl
from numpy.typing import NDArray

from audience_predictor.data.bridge import genome_scores_path, genome_tags_path
from audience_predictor.data.genome_buckets import Bucket, bucket_tag

FloatArray = NDArray[np.float32]
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def parse_year(title: str) -> int | None:
    match = YEAR_RE.search(title)
    return int(match.group(1)) if match else None


def load_genome_tags(root=None) -> pl.DataFrame:
    return pl.read_parquet(genome_tags_path(root)).sort("tagId")


def genome_matrix(movies: pl.DataFrame, *, root=None) -> FloatArray:
    scores = pl.read_parquet(genome_scores_path(root))
    tags = load_genome_tags(root)
    n_movies = int(movies["movie_row"].max()) + 1
    n_tags = tags.height
    joined = scores.join(movies.select(["movieId", "movie_row"]), on="movieId", how="inner")
    mat = np.zeros((n_movies, n_tags), dtype=np.float32)
    tag_index = joined["tagId"].to_numpy() - 1
    mat[joined["movie_row"].to_numpy(), tag_index] = joined["relevance"].to_numpy().astype(np.float32)
    return mat


def genome_tag_names(root=None) -> list[str]:
    return [str(t) for t in load_genome_tags(root)["tag"].to_list()]


def genome_bucket_mask(bucket: Bucket, *, root=None) -> np.ndarray:
    names = genome_tag_names(root)
    return np.array([bucket_tag(name) == bucket for name in names], dtype=bool)


def genome_bucket_matrix(movies: pl.DataFrame, bucket: Bucket, *, root=None) -> FloatArray:
    """Oracle genome columns that belong to one honesty bucket."""
    full = genome_matrix(movies, root=root)
    keep = genome_bucket_mask(bucket, root=root)
    return full[:, keep]


def side_features(movies: pl.DataFrame) -> FloatArray:
    n_movies = int(movies["movie_row"].max()) + 1
    genres = np.zeros((n_movies, len(GENRES)), dtype=np.float32)
    years = np.zeros((n_movies, 2), dtype=np.float32)
    genre_index = {name: i for i, name in enumerate(GENRES)}
    for row in movies.iter_rows(named=True):
        i = int(row["movie_row"])
        for name in str(row["genres"]).split("|"):
            if name in genre_index:
                genres[i, genre_index[name]] = 1.0
        year = parse_year(str(row["title"]))
        if year is None:
            years[i, 1] = 1.0
        else:
            years[i, 0] = (year - 1995.0) / 20.0
    return np.concatenate([genres, years], axis=1)


def content_features(movies: pl.DataFrame, *, root=None) -> FloatArray:
    return np.concatenate([genome_matrix(movies, root=root), side_features(movies)], axis=1)
