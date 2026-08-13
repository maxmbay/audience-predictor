from __future__ import annotations

import polars as pl

from audience_predictor.data.bridge import tags_csv_path


def load_tags(root=None) -> pl.DataFrame:
    return pl.read_csv(tags_csv_path(root))


def movie_tag_text(tags: pl.DataFrame) -> pl.DataFrame:
    return tags.group_by("movieId").agg(
        pl.col("tag").str.join(" ").alias("tag_text"),
        pl.len().alias("n_tags"),
        pl.col("tag").n_unique().alias("n_unique_tags"),
    )
