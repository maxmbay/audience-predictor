"""Join coverage: how much narrative we actually have against MovieLens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from audience_predictor.data.bridge import (
    catalog_value_root,
    movies_path,
    ratings_path,
    titles_path,
)
from audience_predictor.data.genome import parse_year
from audience_predictor.data.moviesum import provenance_path
from audience_predictor.data.subtitles import provenance_path as sublens_provenance_path
from audience_predictor.data.tags import load_tags, movie_tag_text
from audience_predictor.paths import reports_dir


def coverage_tables(root=None) -> dict[str, pl.DataFrame | dict]:
    base = catalog_value_root(root)
    movies = pl.read_parquet(movies_path(base))
    ratings = pl.read_parquet(ratings_path(base))
    titles = np.load(titles_path(base))
    years = [parse_year(t) or 0 for t in movies["title"].to_list()]
    movies = movies.with_columns(pl.Series("year", years))
    tags = movie_tag_text(load_tags(base))
    movies = movies.join(tags, on="movieId", how="left")
    n = movies.height
    n_ratings = movies["n_ratings"].to_numpy()
    n_screenplay = 0
    n_join = 0
    n_join_ge_1000 = 0
    n_subtitle = 0
    n_sub_join = 0
    n_sub_join_ge_1000 = 0
    if provenance_path().exists():
        gold = pl.read_parquet(provenance_path())
        n_screenplay = gold.height
        matched = gold.filter(pl.col("movie_row").is_not_null())
        n_join = matched.height
        if n_join:
            rows = set(int(r) for r in matched["movie_row"].to_list())
            n_join_ge_1000 = int(
                movies.filter(pl.col("movie_row").is_in(list(rows)) & (pl.col("n_ratings") >= 1000)).height
            )
    if sublens_provenance_path().exists():
        silver = pl.read_parquet(sublens_provenance_path())
        n_subtitle = int(silver["imdb_join"].n_unique()) if silver.height else 0
        matched = silver.filter(pl.col("movie_row").is_not_null())
        n_sub_join = int(matched["movie_row"].n_unique()) if matched.height else 0
        if n_sub_join:
            rows = set(int(r) for r in matched["movie_row"].to_list())
            n_sub_join_ge_1000 = int(
                movies.filter(pl.col("movie_row").is_in(list(rows)) & (pl.col("n_ratings") >= 1000)).height
            )
    summary = {
        "n_movielens_core": n,
        "n_with_imdb": int(movies["imdbId"].is_not_null().sum()),
        "n_with_tmdb": int(movies["tmdbId"].is_not_null().sum()),
        "n_with_collab_z": int(titles["z"].shape[0]),
        "n_with_ge_100_ratings": int((n_ratings >= 100).sum()),
        "n_with_ge_1000_ratings": int((n_ratings >= 1000).sum()),
        "n_with_ge_10000_ratings": int((n_ratings >= 10000).sum()),
        "n_with_user_tags": int(movies["n_tags"].fill_null(0).gt(0).sum()),
        "n_screenplay": n_screenplay,
        "n_screenplay_intersect_movielens": n_join,
        "n_screenplay_intersect_ge_1000": n_join_ge_1000,
        "n_subtitle": n_subtitle,
        "n_subtitle_intersect_movielens": n_sub_join,
        "n_subtitle_intersect_ge_1000": n_sub_join_ge_1000,
        "year_min": int(min(y for y in years if y) or 0),
        "year_max": int(max(years)),
        "n_ratings_core": int(ratings.height),
        "catalog_value_root": str(base),
    }
    by_decade = (
        movies.filter(pl.col("year") > 0)
        .with_columns(((pl.col("year") // 10) * 10).alias("decade"))
        .group_by("decade")
        .len()
        .rename({"len": "n"})
        .sort("decade")
    )
    return {"summary": summary, "movies": movies, "by_decade": by_decade}


def write_coverage_report(root=None, dest: Path | None = None) -> Path:
    tables = coverage_tables(root)
    summary = tables["summary"]
    dest = dest or reports_dir() / "coverage.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    decades = "\n".join(
        f"| {int(r['decade'])} | {int(r['n'])} |" for r in tables["by_decade"].iter_rows(named=True)
    )
    dest.write_text(
        f"""# Coverage report

Join spine: `catalog-value` core universe. Gold screenplay *text* is not in git
(MovieSum is CC BY-NC 4.0; we store identifiers and derived features only).
`N_screenplay ∩ MovieLens = {summary['n_screenplay_intersect_movielens']}`.

| Quantity | N |
|----------|--:|
| MovieLens core titles | {summary['n_movielens_core']:,} |
| With IMDb id | {summary['n_with_imdb']:,} |
| With TMDB id | {summary['n_with_tmdb']:,} |
| Collaborative embeddings `z_i` | {summary['n_with_collab_z']:,} |
| ≥100 ratings | {summary['n_with_ge_100_ratings']:,} |
| ≥1,000 ratings | {summary['n_with_ge_1000_ratings']:,} |
| ≥10,000 ratings | {summary['n_with_ge_10000_ratings']:,} |
| With MovieLens user tags (silver text) | {summary['n_with_user_tags']:,} |
| Genuine screenplays (MovieSum) | {summary['n_screenplay']:,} |
| Screenplay ∩ MovieLens core | {summary['n_screenplay_intersect_movielens']:,} |
| Screenplay ∩ ≥1,000 ratings | {summary['n_screenplay_intersect_ge_1000']:,} |
| Silver subtitles (SubLens-20M) | {summary.get('n_subtitle', 0):,} |
| Subtitle ∩ MovieLens core | {summary.get('n_subtitle_intersect_movielens', 0):,} |
| Subtitle ∩ ≥1,000 ratings | {summary.get('n_subtitle_intersect_ge_1000', 0):,} |
| Core ratings | {summary['n_ratings_core']:,} |
| Year range | {summary['year_min']}–{summary['year_max']} |

## By decade

| Decade | Titles |
|--------|-------:|
{decades}

catalog-value root: `{summary['catalog_value_root']}`
"""
    )
    return dest
