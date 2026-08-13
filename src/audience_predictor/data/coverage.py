"""Join coverage: how much narrative we actually have against MovieLens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from audience_predictor.data.bridge import catalog_value_root, movies_path, ratings_path, titles_path
from audience_predictor.data.genome import parse_year
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
    summary = {
        "n_movielens_core": n,
        "n_with_imdb": int(movies["imdbId"].is_not_null().sum()),
        "n_with_tmdb": int(movies["tmdbId"].is_not_null().sum()),
        "n_with_collab_z": int(titles["z"].shape[0]),
        "n_with_ge_100_ratings": int((n_ratings >= 100).sum()),
        "n_with_ge_1000_ratings": int((n_ratings >= 1000).sum()),
        "n_with_ge_10000_ratings": int((n_ratings >= 10000).sum()),
        "n_with_user_tags": int(movies["n_tags"].fill_null(0).gt(0).sum()),
        "n_screenplay": 0,
        "n_screenplay_intersect_movielens": 0,
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

Join spine: `catalog-value` core universe. Narrative documents are counted
without collapsing document types. Gold screenplays are **not** in this
repository; `N_screenplay ∩ MovieLens = {summary['n_screenplay_intersect_movielens']}`.

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
| Genuine screenplays | {summary['n_screenplay']:,} |
| Screenplay ∩ MovieLens | {summary['n_screenplay_intersect_movielens']:,} |
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
