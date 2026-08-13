"""Title/year matching onto the MovieLens core spine."""

from __future__ import annotations

import re
import unicodedata

import polars as pl

_YEAR = re.compile(r"\((\d{4})\)\s*$")
_PUNCT = re.compile(r"[^a-z0-9]+")
_ARTICLES = re.compile(r"^(the|a|an)\s+")


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = _YEAR.sub("", text).strip()
    if "," in text:
        left, right = text.rsplit(",", 1)
        if right.strip() in {"the", "a", "an"}:
            text = f"{right.strip()} {left.strip()}"
    text = _ARTICLES.sub("", text)
    return _PUNCT.sub(" ", text).strip()


def extract_year(title: str) -> int | None:
    match = _YEAR.search(title)
    return int(match.group(1)) if match else None


def movies_lookup(movies: pl.DataFrame) -> pl.DataFrame:
    return movies.with_columns(
        pl.col("title").map_elements(normalize_title, return_dtype=pl.Utf8).alias("title_norm"),
        pl.col("title").map_elements(lambda t: extract_year(str(t)), return_dtype=pl.Int64).alias("year"),
        pl.col("imdbId").cast(pl.Utf8).str.replace(r"^tt", "").str.zfill(7).alias("imdb_join"),
    )


def resolve_imdb(movies: pl.DataFrame, imdb_id: str) -> pl.DataFrame:
    digits = re.sub(r"\D", "", imdb_id).zfill(7)
    table = movies_lookup(movies)
    return table.filter(pl.col("imdb_join") == digits)


def resolve_title_year(movies: pl.DataFrame, title: str, year: int | None = None) -> pl.DataFrame:
    table = movies_lookup(movies)
    hit = table.filter(pl.col("title_norm") == normalize_title(title))
    if year is not None:
        exact = hit.filter(pl.col("year") == year)
        if exact.height:
            return exact
    return hit
