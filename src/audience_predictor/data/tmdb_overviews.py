"""Optional TMDB overviews as silver synopses. Cached; never required for tests."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl
from tqdm import tqdm

from audience_predictor.data.env import tmdb_api_key
from audience_predictor.paths import data_dir

TMDB_BASE = "https://api.themoviedb.org/3"
USER_AGENT = "audience-predictor/0.1 (research; overview snapshot)"


def overview_cache_path() -> Path:
    path = data_dir() / "raw" / "tmdb"
    path.mkdir(parents=True, exist_ok=True)
    return path / "overviews.parquet"


def _request_json(path: str, params: dict[str, str], *, retries: int = 4) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{TMDB_BASE}{path}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"_http_status": 404}
            if exc.code in {429, 500, 502, 503} and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                last_error = exc
                continue
            raise
        except urllib.error.URLError as exc:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                last_error = exc
                continue
            raise
    if last_error:
        raise last_error
    return {}


def fetch_overview(tmdb_id: int, api_key: str) -> str:
    payload = _request_json(f"/movie/{int(tmdb_id)}", {"api_key": api_key, "language": "en-US"})
    if payload.get("_http_status") == 404:
        return ""
    return str(payload.get("overview") or "")


def load_or_fetch_overviews(
    movies: pl.DataFrame,
    *,
    force: bool = False,
    limit: int | None = None,
) -> pl.DataFrame:
    cache = overview_cache_path()
    if cache.exists() and not force:
        cached = pl.read_parquet(cache)
        if cached.height >= min(movies.filter(pl.col("tmdbId").is_not_null()).height, limit or 10**9):
            return cached
    key = tmdb_api_key()
    if not key:
        if cache.exists():
            return pl.read_parquet(cache)
        return pl.DataFrame({"tmdbId": [], "overview": []})
    rows = movies.filter(pl.col("tmdbId").is_not_null()).select(["movie_row", "tmdbId", "title"])
    if limit is not None:
        rows = rows.head(limit)
    records = []
    for row in tqdm(rows.iter_rows(named=True), total=rows.height, desc="tmdb overviews"):
        records.append(
            {
                "movie_row": int(row["movie_row"]),
                "tmdbId": int(row["tmdbId"]),
                "overview": fetch_overview(int(row["tmdbId"]), key),
            }
        )
        time.sleep(0.02)
    frame = pl.DataFrame(records)
    frame.write_parquet(cache)
    return frame
