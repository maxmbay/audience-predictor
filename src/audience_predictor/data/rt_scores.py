"""Rotten Tomatoes critic / audience scores joined onto MovieLens titles.

Labels come from the GroupLens hetrec 2011 snapshot (scores through ~2011).
That is a reception target — a film that already existed — not a script-time
fact. We still predict it from narrative text so the guess card can speak
the language people already use.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import polars as pl
from numpy.typing import NDArray

from audience_predictor.data.bridge import catalog_value_root, movies_path
from audience_predictor.paths import data_dir

HETREC_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"
HETREC_MOVIES = "movies.dat"
USER_AGENT = "audience-predictor/0.1 (research; hetrec snapshot)"

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


def normalize_imdb(value: object) -> str:
    """Strip tt / leading zeros so 0114709 and 114709 match."""
    text = str(value or "").strip().lower()
    if text in {"", "none", "null", "\\n", "nan"}:
        return ""
    if text.startswith("tt"):
        text = text[2:]
    text = text.lstrip("0")
    return text or "0"


def hetrec_dir() -> Path:
    path = data_dir() / "raw" / "hetrec"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scores_path() -> Path:
    path = data_dir() / "processed" / "rt"
    path.mkdir(parents=True, exist_ok=True)
    return path / "rt_scores.parquet"


def parse_percent(value: object) -> float | None:
    text = str(value or "").strip().replace("%", "")
    if text in {"", "\\N", "N/A", "na", "null", "-1"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not np.isfinite(number):
        return None
    if number > 1.0:
        number = number / 100.0
    if number < 0.0 or number > 1.0:
        return None
    return number


def parse_count(value: object) -> int:
    """Review / rating counts. Anything unparseable counts as no reviews."""
    text = str(value or "").strip()
    try:
        return int(float(text))
    except ValueError:
        return 0


def parse_optional_count(value: object) -> int | None:
    """Like parse_count, but \\N / blanks stay missing so 0-fresh is distinct from unknown."""
    text = str(value or "").strip()
    if text in {"", "\\N", "N/A", "na", "null", "-1"}:
        return None
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number >= 0 else None


def _binomial_k(score: float | None, n: int, fresh: int | None) -> int | None:
    """Fresh reviews if hetrec has them; otherwise round(score × n)."""
    if n <= 0 or score is None:
        return None
    k = int(fresh) if fresh is not None else int(round(float(score) * n))
    return int(min(max(k, 0), n))


def parse_hetrec_movies(text: str) -> pl.DataFrame:
    frame = pl.read_csv(io.StringIO(text), separator="\t", infer_schema_length=0, encoding="utf8")
    if "imdbID" not in frame.columns:
        raise ValueError("hetrec movies.dat is missing imdbID")
    payload = {
        "imdb_norm": frame["imdbID"].map_elements(normalize_imdb, return_dtype=pl.Utf8),
        "rt_critics": frame["rtAllCriticsScore"].map_elements(parse_percent, return_dtype=pl.Float64),
        "rt_audience": frame["rtAudienceScore"].map_elements(parse_percent, return_dtype=pl.Float64),
        "n_critics": frame["rtAllCriticsNumReviews"].map_elements(parse_count, return_dtype=pl.Int64),
        "n_audience": frame["rtAudienceNumRatings"].map_elements(parse_count, return_dtype=pl.Int64),
        "fresh_critics": frame["rtAllCriticsNumFresh"].map_elements(
            parse_optional_count, return_dtype=pl.Int64
        ),
    }
    if "rtID" in frame.columns:
        payload["rt_id"] = frame["rtID"]
    # hetrec writes an unrated film as score 0 with zero reviews rather than \N.
    # Taken at face value that pins ~26% of the audience column to a false 0%.
    rows = []
    for rec in pl.DataFrame(payload).iter_rows(named=True):
        n_c = int(rec["n_critics"] or 0)
        n_a = int(rec["n_audience"] or 0)
        critics = rec["rt_critics"] if n_c > 0 else None
        audience = rec["rt_audience"] if n_a > 0 else None
        row = {
            "imdb_norm": rec["imdb_norm"],
            "rt_critics": critics,
            "rt_audience": audience,
            "n_critics": n_c if n_c > 0 else None,
            "n_audience": n_a if n_a > 0 else None,
            "k_critics": _binomial_k(critics, n_c, rec.get("fresh_critics")),
            "k_audience": _binomial_k(audience, n_a, None),
        }
        if "rt_id" in rec:
            row["rt_id"] = rec["rt_id"]
        rows.append(row)
    return pl.DataFrame(rows).filter(pl.col("imdb_norm") != "")


def download_hetrec_movies(dest_dir: Path | None = None) -> Path:
    folder = dest_dir or hetrec_dir()
    movies = folder / HETREC_MOVIES
    if movies.exists():
        return movies
    archive = folder / "hetrec2011-movielens-2k-v2.zip"
    if not archive.exists():
        req = Request(HETREC_URL, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=120) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as zf:
        movies.write_bytes(zf.read(HETREC_MOVIES))
    return movies


def ingest_rt_scores(*, catalog_root: str | Path | None = None) -> pl.DataFrame:
    """Join hetrec RT scores onto the MovieLens spine and cache the table."""
    root = catalog_value_root(catalog_root)
    movies = pl.read_parquet(movies_path(root)).sort("movie_row")
    raw = download_hetrec_movies().read_text(encoding="latin-1")
    scores = parse_hetrec_movies(raw)
    joined = movies.with_columns(
        pl.col("imdbId").map_elements(normalize_imdb, return_dtype=pl.Utf8).alias("imdb_norm")
    ).join(scores, on="imdb_norm", how="left")
    cols = [
        "movie_row",
        "movieId",
        "imdbId",
        "title",
        "rt_critics",
        "rt_audience",
        "n_critics",
        "n_audience",
        "k_critics",
        "k_audience",
    ]
    if "rt_id" in joined.columns:
        cols.append("rt_id")
    out = joined.select(cols)
    dest = scores_path()
    out.write_parquet(dest)
    n_c = int(out["rt_critics"].is_not_null().sum())
    n_a = int(out["rt_audience"].is_not_null().sum())
    print(f"RT snapshot joined  critics={n_c:,}  audience={n_a:,}  of {out.height:,}  → {dest}")
    return out


def load_rt_scores() -> pl.DataFrame:
    path = scores_path()
    if not path.exists():
        return ingest_rt_scores()
    return pl.read_parquet(path)


def rt_target_matrices(n_movies: int, table: pl.DataFrame | None = None) -> tuple[FloatArray, BoolArray]:
    """Return (n, 2) scores in [0, 1] and a mask. Missing stays 0 with mask False."""
    frame = table if table is not None else load_rt_scores()
    y = np.zeros((n_movies, 2), dtype=np.float32)
    mask = np.zeros((n_movies, 2), dtype=bool)
    rows = frame.filter(pl.col("movie_row").is_not_null())
    for row in rows.iter_rows(named=True):
        idx = int(row["movie_row"])
        if idx < 0 or idx >= n_movies:
            continue
        critics = row.get("rt_critics")
        audience = row.get("rt_audience")
        if critics is not None:
            y[idx, 0] = float(critics)
            mask[idx, 0] = True
        if audience is not None:
            y[idx, 1] = float(audience)
            mask[idx, 1] = True
    return y, mask


def rt_count_matrices(
    n_movies: int, table: pl.DataFrame | None = None
) -> tuple[NDArray[np.int32], NDArray[np.int32], BoolArray]:
    """Binomial counts (k, n) per title. k is fresh / liked, n is reviews.

    A 100% Tomatometer with one review is (k=1, n=1), not a delta at 1.0.
    """
    frame = table if table is not None else load_rt_scores()
    needed = {"k_critics", "n_critics", "k_audience", "n_audience"}
    if not needed.issubset(set(frame.columns)):
        frame = ingest_rt_scores()
    k = np.zeros((n_movies, 2), dtype=np.int32)
    n = np.zeros((n_movies, 2), dtype=np.int32)
    mask = np.zeros((n_movies, 2), dtype=bool)
    rows = frame.filter(pl.col("movie_row").is_not_null())
    for row in rows.iter_rows(named=True):
        idx = int(row["movie_row"])
        if idx < 0 or idx >= n_movies:
            continue
        for col, kk, nn in (
            (0, row.get("k_critics"), row.get("n_critics")),
            (1, row.get("k_audience"), row.get("n_audience")),
        ):
            if kk is None or nn is None:
                continue
            reviews = int(nn)
            if reviews <= 0:
                continue
            k[idx, col] = int(min(max(int(kk), 0), reviews))
            n[idx, col] = reviews
            mask[idx, col] = True
    return k, n, mask
