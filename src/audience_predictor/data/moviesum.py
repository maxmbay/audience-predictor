"""Ingest MovieSum gold screenplays. Scripts stay local; git gets identifiers only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from audience_predictor.data.bridge import catalog_value_root, movies_path
from audience_predictor.data.entity_resolution import movies_lookup
from audience_predictor.data.narrative import (
    N_STRUCT,
    NarrativeBatch,
    _pad_blocks,
    blocks_from_parsed,
)
from audience_predictor.data.screenplay_parser import parse_moviesum_xml
from audience_predictor.paths import data_dir

SPLITS = ("train", "val", "test")
SOURCE = "moviesum"
HF_REPO = "rohitsaxena/MovieSum"


def moviesum_raw_dir() -> Path:
    return data_dir() / "raw" / "moviesum"


def gold_processed_dir() -> Path:
    path = data_dir() / "processed" / "gold"
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance_path() -> Path:
    return gold_processed_dir() / "moviesum_provenance.parquet"


def blocks_path() -> Path:
    return gold_processed_dir() / "moviesum_blocks.npz"


def iter_moviesum_records(raw_dir: Path | None = None):
    root = raw_dir or moviesum_raw_dir()
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download MovieSum JSONL from Hugging Face ({HF_REPO})."
            )
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                yield split, row


def _imdb_join(imdb_id: str) -> str:
    digits = "".join(ch for ch in str(imdb_id) if ch.isdigit())
    return digits.zfill(7)


def ingest_moviesum(
    *,
    hash_dim: int,
    max_blocks: int,
    catalog_root: str | Path | None = None,
) -> pl.DataFrame:
    """Parse XML, join IMDb ids to MovieLens, write provenance + hashed blocks.

    Does not write screenplay text.
    """
    cv = catalog_value_root(catalog_root)
    movies = movies_lookup(pl.read_parquet(movies_path(cv)))
    n_movies = int(movies["movie_row"].max()) + 1
    lookup = dict(zip(movies["imdb_join"].to_list(), movies["movie_row"].to_list()))
    id_lookup = dict(zip(movies["imdb_join"].to_list(), movies["movieId"].to_list()))

    records = []
    dialogue_blocks: list[np.ndarray] = [np.zeros((0, hash_dim), np.float32)] * n_movies
    action_blocks: list[np.ndarray] = [np.zeros((0, hash_dim), np.float32)] * n_movies
    structs = np.zeros((n_movies, N_STRUCT), dtype=np.float32)
    types = np.array(["unknown"] * n_movies, dtype=object)
    n_scenes_stored = np.zeros(n_movies, dtype=np.int32)

    for split, row in tqdm(iter_moviesum_records(), total=2200, desc="moviesum"):
        imdb_id = str(row.get("imdb_id") or "")
        key = _imdb_join(imdb_id)
        movie_row = lookup.get(key)
        parsed = parse_moviesum_xml(str(row.get("script") or ""))
        parse_ok = parsed.n_scenes > 0
        rec = {
            "document_id": f"moviesum:{key}:{split}",
            "imdb_id": imdb_id,
            "imdb_join": key,
            "movie_name": str(row.get("movie_name") or ""),
            "source": SOURCE,
            "source_url": f"https://huggingface.co/datasets/{HF_REPO}",
            "document_type": "screenplay",
            "split": split,
            "n_scenes": int(parsed.n_scenes),
            "word_count": len(str(row.get("script") or "").split()),
            "parse_ok": parse_ok,
            "movie_row": int(movie_row) if movie_row is not None else None,
            "movieId": int(id_lookup[key]) if movie_row is not None else None,
            "match_confidence": 1.0 if movie_row is not None else 0.0,
        }
        records.append(rec)
        if movie_row is None or not parse_ok:
            continue
        idx = int(movie_row)
        if types[idx] == "screenplay" and int(parsed.n_scenes) <= int(n_scenes_stored[idx]):
            continue
        d, a, mask, struct = blocks_from_parsed(parsed, hash_dim, max_blocks)
        k = int(mask[0].sum())
        dialogue_blocks[idx] = d[0, :k]
        action_blocks[idx] = a[0, :k]
        structs[idx] = struct[0]
        types[idx] = "screenplay"
        n_scenes_stored[idx] = int(parsed.n_scenes)

    frame = pl.DataFrame(records)
    dest = provenance_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(dest)

    dialogue, mask = _pad_blocks(dialogue_blocks, max_blocks, hash_dim)
    action, _ = _pad_blocks(action_blocks, max_blocks, hash_dim)
    np.savez_compressed(
        blocks_path(),
        dialogue=dialogue,
        action=action,
        mask=mask,
        structure=structs,
        document_type=types.astype(str),
    )
    n_join = int(frame["movie_row"].is_not_null().sum())
    print(f"MovieSum {frame.height} screenplays  joined to MovieLens core {n_join}")
    print(f"Wrote {dest} and {blocks_path()} (no script text)")
    return frame


def load_moviesum_blocks() -> NarrativeBatch:
    payload = np.load(blocks_path(), allow_pickle=True)
    return NarrativeBatch(
        dialogue=payload["dialogue"],
        action=payload["action"],
        mask=payload["mask"],
        structure=payload["structure"],
        document_type=payload["document_type"],
    )


def gold_movie_rows() -> np.ndarray:
    frame = pl.read_parquet(provenance_path())
    rows = (
        frame.filter(pl.col("movie_row").is_not_null() & pl.col("parse_ok"))["movie_row"]
        .to_numpy()
    )
    return np.unique(rows.astype(np.int64))
