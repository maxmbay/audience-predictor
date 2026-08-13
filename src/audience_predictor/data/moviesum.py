"""Ingest MovieSum gold screenplays. Scripts stay local; git gets identifiers only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from audience_predictor.data.bridge import catalog_value_root, movies_path
from audience_predictor.data.encoder import HASH_ENCODER, BlockEncoder, get_encoder
from audience_predictor.data.entity_resolution import movies_lookup
from audience_predictor.data.narrative import (
    N_STRUCT,
    NarrativeBatch,
    _pad_blocks,
    scene_block_texts,
)
from audience_predictor.data.screenplay_parser import (
    STRUCT_KEYS,
    parse_moviesum_xml,
    structural_features,
)
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


def blocks_path(encoder: str = HASH_ENCODER) -> Path:
    if not encoder or encoder == HASH_ENCODER:
        return gold_processed_dir() / "moviesum_blocks.npz"
    return gold_processed_dir() / f"moviesum_blocks_{encoder}.npz"


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
    encoder: str | BlockEncoder = HASH_ENCODER,
) -> pl.DataFrame:
    """Parse XML, join IMDb ids to MovieLens, write provenance + block features.

    Does not write screenplay text.
    """
    backend = encoder if hasattr(encoder, "embed_many") else get_encoder(str(encoder), hash_dim)
    cv = catalog_value_root(catalog_root)
    movies = movies_lookup(pl.read_parquet(movies_path(cv)))
    n_movies = int(movies["movie_row"].max()) + 1
    lookup = dict(zip(movies["imdb_join"].to_list(), movies["movie_row"].to_list()))
    id_lookup = dict(zip(movies["imdb_join"].to_list(), movies["movieId"].to_list()))

    records = []
    pending: dict[int, tuple[list[str], list[str], np.ndarray, int]] = {}

    for split, row in tqdm(iter_moviesum_records(), total=2200, desc="moviesum parse"):
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
        if idx in pending and int(parsed.n_scenes) <= pending[idx][3]:
            continue
        dialogue, action = scene_block_texts(parsed, max_blocks)
        stats = structural_features(parsed)
        struct = np.array([stats[k] for k in STRUCT_KEYS], dtype=np.float32)
        pending[idx] = (dialogue, action, struct, int(parsed.n_scenes))

    frame = pl.DataFrame(records)
    dest = provenance_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(dest)

    dialogue_blocks: list[np.ndarray] = [np.zeros((0, backend.dim), np.float32)] * n_movies
    action_blocks: list[np.ndarray] = [np.zeros((0, backend.dim), np.float32)] * n_movies
    structs = np.zeros((n_movies, N_STRUCT), dtype=np.float32)
    types = np.array(["unknown"] * n_movies, dtype=object)

    d_texts: list[str] = []
    a_texts: list[str] = []
    owners: list[tuple[int, int, int]] = []
    for idx, (dialogue, action, struct, _n) in pending.items():
        start = len(d_texts)
        d_texts.extend(dialogue)
        a_texts.extend(action)
        owners.append((idx, start, len(dialogue)))
        structs[idx] = struct
        types[idx] = "screenplay"

    print(f"encoding {len(d_texts):,} dialogue + {len(a_texts):,} action blocks with {backend.name}")
    d_vecs = backend.embed_many(d_texts) if d_texts else np.zeros((0, backend.dim), np.float32)
    a_vecs = backend.embed_many(a_texts) if a_texts else np.zeros((0, backend.dim), np.float32)
    for idx, start, n in owners:
        dialogue_blocks[idx] = d_vecs[start : start + n]
        action_blocks[idx] = a_vecs[start : start + n]

    packed_d, mask = _pad_blocks(dialogue_blocks, max_blocks, backend.dim)
    packed_a, _ = _pad_blocks(action_blocks, max_blocks, backend.dim)
    out_blocks = blocks_path(backend.name)
    np.savez_compressed(
        out_blocks,
        dialogue=packed_d,
        action=packed_a,
        mask=mask,
        structure=structs,
        document_type=types.astype(str),
        encoder=np.asarray(backend.name),
        dim=np.asarray(backend.dim),
    )
    n_join = int(frame["movie_row"].is_not_null().sum())
    print(f"MovieSum {frame.height} screenplays  joined to MovieLens core {n_join}")
    print(f"Wrote {dest} and {out_blocks} (no script text, encoder={backend.name})")
    return frame


def load_moviesum_blocks(encoder: str = HASH_ENCODER) -> NarrativeBatch:
    payload = np.load(blocks_path(encoder), allow_pickle=True)
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
        frame.filter(pl.col("movie_row").is_not_null() & pl.col("parse_ok"))["movie_row"].to_numpy()
    )
    return np.unique(rows.astype(np.int64))
