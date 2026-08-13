"""Silver subtitle ingest from SubLens-20M. Files stay local; git gets identifiers only.

A subtitle is not a screenplay: document_type is always ``subtitle``, the action
stream is empty, and these rows never replace MovieSum gold.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from audience_predictor.data.bridge import catalog_value_root, movies_path
from audience_predictor.data.encoder import HASH_ENCODER, BlockEncoder, get_encoder
from audience_predictor.data.entity_resolution import movies_lookup
from audience_predictor.data.narrative import (
    HASH_CLIP_CHARS,
    N_STRUCT,
    NarrativeBatch,
    _clip_text,
    _pad_blocks,
    select_block_indices,
)
from audience_predictor.data.screenplay_parser import STRUCT_KEYS
from audience_predictor.paths import data_dir

SOURCE = "sublens-20m"
DRIVE_FILE_ID = "1Xmty1wID7RjZLBXIv09hUctq8OMmhOW5"
DRIVE_URL = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}"
TEXT_SUFFIXES = {".srt", ".vtt", ".txt", ".sub"}

_HTML = re.compile(r"<[^>]+>")
_INDEX = re.compile(r"^\d+\s*$")
_TIMECODE = re.compile(
    r"\d{1,2}:\d{2}:\d{2}[,.]\d{0,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{0,3}"
)
_WEBVTT = re.compile(r"^WEBVTT\b", re.IGNORECASE)
_IMDB_STEM = re.compile(r"^(?:tt)?(\d+)$", re.IGNORECASE)


def sublens_raw_dir() -> Path:
    return data_dir() / "raw" / "sublens"


def silver_processed_dir() -> Path:
    path = data_dir() / "processed" / "silver"
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance_path() -> Path:
    return silver_processed_dir() / "sublens_provenance.parquet"


def blocks_path(encoder: str = HASH_ENCODER) -> Path:
    if not encoder or encoder == HASH_ENCODER:
        return silver_processed_dir() / "sublens_blocks.npz"
    return silver_processed_dir() / f"sublens_blocks_{encoder}.npz"


def parse_srt_cues(text: str) -> list[str]:
    """Strip indices, timecodes, and markup; return spoken lines in order."""
    body = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    if _WEBVTT.match(body):
        body = re.sub(r"^WEBVTT.*\n", "", body, count=1, flags=re.IGNORECASE)
    cues: list[str] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if _INDEX.match(lines[0]):
            lines = lines[1:]
        if lines and _TIMECODE.search(lines[0]):
            lines = lines[1:]
        spoken = _HTML.sub(" ", " ".join(lines))
        spoken = re.sub(r"\s+", " ", spoken).strip()
        if spoken:
            cues.append(spoken)
    return cues


def chunk_cues(cues: list[str], max_chars: int = HASH_CLIP_CHARS) -> list[str]:
    """Pack consecutive cues into ~max_chars windows so BGE sees local context."""
    if not cues:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    n = 0
    for cue in cues:
        extra = len(cue) + (1 if buf else 0)
        if buf and n + extra > max_chars:
            chunks.append(" ".join(buf))
            buf, n = [cue], len(cue)
        else:
            buf.append(cue)
            n += extra
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def subtitle_block_texts(cues: list[str], max_blocks: int) -> list[str]:
    chunks = chunk_cues(cues)
    if not chunks:
        return []
    keep = select_block_indices(len(chunks), max_blocks)
    return [_clip_text(chunks[i]) for i in keep]


def subtitle_structure(cues: list[str], n_blocks: int) -> np.ndarray:
    n_cues = max(len(cues), 1)
    lengths = [len(c) for c in cues] or [0]
    lengths.sort()
    median = float(lengths[len(lengths) // 2])
    struct = np.zeros(N_STRUCT, dtype=np.float32)
    keys = {name: i for i, name in enumerate(STRUCT_KEYS)}
    struct[keys["n_scenes"]] = float(len(cues))
    struct[keys["median_scene_length"]] = median
    struct[keys["dialogue_fraction"]] = 1.0
    struct[keys["scene_transition_rate"]] = float(n_blocks) / float(n_cues)
    return struct


def _imdb_from_stem(stem: str) -> str | None:
    match = _IMDB_STEM.match(stem.strip())
    if not match:
        return None
    return match.group(1).zfill(7)


def iter_subtitle_files(raw_dir: Path | None = None):
    root = raw_dir or sublens_raw_dir()
    if not root.exists():
        raise FileNotFoundError(
            f"Missing {root}. Download SubLens-20M from {DRIVE_URL} and extract "
            "the subtitles folder into data/raw/sublens/."
        )
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        key = _imdb_from_stem(path.parent.name) or _imdb_from_stem(path.stem)
        if key is None:
            continue
        yield key, path


def ingest_sublens(
    *,
    hash_dim: int,
    max_blocks: int,
    catalog_root: str | Path | None = None,
    encoder: str | BlockEncoder = HASH_ENCODER,
    raw_dir: Path | None = None,
) -> pl.DataFrame:
    """Parse SRT/VTT files named by IMDb id, join to MovieLens, write blocks.

    Does not write subtitle text.
    """
    backend = encoder if hasattr(encoder, "embed_many") else get_encoder(str(encoder), hash_dim)
    cv = catalog_value_root(catalog_root)
    movies = movies_lookup(pl.read_parquet(movies_path(cv)))
    n_movies = int(movies["movie_row"].max()) + 1
    lookup = dict(zip(movies["imdb_join"].to_list(), movies["movie_row"].to_list()))
    id_lookup = dict(zip(movies["imdb_join"].to_list(), movies["movieId"].to_list()))

    records = []
    pending: dict[int, tuple[list[str], np.ndarray, int]] = {}
    files = list(iter_subtitle_files(raw_dir))
    for key, path in tqdm(files, desc="sublens parse"):
        movie_row = lookup.get(key)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        cues = parse_srt_cues(text)
        if not cues and text.strip():
            cues = [ln.strip() for ln in text.splitlines() if ln.strip() and not _TIMECODE.search(ln)]
        parse_ok = len(cues) > 0
        rec = {
            "document_id": f"sublens:{key}",
            "imdb_id": f"tt{key}",
            "imdb_join": key,
            "source": SOURCE,
            "source_url": DRIVE_URL,
            "document_type": "subtitle",
            "n_cues": int(len(cues)),
            "word_count": int(sum(len(c.split()) for c in cues)),
            "parse_ok": parse_ok,
            "movie_row": int(movie_row) if movie_row is not None else None,
            "movieId": int(id_lookup[key]) if movie_row is not None else None,
            "match_confidence": 1.0 if movie_row is not None else 0.0,
        }
        records.append(rec)
        if movie_row is None or not parse_ok:
            continue
        idx = int(movie_row)
        blocks = subtitle_block_texts(cues, max_blocks)
        if not blocks:
            continue
        if idx in pending and len(cues) <= pending[idx][2]:
            continue
        pending[idx] = (blocks, subtitle_structure(cues, len(blocks)), len(cues))

    frame = pl.DataFrame(records) if records else pl.DataFrame(
        schema={
            "document_id": pl.Utf8,
            "imdb_id": pl.Utf8,
            "imdb_join": pl.Utf8,
            "source": pl.Utf8,
            "source_url": pl.Utf8,
            "document_type": pl.Utf8,
            "n_cues": pl.Int64,
            "word_count": pl.Int64,
            "parse_ok": pl.Boolean,
            "movie_row": pl.Int64,
            "movieId": pl.Int64,
            "match_confidence": pl.Float64,
        }
    )
    dest = provenance_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(dest)

    dialogue_blocks: list[np.ndarray] = [np.zeros((0, backend.dim), np.float32)] * n_movies
    structs = np.zeros((n_movies, N_STRUCT), dtype=np.float32)
    types = np.array(["unknown"] * n_movies, dtype=object)
    d_texts: list[str] = []
    owners: list[tuple[int, int, int]] = []
    for idx, (blocks, struct, _n) in pending.items():
        start = len(d_texts)
        d_texts.extend(blocks)
        owners.append((idx, start, len(blocks)))
        structs[idx] = struct
        types[idx] = "subtitle"

    print(f"encoding {len(d_texts):,} subtitle blocks with {backend.name}")
    d_vecs = backend.embed_many(d_texts) if d_texts else np.zeros((0, backend.dim), np.float32)
    for idx, start, n in owners:
        dialogue_blocks[idx] = d_vecs[start : start + n]

    packed_d, mask = _pad_blocks(dialogue_blocks, max_blocks, backend.dim)
    packed_a = np.zeros_like(packed_d)
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
    n_join = int(frame["movie_row"].is_not_null().sum()) if frame.height else 0
    print(f"SubLens {frame.height} subtitles  joined to MovieLens core {n_join}")
    print(f"Wrote {dest} and {out_blocks} (no subtitle text, encoder={backend.name})")
    return frame


def load_sublens_blocks(encoder: str = HASH_ENCODER) -> NarrativeBatch:
    payload = np.load(blocks_path(encoder), allow_pickle=True)
    return NarrativeBatch(
        dialogue=payload["dialogue"],
        action=payload["action"],
        mask=payload["mask"],
        structure=payload["structure"],
        document_type=payload["document_type"],
    )


def silver_movie_rows() -> np.ndarray:
    path = provenance_path()
    if not path.exists():
        return np.zeros(0, dtype=np.int64)
    frame = pl.read_parquet(path)
    rows = frame.filter(pl.col("movie_row").is_not_null() & pl.col("parse_ok"))["movie_row"].to_numpy()
    return np.unique(rows.astype(np.int64))
