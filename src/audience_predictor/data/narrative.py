"""Build padded narrative block tensors for the Equinox set encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from audience_predictor.data.hasher import hash_embed, hash_embed_many
from audience_predictor.data.screenplay_parser import (
    STRUCT_KEYS,
    ParsedScreenplay,
    parse_screenplay,
    structural_features,
)

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
N_STRUCT = len(STRUCT_KEYS)


@dataclass
class NarrativeBatch:
    dialogue: FloatArray  # [n, max_blocks, hash_dim]
    action: FloatArray
    mask: BoolArray  # [n, max_blocks]
    structure: FloatArray  # [n, N_STRUCT]
    document_type: np.ndarray  # [n] unicode


def _pad_blocks(blocks: list[np.ndarray], max_blocks: int, dim: int) -> tuple[FloatArray, BoolArray]:
    n = len(blocks)
    out = np.zeros((n, max_blocks, dim), dtype=np.float32)
    mask = np.zeros((n, max_blocks), dtype=bool)
    for i, mat in enumerate(blocks):
        if mat.size == 0:
            continue
        k = min(mat.shape[0], max_blocks)
        out[i, :k] = mat[:k]
        mask[i, :k] = True
    return out, mask


def blocks_from_screenplay(text: str, hash_dim: int, max_blocks: int) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray]:
    parsed = parse_screenplay(text)
    return blocks_from_parsed(parsed, hash_dim, max_blocks)


def blocks_from_parsed(
    parsed: ParsedScreenplay, hash_dim: int, max_blocks: int
) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray]:
    dialogue = [s.dialogue_text for s in parsed.scenes] or [""]
    action = [f"{s.heading} {s.action_text}".strip() for s in parsed.scenes] or [""]
    d = hash_embed_many(dialogue[:max_blocks], hash_dim)
    a = hash_embed_many(action[:max_blocks], hash_dim)
    packed_d, mask = _pad_blocks([d], max_blocks, hash_dim)
    packed_a, _ = _pad_blocks([a], max_blocks, hash_dim)
    stats = structural_features(parsed)
    struct = np.array([[stats[k] for k in STRUCT_KEYS]], dtype=np.float32)
    return packed_d, packed_a, mask, struct


def blocks_from_sentences(text: str, hash_dim: int, max_blocks: int) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray]:
    parts = [p.strip() for p in re_split_sentences(text) if p.strip()]
    if not parts:
        parts = [text]
    parts = parts[:max_blocks]
    hashed = hash_embed_many(parts, hash_dim)
    empty = np.zeros_like(hashed)
    d, mask = _pad_blocks([hashed], max_blocks, hash_dim)
    a, _ = _pad_blocks([empty], max_blocks, hash_dim)
    struct = np.zeros((1, N_STRUCT), dtype=np.float32)
    struct[0, 0] = float(len(parts))
    struct[0, 2] = 1.0
    return d, a, mask, struct


def re_split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=[.!?])\s+", text.strip())


def tag_blocks_for_movies(
    movies: pl.DataFrame,
    tags: pl.DataFrame,
    *,
    hash_dim: int,
    max_blocks: int,
) -> NarrativeBatch:
    n = int(movies["movie_row"].max()) + 1
    counts = tags.group_by(["movieId", "tag"]).len().rename({"len": "n"})
    ranked = counts.sort(["movieId", "n"], descending=[False, True])
    per_movie: dict[int, list[str]] = {}
    for row in ranked.iter_rows(named=True):
        bucket = per_movie.setdefault(int(row["movieId"]), [])
        if len(bucket) < max_blocks:
            bucket.append(str(row["tag"]))
    structs = np.zeros((n, N_STRUCT), dtype=np.float32)
    types = np.array(["unknown"] * n, dtype=object)
    id_to_row = dict(zip(movies["movieId"].to_list(), movies["movie_row"].to_list()))
    empty = np.zeros((0, hash_dim), dtype=np.float32)
    blocks = [empty] * n
    for movie_id, row in id_to_row.items():
        names = per_movie.get(int(movie_id), [])
        if names:
            blocks[int(row)] = hash_embed_many(names, hash_dim)
            types[int(row)] = "transcript"
            structs[int(row), 0] = float(len(names))
            structs[int(row), 4] = float(len(names))
            structs[int(row), 2] = 1.0
        else:
            types[int(row)] = "unknown"
    dialogue, mask = _pad_blocks(blocks, max_blocks, hash_dim)
    action, _ = _pad_blocks([empty] * n, max_blocks, hash_dim)
    return NarrativeBatch(
        dialogue=dialogue,
        action=action,
        mask=mask,
        structure=structs,
        document_type=types,
    )


def bag_features(texts: list[str], hash_dim: int) -> FloatArray:
    return np.stack([hash_embed(t, hash_dim) for t in texts], axis=0)
