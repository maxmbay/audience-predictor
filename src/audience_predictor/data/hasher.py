"""Deterministic character n-gram hashing. No learned tokenizer, no PyTorch."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.lower()).strip()


def _bucket(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def hash_embed(text: str, dim: int, ngram_ns: tuple[int, ...] = (3, 4, 5)) -> FloatArray:
    vec = np.zeros(dim, dtype=np.float32)
    body = f" {normalize_text(text)} "
    if len(body.strip()) == 0:
        return vec
    for n in ngram_ns:
        if len(body) < n:
            continue
        for i in range(len(body) - n + 1):
            vec[_bucket(body[i : i + n], dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def hash_embed_many(texts: list[str], dim: int) -> FloatArray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack([hash_embed(t, dim) for t in texts], axis=0)
