"""Frozen block encoders. Training stays in JAX; this only precomputes features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from audience_predictor.data.hasher import hash_embed_many
from audience_predictor.paths import data_dir

FloatArray = NDArray[np.float32]

BGE_SMALL = "BAAI/bge-small-en-v1.5"
BGE_DIM = 384
HASH_ENCODER = "hash"
BGE_ENCODER = "bge"


def encoder_cache_dir() -> Path:
    path = data_dir() / "raw" / "fastembed"
    path.mkdir(parents=True, exist_ok=True)
    return path


class BlockEncoder(Protocol):
    name: str
    dim: int

    def embed_many(self, texts: list[str]) -> FloatArray: ...


@dataclass(frozen=True)
class HashEncoder:
    name: str = HASH_ENCODER
    dim: int = 256

    def embed_many(self, texts: list[str]) -> FloatArray:
        return hash_embed_many(texts, self.dim)


@dataclass
class BgeEncoder:
    """ONNX BGE-small via fastembed. No PyTorch."""

    name: str = BGE_ENCODER
    dim: int = BGE_DIM
    model_name: str = BGE_SMALL
    _model: object | None = field(default=None, init=False, repr=False)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "BGE ingest needs fastembed (ONNX). Install with: uv sync --extra encoder"
            ) from exc
        self._model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(encoder_cache_dir()),
        )
        return self._model

    def embed_many(self, texts: list[str]) -> FloatArray:
        from tqdm import tqdm

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        indexed = [(i, f"passage: {t.strip()}") for i, t in enumerate(texts) if t.strip()]
        if not indexed:
            return out
        model = self._load()
        bodies = [body for _, body in indexed]
        vecs = list(
            tqdm(
                model.embed(bodies, batch_size=64),
                total=len(bodies),
                desc=f"bge embed ({self.model_name})",
            )
        )
        for (i, _), vec in zip(indexed, vecs, strict=True):
            out[i] = np.asarray(vec, dtype=np.float32)
        return out


def get_encoder(name: str, hash_dim: int = 256) -> HashEncoder | BgeEncoder:
    key = (name or HASH_ENCODER).lower()
    if key in {HASH_ENCODER, "blake2b"}:
        return HashEncoder(dim=hash_dim)
    if key in {BGE_ENCODER, "bge-small", BGE_SMALL.lower()}:
        return BgeEncoder()
    raise ValueError(f"unknown block encoder {name!r}")
