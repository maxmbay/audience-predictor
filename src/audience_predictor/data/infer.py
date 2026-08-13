"""Turn an arbitrary document into padded BGE (or hash) blocks for inference."""

from __future__ import annotations

import numpy as np

from audience_predictor.data.document import classify_document
from audience_predictor.data.encoder import BlockEncoder, get_encoder
from audience_predictor.data.narrative import (
    HASH_CLIP_CHARS,
    N_STRUCT,
    _clip_text,
    _pad_blocks,
    blocks_from_parsed,
    select_block_indices,
)
from audience_predictor.data.screenplay_parser import parse_screenplay
from audience_predictor.data.subtitles import parse_srt_cues, subtitle_block_texts


def _pack_windows(text: str, max_blocks: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    chunks: list[str] = []
    buf: list[str] = []
    n = 0
    for word in words:
        extra = len(word) + (1 if buf else 0)
        if buf and n + extra > HASH_CLIP_CHARS:
            chunks.append(" ".join(buf))
            buf, n = [word], len(word)
        else:
            buf.append(word)
            n += extra
    if buf:
        chunks.append(" ".join(buf))
    keep = select_block_indices(len(chunks), max_blocks)
    return [_clip_text(chunks[i]) for i in keep]


def blocks_from_document(
    text: str,
    *,
    max_blocks: int,
    encoder: BlockEncoder | None = None,
    hash_dim: int = 256,
):
    """Return dialogue, action, mask, structure, document_type for one document."""
    backend = encoder or get_encoder("bge")
    kind = classify_document(text)
    if kind in {"screenplay", "shooting_script"}:
        d, a, mask, struct = blocks_from_parsed(
            parse_screenplay(text), hash_dim, max_blocks, encoder=backend
        )
        return d, a, mask, struct, kind
    if kind == "subtitle":
        parts = subtitle_block_texts(parse_srt_cues(text), max_blocks)
    else:
        parts = _pack_windows(text, max_blocks)
    if not parts:
        parts = [""]
    vecs = backend.embed_many(parts)
    empty = np.zeros_like(vecs)
    d, mask = _pad_blocks([vecs], max_blocks, backend.dim)
    a, _ = _pad_blocks([empty], max_blocks, backend.dim)
    struct = np.zeros((1, N_STRUCT), dtype=np.float32)
    struct[0, 0] = float(len(parts))
    struct[0, 2] = 1.0
    return d, a, mask, struct, kind
