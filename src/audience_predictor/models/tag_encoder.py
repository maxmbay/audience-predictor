"""ISAB + position → independent genome-tag relevances. Not a simplex."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from audience_predictor.models.hierarchical import InducedAttention, _safe_mask


class ScriptTagEncoder(eqx.Module):
    """Variable-length BGE blocks → independent 0–1 tag relevances.

    Position is fractional progress along the kept spine (not permutation
    invariant). Pooling is ISAB then a learned query. The last layer is a
    sigmoid per tag — genome scores do not sum to 1.
    """

    in_proj: eqx.nn.Linear
    pos_proj: eqx.nn.Linear
    isab: InducedAttention
    query: jax.Array
    pma: eqx.nn.MultiheadAttention
    ln_pma: eqx.nn.LayerNorm
    tag_head: eqx.nn.Linear
    in_dim: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    n_tags: int = eqx.field(static=True)
    pool: str = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        n_tags: int,
        n_heads: int,
        n_inducing: int,
        key: jax.Array,
        pool: str = "isab",
    ) -> None:
        if pool not in {"isab", "pma", "mean"}:
            raise ValueError(f"unknown pool {pool!r}")
        keys = jax.random.split(key, 7)
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.n_tags = n_tags
        self.pool = pool
        self.in_proj = eqx.nn.Linear(in_dim, model_dim, key=keys[0])
        self.pos_proj = eqx.nn.Linear(1, model_dim, key=keys[1])
        self.isab = InducedAttention(model_dim, n_heads, n_inducing, keys[2])
        self.query = jax.random.normal(keys[3], (1, model_dim)) * 0.02
        self.pma = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=model_dim, key=keys[4])
        self.ln_pma = eqx.nn.LayerNorm(model_dim)
        self.tag_head = eqx.nn.Linear(model_dim, n_tags, key=keys[5])

    def _tokens(self, blocks: jax.Array, mask: jax.Array) -> jax.Array:
        n = blocks.shape[0]
        progress = jnp.arange(n, dtype=blocks.dtype)[:, None]
        denom = jnp.clip((mask.astype(blocks.dtype).sum() - 1.0), 1.0)
        progress = progress / denom
        return jax.vmap(self.in_proj)(blocks) + jax.vmap(self.pos_proj)(progress)

    def _pool(self, tokens: jax.Array, mask: jax.Array) -> jax.Array:
        keep = _safe_mask(mask)
        if self.pool == "mean":
            weights = keep.astype(tokens.dtype)[:, None]
            return (tokens * weights).sum(axis=0) / jnp.clip(weights.sum(), 1.0)
        encoded = self.isab(tokens, keep) if self.pool == "isab" else tokens
        pooled = self.pma(self.query, encoded, encoded, mask=keep[None, :])
        return self.ln_pma(pooled[0])

    def __call__(self, blocks: jax.Array, mask: jax.Array) -> jax.Array:
        """Return independent tag relevances in (0, 1)."""
        hidden = self._pool(self._tokens(blocks, mask), mask)
        return jax.nn.sigmoid(self.tag_head(jax.nn.gelu(hidden)))


class TagMeanEncoder(eqx.Module):
    """Masked-mean BGE vector → independent tag relevances."""

    layers: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.LayerNorm, ...]
    head: eqx.nn.Linear
    in_dim: int = eqx.field(static=True)
    n_tags: int = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden: int, n_tags: int, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.in_dim = in_dim
        self.n_tags = n_tags
        self.layers = (
            eqx.nn.Linear(in_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
        )
        self.norms = (eqx.nn.LayerNorm(hidden), eqx.nn.LayerNorm(hidden))
        self.head = eqx.nn.Linear(hidden, n_tags, key=k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = x
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.gelu(norm(layer(h)))
        return jax.nn.sigmoid(self.head(h))
