"""Tiny head: pooled narrative vector → RT critic / audience scores in [0, 1]."""

from __future__ import annotations

import equinox as eqx
import jax


class ScoreMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.LayerNorm, ...]
    head: eqx.nn.Linear
    in_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden: int, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.in_dim = in_dim
        self.layers = (
            eqx.nn.Linear(in_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
        )
        self.norms = (eqx.nn.LayerNorm(hidden), eqx.nn.LayerNorm(hidden))
        self.head = eqx.nn.Linear(hidden, 2, key=k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = x
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.gelu(norm(layer(h)))
        return jax.nn.sigmoid(self.head(h))
