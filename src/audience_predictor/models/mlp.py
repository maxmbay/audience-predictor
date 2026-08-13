"""MLP that maps a fixed feature vector into collaborative audience space."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class AudienceMLP(eqx.Module):
    """Genome / metadata / hashed-bag → (μ, log σ², bias, aux).

    aux is four polarization stats: mean rating, rating std, P(love), P(hate).
    """

    layers: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.LayerNorm, ...]
    mu_head: eqx.nn.Linear
    log_var_head: eqx.nn.Linear
    bias_head: eqx.nn.Linear
    aux_head: eqx.nn.Linear
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, hidden: int, key: jax.Array) -> None:
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = (
            eqx.nn.Linear(in_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
        )
        self.norms = (eqx.nn.LayerNorm(hidden), eqx.nn.LayerNorm(hidden))
        self.mu_head = eqx.nn.Linear(hidden, out_dim, key=k3)
        self.log_var_head = eqx.nn.Linear(hidden, 1, key=k4)
        self.bias_head = eqx.nn.Linear(hidden, 1, key=k5)
        self.aux_head = eqx.nn.Linear(hidden, 4, key=k6)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        h = x
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.gelu(norm(layer(h)))
        mu = self.mu_head(h)
        log_var = jnp.clip(self.log_var_head(h).squeeze(-1), -8.0, 4.0)
        bias = self.bias_head(h).squeeze(-1)
        aux = self.aux_head(h)
        return mu, log_var, bias, aux
