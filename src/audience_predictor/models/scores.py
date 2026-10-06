"""Pooled narrative vector → Beta posterior over RT critic / audience scores.

The Tomatometer is a binomial proportion (k fresh of n reviews). A point
estimate of 100 is the wrong object: hetrec's 100%s are often one or two
reviews. The head predicts p(π | text) = Beta(α(x), β(x)), trained as a
beta-binomial on the observed (k, n).
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist

MIN_KAPPA = 2.0
INTERVAL = 0.90


class ScoreBeta(eqx.Module):
    """Two-column Beta posterior. loc is logit μ, conc is pre-softplus κ."""

    layers: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.LayerNorm, ...]
    loc: eqx.nn.Linear
    conc: eqx.nn.Linear
    in_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden: int, key: jax.Array) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.in_dim = in_dim
        self.layers = (
            eqx.nn.Linear(in_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
        )
        self.norms = (eqx.nn.LayerNorm(hidden), eqx.nn.LayerNorm(hidden))
        self.loc = eqx.nn.Linear(hidden, 2, key=k3)
        self.conc = eqx.nn.Linear(hidden, 2, key=k4)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = x
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.gelu(norm(layer(h)))
        mu = jax.nn.sigmoid(self.loc(h))
        kappa = jax.nn.softplus(self.conc(h)) + MIN_KAPPA
        alpha = mu * kappa
        beta = (1.0 - mu) * kappa
        return alpha, beta


def beta_mean(alpha: NDArray[np.floating], beta: NDArray[np.floating]) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    return a / np.clip(a + b, 1e-8, None)


def beta_interval(
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    *,
    level: float = INTERVAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed credible interval for π ~ Beta(α, β)."""
    a = np.clip(np.asarray(alpha, dtype=np.float64), 1e-3, 1e6)
    b = np.clip(np.asarray(beta, dtype=np.float64), 1e-3, 1e6)
    tail = (1.0 - level) / 2.0
    lo = beta_dist.ppf(tail, a, b)
    hi = beta_dist.ppf(1.0 - tail, a, b)
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
