"""Gaussian NLL and residual losses used by every Equinox encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


def bce_soft(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Binary cross-entropy with soft 0–1 labels (genome relevances)."""
    p = jnp.clip(pred, 1e-6, 1.0 - 1e-6)
    return -jnp.mean(target * jnp.log(p) + (1.0 - target) * jnp.log(1.0 - p))


def gaussian_nll(mu: jax.Array, log_var: jax.Array, target: jax.Array) -> jax.Array:
    """Isotropic or per-dim NLL. log_var broadcasts against mu."""
    var = jnp.exp(log_var) + 1e-6
    residual = (mu - target) ** 2
    return jnp.mean(0.5 * (jnp.log(var) + residual / var))


def gaussian_crps(mu: jax.Array, log_var: jax.Array, target: jax.Array) -> jax.Array:
    """Closed-form CRPS for N(mu, sigma^2 I), averaged over dimensions."""
    sigma = jnp.sqrt(jnp.exp(log_var) + 1e-6)
    z = (target - mu) / sigma
    pdf = jax.scipy.stats.norm.pdf(z)
    cdf = jax.scipy.stats.norm.cdf(z)
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / jnp.sqrt(jnp.pi))
    return jnp.mean(crps)
