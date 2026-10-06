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


def beta_binomial_nll(
    alpha: jax.Array,
    beta: jax.Array,
    k: jax.Array,
    n: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Mean NLL of k ~ BetaBinomial(n, α, β). mask ignores missing labels.

    This is the right likelihood for a Tomatometer: a 100% with one review
    is (k=1, n=1), which barely updates a Beta posterior, while 73/73 does.
    """
    alpha = jnp.clip(alpha, 1e-3, 1e6)
    beta = jnp.clip(beta, 1e-3, 1e6)
    k = k.astype(alpha.dtype)
    n = n.astype(alpha.dtype)
    log_choose = (
        jax.scipy.special.gammaln(n + 1.0)
        - jax.scipy.special.gammaln(k + 1.0)
        - jax.scipy.special.gammaln(n - k + 1.0)
    )
    log_lik = (
        log_choose
        + jax.scipy.special.betaln(alpha + k, beta + n - k)
        - jax.scipy.special.betaln(alpha, beta)
    )
    weights = mask.astype(alpha.dtype) * (n > 0).astype(alpha.dtype)
    denom = jnp.clip(jnp.sum(weights), 1.0)
    return -jnp.sum(weights * log_lik) / denom


def gaussian_crps(mu: jax.Array, log_var: jax.Array, target: jax.Array) -> jax.Array:
    """Closed-form CRPS for N(mu, sigma^2 I), averaged over dimensions."""
    sigma = jnp.sqrt(jnp.exp(log_var) + 1e-6)
    z = (target - mu) / sigma
    pdf = jax.scipy.stats.norm.pdf(z)
    cdf = jax.scipy.stats.norm.cdf(z)
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / jnp.sqrt(jnp.pi))
    return jnp.mean(crps)
