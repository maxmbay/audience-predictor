"""Set-Transformer-lite over narrative blocks, implemented in Equinox.

A movie is a *set* of scenes (or tags, or synopsis sentences). A learned
query attends over that set; dialogue and action are residual streams.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _safe_mask(mask: jax.Array) -> jax.Array:
    """Guarantee at least one True so attention does not see an empty set."""
    mask = jnp.asarray(mask)
    return jnp.where(mask.any(), mask, mask.at[0].set(True))


class InducedAttention(eqx.Module):
    """One ISAB layer: inducing points compress a variable-length set."""

    inducing: jax.Array
    attn_in: eqx.nn.MultiheadAttention
    attn_out: eqx.nn.MultiheadAttention
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, dim: int, n_heads: int, n_inducing: int, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.inducing = jax.random.normal(k1, (n_inducing, dim)) * 0.02
        self.attn_in = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=dim, key=k2)
        self.attn_out = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=dim, key=k3)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)

    def __call__(self, scenes: jax.Array, mask: jax.Array) -> jax.Array:
        keep = _safe_mask(mask)
        kv_mask = jnp.broadcast_to(keep[None, :], (self.inducing.shape[0], scenes.shape[0]))
        compressed = jax.vmap(self.ln1)(self.attn_in(self.inducing, scenes, scenes, mask=kv_mask))
        q_mask = jnp.broadcast_to(keep[:, None], (scenes.shape[0], compressed.shape[0]))
        return jax.vmap(self.ln2)(scenes + self.attn_out(scenes, compressed, compressed, mask=q_mask))


class NarrativeSetEncoder(eqx.Module):
    """Hash-embedded blocks → probabilistic audience fingerprint.

    Residual formulation:
        z_full = z_dialogue + Δ(action, structure)
    When ``action`` is all zeros, Δ is near zero and z_full ≈ z_dialogue.
    """

    in_proj_d: eqx.nn.Linear
    in_proj_a: eqx.nn.Linear
    struct_proj: eqx.nn.Linear
    isab: InducedAttention
    query: jax.Array
    pma: eqx.nn.MultiheadAttention
    ln_pma: eqx.nn.LayerNorm
    ln_struct: eqx.nn.LayerNorm
    delta_mlp: eqx.nn.Linear
    mu_head: eqx.nn.Linear
    log_var_head: eqx.nn.Linear
    bias_head: eqx.nn.Linear
    aux_head: eqx.nn.Linear
    hash_dim: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        hash_dim: int,
        model_dim: int,
        out_dim: int,
        n_heads: int,
        n_inducing: int,
        n_struct: int,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, 11)
        self.hash_dim = hash_dim
        self.model_dim = model_dim
        self.out_dim = out_dim
        self.in_proj_d = eqx.nn.Linear(hash_dim, model_dim, key=keys[0])
        self.in_proj_a = eqx.nn.Linear(hash_dim, model_dim, key=keys[1])
        self.struct_proj = eqx.nn.Linear(n_struct, model_dim, key=keys[2])
        self.isab = InducedAttention(model_dim, n_heads, n_inducing, keys[3])
        self.query = jax.random.normal(keys[4], (1, model_dim)) * 0.02
        self.pma = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=model_dim, key=keys[5])
        self.ln_pma = eqx.nn.LayerNorm(model_dim)
        self.ln_struct = eqx.nn.LayerNorm(model_dim)
        self.delta_mlp = eqx.nn.Linear(model_dim, out_dim, key=keys[6])
        self.mu_head = eqx.nn.Linear(model_dim, out_dim, key=keys[7])
        self.log_var_head = eqx.nn.Linear(model_dim, 1, key=keys[8])
        self.bias_head = eqx.nn.Linear(model_dim, 1, key=keys[9])
        self.aux_head = eqx.nn.Linear(model_dim, 4, key=keys[10])

    def _pool(self, tokens: jax.Array, mask: jax.Array) -> jax.Array:
        keep = _safe_mask(mask)
        encoded = self.isab(tokens, keep)
        attn_mask = keep[None, :]
        pooled = self.pma(self.query, encoded, encoded, mask=attn_mask)
        return self.ln_pma(pooled[0])

    def encode_dialogue(self, dialogue: jax.Array, mask: jax.Array) -> jax.Array:
        tokens = jax.vmap(self.in_proj_d)(dialogue)
        return self._pool(tokens, mask)

    def encode_delta(self, action: jax.Array, structure: jax.Array, mask: jax.Array) -> jax.Array:
        tokens = jax.vmap(self.in_proj_a)(action)
        pooled = self._pool(tokens, mask) + self.ln_struct(self.struct_proj(structure))
        return self.delta_mlp(jax.nn.gelu(pooled))

    def __call__(
        self,
        dialogue: jax.Array,
        action: jax.Array,
        mask: jax.Array,
        structure: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return μ_full, log_var, bias, aux, μ_dialogue, delta."""
        h_d = self.encode_dialogue(dialogue, mask)
        z_d = self.mu_head(h_d)
        delta = self.encode_delta(action, structure, mask)
        mu = z_d + delta
        log_var = jnp.clip(self.log_var_head(h_d).squeeze(-1), -8.0, 4.0)
        bias = self.bias_head(h_d).squeeze(-1)
        aux = self.aux_head(h_d)
        return mu, log_var, bias, aux, z_d, delta
