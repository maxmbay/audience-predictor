"""Optax training loops for MLP and set-encoder modalities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray
from tqdm import tqdm

from audience_predictor.models.hierarchical import NarrativeSetEncoder
from audience_predictor.models.losses import gaussian_nll, mse
from audience_predictor.models.mlp import AudienceMLP


def _adamw(lr: float, weight_decay: float):
    return optax.adamw(lr, weight_decay=weight_decay)


def save_checkpoint(path: Path, model: eqx.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_checkpoint(path: Path, like: eqx.Module) -> eqx.Module:
    return eqx.tree_deserialise_leaves(path, like)


def train_mlp(
    x: NDArray[np.floating],
    z: NDArray[np.floating],
    bias: NDArray[np.floating],
    aux: NDArray[np.floating],
    train_idx: NDArray[np.integer],
    *,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    nll_weight: float,
    aux_weight: float,
    bias_weight: float,
    seed: int,
) -> tuple[AudienceMLP, list[float]]:
    key = jax.random.key(seed)
    model = AudienceMLP(in_dim=x.shape[1], out_dim=z.shape[1], hidden=hidden, key=key)
    optim = _adamw(lr, weight_decay)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    xt = jnp.asarray(x[train_idx])
    zt = jnp.asarray(z[train_idx])
    bt = jnp.asarray(bias[train_idx])
    at = jnp.asarray(aux[train_idx])

    @eqx.filter_jit
    def step(model, opt_state):
        def loss_fn(model):
            pred = jax.vmap(model)(xt)
            mu, log_var, b, aux_hat = pred
            loss = (
                mse(mu, zt)
                + bias_weight * mse(b, bt)
                + nll_weight * gaussian_nll(mu, log_var[:, None], zt)
                + aux_weight * mse(aux_hat, at)
            )
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    history: list[float] = []
    for epoch in tqdm(range(epochs), desc="audience mlp"):
        model, opt_state, loss = step(model, opt_state)
        history.append(float(loss))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={float(loss):.4f}")
    return model, history


def predict_mlp(model: AudienceMLP, x: NDArray[np.floating]) -> dict[str, np.ndarray]:
    mu, log_var, bias, aux = jax.vmap(model)(jnp.asarray(x))
    return {
        "mu": np.asarray(mu, dtype=np.float64),
        "log_var": np.asarray(log_var, dtype=np.float64),
        "bias": np.asarray(bias, dtype=np.float64),
        "aux": np.asarray(aux, dtype=np.float64),
    }


def train_set_encoder(
    dialogue: NDArray[np.floating],
    action: NDArray[np.floating],
    mask: NDArray[np.bool_],
    structure: NDArray[np.floating],
    z: NDArray[np.floating],
    bias: NDArray[np.floating],
    aux: NDArray[np.floating],
    train_idx: NDArray[np.integer],
    *,
    model_dim: int,
    out_dim: int,
    n_heads: int,
    n_inducing: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    nll_weight: float,
    aux_weight: float,
    bias_weight: float,
    seed: int,
    pool: str = "isab",
    make_model: Callable[[jax.Array], NarrativeSetEncoder] | None = None,
) -> tuple[NarrativeSetEncoder, list[float]]:
    key = jax.random.key(seed)
    if make_model is None:
        model = NarrativeSetEncoder(
            hash_dim=dialogue.shape[-1],
            model_dim=model_dim,
            out_dim=out_dim,
            n_heads=n_heads,
            n_inducing=n_inducing,
            n_struct=structure.shape[-1],
            key=key,
            pool=pool,
        )
    else:
        model = make_model(key)
    optim = _adamw(lr, weight_decay)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    d = jnp.asarray(dialogue[train_idx])
    a = jnp.asarray(action[train_idx])
    m = jnp.asarray(mask[train_idx])
    s = jnp.asarray(structure[train_idx])
    zt = jnp.asarray(z[train_idx])
    bt = jnp.asarray(bias[train_idx])
    at = jnp.asarray(aux[train_idx])

    @eqx.filter_jit
    def step(model, opt_state):
        def loss_fn(model):
            mu, log_var, b, aux_hat, _, _ = jax.vmap(model)(d, a, m, s)
            return (
                mse(mu, zt)
                + bias_weight * mse(b, bt)
                + nll_weight * gaussian_nll(mu, log_var[:, None], zt)
                + aux_weight * mse(aux_hat, at)
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    history: list[float] = []
    for epoch in tqdm(range(epochs), desc="narrative set encoder"):
        model, opt_state, loss = step(model, opt_state)
        history.append(float(loss))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={float(loss):.4f}")
    return model, history


def predict_set_encoder(model: NarrativeSetEncoder, batch) -> dict[str, np.ndarray]:
    mu, log_var, bias, aux, z_d, delta = jax.vmap(model)(
        jnp.asarray(batch.dialogue),
        jnp.asarray(batch.action),
        jnp.asarray(batch.mask),
        jnp.asarray(batch.structure),
    )
    return {
        "mu": np.asarray(mu, dtype=np.float64),
        "log_var": np.asarray(log_var, dtype=np.float64),
        "bias": np.asarray(bias, dtype=np.float64),
        "aux": np.asarray(aux, dtype=np.float64),
        "z_dialogue": np.asarray(z_d, dtype=np.float64),
        "delta": np.asarray(delta, dtype=np.float64),
    }
