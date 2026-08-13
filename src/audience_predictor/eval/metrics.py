from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from audience_predictor.models.losses import gaussian_crps, gaussian_nll


def cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> np.ndarray:
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return (an * bn).sum(axis=1)


def rmse(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def recovery_metrics(
    pred_z: NDArray[np.floating],
    true_z: NDArray[np.floating],
    log_var: NDArray[np.floating] | None = None,
    pred_aux: NDArray[np.floating] | None = None,
    true_aux: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    cos = cosine(pred_z, true_z)
    out = {
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(np.median(cos)),
        "rmse": rmse(pred_z, true_z),
        "n": int(pred_z.shape[0]),
    }
    if log_var is not None:
        mu = jnp.asarray(pred_z)
        lv = jnp.asarray(log_var)[:, None]
        tgt = jnp.asarray(true_z)
        out["nll"] = float(gaussian_nll(mu, lv, tgt))
        out["crps"] = float(gaussian_crps(mu, lv, tgt))
        out["mean_sigma"] = float(np.exp(0.5 * np.asarray(log_var)).mean())
        out["error_vs_sigma_spearman"] = _spearman(1.0 - cos, np.exp(0.5 * np.asarray(log_var)))
    if pred_aux is not None and true_aux is not None:
        names = ("mean_rating", "rating_std", "p_love", "p_hate")
        for i, name in enumerate(names):
            out[f"spearman_{name}"] = _spearman(pred_aux[:, i], true_aux[:, i])
    n = int(out["n"])
    cleaned = {k: _finite(v) for k, v in out.items() if k != "n"}
    cleaned["n"] = n
    return cleaned


def _spearman(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    stat = spearmanr(a, b).statistic
    return _finite(stat)


def _finite(value: object) -> float:
    if value is None:
        return 0.0
    number = float(value)
    return number if np.isfinite(number) else 0.0
