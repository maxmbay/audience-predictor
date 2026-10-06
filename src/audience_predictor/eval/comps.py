"""Nearest-neighbor comps in collaborative vs predicted audience space."""

from __future__ import annotations

import numpy as np
import polars as pl

COMP_MIN_RATINGS = 1500
COMP_FAMOUS_RATINGS = 15_000
COMP_K = 10
_CHUNK = 512


def popular_rows(movies: pl.DataFrame, min_ratings: int = COMP_MIN_RATINGS) -> np.ndarray:
    return (
        movies.filter(pl.col("n_ratings") >= min_ratings)["movie_row"]
        .to_numpy()
        .astype(np.int64)
    )


def neighbor_table(
    movies: pl.DataFrame,
    z: np.ndarray,
    title: str,
    *,
    k: int = 6,
    min_ratings: int = COMP_MIN_RATINGS,
) -> pl.DataFrame | None:
    hit = movies.filter(pl.col("title") == title)
    if hit.is_empty():
        return None
    idx = int(hit["movie_row"][0])
    unit = _unit(z)
    scores = unit @ unit[idx]
    scores[idx] = -np.inf
    popular = set(int(r) for r in popular_rows(movies, min_ratings))
    ranked = [int(i) for i in np.argsort(scores)[::-1] if int(i) in popular][:k]
    return _comp_frame(movies, ranked, scores)


def neighbors_from_vector(
    movies: pl.DataFrame,
    catalog_z: np.ndarray,
    mu: np.ndarray,
    *,
    k: int = 6,
    min_ratings: int = COMP_MIN_RATINGS,
) -> pl.DataFrame:
    """Nearest catalog titles to an arbitrary predicted fingerprint."""
    unit = _unit(catalog_z)
    query = mu / max(float(np.linalg.norm(mu)), 1e-8)
    scores = unit @ query
    popular = set(int(r) for r in popular_rows(movies, min_ratings))
    ranked = [int(i) for i in np.argsort(scores)[::-1] if int(i) in popular][:k]
    return _comp_frame(movies, ranked, scores)


def neighbor_recall(
    catalog_z: np.ndarray,
    pred_z: np.ndarray,
    hold_idx: np.ndarray,
    popular: np.ndarray,
    *,
    k: int = COMP_K,
) -> dict[str, float]:
    """Recall@k of collaborative comps among films nearest the predicted μ."""
    hold = np.asarray(hold_idx, dtype=np.int64)
    if hold.size == 0:
        return {
            "neighbor_recall_at_k": 0.0,
            "neighbor_k": float(k),
            "neighbor_recall_chance": 0.0,
            "n": 0.0,
        }
    collab = _unit(np.asarray(catalog_z, dtype=np.float64))
    pred = _unit(np.asarray(pred_z, dtype=np.float64))
    allowed = np.zeros(collab.shape[0], dtype=bool)
    allowed[np.asarray(popular, dtype=np.int64)] = True
    n_pop = int(allowed.sum())
    chance = float(k) / float(max(n_pop - 1, 1))
    hits = np.empty(hold.size, dtype=np.float64)
    for start in range(0, hold.size, _CHUNK):
        sl = hold[start : start + _CHUNK]
        true_scores = collab @ collab[sl].T
        pred_scores = collab @ pred[sl].T
        for col, idx in enumerate(sl):
            i = int(idx)
            truth = _topk_allowed(true_scores[:, col], k, i, allowed)
            guessed = _topk_allowed(pred_scores[:, col], k, i, allowed)
            hits[start + col] = len(truth & guessed) / float(k)
    return {
        "neighbor_recall_at_k": float(hits.mean()),
        "neighbor_k": float(k),
        "neighbor_recall_chance": chance,
        "n": float(hits.size),
    }


def product_metrics(
    catalog_z: np.ndarray,
    pred_z: np.ndarray,
    hold_idx: np.ndarray,
    n_ratings: np.ndarray,
    popular: np.ndarray,
    *,
    k: int = COMP_K,
) -> dict[str, float]:
    """Comps recall and popularity-stratified recovery on a holdout slice."""
    hold = np.asarray(hold_idx, dtype=np.int64)
    if hold.size == 0:
        return {}
    out = neighbor_recall(catalog_z, pred_z, hold, popular, k=k)
    prefix = f"neighbor_recall_at_{k}"
    result = {
        prefix: out["neighbor_recall_at_k"],
        "neighbor_k": out["neighbor_k"],
        "neighbor_recall_chance": out["neighbor_recall_chance"],
    }
    counts = np.asarray(n_ratings, dtype=np.float64)[hold]
    median = float(np.median(counts))
    result["popularity_median"] = median
    famous = hold[counts >= median]
    obscure = hold[counts < median]
    for name, idx in (("famous", famous), ("obscure", obscure)):
        if idx.size < 8:
            continue
        rec = neighbor_recall(catalog_z, pred_z, idx, popular, k=k)
        result[f"{prefix}_{name}"] = rec["neighbor_recall_at_k"]
        result[f"n_{name}"] = rec["n"]
        result[f"cosine_{name}"] = _mean_cosine(pred_z[idx], catalog_z[idx])
    return result


def _unit(z: np.ndarray) -> np.ndarray:
    return z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-8, None)


def _mean_cosine(pred: np.ndarray, catalog: np.ndarray) -> float:
    return float((_unit(pred) * _unit(catalog)).sum(axis=1).mean())


def _topk_allowed(scores: np.ndarray, k: int, skip: int, allowed: np.ndarray) -> set[int]:
    ranked = scores.copy()
    ranked[skip] = -np.inf
    ranked[~allowed] = -np.inf
    n = int(np.isfinite(ranked).sum())
    if n <= 0:
        return set()
    kk = min(k, n)
    idx = np.argpartition(ranked, -kk)[-kk:]
    return {int(i) for i in idx if np.isfinite(ranked[i])}


def _comp_frame(movies: pl.DataFrame, ranked: list[int], scores: np.ndarray) -> pl.DataFrame:
    titles = dict(zip(movies["movie_row"].to_list(), movies["title"].to_list()))
    genres = dict(zip(movies["movie_row"].to_list(), movies["genres"].to_list()))
    return pl.DataFrame(
        {
            "title": [titles[i] for i in ranked],
            "genres": [genres[i] for i in ranked],
            "cosine": [float(scores[i]) for i in ranked],
        }
    )
