"""Research figures for the narrative → audience report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from audience_predictor.viz.style import (
    BLUE,
    GOLD,
    INK,
    MUTED,
    RED,
    TEAL,
    VIOLET,
    apply_style,
    savefig,
)


def plot_reconstruction(
    n_ratings: np.ndarray,
    cosine: np.ndarray,
    holdout: np.ndarray,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(
        n_ratings[~holdout],
        cosine[~holdout],
        s=10,
        c=MUTED,
        alpha=0.35,
        linewidths=0,
        label="train titles",
    )
    ax.scatter(
        n_ratings[holdout],
        cosine[~holdout] * 0 + cosine[holdout],
        s=16,
        c=TEAL,
        alpha=0.8,
        linewidths=0,
        label="held-out cold start",
    )
    ax.set_xscale("log")
    ax.set_xlabel("MovieLens rating count")
    ax.set_ylabel("cosine(predicted z, collaborative z)")
    ax.set_title("Cold-start recovery of the audience fingerprint")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return savefig(fig, path)


def plot_ablation(rows: list[tuple[str, float, float]], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    labels = [r[0] for r in rows]
    train = [r[1] for r in rows]
    hold = [r[2] for r in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.barh(y + 0.18, train, height=0.32, color=MUTED, label="train")
    ax.barh(y - 0.18, hold, height=0.32, color=TEAL, label="holdout")
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("mean cosine to collaborative z")
    ax.set_xlim(0, 1)
    ax.set_title("What each information source knows about the audience")
    ax.legend(loc="lower right")
    return savefig(fig, path)


def plot_uncertainty(error: np.ndarray, sigma: np.ndarray, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(sigma, error, s=14, c=VIOLET, alpha=0.55, linewidths=0)
    ax.set_xlabel("predicted σ (isotropic)")
    ax.set_ylabel("1 − cosine(pred, collab)")
    ax.set_title("Does the model know when it does not know?")
    return savefig(fig, path)


def plot_polarization(pred: np.ndarray, true: np.ndarray, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    titles = ("P(rating ≥ 4)  love", "P(rating ≤ 2)  hate")
    colors = (TEAL, RED)
    for ax, i, title, color in zip(axes, (2, 3), titles, colors, strict=True):
        ax.scatter(true[:, i], pred[:, i], s=10, c=color, alpha=0.45, linewidths=0)
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8, alpha=0.4)
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Narrative features and audience polarization", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_scaling(sizes: list[int], holdout_cos: list[float], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(sizes, holdout_cos, marker="o", color=BLUE, lw=2)
    ax.set_xlabel("train titles")
    ax.set_ylabel("holdout cosine")
    ax.set_title("Does more labeled narrative still help?")
    ax.set_ylim(0, 1)
    return savefig(fig, path)


def plot_atlas(
    xy: np.ndarray,
    pred_xy: np.ndarray,
    holdout: np.ndarray,
    sigma: np.ndarray,
    titles: list[str],
    highlight: list[int],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c=MUTED, alpha=0.25, linewidths=0, label="collaborative atlas")
    ax.scatter(
        pred_xy[holdout, 0],
        pred_xy[holdout, 1],
        s=18 + 80 * (sigma[holdout] / (sigma.max() + 1e-8)),
        c=TEAL,
        alpha=0.7,
        linewidths=0,
        label="cold-start prediction",
    )
    for idx in highlight:
        ax.annotate(titles[idx], xy[idx], fontsize=8, color=GOLD)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Where the story lands on the audience map")
    ax.legend(loc="upper right")
    return savefig(fig, path)


def plot_neighbors(panels: list[tuple[str, pl.DataFrame, pl.DataFrame]], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(10.4, 2.5 * n))
    if n == 1:
        axes = np.array([axes])
    for row, (name, collab, pred) in enumerate(panels):
        for col, (table, color, subtitle) in enumerate(
            ((collab, BLUE, "observed audience comps"), (pred, TEAL, "predicted from narrative"))
        ):
            ax = axes[row, col]
            labels = [t.rsplit(" (", 1)[0] for t in table["title"].to_list()]
            scores = table["cosine"].to_numpy()
            y = np.arange(len(labels))[::-1]
            ax.barh(y, scores, color=color, height=0.62)
            ax.set_yticks(y, labels=labels, fontsize=8)
            ax.set_xlim(0, max(0.4, float(scores.max()) * 1.15 if len(scores) else 1.0))
            if row == 0:
                ax.set_title(subtitle, fontsize=11)
            if col == 0:
                short = name.rsplit(" (", 1)[0]
                ax.set_ylabel(short.replace(" ", "\n"), fontsize=9)
    fig.suptitle("Audience comps: observed vs predicted", fontsize=13, y=0.995)
    fig.subplots_adjust(wspace=0.55, hspace=0.38, left=0.16, right=0.98, top=0.90, bottom=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_coverage_funnel(counts: list[tuple[str, int]], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    labels = [c[0] for c in counts]
    values = [c[1] for c in counts]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=BLUE, height=0.6)
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("titles")
    ax.set_title("Coverage funnel")
    for yi, val in zip(y, values, strict=True):
        ax.text(val * 1.01, yi, f"{val:,}", va="center", fontsize=8, color=INK)
    return savefig(fig, path)


def plot_loss(history: list[float], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(np.arange(1, len(history) + 1), history, color=BLUE, lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("Equinox training")
    return savefig(fig, path)
