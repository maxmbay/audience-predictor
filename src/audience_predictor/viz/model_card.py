"""Figures for the parked baseline model card."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from audience_predictor.viz.style import (
    GOLD,
    GRID,
    INK,
    MUTED,
    PAPER,
    RED,
    TEAL,
    apply_style,
    savefig,
)


def plot_architecture_dag(path: Path) -> Path:
    """Hand-laid DAG of the parked baseline. Teal is the live inference path."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("Parked baseline  ·  narrative → audience fingerprint", loc="left", fontsize=14, pad=8)

    def box(x, y, w, h, text, *, fill="#efece6", edge=INK, lw=1.1, ts=8.5, color=INK):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=ts, color=color, zorder=4)

    def arrow(x1, y1, x2, y2, *, color=MUTED, lw=1.4):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>", mutation_scale=11, lw=lw,
                color=color, zorder=2,
            )
        )

    # Row 0: frozen collab space
    box(8.6, 6.15, 3.4, 0.7, "catalog-value Phase A\n$z_i^{collab} \\in \\mathbb{R}^{64}$  ·  train target", fill="#efece6", ts=8)

    # Inputs
    box(0.25, 4.85, 2.15, 0.85, "screenplay\nMovieSum XML", fill="#e8e4dc")
    box(0.25, 3.75, 2.15, 0.85, "subtitles\nSubLens SRT", fill="#e8e4dc")
    box(0.25, 2.65, 2.15, 0.85, "any text\npredict CLI", fill=TEAL, edge=TEAL, color="white", ts=9)
    box(0.25, 1.15, 2.15, 0.85, "genome tags\n1,128 relevances", fill="#e8e4dc")

    box(2.75, 3.55, 2.35, 1.35, "document_type\nblocks ≤ 32\n~800-char windows", fill="#e8e4dc", ts=8.5)
    box(5.4, 3.55, 2.5, 1.35, "frozen BGE-small\nONNX 384-d\nnever trained here", fill="#dbeae8", edge=TEAL, ts=8.5)
    box(8.2, 3.7, 2.0, 1.05, "masked mean\nDeep Set ρ", fill="#dbeae8", edge=TEAL, ts=9)
    box(10.4, 3.55, 1.75, 1.35, "AudienceMLP\nμ, σ², bias\npolarization", fill=TEAL, edge=TEAL, color="white", ts=8)

    box(5.4, 1.15, 2.5, 0.85, "honesty split\nscript / prod / rec.", fill="#f3e6d4", edge=GOLD, ts=8)
    box(8.2, 1.15, 2.0, 0.85, "oracle MLP\nceiling 0.32", fill="#f3e6d4", edge=GOLD, ts=8)
    box(10.4, 1.15, 1.75, 0.85, "comps +\natlas", fill="#dbeae8", edge=TEAL, color=INK, ts=8.5)

    arrow(2.4, 5.27, 2.75, 4.55)
    arrow(2.4, 4.17, 2.75, 4.22, color=TEAL, lw=1.8)
    arrow(2.4, 3.07, 2.75, 3.9, color=TEAL, lw=1.8)
    arrow(5.1, 4.22, 5.4, 4.22, color=TEAL, lw=1.8)
    arrow(7.9, 4.22, 8.2, 4.22, color=TEAL, lw=1.8)
    arrow(10.2, 4.22, 10.4, 4.22, color=TEAL, lw=1.8)
    arrow(11.27, 3.55, 11.27, 2.0, color=TEAL, lw=1.6)
    arrow(2.4, 1.57, 5.4, 1.57, color=GOLD, lw=1.4)
    arrow(7.9, 1.57, 8.2, 1.57, color=GOLD, lw=1.4)

    ax.annotate(
        "supervise",
        xy=(11.27, 4.9),
        xytext=(10.4, 6.15),
        fontsize=7.5,
        color=MUTED,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0, ls=(0, (3, 2))),
    )

    ax.text(0.25, 6.55, "live path (teal)  ·  genome ceiling (gold)  ·  collab target (top right)", fontsize=8.5, color=MUTED)
    ax.text(0.25, 0.35, "BGE and $z_i$ stay frozen. Equinox only learns the map from pooled blocks (or oracle tags) into audience space.", fontsize=8, color=MUTED)
    return savefig(fig, path)


def plot_honesty_ladder(rows: list[tuple[str, float, str]], path: Path) -> Path:
    """When each signal exists, and how much audience it recovers."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows], dtype=float)
    eras = [r[2] for r in rows]
    era_color = {
        "script": TEAL,
        "production": GOLD,
        "release": RED,
        "baseline": MUTED,
    }
    y = np.arange(len(labels))
    colors = [era_color.get(e, MUTED) for e in eras]
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y, labels=labels)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("holdout cosine to collaborative $z_i$")
    ax.set_title("What you can know, and when you can know it")
    ax.axvline(0.185, color=MUTED, lw=1, ls="--", alpha=0.7, zorder=1)
    ax.text(0.187, len(labels) - 0.35, " metadata", color=MUTED, fontsize=8, va="top")
    for yi, val in zip(y, vals, strict=True):
        ax.text(val + 0.008, yi, f"{val:.3f}", va="center", fontsize=8.5, color=INK)
    handles = [
        plt.Line2D([0], [0], color=TEAL, lw=6, label="on the page (script time)"),
        plt.Line2D([0], [0], color=GOLD, lw=6, label="in production"),
        plt.Line2D([0], [0], color=RED, lw=6, label="after an audience"),
        plt.Line2D([0], [0], color=MUTED, lw=6, label="weak / failed maps"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.invert_yaxis()
    return savefig(fig, path)


def plot_usefulness_plane(points: list[tuple[str, float, float, str]], path: Path) -> Path:
    """Cosine vs polarization. Genome is a comparison, not a gate."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    palette = {"ceiling": GOLD, "narrative": TEAL, "leak": RED, "fail": MUTED}
    for name, cos, spear, kind in points:
        ax.scatter(cos, spear, s=90, c=palette.get(kind, MUTED), zorder=4, linewidths=0)
        ax.annotate(name, (cos, spear), textcoords="offset points", xytext=(7, 5), fontsize=8, color=INK)
    ax.set_xlabel("holdout cosine to $z_i$")
    ax.set_ylabel("Spearman with observed mean rating")
    ax.set_title("Useful is not the same as matching a realized film")
    ax.set_xlim(-0.05, 0.48)
    ax.set_ylim(-0.25, 1.05)
    ax.scatter([], [], c=TEAL, s=70, label="narrative (this baseline)")
    ax.scatter([], [], c=GOLD, s=70, label="oracle genome slices")
    ax.scatter([], [], c=RED, s=70, label="post-release leak")
    ax.scatter([], [], c=MUTED, s=70, label="collapsed / hash")
    ax.legend(loc="lower right")
    ax.text(0.123, 0.42, "subtitle BGE already\nranks who loves it", fontsize=8, color=TEAL)
    return savefig(fig, path)


def plot_tag_constellation(
    genome: np.ndarray,
    buckets: list[str],
    names: list[str],
    path: Path,
    landmarks: list[str] | None = None,
) -> Path:
    """PCA of tag-by-movie relevances. Color = honesty bucket."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    apply_style()
    mat = np.asarray(genome, dtype=np.float32)
    xy = PCA(n_components=2, random_state=0).fit_transform(mat.T)
    color = {"script": TEAL, "production": GOLD, "reception": RED}
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    for bucket, c in color.items():
        idx = [i for i, b in enumerate(buckets) if b == bucket]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=18, c=c, alpha=0.72, linewidths=0, label=f"{bucket}  n={len(idx)}", zorder=3)
    mark = landmarks or [
        "twist ending", "dystopia", "romantic comedy", "imax", "oscar",
        "sequel", "horror", "masterpiece", "time travel", "pixar",
    ]
    name_to_i = {n.lower(): i for i, n in enumerate(names)}
    for tag in mark:
        i = name_to_i.get(tag.lower())
        if i is None:
            continue
        ax.annotate(tag, xy[i], fontsize=8, color=INK, xytext=(6, 4), textcoords="offset points")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("The genome is three maps sharing one space")
    ax.legend(loc="upper right")
    ax.text(
        0.02, 0.02,
        "Each point is a MovieLens tag. Nearby tags fire on similar films.\nTeal can exist on a spec. Gold needs a production. Red needs an audience.",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom",
    )
    return savefig(fig, path)


def plot_two_step(oracle: float, isab: float, mean_tags: float, narrative: float, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [
        "oracle script-visible tags",
        "BGE subtitle mean-pool\n(parked baseline)",
        "ISAB + position → tags",
        "mean-pool → tags",
    ]
    vals = [oracle, narrative, isab, mean_tags]
    colors = [GOLD, TEAL, MUTED, MUTED]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.55)
    ax.set_yticks(y, labels=labels)
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("holdout cosine")
    ax.set_title("The two-step has the right shape. The tag reader is not there yet.")
    for yi, val in zip(y, vals, strict=True):
        ax.text(val + 0.008, yi, f"{val:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    return savefig(fig, path)


def plot_baseline_atlas(
    xy: np.ndarray,
    genome_xy: np.ndarray,
    narrative_xy: np.ndarray,
    genome_hold: np.ndarray,
    narrative_hold: np.ndarray,
    titles: list[str],
    highlight: list[int],
    path: Path,
) -> Path:
    """Collaborative atlas with genome vs narrative cold-start overlays."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    panels = (
        (axes[0], genome_xy, genome_hold, GOLD, "oracle genome  ·  after the film exists"),
        (axes[1], narrative_xy, narrative_hold, TEAL, "BGE mean-pool  ·  parked baseline"),
    )
    for ax, pred, hold, color, title in panels:
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c=MUTED, alpha=0.22, linewidths=0)
        ax.scatter(
            pred[hold, 0], pred[hold, 1],
            s=14, c=color, alpha=0.75, linewidths=0, zorder=3,
        )
        for idx in highlight:
            ax.annotate(titles[idx].rsplit(" (", 1)[0], xy[idx], fontsize=7, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_facecolor(PAPER)
    fig.suptitle("Same holdout films, two maps", fontsize=13, y=0.98)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path
