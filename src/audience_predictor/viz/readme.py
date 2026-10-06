"""Figures for the repository front page."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import beta as beta_dist

from audience_predictor.viz.style import (
    BLUE,
    GOLD,
    GRID,
    INK,
    MUTED,
    PAPER,
    RED,
    SLATE,
    TEAL,
    apply_style,
)

TEAL_SOFT = "#dbeae8"
GOLD_SOFT = "#f3e6d4"
STONE = "#e8e4dc"
JEV_SERIES = (
    ("turns", "turns", BLUE),
    ("want", "want", TEAL),
    ("specific", "specific", GOLD),
    ("on_the_nose", "on the nose", RED),
    ("dead_air", "dead air", "#a8a29e"),
)
STAMP_COLOR = {"recommend": TEAL, "consider": GOLD, "pass": RED}


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def plot_system(path: Path) -> Path:
    """Two questions, one desk: who is it for, and is it any good."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(13.6, 7.4))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", *, fill=STONE, edge=INK, color=INK, lw=1.1):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
            )
        )
        if sub:
            ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                    fontsize=9.6, fontweight="bold", color=color, zorder=4)
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                    fontsize=7.6, color=color if color != INK else SLATE, zorder=4,
                    linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=9.6, fontweight="bold", color=color, zorder=4)

    def arrow(x1, y1, x2, y2, *, color=MUTED, lw=1.5, ls="-", rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>", mutation_scale=12, lw=lw, color=color,
                linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2,
            )
        )

    def lane(y, h, label, color):
        ax.add_patch(
            FancyBboxPatch(
                (1.95, y), 9.35, h,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor="none", edgecolor=color, linewidth=0.9,
                linestyle=(0, (4, 3)), alpha=0.7, zorder=1,
            )
        )
        ax.text(2.1, y + h - 0.2, label, fontsize=8.4, color=color,
                fontweight="bold", va="top", zorder=4)

    ax.text(0.15, 7.12, "One script, two questions", fontsize=16, fontweight="bold", color=INK)
    ax.text(0.15, 6.78,
            "Audience fit and craft are separate readings. A tight script can be for nobody;\n"
            "a broad crowd can love a sloppy one.",
            fontsize=9.2, color=MUTED, va="top", linespacing=1.35)

    box(0.15, 3.0, 1.55, 1.5, "PAGES", ".fountain · .txt\n.srt · prose",
        fill=INK, edge=INK, color="white")

    lane(3.75, 2.3, "WHO IS IT FOR  ·  audience map", TEAL)
    box(2.25, 4.2, 1.85, 1.2, "Typed blocks", "screenplay ≠ subtitle\n≤ 32 windows")
    box(4.4, 4.2, 1.85, 1.2, "BGE-small", "frozen · 384-d\nONNX", fill=TEAL_SOFT, edge=TEAL)
    box(6.55, 4.2, 1.65, 1.2, "Mean pool", "Deep Set ρ", fill=TEAL_SOFT, edge=TEAL)
    box(8.5, 4.85, 2.6, 0.95, "AudienceMLP", "μ ∈ ℝ⁶⁴ · σ² · P(love) · P(hate)",
        fill=TEAL, edge=TEAL, color="white")
    box(8.5, 3.9, 2.6, 0.82, "ScoreBeta", "Beta posterior · critics & audience",
        fill=TEAL_SOFT, edge=TEAL)

    lane(0.55, 2.75, "IS IT GOOD  ·  scene judge", GOLD)
    box(2.25, 1.0, 1.85, 1.2, "Scene parser", "sluglines · cast\naction · dialogue")
    box(4.4, 1.0, 1.85, 1.2, "Typed questions", "5 per scene\n+ genre · genericness",
        fill=GOLD_SOFT, edge=GOLD)
    box(6.55, 1.0, 2.0, 1.2, "OpenJev NLI", "Qwen3.5-4B cross-encoder\nP(entail | scene, claim)",
        fill=GOLD_SOFT, edge=GOLD)
    box(8.85, 1.0, 2.25, 1.2, "Craft + stamp", "scene timeline\npass · consider · recommend",
        fill=GOLD, edge=GOLD, color="white")

    box(11.65, 2.55, 1.8, 2.4, "COVERAGE\nDESK", "FastAPI\nstamp · timeline\ncomps · RT",
        fill=INK, edge=INK, color="white")

    box(8.5, 6.2, 2.6, 0.62, "Collaborative z", "24.6M ratings · 13,176 films", fill="#efece6", lw=0.9)

    arrow(1.7, 4.0, 2.25, 4.8, color=TEAL, lw=1.8, rad=-0.15)
    arrow(1.7, 3.5, 2.25, 1.65, color=GOLD, lw=1.8, rad=0.15)
    for x1, x2 in ((4.1, 4.4), (6.25, 6.55)):
        arrow(x1, 4.8, x2, 4.8, color=TEAL, lw=1.8)
        arrow(x1, 1.6, x2, 1.6, color=GOLD, lw=1.8)
    arrow(8.2, 4.95, 8.5, 5.3, color=TEAL, lw=1.8)
    arrow(8.2, 4.7, 8.5, 4.32, color=TEAL, lw=1.8)
    arrow(8.55, 1.6, 8.85, 1.6, color=GOLD, lw=1.8)
    arrow(9.8, 6.2, 9.8, 5.8, color=MUTED, lw=1.1, ls=(0, (3, 2)))
    ax.text(9.95, 6.0, "frozen target", fontsize=7, color=MUTED, va="center")
    arrow(11.1, 4.9, 11.65, 4.3, color=TEAL, lw=1.8, rad=-0.1)
    arrow(11.1, 1.6, 11.65, 3.15, color=GOLD, lw=1.8, rad=0.12)

    ax.text(2.25, 0.18,
            "Frozen: rating-space z, BGE, OpenJev. Trained here: small Equinox heads only. "
            "Judge falls back to a lexical stand-in when 4B weights are not cached.",
            fontsize=7.8, color=MUTED)
    return _save(fig, path)


def plot_scene_timeline(cards: list[tuple[str, dict]], path: Path) -> Path:
    """The judge's softmax over five craft heads, scene by scene."""
    import matplotlib.pyplot as plt

    apply_style()
    n = len(cards)
    fig, axes = plt.subplots(n, 1, figsize=(11.6, 2.4 * n + 1.1), sharex=True)
    axes = np.atleast_1d(axes)
    judge = "heuristic"
    for ax, (title, card) in zip(axes, cards, strict=True):
        judge = card["judge"]["kind"]
        scenes = card["scenes"]
        total = max(int(card["stats"]["n_scenes"]) - 1, 1)
        x = np.array([row["index"] / total for row in scenes], dtype=float)
        stack = np.array([[row["softmax"][k] for k, _, _ in JEV_SERIES] for row in scenes]).T
        colors = [c for _, _, c in JEV_SERIES]
        ax.stackplot(x, stack, colors=colors, alpha=0.26, linewidth=0, zorder=2)
        acc = np.zeros_like(x)
        for layer in stack[:-1]:
            acc = acc + layer
            ax.plot(x, acc, color=PAPER, lw=0.8, zorder=3)
        craft = np.array([row["craft"] for row in scenes], dtype=float)
        ax.plot(x, craft, color=INK, lw=2.0, ls=(0, (5, 2.5)), zorder=5)
        ax.scatter(x, craft, s=22, color=INK, zorder=6, linewidths=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.01, 1.01)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        stamp = card["stamp"]
        ax.text(0.0, 1.06, title, transform=ax.transAxes, fontsize=11.5,
                fontweight="bold", color=INK, va="bottom")
        ax.text(0.86, 1.07,
                f"{card['genre']}  ·  {card['stats']['n_scenes']} scenes  ·  craft {card['craft']:.2f}",
                transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom", ha="right")
        ax.text(1.0, 1.05, stamp.upper(), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, fontweight="bold", color=STAMP_COLOR.get(stamp, INK),
                bbox=dict(boxstyle="square,pad=0.28", fc=PAPER,
                          ec=STAMP_COLOR.get(stamp, INK), lw=1.6))
    axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0], labels=["FADE IN", "", "midpoint", "", "FADE OUT"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for _, _, c in JEV_SERIES]
    labels = [label for _, label, _ in JEV_SERIES]
    handles.append(plt.Line2D([0], [0], color=INK, lw=2, ls=(0, (5, 2.5))))
    labels.append("scene craft")
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Scene judge, page by page", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.text(0.01, 0.955,
             f"Bands: softmax over five yes/no craft questions per scene. Line: weighted scene craft. "
             f"Judge backend here: {judge}.",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=2.6)
    return _save(fig, path)


def plot_posterior_story(
    examples: list[tuple[str, float, float, float, int, int]],
    path: Path,
) -> Path:
    """Why a Beta posterior: thin counts barely move it; text gives a wide prior."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    grid = np.linspace(0.001, 0.999, 500)

    cases = (
        ("1 of 1 fresh  ·  “100%”", 1, 1, GOLD),
        ("13 of 28 fresh  ·  46%", 13, 28, SLATE),
        ("73 of 73 fresh  ·  100%", 73, 73, TEAL),
    )
    for label, k, n, color in cases:
        dens = beta_dist.pdf(grid, 1 + k, 1 + n - k)
        left.fill_between(grid * 100, dens, color=color, alpha=0.18, lw=0)
        left.plot(grid * 100, dens, color=color, lw=1.8, label=label)
    left.set_xlim(0, 100)
    left.set_ylim(0, 12)
    left.set_yticks([])
    left.set_xlabel("Tomatometer")
    left.set_title("Same headline, different evidence", loc="left", fontsize=12, fontweight="bold")
    left.legend(loc="upper left", fontsize=8.6)
    left.spines["left"].set_visible(False)

    labels = []
    for i, (title, a, b, observed, k, n) in enumerate(examples):
        mean = a / (a + b)
        lo90, lo50, hi50, hi90 = beta_dist.ppf([0.05, 0.25, 0.75, 0.95], a, b)
        right.plot([lo90 * 100, hi90 * 100], [i, i], color=TEAL, lw=2.0, alpha=0.35,
                   solid_capstyle="round", zorder=2)
        right.plot([lo50 * 100, hi50 * 100], [i, i], color=TEAL, lw=6.0, alpha=0.75,
                   solid_capstyle="butt", zorder=3)
        right.scatter([mean * 100], [i], s=36, color=PAPER, edgecolors=TEAL, linewidths=1.6, zorder=4)
        right.scatter([observed * 100], [i], marker="D", s=30, color=INK, zorder=5)
        labels.append(f"{title}  ({k}/{n})")
    right.set_yticks(range(len(examples)), labels=labels, fontsize=8.6)
    right.set_xlim(0, 100)
    right.set_ylim(-0.7, len(examples) - 0.3)
    right.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.set_xlabel("critics score  ·  thick = 50%, thin = 90% interval  ·  ◆ = observed")
    right.set_title("Story text → a posterior, not a point", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def pick_posterior_examples(
    titles: list[str],
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    holdout: np.ndarray,
    *,
    col: int = 0,
    min_reviews: int = 120,
    n_pick: int = 9,
) -> list[tuple[str, float, float, float, int, int]]:
    """Holdout films with plenty of reviews, spread across predicted score."""
    idx = np.asarray(holdout, dtype=np.int64)
    idx = idx[n[idx, col] >= min_reviews]
    if idx.size == 0:
        return []
    mean = alpha[idx, col] / (alpha[idx, col] + beta[idx, col])
    order = idx[np.argsort(mean)]
    qs = np.linspace(0.02, 0.98, n_pick)
    picks = list(dict.fromkeys(order[int(q * (order.size - 1))] for q in qs))
    rows = []
    for i in picks:
        rows.append(
            (
                _short_title(titles[i]),
                float(alpha[i, col]),
                float(beta[i, col]),
                float(target[i, col]),
                int(k[i, col]),
                int(n[i, col]),
            )
        )
    return rows


def _short_title(title: str, limit: int = 30) -> str:
    text = title.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
