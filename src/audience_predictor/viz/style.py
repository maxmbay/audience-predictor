"""Shared matplotlib style — visual cousin of catalog-value figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

from audience_predictor.paths import project_root

INK = "#1c1917"
PAPER = "#f7f4ee"
MUTED = "#78716c"
GRID = "#e7e5e4"
BLUE = "#1d4e89"
RED = "#b42318"
TEAL = "#0f766e"
GOLD = "#b45309"
SLATE = "#57534e"
VIOLET = "#5b21b6"


def apply_style() -> None:
    rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titlecolor": INK,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def figures_dir() -> Path:
    path = project_root() / "reports" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
