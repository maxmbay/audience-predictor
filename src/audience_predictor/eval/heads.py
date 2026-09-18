"""Which parked head to run for a document type."""

from __future__ import annotations

from pathlib import Path

from audience_predictor.paths import output_dir

SCREENPLAY_TYPES = frozenset({"screenplay", "shooting_script"})
SCREENPLAY_HEADS = ("dialogue_mean", "screenplay_mean")
DEFAULT_HEAD = "subtitle_mean"


def available_heads(root: Path | None = None) -> frozenset[str]:
    base = root or output_dir()
    if not base.exists():
        return frozenset()
    return frozenset(p.name for p in base.iterdir() if (p / "model.eqx").exists())


def predict_head_for_document(
    kind: str,
    available: frozenset[str] | None = None,
) -> str:
    """Screenplays use the gold dialogue/screenplay mean-pool head when trained.

    Subtitles, transcripts, synopses, and unknown text stay on the parked
    subtitle_mean baseline. Falls back to that baseline if a screenplay head
    is missing.
    """
    have = available if available is not None else available_heads()
    if kind in SCREENPLAY_TYPES:
        for name in SCREENPLAY_HEADS:
            if name in have:
                return name
    if DEFAULT_HEAD in have:
        return DEFAULT_HEAD
    if have:
        return sorted(have)[0]
    return DEFAULT_HEAD


def format_predict_card(result: dict) -> str:
    """Human readout for `predict`. JSON is the `--json` path."""
    pol = result["polarization"]
    sigma = result.get("sigma")
    sigma_txt = f"  ·  σ = {sigma:.2f}" if isinstance(sigma, (int, float)) else ""
    lines = [
        f"{result['document_type']}  ·  head {result['path']}  ·  {result['n_blocks']} blocks",
        f"this is a guess{sigma_txt}",
        "σ is not calibrated; comps are a neighborhood, not a fact.",
        "",
        f"mean rating  {pol['mean_rating']:.2f}",
        f"P(love)      {pol['p_love']:.2f}",
        f"P(hate)      {pol['p_hate']:.2f}",
        "",
        "comps among well-rated catalog titles",
    ]
    for i, row in enumerate(result.get("comps") or [], 1):
        title = str(row["title"])
        lines.append(f"  {i:2d}.  {title:<42s}  {float(row['cosine']):.3f}")
    return "\n".join(lines)
