"""Assemble the three-register report from metrics, coverage, and figures."""

from __future__ import annotations

import json
from pathlib import Path

from audience_predictor.config import Config
from audience_predictor.paths import output_dir, reports_dir
from audience_predictor.viz.style import figures_dir


def _metrics_block() -> str:
    path = output_dir() / "ablation.json"
    if not path.exists():
        parts = []
        for name in ("metadata", "genome", "tags", "hierarchical"):
            p = output_dir() / name / "metrics.json"
            if p.exists():
                parts.append((name, json.loads(p.read_text())))
        data = {k: v for k, v in parts}
    else:
        data = json.loads(path.read_text())
    if not data:
        return "_No trained modalities yet. Run `audience-predictor train --modality genome`._"
    lines = [
        "| Modality | Train cosine | Holdout cosine | Holdout RMSE | Holdout NLL |",
        "|----------|-------------:|---------------:|-------------:|------------:|",
    ]
    for name, block in data.items():
        tr = block["train"]
        ho = block["holdout"]
        lines.append(
            f"| {name} | {tr['cosine_mean']:.3f} | {ho['cosine_mean']:.3f} | {ho['rmse']:.3f} | {ho.get('nll', float('nan')):.3f} |"
        )
    return "\n".join(lines)


def _fig(name: str) -> str:
    path = figures_dir() / name
    rel = Path("figures") / name
    if path.exists():
        return f"![{name}]({rel.as_posix()})"
    return f"_{name} not generated yet._"


def write_report(config: Config) -> Path:
    dest = reports_dir() / "narrative-audience.md"
    dest.parent.mkdir(parents=True, exist_all=True) if False else dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_render(config))
    return dest


def _render(config: Config) -> str:
    return f"""# From narrative to audience

A JAX/Equinox system that predicts **who a film is for** from story-shaped
information, before anyone has rated it.

The target is not box office. It is a point — with uncertainty — in the
64-dimensional audience space already learned from MovieLens ratings by the
sibling catalog-value project.

---

## Low — what this is, and why it is interesting

Imagine you are handed a script, or even just the way people *talk about* a
movie, and you have to guess **which kinds of viewers will love it**. Not how
much money it will make. Not whether critics will smile. *Who it is for.*

That is a different question from "is this good?" Two films can have the same
average rating and completely different crowds: a consensus crowd-pleaser, a
niche film that a small group adores, a polarizer that splits the room.

This project builds a map of movie taste from 25 million ratings, then asks a
new encoder, written in JAX, to land an unseen film on that map using only
content-like signals:

- genre and year (what a one-sheet already tells you)
- MovieLens "genome" tags (a rich but non-narrative content baseline)
- the words viewers themselves tag the film with (a silver stand-in for
  dialogue and cultural reception)
- a **set transformer** that treats those tags (and, later, screenplay scenes)
  as an unordered collection of story blocks

Until a gold corpus of several thousand genuine screenplays is licensed and
parsed, we do **not** claim "the screenplay predicts the audience." We claim
something more careful and already useful:

> Story-shaped information, encoded as a set of blocks with a probabilistic
> Equinox model, recovers a meaningful slice of the audience a film eventually
> finds — and it tells you when it is unsure.

{_fig("atlas.png")}

The beige cloud is the collaborative audience map. Teal points are films the
model had to place **without using their ratings**. Bigger teal marks are more
uncertain guesses. That uncertainty is a feature: some stories really are
harder to place than others.

{_fig("comps.png")}

For a handful of famous films, the left column is "who actually showed up"
(nearest neighbors in rating space). The right column is "who the model
thought would show up" from content alone. When those lists rhyme, the
encoder has learned taste, not trivia.

---

## Medium — how the machine is put together

### Two repositories, one latent space

| Piece | Where it lives |
|-------|----------------|
| Who likes what (ratings only) | `catalog-value` Phase A |
| `z_i` collaborative fingerprint, 13,176 × 64 | `outputs/phase_a/titles.npz` |
| Genome MLP (PyTorch, the bar to beat) | `catalog-value` Phase B |
| Narrative / content encoders (Equinox) | **this repo** |
| Catalog value (MCV / PACV) | `catalog-value` Phases C–D, downstream |

The Equinox models emit the same 64-d vector plus a scalar variance, so a
predicted script can be dropped into the existing catalog-value posterior
without retraining the audience encoder.

### Modalities (ablations)

```text
metadata     genre + year
genome       1,128 tag relevances + genre + year     ← bar to beat
tags         hashed MovieLens user tags (bag)
hierarchical set transformer over the same tags as blocks
screenplay   parser ready; corpus not in git
```

Movie-level split: 20% of well-rated titles are held out entirely. The text
model never sees their collaborative embedding during training.

### Architecture, in one picture

```text
scene / tag / sentence
        │
   blake2b n-gram hash (256-d, frozen)
        │
   linear → GELU  (dialogue stream and action stream)
        │
   Induced Set Attention (learned inducing points)
        │
   Pooling by Multihead Attention (one movie query)
        │
   μ ∈ R⁶⁴ ,  log σ² ,  bias ,  polarization aux
        │
   residual:  μ_full = μ_dialogue + Δ(action, structure)
```

Training is Optax AdamW on a sum of embedding MSE, Gaussian NLL (so the
variance means something), bias MSE, and an auxiliary loss on
\\(E[r], \\sqrt{{\\mathrm{{Var}}(r)}}, P(r\\ge 4), P(r\\le 2)\\).

### Results so far

{_metrics_block()}

{_fig("ablation.png")}

Read the teal bars: that is cold-start cosine. Genome should beat metadata.
Tags test whether *language* about a film carries audience signal beyond
structured tags. The set transformer tests whether treating language as a
**set of blocks** (the same inductive bias we want for scenes) helps.

{_fig("reconstruction.png")}

Well-observed films are easier; the interesting points are the teal
cold-start titles. If cosine stays high there, content is doing real work.

{_fig("polarization.png")}

Love and hate are not the same as the mean. A film can be a 3.5 that everyone
shrugs at, or a 3.5 that half the room worships.

{_fig("uncertainty.png")}

Hypothesis H5: bigger predicted σ should line up with larger embedding error.
If that cloud slopes up, the model is calibrated enough to be useful in
development ("this script is genuinely ambiguous").

{_fig("scaling.png")}

If the curve is still climbing at a few thousand titles, collecting more gold
screenplays is worth the cost. If it has flattened, architecture and
pretraining matter more than corpus size.

{_fig("coverage_funnel.png")}

The binding constraint is the last bar. Everything else — IMDb ids, rating
depth, silver tags — is already in place.

### How to run

```bash
uv sync --dev
uv run pytest
uv run python -m audience_predictor coverage
uv run python -m audience_predictor train --modality genome
uv run python -m audience_predictor train --modality metadata
uv run python -m audience_predictor train --modality tags
uv run python -m audience_predictor train --modality hierarchical
uv run python -m audience_predictor evaluate
uv run python -m audience_predictor scaling
uv run python -m audience_predictor figures
uv run python -m audience_predictor report
```

Checkpoints live in `outputs/<modality>/`. Figures live in `reports/figures/`.

---

## High — model, protocol, and claims

### Target

Let \(z_i^{{\\mathrm{{collab}}}} \\in \\mathbb{{R}}^{{64}}\) be the catalog-value
item embedding. Users are mixtures \(Z_u = \\{{(\\pi_{{uk}}, z_{{uk}}))\\}}_{{k=1}}^{{8}}\).
We do not retrain that space. We learn

\\[
f_\\theta(X_i) \\;\\mapsto\\; \\mathcal{{N}}\\big(\\mu_\\theta(X_i),\\, \\sigma_\\theta^2(X_i)\\, I\\big)
\\]

and score \(\\cos(\\mu_\\theta(X_i), z_i^{{\\mathrm{{collab}}}})\) on titles whose
ratings were **never** used to supervise \(f_\\theta\).

A Gaussian shrinkage posterior, identical in contract to catalog-value Phase B,

\\[
w_i = \\frac{{n_i}}{{n_i + n_0}},\\qquad
\\mu_i = w_i z_i^{{\\mathrm{{collab}}}} + (1-w_i)\\mu_\\theta(X_i)
\\]

keeps MCV plug-and-play. For true cold start, \(n_i = 0\) and the posterior
collapses to the narrative prior.

### Set encoder

Blocks \(x_{{i1}},\\ldots,x_{{iT}}\) are hashed into \(\\mathbb{{R}}^{{d_h}}\).
An ISAB layer with \(m\) inducing points compresses the set; PMA with a single
learned query yields a movie token. Dialogue and action are separate
projections; structure (scene counts, character entropy, graph density) is
added into the residual stream:

\\[
z_i = z_i^{{\\mathrm{{dialogue}}}} + \\Delta_i(\\text{{action}},\\, \\text{{structure}}).
\\]

This is the flagship scientific object. On silver data, "dialogue" is
folksonomy tags and "action" is empty, so \(\\Delta \\approx 0\) and the
identity is testable. On gold screenplays, \(\\Delta\) is the incremental
audience signal in description, setting, and pacing — hypothesis **H2**.

Loss (per train movie):

\\[
\\mathcal{{L}} = \\|\\mu - z\\|_2^2
  + \\lambda_b (b - b_i)^2
  + \\lambda_n \\, \\mathrm{{NLL}}(\\mu,\\sigma^2; z)
  + \\lambda_a \\|\\mathrm{{aux}} - y^{{\\mathrm{{pol}}}}\\|_2^2.
\\]

CRPS has a closed form for Gaussians and is reported on holdout for
calibration, not just accuracy.

### Protocol details

- Split unit: **movie**, not rating. `tests/test_splits.py` asserts disjoint
  index sets.
- Eligible train/holdout titles have ≥ {config.train.min_train_ratings} ratings.
- Holdout fraction: {config.train.holdout_frac}.
- Seed: {config.seed}. Embedding dim: {config.representation.embedding_dim}.
- Hashing is blake2b modulo {config.data.hash_dim}, character 3–5 grams,
  \(\\ell_2\)-normalized. Deterministic across processes (unlike `hash()`).
- JAX `vmap` + `eqx.filter_jit` over the full eligible train set. 13k titles
  is a small problem once features are precomputed; we do not re-tokenize.

### Hypotheses

| ID | Claim | Status in this milestone |
|----|-------|--------------------------|
| H1 | Content predicts audience beyond metadata | Testable: genome vs metadata holdout cosine |
| H2 | Full screenplay > dialogue | Structurally implemented; empirically pending gold scripts |
| H3 | Silver pretrain helps gold | Pending gold scripts |
| H4 | Multi-interest targets beat a single z | Target space is already multi-interest; not re-ablated here |
| H5 | σ tracks error | `uncertainty.png` / Spearman(error, σ) |
| H6 | Fingerprints feed catalog value | Interface-compatible; MCV transfer lives in catalog-value |

### Guardrails

- MovieLens users are not representative of moviegoers.
- Missing ratings are not dislikes.
- User tags leak *reception*, not pre-production text. They are an upper bound
  on "language about a film," not a substitute for a shooting script.
- A screenplay does not uniquely determine the finished film.
- This repo does not ship copyrighted screenplay text. The parser is tested on
  original fixtures.

### What "done" looks like for the gold-script era

Keep the Equinox encoder. Replace hashed tags with parsed scenes. Precompute
block embeddings once. Fine-tune. The residual head is already the experiment.
The coverage report is already the go/no-go dashboard
(`N_{{screenplay \\cap MovieLens}}`).

---

*Config: `{config.source.name}`. Embedding dim {config.representation.embedding_dim},
{config.representation.n_inducing} inducing points, {config.representation.n_heads} heads.*
"""
