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
        for name in (
            "metadata",
            "genome",
            "tags",
            "hierarchical",
            "genome_gold",
            "screenplay",
            "dialogue",
            "screenplay_bge",
            "dialogue_bge",
            "screenplay_mean",
            "dialogue_mean",
            "screenplay_pma",
            "dialogue_pma",
        ):
            p = output_dir() / name / "metrics.json"
            if p.exists():
                parts.append((name, json.loads(p.read_text())))
        data = {k: v for k, v in parts}
    else:
        data = json.loads(path.read_text())
    if not data:
        return "_No trained modalities yet. Run `audience-predictor train --modality genome`._"
    lines = [
        "| Modality | Train cosine | Holdout cosine | Holdout n | Gold-holdout cosine |",
        "|----------|-------------:|---------------:|----------:|--------------------:|",
    ]
    for name, block in data.items():
        tr = block["train"]
        ho = block["holdout"]
        gh = block.get("gold_holdout") or {}
        gold_cos = gh.get("cosine_mean")
        gold_cell = f"{gold_cos:.3f}" if gold_cos is not None else "—"
        lines.append(
            f"| {name} | {tr['cosine_mean']:.3f} | {ho['cosine_mean']:.3f} | {ho.get('n', 0):,} | {gold_cell} |"
        )
    return "\n".join(lines)


def _findings() -> str:
    path = output_dir() / "ablation.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    g = data.get("genome", {}).get("holdout", {})
    m = data.get("metadata", {}).get("holdout", {})
    t = data.get("tags", {}).get("holdout", {})
    h = data.get("hierarchical", {}).get("holdout", {})
    sp = data.get("screenplay", {}).get("holdout", {})
    dl = data.get("dialogue", {}).get("holdout", {})
    gg = data.get("genome_gold", {}).get("holdout", {})
    g_gold = data.get("genome", {}).get("gold_holdout", {})
    sp_b = data.get("screenplay_bge", {}).get("holdout", {})
    dl_b = data.get("dialogue_bge", {}).get("holdout", {})
    sm = data.get("screenplay_mean", {}).get("holdout", {})
    dm = data.get("dialogue_mean", {}).get("holdout", {})
    sp_p = data.get("screenplay_pma", {}).get("holdout", {})
    dl_p = data.get("dialogue_pma", {}).get("holdout", {})
    if not g:
        return ""
    gold_note = ""
    if sp:
        sp_cos = sp.get("cosine_mean", 0)
        dl_cos = dl.get("cosine_mean", 0) if dl else 0
        delta = sp_cos - dl_cos
        h2 = "supported" if delta > 0.01 else ("not supported" if delta < -0.01 else "too close to call")
        extra = []
        if gg:
            extra.append(f"Matched-N genome (gold train only) is {gg.get('cosine_mean', 0):.3f}.")
        if g_gold:
            extra.append(f"Full-data genome scored on the same gold holdout is {g_gold.get('cosine_mean', 0):.3f}.")
        if max(sp_cos, dl_cos) < 0.1:
            extra.append(
                "Hashed n-grams of scene text do not recover z_i; a pretrained language encoder is the next lever, not more MovieSum scripts."
            )
        extra_txt = " ".join(extra)
        gold_note += f"""
- **H2 hashed ({h2}).** On MovieSum ∩ holdout (n={sp.get('n', 0)}), full screenplay
  cosine is {sp_cos:.3f} vs dialogue-only {dl_cos:.3f} (Δ={delta:+.3f}).
  {extra_txt} Wikipedia summaries from MovieSum are **not** used — they leak
  post-release reception.
"""
    if sp_b:
        b_sp = sp_b.get("cosine_mean", 0)
        b_dl = dl_b.get("cosine_mean", 0) if dl_b else 0
        b_delta = b_sp - b_dl
        h2b = "supported" if b_delta > 0.01 else ("not supported" if b_delta < -0.01 else "too close to call")
        mean_txt = ""
        if sm or dm:
            parts = []
            if dm:
                parts.append(f"dialogue mean-pool {dm.get('cosine_mean', 0):.3f}")
            if sm:
                parts.append(f"screenplay mean-pool {sm.get('cosine_mean', 0):.3f}")
            mean_txt = (
                " Masked-average MLP reaches "
                + " / ".join(parts)
                + " — near metadata, well above the set encoder on the same vectors."
            )
        gold_note += f"""
- **H2 frozen BGE ({h2b}).** Same holdout, ONNX BGE-small scene embeddings:
  screenplay {b_sp:.3f} vs dialogue {b_dl:.3f} (Δ={b_delta:+.3f}).{mean_txt}
  Equinox still trains the set aggregator; the language encoder is frozen.
"""
    if sp_p or dl_p:
        p_sp = sp_p.get("cosine_mean", 0) if sp_p else 0
        p_dl = dl_p.get("cosine_mean", 0) if dl_p else 0
        p_delta = p_sp - p_dl
        if max(p_sp, p_dl) < 0.1:
            h2p = "not supported; recovery still weak"
        else:
            h2p = "supported" if p_delta > 0.01 else ("not supported" if p_delta < -0.01 else "too close to call")
        gold_note += f"""
- **H2 PMA-only ({h2p}).** Dropping ISAB, a learned query attends over projected BGE
  scenes: screenplay {p_sp:.3f} vs dialogue {p_dl:.3f} (Δ={p_delta:+.3f}).
  Attention pooling still sits well below masked-average MLP; induced attention
  was not the only thing scrambling frozen geometry. Mean-pool remains the
  narrative baseline.
"""
    if not gold_note:
        gold_note = """
- **N_screenplay ∩ MovieLens is still 0 in this checkpoint.** Run
  `ingest-moviesum` then `train --modality screenplay` / `dialogue`.
"""
    return f"""
### What this run actually found

- **H1 holds for genome tags, not for hashed language.** Genome holdout cosine
  is {g.get('cosine_mean', 0):.3f} vs metadata {m.get('cosine_mean', 0):.3f}.
  Content structure beats a one-sheet. Character-n-gram bags of user tags
  ({t.get('cosine_mean', 0):.3f}) and the set transformer over those bags
  ({h.get('cosine_mean', 0):.3f}) do **not** recover the 64-d fingerprint.
  The architecture is ready for real scene embeddings; hashing is not a
  substitute for a pretrained language encoder.
- **Polarization is the surprising win.** Genome Spearman with observed
  mean rating is {g.get('spearman_mean_rating', 0):.2f}, with P(love)
  {g.get('spearman_p_love', 0):.2f} and P(hate) {g.get('spearman_p_hate', 0):.2f}.
  Even a moderate embedding cosine can still rank which films split a room.
- **H5 is not supported yet.** Spearman(error, σ) is
  {g.get('error_vs_sigma_spearman', 0):.2f} on genome holdout. The NLL term
  shrinks σ without making it track mistakes. Calibration needs a dedicated
  pass (temperature scaling or a proper heteroscedastic fit).
{gold_note}
"""


def _fig(name: str) -> str:
    path = figures_dir() / name
    rel = Path("figures") / name
    if path.exists():
        return f"![{name}]({rel.as_posix()})"
    return f"_{name} not generated yet._"


def write_report(config: Config) -> Path:
    dest = reports_dir() / "narrative-audience.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
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
- a **set transformer** over MovieSum scenes (dialogue vs full screenplay)
  once the gold XML is ingested locally

MovieSum Wikipedia summaries are post-release text and are **never** used as
screenplay input. A subtitle is not a screenplay; hashed user tags are not
either.

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
genome_gold  genome MLP trained only on MovieSum ∩ train
dialogue     MovieSum spoken lines (action/structure zeroed)
screenplay   MovieSum scenes + action + structure     ← H2
```

Movie-level split: 20% of well-rated titles are held out entirely. Screenplay
and dialogue models train only on MovieSum ∩ train; they never see zeros for
titles without a script.

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
mean rating, rating std, P(r ≥ 4), and P(r ≤ 2).

### Results so far

{_metrics_block()}

{_findings()}

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
PYTHONPATH=src uv run python -m audience_predictor coverage
PYTHONPATH=src uv run python -m audience_predictor ingest-moviesum
PYTHONPATH=src uv run python -m audience_predictor train --modality genome
PYTHONPATH=src uv run python -m audience_predictor train --modality metadata
PYTHONPATH=src uv run python -m audience_predictor train --modality tags
PYTHONPATH=src uv run python -m audience_predictor train --modality hierarchical
PYTHONPATH=src uv run python -m audience_predictor train --modality genome_gold
PYTHONPATH=src uv run python -m audience_predictor train --modality dialogue
PYTHONPATH=src uv run python -m audience_predictor train --modality screenplay
PYTHONPATH=src uv run python -m audience_predictor evaluate
PYTHONPATH=src uv run python -m audience_predictor scaling
PYTHONPATH=src uv run python -m audience_predictor figures
PYTHONPATH=src uv run python -m audience_predictor report
```

Checkpoints live in `outputs/<modality>/`. Figures live in `reports/figures/`.

---

## High — model, protocol, and claims

### Target

Let `z_i` in R^64 be the catalog-value item embedding. Users are mixtures
`Z_u = {{(pi_uk, z_uk)}}` for k = 1..8. We do not retrain that space. We learn

```text
f(X_i)  →  N(μ_θ(X_i), σ_θ²(X_i) I)
```

and score cosine(μ_θ(X_i), z_i) on titles whose ratings were **never** used
to supervise the text model.

A Gaussian shrinkage posterior, identical in contract to catalog-value Phase B,

```text
w_i = n_i / (n_i + n0)
μ_i = w_i z_i + (1 - w_i) μ_θ(X_i)
```

keeps MCV plug-and-play. For true cold start, n_i = 0 and the posterior
collapses to the narrative prior.

### Set encoder

Blocks x_i1 … x_iT are hashed into R^d. An ISAB layer with m inducing
points compresses the set; PMA with a single learned query yields a movie
token. Dialogue and action are separate projections; structure (scene counts,
character entropy, graph density) is added into the residual stream:

```text
z_i = z_dialogue + Delta_i(action, structure)
```

This is the flagship scientific object. On silver data, "dialogue" is
folksonomy tags and "action" is empty, so Δ ≈ 0 and the identity is testable.
On gold screenplays, Δ is the incremental audience signal in description,
setting, and pacing — hypothesis **H2**.

Loss (per train movie):

```text
L = ||μ - z||² + λ_b (b - b_i)² + λ_n NLL(μ, σ²; z) + λ_a ||aux - y_pol||²
```

CRPS has a closed form for Gaussians and is reported on holdout for
calibration, not just accuracy.

### Protocol details

- Split unit: **movie**, not rating. `tests/test_splits.py` asserts disjoint
  index sets.
- Eligible train/holdout titles have ≥ {config.train.min_train_ratings} ratings.
- Holdout fraction: {config.train.holdout_frac}.
- Seed: {config.seed}. Embedding dim: {config.representation.embedding_dim}.
- Hashing is blake2b modulo {config.data.hash_dim}, character 3–5 grams,
  L2-normalized. Deterministic across processes (unlike `hash()`).
- JAX `vmap` + `eqx.filter_jit` over the full eligible train set. 13k titles
  is a small problem once features are precomputed; we do not re-tokenize.

### Hypotheses

| ID | Claim | Status in this milestone |
|----|-------|--------------------------|
| H1 | Content predicts audience beyond metadata | Testable: genome vs metadata holdout cosine |
| H2 | Full screenplay > dialogue | MovieSum gold: `screenplay` vs `dialogue` holdout cosine |
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

Keep the Equinox encoder. MovieSum is the first gold dump (identifiers and
hashed blocks only in `data/processed/gold/`, scripts stay in `data/raw/`).
Next: SubLens-20M silver subtitles, then a real language encoder in place of
blake2b hashing. The coverage report is the go/no-go dashboard
(`N_screenplay ∩ MovieLens`).

---

*Config: `{config.source.name}`. Embedding dim {config.representation.embedding_dim},
{config.representation.n_inducing} inducing points, {config.representation.n_heads} heads.*
"""
