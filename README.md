# Audience Predictor

Predict **who a film will appeal to** from its narrative content, before any audience
behavior is observed.

The target is not box office and not an IMDb star rating. It is the film's
**Audience Fingerprint**: a point (and uncertainty) in the 64-dimensional
multi-interest preference space already learned from MovieLens ratings in the
sibling [`catalog-value`](../catalog-value) project.

```text
Narrative blocks  →  Equinox set encoder  →  p(z_i | text)
                                              ↓
                                    catalog-value MCV / PACV
```

## Stack

JAX, Equinox, Optax, Polars. PyTorch stays in `catalog-value`; this repo does not depend on it.

## Quick start

```bash
uv sync --dev
cp .env.example .env   # set CATALOG_VALUE_ROOT if the sibling path is nonstandard
uv run pytest
uv run python -m audience_predictor coverage
uv run python -m audience_predictor train --modality genome
uv run python -m audience_predictor evaluate
uv run python -m audience_predictor figures
```

## Documentation

- [PR roadmap](docs/ROADMAP.md) — six stacked PRs, testing/training/narrative plan
- [Research report](reports/narrative-audience.md) — three registers, from lay to technical (PR 6)

## Scientific guardrails

A subtitle is not a screenplay. MovieLens users are not the general population.
Missing ratings are not dislikes. A screenplay does not uniquely determine the
finished film. This repository does not redistribute copyrighted script text.
