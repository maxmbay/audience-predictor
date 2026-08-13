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

JAX, Equinox, Optax, Polars. PyTorch stays in `catalog-value`. Frozen scene
embeddings use ONNX BGE-small via `fastembed` (`uv sync --extra encoder`).

## Quick start

```bash
uv sync --dev --extra encoder
cp .env.example .env   # set CATALOG_VALUE_ROOT if the sibling path is nonstandard
PYTHONPATH=src uv run pytest
PYTHONPATH=src uv run python -m audience_predictor coverage
PYTHONPATH=src uv run python -m audience_predictor ingest-moviesum --encoder bge
PYTHONPATH=src uv run python -m audience_predictor ingest-sublens --encoder hash
PYTHONPATH=src uv run python -m audience_predictor train --modality genome
PYTHONPATH=src uv run python -m audience_predictor train --modality screenplay_bge
PYTHONPATH=src uv run python -m audience_predictor train --modality dialogue_bge
PYTHONPATH=src uv run python -m audience_predictor evaluate
PYTHONPATH=src uv run python -m audience_predictor figures
PYTHONPATH=src uv run python -m audience_predictor report
```

Gold screenplays come from [MovieSum](https://huggingface.co/datasets/rohitsaxena/MovieSum)
(`data/raw/moviesum/{train,val,test}.jsonl`). Scripts stay local; git stores
parsers, IMDb identifiers, and hashed scene features only. Wikipedia summaries
in that dump are post-release text and are not used as input.

## Documentation

- [PR roadmap](docs/ROADMAP.md) — six stacked PRs, testing/training/narrative plan
- [Research report](reports/narrative-audience.md) — three registers, from lay to technical (PR 6)

## Scientific guardrails

A subtitle is not a screenplay. MovieLens users are not the general population.
Missing ratings are not dislikes. A screenplay does not uniquely determine the
finished film. This repository does not redistribute copyrighted script text.
