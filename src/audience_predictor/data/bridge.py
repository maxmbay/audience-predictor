from __future__ import annotations

import os
from pathlib import Path

from audience_predictor.paths import project_root


class BridgeError(FileNotFoundError):
    """catalog-value artifacts are missing or CATALOG_VALUE_ROOT is wrong."""


def catalog_value_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("CATALOG_VALUE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    sibling = project_root().parent / "catalog-value"
    if sibling.is_dir():
        return sibling.resolve()
    raise BridgeError(
        "catalog-value not found. Set CATALOG_VALUE_ROOT or clone it next to this repo."
    )


def require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise BridgeError(f"Missing {path}. {hint}")
    return path


def movies_path(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return require_file(
        base / "data" / "processed" / "core" / "movies.parquet",
        "Run `python -m catalog_value ingest` in catalog-value.",
    )


def ratings_path(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return require_file(
        base / "data" / "processed" / "core" / "ratings.parquet",
        "Run `python -m catalog_value ingest` in catalog-value.",
    )


def titles_path(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return require_file(
        base / "outputs" / "phase_a" / "titles.npz",
        "Run `python -m catalog_value fit` in catalog-value.",
    )


def atlas_path(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return require_file(
        base / "outputs" / "phase_a" / "title_atlas.npz",
        "Run `python -m catalog_value phase-a` in catalog-value.",
    )


def genome_scores_path(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return require_file(
        base / "data" / "processed" / "genome_scores.parquet",
        "Run `python -m catalog_value ingest` in catalog-value.",
    )


def genome_tags_path(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return require_file(
        base / "data" / "processed" / "genome_tags.parquet",
        "Run `python -m catalog_value ingest` in catalog-value.",
    )


def tags_csv_path(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return require_file(
        base / "data" / "raw" / "ml-25m" / "tags.csv",
        "MovieLens 25M extract missing tags.csv.",
    )


def processed_dir(root: Path | None = None) -> Path:
    base = root or catalog_value_root()
    return base / "data" / "processed"
