from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from audience_predictor.paths import config_dir, project_root


@dataclass(frozen=True)
class DataConfig:
    catalog_value_root: str | None
    movielens_url: str
    min_user_ratings: int
    min_movie_ratings: int
    hash_dim: int
    max_blocks: int
    tmdb_overview_limit: int | None


@dataclass(frozen=True)
class RepresentationConfig:
    embedding_dim: int
    n_interests: int
    n_heads: int
    n_inducing: int
    hidden: int
    model_dim: int


@dataclass(frozen=True)
class TrainConfig:
    holdout_frac: float
    min_train_ratings: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    nll_weight: float
    aux_weight: float
    bias_weight: float


@dataclass(frozen=True)
class EvalConfig:
    n0: float
    probes: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    representation: RepresentationConfig
    train: TrainConfig
    eval: EvalConfig
    source: Path


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path is not None else config_dir() / "default.yaml"
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())
    data = raw["data"]
    limit = data.get("tmdb_overview_limit")
    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(
            catalog_value_root=data.get("catalog_value_root"),
            movielens_url=str(data["movielens_url"]),
            min_user_ratings=int(data["min_user_ratings"]),
            min_movie_ratings=int(data["min_movie_ratings"]),
            hash_dim=int(data["hash_dim"]),
            max_blocks=int(data["max_blocks"]),
            tmdb_overview_limit=None if limit in (None, "null") else int(limit),
        ),
        representation=RepresentationConfig(**raw["representation"]),
        train=TrainConfig(**raw["train"]),
        eval=EvalConfig(
            n0=float(raw["eval"]["n0"]),
            probes=tuple(raw["eval"]["probes"]),
        ),
        source=config_path,
    )
