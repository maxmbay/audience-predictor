from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def data_dir() -> Path:
    return project_root() / "data"


def config_dir() -> Path:
    return project_root() / "configs"


def output_dir() -> Path:
    return project_root() / "outputs"


def reports_dir() -> Path:
    return project_root() / "reports"
