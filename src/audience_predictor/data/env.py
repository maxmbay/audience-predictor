from __future__ import annotations

import os
from pathlib import Path

from audience_predictor.paths import project_root


def load_dotenv(path: Path | None = None) -> None:
    env_path = path if path is not None else project_root() / ".env"
    if not env_path.exists():
        sibling = project_root().parent / "catalog-value" / ".env"
        env_path = sibling if sibling.exists() else env_path
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tmdb_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("TMDB_API_KEY", "").strip()
    return key or None
