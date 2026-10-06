"""Lazy studio: coverage first, audience map if a checkpoint exists."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from audience_predictor.config import Config, load_config
from audience_predictor.coverage import cover_pages
from audience_predictor.data.document import classify_document
from audience_predictor.eval.heads import available_heads
from audience_predictor.pipeline import run_predict

_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def predict_pages(text: str, *, k: int = 8) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("empty script")
    coverage = cover_pages(text)
    kind = classify_document(text)
    result: dict[str, Any] = {
        "coverage": coverage,
        "audience_available": False,
        "document_type": kind,
        "path": None,
        "n_blocks": int(coverage["stats"]["n_scenes"]),
        "sigma": None,
        "polarization": None,
        "comps": [],
        "rt": None,
    }
    if not available_heads():
        result["audience_error"] = "Train a parked head before the audience sidecar will load."
        return json_safe(result)
    try:
        audience = run_predict(get_config(), text, k=k, emit=False)
        result.update(audience)
        result["coverage"] = coverage
        result["audience_available"] = True
    except Exception as exc:
        result["audience_error"] = str(exc)
    return json_safe(result)
