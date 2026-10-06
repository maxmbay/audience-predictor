"""OpenJev / MNLI backend: premise + hypothesis → class probabilities.

Default is heuristic so tests and the desk run without a 4B download.
Set OPENJEV_BACKEND=openjev and OPENJEV_DOWNLOAD=1 to pull
AlexWortega/openjev (qwen3.5-4b-nli). Cached weights load on auto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

Pair = tuple[str, str]


@dataclass(frozen=True)
class Triple:
    contradiction: float
    entailment: float
    neutral: float

    def as_dict(self) -> dict[str, float]:
        return {
            "contradiction": self.contradiction,
            "entailment": self.entailment,
            "neutral": self.neutral,
        }


class NLIBackend(Protocol):
    name: str
    kind: str

    def predict(self, pairs: list[Pair]) -> list[Triple]: ...


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def backend_kind_requested() -> str:
    return _env("OPENJEV_BACKEND", "auto").lower()


def model_id() -> str:
    return _env("OPENJEV_MODEL", "AlexWortega/openjev")


def model_subfolder() -> str | None:
    raw = _env("OPENJEV_SUBFOLDER", "qwen3.5-4b-nli")
    return raw or None


def allow_download() -> bool:
    return _env("OPENJEV_DOWNLOAD", "0").lower() in {"1", "true", "yes"}


class HeuristicNLI:
    """Lexical stand-in for the same (premise, hypothesis) → noul interface."""

    name = "heuristic"
    kind = "heuristic"

    def predict(self, pairs: list[Pair]) -> list[Triple]:
        return [_heuristic_triple(p, h) for p, h in pairs]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _heuristic_triple(premise: str, hypothesis: str) -> Triple:
    from audience_predictor.coverage.craft import heuristic_entailment

    p_yes = _clip01(heuristic_entailment(premise, hypothesis))
    p_no = _clip01((1.0 - p_yes) * 0.55)
    p_neutral = _clip01(1.0 - p_yes - p_no)
    mass = p_yes + p_no + p_neutral
    if mass <= 0:
        return Triple(0.33, 0.34, 0.33)
    return Triple(p_no / mass, p_yes / mass, p_neutral / mass)


class TransformersNLI:
    """AlexWortega OpenJev or any 3-way MNLI sequence classifier."""

    kind = "openjev"

    def __init__(self, model_name: str, *, subfolder: str | None, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        kwargs: dict = {"trust_remote_code": True, "local_files_only": not allow_download()}
        if subfolder:
            kwargs["subfolder"] = subfolder
        self.tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.torch = torch
        self.name = f"{model_name}" + (f"/{subfolder}" if subfolder else "")
        self._template = getattr(self.model.config, "nli_template", None)
        self._labels = _label_index(self.model.config)

    def predict(self, pairs: list[Pair]) -> list[Triple]:
        if not pairs:
            return []
        out: list[Triple] = []
        batch_size = max(1, int(_env("OPENJEV_BATCH", "4")))
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            out.extend(self._forward(chunk))
        return out

    def _forward(self, pairs: list[Pair]) -> list[Triple]:
        torch = self.torch
        if self._template:
            texts = [
                str(self._template).format(premise=p[:1800], hypothesis=h) for p, h in pairs
            ]
            enc = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        else:
            enc = self.tok(
                [p[:1800] for p, _ in pairs],
                [h for _, h in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        triples = []
        for row in probs:
            triples.append(
                Triple(
                    contradiction=float(row[self._labels["contradiction"]]),
                    entailment=float(row[self._labels["entailment"]]),
                    neutral=float(row[self._labels["neutral"]]),
                )
            )
        return triples


def _label_index(config) -> dict[str, int]:
    raw = getattr(config, "id2label", None) or {}
    mapping = {int(k): str(v).lower() for k, v in raw.items()}
    found = {"contradiction": 0, "entailment": 2, "neutral": 1}
    for idx, name in mapping.items():
        if "entail" in name:
            found["entailment"] = idx
        elif "contrad" in name:
            found["contradiction"] = idx
        elif "neutral" in name:
            found["neutral"] = idx
    return found


def _torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _try_openjev() -> NLIBackend | None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return None
    try:
        return TransformersNLI(model_id(), subfolder=model_subfolder(), device=_torch_device())
    except Exception:
        return None


_backend: NLIBackend | None = None


def reset_backend() -> None:
    global _backend
    _backend = None


def get_backend() -> NLIBackend:
    global _backend
    if _backend is not None:
        return _backend
    requested = backend_kind_requested()
    if requested == "heuristic":
        _backend = HeuristicNLI()
        return _backend
    loaded = _try_openjev()
    if requested == "openjev":
        if loaded is None:
            raise RuntimeError(
                "OpenJev weights missing. Cache AlexWortega/openjev "
                "(subfolder qwen3.5-4b-nli) or set OPENJEV_DOWNLOAD=1. "
                "Install extras: uv sync --extra openjev"
            )
        _backend = loaded
        return _backend
    _backend = loaded if loaded is not None else HeuristicNLI()
    return _backend


def describe_backend() -> dict[str, str]:
    backend = get_backend()
    return {
        "kind": backend.kind,
        "name": backend.name,
        "model": model_id() if backend.kind == "openjev" else "",
    }
