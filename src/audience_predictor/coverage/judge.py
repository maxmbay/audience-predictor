"""Scene loop: parse → typed questions → probabilities → stamp."""

from __future__ import annotations

import math
import os
from typing import Any

from audience_predictor.coverage.craft import (
    document_premise,
    genericness_from_stats,
    genre_choice,
    heuristic_scene_scores,
    scene_premise,
    screenplay_stats,
)
from audience_predictor.coverage.nli import NLIBackend, Triple, get_backend
from audience_predictor.coverage.schema import (
    DOCUMENT_QUESTIONS,
    GENRE_OPTIONS,
    SCENE_QUESTIONS,
    Question,
)
from audience_predictor.data.screenplay_parser import ParsedScreenplay, Scene, parse_screenplay


def max_scenes() -> int:
    raw = os.environ.get("OPENJEV_MAX_SCENES", "12")
    try:
        return max(1, min(int(raw), 40))
    except ValueError:
        return 12


def select_scene_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _entailment(triple: Triple) -> float:
    return float(triple.entailment)


def _noul_from_triples(questions: tuple[Question, ...], triples: list[Triple]) -> dict[str, float]:
    out: dict[str, float] = {}
    for question, triple in zip(questions, triples, strict=True):
        out[question.key] = _entailment(triple)
    return out


def _choice(backend: NLIBackend, premise: str, options: tuple[str, ...]) -> dict[str, float]:
    hyps = [f"This screenplay is {opt}." for opt in options]
    triples = backend.predict([(premise, h) for h in hyps])
    raw = {opt: _entailment(t) for opt, t in zip(options, triples)}
    mass = sum(raw.values())
    if mass <= 1e-8:
        return genre_choice(premise)
    return {k: v / mass for k, v in raw.items()}


def _craft_score(means: dict[str, float]) -> float:
    return (
        0.30 * means.get("turns", 0)
        + 0.25 * means.get("want", 0)
        + 0.20 * means.get("specific", 0)
        + 0.15 * (1.0 - means.get("on_the_nose", 0))
        + 0.10 * (1.0 - means.get("dead_air", 0))
    )


def stamp_for(score: float) -> str:
    if score >= 0.58:
        return "recommend"
    if score >= 0.40:
        return "consider"
    return "pass"


def _score_scenes(
    scenes: list[Scene],
    indices: list[int],
    backend: NLIBackend,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if backend.kind == "heuristic":
        for idx in indices:
            scene = scenes[idx]
            scores = heuristic_scene_scores(scene)
            rows.append(_scene_row(idx, scene, scores, source="heuristic"))
        return rows

    pairs: list[tuple[str, str]] = []
    premises: list[str] = []
    for idx in indices:
        premise = scene_premise(scenes[idx])
        premises.append(premise)
        for question in SCENE_QUESTIONS:
            pairs.append((premise, question.hypothesis))
    triples = backend.predict(pairs)
    width = len(SCENE_QUESTIONS)
    for i, idx in enumerate(indices):
        chunk = triples[i * width : (i + 1) * width]
        scores = _noul_from_triples(SCENE_QUESTIONS, chunk)
        rows.append(_scene_row(idx, scenes[idx], scores, source=backend.kind))
    return rows


def _softmax_row(scores: dict[str, float], *, temperature: float = 0.35) -> dict[str, float]:
    keys = [q.key for q in SCENE_QUESTIONS]
    xs = [float(scores.get(k, 0.0)) / max(temperature, 1e-6) for k in keys]
    peak = max(xs) if xs else 0.0
    exps = [math.exp(x - peak) for x in xs]
    mass = sum(exps) or 1.0
    return {k: round(e / mass, 4) for k, e in zip(keys, exps)}


def _scene_row(idx: int, scene: Scene, scores: dict[str, float], *, source: str) -> dict[str, Any]:
    craft = _clip(_craft_score(scores))
    return {
        "index": idx,
        "heading": scene.heading,
        "n_lines": len(scene.dialogue),
        "n_action": len(scene.action),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "softmax": _softmax_row(scores),
        "craft": round(craft, 4),
        "source": source,
    }


def cover_pages(text: str, *, backend: NLIBackend | None = None) -> dict[str, Any]:
    parsed: ParsedScreenplay = parse_screenplay(text)
    backend = backend or get_backend()
    indices = select_scene_indices(parsed.n_scenes, max_scenes())
    scene_rows = _score_scenes(parsed.scenes, indices, backend) if indices else []
    keys = [q.key for q in SCENE_QUESTIONS]
    means = {key: _mean([row["scores"][key] for row in scene_rows]) for key in keys}

    stats = screenplay_stats(parsed)
    premise = document_premise(parsed)
    if backend.kind == "heuristic":
        genericness = genericness_from_stats(stats)
        genres = genre_choice(text)
    else:
        doc_triples = backend.predict([(premise, q.hypothesis) for q in DOCUMENT_QUESTIONS])
        genericness = _entailment(doc_triples[0]) if doc_triples else genericness_from_stats(stats)
        genres = _choice(backend, premise, GENRE_OPTIONS)

    craft = _clip(_craft_score(means) * (1.0 - 0.25 * genericness))
    stamp = stamp_for(craft)
    top_genre = max(genres, key=genres.get) if genres else "drama"
    return {
        "judge": {"kind": backend.kind, "name": backend.name},
        "stamp": stamp,
        "craft": round(craft, 4),
        "scene_means": {k: round(v, 4) for k, v in means.items()},
        "genericness": round(float(genericness), 4),
        "genre": top_genre,
        "genre_probs": {k: round(v, 4) for k, v in genres.items()},
        "stats": {
            "n_scenes": stats["n_scenes"],
            "n_characters": stats["n_characters"],
            "characters": stats["characters"],
            "dialogue_fraction": round(float(stats["dialogue_fraction"]), 4),
            "n_locations": stats["n_locations"],
            "stock_location_rate": round(float(stats["stock_location_rate"]), 4),
        },
        "n_judged": len(scene_rows),
        "scenes": scene_rows,
        "timeline": {
            "keys": [q.key for q in SCENE_QUESTIONS],
            "n_scenes": int(stats["n_scenes"]),
            "x": [row["index"] for row in scene_rows],
            "craft": [row["craft"] for row in scene_rows],
            "softmax": [row["softmax"] for row in scene_rows],
            "scores": [row["scores"] for row in scene_rows],
        },
    }


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
