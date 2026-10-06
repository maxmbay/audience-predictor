"""Parser-side craft and a lexical NLI stand-in.

The heuristic is not a substitute for OpenJev. It exists so the desk and
tests answer the same typed questions when 4B weights are not on disk.
"""

from __future__ import annotations

import re

from audience_predictor.coverage.schema import DOCUMENT_QUESTIONS, GENRE_OPTIONS, SCENE_QUESTIONS
from audience_predictor.data.screenplay_parser import ParsedScreenplay, Scene

STOCK_LOCATIONS = {
    "APARTMENT",
    "HOUSE",
    "HOME",
    "OFFICE",
    "BEDROOM",
    "KITCHEN",
    "LIVING ROOM",
    "BATHROOM",
    "STREET",
    "BAR",
    "RESTAURANT",
    "CAR",
    "ROOM",
    "HALLWAY",
}

WANT_RE = re.compile(
    r"\b(want|need|get|stop|don't|dont|must|please|give|let me|before|"
    r"go\b|move|won't|wont|can't|cant|find|keep|leave|stay|open|shut|"
    r"help|wait|come|should)\b",
    re.I,
)
FEEL_RE = re.compile(
    r"\b(i (love|hate|feel)|(i am|i'm).{0,16}(sad|angry|happy|scared|sorry)|"
    r"i know you|you (always|never|make me)|because i )\b",
    re.I,
)
CHANGE_RE = re.compile(
    r"\b(then|but|instead|doesn't|doesnt|already|anyway|left|slides|"
    r"stops|turns|before the next|no longer|too late)\b",
    re.I,
)
CONCRETE_RE = re.compile(
    r"\b(\d+|keycard|viewport|airlock|drawing|pipe|well|ticket|tickets|"
    r"bus|blood|ghost|silo|thermos|bicycle|plywood|gravel|orange|"
    r"paperback|headphones|duct tape|pickup|corn|moon|brick)\b",
    re.I,
)
PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,}|[A-Z]{2,})\b")

GENRE_CUES: dict[str, tuple[str, ...]] = {
    "horror": ("well", "blood", "ghost", "scream", "dark", "dead", "fear", "wet tick", "drain"),
    "tragedy": ("death", "kill", "revenge", "blood", "fate", "grave", "poison"),
    "romance": ("love", "kiss", "ticket", "heart", "maybe", "together", "date"),
    "comedy": ("joke", "laugh", "funny", "ridiculous", "anyway"),
    "thriller": ("map", "phone", "don't", "chase", "run", "watch"),
    "drama": ("family", "said", "leave", "stay", "sorry"),
    "action": ("gun", "run", "fight", "explode", "chase"),
    "mystery": ("who", "note", "clue", "why", "secret"),
}


def scene_premise(scene: Scene, *, limit: int = 1800) -> str:
    lines = [
        f"SCENE: {scene.heading}",
        f"CHARACTERS: {', '.join(scene.characters) or 'none'}",
    ]
    if scene.action:
        lines.append("ACTION: " + " ".join(scene.action))
    for speaker, line in scene.dialogue:
        lines.append(f"{speaker}: {line}")
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


def document_premise(parsed: ParsedScreenplay, *, n_scenes: int = 6, limit: int = 2400) -> str:
    chunks = [scene_premise(scene, limit=800) for scene in parsed.scenes[:n_scenes]]
    text = "\n\n".join(chunks) if chunks else ""
    if len(text) <= limit:
        return text
    return text[:limit]


def screenplay_stats(parsed: ParsedScreenplay) -> dict[str, float | int | list[str]]:
    n_scenes = max(parsed.n_scenes, 1)
    dialogue_chars = sum(len(s.dialogue_text) for s in parsed.scenes)
    action_chars = sum(len(s.action_text) for s in parsed.scenes)
    total = max(dialogue_chars + action_chars, 1)
    locations = [s.location for s in parsed.scenes if s.location]
    loc_set = {loc.upper() for loc in locations}
    stock = sum(1 for loc in loc_set if any(s in loc for s in STOCK_LOCATIONS))
    return {
        "n_scenes": parsed.n_scenes,
        "n_characters": len(parsed.characters),
        "characters": parsed.characters[:16],
        "dialogue_fraction": dialogue_chars / total,
        "n_locations": len(loc_set),
        "stock_location_rate": stock / max(len(loc_set), 1),
        "mean_scene_chars": (dialogue_chars + action_chars) / n_scenes,
    }


def genericness_from_stats(stats: dict[str, float | int | list[str]]) -> float:
    stock = float(stats["stock_location_rate"])
    n_loc = int(stats["n_locations"])
    n_scenes = max(int(stats["n_scenes"]), 1)
    uniqueness = min(1.0, n_loc / n_scenes)
    return max(0.0, min(1.0, 0.55 * stock + 0.45 * (1.0 - uniqueness)))


def heuristic_scene_scores(scene: Scene) -> dict[str, float]:
    dialogue = scene.dialogue_text
    action = scene.action_text
    body = f"{action} {dialogue}"
    n_lines = max(len(scene.dialogue), 1)
    want_hits = len(WANT_RE.findall(body))
    feel_hits = len(FEEL_RE.findall(dialogue))
    change_hits = len(CHANGE_RE.findall(body))
    concrete_hits = len(CONCRETE_RE.findall(body))
    proper_hits = len(PROPER_RE.findall(action))
    loc = (scene.location or "").upper()
    stock = any(s in loc for s in STOCK_LOCATIONS)

    want = _clip(0.12 * want_hits + 0.08 * min(len(scene.dialogue), 6))
    if any(verb in dialogue.lower() for verb in ("go", "don't", "need", "should", "have to")):
        want = _clip(want + 0.2)

    turns = _clip(0.18 * change_hits + 0.1 * min(len(scene.action), 4))
    if scene.action and scene.dialogue:
        turns = _clip(turns + 0.18)
    if len({name for name, _ in scene.dialogue}) >= 2:
        turns = _clip(turns + 0.12)

    specific = _clip(0.16 * concrete_hits + 0.04 * proper_hits)
    if loc and not stock:
        specific = _clip(specific + 0.28)
    elif loc and stock:
        specific = _clip(specific + 0.05)
    if scene.heading and scene.heading != "UNSLUGGED":
        specific = _clip(specific + 0.08)

    on_the_nose = _clip(feel_hits / n_lines)
    if feel_hits and want_hits == 0:
        on_the_nose = _clip(on_the_nose + 0.2)

    mass = len(action) + len(dialogue)
    dead_air = 0.08
    if want < 0.22 and turns < 0.22:
        dead_air = 0.62
    if mass < 40 and want < 0.22:
        dead_air = max(dead_air, 0.78)
    if not scene.dialogue and not scene.action:
        dead_air = 1.0
    dead_air = _clip(dead_air)

    return {
        "turns": turns,
        "want": want,
        "specific": specific,
        "on_the_nose": on_the_nose,
        "dead_air": dead_air,
    }


def heuristic_entailment(premise: str, hypothesis: str) -> float:
    """Map a typed hypothesis back onto parser craft for the heuristic backend."""
    hypo = hypothesis.lower()
    scene = _scene_from_premise(premise)
    scores = heuristic_scene_scores(scene) if scene is not None else {}
    if "situation has changed" in hypo or "end of this scene" in hypo:
        return scores.get("turns", 0.4)
    if "trying to get" in hypo or "prevent" in hypo:
        return scores.get("want", 0.4)
    if "concrete and particular" in hypo:
        return scores.get("specific", 0.4)
    if "say exactly what they feel" in hypo:
        return scores.get("on_the_nose", 0.3)
    if "could be removed" in hypo:
        return scores.get("dead_air", 0.3)
    if "generic version" in hypo:
        return _genericness_from_premise(premise)
    for genre in GENRE_OPTIONS:
        if hypo.strip().endswith(genre) or f"a {genre}" in hypo or f"an {genre}" in hypo:
            return genre_affinity(premise, genre)
    hits = sum(1 for token in re.findall(r"[a-z]{4,}", hypo) if token in premise.lower())
    return _clip(0.15 + 0.08 * hits)


def genre_affinity(text: str, genre: str) -> float:
    hay = text.lower()
    cues = GENRE_CUES.get(genre, ())
    if not cues:
        return 0.1
    hits = sum(1 for cue in cues if cue in hay)
    return _clip(0.08 + 0.18 * hits)


def genre_choice(text: str) -> dict[str, float]:
    raw = {g: genre_affinity(text, g) for g in GENRE_OPTIONS}
    return _softmax(raw)


def _genericness_from_premise(premise: str) -> float:
    headings = re.findall(r"^SCENE: (.+)$", premise, re.M)
    if not headings:
        return 0.45
    locs = [h.upper() for h in headings]
    stock = sum(1 for loc in locs if any(s in loc for s in STOCK_LOCATIONS))
    unique = len(set(locs))
    return _clip(0.5 * (stock / max(len(locs), 1)) + 0.5 * (1.0 - unique / max(len(locs), 1)))


def _scene_from_premise(premise: str) -> Scene | None:
    if "SCENE:" not in premise:
        return None
    heading = "UNSLUGGED"
    action: list[str] = []
    dialogue: list[tuple[str, str]] = []
    characters: list[str] = []
    for raw in premise.splitlines():
        line = raw.strip()
        if line.startswith("SCENE:"):
            heading = line[6:].strip() or heading
        elif line.startswith("CHARACTERS:"):
            names = [n.strip() for n in line[11:].split(",") if n.strip() and n.strip() != "none"]
            characters.extend(names)
        elif line.startswith("ACTION:"):
            action.append(line[7:].strip())
        elif ": " in line:
            speaker, text = line.split(": ", 1)
            if speaker == speaker.upper() and len(speaker) < 40:
                dialogue.append((speaker, text))
                if speaker not in characters:
                    characters.append(speaker)
    interior = True if heading.upper().startswith("INT") else False if heading.upper().startswith("EXT") else None
    location = ""
    if " - " in heading:
        location = heading.split(" - ", 1)[0]
        location = re.sub(r"^(INT\.|EXT\.|INT/EXT\.|I/E\.)\s*", "", location, flags=re.I)
    return Scene(
        heading=heading,
        interior=interior,
        location=location.strip(),
        time_of_day="",
        action=action,
        dialogue=dialogue,
        characters=characters,
    )


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _softmax(raw: dict[str, float], temperature: float = 0.35) -> dict[str, float]:
    import math

    keys = list(raw)
    xs = [raw[k] / max(temperature, 1e-6) for k in keys]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps) or 1.0
    return {k: e / z for k, e in zip(keys, exps)}


# Keep question lists imported so a missing schema key fails at import.
assert {q.key for q in SCENE_QUESTIONS} == {"turns", "want", "specific", "on_the_nose", "dead_air"}
assert DOCUMENT_QUESTIONS[0].key == "genericness"
