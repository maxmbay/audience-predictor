"""Parse production-format screenplays into scenes, dialogue, and action."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SCENE_RE = re.compile(
    r"^(INT\.|EXT\.|INT/EXT\.|I/E\.)[.\s].+",
    re.IGNORECASE,
)
TRANSITION_RE = re.compile(
    r"^(CUT TO:|FADE IN:|FADE OUT:|SMASH CUT:|DISSOLVE TO:|BACK TO:)\s*$",
    re.IGNORECASE,
)
CHARACTER_RE = re.compile(r"^[A-Z][A-Z0-9 #.'\-]{1,36}$")
PARENTHETICAL_RE = re.compile(r"^\(.*\)$")


@dataclass
class Scene:
    heading: str
    interior: bool | None
    location: str
    time_of_day: str
    action: list[str] = field(default_factory=list)
    dialogue: list[tuple[str, str]] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)

    @property
    def action_text(self) -> str:
        return " ".join(self.action)

    @property
    def dialogue_text(self) -> str:
        return " ".join(line for _, line in self.dialogue)


@dataclass
class ParsedScreenplay:
    scenes: list[Scene]
    transitions: list[str]

    @property
    def n_scenes(self) -> int:
        return len(self.scenes)

    @property
    def characters(self) -> list[str]:
        seen: list[str] = []
        for scene in self.scenes:
            for name in scene.characters:
                if name not in seen:
                    seen.append(name)
        return seen


def _split_heading(heading: str) -> tuple[bool | None, str, str]:
    upper = heading.strip().upper()
    interior: bool | None
    if upper.startswith("INT/EXT") or upper.startswith("I/E"):
        interior = None
        rest = re.sub(r"^(INT/EXT\.?|I/E\.?)\s*", "", upper)
    elif upper.startswith("INT"):
        interior = True
        rest = re.sub(r"^INT\.?\s*", "", upper)
    elif upper.startswith("EXT"):
        interior = False
        rest = re.sub(r"^EXT\.?\s*", "", upper)
    else:
        interior = None
        rest = upper
    if " - " in rest:
        location, time_of_day = rest.rsplit(" - ", 1)
    else:
        location, time_of_day = rest, ""
    return interior, location.strip(" ."), time_of_day.strip(" .")


def _is_character(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return False
    if stripped.endswith("TO:"):
        return False
    return bool(CHARACTER_RE.match(stripped.replace("(V.O.)", "").replace("(O.S.)", "").strip()))


def parse_screenplay(text: str) -> ParsedScreenplay:
    scenes: list[Scene] = []
    transitions: list[str] = []
    current: Scene | None = None
    speaker: str | None = None

    def ensure_scene() -> Scene:
        nonlocal current
        if current is None:
            current = Scene(
                heading="UNSLUGGED",
                interior=None,
                location="",
                time_of_day="",
            )
            scenes.append(current)
        return current

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            speaker = None
            continue
        if TRANSITION_RE.match(stripped):
            transitions.append(stripped.upper())
            speaker = None
            continue
        if SCENE_RE.match(stripped):
            interior, location, time_of_day = _split_heading(stripped)
            current = Scene(
                heading=stripped,
                interior=interior,
                location=location,
                time_of_day=time_of_day,
            )
            scenes.append(current)
            speaker = None
            continue
        if _is_character(stripped):
            speaker = re.sub(r"\s*\((V\.O\.|O\.S\.)\)\s*$", "", stripped).strip()
            scene = ensure_scene()
            if speaker not in scene.characters:
                scene.characters.append(speaker)
            continue
        if speaker is not None:
            if PARENTHETICAL_RE.match(stripped):
                continue
            ensure_scene().dialogue.append((speaker, stripped))
            continue
        ensure_scene().action.append(stripped)

    return ParsedScreenplay(scenes=scenes, transitions=transitions)


def structural_features(parsed: ParsedScreenplay) -> dict[str, float]:
    n_scenes = max(parsed.n_scenes, 1)
    scene_lens = [len(s.action_text) + len(s.dialogue_text) for s in parsed.scenes] or [0]
    dialogue_chars = sum(len(s.dialogue_text) for s in parsed.scenes)
    action_chars = sum(len(s.action_text) for s in parsed.scenes)
    total = max(dialogue_chars + action_chars, 1)
    names = parsed.characters
    counts = [sum(s.characters.count(n) for s in parsed.scenes) for n in names] or [1]
    mass = float(sum(counts))
    entropy = 0.0
    if mass > 0 and len(counts) > 1:
        import math

        entropy = -sum((c / mass) * math.log(c / mass + 1e-12) for c in counts)
    locations = [s.location for s in parsed.scenes if s.location]
    loc_set = set(locations)
    loc_entropy = 0.0
    if locations:
        import math
        from collections import Counter

        c = Counter(locations)
        n = len(locations)
        loc_entropy = -sum((v / n) * math.log(v / n + 1e-12) for v in c.values())
    n_chars = len(names)
    co_edges = 0
    for scene in parsed.scenes:
        k = len(scene.characters)
        co_edges += k * (k - 1) // 2
    density = co_edges / max(n_chars * (n_chars - 1) / 2, 1)
    return {
        "n_scenes": _log1p(parsed.n_scenes),
        "median_scene_length": _log1p(sorted(scene_lens)[len(scene_lens) // 2]),
        "dialogue_fraction": dialogue_chars / total,
        "action_density": _log1p(action_chars / n_scenes),
        "n_characters": _log1p(n_chars),
        "character_entropy": float(entropy),
        "location_entropy": float(loc_entropy),
        "n_locations": _log1p(len(loc_set)),
        "graph_n_edges": _log1p(co_edges),
        "graph_density": float(density),
        "scene_transition_rate": float(len(parsed.transitions)) / n_scenes,
    }


def _log1p(value: float | int) -> float:
    import math

    return math.log1p(max(float(value), 0.0))


STRUCT_KEYS = (
    "n_scenes",
    "median_scene_length",
    "dialogue_fraction",
    "action_density",
    "n_characters",
    "character_entropy",
    "location_entropy",
    "n_locations",
    "graph_n_edges",
    "graph_density",
    "scene_transition_rate",
)


def parse_moviesum_xml(xml_text: str) -> ParsedScreenplay:
    """Parse MovieSum's scene-tagged XML into the shared screenplay schema."""
    import xml.etree.ElementTree as ET

    text = xml_text.strip()
    if not text.startswith("<"):
        return parse_screenplay(text)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        wrapped = f"<script>{text}</script>"
        try:
            root = ET.fromstring(wrapped)
        except ET.ParseError:
            return parse_screenplay(re.sub(r"<[^>]+>", " ", text))

    scenes: list[Scene] = []
    transitions: list[str] = []
    for scene_el in root.findall(".//scene"):
        heading = " ".join(scene_el.findtext("stage_direction", default="").split())
        interior, location, time_of_day = _split_heading(heading) if heading else (None, "", "")
        scene = Scene(
            heading=heading or "UNSLUGGED",
            interior=interior,
            location=location,
            time_of_day=time_of_day,
        )
        speaker: str | None = None
        for child in list(scene_el):
            tag = child.tag.lower()
            body = " ".join((child.text or "").split())
            if not body:
                continue
            if tag == "stage_direction":
                continue
            if tag in {"scene_description", "action"}:
                scene.action.append(body)
                speaker = None
            elif tag == "character":
                speaker = body.upper()
                if speaker not in scene.characters:
                    scene.characters.append(speaker)
            elif tag == "dialogue" and speaker is not None:
                scene.dialogue.append((speaker, body))
            elif tag == "dialogue":
                scene.dialogue.append(("UNKNOWN", body))
            elif tag == "parenthetical":
                continue
            elif TRANSITION_RE.match(body):
                transitions.append(body.upper())
        scenes.append(scene)
    if not scenes:
        return parse_screenplay(re.sub(r"<[^>]+>", " ", text))
    return ParsedScreenplay(scenes=scenes, transitions=transitions)
