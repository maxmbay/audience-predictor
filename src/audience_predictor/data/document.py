"""Document typing and provenance. Never collapse screenplay vs transcript."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

DOCUMENT_TYPES = (
    "screenplay",
    "shooting_script",
    "transcript",
    "subtitle",
    "synopsis",
    "unknown",
)

_TIMECODE = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}")
_SRT_ARROW = re.compile(r"-->")
_SCENE = re.compile(r"^(INT\.|EXT\.|INT/EXT\.)", re.MULTILINE | re.IGNORECASE)
_SPEAKER_COLON = re.compile(r"^[A-Z][A-Za-z .]{1,24}:\s+\S", re.MULTILINE)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    movie_id: str | None
    imdb_id: str | None
    tmdb_id: int | None
    source: str
    source_url: str | None
    document_type: str
    draft_type: str | None
    retrieval_date: str
    word_count: int
    scene_count: int
    parse_quality: float
    match_confidence: float

    def as_dict(self) -> dict:
        return asdict(self)


def classify_document(text: str) -> str:
    sample = text[:80_000]
    n_timecode = len(_TIMECODE.findall(sample))
    n_arrow = len(_SRT_ARROW.findall(sample))
    n_scene = len(_SCENE.findall(sample))
    n_colon = len(_SPEAKER_COLON.findall(sample))
    if n_timecode >= 2 or (n_arrow >= 2 and n_scene == 0):
        return "subtitle"
    if n_scene >= 3:
        if re.search(r"\b(MORE|CONT'D|CONTINUED:)\b", sample):
            return "shooting_script"
        return "screenplay"
    if n_colon >= 8 and n_scene == 0:
        return "transcript"
    if len(sample.split()) < 400 and n_scene == 0:
        return "synopsis"
    return "unknown"


def make_record(
    *,
    document_id: str,
    text: str,
    source: str,
    document_type: str | None = None,
    movie_id: str | None = None,
    imdb_id: str | None = None,
    tmdb_id: int | None = None,
    source_url: str | None = None,
    draft_type: str | None = None,
    scene_count: int = 0,
    parse_quality: float = 0.0,
    match_confidence: float = 0.0,
) -> DocumentRecord:
    kind = document_type or classify_document(text)
    if kind not in DOCUMENT_TYPES:
        raise ValueError(f"unknown document_type {kind}")
    return DocumentRecord(
        document_id=document_id,
        movie_id=movie_id,
        imdb_id=imdb_id,
        tmdb_id=tmdb_id,
        source=source,
        source_url=source_url,
        document_type=kind,
        draft_type=draft_type,
        retrieval_date=datetime.now(UTC).date().isoformat(),
        word_count=len(text.split()),
        scene_count=scene_count,
        parse_quality=parse_quality,
        match_confidence=match_confidence,
    )
