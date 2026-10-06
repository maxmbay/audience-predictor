"""Typed questions the scene judge answers. No generation.

Each item is a noul (true/false) whose readout is P(entailment) from an
NLI/OpenJev backend. Document-level items are the same primitive over a
truncated script premise, plus a choice over genre labels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    key: str
    hypothesis: str
    invert: bool = False
    """If True, high entailment is a craft defect (on-the-nose, dead air)."""


SCENE_QUESTIONS: tuple[Question, ...] = (
    Question(
        "turns",
        "By the end of this scene the situation has changed from the beginning.",
    ),
    Question(
        "want",
        "A character is actively trying to get something or prevent something.",
    ),
    Question(
        "specific",
        "The details in this scene are concrete and particular, not interchangeable stock.",
    ),
    Question(
        "on_the_nose",
        "The characters say exactly what they feel instead of implying it.",
        invert=True,
    ),
    Question(
        "dead_air",
        "This scene could be removed without changing what happens in the story.",
        invert=True,
    ),
)

DOCUMENT_QUESTIONS: tuple[Question, ...] = (
    Question(
        "genericness",
        "This screenplay is a generic version of a story that has been told many times.",
        invert=True,
    ),
)

GENRE_OPTIONS: tuple[str, ...] = (
    "tragedy",
    "drama",
    "horror",
    "thriller",
    "romance",
    "comedy",
    "action",
    "mystery",
)

STAMPS = ("pass", "consider", "recommend")
