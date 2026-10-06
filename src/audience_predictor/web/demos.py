"""Spec pages for the coverage desk.

Hollywood shooting scripts are still in copyright. The classics here are
public-domain plays (Shakespeare via Project Gutenberg), plus a couple of
original specs to exercise horror / romcom. Upload anything you have rights to.
"""

from __future__ import annotations

from pathlib import Path

DEMOS_DIR = Path(__file__).resolve().parent / "specs"

DEMOS: tuple[dict[str, str], ...] = (
    {
        "id": "hamlet",
        "file": "hamlet.fountain",
        "title": "HAMLET",
        "slugline": "EXT. ELSINORE. A PLATFORM BEFORE THE CASTLE - NIGHT",
        "logline": "Shakespeare, c.1600. Public domain. Who's there?",
        "tone": "classic",
    },
    {
        "id": "macbeth",
        "file": "macbeth.fountain",
        "title": "MACBETH",
        "slugline": "EXT. A DESERT PLACE - NIGHT",
        "logline": "Shakespeare, c.1606. Public domain. Fair is foul.",
        "tone": "classic",
    },
    {
        "id": "romeo_and_juliet",
        "file": "romeo_and_juliet.fountain",
        "title": "ROMEO AND JULIET",
        "slugline": "EXT. VERONA. A PUBLIC PLACE - DAY",
        "logline": "Shakespeare, c.1597. Public domain. Two households.",
        "tone": "classic",
    },
    {
        "id": "the_well",
        "file": "the_well.fountain",
        "title": "THE WELL",
        "slugline": "EXT. DIRT TRACK - NIGHT",
        "logline": "Original spec. A farmhouse, a well, a note in your own hand.",
        "tone": "horror",
    },
    {
        "id": "two_tickets",
        "file": "two_tickets.fountain",
        "title": "TWO TICKETS",
        "slugline": "INT. GREYHOUND STATION - DAY",
        "logline": "Original spec. Two seats, one maybe, a bus that does not care.",
        "tone": "romcom",
    },
)


def demo_catalog() -> list[dict[str, str]]:
    return [dict(item) for item in DEMOS]


def load_demo(demo_id: str) -> tuple[dict[str, str], str]:
    for item in DEMOS:
        if item["id"] == demo_id:
            path = DEMOS_DIR / item["file"]
            if not path.exists():
                raise FileNotFoundError(path)
            return dict(item), path.read_text(encoding="utf-8")
    raise KeyError(demo_id)
