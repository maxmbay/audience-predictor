"""Partition MovieLens genome tags into what a *script* could know.

Genome recovers audience (~0.38 cosine) but many tags are about the released
film: Oscars, CGI, sequels, 'masterpiece'. Those are descriptive, not
available for an undeveloped original. The predictive ceiling is the
script-visible subset.

Buckets
-------
script       Tone, tropes, themes, setting, plot devices, character types.
production   How it was made or sold: format, budget, studio, franchise, talent, adaptations.
reception    Quality, awards, lists, 'good/bad' judgments after an audience saw it.

When a tag is ambiguous, it is *not* script. That keeps the oracle ceiling honest.
"""

from __future__ import annotations

import re
from typing import Literal

Bucket = Literal["script", "production", "reception"]
BUCKETS: tuple[Bucket, ...] = ("script", "production", "reception")

# Whole-string quality / list / award judgments.
_RECEPTION_EXACT = {
    "awesome",
    "awful",
    "bad",
    "beautiful",
    "boring",
    "boring!",
    "breathtaking",
    "brilliant",
    "classic",
    "clever",
    "cool",
    "corny",
    "cult",
    "cult classic",
    "cult film",
    "cute",
    "cute!",
    "disappointing",
    "dumb",
    "entertaining",
    "excellent",
    "exceptional acting",
    "exciting",
    "fun",
    "fun movie",
    "funniest movies",
    "funny as hell",
    "good",
    "good acting",
    "good action",
    "good dialogue",
    "good music",
    "good romantic comedies",
    "good soundtrack",
    "good story",
    "great",
    "great acting",
    "great cinematography",
    "great dialogue",
    "great ending",
    "great movie",
    "great music",
    "great soundtrack",
    "guilty pleasure",
    "hard to watch",
    "hilarious",
    "hillarious",
    "horrible",
    "idiotic",
    "imdb top 250",
    "interesting",
    "lame",
    "long",
    "masterpiece",
    "movielens top pick",
    "moving",
    "original",
    "original plot",
    "overrated",
    "perfect",
    "plot holes",
    "pointless",
    "predictable",
    "pretentious",
    "quotable",
    "highly quotable",
    "ridiculous",
    "sad but good",
    "shallow",
    "so bad it's funny",
    "so bad it's good",
    "special",
    "stupid",
    "stupid as hell",
    "stupidity",
    "stunning",
    "suprisingly clever",
    "too long",
    "too short",
    "touching",
    "underrated",
    "unfunny",
    "unintentionally funny",
    "unique",
    "unlikeable characters",
    "unrealistic",
    "very funny",
    "very good",
    "very interesting",
    "visually appealing",
    "visually stunning",
    "visuals",
    "waste of time",
    "watch the credits",
    "better than expected",
}

_RECEPTION_RE = re.compile(
    r"("
    r"oscar|\baoskar\b|golden palm|saturn award|afi 100|criterion|"
    r"best of |potential oscar|overrated|underrated|"
    r"amazing cinematography|amazing photography|beautifully filmed|"
    r"bad acting|bad cgi|bad ending|bad plot|bad science|bad script|"
    r"better than|not as good|crappy sequel|good sequel|bad sequel|"
    r"no plot|plot holes|free to download|easily confused"
    r")",
    re.IGNORECASE,
)

_PRODUCTION_EXACT = {
    "3d",
    "70mm",
    "aardman",
    "aardman studios",
    "adaptation",
    "adapted from:book",
    "adapted from:comic",
    "adapted from:game",
    "almodovar",
    "amy smart",
    "arnold",
    "art house",
    "author:alan moore",
    "author:neil gaiman",
    "awesome soundtrack",
    "based on a book",
    "based on a comic",
    "based on a play",
    "based on a tv show",
    "based on a video game",
    "based on book",
    "based on comic",
    "batman",
    "beatles",
    "big budget",
    "black and white",
    "bond",
    "book",
    "book was better",
    "books",
    "c.s. lewis",
    "carrie-anne moss",
    "cgi",
    "chris tucker",
    "cinematography",
    "coen bros",
    "coen brothers",
    "comic",
    "comic book",
    "comic book adaption",
    "comics",
    "computer animation",
    "computer game",
    "dc comics",
    "depp & burton",
    "directorial debut",
    "disney",
    "disney animated feature",
    "dr. seuss",
    "dreamworks",
    "dynamic cgi action",
    "effects",
    "emma watson",
    "ennio morricone",
    "ensemble cast",
    "firefly",
    "franchise",
    "francis ford copolla",
    "george orwell",
    "gilliam",
    "graphic novel",
    "handycam",
    "hannibal lecter",
    "harry potter",
    "hitchcock",
    "hollywood",
    "imax",
    "independent film",
    "indiana jones",
    "indie",
    "james bond",
    "jane austen",
    "jay and silent bob",
    "kubrick",
    "kurosawa",
    "literary adaptation",
    "liv tyler",
    "low budget",
    "lynch",
    "made for tv",
    "marvel",
    "marx brothers",
    "miyazaki",
    "monty python",
    "movie business",
    "mtv",
    "muppets",
    "neil gaiman",
    "notable soundtrack",
    "pg",
    "pg-13",
    "philip k. dick",
    "pixar",
    "pixar animation",
    "prequel",
    "product placement",
    "remade",
    "remake",
    "robert downey jr",
    "robert ludlum",
    "roald dahl",
    "saturday night live",
    "sequel",
    "sequels",
    "series",
    "sophia coppola",
    "special effects",
    "spielberg",
    "spock",
    "star trek",
    "star wars",
    "studio ghibli",
    "tarantino",
    "tolkien",
    "tom clancy",
    "trilogy",
    "truman capote",
    "video game",
    "video game adaptation",
    "video games",
    "videogame",
    "view askew",
    "007",
    "007 (series)",
}

_PRODUCTION_RE = re.compile(
    r"("
    r"^author:|adapted from|based on a |based on book|based on comic|"
    r"\bcgi\b|special effects|big budget|low budget|\bindie\b|"
    r"\bimax\b|\b3d\b|70mm|pixar|disney|dreamworks|aardman|ghibli|"
    r"\bmarvel\b|\bfranchise\b|\bsequel|\bprequel|\bremake|\btrilogy|"
    r"directorial debut|ensemble cast|product placement|made for tv|"
    r"black and white|computer animation|comic book adapt"
    r")",
    re.IGNORECASE,
)


def _norm(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip().lower())


def bucket_tag(tag: str) -> Bucket:
    """Assign one genome tag. Ambiguous tags are production or reception, not script."""
    name = _norm(tag)
    if name in _RECEPTION_EXACT or _RECEPTION_RE.search(name):
        return "reception"
    if name in _PRODUCTION_EXACT or _PRODUCTION_RE.search(name):
        return "production"
    return "script"


def bucket_mask(tags: list[str], bucket: Bucket) -> list[bool]:
    return [bucket_tag(name) == bucket for name in tags]


def partition_tags(tags: list[str]) -> dict[Bucket, list[str]]:
    out: dict[Bucket, list[str]] = {b: [] for b in BUCKETS}
    for name in tags:
        out[bucket_tag(name)].append(name)
    return out
