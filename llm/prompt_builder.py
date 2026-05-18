"""
Prompt builder for all datasets (book, movie, flight).

Ablation changes:
  c1   → "Only use values that appear in the sources above."
  c2   → "If the same value appears in different formats, count them as one."
  c1c2 → both c1 and c2

Set config.PROMPT_DOMAIN and config.PROMPT_SHOT before calling any build_*_prompt function.
"""

import config
from data.loader import FLIGHT_ATTRS


def _shot() -> int:
    return getattr(config, "PROMPT_SHOT", 0)


# ─────────────────────────────────────────────────────────────────────────────
# 1-shot examples
# ─────────────────────────────────────────────────────────────────────────────

_BOOK_EXAMPLE = """\
[EXAMPLE]
Book ISBN: 019923356X
Multiple sellers report the following author(s):
  - seller_A: "STEPHEN KING; PETER STRAUB"
  - seller_B: "King, Stephen"
  - seller_C: "Stephen King, Peter Straub"

Answer: Stephen King; Peter Straub
[/EXAMPLE]

"""

_MOVIE_EXAMPLE = """\
[EXAMPLE]
Movie: "The Godfather" (1972)
Multiple sources report the following director(s):
  - source_A: "FRANCIS FORD COPPOLA"
  - source_B: "Coppola, Francis Ford"
  - source_C: "Francis Ford Coppola"

Answer: Francis Ford Coppola
[/EXAMPLE]

"""

_INDEPENDENT_EXAMPLE = """\
[EXAMPLE]
Entity: 019923356X
Multiple sources report conflicting values for the same attribute:
  - seller_A: "DAVID J. HAND"
  - seller_B: "David J Hand"
  - seller_C: "Hand, David J."

Answer: David J Hand
[/EXAMPLE]

"""

_FLIGHT_DD_EXAMPLE = """\
[EXAMPLE]
Flight: AA-1234-JFK-LAX
Multiple sources report the following data:

Scheduled departure:
  - aa: "12/01/2011 08:00 AM"
  - flightaware: "2011-12-01 08:00AM EST"

Actual departure:
  - aa: "12/01/2011 08:15 AM"
  - flightaware: "2011-12-01 08:15AM EST"

Departure gate:
  - aa: "B12"
  - flightaware: " B12 "

Scheduled arrival:
  - aa: "12/01/2011 11:30 AM"

Actual arrival:
  - aa: "12/01/2011 11:45 AM"

Arrival gate:
  - aa: "C5"

Scheduled departure: 12/01/2011 08:00 AM
Actual departure: 12/01/2011 08:15 AM
Departure gate: B12
Scheduled arrival: 12/01/2011 11:30 AM
Actual arrival: 12/01/2011 11:45 AM
Arrival gate: C5
[/EXAMPLE]

"""

_FLIGHT_DI_EXAMPLE = """\
[EXAMPLE]
Entity: AA-1234-JFK-LAX
Multiple sources report the following data:

Attribute 1:
  - source_A: "12/01/2011 08:00 AM"
  - source_B: "2011-12-01 08:00AM EST"

Attribute 2:
  - source_A: "12/01/2011 08:15 AM"
  - source_B: "2011-12-01 08:15AM EST"

Attribute 3:
  - source_A: "B12"
  - source_B: " B12 "

Attribute 4:
  - source_A: "12/01/2011 11:30 AM"

Attribute 5:
  - source_A: "12/01/2011 11:45 AM"

Attribute 6:
  - source_A: "C5"

Attribute 1: 12/01/2011 08:00 AM
Attribute 2: 12/01/2011 08:15 AM
Attribute 3: B12
Attribute 4: 12/01/2011 11:30 AM
Attribute 5: 12/01/2011 11:45 AM
Attribute 6: C5
[/EXAMPLE]

"""

# Flight DI attribute label mapping
_ATTR_LABELS = {attr: f"Attribute {i+1}" for i, attr in enumerate(FLIGHT_ATTRS)}
LABEL_TO_ATTR = {v: k for k, v in _ATTR_LABELS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Book / Movie builders
# ─────────────────────────────────────────────────────────────────────────────

def _book_dd(isbn: str, claims: list, change: str) -> str:
    lines = []
    if _shot() == 1:
        lines.append(_BOOK_EXAMPLE)
    lines += [
        f"Book ISBN: {isbn}",
        "Multiple sellers report the following author(s):",
    ]
    for seller, author in claims:
        lines.append(f'  - {seller}: "{author}"')
    lines.append("")
    lines.append("Who are all the correct authors?")
    if change in ("c1", "c1c2"):
        lines.append("Only use names that appear in the sources above.")
    if change in ("c2", "c1c2"):
        lines.append("If the same person appears in different formats, count them as one.")
    lines.append("If there are multiple authors, separate them with a semicolon (;).")
    lines.append("Reply with ONLY the author name(s), nothing else.")
    return "\n".join(lines)


def _movie_dd(title: str, year: str, claims: list, change: str) -> str:
    lines = []
    if _shot() == 1:
        lines.append(_MOVIE_EXAMPLE)
    lines += [
        f'Movie: "{title}" ({year})',
        "Multiple sources report the following director(s):",
    ]
    for source, director in claims:
        lines.append(f'  - {source}: "{director}"')
    lines.append("")
    lines.append("Who are all the correct directors?")
    if change in ("c1", "c1c2"):
        lines.append("Only use names that appear in the sources above.")
    if change in ("c2", "c1c2"):
        lines.append("If the same person appears in different formats, count them as one.")
    lines.append("If there are multiple directors, separate them with a semicolon (;).")
    lines.append("Reply with ONLY the director name(s), nothing else.")
    return "\n".join(lines)


def _book_movie_di(entity_id: str, claims: list, change: str) -> str:
    lines = []
    if _shot() == 1:
        lines.append(_INDEPENDENT_EXAMPLE)
    lines += [
        f"Entity: {entity_id}",
        "Multiple sources report conflicting values for the same attribute:",
    ]
    for source, value in claims:
        lines.append(f'  - {source}: "{value}"')
    lines.append("")
    lines.append("What is the correct value?")
    if change in ("c1", "c1c2"):
        lines.append("Only use values that appear in the sources above.")
    if change in ("c2", "c1c2"):
        lines.append("If the same entity appears in different formats, count them as one.")
    lines.append("If there are multiple correct values, separate them with a semicolon (;).")
    lines.append("Reply with ONLY the value(s), nothing else.")
    return "\n".join(lines)


_DEPENDENT_MAP   = {"dependent": "", "dependent_c1": "c1", "dependent_c2": "c2", "dependent_c1c2": "c1c2"}
_INDEPENDENT_MAP = {"independent": "", "independent_c1": "c1", "independent_c2": "c2", "independent_c1c2": "c1c2"}


def build_book_prompt(isbn: str, claims: list) -> str:
    domain = getattr(config, "PROMPT_DOMAIN", "dependent")
    if domain in _DEPENDENT_MAP:
        return _book_dd(isbn, claims, _DEPENDENT_MAP[domain])
    if domain in _INDEPENDENT_MAP:
        return _book_movie_di(isbn, claims, _INDEPENDENT_MAP[domain])
    raise ValueError(f"Unknown PROMPT_DOMAIN: {domain}")


def build_movie_prompt(title: str, year: str, claims: list) -> str:
    domain = getattr(config, "PROMPT_DOMAIN", "dependent")
    entity_id = f"{title} ({year})"
    if domain in _DEPENDENT_MAP:
        return _movie_dd(title, year, claims, _DEPENDENT_MAP[domain])
    if domain in _INDEPENDENT_MAP:
        return _book_movie_di(entity_id, claims, _INDEPENDENT_MAP[domain])
    raise ValueError(f"Unknown PROMPT_DOMAIN: {domain}")


# ─────────────────────────────────────────────────────────────────────────────
# Flight builders
# ─────────────────────────────────────────────────────────────────────────────

def _flight_dd(flight_id: str, claims_by_attr: dict, change: str) -> str:
    lines = []
    if _shot() == 1:
        lines.append(_FLIGHT_DD_EXAMPLE)
    lines.append(f"Flight: {flight_id}")
    lines.append("Multiple sources report the following data:")
    lines.append("")
    for attr in FLIGHT_ATTRS:
        sources = claims_by_attr.get(attr, [])
        lines.append(f"{attr}:")
        for source, value in sources:
            lines.append(f'  - {source}: "{value}"')
        if not sources:
            lines.append("  (no data)")
        lines.append("")
    lines.append("What is the correct value for each attribute?")
    if change in ("c1", "c1c2"):
        lines.append("Only use values that appear in the sources above.")
    if change in ("c2", "c1c2"):
        lines.append("If the same value appears in different formats, treat them as equivalent.")
    lines.append("Reply ONLY in this exact format (use N/A if unknown):")
    for attr in FLIGHT_ATTRS:
        lines.append(f"{attr}: <value>")
    return "\n".join(lines)


def _flight_di(flight_id: str, claims_by_attr: dict, change: str) -> str:
    lines = []
    if _shot() == 1:
        lines.append(_FLIGHT_DI_EXAMPLE)
    lines.append(f"Entity: {flight_id}")
    lines.append("Multiple sources report the following data:")
    lines.append("")
    for attr in FLIGHT_ATTRS:
        label = _ATTR_LABELS[attr]
        sources = claims_by_attr.get(attr, [])
        lines.append(f"{label}:")
        for source, value in sources:
            lines.append(f'  - {source}: "{value}"')
        if not sources:
            lines.append("  (no data)")
        lines.append("")
    lines.append("What is the correct value for each attribute?")
    if change in ("c1", "c1c2"):
        lines.append("Only use values that appear in the sources above.")
    if change in ("c2", "c1c2"):
        lines.append("If the same value appears in different formats, treat them as equivalent.")
    lines.append("Reply ONLY in this exact format (use N/A if unknown):")
    for attr in FLIGHT_ATTRS:
        lines.append(f"{_ATTR_LABELS[attr]}: <value>")
    return "\n".join(lines)


_FLIGHT_DEPENDENT_MAP   = {"flight_dependent": "", "flight_dependent_c1": "c1", "flight_dependent_c2": "c2", "flight_dependent_c1c2": "c1c2"}
_FLIGHT_INDEPENDENT_MAP = {"flight_independent": "", "flight_independent_c1": "c1", "flight_independent_c2": "c2", "flight_independent_c1c2": "c1c2"}


def build_flight_prompt(flight_id: str, claims_by_attr: dict) -> str:
    domain = getattr(config, "PROMPT_DOMAIN", "flight_dependent")
    if domain in _FLIGHT_DEPENDENT_MAP:
        return _flight_dd(flight_id, claims_by_attr, _FLIGHT_DEPENDENT_MAP[domain])
    if domain in _FLIGHT_INDEPENDENT_MAP:
        return _flight_di(flight_id, claims_by_attr, _FLIGHT_INDEPENDENT_MAP[domain])
    raise ValueError(f"Unknown PROMPT_DOMAIN: {domain}")
