"""
Parses LLM responses for book and movie into a set of normalized values.
"""

from data.normalizer import normalize


def parse_response(raw: str) -> set:
    # Splits on ";", normalizes each part, and discards empty results.
    parts = raw.split(";")
    result = set()
    for part in parts:
        normalized = normalize(part)
        if normalized:
            result.add(normalized)
    return result
