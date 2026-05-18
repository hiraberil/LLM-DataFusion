"""
Parses LLM responses for the flight dataset into {attr: normalized_value}.

  - Times  → "HH:MM" (24h)
  - Gates  → uppercase, stripped
  - N/A or empty → omitted
"""

import re
from data.loader import FLIGHT_ATTRS
from .prompt_builder import LABEL_TO_ATTR

_DATE_ATTRS = {
    "Scheduled departure", "Actual departure",
    "Scheduled arrival",  "Actual arrival",
}
_GATE_ATTRS = {"Departure gate", "Arrival gate"}


_TIME_RE = re.compile(r'\b(\d{1,2}):(\d{2})\s*([aApP]\.?[mM]\.?)?', re.IGNORECASE)


def parse_time(value: str) -> str | None:
    """Converts a time string to HH:MM (24h). Ignores date."""
    if not value:
        return None
    value = value.strip()
    if value.upper() == "N/A":
        return None

    match = _TIME_RE.search(value)
    if not match:
        return None

    hour   = int(match.group(1))
    minute = int(match.group(2))
    ampm   = match.group(3)

    if ampm:
        ampm_clean = ampm.replace(".", "").upper()
        if ampm_clean == "PM" and hour != 12:
            hour += 12
        elif ampm_clean == "AM" and hour == 12:
            hour = 0

    return f"{hour:02d}:{minute:02d}"


def _normalize_datetime(value: str) -> str | None:
    return parse_time(value)


def _normalize_gate(value: str) -> str | None:
    value = value.strip().upper()
    if not value or value == "N/A":
        return None
    return value


def parse_flight_di_response(raw: str) -> dict:
    """Parses 'Attribute N: <value>' lines (domain-independent) and maps them to real attribute names."""
    result = {}
    for label, attr in LABEL_TO_ATTR.items():
        pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(raw)
        if not match:
            continue
        value = match.group(1).strip()
        if attr in _DATE_ATTRS:
            normalized = _normalize_datetime(value)
        else:
            normalized = _normalize_gate(value)
        if normalized:
            result[attr] = normalized
    return result


def parse_flight_response(raw: str) -> dict:
    """Parses LLM output into {attr: normalized_value}. Skips unrecognized or N/A values."""
    result = {}
    for attr in FLIGHT_ATTRS:
        pattern = re.compile(rf"^{re.escape(attr)}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(raw)
        if not match:
            continue
        value = match.group(1).strip()
        if attr in _DATE_ATTRS:
            normalized = _normalize_datetime(value)
        else:
            normalized = _normalize_gate(value)
        if normalized:
            result[attr] = normalized
    return result
