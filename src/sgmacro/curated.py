"""The hand-maintained policy event list: loading, validation, quarterly counts.

One module owns the file format so that parsing, fetching, tracing and the report
all read the list the same way, and a malformed edit fails loudly in one place.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import yaml

from sgmacro.periods import Period
from sgmacro.registry import Frequency

DIRECTIONS = {"tightening": 1, "easing": -1}


class CuratedError(ValueError):
    """The event list is malformed."""


@dataclass(frozen=True)
class PolicyEvent:
    announced: dt.date
    direction: str
    instruments: tuple[str, ...]
    summary: str
    source: str

    @property
    def sign(self) -> int:
        return DIRECTIONS[self.direction]

    @property
    def quarter(self) -> Period:
        return Period(Frequency.QUARTERLY, self.announced.year, (self.announced.month - 1) // 3 + 1)


@dataclass(frozen=True)
class PolicyEvents:
    reviewed_on: dt.date
    coverage_start: dt.date
    events: tuple[PolicyEvent, ...]

    @property
    def first_quarter(self) -> Period:
        d = self.coverage_start
        return Period(Frequency.QUARTERLY, d.year, (d.month - 1) // 3 + 1)

    @property
    def last_complete_quarter(self) -> Period:
        """The newest quarter that had ended by the review date."""
        d = self.reviewed_on
        q = Period(Frequency.QUARTERLY, d.year, (d.month - 1) // 3 + 1)
        return q if q.end_date <= d else q.shift(-1)

    def net_rounds(self, quarter: Period) -> int:
        return sum(e.sign for e in self.events if e.quarter == quarter)


def _date(value: Any, field: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise CuratedError(f"{field}: {value!r} is not a date") from exc


def load_policy_events(content: bytes) -> PolicyEvents:
    try:
        raw = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CuratedError(f"event list is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
        raise CuratedError("event list needs reviewed_on, coverage_start and an events list")
    events = []
    for i, item in enumerate(raw["events"], 1):
        if not isinstance(item, dict):
            raise CuratedError(f"event {i} is not a mapping")
        direction = str(item.get("direction", "")).strip()
        if direction not in DIRECTIONS:
            raise CuratedError(f"event {i}: direction must be one of {sorted(DIRECTIONS)}, got {direction!r}")
        source = str(item.get("source", "")).strip()
        if not source.startswith("https://"):
            raise CuratedError(f"event {i}: every event needs an official https source link")
        instruments = item.get("instruments") or []
        if not instruments or not str(item.get("summary", "")).strip():
            raise CuratedError(f"event {i}: instruments and summary are required")
        events.append(PolicyEvent(_date(item.get("announced"), f"event {i} announced"), direction,
                                  tuple(str(x) for x in instruments), str(item["summary"]).strip(), source))
    parsed = PolicyEvents(_date(raw.get("reviewed_on"), "reviewed_on"),
                          _date(raw.get("coverage_start"), "coverage_start"),
                          tuple(sorted(events, key=lambda e: e.announced)))
    outside = [e.announced.isoformat() for e in parsed.events
               if not parsed.coverage_start <= e.announced <= parsed.reviewed_on]
    if outside:
        raise CuratedError(f"events {outside} fall outside coverage_start..reviewed_on")
    return parsed


def summarise(content: bytes) -> dict[str, Any]:
    parsed = load_policy_events(content)
    return {
        "title": "Singapore residential property market measures (hand-maintained from official releases)",
        "publisher": "MAS / MND / MOF / HDB releases, curated",
        "last_updated": parsed.reviewed_on.isoformat(),
        "coverage_start": parsed.coverage_start.isoformat(),
        "events": len(parsed.events),
    }
