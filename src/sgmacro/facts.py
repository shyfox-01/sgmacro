"""The fact sheet: every number the report prints, with its lineage.

A :class:`Fact` is the unit of traceability. The report template only ever prints
``fact.display``; the LLM summary may only cite facts by id; ``sgmacro trace``
resolves an id back to raw-snapshot cells and recomputes it. So "where did this
number come from" has one answer for every figure, whoever wrote the sentence
around it.

Ids are deterministic in (series, period, kind), so the same figure keeps the
same id across runs and a reviewer can cite ``F:1a2b3c4d`` from an old report.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from sgmacro.transform import Change, ObsRef

CHANGE_SUFFIX = {"pct": "%", "pp": " pp", "bp": " bp", "count": " rounds"}


def fact_id(*parts: object) -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"F:{digest[:8]}"


def signed(value: float, decimals: int) -> str:
    """``+1.2`` / ``-0.4`` / ``0.0``: never ``-0.0``, never ``+0.0``."""
    rounded = round(value, decimals)
    if rounded == 0:
        return f"{0:.{decimals}f}"
    return f"{rounded:+.{decimals}f}"


def format_change(value: float | None, unit: str) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if unit == "count":
        n = int(round(value))
        return f"{signed(float(n), 0)} round{'' if abs(n) == 1 else 's'}"
    decimals = {"bp": 0, "pp": 2}.get(unit, 1)
    return f"{signed(value, decimals)}{CHANGE_SUFFIX[unit]}"


def format_level(value: float | None, decimals: int) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


@dataclass
class Fact:
    fact_id: str
    kind: str
    series_id: str
    label: str
    period: str | None
    value: float | None
    unit: str
    display: str
    formula: str
    inputs: list[ObsRef] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> str | None:
        if self.kind not in ("change", "real_change") or self.value is None:
            return None
        if self.display.startswith("+"):
            return "up"
        if self.display.startswith("-"):
            return "down"
        return "flat"

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["inputs"] = [asdict(i) for i in self.inputs]
        out["value"] = None if self.value is None else round(self.value, 10)
        return out


class FactSheet:
    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}

    def add(self, fact: Fact) -> Fact:
        existing = self._facts.get(fact.fact_id)
        if existing is not None and existing.value != fact.value:
            raise ValueError(f"fact id collision for {fact.fact_id}: {existing.label} vs {fact.label}")
        self._facts[fact.fact_id] = fact
        return fact

    def get(self, fid: str) -> Fact:
        return self._facts[fid]

    def __contains__(self, fid: str) -> bool:
        return fid in self._facts

    def __iter__(self):
        return iter(sorted(self._facts.values(), key=lambda f: (f.series_id, f.kind, f.fact_id)))

    def __len__(self) -> int:
        return len(self._facts)

    # -- constructors -------------------------------------------------------
    def level(self, spec_id: str, label: str, obs: ObsRef, unit: str, decimals: int) -> Fact:
        return self.add(Fact(
            fact_id=fact_id(spec_id, obs.period, "level"),
            kind="level", series_id=spec_id, label=f"{label}, level", period=obs.period,
            value=obs.value, unit=unit, display=format_level(obs.value, decimals),
            formula="v[t] as published", inputs=[obs]))

    def change(self, label: str, change: Change, kind: str = "change") -> Fact:
        name = "real (CPI-deflated) year-on-year change" if kind == "real_change" else change.human
        return self.add(Fact(
            fact_id=fact_id(change.series_id, change.period.label, kind, change.transform.value),
            kind=kind, series_id=change.series_id, label=f"{label}, {name}",
            period=change.period.label, value=change.value, unit=change.unit,
            display=format_change(change.value, change.unit), formula=change.formula,
            inputs=list(change.inputs), note=change.note,
            extra={"transform": change.transform.value}))

    def percentile(self, spec_id: str, label: str, period: str, value: float | None,
                   history_start: str | None, headline_fact: str) -> Fact:
        display = "n/a" if value is None else ordinal_suffix(int(round(value)))
        return self.add(Fact(
            fact_id=fact_id(spec_id, period, "percentile"),
            kind="percentile", series_id=spec_id, label=f"{label}, historical percentile of the headline change",
            period=period, value=value, unit="percentile", display=display,
            formula="share of headline-change history at or below the latest value",
            depends_on=[headline_fact], extra={"history_start": history_start}))

    def to_json(self) -> list[dict[str, Any]]:
        return [f.to_json() for f in self]
