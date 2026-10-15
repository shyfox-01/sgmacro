"""End-to-end orchestration: as-of date in, report bundle out.

The order is fixed and every stage after retrieval is pure (no network, no
clock), which is what makes a repeated request reproduce byte-identical outputs:

    fetch      select (or download) one snapshot vintage per upstream object
    parse      raw bytes -> tidy observations with raw-cell locators
    validate   contract, gate, staleness, lag, cross-source, revisions, ...
    transform  changes on *gated* observations only
    signals    declared effect x observed direction, per market
    diagnose   lead-lag correlations against the targets
    facts      every printable number, with lineage
    narrate    LLM executive summary, claim-checked (optional, cached)
    render     Markdown and HTML report + charts + processed tables + run manifest

Nothing downstream of ``validate`` ever sees an observation the publication-lag
gate withheld, so no observation published after the as-of date is used. Values
are as revised in the vintage read: the newest vintage retrieved by the as-of
date, or, if none was retrieved that early, the oldest one, which the report
flags because later revisions may show through.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from sgmacro import __version__
from sgmacro.credentials import redact
from sgmacro.diagnostics import Diagnostic, lead_lag
from sgmacro.facts import Fact, FactSheet, fact_id
from sgmacro.fetch import Fetcher, ObjectState, plan
from sgmacro.parse import Extracted, ParseError, extract
from sgmacro.registry import Registry, Role, SeriesSpec, SourceRef, load_registry
from sgmacro.signals import Signal, signals_for
from sgmacro.store import SnapshotStore, Vintage, sha256
from sgmacro.transform import Change, SeriesChanges, compute_changes, real_yoy
from sgmacro.validate import SeriesResult, SourceView, validate_series

if TYPE_CHECKING:  # the LLM layer is imported only when a run drafts a summary
    from sgmacro.llm.summary import Summary

DEFLATOR_ID = "cpi_all_items"


@dataclass
class RunConfig:
    as_of: date
    registry_path: Path = Path("config/sources.yaml")
    store_root: Path = Path("data/snapshots")
    processed_root: Path = Path("data/processed")
    runs_root: Path = Path("runs")
    reports_root: Path = Path("reports")
    llm_cache: Path = Path("llm_cache")
    llm: str = "auto"
    refetch: bool = False
    offline: bool = False


@dataclass
class SeriesOutcome:
    spec: SeriesSpec
    result: SeriesResult
    changes: SeriesChanges | None = None
    signals: list[Signal] = field(default_factory=list)
    real: Change | None = None
    facts: dict[str, str] = field(default_factory=dict)
    """Role of each fact for the template: ``level``, ``headline``, ``<transform>``, ..."""


@dataclass
class RunResult:
    config: RunConfig
    registry: Registry
    run_id: str
    states: dict[str, ObjectState]
    outcomes: dict[str, SeriesOutcome]
    diagnostics: list[Diagnostic]
    facts: FactSheet
    summary: Summary | None = None
    outputs: dict[str, str] = field(default_factory=dict)

    def by_role(self, role: Role) -> list[SeriesOutcome]:
        return [o for o in self.outcomes.values() if o.spec.role is role]

    def reportable(self, role: Role) -> list[SeriesOutcome]:
        return [o for o in self.by_role(role) if o.result.reportable]

    @property
    def excluded(self) -> list[SeriesOutcome]:
        return [o for o in self.outcomes.values() if not o.result.reportable]

    def signals(self) -> list[Signal]:
        return [s for o in self.reportable(Role.PREDICTOR) for s in o.signals]


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def fetch_stage(config: RunConfig, registry: Registry, log: Callable[[str], None]) -> dict[str, ObjectState]:
    store = SnapshotStore(config.store_root)
    fetcher = Fetcher(store, log=log)
    return fetcher.resolve(plan(registry), config.as_of, refetch=config.refetch, offline=config.offline)


class _ExtractCache:
    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[Extracted | None, str | None]] = {}

    def get(self, ref: SourceRef, vintage: Vintage, spec: SeriesSpec) -> tuple[Extracted | None, str | None]:
        key = (ref.model_dump_json(), vintage.object_key, vintage.vintage_id, spec.frequency)
        if key not in self._cache:
            try:
                self._cache[key] = (extract(ref, vintage.data_bytes(), spec.frequency, vintage.upstream), None)
            except ParseError as exc:
                self._cache[key] = (None, str(exc))
        return self._cache[key]


def analyse(config: RunConfig, registry: Registry, states: dict[str, ObjectState]) -> RunResult:
    """Every stage after retrieval. Pure: same inputs, same result."""
    store = SnapshotStore(config.store_root)
    cache = _ExtractCache()

    def view(ref: SourceRef, spec: SeriesSpec) -> SourceView:
        state = states.get(ref.object_key)
        if state is None or state.vintage is None:
            return SourceView(ref, state, None, state.error if state else "not fetched")
        extracted, error = cache.get(ref, state.vintage, spec)
        return SourceView(ref, state, extracted, error)

    outcomes: dict[str, SeriesOutcome] = {}
    for spec in registry.series:
        primary = view(spec.source, spec)
        previous = None
        if primary.vintage is not None:
            prior_vintage = store.previous(primary.vintage)
            if prior_vintage is not None:
                previous, _ = cache.get(spec.source, prior_vintage, spec)
        cross = view(spec.cross_check.ref, spec) if spec.cross_check else None
        result = validate_series(spec, primary, as_of=config.as_of, previous=previous, cross=cross)
        outcome = SeriesOutcome(spec, result)
        if result.reportable:
            outcome.changes = compute_changes(result)
            if spec.role is Role.PREDICTOR:
                outcome.signals = signals_for(spec, outcome.changes)
        outcomes[spec.id] = outcome

    deflator = outcomes.get(DEFLATOR_ID)
    for outcome in outcomes.values():
        if outcome.spec.role is Role.TARGET and outcome.changes and deflator and deflator.result.reportable:
            outcome.real = real_yoy(outcome.result, outcome.changes, deflator.result)

    diagnostics: list[Diagnostic] = []
    targets = [o for o in outcomes.values() if o.spec.role is Role.TARGET and o.result.reportable]
    for predictor in (o for o in outcomes.values() if o.spec.role is Role.PREDICTOR and o.changes):
        for target in targets:
            diagnostics.append(lead_lag(predictor.changes, target.spec, target.result.gated_series()))

    facts = build_facts(outcomes, diagnostics)
    run_id = run_identity(config.as_of, registry, states)
    return RunResult(config, registry, run_id, states, outcomes, diagnostics, facts)


def build_facts(outcomes: dict[str, SeriesOutcome], diagnostics: list[Diagnostic]) -> FactSheet:
    sheet = FactSheet()
    for outcome in outcomes.values():
        spec, changes = outcome.spec, outcome.changes
        if changes is None:
            continue
        level = sheet.level(spec.id, spec.short_label, changes.latest, spec.unit, spec.decimals)
        outcome.facts["level"] = level.fact_id
        for change in changes.changes:
            f = sheet.change(spec.short_label, change)
            outcome.facts[change.transform.value] = f.fact_id
        outcome.facts["headline"] = outcome.facts[spec.headline.value]
        pct = sheet.percentile(spec.id, spec.short_label, changes.headline.period.label,
                               changes.percentile, changes.history_start, outcome.facts["headline"])
        outcome.facts["percentile"] = pct.fact_id
        if outcome.real is not None:
            outcome.facts["real_yoy"] = sheet.change(spec.short_label, outcome.real, kind="real_change").fact_id
        for signal in outcome.signals:
            f = sheet.add(Fact(
                fact_id=fact_id(spec.id, changes.headline.period.label, "signal", signal.market.value),
                kind="signal", series_id=spec.id,
                label=f"{spec.short_label}: reading for {signal.market.value}",
                period=changes.headline.period.label, value=float(signal.direction),
                unit="signal", display=signal.reading,
                formula="direction(headline, neutral band) x declared effect",
                depends_on=[outcome.facts["headline"]],
                extra={"market": signal.market.value, "effect": signal.effect.value,
                       "neutral_band": spec.neutral_band}))
            outcome.facts[f"signal_{signal.market.value}"] = f.fact_id
    for diag in diagnostics:
        for corr in diag.correlations:
            if corr.r is None:
                continue
            sheet.add(Fact(
                fact_id=fact_id(diag.predictor_id, diag.target_id, "corr", corr.lag),
                kind="correlation", series_id=diag.predictor_id,
                label=f"correlation of {diag.predictor_id} headline (lead {corr.lag}q) with {diag.target_id} YoY",
                period=None, value=corr.r, unit="r", display=f"{corr.r:+.2f}",
                formula="pearson(x[q-k], y[q]) over gated quarterly observations",
                extra={"lag": corr.lag, "pairs": corr.pairs, "target": diag.target_id}))
    return sheet


def run_identity(as_of: date, registry: Registry, states: dict[str, ObjectState],
                 summary_identity: str | None = None) -> str:
    """Hash of everything the report's content depends on: the request, the
    registry, each object's fetch state as the report shows it (vintage read, or
    none; degraded fallback; the error text) and, once drafted, the summary. Two
    runs share an id only if they produce the same report."""
    objects = sorted(f"{k}@{s.vintage.vintage_id if s.vintage else 'none'}{'+degraded' if s.degraded else ''}"
                     + (f"!{s.error}" if s.error else "")
                     for k, s in states.items())
    parts = [as_of.isoformat(), registry.fingerprint(), objects, __version__]
    if summary_identity is not None:
        parts.append(summary_identity)
    blob = json.dumps(parts)
    return f"{as_of.isoformat()}_{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# full run
# ---------------------------------------------------------------------------

def run(config: RunConfig, *, log: Callable[[str], None] = print, registry: Registry | None = None,
        states: dict[str, ObjectState] | None = None) -> RunResult:
    started = datetime.now(timezone.utc)
    registry = registry or load_registry(config.registry_path)
    states = states if states is not None else fetch_stage(config, registry, log)
    result = analyse(config, registry, states)

    from sgmacro.llm.summary import summarise  # local import: optional heavy deps
    result.summary = summarise(result, backend_preference=config.llm, cache_dir=config.llm_cache, log=log,
                               offline=config.offline)
    result.run_id = run_identity(config.as_of, registry, states, result.summary.identity)

    from sgmacro.report.render import write_outputs
    write_outputs(result)
    _log_execution(result, started)
    return result


def _log_execution(result: RunResult, started: datetime) -> None:
    """Append-only record of each invocation. The manifest itself carries no
    wall-clock time, so repeated runs leave it byte-identical."""
    run_dir = result.config.runs_root / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "started_at": started.replace(microsecond=0).isoformat(),
        "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "as_of": result.config.as_of.isoformat(),
        "refetch": result.config.refetch,
        "llm": result.config.llm,
        "downloads": sum(1 for s in result.states.values() if s.downloaded),
        "new_vintages": sum(1 for s in result.states.values() if s.created),
        "degraded": sorted(k for k, s in result.states.items() if s.degraded),
        "summary_source": result.summary.source if result.summary else None,
        "summary_from_cache": result.summary.from_cache if result.summary else None,
        "summary_notes": [redact(n) for n in result.summary.notes + result.summary.skipped_backends
                          + result.summary.execution_notes]
                         if result.summary else [],
    }
    with (run_dir / "executions.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# processed tables
# ---------------------------------------------------------------------------

def table_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _cell(row.get(k)) for k in columns})
    return buffer.getvalue()


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return "" if value != value else repr(round(value, 10))
    return "" if value is None else value


def processed_tables(result: RunResult) -> dict[str, str]:
    observations, validation, changes, signals, diagnostics = [], [], [], [], []
    for outcome in result.outcomes.values():
        r = outcome.result
        if r.observations is not None and r.vintage is not None:
            for row in r.observations.itertuples(index=False):
                observations.append({
                    "series_id": outcome.spec.id, "role": outcome.spec.role.value,
                    "period": row.period, "value": None if row.value != row.value else float(row.value),
                    "admitted": bool(row.admitted), "publishable_on": row.publishable_on,
                    "unit": outcome.spec.unit, "vintage_id": r.vintage.vintage_id,
                    "object": r.vintage.object_key, "locator": row.locator,
                })
        for finding in r.findings:
            validation.append({"series_id": outcome.spec.id, "verdict": r.verdict, "check": finding.check,
                               "severity": finding.severity.value, "message": finding.message})
        for s in outcome.signals:
            signals.append({"series_id": s.series_id, "market": s.market.value, "effect": s.effect.value,
                            "direction": s.direction, "reading": s.reading,
                            "fact_id": outcome.facts.get(f"signal_{s.market.value}")})
    for fact in result.facts:
        if fact.kind in ("level", "change", "real_change", "percentile"):
            changes.append({
                "fact_id": fact.fact_id, "series_id": fact.series_id, "kind": fact.kind,
                "transform": fact.extra.get("transform", ""), "period": fact.period, "value": fact.value,
                "unit": fact.unit, "display": fact.display, "formula": fact.formula,
                "inputs": "; ".join(f"{i.series_id}:{i.period}={i.value}" for i in fact.inputs),
            })
    for d in result.diagnostics:
        for c in d.correlations:
            diagnostics.append({"predictor_id": d.predictor_id, "target_id": d.target_id,
                                "market": d.market.value, "lag_quarters": c.lag, "r": c.r, "pairs": c.pairs,
                                "in_declared_window": d.in_window(c.lag),
                                "verdict_lag": d.best.lag if d.best else None,
                                "expected_sign": d.expected_sign, "agreement": d.agreement})
    series = [{
        "series_id": o.spec.id, "role": o.spec.role.value, "label": o.spec.label, "publisher": o.spec.publisher,
        "source": o.spec.source.source, "identifier": o.spec.source.identifier, "link": o.spec.source.link,
        "definition": " ".join(o.spec.definition.split()), "unit": o.spec.unit, "unit_kind": o.spec.unit_kind.value,
        "observation_frequency": o.spec.frequency.value,
        "seasonal_adjustment": o.spec.seasonal_adjustment.value, "update_frequency": o.spec.update_frequency,
        "publication_lag_days": o.spec.publication_lag_days, "reference_month": o.spec.reference_month,
        "verdict": o.result.verdict, "latest_period": o.result.latest[0].label if o.result.latest else None,
        "as_of": result.config.as_of.isoformat(),
    } for o in result.outcomes.values()]
    return {
        "series.csv": table_csv(series, list(series[0]) if series else []),
        "observations.csv": table_csv(observations, ["series_id", "role", "period", "value", "admitted",
                                                     "publishable_on", "unit", "vintage_id", "object", "locator"]),
        "facts.csv": table_csv(changes, ["fact_id", "series_id", "kind", "transform", "period", "value",
                                         "unit", "display", "formula", "inputs"]),
        "signals.csv": table_csv(signals, ["series_id", "market", "effect", "direction", "reading", "fact_id"]),
        "validation.csv": table_csv(validation, ["series_id", "verdict", "check", "severity", "message"]),
        "diagnostics.csv": table_csv(diagnostics, ["predictor_id", "target_id", "market", "lag_quarters", "r",
                                                   "pairs", "in_declared_window", "verdict_lag", "expected_sign",
                                                   "agreement"]),
        "facts.json": json.dumps(result.facts.to_json(), indent=1, ensure_ascii=False) + "\n",
    }


def manifest(result: RunResult) -> dict[str, Any]:
    """The run record. Deliberately free of wall-clock time (see executions.jsonl)."""
    summary = result.summary
    return {
        "schema": 2,
        "run_id": result.run_id,
        "as_of": result.config.as_of.isoformat(),
        "pipeline_version": __version__,
        "registry": {"version": result.registry.version, "fingerprint": result.registry.fingerprint()},
        "vintages": {
            key: {
                **(state.vintage.summary() if state.vintage else {}),
                "after_as_of": bool(state.selection and state.selection.after_as_of),
                "degraded": state.degraded,
                "error": state.error,
            }
            for key, state in sorted(result.states.items())
        },
        "series": [o.result.to_json() for o in result.outcomes.values()],
        "facts": len(result.facts),
        "summary": summary.to_json() if summary is not None else None,
        "outputs": dict(sorted(result.outputs.items())),
    }


def file_digest(content: str | bytes) -> str:
    return sha256(content.encode() if isinstance(content, str) else content)
