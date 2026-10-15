"""Deterministic verification of the agent's proposals.

The agent's claims about a series (that it exists, is current, has a given unit
and frequency) are re-established here from the publisher's own responses, never
taken on trust. Two kinds of check run on every proposal:

* data checks: the series exists, is still published, has enough history, has
  the stated frequency and unit, and is not already in the registry;
* a relationship check: the same lead-lag correlation the report runs, against
  each target, compared with the effect the agent claims. Among targets with a
  declared positive or negative effect, it passes only when the data agree for at
  least one and contradict none; effects declared ambiguous are recorded, not
  tested, and pass. It informs the person
  deciding; it never rejects on its own (the registry keeps one predictor whose
  data contradict theory, for a documented reason).

Each proposal ends as one of

``accept``   every check passed; a registry draft is produced for human review
``review``   usable but with warnings (short history, unit wording, no SA, ...)
``reject``   does not exist, cannot be parsed, is stale, or duplicates the registry

and a draft registry entry is written for anything not rejected, with the fields
only a person should set (publication lag, neutral band, limitations) marked TODO.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from sgmacro.diagnostics import correlate
from sgmacro.parse import Extracted, ParseError, extract
from sgmacro.periods import Period, parse_period
from sgmacro.registry import DataGovRef, Frequency, Registry, SeriesSpec, SingStatRef
from sgmacro.validate import _parse_stamp
from sgmacro.sources.datagovsg import DataGovClient, summarise_metadata
from sgmacro.sources.singstat import SingStatClient

#: Generous publication lags for freshness screening of an unknown series.
SCREENING_LAG_DAYS = {Frequency.MONTHLY: 75, Frequency.QUARTERLY: 135, Frequency.ANNUAL: 300}
MIN_HISTORY_YEARS = 8


@dataclass
class Check:
    name: str
    severity: str  # pass | warn | fail
    message: str


@dataclass
class Verification:
    proposal: dict[str, Any]
    checks: list[Check] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    draft: dict[str, Any] | None = None

    @property
    def verdict(self) -> str:
        severities = {c.severity for c in self.checks}
        if "fail" in severities:
            return "reject"
        return "review" if "warn" in severities else "accept"

    def add(self, name: str, severity: str, message: str) -> None:
        self.checks.append(Check(name, severity, message))

    def to_json(self) -> dict[str, Any]:
        return {"label": self.proposal.get("label"), "verdict": self.verdict, "proposal": self.proposal,
                "observed": self.observed, "draft": self.draft,
                "checks": [{"check": c.name, "severity": c.severity, "message": c.message} for c in self.checks]}


def _norm_unit(text: str | None) -> str:
    t = (text or "").lower().replace("per cent", "percent").replace("%", "percent")
    return re.sub(r"[^a-z]", "", t)


def is_rate_unit(unit: str | None) -> bool:
    """Percent, per cent, % or percentage: a rate or share, changed in points, never in percent."""
    return "percent" in _norm_unit(unit)


def _proposal_filters(proposal: dict[str, Any]) -> dict[str, str] | None:
    raw = proposal.get("filters") or "{}"
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else None


def _registry_duplicate(registry: Registry, proposal: dict[str, Any]) -> str | None:
    """The registry series this proposal is the same series as, if any.

    Identity is the series, not the file: another property type from the same
    data.gov.sg dataset, or one component of a SingStat ratio, is a different series.
    Selectors are compared as the extractor reads them (a wide row label without its
    surrounding blanks, a SingStat series number as text), so a variant spelling of a
    registry series is not taken for a new one.
    """
    table = str(proposal.get("table_id") or "").strip().upper()
    series = str(proposal.get("series_no") or "").strip()
    dataset = str(proposal.get("dataset_id") or "").strip()
    row = str(proposal.get("value_column") or "").strip()
    for spec in registry.series:
        for ref in spec.refs():
            if isinstance(ref, SingStatRef) and proposal.get("source") == "singstat":
                if ref.table_id.upper() == table and str(ref.series) == series:
                    return spec.id
            if isinstance(ref, DataGovRef) and proposal.get("source") == "datagovsg":
                if ref.dataset_id != dataset:
                    continue
                if ref.layout == "wide" and (ref.row_label or "").strip() == row:
                    return spec.id
                if (ref.layout == "tidy" and ref.aggregate == "single"
                        and ref.value_column == proposal.get("value_column")
                        and ref.filters == _proposal_filters(proposal)):
                    return spec.id
    return None


def periods_behind(freq: Frequency, latest: Period, today: date) -> int:
    lag = timedelta(days=SCREENING_LAG_DAYS[freq])
    behind, candidate = 0, latest
    while (candidate := candidate.shift(1)).end_date + lag <= today:
        behind += 1
        if behind > 500:
            break
    return behind


def known_titles(registry: Registry, store_root) -> dict[str, str]:
    """Publisher titles of every object the registry reads, from the stored vintages.

    data.gov.sg republishes many SingStat tables under the same title, so a
    proposal can name a different id for a series the registry already has.
    """
    from sgmacro.store import SnapshotStore

    store = SnapshotStore(store_root)
    titles: dict[str, str] = {}
    for spec in registry.series:
        for ref in spec.refs():
            latest = store.latest(ref.object_key)
            title = (latest.upstream.get("title") if latest else None) or ""
            if title:
                titles.setdefault(title.strip().lower(), spec.id)
    return titles


def _canonical_ids(proposal: dict[str, Any]) -> dict[str, Any]:
    """The proposal with its source identifiers written as the series identity
    (:func:`sgmacro.agent.tools.candidate_key`) reads them, so what is requested,
    recorded and remembered is one series: a padded or numeric SingStat series
    number, or a padded table or dataset id, is not a different series, nor a
    missing one. ``replaces`` is a registry id, read without blanks here as by the
    tool and by adoption. The value column is left as written: only the table's
    layout decides whether it names a wide row (read without blanks) or a tidy
    column (read exactly)."""
    canonical = dict(proposal)
    if proposal.get("replaces"):
        canonical["replaces"] = str(proposal["replaces"]).strip()
    if proposal.get("source") == "singstat":
        canonical["table_id"] = str(proposal.get("table_id") or "").strip().upper()
        canonical["series_no"] = str(proposal.get("series_no") or "").strip()
    elif proposal.get("source") == "datagovsg":
        canonical["dataset_id"] = str(proposal.get("dataset_id") or "").strip()
    return canonical


def verify_proposal(proposal: dict[str, Any], registry: Registry, today: date, *,
                    singstat: SingStatClient, datagov: DataGovClient | None = None,
                    titles: dict[str, str] | None = None,
                    targets: list[tuple[SeriesSpec, pd.Series]] | None = None) -> Verification:
    proposal = _canonical_ids(proposal)
    v = Verification(proposal)
    try:
        freq = Frequency(proposal.get("frequency"))
    except ValueError:
        v.add("frequency", "fail", f"unknown frequency {proposal.get('frequency')!r}")
        return v

    duplicate = _registry_duplicate(registry, proposal)
    replaces = (proposal.get("replaces") or "").strip()
    if replaces and replaces not in {s.id for s in registry.series}:
        v.add("replaces", "warn", f"`replaces` names {replaces!r}, which is not a registry series")
    if duplicate and duplicate != replaces:
        v.add("duplicate", "fail", f"already in the registry as {duplicate!r}")

    try:
        if proposal.get("source") == "singstat":
            extracted, upstream = _singstat(proposal, freq, singstat, v)
        elif proposal.get("source") == "datagovsg" and datagov is not None:
            extracted, upstream = _datagov(proposal, freq, datagov, v)
        else:
            v.add("source", "fail", f"unsupported source {proposal.get('source')!r}")
            return v
    except Exception as exc:  # noqa: BLE001 - any failure to resolve is a rejection
        v.add("resolve", "fail", f"{type(exc).__name__}: {exc}")
        return v
    if extracted is None:
        return v
    mirror = (titles or {}).get((upstream.get("title") or "").strip().lower())
    if mirror and mirror != replaces and not duplicate:
        v.add("duplicate", "warn", f"same publication title as the table behind registry series {mirror!r}; "
                                   "likely another channel's copy of data already covered")

    # An annual figure's reference date (end-June vs end-December) decides both its
    # publication lag and which quarter it sits in for the lead-lag check.
    reference_month = None
    if freq is Frequency.ANNUAL:
        raw = proposal.get("reference_month")
        reference_month = raw if isinstance(raw, int) and 1 <= raw <= 12 else None
        v.observed["reference_month"] = reference_month
        if reference_month is None:
            v.add("reference_month", "warn", "annual series with no stated reference month: lag and lead-lag "
                                             "assume end-December; confirm the month before adopting")

    latest = extracted.latest()
    populated = extracted.populated
    if latest is None:
        v.add("data", "fail", "no populated observations")
        return v
    first = extracted.period(populated["ordinal"].iloc[0])
    years = len(populated) / freq.periods_per_year
    behind = periods_behind(freq, latest[0], today)
    v.observed.update({"first": first.label, "latest": latest[0].label, "latest_value": latest[1],
                       "observations": int(len(populated)), "min": float(populated["value"].min()),
                       "max": float(populated["value"].max()), "periods_behind": behind})
    if behind >= 2:
        v.add("freshness", "fail", f"latest observation {latest[0].label} is {behind} periods behind what a "
                                   f"current {freq.value} series would show by {today}")
    elif behind == 1:
        v.add("freshness", "warn", f"latest observation {latest[0].label} is one period behind; check the release")
    else:
        v.add("freshness", "pass", f"latest observation {latest[0].label} is current")
    if years < MIN_HISTORY_YEARS:
        v.add("history", "warn", f"only {years:.1f} years of history ({first.label} onwards); lead-lag "
                                 "estimates will be weak")
    else:
        v.add("history", "pass", f"{years:.1f} years of history from {first.label}")

    for market in proposal.get("markets") or []:
        if proposal.get(f"effect_on_{market}") in (None, "not_applicable"):
            v.add("effects", "warn", f"claims market {market!r} but gives no expected effect on it")
    stamp = _parse_stamp(upstream.get("last_updated"))
    if stamp is not None:
        v.observed["update_stamp"] = stamp.isoformat()
        v.observed["observed_lag_days"] = (stamp - latest[0].reference_date(reference_month)).days
    if targets:
        relationship_check(v, proposal, freq, extracted, targets, reference_month)
    v.draft = draft_entry(proposal, freq, upstream, v.observed)
    return v


def candidate_changes(extracted: Extracted, freq: Frequency, unit: str | None,
                      reference_month: int | None = None) -> pd.Series:
    """Year-on-year change of a candidate, on quarterly ordinals (pp for rates); an
    annual figure sits in the quarter of its reference date, as in the report."""
    values = extracted.values()
    values = values.reindex(range(int(values.index.min()), int(values.index.max()) + 1)).astype("float64")
    k = freq.periods_per_year
    if is_rate_unit(unit):
        changes = values - values.shift(k)
    else:
        base = values.shift(k)
        changes = (values / base.where(base > 0) - 1.0) * 100.0
    if freq is Frequency.QUARTERLY:
        return changes.dropna()
    mapped: dict[int, float] = {}
    for ordinal, value in changes.dropna().items():
        mapped[Period.from_ordinal(freq, int(ordinal)).to_quarter(reference_month).ordinal] = float(value)
    return pd.Series(mapped, dtype="float64").sort_index()


def relationship_check(v: Verification, proposal: dict[str, Any], freq: Frequency, extracted: Extracted,
                       targets: list[tuple[SeriesSpec, pd.Series]], reference_month: int | None = None) -> None:
    """Does the candidate move with the targets the way the agent says it should?"""
    x = candidate_changes(extracted, freq, v.observed.get("unit") or proposal.get("unit"), reference_month)
    sign = {"positive": 1, "negative": -1, "ambiguous": 0}
    rows = []
    for spec, values in targets:
        market = spec.market.value  # type: ignore[union-attr]
        declared = proposal.get(f"effect_on_{market}") if market in (proposal.get("markets") or []) else None
        diag = correlate(proposal.get("label") or "candidate", x, spec, values, sign.get(declared))
        best = diag.best
        rows.append({"target": spec.short_label, "market": market, "declared_effect": declared,
                     "best_lag_quarters": best.lag if best else None,
                     "r": round(best.r, 3) if best else None, "pairs": best.pairs if best else
                     max((c.pairs for c in diag.correlations), default=0),
                     "agreement": diag.agreement})
    v.observed["relationship"] = rows
    tested = [r for r in rows if r["declared_effect"] in ("positive", "negative")]
    describe = "; ".join(f"{r['target']} r={r['r']:+.2f} at {r['best_lag_quarters']}q ({r['agreement']})"
                         for r in tested if r["r"] is not None)
    agrees = any(r["agreement"] == "agrees" for r in tested)
    disagrees = any(r["agreement"] == "disagrees" for r in tested)
    if not tested:
        v.add("relationship", "pass", "effects declared ambiguous; correlations recorded for reference only")
    elif agrees and disagrees:
        v.add("relationship", "warn", f"mixed evidence: the data agree with the declared effect for some targets "
                                      f"and contradict it for others: {describe}; adopt only with an explanation")
    elif agrees:
        v.add("relationship", "pass", f"data consistent with the declared effect: {describe}")
    elif disagrees:
        v.add("relationship", "warn", f"data contradict the declared effect: {describe}; "
                                      "adopt only with an explanation")
    elif all(r["agreement"] == "insufficient data" for r in tested):
        v.add("relationship", "warn", "too little overlapping history with the targets to test the declared effect")
    else:
        v.add("relationship", "warn", f"no measurable relationship with the targets (|r| < 0.2): {describe}")


def _singstat(proposal: dict[str, Any], freq: Frequency, client: SingStatClient, v: Verification):
    table, series = proposal["table_id"], str(proposal["series_no"])
    records = json.loads(client.metadata(table))["Data"]["records"]
    rows = {str(r.get("seriesNo")): r for r in records.get("row") or []}
    if series not in rows:
        v.add("exists", "fail", f"{table} has no series {series!r}")
        return None, {}
    v.add("exists", "pass", f"{table} series {series}: {rows[series].get('rowText')}")
    upstream_freq = (records.get("frequency") or "").lower()
    if upstream_freq != freq.upstream_label.lower():
        v.add("frequency", "fail", f"publisher reports {records.get('frequency')!r}, proposal says {freq.value}")
        return None, {}
    unit = rows[series].get("uoM")
    if _norm_unit(unit) not in _norm_unit(proposal.get("unit")) and _norm_unit(proposal.get("unit")) not in _norm_unit(unit):
        v.add("unit", "warn", f"proposal says {proposal.get('unit')!r}; publisher's unit is {unit!r} (the publisher's is used)")
    else:
        v.add("unit", "pass", f"unit {unit!r}")
    adjustment = records.get("adjustmentType") or ""
    if freq is not Frequency.ANNUAL and adjustment.lower() == "non-seasonally adjusted":
        v.add("seasonality", "warn", "not seasonally adjusted: use year-on-year changes only")
    ref = SingStatRef(table_id=table, series=series, unit_of_measure=unit or "unknown")
    content = client.tabledata(table, [series])
    upstream = {"publisher": records.get("dataSource"), "title": records.get("title"), "unit": unit,
                "adjustment": adjustment, "last_updated": records.get("dataLastUpdated"),
                "link": ref.link, "identifier": ref.identifier}
    v.observed.update({"publisher": records.get("dataSource"), "unit": unit, "adjustment": adjustment or "n/a",
                       "last_updated": records.get("dataLastUpdated")})
    return extract(ref, content, freq, {}), upstream


def _label_frequency(label: str) -> Frequency | None:
    """The frequency a period label is written in ('2026 2Q', '2026Jun', '2026')."""
    for freq in Frequency:
        try:
            parse_period(label, freq)
        except Exception:  # noqa: BLE001 - any parse failure means "not this frequency"
            continue
        return freq
    return None


def _datagov(proposal: dict[str, Any], freq: Frequency, client: DataGovClient, v: Verification):
    dataset = proposal["dataset_id"]
    filters = proposal.get("filters") or "{}"
    try:
        filters = json.loads(filters) if isinstance(filters, str) else dict(filters)
    except json.JSONDecodeError:
        v.add("filters", "fail", f"filters are not a JSON object: {filters!r}")
        return None, {}
    metadata = client.metadata(dataset)
    upstream = summarise_metadata(metadata)
    content = client.download(dataset)
    # Read as CSV, as the extractor does: a quoted header ("2026 2Q") must give
    # the same labels here as there, or a wrong frequency looks like bad data.
    header = [h.strip() for h in next(csv.reader(io.StringIO(content.decode("utf-8-sig"))), [])]
    if header and header[0].strip() == "DataSeries":
        # One row per series: the agent's value_column names the row. Added after
        # the first repair session, where the agent found the right series in this
        # layout and the verifier could not read it.
        v.observed["layout"] = "wide"          # recorded before any failure: it is part of the identity
        published = _label_frequency(next((h.strip() for h in header[1:] if h.strip()), ""))
        ref = DataGovRef(dataset_id=dataset, layout="wide", row_label=proposal["value_column"])
    else:
        period_column = next((c for c in ("quarter", "month", "year") if c in header), None)
        if period_column is None:
            v.add("layout", "fail", f"no quarter/month/year column in {header[:6]}")
            return None, {}
        v.observed["layout"] = "tidy"
        published = {"quarter": Frequency.QUARTERLY, "month": Frequency.MONTHLY, "year": Frequency.ANNUAL}[period_column]
        ref = DataGovRef(dataset_id=dataset, layout="tidy", period_column=period_column,
                         value_column=proposal["value_column"], filters={str(k): str(x) for k, x in filters.items()})
    if published is not None and published is not freq:
        # A wrong frequency in the proposal, not a fault in the data: reported as
        # such (the parser would call it a parse failure), so it can be corrected.
        v.add("frequency", "fail", f"the table holds {published.value} periods, proposal says {freq.value}")
        return None, {}
    try:
        extracted = extract(ref, content, freq)
    except ParseError as exc:
        v.add("parse", "fail", str(exc))
        return None, {}
    v.add("exists", "pass", f"{dataset}: {upstream.get('title')}")
    v.add("unit", "warn", "data.gov.sg exposes no unit metadata; confirm the unit by hand")
    upstream.update({"link": ref.link, "identifier": ref.identifier, "layout": ref.layout,
                     "period_column": ref.period_column})
    v.observed.update({"publisher": upstream.get("publisher"), "last_updated": upstream.get("last_updated")})
    return extracted, upstream


def draft_entry(p: dict[str, Any], freq: Frequency, upstream: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    unit = (upstream.get("unit") or p.get("unit") or "").strip()
    kind = "rate" if is_rate_unit(unit) else ("index" if "index" in unit.lower() else "level")
    if kind == "rate":
        transforms = ["mom_pp" if freq is Frequency.MONTHLY else "qoq_pp", "yoy_pp"] if freq is not Frequency.ANNUAL else ["yoy_pp"]
        headline = "yoy_pp"
        band = [math.floor(obs["min"]) - 5, math.ceil(obs["max"]) + 5]
    else:
        transforms = (["mom" if freq is Frequency.MONTHLY else "qoq", "yoy"] if freq is not Frequency.ANNUAL else ["yoy"])
        headline = "yoy"
        band = [round(obs["min"] * 0.5, 4), round(obs["max"] * 2, 4)]
    slug = re.sub(r"[^a-z0-9]+", "_", (p.get("label") or "candidate").lower()).strip("_")[:40] or "candidate"
    markets = [m for m in p.get("markets") or [] if p.get(f"effect_on_{m}") not in (None, "not_applicable")]
    if p.get("source") == "singstat":
        source = {"source": "singstat", "table_id": p["table_id"], "series": str(p["series_no"]),
                  "unit_of_measure": unit}
    elif upstream.get("layout") == "wide":
        source = {"source": "datagovsg", "dataset_id": p["dataset_id"], "layout": "wide",
                  "row_label": str(p["value_column"]).strip()}
    else:
        source = {"source": "datagovsg", "dataset_id": p["dataset_id"], "layout": "tidy",
                  "period_column": upstream.get("period_column") or "quarter",
                  "value_column": p["value_column"], "filters": json.loads(p.get("filters") or "{}")}
    return {
        "id": slug, "label": p.get("label"), "short_label": p.get("label"), "role": "predictor",
        "markets": markets, "channel": p.get("channel"), "publisher": upstream.get("publisher") or "TODO",
        "source": source,
        "definition": f"TODO (publisher title: {upstream.get('title')})",
        "unit": unit or "TODO", "unit_kind": kind, "frequency": freq.value,
        "seasonal_adjustment": "seasonally_adjusted" if (upstream.get("adjustment") or "").lower() == "seasonally adjusted"
        else "not_applicable",
        "update_frequency": "TODO", "publication_lag_days": "TODO (cross-check against the update stamp)",
        "plausible_range": band, "transforms": transforms, "headline": headline,
        "effects": {m: p.get(f"effect_on_{m}") for m in markets},
        "neutral_band": "TODO", "expected_lead_quarters": "TODO",
        "rationale": p.get("rationale"), "limitations": "TODO (reviewer)",
        # An annual figure's reference date (end-June, end-December) is not in the
        # metadata and decides when it is publishable, so a person must state it.
        **({"reference_month": obs.get("reference_month")
            or "TODO (month the annual figure refers to, 1-12; e.g. 6 for end-June)"}
           if freq is Frequency.ANNUAL else {}),
    }
