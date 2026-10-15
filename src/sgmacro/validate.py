"""Validation: decide what is fit to report, and say why when it is not.

Checks run per *series*, never per dataset. The headline case for that rule is
the URA rental-index file: the dataset is refreshed every quarter, yet its
"All Residential" slice stopped in 2021-Q2. A staleness check on the dataset
sees a fresh file and passes; only a check on the selected series catches it.

In the order they matter:

contract     The publisher's own metadata must agree with the registry on unit
             of measure, frequency and seasonal adjustment. A mismatch means the
             registry describes a different series from the one being read, so
             every downstream number would be mislabelled. ERROR.
gate         An observation is admitted only once its reference date plus the
             declared publication lag has passed at the as-of date. This is what
             keeps look-ahead bias out of a historical run.
staleness    Measured from the newest *observation*, not the upstream's update
             stamp, which can be refreshed with no new data behind it. Years
             behind is ERROR (discontinued); a release or two behind is WARN.
declared lag The observed lag (update stamp minus reference date) must not exceed
             the declared one, or the gate could admit a figure before it existed.
cross-source The same series published elsewhere must agree within rounding, and
             the fresher publication is noted.
revisions    Values are compared point by point with the previous vintage.
plausibility Levels inside the declared band; recent moves not extreme relative
             to the series' own history.

A series that fails an ERROR check is excluded from the report *and listed with
its reason*; silently dropping it would hide that a declared predictor is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from sgmacro.fetch import ObjectState
from sgmacro.parse import Extracted
from sgmacro.periods import Period
from sgmacro.registry import (Adjustment, CuratedRef, DataGovRef, Frequency, SeriesSpec, SingStatRef, SourceRef,
                               UnitKind)
from sgmacro.store import Vintage


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class Status(str, Enum):
    OK = "ok"
    OK_WITH_WARNINGS = "ok_with_warnings"
    REJECTED = "rejected"

    @property
    def reportable(self) -> bool:
        return self is not Status.REJECTED


#: Periods behind schedule beyond which a series is treated as discontinued.
DISCONTINUED_AFTER = {Frequency.MONTHLY: 24, Frequency.QUARTERLY: 8, Frequency.ANNUAL: 3}
#: Robust z-score above which a recent period-on-period move is flagged.
UNUSUAL_MOVE_Z = 8.0
UNUSUAL_MOVE_WINDOW = 4


@dataclass
class Finding:
    check: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"check": self.check, "severity": self.severity.value,
                "message": self.message, "details": _jsonable(self.details)}


@dataclass
class SourceView:
    """One source reference as read at its selected vintage."""

    ref: SourceRef
    state: ObjectState | None
    extracted: Extracted | None = None
    error: str | None = None

    @property
    def vintage(self) -> Vintage | None:
        return self.state.vintage if self.state else None


@dataclass
class SeriesResult:
    spec: SeriesSpec
    status: Status
    findings: list[Finding]
    reason: str | None = None
    extracted: Extracted | None = None
    gated: pd.DataFrame | None = None
    observations: pd.DataFrame | None = None
    withheld: list[str] = field(default_factory=list)
    latest: tuple[Period, float] | None = None
    expected_latest: Period | None = None
    observed_lag_days: int | None = None
    vintage: Vintage | None = None
    cross_check: dict[str, Any] | None = None
    revisions: dict[str, Any] | None = None

    @property
    def reportable(self) -> bool:
        return self.status.reportable and self.gated is not None

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity is severity]

    @property
    def as_expected(self) -> bool | None:
        """For quarantined series: did validation reach the declared verdict?"""
        if not self.spec.expected_status:
            return None
        return self.spec.expected_status == self.verdict

    @property
    def verdict(self) -> str:
        return f"rejected:{self.reason}" if self.status is Status.REJECTED else self.status.value

    def gated_series(self) -> pd.Series:
        """Admitted values indexed by ordinal (NaN where missing)."""
        assert self.gated is not None
        return self.gated.set_index("ordinal")["value"]

    def to_json(self) -> dict[str, Any]:
        return {
            "series_id": self.spec.id,
            "role": self.spec.role.value,
            "verdict": self.verdict,
            "expected": self.spec.expected_status,
            "as_expected": self.as_expected,
            "latest": self.latest[0].label if self.latest else None,
            "expected_latest": self.expected_latest.label if self.expected_latest else None,
            "withheld": self.withheld[-4:],
            "observed_lag_days": self.observed_lag_days,
            "vintage": self.vintage.vintage_id if self.vintage else None,
            "cross_check": _jsonable(self.cross_check),
            "revisions": _jsonable(self.revisions),
            "findings": [f.to_json() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def validate_series(
    spec: SeriesSpec,
    primary: SourceView,
    *,
    as_of: date,
    previous: Extracted | None = None,
    cross: SourceView | None = None,
) -> SeriesResult:
    findings: list[Finding] = []

    def reject(reason: str, **kw: Any) -> SeriesResult:
        return SeriesResult(spec, Status.REJECTED, findings, reason=reason, vintage=primary.vintage, **kw)

    # -- retrieval ---------------------------------------------------------
    state = primary.state
    if state is None or state.vintage is None:
        findings.append(Finding("retrieval", Severity.ERROR,
                                f"no snapshot available: {state.error if state else 'not planned'}"))
        return reject("no_data")
    if state.degraded:
        findings.append(Finding("retrieval", Severity.WARN,
                                f"refetch failed ({state.error}); using stored vintage "
                                f"{state.vintage.vintage_id}", {"error": state.error}))
    curated = isinstance(primary.ref, CuratedRef)
    if state.selection and state.selection.after_as_of and curated:
        # Each event carries its own announcement date and is gated by it, so a
        # list compiled later cannot leak a future event into this run.
        findings.append(Finding(
            "vintage", Severity.INFO,
            f"event list compiled on {state.vintage.retrieved_date_sgt}, after the as-of date; events "
            "are gated by their announcement dates"))
    elif state.selection and state.selection.after_as_of:
        findings.append(Finding(
            "vintage", Severity.WARN,
            f"no snapshot predates the as-of date; using vintage retrieved "
            f"{state.vintage.retrieved_date_sgt}, so values may include revisions published "
            "after the as-of date (availability is still gated)"))

    # -- parse -------------------------------------------------------------
    if primary.extracted is None:
        findings.append(Finding("parse", Severity.ERROR, primary.error or "could not be parsed"))
        return reject("parse_error")
    extracted = primary.extracted
    findings += [Finding("parse", Severity.INFO, note) for note in extracted.notes]

    # -- contract with the publisher's metadata ------------------------------
    findings += contract_checks(spec, primary.ref, extracted, state.vintage.upstream, severity=Severity.ERROR)
    if any(f.severity is Severity.ERROR for f in findings):
        return reject("contract_mismatch", extracted=extracted)

    populated = extracted.populated
    if populated.empty:
        findings.append(Finding("emptiness", Severity.ERROR, "no populated observations"))
        return reject("empty", extracted=extracted)

    # -- publication-lag gate ----------------------------------------------
    observations = gate_frame(spec, extracted, as_of)
    gated = observations[observations["admitted"]].drop(columns=["admitted"]).reset_index(drop=True)
    withheld = observations.loc[~observations["admitted"] & observations["value"].notna(), "period"].tolist()
    if withheld:
        findings.append(Finding(
            "gate", Severity.INFO,
            f"withheld {len(withheld)} observation(s) not yet publishable at {as_of}: newest "
            f"{withheld[-1]} (declared lag {spec.publication_lag_days}d"
            + (f" from end-{_month_name(spec.reference_month)}" if spec.reference_month else "")
            + ")", {"withheld": withheld[-4:]}))
    admitted_populated = gated[gated["value"].notna()]
    if admitted_populated.empty:
        findings.append(Finding("gate", Severity.ERROR,
                                f"no observation would have been published by {as_of}"))
        return reject("nothing_admitted", extracted=extracted, observations=observations, withheld=withheld)

    last = admitted_populated.iloc[-1]
    latest = (extracted.period(last["ordinal"]), float(last["value"]))
    expected = expected_latest(spec, latest[0], as_of)
    result_kw: dict[str, Any] = dict(extracted=extracted, observations=observations, withheld=withheld,
                                     latest=latest, expected_latest=expected, vintage=state.vintage)

    # -- staleness -----------------------------------------------------------
    stale = staleness_check(spec, latest[0], expected, as_of)
    findings += stale
    discontinued = any(f.severity is Severity.ERROR for f in stale)

    # -- declared vs observed publication lag -----------------------------
    observed_lag = None
    newest_served = extracted.latest()
    if curated:
        findings.append(Finding("declared_lag", Severity.INFO,
                                f"hand-maintained list, last reviewed {state.vintage.upstream.get('last_updated')}; "
                                "a quarter is admitted once it has ended"))
    elif newest_served and not discontinued:
        observed_lag = observed_lag_days(state.vintage.upstream.get("last_updated"),
                                         newest_served[0].reference_date(spec.reference_month))
        findings += declared_lag_check(spec, newest_served[0], observed_lag)

    findings += coverage_check(spec, primary.ref, extracted, state.vintage.upstream)
    findings += range_check(spec, admitted_populated)
    findings += unusual_move_check(spec, admitted_populated)
    findings += gap_check(spec, gated)
    revisions, revision_findings = revision_check(extracted, previous)
    findings += revision_findings
    cross_summary, cross_findings = cross_source_check(spec, extracted, cross)
    findings += cross_findings
    if spec.history_note:
        findings.append(Finding("history", Severity.INFO, spec.history_note))
    for item in spec.breaks:
        findings.append(Finding("history", Severity.INFO,
                                f"series break at {item.period}: {item.note}; changes spanning it are excluded"))

    if discontinued:
        return SeriesResult(spec, Status.REJECTED, findings, reason="discontinued",
                            revisions=revisions, cross_check=cross_summary, **result_kw)
    status = Status.OK_WITH_WARNINGS if any(f.severity is Severity.WARN for f in findings) else Status.OK
    return SeriesResult(spec, status, findings, gated=gated, observed_lag_days=observed_lag,
                        revisions=revisions, cross_check=cross_summary, **result_kw)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

_ADJUSTMENT_LABELS = {
    "seasonally adjusted": Adjustment.SA,
    "non-seasonally adjusted": Adjustment.NSA,
}


def contract_checks(spec: SeriesSpec, ref: SourceRef, extracted: Extracted,
                    upstream: dict[str, Any], *, severity: Severity) -> list[Finding]:
    """Registry declaration vs the publisher's metadata (SingStat only)."""
    if not isinstance(ref, SingStatRef):
        return []
    out: list[Finding] = []
    freq = (upstream.get("frequency") or "").strip()
    if freq and freq.lower() != spec.frequency.upstream_label.lower():
        out.append(Finding("contract", severity,
                           f"{ref.table_id}: registry declares {spec.frequency.value} but the "
                           f"publisher reports {freq!r}"))
    adjustment = _ADJUSTMENT_LABELS.get((upstream.get("adjustment") or "").strip().lower())
    declared_sa = spec.seasonal_adjustment is Adjustment.SA
    if declared_sa and adjustment is not Adjustment.SA:
        out.append(Finding("contract", severity,
                           f"{ref.table_id}: registry declares seasonally adjusted but the publisher "
                           f"reports {upstream.get('adjustment')!r}"))
    if not declared_sa and adjustment is Adjustment.SA:
        out.append(Finding("contract", severity,
                           f"{ref.table_id}: publisher reports seasonally adjusted data but the "
                           f"registry declares {spec.seasonal_adjustment.value}"))
    expected_unit = _norm(ref.unit_of_measure)
    for number, unit in extracted.publisher_units.items():
        if unit is None:
            out.append(Finding("contract", Severity.INFO,
                               f"{ref.table_id} series {number}: publisher reports no unit of measure"))
        elif _norm(unit) != expected_unit:
            out.append(Finding(
                "contract", severity,
                f"{ref.table_id} series {number}: publisher's unit of measure is {unit!r} but the "
                f"registry declares {ref.unit_of_measure!r}; every level and change would be "
                "mislabelled", {"publisher_unit": unit, "declared": ref.unit_of_measure}))
    return out


def gate_frame(spec: SeriesSpec, extracted: Extracted, as_of: date) -> pd.DataFrame:
    frame = extracted.frame.copy()
    lag = timedelta(days=spec.publication_lag_days)
    publishable = [extracted.period(o).reference_date(spec.reference_month) + lag for o in frame["ordinal"]]
    frame["publishable_on"] = [d.isoformat() for d in publishable]
    frame["admitted"] = [d <= as_of for d in publishable]
    return frame


def publishable_on(spec: SeriesSpec, period: Period) -> date:
    return period.reference_date(spec.reference_month) + timedelta(days=spec.publication_lag_days)


def expected_latest(spec: SeriesSpec, latest: Period, as_of: date) -> Period:
    """Newest period the declared schedule says should be published by ``as_of``."""
    candidate = latest
    for _ in range(2000):
        nxt = candidate.shift(1)
        if publishable_on(spec, nxt) > as_of:
            return candidate
        candidate = nxt
    return candidate  # pragma: no cover - absurd as-of date


def staleness_check(spec: SeriesSpec, latest: Period, expected: Period, as_of: date) -> list[Finding]:
    behind = expected.ordinal - latest.ordinal
    details = {"latest": latest.label, "expected": expected.label, "periods_behind": behind}
    if behind <= 0:
        return []
    if behind > DISCONTINUED_AFTER[spec.frequency]:
        return [Finding("staleness", Severity.ERROR,
                        f"newest observation {latest.label} is {behind} {spec.frequency.value} periods "
                        f"behind the {expected.label} that the as-of date {as_of} should allow; "
                        "treated as discontinued and excluded", details)]
    if behind == 1:
        return [Finding("staleness", Severity.INFO,
                        f"{expected.label} was due by {as_of} under the declared lag but is not in the "
                        f"snapshot; the release may simply be late, or a refetch is needed", details)]
    return [Finding("staleness", Severity.WARN,
                    f"newest observation {latest.label} is {behind} periods behind schedule "
                    f"({expected.label} expected by {as_of}); refetch or check the source", details)]


def observed_lag_days(last_updated: str | None, reference: date) -> int | None:
    stamp = _parse_stamp(last_updated)
    return None if stamp is None else (stamp - reference).days


def declared_lag_check(spec: SeriesSpec, newest: Period, observed: int | None) -> list[Finding]:
    if observed is None:
        return [Finding("declared_lag", Severity.INFO,
                        "no upstream update stamp, so the declared lag could not be cross-checked")]
    details = {"observed_days": observed, "declared_days": spec.publication_lag_days, "period": newest.label}
    if observed > spec.publication_lag_days:
        return [Finding("declared_lag", Severity.WARN,
                        f"declared publication lag {spec.publication_lag_days}d is shorter than the "
                        f"{observed}d observed for {newest.label}; the gate may admit a figure before "
                        "it was published", details)]
    return [Finding("declared_lag", Severity.INFO,
                    f"declared lag {spec.publication_lag_days}d covers the {observed}d observed for "
                    f"{newest.label}", details)]


def coverage_check(spec: SeriesSpec, ref: SourceRef, extracted: Extracted,
                   upstream: dict[str, Any]) -> list[Finding]:
    if not isinstance(ref, DataGovRef):
        return []
    declared = _parse_stamp(upstream.get("coverage_end"))
    newest = extracted.latest()
    if declared is None or newest is None:
        return []
    served = newest[0].end_date
    gap = (declared - served).days
    if gap <= 92:
        return []
    return [Finding("coverage", Severity.WARN,
                    f"the dataset declares coverage to {declared} but this series ends at "
                    f"{newest[0].label} ({gap} days earlier): the dataset is alive, the series "
                    "may not be", {"declared_end": declared.isoformat(), "series_end": served.isoformat()})]


def range_check(spec: SeriesSpec, admitted: pd.DataFrame) -> list[Finding]:
    low, high = spec.plausible_range
    bad = admitted[(admitted["value"] < low) | (admitted["value"] > high)]
    if bad.empty:
        return []
    sample = list(zip(bad["period"], bad["value"].round(4)))[:3]
    return [Finding("range", Severity.WARN,
                    f"{len(bad)} observation(s) outside the plausible band [{low:g}, {high:g}] "
                    f"declared for this series, e.g. {sample}; either the band or the unit is wrong",
                    {"offenders": sample})]


def period_changes(spec: SeriesSpec, values: pd.Series) -> pd.Series:
    """Period-on-period change in the arithmetic the unit kind allows."""
    from sgmacro.transform import mask_breaks

    values = values.astype("float64")
    if spec.unit_kind.is_difference_based:
        return mask_breaks(spec, values.diff(), 1)
    prior = values.shift(1)
    return mask_breaks(spec, (values / prior.where(prior > 0) - 1.0) * 100.0, 1)


def unusual_move_check(spec: SeriesSpec, admitted: pd.DataFrame) -> list[Finding]:
    series = admitted.set_index("ordinal")["value"]
    series = series.reindex(range(int(series.index.min()), int(series.index.max()) + 1))
    changes = period_changes(spec, series).dropna()
    if len(changes) < 24:
        return []
    median = changes.median()
    mad = (changes - median).abs().median() * 1.4826
    if not mad or not math.isfinite(mad):
        return []
    recent = changes.iloc[-UNUSUAL_MOVE_WINDOW:]
    z = (recent - median) / mad
    flagged = z[z.abs() > UNUSUAL_MOVE_Z]
    if flagged.empty:
        return []
    labels = [Period.from_ordinal(spec.frequency, int(o)).label for o in flagged.index]
    return [Finding("unusual_move", Severity.WARN,
                    f"period-on-period move at {labels} is {float(flagged.abs().max()):.1f} robust "
                    "standard deviations from this series' typical move; it may be genuine, but "
                    "verify against the publisher before relying on it",
                    {"periods": labels, "robust_z": [round(float(v), 2) for v in flagged]})]


def gap_check(spec: SeriesSpec, gated: pd.DataFrame) -> list[Finding]:
    window = gated.iloc[-(2 * spec.frequency.periods_per_year + 1):]
    if window.empty:
        return []
    ordinals = window["ordinal"].astype(int).tolist()
    absent = sorted(set(range(ordinals[0], ordinals[-1] + 1)) - set(ordinals))
    blank = window.loc[window["value"].isna(), "period"].tolist()
    out = []
    if absent:
        out.append(Finding("gaps", Severity.WARN,
                           f"{len(absent)} period(s) absent from the trailing two years"))
    if blank:
        out.append(Finding("gaps", Severity.WARN,
                           f"trailing-two-year observations are empty at {blank[:6]}", {"periods": blank}))
    return out


def revision_check(current: Extracted, previous: Extracted | None) -> tuple[dict[str, Any] | None, list[Finding]]:
    if previous is None:
        return None, []
    before = previous.values()
    after = current.values()
    common = before.index.intersection(after.index)
    b, a = before.loc[common], after.loc[common]
    changed_mask = ~((b.isna() & a.isna()) | np.isclose(b.fillna(np.inf), a.fillna(np.inf), rtol=1e-9, atol=1e-9))
    changed = common[changed_mask.to_numpy()]
    added = after.index.difference(before.index)
    summary = {
        "compared_periods": int(len(common)),
        "revised_periods": int(len(changed)),
        "new_periods": [current.period(o).label for o in added[-6:]],
        "max_abs_revision": float((a.loc[changed] - b.loc[changed]).abs().max()) if len(changed) else 0.0,
        "examples": [
            {"period": current.period(o).label, "before": _num(b.loc[o]), "after": _num(a.loc[o])}
            for o in changed[-5:]
        ],
    }
    findings: list[Finding] = []
    if len(changed):
        findings.append(Finding("revision", Severity.WARN,
                                f"{len(changed)} historical observation(s) revised since the previous "
                                f"vintage (largest |change| {summary['max_abs_revision']:.4g}), e.g. "
                                f"{summary['examples'][-3:]}", summary))
    if len(added):
        findings.append(Finding("revision", Severity.INFO,
                                f"{len(added)} new observation(s) since the previous vintage: "
                                f"{summary['new_periods'][-3:]}"))
    return summary, findings


def cross_source_check(spec: SeriesSpec, primary: Extracted,
                       cross: SourceView | None) -> tuple[dict[str, Any] | None, list[Finding]]:
    if cross is None or spec.cross_check is None:
        return None, []
    name = _source_name(cross.ref)
    if cross.extracted is None:
        reason = cross.error or (cross.state.error if cross.state else None) or "no snapshot"
        return ({"source": name, "available": False, "reason": reason},
                [Finding("cross_source", Severity.WARN, f"cross-check source {name} unavailable: {reason}")])
    findings: list[Finding] = []
    if cross.vintage is not None:
        findings += [
            Finding("cross_source", Severity.WARN, f"cross-check source: {f.message}")
            for f in contract_checks(spec, cross.ref, cross.extracted, cross.vintage.upstream,
                                     severity=Severity.WARN)
            if f.severity is Severity.WARN
        ]
    a = primary.values().dropna()
    b = cross.extracted.values().dropna()
    common = a.index.intersection(b.index)
    tolerance = spec.cross_check.tolerance
    diffs = (a.loc[common] - b.loc[common]).abs()
    mismatched = diffs[diffs > tolerance]
    primary_latest, cross_latest = primary.latest(), cross.extracted.latest()
    lead = (primary_latest[0].ordinal - cross_latest[0].ordinal) if primary_latest and cross_latest else None
    summary = {
        "source": name,
        "identifier": cross.ref.identifier,
        "available": True,
        "overlap_periods": int(len(common)),
        "mismatches": int(len(mismatched)),
        "max_abs_diff": float(diffs.max()) if len(diffs) else None,
        "tolerance": tolerance,
        "primary_latest": primary_latest[0].label if primary_latest else None,
        "cross_latest": cross_latest[0].label if cross_latest else None,
        "primary_lead_periods": lead,
    }
    if not len(common):
        findings.append(Finding("cross_source", Severity.WARN, f"no overlapping periods with {name}"))
    elif len(mismatched):
        examples = [
            {"period": primary.period(o).label, "primary": _num(a.loc[o]), "cross": _num(b.loc[o])}
            for o in mismatched.index[-3:]
        ]
        findings.append(Finding("cross_source", Severity.WARN,
                                f"{len(mismatched)} of {len(common)} overlapping periods differ from "
                                f"{name} by more than {tolerance:g}, e.g. {examples}", summary))
    else:
        findings.append(Finding("cross_source", Severity.INFO,
                                f"agrees with {name} on all {len(common)} overlapping periods "
                                f"(max |difference| {summary['max_abs_diff']:.4g}, tolerance {tolerance:g})"))
    if lead is not None and lead < 0:
        findings.append(Finding("cross_source", Severity.WARN,
                                f"{name} is {-lead} period(s) fresher (latest {summary['cross_latest']} vs "
                                f"{summary['primary_latest']}); consider making it the primary source"))
    elif lead:
        findings.append(Finding("cross_source", Severity.INFO,
                                f"{name} lags the primary source by {lead} period(s) "
                                f"(its latest is {summary['cross_latest']})"))
    return summary, findings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _source_name(ref: SourceRef) -> str:
    return "SingStat" if isinstance(ref, SingStatRef) else "data.gov.sg"


def _parse_stamp(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _month_name(month: int | None) -> str:
    import calendar

    return calendar.month_name[month] if month else ""


def _num(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, 6)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) else round(value, 10)
    return value
