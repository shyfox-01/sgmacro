"""Change calculations, dispatched on unit kind and observation frequency.

Two errors this module exists to prevent:

*Percentage change of a rate.* SORA moving from 1.00% to 1.50% is +50 basis
points, not +50 percent. The arithmetic follows the declared ``unit_kind``,
which the registry has already tied to the permitted transforms.

*A fixed year-on-year lag.* "A year ago" is 12 observations on a monthly series
and 4 on a quarterly one; the lag comes from the frequency.

Every :class:`Change` records the exact observations it consumed, each with its
raw-snapshot locator, vintage and content hash, plus the formula as written. That
record is what ``sgmacro trace`` re-verifies against the raw bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sgmacro.periods import Period
from sgmacro.registry import Frequency, SeriesSpec, Transform
from sgmacro.validate import SeriesResult

FORMULAS = {
    "pct": "(v[t] / v[t-k] - 1) * 100",
    "pp": "v[t] - v[t-k]",
    "bp": "(v[t] - v[t-k]) * 100",
    "t4q": "(sum(v[t-3..t]) / sum(v[t-7..t-4]) - 1) * 100",
    "t4q_sum": "sum(v[t-3..t])",
}

HUMAN = {
    Transform.MOM: "month-on-month change",
    Transform.QOQ: "quarter-on-quarter change",
    Transform.YOY: "year-on-year change",
    Transform.MOM_PP: "month-on-month change",
    Transform.QOQ_PP: "quarter-on-quarter change",
    Transform.YOY_PP: "year-on-year change",
    Transform.MOM_BP: "month-on-month change",
    Transform.YOY_BP: "year-on-year change",
    Transform.T4Q_YOY: "trailing four quarters vs the four before",
    Transform.T4Q_SUM: "net rounds in the trailing four quarters",
}


@dataclass(frozen=True)
class ObsRef:
    """One raw observation a figure depends on, and where it lives."""

    series_id: str
    period: str
    value: float | None
    locator: str
    vintage_id: str
    data_sha256: str
    object_key: str


@dataclass
class Change:
    series_id: str
    transform: Transform
    period: Period
    value: float | None
    inputs: list[ObsRef]
    formula: str
    note: str | None = None

    @property
    def unit(self) -> str:
        return self.transform.change_unit

    @property
    def human(self) -> str:
        return HUMAN[self.transform]


@dataclass
class SeriesChanges:
    spec: SeriesSpec
    latest: ObsRef
    changes: list[Change]
    headline: Change
    headline_history: pd.Series = field(repr=False)
    """Headline change at every admitted period, indexed by ordinal."""
    percentile: float | None
    """Share of the headline's own history at or below the latest value (0-100)."""
    history_start: str | None

    def change(self, transform: Transform) -> Change | None:
        return next((c for c in self.changes if c.transform is transform), None)


def lag_for(transform: Transform, freq: Frequency) -> int:
    if transform.step == "yoy":
        return freq.periods_per_year
    return 1


def _dense(series: pd.Series) -> pd.Series:
    """Reindex to a gap-free ordinal range so ``shift(k)`` means k periods."""
    if series.empty:
        return series
    index = range(int(series.index.min()), int(series.index.max()) + 1)
    return series.reindex(index).astype("float64")


def window_for(transform: Transform, freq: Frequency) -> int:
    """How many periods back a transform's inputs reach."""
    if transform is Transform.T4Q_YOY:
        return 7
    if transform is Transform.T4Q_SUM:
        return 3
    return lag_for(transform, freq)


def spanned_break(spec: SeriesSpec, ordinal: int, window: int) -> str | None:
    """The first series break inside ``(ordinal - window, ordinal]``, if any."""
    from sgmacro.periods import parse_period

    for item in spec.breaks:
        b = parse_period(item.period, spec.frequency).ordinal
        if ordinal - window < b <= ordinal:
            return item.period
    return None


def mask_breaks(spec: SeriesSpec, history: pd.Series, window: int) -> pd.Series:
    if not spec.breaks:
        return history
    keep = [spanned_break(spec, int(o), window) is None for o in history.index]
    return history.where(pd.Series(keep, index=history.index))


def change_history(spec: SeriesSpec, transform: Transform, values: pd.Series) -> pd.Series:
    """The transform evaluated at every period (NaN where undefined or where the
    window spans a declared series break)."""
    s = _dense(values)
    k = lag_for(transform, spec.frequency)
    if transform is Transform.T4Q_YOY:
        rolling = s.rolling(4, min_periods=4).sum()
        base = rolling.shift(4)
        out = (rolling / base.where(base > 0) - 1.0) * 100.0
    elif transform is Transform.T4Q_SUM:
        out = s.rolling(4, min_periods=4).sum()
    elif transform.change_unit == "pct":
        base = s.shift(k)
        out = (s / base.where(base > 0) - 1.0) * 100.0
    else:
        diff = s - s.shift(k)
        out = diff * 100.0 if transform.change_unit == "bp" else diff
    return mask_breaks(spec, out, window_for(transform, spec.frequency))


def obs_ref(result: SeriesResult, ordinal: int) -> ObsRef:
    assert result.gated is not None and result.vintage is not None
    row = result.gated.loc[result.gated["ordinal"] == ordinal]
    period = Period.from_ordinal(result.spec.frequency, ordinal)
    if row.empty:
        return ObsRef(result.spec.id, period.label, None, "(absent)", result.vintage.vintage_id,
                      result.vintage.data_sha256, result.vintage.object_key)
    value = row["value"].iloc[0]
    return ObsRef(
        series_id=result.spec.id,
        period=period.label,
        value=None if pd.isna(value) else float(value),
        locator=str(row["locator"].iloc[0]),
        vintage_id=result.vintage.vintage_id,
        data_sha256=result.vintage.data_sha256,
        object_key=result.vintage.object_key,
    )


def compute_changes(result: SeriesResult) -> SeriesChanges:
    spec = result.spec
    assert result.latest is not None
    values = result.gated_series()
    t = result.latest[0]
    changes: list[Change] = []
    for transform in spec.transforms:
        history = change_history(spec, transform, values)
        value = history.get(t.ordinal, np.nan)
        k = lag_for(transform, spec.frequency)
        if transform is Transform.T4Q_YOY:
            ordinals = list(range(t.ordinal - 7, t.ordinal + 1))
            formula = FORMULAS["t4q"]
        elif transform is Transform.T4Q_SUM:
            ordinals = list(range(t.ordinal - 3, t.ordinal + 1))
            formula = FORMULAS["t4q_sum"]
        else:
            ordinals = [t.ordinal - k, t.ordinal]
            formula = FORMULAS[transform.change_unit].replace("k", str(k))
        inputs = [obs_ref(result, o) for o in ordinals]
        note = None
        if pd.isna(value):
            missing = [i.period for i in inputs if i.value is None]
            crossed = spanned_break(spec, t.ordinal, window_for(transform, spec.frequency))
            if crossed:
                note = f"undefined: the window spans the series break at {crossed}"
            elif missing:
                note = f"undefined: missing observation(s) {missing}"
            else:
                note = "undefined: base is zero or negative"
        changes.append(Change(spec.id, transform, t, None if pd.isna(value) else float(value),
                              inputs, formula, note))

    headline = next(c for c in changes if c.transform is spec.headline)
    history = change_history(spec, spec.headline, values).dropna()
    percentile = None
    if headline.value is not None and len(history) >= 8:
        percentile = float((history <= headline.value + 1e-12).mean() * 100.0)
    start = Period.from_ordinal(spec.frequency, int(history.index.min())).label if len(history) else None
    return SeriesChanges(spec, obs_ref(result, t.ordinal), changes, headline, history, percentile, start)


def quarterly_average(values: pd.Series, freq: Frequency, quarter: Period) -> tuple[float | None, list[int]]:
    """Average of the three monthly values in a quarter, or None if any is missing."""
    if freq is not Frequency.MONTHLY:
        raise ValueError("quarterly_average expects a monthly series")
    ordinals = [m.ordinal for m in quarter.months()]
    picked = [values.get(o, np.nan) for o in ordinals]
    if any(pd.isna(v) for v in picked):
        return None, ordinals
    return float(sum(picked) / 3.0), ordinals


def real_yoy(target: SeriesResult, target_changes: SeriesChanges, cpi: SeriesResult) -> Change | None:
    """CPI-deflated year-on-year change of a quarterly target.

    ``real = ((1 + nominal) / (1 + cpi) - 1) * 100``, with CPI averaged over the
    three months of each quarter. Undefined (None) unless all six CPI months were
    admitted by the as-of gate, so a real figure never leans on an unpublished
    month.
    """
    nominal = target_changes.change(Transform.YOY)
    if nominal is None or nominal.value is None or not cpi.reportable:
        return None
    t = nominal.period
    base = t.shift(-4)
    cpi_values = cpi.gated_series()
    now, now_ord = quarterly_average(cpi_values, Frequency.MONTHLY, t)
    then, then_ord = quarterly_average(cpi_values, Frequency.MONTHLY, base)
    inputs = list(nominal.inputs) + [obs_ref(cpi, o) for o in then_ord + now_ord]
    formula = ("((1 + nominal_yoy/100) / (avg(cpi[t]) / avg(cpi[t-4])) - 1) * 100, "
               "cpi averaged over the quarter's three months")
    if now is None or then is None:
        return Change(target.spec.id, Transform.YOY, t, None, inputs, formula,
                      note="undefined: a CPI month in the window is not yet admitted")
    value = ((1.0 + nominal.value / 100.0) / (now / then) - 1.0) * 100.0
    return Change(target.spec.id, Transform.YOY, t, value, inputs, formula)
