"""Lead-lag diagnostics recover a known lag and flag disagreement with the declared sign."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sgmacro.diagnostics import LAGS, lead_lag, to_quarterly
from sgmacro.registry import Frequency, Transform
from sgmacro.transform import SeriesChanges
from tests.conftest import make_spec, rate_spec


def series_changes(spec, history: pd.Series) -> SeriesChanges:
    return SeriesChanges(spec=spec, latest=None, changes=[], headline=None, headline_history=history,
                         percentile=None, history_start=None)


def test_known_two_quarter_lead_is_recovered():
    n = 80
    start = 2000 * 4
    signal = pd.Series([math.sin(i / 3.0) + 0.3 * math.cos(i / 1.7) for i in range(n)], index=range(start, start + n))
    # Target YoY copies the predictor two quarters later, with the opposite sign.
    target_yoy = -signal.shift(2)
    level = [100.0] * 4
    for i in range(4, n):
        level.append(level[i - 4] * (1 + (target_yoy.iloc[i] if not math.isnan(target_yoy.iloc[i]) else 0) / 100))
    target_values = pd.Series(level, index=range(start, start + n))
    predictor = make_spec(id="p", role="predictor", markets=["sales"], channel="c", effects={"sales": "negative"},
                          neutral_band=1, expected_lead_quarters=[1, 4], rationale="r", limitations="l",
                          unit_kind="level")
    target = make_spec(id="t")
    diag = lead_lag(series_changes(predictor, signal), target, target_values)
    assert diag.best.lag == 2 and diag.best.r < -0.95
    assert diag.agreement == "agrees"
    assert [c.lag for c in diag.correlations] == list(LAGS)


def _target_with_yoy(yoy: pd.Series) -> pd.Series:
    """Index levels whose year-on-year change is exactly ``yoy`` (NaN treated as 0)."""
    levels = [100.0] * 4
    for i in range(4, len(yoy)):
        change = 0.0 if math.isnan(yoy.iloc[i]) else yoy.iloc[i]
        levels.append(levels[i - 4] * (1 + change / 100))
    return pd.Series(levels, index=yoy.index)


def _pipeline_spec():
    return make_spec(id="pipeline", role="predictor", markets=["sales"], channel="c",
                     effects={"sales": "negative"}, neutral_band=1, expected_lead_quarters=[4, 12],
                     rationale="r", limitations="l", unit_kind="level")


def _noise(n, start):
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(size=n), index=range(start, start + n))


def test_a_correlation_outside_the_declared_window_is_reported_not_judged():
    """The pipeline case: prices and construction move together in the same quarter
    (developers build into strong markets), but the declared supply effect is 4-12
    quarters out. Lag 0 is reported as the strongest finding; the verdict comes from
    the declared window, where there is nothing, so it is weak, not 'disagrees'."""
    x = _noise(80, 2000 * 4)
    diag = lead_lag(series_changes(_pipeline_spec(), x), make_spec(id="t"), _target_with_yoy(x))
    assert diag.strongest.lag == 0 and diag.strongest_outside.lag == 0
    assert 4 <= diag.best.lag <= 12 and diag.agreement == "weak"


def test_opposite_sign_inside_the_window_is_flagged_as_disagreement():
    x = _noise(80, 2000 * 4)
    diag = lead_lag(series_changes(_pipeline_spec(), x), make_spec(id="t"), _target_with_yoy(x.shift(6)))
    assert diag.best.lag == 6 and diag.best.r > 0.9 and diag.agreement == "disagrees"
    assert diag.strongest_outside is None


def test_short_history_is_not_correlated():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=range(8000, 8005))
    predictor = rate_spec(frequency="quarterly", transforms=["yoy_bp"], headline="yoy_bp")
    diag = lead_lag(series_changes(predictor, x), make_spec(id="t"),
                    pd.Series([100.0 + i for i in range(10)], index=range(8000, 8010)))
    assert diag.best is None and diag.agreement == "insufficient data"


def test_monthly_maps_to_last_month_and_annual_to_reference_quarter():
    monthly = rate_spec()
    hist = pd.Series([1.0, 2.0, 3.0, 4.0], index=[2026 * 12 + i for i in range(4)])  # Jan-Apr 2026
    q = to_quarterly(monthly, hist)
    assert q.loc[2026 * 4] == 3.0 and q.loc[2026 * 4 + 1] == 4.0
    annual = make_spec(id="a", frequency="annual", reference_month=6, unit_kind="level", transforms=["yoy"])
    qa = to_quarterly(annual, pd.Series([5.0], index=[2026]))
    assert list(qa.index) == [2026 * 4 + 1]  # Q2, the quarter of the end-June reference date
    assert Frequency.QUARTERLY.periods_per_year == 4 and Transform.YOY.step == "yoy"
