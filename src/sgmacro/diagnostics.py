"""Lead-lag diagnostics: descriptive evidence on the declared lags and signs.

For each predictor and target, the Pearson correlation between the predictor's
headline change and the target's year-on-year change, with the predictor leading
by ``k`` quarters. Computed only on observations the as-of gate admitted.

What this is not: a model, a forecast or a significance test. Year-on-year
windows overlap, so neighbouring observations are strongly autocorrelated and
the effective sample is far smaller than the pair count; there is no correction
for the hundreds of correlations tabulated; predictors are collinear in a small
open economy. No p-values are reported, deliberately.

What it is for: confronting the registry's *declared* hypothesis with the data.
The declaration says both the sign of the effect and *when* it should show (the
expected lead, in quarters), so the verdict is judged only on lags inside that
window. The strongest correlation at any lag is still reported, as a finding:
the supply pipeline correlates positively with prices in the same quarter
(developers build into strong markets), which says nothing about what extra
supply does two years later, and must not be read as evidence against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sgmacro.periods import Period
from sgmacro.registry import Frequency, Market, SeriesSpec, Transform
from sgmacro.transform import SeriesChanges, change_history

LAGS = (0, 1, 2, 4, 6, 8, 12)
#: Fewer overlapping quarters than this and no correlation is reported. Raised from 12
#: after bank housing loans (history from 2021-07, 12-16 pairs) was reported as
#: 'disagrees' on a sample far too short to support any reading.
MIN_PAIRS = 20
WEAK = 0.2
#: How much stronger a correlation outside the declared window must be to be called out.
NOTABLE = 0.1


@dataclass(frozen=True)
class LagCorrelation:
    lag: int
    r: float | None
    pairs: int


@dataclass
class Diagnostic:
    predictor_id: str
    target_id: str
    market: Market
    expected_sign: int | None
    correlations: list[LagCorrelation] = field(default_factory=list)
    window: tuple[int, int] | None = None
    """The declared lead in quarters. None (an agent's candidate, which has no
    declaration yet) judges on every lag."""

    def in_window(self, lag: int) -> bool:
        return self.window is None or self.window[0] <= lag <= self.window[1]

    @property
    def best(self) -> LagCorrelation | None:
        """The strongest correlation inside the declared window: the verdict's basis."""
        scored = [c for c in self.correlations if c.r is not None and self.in_window(c.lag)]
        return max(scored, key=lambda c: abs(c.r)) if scored else None  # type: ignore[arg-type]

    @property
    def strongest(self) -> LagCorrelation | None:
        """The strongest correlation at any lag, reported but not judged."""
        scored = [c for c in self.correlations if c.r is not None]
        return max(scored, key=lambda c: abs(c.r)) if scored else None  # type: ignore[arg-type]

    @property
    def strongest_outside(self) -> LagCorrelation | None:
        """The overall strongest, when it lies outside the window and tells a different
        story: clearly stronger than the in-window best, or of the opposite sign."""
        top, best = self.strongest, self.best
        if top is None or self.in_window(top.lag):
            return None
        if best is None:
            return top
        clearly_stronger = abs(top.r) - abs(best.r) >= NOTABLE  # type: ignore[operator]
        flips = np.sign(top.r) != np.sign(best.r) and abs(top.r) >= WEAK  # type: ignore[arg-type]
        return top if clearly_stronger or flips else None

    @property
    def agreement(self) -> str:
        best = self.best
        if best is None:
            return "insufficient data"
        if self.expected_sign is None:
            return "market not declared"
        if abs(best.r) < WEAK:  # type: ignore[arg-type]
            return "weak"
        if self.expected_sign == 0:
            return "ambiguous by design"
        return "agrees" if np.sign(best.r) == self.expected_sign else "disagrees"


def to_quarterly(spec: SeriesSpec, history: pd.Series) -> pd.Series:
    """Map a change history onto quarterly ordinals.

    Monthly: the last available month of each quarter (these are end-of-period or
    monthly-average series and the quarter-end reading is the conventional one).
    Annual: the quarter containing the reference date, so an end-June population
    estimate sits in Q2 rather than being pushed to Q4.
    """
    if spec.frequency is Frequency.QUARTERLY:
        return history
    mapped: dict[int, float] = {}
    for ordinal, value in history.dropna().items():
        quarter = Period.from_ordinal(spec.frequency, int(ordinal)).to_quarter(spec.reference_month)
        mapped[quarter.ordinal] = float(value)  # later months overwrite earlier ones
    return pd.Series(mapped, dtype="float64").sort_index()


def lead_lag(predictor: SeriesChanges, target_spec: SeriesSpec, target_values: pd.Series,
             lags: tuple[int, ...] = LAGS) -> Diagnostic:
    market = target_spec.market
    assert market is not None
    effect = predictor.spec.effects.for_market(market) if market in predictor.spec.markets else None
    lead = predictor.spec.expected_lead_quarters
    return correlate(predictor.spec.id, to_quarterly(predictor.spec, predictor.headline_history), target_spec,
                     target_values, effect.sign if effect is not None else None, lags,
                     window=(lead[0], lead[1]) if lead else None)


def correlate(predictor_id: str, x: pd.Series, target_spec: SeriesSpec, target_values: pd.Series,
              expected_sign: int | None, lags: tuple[int, ...] = LAGS,
              window: tuple[int, int] | None = None) -> Diagnostic:
    """Correlate a quarterly change series ``x`` with a target's year-on-year change.

    Shared by the report diagnostics and by the verifier, which runs the same
    check on an agent's candidate before a person decides whether to adopt it.
    """
    target_yoy = change_history(target_spec, Transform.YOY, target_values).dropna()
    market = target_spec.market
    assert market is not None
    diagnostic = Diagnostic(predictor_id, target_spec.id, market, expected_sign, window=window)
    x = x.dropna()
    if x.empty or target_yoy.empty:
        diagnostic.correlations = [LagCorrelation(lag, None, 0) for lag in lags]
        return diagnostic
    index = range(int(min(x.index.min(), target_yoy.index.min())),
                  int(max(x.index.max(), target_yoy.index.max())) + 1)
    x = x.reindex(index)
    y = target_yoy.reindex(index)
    for lag in lags:
        pairs = pd.concat([x.shift(lag), y], axis=1, keys=["x", "y"]).dropna()
        n = len(pairs)
        if n < MIN_PAIRS or pairs["x"].std() == 0 or pairs["y"].std() == 0:
            diagnostic.correlations.append(LagCorrelation(lag, None, n))
            continue
        diagnostic.correlations.append(LagCorrelation(lag, float(pairs["x"].corr(pairs["y"])), n))
    return diagnostic
