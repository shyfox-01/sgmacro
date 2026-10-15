"""Deterministic interpretation: what each predictor's latest move implies.

For every predictor and every market it is declared to bear on::

    direction = sign(headline change), or 0 inside the declared neutral band
    reading   = direction x expected effect
                 +1 -> supportive    (pushes the market up)
                 -1 -> headwind      (pushes the market down)
                  0 -> neutral       (the predictor barely moved)
    an expected effect of "ambiguous" always reads "ambiguous"

This is a structured restatement of the registry's hypotheses applied to the
latest data, not a forecast. Its value is that it is computed, so the LLM's
summary can be *checked against it*: prose that calls a rate rise "supportive for
prices" contradicts a declared negative effect and is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from sgmacro.registry import Effect, Market, SeriesSpec
from sgmacro.transform import SeriesChanges


@dataclass(frozen=True)
class Signal:
    series_id: str
    market: Market
    effect: Effect
    direction: int
    reading: str


def direction_of(value: float | None, neutral_band: float) -> int | None:
    if value is None:
        return None
    if abs(value) <= neutral_band:
        return 0
    return 1 if value > 0 else -1


def read(effect: Effect, direction: int | None) -> str:
    if direction is None:
        return "unavailable"
    if effect is Effect.AMBIGUOUS:
        return "ambiguous"
    if direction == 0:
        return "neutral"
    return "supportive" if direction * effect.sign > 0 else "headwind"


def signals_for(spec: SeriesSpec, changes: SeriesChanges) -> list[Signal]:
    direction = direction_of(changes.headline.value, spec.neutral_band or 0.0)
    out = []
    for market in spec.markets:
        effect = spec.effects.for_market(market)
        assert effect is not None  # guaranteed by the registry
        out.append(Signal(spec.id, market, effect, direction if direction is not None else 0,
                          read(effect, direction)))
    return out
