"""Change arithmetic, signals and fact formatting."""

from __future__ import annotations

from datetime import date

import pytest

from sgmacro.facts import format_change, ordinal_suffix, signed
from sgmacro.parse import extract
from sgmacro.registry import Effect, Market, Transform
from sgmacro.signals import direction_of, read, signals_for
from sgmacro.transform import change_history, compute_changes, real_yoy
from sgmacro.validate import SourceView, validate_series
from tests.conftest import (fixture_bytes, make_spec, quarters, rate_spec, singstat_payload, state_for,
                            store_vintage)

AS_OF = date(2026, 10, 2)


def validated(tmp_path, spec, content, extension="csv", upstream=None):
    vintage = store_vintage(tmp_path, spec.source.object_key, content, extension=extension, upstream=upstream or {})
    view = SourceView(spec.source, state_for(vintage), extract(spec.source, content, spec.frequency, vintage.upstream))
    return validate_series(spec, view, as_of=AS_OF)


def test_index_changes_are_percentages(tmp_path):
    result = validated(tmp_path, make_spec(), fixture_bytes("tidy_quarter.csv"))
    changes = compute_changes(result)
    yoy = changes.change(Transform.YOY)
    assert yoy.value == pytest.approx((219.4 / 213.2 - 1) * 100)
    assert [i.period for i in yoy.inputs] == ["2025-Q2", "2026-Q2"]
    assert yoy.formula == "(v[t] / v[t-4] - 1) * 100"
    assert changes.change(Transform.QOQ).value == pytest.approx((219.4 / 218.3 - 1) * 100)


def test_rate_changes_are_basis_points_and_year_ago_is_twelve_months(tmp_path):
    months = [("2025 Aug", 1.5638), ("2026 Jul", 1.1354), ("2026 Aug", 1.1863)]
    spec = rate_spec(source={"source": "singstat", "table_id": "M700071", "series": "23",
                             "unit_of_measure": "Per Cent Per Annum"})
    content = singstat_payload({"23": months}, uom="Per Cent Per Annum")
    result = validated(tmp_path, spec, content, "json", {"frequency": "Monthly"})
    changes = compute_changes(result)
    assert changes.change(Transform.YOY_BP).value == pytest.approx(-37.75)
    assert changes.change(Transform.MOM_BP).value == pytest.approx(5.09)


def test_trailing_four_quarter_change(tmp_path):
    spec = make_spec(source={"source": "datagovsg", "dataset_id": "d_t", "layout": "tidy", "value_column": "units"},
                     unit_kind="level", transforms=["qoq", "t4q_yoy"], headline="t4q_yoy", plausible_range=[1, 10**6])
    values = [100, 100, 100, 100, 110, 120, 130, 140]
    content = b"quarter,units\n" + b"".join(f"{p},{v}\n".encode() for p, v in quarters(2024, values))
    changes = compute_changes(validated(tmp_path, spec, content))
    t4q = changes.change(Transform.T4Q_YOY)
    assert t4q.value == pytest.approx((500 / 400 - 1) * 100)
    assert len(t4q.inputs) == 8


def test_missing_base_leaves_the_change_undefined_with_a_reason(tmp_path):
    content = fixture_bytes("tidy_quarter.csv").replace(b"2025-Q2,All Residential,213.2", b"2025-Q2,All Residential,-")
    changes = compute_changes(validated(tmp_path, make_spec(), content))
    yoy = changes.change(Transform.YOY)
    assert yoy.value is None and "2025-Q2" in yoy.note


def test_zero_base_is_not_a_percentage(tmp_path):
    spec = make_spec(unit_kind="level", plausible_range=[-1, 1000])
    content = b"quarter,property_type,index\n" + b"".join(
        f"{p},All Residential,{v}\n".encode() for p, v in quarters(2025, [0, 5, 6, 7, 8]))
    yoy = compute_changes(validated(tmp_path, spec, content)).change(Transform.YOY)
    assert yoy.value is None and "zero or negative" in yoy.note


def test_change_history_matches_point_calculation(tmp_path):
    result = validated(tmp_path, make_spec(), fixture_bytes("tidy_quarter.csv"))
    changes = compute_changes(result)
    history = change_history(result.spec, Transform.YOY, result.gated_series())
    assert history.dropna().iloc[-1] == pytest.approx(changes.headline.value)


def test_real_change_uses_quarterly_average_cpi(tmp_path):
    target = validated(tmp_path, make_spec(), fixture_bytes("tidy_quarter.csv"))
    cpi_spec = make_spec(id="cpi", source={"source": "singstat", "table_id": "M213751", "series": "1",
                                           "unit_of_measure": "Index"},
                         frequency="monthly", transforms=["mom", "yoy"], publication_lag_days=28)
    months = [(f"2025 {m}", 100.0) for m in ("Apr", "May", "Jun")] + [(f"2026 {m}", 102.0) for m in ("Apr", "May", "Jun")]
    cpi = validated(tmp_path / "cpi", cpi_spec, singstat_payload({"1": months}), "json", {"frequency": "Monthly"})
    real = real_yoy(target, compute_changes(target), cpi)
    nominal = 219.4 / 213.2
    assert real.value == pytest.approx((nominal / 1.02 - 1) * 100)
    assert len(real.inputs) == 8


def test_real_change_refuses_an_unadmitted_cpi_month(tmp_path):
    target = validated(tmp_path, make_spec(), fixture_bytes("tidy_quarter.csv"))
    cpi_spec = make_spec(id="cpi", source={"source": "singstat", "table_id": "M213751", "series": "1",
                                           "unit_of_measure": "Index"},
                         frequency="monthly", transforms=["mom", "yoy"], publication_lag_days=28)
    months = [(f"2025 {m}", 100.0) for m in ("Apr", "May", "Jun")] + [("2026 Apr", 102.0), ("2026 May", 102.0)]
    cpi = validated(tmp_path / "cpi", cpi_spec, singstat_payload({"1": months}), "json", {"frequency": "Monthly"})
    real = real_yoy(target, compute_changes(target), cpi)
    assert real.value is None and "not yet admitted" in real.note


@pytest.mark.parametrize("effect,direction,expected", [
    (Effect.NEGATIVE, -1, "supportive"), (Effect.NEGATIVE, 1, "headwind"), (Effect.POSITIVE, 1, "supportive"),
    (Effect.POSITIVE, 0, "neutral"), (Effect.AMBIGUOUS, 1, "ambiguous"), (Effect.POSITIVE, None, "unavailable"),
])
def test_signal_readings(effect, direction, expected):
    assert read(effect, direction) == expected


def test_neutral_band():
    assert direction_of(9.9, 10) == 0 and direction_of(-10.1, 10) == -1 and direction_of(None, 1) is None


def test_signals_per_market(tmp_path):
    spec = rate_spec(source={"source": "singstat", "table_id": "M700071", "series": "23",
                             "unit_of_measure": "Per Cent Per Annum"})
    content = singstat_payload({"23": [("2025 Aug", 1.5638), ("2026 Aug", 1.1863)]}, uom="Per Cent Per Annum")
    changes = compute_changes(validated(tmp_path, spec, content, "json", {"frequency": "Monthly"}))
    readings = {s.market: s.reading for s in signals_for(spec, changes)}
    assert readings == {Market.SALES: "supportive", Market.RENTS: "ambiguous"}


def test_formatting_never_prints_negative_zero():
    assert signed(-0.04, 1) == "0.0" and format_change(-0.049, "pct") == "0.0%"
    assert format_change(-37.75, "bp") == "-38 bp" and format_change(0.021, "pp") == "+0.02 pp"
    assert format_change(None, "pct") == "n/a"
    assert [ordinal_suffix(n) for n in (1, 2, 3, 11, 12, 13, 21, 100)] == \
        ["1st", "2nd", "3rd", "11th", "12th", "13th", "21st", "100th"]


def test_trace_recomputes_an_undefined_change_as_undefined():
    """The independent recomputation shares the transform layer's domain: a zero
    base gives n/a there and None here, not a ZeroDivisionError."""
    from sgmacro.trace import recompute

    assert recompute({"kind": "change", "unit": "pct"}, [0.0, 5.0]) is None
    assert recompute({"kind": "change", "unit": "pct", "extra": {"transform": "t4q_yoy"}},
                     [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]) is None
    assert recompute({"kind": "change", "unit": "pct"}, [4.0, 5.0]) == 25.0
