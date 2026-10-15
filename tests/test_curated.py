"""The hand-maintained policy event list, and series breaks."""

from __future__ import annotations

from datetime import date

import pytest

from sgmacro.curated import CuratedError, load_policy_events
from sgmacro.facts import format_change
from sgmacro.parse import ParseError, extract
from sgmacro.registry import CuratedRef, Frequency, Transform
from sgmacro.transform import change_history, compute_changes
from sgmacro.validate import Severity, SourceView, validate_series
from tests.conftest import PREDICTOR_FIELDS, ROOT, fixture_bytes, make_spec, quarters, state_for, store_vintage

EVENTS = b"""
reviewed_on: 2024-10-02
coverage_start: 2023-01-01
events:
  - {announced: 2023-04-26, direction: tightening, instruments: [ABSD], summary: up, source: https://example.gov.sg/a}
  - {announced: 2024-08-19, direction: tightening, instruments: [LTV], summary: down, source: https://example.gov.sg/b}
  - {announced: 2024-09-02, direction: easing, instruments: [TDSR], summary: eased, source: https://example.gov.sg/c}
"""


def policy_spec(**kw):
    fields = {**PREDICTOR_FIELDS, "id": "policy", "source": {"source": "curated", "path": "events.yaml"},
              "unit": "net rounds", "unit_kind": "count", "publication_lag_days": 0, "plausible_range": [-2, 3],
              "transforms": ["t4q_sum"], "headline": "t4q_sum", "decimals": 0,
              "effects": {"sales": "negative", "rents": "ambiguous"}, "neutral_band": 0.5}
    fields.update(kw)
    return make_spec(**fields)


def test_events_become_net_rounds_per_quarter_with_zeros():
    ex = extract(CuratedRef(path="events.yaml"), EVENTS, Frequency.QUARTERLY)
    values = dict(zip(ex.frame["period"], ex.frame["value"]))
    assert values["2023-Q1"] == 0 and values["2023-Q2"] == 1
    assert values["2024-Q3"] == 0            # one tightening and one easing in the same quarter
    assert ex.frame["period"].iloc[-1] == "2024-Q3"   # 2024-Q4 had not ended by the review date
    assert ex.frame["locator"].iloc[1] == "quarter=2023-Q2"


@pytest.mark.parametrize("bad, message", [
    (EVENTS.replace(b"direction: easing", b"direction: loosening"), "direction"),
    (EVENTS.replace(b"source: https://example.gov.sg/c", b"source: a blog"), "official https source"),
    (EVENTS.replace(b"announced: 2024-09-02", b"announced: 2025-01-01"), "outside"),
])
def test_malformed_lists_are_rejected(bad, message):
    with pytest.raises(CuratedError, match=message):
        load_policy_events(bad)
    with pytest.raises(ParseError):
        extract(CuratedRef(path="events.yaml"), bad, Frequency.QUARTERLY)


def test_count_unit_only_takes_trailing_sums():
    with pytest.raises(Exception, match="forbids"):
        policy_spec(transforms=["yoy"], headline="yoy")


def test_trailing_sum_signal_and_formatting(tmp_path):
    spec = policy_spec()
    vintage = store_vintage(tmp_path, spec.source.object_key, EVENTS, extension="yaml",
                            upstream={"last_updated": "2024-10-02"})
    view = SourceView(spec.source, state_for(vintage, after_as_of=True),
                      extract(spec.source, EVENTS, Frequency.QUARTERLY))
    result = validate_series(spec, view, as_of=date(2024, 10, 1))
    # a list compiled after the as-of date is not a revision risk: events carry their own dates
    assert not result.by_severity(Severity.WARN)
    changes = compute_changes(result)
    head = changes.headline
    # 2024-Q3 holds one tightening and one easing; 2023-Q2's round is outside the window
    assert head.value == 0.0 and [i.period for i in head.inputs] == ["2023-Q4", "2024-Q1", "2024-Q2", "2024-Q3"]
    from sgmacro.signals import signals_for
    assert {s.market.value: s.reading for s in signals_for(spec, changes)} == {"sales": "neutral", "rents": "ambiguous"}
    assert format_change(1.0, "count") == "+1 round" and format_change(head.value, "count") == "0 rounds"
    assert format_change(-2.0, "count") == "-2 rounds"


def test_committed_policy_list_is_valid_and_sourced():
    events = load_policy_events((ROOT / "config" / "policy_events.yaml").read_bytes())
    assert len(events.events) >= 20 and events.coverage_start == date(2005, 1, 1)
    assert all(e.source.startswith("https://") and e.source.split("/")[2].endswith(".gov.sg") for e in events.events)


def test_changes_spanning_a_series_break_are_excluded(tmp_path):
    spec = make_spec(breaks=[{"period": "2025-Q1", "note": "coverage widened"}])
    values = [100 + i for i in range(14)]   # 2023-Q1 .. 2026-Q2
    content = b"quarter,property_type,index\n" + b"".join(
        f"{p},All Residential,{v}\n".encode() for p, v in quarters(2023, values))
    history = change_history(spec, Transform.YOY, extract(spec.source, content, Frequency.QUARTERLY).values())
    by_label = {f"{o // 4}-Q{o % 4 + 1}": v for o, v in history.items()}
    assert by_label["2024-Q4"] == pytest.approx((107 / 103 - 1) * 100)   # entirely before the break
    for label in ("2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"):          # window crosses 2025-Q1
        assert by_label[label] != by_label[label]                         # NaN
    assert by_label["2026-Q1"] == pytest.approx((112 / 108 - 1) * 100)   # entirely after
    vintage = store_vintage(tmp_path, spec.source.object_key, content)
    result = validate_series(spec, SourceView(spec.source, state_for(vintage),
                                              extract(spec.source, content, Frequency.QUARTERLY)),
                             as_of=date(2026, 6, 1))
    assert any("series break at 2025-Q1" in f.message for f in result.findings)


def test_break_period_must_match_frequency():
    with pytest.raises(Exception):
        make_spec(breaks=[{"period": "2025-07", "note": "monthly label on a quarterly series"}])
