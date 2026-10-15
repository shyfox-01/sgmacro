"""Parsing: strict by design, with a raw-cell locator on every observation."""

from __future__ import annotations

import json
import math

import pytest

from sgmacro.parse import ParseError, extract
from sgmacro.registry import DataGovRef, Frequency, SingStatRef
from tests.conftest import fixture_bytes, singstat_payload

Q, M = Frequency.QUARTERLY, Frequency.MONTHLY


def tidy(**kw):
    base = {"dataset_id": "d_x", "layout": "tidy", "value_column": "index"}
    base.update(kw)
    return DataGovRef(**base)


def test_tidy_filter_selects_one_series():
    ex = extract(tidy(filters={"property_type": "All Residential"}), fixture_bytes("tidy_quarter.csv"), Q)
    assert ex.latest()[0].label == "2026-Q2" and ex.latest()[1] == 219.4
    assert ex.frame["locator"].iloc[-1] == "csv line 20"


def test_tidy_ambiguous_filter_is_an_error():
    with pytest.raises(ParseError, match="does not identify a single series"):
        extract(tidy(), fixture_bytes("tidy_quarter.csv"), Q)


def test_schema_change_is_reported():
    with pytest.raises(ParseError, match="schema may have changed"):
        extract(tidy(value_column="price"), fixture_bytes("tidy_quarter.csv"), Q)


def test_filter_matching_nothing_is_an_error():
    with pytest.raises(ParseError, match="matched no rows"):
        extract(tidy(filters={"property_type": "Castle"}), fixture_bytes("tidy_quarter.csv"), Q)


def test_sum_aggregate_and_incomplete_periods():
    csv = (b"quarter,type,units\n"
           b"2026-Q1,a,1\n2026-Q1,b,2\n2026-Q1,c,3\n"
           b"2026-Q2,a,4\n2026-Q2,b,5\n")          # Q2 is missing component c
    ref = tidy(value_column="units", aggregate="sum", rows_per_period=3)
    ex = extract(ref, csv, Q)
    values = ex.values()
    assert values.iloc[0] == 6.0
    assert math.isnan(values.iloc[1])
    assert "left empty rather than summed" in ex.notes[0]
    assert ex.frame["locator"].iloc[0] == "csv lines 2,3,4"


def test_sum_refuses_a_component_counted_twice():
    """Right row count, wrong rows: c is missing and a appears twice. Summing would
    give a plausible but wrong total, so the series stops."""
    csv = (b"quarter,type,units\n"
           b"2026-Q1,a,1\n2026-Q1,b,2\n2026-Q1,c,3\n"
           b"2026-Q2,a,4\n2026-Q2,b,5\n2026-Q2,a,4\n")
    with pytest.raises(ParseError, match="appears more than once in 2026-Q2"):
        extract(tidy(value_column="units", aggregate="sum", rows_per_period=3), csv, Q)


def test_sum_refuses_components_that_change_between_periods():
    """Right row count, other components: the totals are not comparable. The series
    stops with both sets named; no vote over the history decides which set is right,
    so the outcome cannot depend on hash order or on periods added later."""
    csv = (b"quarter,type,units\n"
           b"2026-Q1,a,1\n2026-Q1,b,2\n2026-Q1,c,3\n"
           b"2026-Q2,a,1\n2026-Q2,b,2\n2026-Q2,c,3\n"
           b"2026-Q3,a,4\n2026-Q3,b,5\n2026-Q3,d,6\n")   # d instead of c
    with pytest.raises(ParseError, match="2026-Q1 sums .* but 2026-Q3 sums .*components changed"):
        extract(tidy(value_column="units", aggregate="sum", rows_per_period=3), csv, Q)
    tie = b"quarter,type,units\n2026-Q1,a,1\n2026-Q1,b,2\n2026-Q2,a,4\n2026-Q2,c,5\n"
    with pytest.raises(ParseError, match="components changed"):
        extract(tidy(value_column="units", aggregate="sum", rows_per_period=2), tie, Q)


def test_sum_with_a_missing_component_value_is_missing():
    csv = b"quarter,type,units\n2026-Q1,a,1\n2026-Q1,b,na\n"
    ex = extract(tidy(value_column="units", aggregate="sum", rows_per_period=2), csv, Q)
    assert math.isnan(ex.values().iloc[0])


def test_wide_month_descending_columns():
    ref = DataGovRef(dataset_id="d_x", layout="wide", row_label="Government Securities - 10-Year Bond Yield")
    ex = extract(ref, fixture_bytes("wide_month.csv"), M)
    assert ex.frame["period"].iloc[0] == "2025-02" and ex.latest() [1] == 2.35
    assert ex.frame["locator"].iloc[-1] == 'csv line 2 column "2026Jul"'


def test_wide_repeated_child_label_is_ambiguous():
    """The pipeline table has 'Under Construction' under landed and non-landed."""
    ref = DataGovRef(dataset_id="d_x", layout="wide", row_label="Under Construction")
    with pytest.raises(ParseError, match="ambiguous"):
        extract(ref, fixture_bytes("wide_quarter_dup_labels.csv"), Q)


def test_wide_ratio_of_two_rows():
    ref = DataGovRef(dataset_id="d_x", layout="wide",
                     ratio_labels=("Vacant Non-Landed Properties", "Available Non-Landed Properties"))
    ex = extract(ref, fixture_bytes("wide_quarter_indent.csv"), Q)
    assert ex.latest()[1] == pytest.approx(18902 / 350679 * 100)


def test_singstat_series_with_units():
    ref = SingStatRef(table_id="M213751", series="1.03.1", unit_of_measure="Index")
    ex = extract(ref, fixture_bytes("singstat_rows.json"), M)
    assert ex.publisher_units == {"1.03.1": "Index"}
    assert ex.frame["locator"].iloc[-1] == 'seriesNo=1.03.1 key="2026 Aug"'


def test_singstat_metadata_unit_takes_precedence():
    ref = SingStatRef(table_id="M1", series="5", unit_of_measure="Number")
    payload = singstat_payload({"5": [("2025", 1)]}, uom=None)
    ex = extract(ref, payload, Frequency.ANNUAL, {"series": {"5": {"unit_of_measure": "Number"}}})
    assert ex.publisher_units == {"5": "Number"}


def test_a_stored_snapshot_with_a_non_object_cell_rejects_the_series():
    bad = json.dumps({"Data": {"row": [{"seriesNo": "1", "uoM": "Index", "columns": ["x"]}]}}).encode()
    with pytest.raises(ParseError, match="object columns"):
        extract(SingStatRef(table_id="M1", series="1", unit_of_measure="Index"), bad, Q)


def test_singstat_missing_series():
    ref = SingStatRef(table_id="M213751", series="9.9", unit_of_measure="Index")
    with pytest.raises(ParseError, match="absent"):
        extract(ref, fixture_bytes("singstat_rows.json"), M)


def test_singstat_ratio_and_zero_denominator():
    ref = SingStatRef(table_id="M1", ratio=("2", "1"), unit_of_measure="Number")
    payload = singstat_payload({"1": [("2026 1Q", 200), ("2026 2Q", 0)], "2": [("2026 1Q", 10), ("2026 2Q", 5)]})
    values = extract(ref, payload, Q).values()
    assert values.iloc[0] == 5.0 and math.isnan(values.iloc[1])


@pytest.mark.parametrize("raw", ["-", "na", "", "n.a.", ".."])
def test_missing_markers_become_nan(raw):
    csv = f"quarter,index\n2026-Q1,{raw}\n2026-Q2,1\n".encode()
    assert math.isnan(extract(tidy(), csv, Q).values().iloc[0])


def test_unknown_non_numeric_value_is_an_error():
    with pytest.raises(ParseError, match="neither numeric"):
        extract(tidy(), b"quarter,index\n2026-Q1,approx 5\n", Q)


def test_duplicate_period_is_an_error():
    payload = singstat_payload({"1": [("2026 1Q", 1), ("2026 1Q", 2)]})
    with pytest.raises(ParseError, match="repeats period"):
        extract(SingStatRef(table_id="M1", series="1", unit_of_measure="Index"), payload, Q)


def test_unparseable_period_is_an_error():
    with pytest.raises(ParseError, match="not a quarterly label"):
        extract(tidy(), b"quarter,index\n2026-13,1\n", Q)
