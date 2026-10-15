from __future__ import annotations

from datetime import date

import pytest

from sgmacro.periods import Period, PeriodParseError, parse_period
from sgmacro.registry import Frequency

Q, M, A = Frequency.QUARTERLY, Frequency.MONTHLY, Frequency.ANNUAL


@pytest.mark.parametrize("label", ["2026-Q2", "2026Q2", "2026 2Q", "20262Q", " 2026-q2 "])
def test_quarter_dialects(label):
    assert parse_period(label, Q) == Period(Q, 2026, 2)


@pytest.mark.parametrize("label", ["2026Jul", "2026 Jul", "2026-07", "2026-M07", "2026 July"])
def test_month_dialects(label):
    assert parse_period(label, M) == Period(M, 2026, 7)


def test_annual():
    assert parse_period("2025", A).end_date == date(2025, 12, 31)


@pytest.mark.parametrize("label,freq", [("2026", Q), ("2026-Q5", Q), ("2026Foo", M), ("2026-Q1", A), ("", M)])
def test_wrong_dialect_is_fatal(label, freq):
    with pytest.raises(PeriodParseError):
        parse_period(label, freq)


def test_shift_and_end_dates():
    assert Period(Q, 2026, 1).shift(-1) == Period(Q, 2025, 4)
    assert Period(M, 2026, 1).shift(-12) == Period(M, 2025, 1)
    assert Period(M, 2024, 2).end_date == date(2024, 2, 29)
    assert Period(Q, 2026, 2).end_date == date(2026, 6, 30)


def test_reference_date_for_mid_year_annual_series():
    """End-June population estimates refer to 30 June, not 31 December."""
    p = Period(A, 2026)
    assert p.reference_date(6) == date(2026, 6, 30)
    assert p.reference_date() == date(2026, 12, 31)
    assert p.to_quarter(6) == Period(Q, 2026, 2)


def test_cross_frequency_comparison_refuses():
    with pytest.raises(TypeError):
        _ = Period(M, 2026, 7) < Period(Q, 2026, 3)


def test_quarter_months():
    assert [m.label for m in Period(Q, 2026, 2).months()] == ["2026-04", "2026-05", "2026-06"]


def test_display_labels():
    assert Period(M, 2026, 8).display == "Aug 2026"
    assert Period(Q, 2026, 2).display == "2026 Q2"
