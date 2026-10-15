"""Canonical observation periods.

The two APIs label time in six different ways::

    data.gov.sg tidy        "2026-Q2"
    data.gov.sg wide        "20262Q", "2026Jul"
    SingStat quarterly      "2026 1Q"
    SingStat monthly        "2026 Aug"
    SingStat annual         "2025"

Everything is parsed into one :class:`Period` so that ordering, lagging and the
publication-lag gate have a single implementation. Periods are comparable only
within a frequency: "is 2026-M07 before 2026-Q3" has no single right answer, and
silently picking one is how off-by-one-quarter bugs get in.

The declared frequency is always required when parsing. ``"2026"`` is a valid
annual label and a plausible truncation of a quarterly one; guessing would turn
a configuration error into a wrong number instead of a loud failure.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from functools import total_ordering

from sgmacro.registry import Frequency

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})


class PeriodParseError(ValueError):
    """A label did not match the declared frequency. Deliberately fatal."""


@total_ordering
@dataclass(frozen=True)
class Period:
    """A month, quarter or year. ``index`` is 1-12, 1-4 or always 1."""

    freq: Frequency
    year: int
    index: int = 1

    def __post_init__(self) -> None:
        limit = self.freq.periods_per_year
        if not 1 <= self.index <= limit:
            raise PeriodParseError(f"index {self.index} out of range 1..{limit} for {self.freq.value}")
        if not 1900 <= self.year <= 2200:
            raise PeriodParseError(f"implausible year {self.year}")

    @property
    def ordinal(self) -> int:
        return self.year * self.freq.periods_per_year + (self.index - 1)

    @classmethod
    def from_ordinal(cls, freq: Frequency, ordinal: int) -> Period:
        ppy = freq.periods_per_year
        return cls(freq, ordinal // ppy, ordinal % ppy + 1)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Period):
            return NotImplemented
        if self.freq is not other.freq:
            raise TypeError(f"refusing to compare {self.freq.value} with {other.freq.value}")
        return self.ordinal < other.ordinal

    def shift(self, steps: int) -> Period:
        return Period.from_ordinal(self.freq, self.ordinal + steps)

    @property
    def end_date(self) -> date:
        """Last calendar day the period covers."""
        if self.freq is Frequency.ANNUAL:
            return date(self.year, 12, 31)
        month = self.index if self.freq is Frequency.MONTHLY else self.index * 3
        return date(self.year, month, calendar.monthrange(self.year, month)[1])

    def reference_date(self, reference_month: int | None = None) -> date:
        """The date the observation describes.

        Period end by default. An annual series measured at end-June refers to
        30 June, and treating it as 31 December would withhold a figure for six
        months after it was actually published.
        """
        if reference_month is None or self.freq is not Frequency.ANNUAL:
            return self.end_date
        return date(self.year, reference_month, calendar.monthrange(self.year, reference_month)[1])

    def to_quarter(self, reference_month: int | None = None) -> Period:
        """The quarter containing this period's reference date."""
        if self.freq is Frequency.QUARTERLY:
            return self
        month = self.reference_date(reference_month).month
        return Period(Frequency.QUARTERLY, self.year, (month - 1) // 3 + 1)

    def months(self) -> list[Period]:
        """Calendar months covered by a quarter (used to average monthly CPI)."""
        if self.freq is not Frequency.QUARTERLY:
            raise ValueError("months() is defined for quarters only")
        first = (self.index - 1) * 3 + 1
        return [Period(Frequency.MONTHLY, self.year, m) for m in range(first, first + 3)]

    @property
    def label(self) -> str:
        if self.freq is Frequency.ANNUAL:
            return f"{self.year}"
        if self.freq is Frequency.QUARTERLY:
            return f"{self.year}-Q{self.index}"
        return f"{self.year}-{self.index:02d}"

    @property
    def display(self) -> str:
        """Human label for the report: ``Aug 2026``, ``2026 Q2``, ``2025``."""
        if self.freq is Frequency.MONTHLY:
            return f"{calendar.month_abbr[self.index]} {self.year}"
        if self.freq is Frequency.QUARTERLY:
            return f"{self.year} Q{self.index}"
        return str(self.year)

    def __str__(self) -> str:
        return self.label


_QUARTER_PATTERNS = (
    re.compile(r"^(\d{4})[-\s]?Q([1-4])$", re.I),    # 2026-Q2, 2026Q2, 2026 Q2
    re.compile(r"^(\d{4})\s+([1-4])Q$", re.I),       # 2026 2Q (SingStat)
    re.compile(r"^(\d{4})([1-4])Q$", re.I),          # 20262Q (data.gov.sg wide)
)
_MONTH_NAMED = re.compile(r"^(\d{4})\s*([A-Za-z]{3,9})$")   # 2026Jul, 2026 Aug
_MONTH_NUMERIC = re.compile(r"^(\d{4})-(?:M)?(\d{2})$")      # 2026-07, 2026-M07
_YEAR = re.compile(r"^(\d{4})$")


def parse_period(label: object, freq: Frequency) -> Period:
    """Parse any supported label into a :class:`Period` of the declared frequency."""
    if not isinstance(label, str):
        raise PeriodParseError(f"expected a string period label, got {type(label).__name__}")
    text = label.strip()
    if not text:
        raise PeriodParseError("empty period label")

    if freq is Frequency.QUARTERLY:
        for pattern in _QUARTER_PATTERNS:
            m = pattern.match(text)
            if m:
                return Period(freq, int(m.group(1)), int(m.group(2)))
        raise PeriodParseError(f"not a quarterly label: {label!r}")

    if freq is Frequency.MONTHLY:
        m = _MONTH_NUMERIC.match(text)
        if m:
            return Period(freq, int(m.group(1)), int(m.group(2)))
        m = _MONTH_NAMED.match(text)
        if m:
            month = _MONTHS.get(m.group(2).lower())
            if month is None:
                raise PeriodParseError(f"unrecognised month name in {label!r}")
            return Period(freq, int(m.group(1)), month)
        raise PeriodParseError(f"not a monthly label: {label!r}")

    m = _YEAR.match(text)
    if m:
        return Period(freq, int(m.group(1)), 1)
    raise PeriodParseError(f"not an annual label: {label!r}")
