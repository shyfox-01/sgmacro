"""Raw snapshot bytes -> one tidy observation table per source reference.

Every observation keeps a *locator*: the exact place in the raw snapshot it was
read from (a SingStat series number and period key, or CSV line numbers). That
is what lets ``sgmacro trace`` re-open the original bytes, re-read the cell and
confirm the reported figure without trusting any intermediate file.

Design rule: anything unexpected raises :class:`ParseError`. A filter that
matches nothing, a label that matches twice, an unparseable period, a duplicate
period, or a value that is neither numeric nor a recognised missing-data marker
all stop the series. A series that silently loses observations produces a report
that looks fine and is wrong, which is the worse failure.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sgmacro.curated import CuratedError, load_policy_events
from sgmacro.periods import Period, PeriodParseError, parse_period
from sgmacro.registry import CuratedRef, DataGovRef, Frequency, SingStatRef, SourceRef

#: Markers these publishers use for "no observation". Anything else that does not
#: parse as a number is an error, not a silent gap.
MISSING_MARKERS = frozenset({"", "-", "na", "n.a.", "n/a", "nil", "null", ".."})

COLUMNS = ["period", "ordinal", "value", "locator"]


class ParseError(ValueError):
    """Raw content did not match what the registry declares."""


@dataclass
class Extracted:
    ref: SourceRef
    frequency: Frequency
    frame: pd.DataFrame
    """Columns ``period`` (label), ``ordinal``, ``value`` (NaN when missing),
    ``locator``; one row per period, ascending."""
    publisher_units: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def period(self, ordinal: int) -> Period:
        return Period.from_ordinal(self.frequency, int(ordinal))

    @property
    def populated(self) -> pd.DataFrame:
        return self.frame[self.frame["value"].notna()]

    def latest(self) -> tuple[Period, float] | None:
        populated = self.populated
        if populated.empty:
            return None
        row = populated.iloc[-1]
        return self.period(row["ordinal"]), float(row["value"])

    def values(self) -> pd.Series:
        """Values indexed by ordinal."""
        return self.frame.set_index("ordinal")["value"]


def extract(ref: SourceRef, content: bytes, frequency: Frequency,
            upstream: dict[str, Any] | None = None) -> Extracted:
    if isinstance(ref, SingStatRef):
        return _singstat(ref, content, frequency, upstream or {})
    if isinstance(ref, CuratedRef):
        return _curated(ref, content, frequency)
    if ref.layout == "tidy":
        return _tidy(ref, content, frequency)
    return _wide(ref, content, frequency)


# ---------------------------------------------------------------------------
# SingStat JSON
# ---------------------------------------------------------------------------

def _singstat(ref: SingStatRef, content: bytes, freq: Frequency, upstream: dict[str, Any]) -> Extracted:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParseError(f"{ref.table_id}: snapshot is not JSON ({exc})") from exc
    data = payload.get("Data") if isinstance(payload, dict) else None
    rows = (data.get("row") if isinstance(data, dict) else None) or []
    # The client checks this shape before storing; a snapshot stored by an older
    # client may not have been, and a bad one must reject the series, not the run.
    if not isinstance(rows, list) or not all(
            isinstance(r, dict) and isinstance(r.get("columns") or [], list)
            and all(isinstance(c, dict) for c in r.get("columns") or []) for r in rows):
        raise ParseError(f"{ref.table_id}: snapshot rows are not objects with object columns")
    if not rows:
        raise ParseError(f"{ref.table_id}: snapshot has no rows")

    series_values: dict[str, dict[int, tuple[float | None, str]]] = {}
    units: dict[str, str | None] = {}
    meta_series = upstream.get("series") or {}
    for number in ref.series_numbers:
        hits = [r for r in rows if str(r.get("seriesNo")) == number]
        if not hits:
            present = sorted(str(r.get("seriesNo")) for r in rows)
            raise ParseError(f"{ref.table_id}: series {number!r} absent; present {present[:12]}")
        if len(hits) > 1:
            raise ParseError(f"{ref.table_id}: series {number!r} appears {len(hits)} times")
        row = hits[0]
        # The metadata endpoint is the authoritative unit; the tabledata row also
        # carries uoM and is used when metadata was not stored.
        units[number] = (meta_series.get(number) or {}).get("unit_of_measure") or row.get("uoM")
        values: dict[int, tuple[float | None, str]] = {}
        for cell in row.get("columns") or []:
            key = str(cell.get("key"))
            period = _period(key, freq, ref.table_id)
            if period.ordinal in values:
                raise ParseError(f"{ref.table_id}: series {number} repeats period {period.label}")
            values[period.ordinal] = (_number(cell.get("value"), ref.table_id, key),
                                      f'seriesNo={number} key="{key}"')
        series_values[number] = values

    if ref.series is not None:
        records = [(o, v, loc) for o, (v, loc) in series_values[ref.series].items()]
    else:
        num, den = ref.ratio  # type: ignore[misc]
        records = []
        for ordinal in sorted(set(series_values[num]) | set(series_values[den])):
            n, n_loc = series_values[num].get(ordinal, (None, ""))
            d, d_loc = series_values[den].get(ordinal, (None, ""))
            value = None if n is None or d is None or d == 0 else n / d * 100.0
            records.append((ordinal, value, f"{n_loc} / {d_loc}".strip(" /")))
    return Extracted(ref, freq, _frame(records, freq), publisher_units=units)


# ---------------------------------------------------------------------------
# data.gov.sg tidy CSV
# ---------------------------------------------------------------------------

def _tidy(ref: DataGovRef, content: bytes, freq: Frequency) -> Extracted:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    header = reader.fieldnames or []
    needed = [ref.period_column, ref.value_column, *ref.filters]
    absent = [c for c in needed if c not in header]
    if absent:
        raise ParseError(f"{ref.dataset_id}: columns {absent} absent; the file has {header}. "
                         "The upstream schema may have changed.")

    groups: dict[int, list[tuple[int, str]]] = {}
    components: dict[int, list[tuple[str, ...]]] = {}
    labels: dict[int, str] = {}
    # A sum adds the rows that share a period; the remaining columns say which
    # component each row is (sale type, market segment, ...).
    component_columns = [c for c in header if c not in needed]
    matched = 0
    for row in reader:
        line = reader.line_num
        if not any((v or "").strip() for v in row.values()):
            continue
        if any((row.get(k) or "").strip() != v for k, v in ref.filters.items()):
            continue
        matched += 1
        label = (row.get(ref.period_column) or "").strip()
        period = _period(label, freq, ref.dataset_id)
        groups.setdefault(period.ordinal, []).append((line, row.get(ref.value_column) or ""))
        components.setdefault(period.ordinal, []).append(
            tuple((row.get(c) or "").strip() for c in component_columns))
        labels[period.ordinal] = label
    if not matched:
        raise ParseError(f"{ref.dataset_id}: filters {ref.filters} matched no rows")

    records: list[tuple[int, float | None, str]] = []
    notes: list[str] = []
    incomplete: list[str] = []
    if ref.aggregate == "sum":
        for ordinal, keys in components.items():
            if len(keys) != len(set(keys)):
                repeated = sorted({k for k in keys if keys.count(k) > 1})
                raise ParseError(f"{ref.dataset_id}: component {repeated[0]} appears more than once in "
                                 f"{labels[ordinal]}; a total would count it twice")
        # Every complete period must add up the same components. If they differ
        # (one missing, another present), the totals are not of the same thing, and
        # which set is "right" is a question for a person, not for a vote over the
        # history: stop the series and name the periods.
        complete = {o: frozenset(k) for o, k in components.items() if len(k) == ref.rows_per_period}
        if len(set(complete.values())) > 1:
            first = min(complete)
            other = min(o for o, k in complete.items() if k != complete[first])
            raise ParseError(
                f"{ref.dataset_id}: {labels[first]} sums {sorted(complete[first])} but {labels[other]} sums "
                f"{sorted(complete[other])}; the components changed, so the total is not comparable")
    for ordinal in sorted(groups):
        cells = groups[ordinal]
        lines = [line for line, _ in cells]
        locator = f"csv line {lines[0]}" if len(lines) == 1 else f"csv lines {','.join(map(str, lines))}"
        if ref.aggregate == "single":
            if len(cells) > 1:
                raise ParseError(
                    f"{ref.dataset_id}: {len(cells)} rows for {labels[ordinal]} under filters "
                    f"{ref.filters}; the selector does not identify a single series"
                )
            records.append((ordinal, _number(cells[0][1], ref.dataset_id, labels[ordinal]), locator))
            continue
        values = [_number(raw, ref.dataset_id, labels[ordinal]) for _, raw in cells]
        if len(cells) != ref.rows_per_period or any(v is None for v in values):
            incomplete.append(labels[ordinal])
            records.append((ordinal, None, locator))
        else:
            records.append((ordinal, float(sum(values)), locator))  # type: ignore[arg-type]
    if incomplete:
        notes.append(
            f"{len(incomplete)} period(s) lack the {ref.rows_per_period} component rows a total "
            f"needs and are left empty rather than summed: {incomplete[:4]}"
            + ("..." if len(incomplete) > 4 else "")
        )
    return Extracted(ref, freq, _frame(records, freq), notes=notes)


# ---------------------------------------------------------------------------
# data.gov.sg wide CSV (one row per series, one column per period)
# ---------------------------------------------------------------------------

def _wide(ref: DataGovRef, content: bytes, freq: Frequency) -> Extracted:
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    table = [(i + 1, r) for i, r in enumerate(reader) if r and any(c.strip() for c in r)]
    if len(table) < 2:
        raise ParseError(f"{ref.dataset_id}: wide table has no data rows")
    (_, header), body = table[0], table[1:]
    if header[0].strip() != "DataSeries":
        raise ParseError(f"{ref.dataset_id}: expected first column 'DataSeries', got {header[0]!r}")
    period_labels = [h.strip() for h in header[1:]]
    ragged = [r[0] for _, r in body if len(r) != len(header)]
    if ragged:
        raise ParseError(f"{ref.dataset_id}: rows {ragged[:3]} do not match the header width")

    def row_for(label: str) -> tuple[int, list[str]]:
        hits = [(line, r) for line, r in body if r[0].strip() == label.strip()]
        if not hits:
            raise ParseError(f"{ref.dataset_id}: no row labelled {label!r}; "
                             f"available {[r[0].strip() for _, r in body][:10]}")
        if len(hits) > 1:
            raise ParseError(
                f"{ref.dataset_id}: row label {label!r} matches {len(hits)} rows; this table "
                "repeats child labels under several parents, so the selector is ambiguous"
            )
        return hits[0][0], hits[0][1][1:]

    def series_for(label: str) -> dict[int, tuple[float | None, str]]:
        line, cells = row_for(label)
        out: dict[int, tuple[float | None, str]] = {}
        for raw_label, raw in zip(period_labels, cells):
            period = _period(raw_label, freq, ref.dataset_id)
            if period.ordinal in out:
                raise ParseError(f"{ref.dataset_id}: column {raw_label!r} repeats a period")
            out[period.ordinal] = (_number(raw, ref.dataset_id, raw_label),
                                   f'csv line {line} column "{raw_label}"')
        return out

    if ref.row_label:
        records = [(o, v, loc) for o, (v, loc) in series_for(ref.row_label).items()]
    else:
        num_label, den_label = ref.ratio_labels  # type: ignore[misc]
        num, den = series_for(num_label), series_for(den_label)
        records = []
        for ordinal in sorted(set(num) | set(den)):
            n, n_loc = num.get(ordinal, (None, ""))
            d, d_loc = den.get(ordinal, (None, ""))
            value = None if n is None or d is None or d == 0 else n / d * 100.0
            records.append((ordinal, value, f"{n_loc} / {d_loc}"))
    return Extracted(ref, freq, _frame(records, freq))


# ---------------------------------------------------------------------------
# curated event list -> net rounds per quarter
# ---------------------------------------------------------------------------

def _curated(ref: CuratedRef, content: bytes, freq: Frequency) -> Extracted:
    """Net policy rounds (tightening +1, easing -1) per quarter, zeros included,
    from the coverage start to the last quarter fully covered by the review."""
    if freq is not Frequency.QUARTERLY:
        raise ParseError(f"{ref.path}: the policy event list is aggregated to quarters only")
    try:
        events = load_policy_events(content)
    except CuratedError as exc:
        raise ParseError(f"{ref.path}: {exc}") from exc
    first, last = events.first_quarter, events.last_complete_quarter
    records = []
    for ordinal in range(first.ordinal, last.ordinal + 1):
        quarter = Period.from_ordinal(freq, ordinal)
        records.append((ordinal, float(events.net_rounds(quarter)), f"quarter={quarter.label}"))
    return Extracted(ref, freq, _frame(records, freq))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _frame(records: list[tuple[int, float | None, str]], freq: Frequency) -> pd.DataFrame:
    ordered = sorted(records, key=lambda r: r[0])
    frame = pd.DataFrame(
        {
            "period": [Period.from_ordinal(freq, o).label for o, _, _ in ordered],
            "ordinal": pd.array([o for o, _, _ in ordered], dtype="int64"),
            "value": pd.array([math.nan if v is None else v for _, v, _ in ordered], dtype="float64"),
            "locator": [loc for _, _, loc in ordered],
        },
        columns=COLUMNS,
    )
    return frame.reset_index(drop=True)


def _period(label: str, freq: Frequency, where: str) -> Period:
    try:
        return parse_period(label, freq)
    except PeriodParseError as exc:
        raise ParseError(f"{where}: {exc}") from exc


def _number(raw: Any, where: str, period_label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "")
    if text.lower() in MISSING_MARKERS:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ParseError(
            f"{where}: value {raw!r} at {period_label!r} is neither numeric nor a recognised "
            f"missing-data marker {sorted(MISSING_MARKERS)}"
        ) from exc
    if not math.isfinite(value):
        raise ParseError(f"{where}: non-finite value {raw!r} at {period_label!r}")
    return value
