"""Independent re-verification of a reported figure from the raw bytes.

``sgmacro trace F:xxxxxxxx`` does not trust any intermediate file. For each raw
observation a fact depends on it

1. re-hashes the snapshot file and compares it with the recorded sha256,
2. re-reads the single cell named by the locator with a minimal reader written
   separately from :mod:`sgmacro.parse`, and
3. recomputes the figure from the re-read inputs with the formula for its kind.

The test suite runs the same procedure over every fact in the report, so the
production arithmetic is checked by a second, independent implementation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sgmacro.periods import parse_period
from sgmacro.registry import CuratedRef, DataGovRef, SeriesSpec, SingStatRef

_SINGSTAT = re.compile(r'seriesNo=(?P<series>\S+) key="(?P<key>[^"]+)"')
_CSV_LINES = re.compile(r"csv lines? (?P<lines>[\d,]+)(?: column \"(?P<column>[^\"]+)\")?")


class TraceError(RuntimeError):
    pass


@dataclass
class InputCheck:
    period: str
    recorded: float | None
    reread: float | None
    locator: str
    file: str
    hash_ok: bool

    @property
    def ok(self) -> bool:
        return self.hash_ok and _close(self.recorded, self.reread)


def _close(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def snapshot_file(store_root: Path, inp: dict[str, Any]) -> Path:
    prefix = inp["object_key"].split("/", 1)[0]
    name = {"singstat": "data.json", "curated": "data.yaml"}.get(prefix, "data.csv")
    return Path(store_root) / inp["object_key"] / inp["vintage_id"] / name


def reread(spec: SeriesSpec, store_root: Path, inp: dict[str, Any]) -> InputCheck:
    path = snapshot_file(store_root, inp)
    if not path.exists():
        raise TraceError(f"snapshot file missing: {path}")
    content = path.read_bytes()
    hash_ok = hashlib.sha256(content).hexdigest() == inp["data_sha256"]
    locator = inp["locator"]
    if locator == "(absent)":
        value = None
    else:
        parts = [p.strip() for p in locator.split(" / ")]
        values = [_read_cell(spec, content, part) for part in parts]
        if len(values) == 1:
            value = values[0]
        else:
            num, den = values
            value = None if num is None or den in (None, 0) else num / den * 100.0
    return InputCheck(inp["period"], inp["value"], value, locator, path.as_posix(), hash_ok)


def _read_cell(spec: SeriesSpec, content: bytes, locator: str) -> float | None:
    ref = spec.source
    if isinstance(ref, CuratedRef):
        # Recount the events from the raw list, independently of the parser.
        import yaml

        m = re.fullmatch(r"quarter=(\d{4})-Q([1-4])", locator)
        if not m:
            raise TraceError(f"unrecognised curated locator {locator!r}")
        year, q = int(m.group(1)), int(m.group(2))
        events = yaml.safe_load(content.decode("utf-8"))["events"]
        signs = {"tightening": 1, "easing": -1}
        return float(sum(signs[e["direction"]] for e in events
                         if str(e["announced"])[:4] == str(year) and (int(str(e["announced"])[5:7]) - 1) // 3 + 1 == q))
    if isinstance(ref, SingStatRef):
        m = _SINGSTAT.fullmatch(locator)
        if not m:
            raise TraceError(f"unrecognised SingStat locator {locator!r}")
        rows = json.loads(content.decode("utf-8-sig"))["Data"]["row"]
        row = next(r for r in rows if str(r["seriesNo"]) == m["series"])
        cell = next(c for c in row["columns"] if str(c["key"]) == m["key"])
        return _float(cell["value"])
    assert isinstance(ref, DataGovRef)
    m = _CSV_LINES.fullmatch(locator)
    if not m:
        raise TraceError(f"unrecognised CSV locator {locator!r}")
    lines = [int(x) for x in m["lines"].split(",")]
    table = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    header = table[0]
    if m["column"]:
        (line,) = lines
        return _float(table[line - 1][header.index(m["column"])])
    column = header.index(ref.value_column)  # type: ignore[arg-type]
    values = [_float(table[line - 1][column]) for line in lines]
    if any(v is None for v in values):
        return None
    return float(sum(values))  # type: ignore[arg-type]


def _float(raw: Any) -> float | None:
    text = str(raw).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def recompute(fact: dict[str, Any], values: list[float | None]) -> float | None:
    """Recompute a fact from re-read input values, by kind."""
    if any(v is None for v in values):
        return None
    kind, unit = fact["kind"], fact["unit"]
    if kind == "level":
        return values[0]
    # Same domain as the transform layer: a percentage change from a zero or
    # negative base is undefined, so it recomputes to None rather than dividing.
    if kind == "real_change":
        base, cur, *cpi = values
        then, now = sum(cpi[:3]) / 3.0, sum(cpi[3:]) / 3.0
        if base <= 0 or then <= 0:
            return None
        return ((cur / base) / (now / then) - 1.0) * 100.0
    if fact.get("extra", {}).get("transform") == "t4q_sum":
        return float(sum(values))
    if fact.get("extra", {}).get("transform") == "t4q_yoy":
        prior = sum(values[:4])
        return None if prior <= 0 else (sum(values[4:]) / prior - 1.0) * 100.0
    base, cur = values
    if unit == "pct":
        return None if base <= 0 else (cur / base - 1.0) * 100.0
    if unit == "pp":
        return cur - base
    if unit == "bp":
        return (cur - base) * 100.0
    raise TraceError(f"cannot recompute fact kind {kind!r}")


def spans_break(spec: SeriesSpec, fact: dict[str, Any]) -> bool:
    """Does a change's window cross a declared series break? Judged here from the
    fact's own input periods, not from the transform layer's note: a change from a
    period before the break to one on or after it compares two definitions."""
    if not spec.breaks:
        return False
    own = [parse_period(i["period"], spec.frequency).ordinal for i in fact["inputs"]
           if i["series_id"] == fact["series_id"]]
    if len(own) < 2:
        return False
    lo, hi = min(own), max(own)
    return any(lo < parse_period(b.period, spec.frequency).ordinal <= hi for b in spec.breaks)


def verify_fact(fact: dict[str, Any], specs: dict[str, SeriesSpec],
                store_root: Path) -> tuple[bool, list[InputCheck], float | None]:
    """Re-read every input of ``fact`` (each with its own series' layout) and recompute."""
    checks = [reread(specs[inp["series_id"]], store_root, inp) for inp in fact["inputs"]]
    spec = specs[fact["series_id"]]
    recomputed = None if spans_break(spec, fact) else recompute(fact, [c.reread for c in checks])
    value = fact["value"]
    ok = all(c.ok for c in checks) and (
        (value is None and recomputed is None)
        or (value is not None and recomputed is not None and math.isclose(value, recomputed, rel_tol=1e-9, abs_tol=1e-9))
    )
    return ok, checks, recomputed
