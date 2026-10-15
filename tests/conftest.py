"""Shared helpers. No test touches the network.

Fixtures under ``tests/fixtures`` are trimmed from real upstream responses, so the
parsers meet the structures these APIs genuinely serve (hierarchy as leading
whitespace, labels repeated under several parents, periods in descending
columns). Synthetic payloads are built with the helpers below when a test needs
a precise edge case.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from sgmacro.fetch import ObjectState
from sgmacro.pipeline import RunConfig, run
from sgmacro.registry import SeriesSpec, load_registry
from sgmacro.store import Selection, SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
REGISTRY_PATH = ROOT / "config" / "sources.yaml"
STORE_ROOT = ROOT / "data" / "snapshots"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture(scope="session")
def registry():
    return load_registry(REGISTRY_PATH)


AS_OF = date(2026, 10, 2)


def config_in(tmp: Path, **kw) -> RunConfig:
    """The sample request (offline, committed snapshots and LLM cache) writing into ``tmp``."""
    base = RunConfig(as_of=AS_OF, registry_path=REGISTRY_PATH, store_root=STORE_ROOT,
                     processed_root=tmp / "processed", runs_root=tmp / "runs", reports_root=tmp / "reports",
                     llm_cache=ROOT / "llm_cache", llm="cache", offline=True)
    return replace(base, **kw)


@pytest.fixture(scope="session")
def committed_run(tmp_path_factory):
    """The sample run reproduced offline. Its run id names the committed sample: other
    runs for the same date may sit beside it in reports/ and are not assumed away."""
    tmp = tmp_path_factory.mktemp("run")
    return tmp, run(config_in(tmp), log=lambda m: None)


BASE_SPEC: dict[str, Any] = {
    "id": "test_series",
    "label": "Test series",
    "short_label": "Test",
    "role": "target",
    "markets": ["sales"],
    "publisher": "Test Agency",
    "source": {"source": "datagovsg", "dataset_id": "d_test", "layout": "tidy",
               "value_column": "index", "filters": {"property_type": "All Residential"}},
    "definition": "A test series.",
    "unit": "index",
    "unit_kind": "index",
    "frequency": "quarterly",
    "seasonal_adjustment": "not_applicable",
    "update_frequency": "quarterly",
    "publication_lag_days": 30,
    "plausible_range": [1, 1000],
    "transforms": ["qoq", "yoy"],
    "headline": "yoy",
}

PREDICTOR_FIELDS: dict[str, Any] = {
    "role": "predictor", "markets": ["sales", "rents"], "channel": "Test channel",
    "effects": {"sales": "negative", "rents": "ambiguous"}, "neutral_band": 10,
    "expected_lead_quarters": [1, 4], "rationale": "Because.", "limitations": "Some.",
}


def make_spec(**overrides: Any) -> SeriesSpec:
    data = json.loads(json.dumps(BASE_SPEC))
    data.update(overrides)
    return SeriesSpec.model_validate(data)


def rate_spec(**overrides: Any) -> SeriesSpec:
    base = {**PREDICTOR_FIELDS, "id": "rate_series",
            "source": {"source": "singstat", "table_id": "M700071", "series": "23",
                       "unit_of_measure": "Per Cent Per Annum"},
            "unit": "percent per annum", "unit_kind": "rate", "frequency": "monthly",
            "publication_lag_days": 21, "plausible_range": [-1, 10], "transforms": ["mom_bp", "yoy_bp"],
            "headline": "yoy_bp", "decimals": 2}
    base.update(overrides)
    return make_spec(**base)


def singstat_payload(series: dict[str, list[tuple[str, Any]]], *, uom: str | None = "Index",
                     title: str = "Test table", frequency: str = "Quarterly") -> bytes:
    rows = [{"seriesNo": no, "rowText": f"Series {no}", "uoM": uom,
             "columns": [{"key": k, "value": None if v is None else str(v)} for k, v in cells]}
            for no, cells in series.items()]
    return json.dumps({"Data": {"title": title, "frequency": frequency, "row": rows}}).encode()


def singstat_metadata(series: dict[str, str], *, frequency: str = "Quarterly", adjustment: str = "",
                      title: str = "Test table", last_updated: str = "24/07/2026") -> bytes:
    rows = [{"seriesNo": no, "rowText": f"Series {no}", "uoM": unit} for no, unit in series.items()]
    return json.dumps({"Data": {"records": {"title": title, "frequency": frequency, "adjustmentType": adjustment,
                                            "dataSource": "TEST AGENCY", "dataLastUpdated": last_updated,
                                            "row": rows}}}).encode()


def quarters(start_year: int, values: list[float | None], label=lambda y, q: f"{y}-Q{q}") -> list[tuple[str, Any]]:
    out, y, q = [], start_year, 1
    for v in values:
        out.append((label(y, q), v))
        q += 1
        if q == 5:
            q, y = 1, y + 1
    return out


def store_vintage(root: Path, object_key: str, data: bytes, *, extension: str = "json",
                  metadata: bytes | None = None, request: dict | None = None, upstream: dict | None = None,
                  retrieved_at: datetime | None = None):
    store = SnapshotStore(root)
    vintage, _ = store.put(object_key, data=data, extension=extension, metadata=metadata,
                           request=request or {"object": object_key}, upstream=upstream or {},
                           retrieved_at=retrieved_at or datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc))
    return vintage


def state_for(vintage, *, after_as_of: bool = False, degraded: bool = False, error: str | None = None) -> ObjectState:
    from sgmacro.fetch import FetchObject

    obj = FetchObject(vintage.object_key, vintage.object_key.split("/")[0], vintage.object_key.split("/")[1])
    return ObjectState(obj, Selection(vintage, after_as_of), degraded=degraded, error=error)
