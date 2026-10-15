"""SingStat Table Builder adapter.

Three endpoints are used: ``tabledata`` (observations), ``metadata`` (title,
frequency, seasonal adjustment, publisher, and the unit of measure of every
series) and ``resourceid`` (keyword search, used by the discovery agent).

The tabledata endpoint has two silent-failure modes, both returning HTTP 200:

1. **Datapoint truncation.** The response is capped at 5000 *datapoints*, not
   rows. Requesting monthly CPI without a series filter returns 22 of its 207
   series, exactly 5000 points, with the tail of the table dropped and no flag.
2. **Ignored filters.** A malformed parameter is not rejected; it is dropped and
   the response comes back with zero rows.

Both are handled the same way: request exactly the series the registry needs,
then assert that exactly those came back, each with observations, under the
datapoint budget. A short or empty response is a failure, never a quietly
shorter series list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sgmacro.sources.http import HttpClient

BASE = "https://tablebuilder.singstat.gov.sg/api/table"
DATAPOINT_LIMIT = 5000


class SingStatError(RuntimeError):
    """The response was empty, truncated or missing a requested series."""


@dataclass
class SingStatPayload:
    data: bytes
    metadata: bytes
    request: dict[str, Any]
    upstream: dict[str, Any] = field(default_factory=dict)


class SingStatClient:
    def __init__(self, client: HttpClient) -> None:
        self.client = client

    def tabledata(self, table_id: str, series: list[str]) -> bytes:
        if not series:
            raise ValueError(f"{table_id}: no series requested")
        response = self.client.get(
            f"{BASE}/tabledata/{table_id}",
            params={"seriesNoORrowNo": ",".join(series)},
            accept="application/json",
        )
        check_tabledata(table_id, series, response.content)
        return response.content

    def metadata(self, table_id: str) -> bytes:
        response = self.client.get(f"{BASE}/metadata/{table_id}", accept="application/json")
        data = _json(response.content, table_id).get("Data")
        records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(records, dict):
            raise SingStatError(f"{table_id}: metadata response had no records object")
        rows = records.get("row") or []
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise SingStatError(f"{table_id}: metadata series list is not a list of objects")
        return response.content

    def fetch(self, table_id: str, series: list[str]) -> SingStatPayload:
        data = self.tabledata(table_id, series)
        metadata = self.metadata(table_id)
        return SingStatPayload(
            data=data,
            metadata=metadata,
            request={"source": "singstat", "table_id": table_id, "series": sorted(series)},
            upstream=summarise_metadata(metadata, series),
        )

    def search(self, keyword: str) -> list[dict[str, str]]:
        """Title keyword search. Noisy by nature: candidates, never answers."""
        response = self.client.get(
            f"{BASE}/resourceid",
            params={"keyword": keyword, "searchOption": "all"},
            accept="application/json",
        )
        records = ((_json(response.content, "search").get("Data") or {}).get("records")) or []
        return [
            {"table_id": r.get("id", ""), "title": r.get("title", ""), "topic": r.get("topic", "")}
            for r in records
            if r.get("tableType") == "time series"
        ]


def check_tabledata(table_id: str, requested: list[str], content: bytes) -> None:
    """Raise unless the response holds every requested series, untruncated."""
    data = _json(content, table_id).get("Data")
    if not isinstance(data, dict):
        raise SingStatError(f"{table_id}: response had no Data object")
    rows = data.get("row") or []
    # The response shape is checked here, at the boundary, so a malformed answer is
    # an upstream error (and the fetcher falls back to the stored vintage), not a crash.
    if not isinstance(rows, list) or not all(
            isinstance(r, dict) and isinstance(r.get("columns") or [], list)
            and all(isinstance(c, dict) for c in r.get("columns") or []) for r in rows):
        raise SingStatError(f"{table_id}: rows are not a list of objects whose columns are objects")
    if not rows:
        raise SingStatError(
            f"{table_id}: zero rows for series {requested}. This API answers an "
            "unsupported parameter with HTTP 200 and no rows, so empty is a failure."
        )
    returned = {str(r.get("seriesNo")) for r in rows}
    missing = [s for s in requested if s not in returned]
    if missing:
        raise SingStatError(
            f"{table_id}: requested {requested} but {missing} are absent (got {sorted(returned)}); "
            "this is the silent truncation mode of the API"
        )
    empty = [str(r.get("seriesNo")) for r in rows if not r.get("columns")]
    if empty:
        raise SingStatError(f"{table_id}: series {empty} returned with no observations")
    points = sum(len(r.get("columns") or []) for r in rows)
    if points >= DATAPOINT_LIMIT:
        raise SingStatError(
            f"{table_id}: {points} datapoints reached the {DATAPOINT_LIMIT} budget, so the "
            "tail may be truncated; request fewer series"
        )


def summarise_metadata(metadata: bytes, series: list[str]) -> dict[str, Any]:
    """The fields of SingStat metadata the validator cross-checks."""
    records = (_json(metadata, "metadata").get("Data") or {}).get("records") or {}
    rows = {str(r.get("seriesNo")): r for r in records.get("row") or []}
    return {
        "title": records.get("title"),
        "publisher": records.get("dataSource"),
        "frequency": records.get("frequency"),
        "adjustment": records.get("adjustmentType"),
        "last_updated": records.get("dataLastUpdated"),
        "start_period": records.get("startPeriod"),
        "end_period": records.get("endPeriod"),
        "series": {
            s: {
                "label": (rows.get(s) or {}).get("rowText"),
                "unit_of_measure": (rows.get(s) or {}).get("uoM"),
                "footnote": (rows.get(s) or {}).get("footnote") or None,
            }
            for s in sorted(series)
        },
    }


def _json(content: bytes, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SingStatError(f"{what}: response was not JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise SingStatError(f"{what}: expected a JSON object")
    return payload
