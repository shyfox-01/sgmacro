"""data.gov.sg adapter.

Download is a two-step flow, ``initiate-download`` then ``poll-download``, which
returns a presigned S3 URL. Three consequences:

1. Two throttled calls per dataset against a 5 requests/minute quota.
2. The presigned URL embeds temporary AWS credentials, so it is used once and
   never recorded. Provenance keeps the dataset id and the content hash.
3. The blob lives on S3, a different host, so it does not spend the quota.

Dataset metadata comes from a separate host. Its ``contactEmails`` field names
individual civil servants and is dropped before anything is written to disk.

The catalogue endpoint has no server-side search, so the discovery agent searches
a local snapshot of all ~4,600 dataset records instead (:func:`crawl_catalogue`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from sgmacro.sources.http import FetchError, HttpClient

DOWNLOAD_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"
METADATA_BASE = "https://api-production.data.gov.sg/v2/public/api/datasets"

#: Fields removed from dataset metadata before it is stored: personal data.
_DROPPED_METADATA_FIELDS = ("contactEmails",)


class DataGovError(RuntimeError):
    """A response did not have the shape the adapter requires."""


@dataclass
class DataGovPayload:
    data: bytes
    metadata: bytes
    request: dict[str, Any]
    upstream: dict[str, Any] = field(default_factory=dict)


class DataGovClient:
    def __init__(self, api: HttpClient, metadata_api: HttpClient, blob: HttpClient,
                 sleep: Callable[[float], None] | None = None) -> None:
        self.api = api                # throttled: initiate/poll
        self.metadata_api = metadata_api
        self.blob = blob              # presigned S3 downloads
        self._sleep = sleep or (lambda s: None)

    def metadata(self, dataset_id: str) -> bytes:
        response = self.metadata_api.get(f"{METADATA_BASE}/{dataset_id}/metadata", accept="application/json")
        data = _envelope(response.content, dataset_id)
        for key in _DROPPED_METADATA_FIELDS:
            data.pop(key, None)
        return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()

    def download(self, dataset_id: str, *, max_polls: int = 4) -> bytes:
        _envelope(self.api.get(f"{DOWNLOAD_BASE}/{dataset_id}/initiate-download",
                               accept="application/json").content, dataset_id)
        status = None
        for attempt in range(max_polls):
            poll = _envelope(self.api.get(f"{DOWNLOAD_BASE}/{dataset_id}/poll-download",
                                          accept="application/json").content, dataset_id)
            status, url = poll.get("status"), poll.get("url")
            if url:
                blob = self.blob.get(url)
                if not blob.content:
                    raise DataGovError(f"{dataset_id}: download returned zero bytes")
                return blob.content
            self._sleep(2.0 * (attempt + 1))
        raise DataGovError(f"{dataset_id}: poll-download never produced a URL (status={status!r})")

    def fetch(self, dataset_id: str) -> DataGovPayload:
        metadata = self.metadata(dataset_id)
        data = self.download(dataset_id)
        return DataGovPayload(
            data=data,
            metadata=metadata,
            request={"source": "datagovsg", "dataset_id": dataset_id},
            upstream=summarise_metadata(metadata),
        )


def summarise_metadata(metadata: bytes) -> dict[str, Any]:
    data = json.loads(metadata.decode("utf-8"))
    columns = data.get("columnMetadata") or {}
    mapping = columns.get("map") or {}
    order = columns.get("order") or []
    names = [mapping.get(c, c) for c in order] or list(mapping.values())
    return {
        "title": data.get("name"),
        "publisher": data.get("managedBy"),
        "last_updated": data.get("lastUpdatedAt"),
        "coverage_start": data.get("coverageStart"),
        "coverage_end": data.get("coverageEnd"),
        "columns_head": names[:8],
        "column_count": len(names),
    }


def crawl_catalogue(client: HttpClient) -> list[dict[str, Any]]:
    """Every dataset record in the catalogue, ~464 pages of ten."""
    records: list[dict[str, Any]] = []
    page = 1
    while True:
        response = client.get(METADATA_BASE, params={"page": page}, accept="application/json")
        data = _envelope(response.content, "catalogue")
        for item in data.get("datasets") or []:
            records.append({
                "dataset_id": item.get("datasetId"),
                "name": item.get("name"),
                "publisher": item.get("managedByAgencyName"),
                "last_updated": item.get("lastUpdatedAt"),
                "format": item.get("format"),
                "description": (item.get("description") or "")[:400],
            })
        if page >= int(data.get("pages") or 1):
            return records
        page += 1


def _envelope(content: bytes, what: str) -> dict[str, Any]:
    """Unwrap ``{code, data, errorMsg}``. A non-zero code arrives with HTTP 200."""
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataGovError(f"{what}: response was not JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise DataGovError(f"{what}: expected a JSON object")
    if payload.get("code") not in (0, None):
        raise FetchError(what, 200, f"code={payload.get('code')} {payload.get('errorMsg')}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise DataGovError(f"{what}: envelope had no data object")
    return data
