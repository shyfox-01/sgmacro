"""Content-addressed snapshot store with point-in-time vintage selection.

Layout::

    data/snapshots/<source>/<object>/<YYYYMMDDTHHMMSSZ>-<sha8>/
        data.csv | data.json     exact bytes the upstream served
        metadata.json            upstream metadata (personal fields removed)
        vintage.json             request, retrieval time, hashes, extracted fields

Three properties matter:

*Immutability.* A vintage directory is written once. Re-fetching an object whose
bytes have not changed records nothing new (deduplication by content hash), so
running the same request twice leaves the store untouched.

*Point-in-time selection.* A run at ``--as-of D`` reads the newest vintage
retrieved on or before ``D``, i.e. the data as this workflow actually saw it on
that day, including whatever revisions had or had not happened yet. When no such
vintage exists (an as-of date before the first fetch), the oldest vintage is used
and flagged, because later revisions may then leak into a historical run.

*Lineage.* Every observation carries the vintage id and the content hash of the
bytes it came from, so a reported figure can be re-derived from the store alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

SGT = timezone(timedelta(hours=8))
VINTAGE_FILE = "vintage.json"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class Vintage:
    object_key: str
    vintage_id: str
    path: Path
    retrieved_at: datetime
    data_file: str
    data_sha256: str
    request: dict[str, Any]
    upstream: dict[str, Any]

    @property
    def retrieved_date_sgt(self) -> date:
        return self.retrieved_at.astimezone(SGT).date()

    @property
    def data_path(self) -> Path:
        return self.path / self.data_file

    def data_bytes(self) -> bytes:
        return self.data_path.read_bytes()

    def metadata(self) -> dict[str, Any] | None:
        path = self.path / "metadata.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def summary(self) -> dict[str, Any]:
        return {
            "object": self.object_key,
            "vintage_id": self.vintage_id,
            "retrieved_at": self.retrieved_at.isoformat(),
            "data_file": self.data_file,
            "data_sha256": self.data_sha256,
            "upstream_last_updated": self.upstream.get("last_updated"),
        }


@dataclass(frozen=True)
class Selection:
    vintage: Vintage
    after_as_of: bool
    """The vintage was retrieved after the as-of date (no earlier one exists)."""


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- reading -----------------------------------------------------------
    def vintages(self, object_key: str) -> list[Vintage]:
        directory = self.root / object_key
        if not directory.exists():
            return []
        found = []
        for child in directory.iterdir():
            sidecar = child / VINTAGE_FILE
            if child.is_dir() and sidecar.exists():
                found.append(self._load(object_key, child, sidecar))
        return sorted(found, key=lambda v: (v.retrieved_at, v.vintage_id))

    def latest(self, object_key: str) -> Vintage | None:
        found = self.vintages(object_key)
        return found[-1] if found else None

    def select(
        self,
        object_key: str,
        as_of: date,
        accept: Callable[[Vintage], bool] = lambda v: True,
    ) -> Selection | None:
        """Newest acceptable vintage retrieved on or before ``as_of`` (SGT)."""
        usable = [v for v in self.vintages(object_key) if accept(v)]
        if not usable:
            return None
        on_or_before = [v for v in usable if v.retrieved_date_sgt <= as_of]
        if on_or_before:
            return Selection(on_or_before[-1], after_as_of=False)
        return Selection(usable[0], after_as_of=True)

    def previous(self, vintage: Vintage) -> Vintage | None:
        """The vintage of the same object immediately before this one."""
        earlier = [v for v in self.vintages(vintage.object_key) if v.retrieved_at < vintage.retrieved_at]
        return earlier[-1] if earlier else None

    # -- writing -----------------------------------------------------------
    def put(
        self,
        object_key: str,
        *,
        data: bytes,
        extension: str,
        metadata: bytes | None,
        request: dict[str, Any],
        upstream: dict[str, Any],
        retrieved_at: datetime | None = None,
    ) -> tuple[Vintage, bool]:
        """Store a download. Returns ``(vintage, created)``.

        A download identical to the latest vintage (same request, same data bytes,
        same publisher metadata) is not stored again. If only the metadata changed
        (a new unit, title or update stamp over the same numbers), a new vintage is
        written: the contract and lag checks read the metadata, so it must not be
        silently kept at its old value.
        """
        digest = sha256(data)
        latest = self.latest(object_key)
        if (latest is not None and latest.data_sha256 == digest and latest.request == request
                and latest.upstream == json.loads(json.dumps(upstream, ensure_ascii=False))):
            return latest, False

        retrieved_at = (retrieved_at or datetime.now(timezone.utc)).replace(microsecond=0)
        vintage_id = f"{retrieved_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{digest[:8]}"
        directory = self.root / object_key / vintage_id
        directory.mkdir(parents=True, exist_ok=False)
        data_file = f"data.{extension}"
        (directory / data_file).write_bytes(data)
        files = {data_file: digest}
        if metadata is not None:
            (directory / "metadata.json").write_bytes(metadata)
            files["metadata.json"] = sha256(metadata)
        sidecar = {
            "object": object_key,
            "vintage_id": vintage_id,
            "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat(),
            "request": request,
            "files": files,
            "data_file": data_file,
            "upstream": upstream,
        }
        assert_no_credentials(json.dumps(sidecar))
        (directory / VINTAGE_FILE).write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8", newline="\n")
        return self._load(object_key, directory, directory / VINTAGE_FILE), True

    # -- internals ---------------------------------------------------------
    def _load(self, object_key: str, directory: Path, sidecar: Path) -> Vintage:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        return Vintage(
            object_key=object_key,
            vintage_id=raw["vintage_id"],
            path=directory,
            retrieved_at=datetime.fromisoformat(raw["retrieved_at"]),
            data_file=raw["data_file"],
            data_sha256=raw["files"][raw["data_file"]],
            request=raw.get("request") or {},
            upstream=raw.get("upstream") or {},
        )


#: Query parameters that would carry credentials if a presigned URL leaked.
_CREDENTIAL_MARKERS = ("X-Amz-Credential", "X-Amz-Signature", "X-Amz-Security-Token",
                       "AWSAccessKeyId", "api_key", "apikey", "access_token")


def assert_no_credentials(blob: str) -> None:
    """Refuse to persist anything that looks like a signed URL or a key."""
    from sgmacro.credentials import KEY_PATTERN, redact_keys

    lowered = blob.lower()
    for marker in _CREDENTIAL_MARKERS:
        if marker.lower() in lowered:
            raise ValueError(f"refusing to write a record containing {marker!r}")
    if KEY_PATTERN.search(blob) or redact_keys(blob) != blob:
        raise ValueError("refusing to write a record containing an API key")
