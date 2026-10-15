"""Retrieval: registry in, selected snapshot vintages out.

Series are grouped by the upstream object that serves them, so the two CPI
series cost one SingStat call and SORA plus the 10-year yield cost another.

The network is touched only when it has to be: on ``--refetch``, or when the
store holds no usable vintage of an object (a fresh registry entry). Otherwise a
run reads the committed store, which is what makes the report reproducible
offline and makes a repeated request a no-op upstream.

A failed refetch degrades instead of aborting: the newest stored vintage is used
and the run is labelled, because an analyst reading a clearly marked older figure
is better served than one reading a report from which a series silently vanished.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from sgmacro import curated
from sgmacro.registry import CuratedRef, DataGovRef, Registry, SingStatRef
from sgmacro.sources.datagovsg import DataGovClient, DataGovError
from sgmacro.sources.http import FetchError, HttpClient, RateLimiter
from sgmacro.sources.singstat import SingStatClient, SingStatError
from sgmacro.store import Selection, SnapshotStore, Vintage


@dataclass(frozen=True)
class FetchObject:
    object_key: str
    source: str
    upstream_id: str
    series: tuple[str, ...] = ()

    @property
    def extension(self) -> str:
        return {"singstat": "json", "curated": "yaml"}.get(self.source, "csv")

    def accepts(self, vintage: Vintage) -> bool:
        """A SingStat vintage is usable only if it contains every needed series."""
        if self.source != "singstat":
            return True
        return set(self.series) <= set(vintage.request.get("series") or [])


@dataclass
class ObjectState:
    obj: FetchObject
    selection: Selection | None
    downloaded: bool = False
    created: bool = False
    degraded: bool = False
    error: str | None = None

    @property
    def vintage(self) -> Vintage | None:
        return self.selection.vintage if self.selection else None


def plan(registry: Registry) -> list[FetchObject]:
    singstat: dict[str, set[str]] = {}
    datagov: set[str] = set()
    curated_paths: dict[str, str] = {}
    for spec in registry.series:
        for ref in spec.refs():
            if isinstance(ref, SingStatRef):
                singstat.setdefault(ref.table_id, set()).update(ref.series_numbers)
            elif isinstance(ref, DataGovRef):
                datagov.add(ref.dataset_id)
            elif isinstance(ref, CuratedRef):
                curated_paths[ref.object_key] = ref.path
    objects = [
        FetchObject(f"singstat/{t}", "singstat", t, tuple(sorted(s, key=_series_sort_key)))
        for t, s in singstat.items()
    ]
    objects += [FetchObject(f"datagovsg/{d}", "datagovsg", d) for d in datagov]
    objects += [FetchObject(key, "curated", path) for key, path in curated_paths.items()]
    return sorted(objects, key=lambda o: o.object_key)


def _series_sort_key(series_no: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in series_no.split("."))


def build_clients(sleep: Callable[[float], None] | None = None) -> tuple[SingStatClient, DataGovClient]:
    kwargs = {"sleep": sleep} if sleep else {}
    limiter_kwargs = {"sleep": sleep} if sleep else {}
    singstat = SingStatClient(HttpClient(RateLimiter(60, **limiter_kwargs), **kwargs))
    datagov = DataGovClient(
        api=HttpClient(RateLimiter(5, **limiter_kwargs), **kwargs),        # documented quota
        metadata_api=HttpClient(RateLimiter(60, **limiter_kwargs), **kwargs),
        blob=HttpClient(RateLimiter(600, **limiter_kwargs), **kwargs),     # S3, no shared quota
        sleep=sleep,
    )
    return singstat, datagov


class Fetcher:
    def __init__(
        self,
        store: SnapshotStore,
        singstat: SingStatClient | None = None,
        datagov: DataGovClient | None = None,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.store = store
        self._singstat = singstat
        self._datagov = datagov
        self.log = log

    def _clients(self) -> tuple[SingStatClient, DataGovClient]:
        if self._singstat is None or self._datagov is None:
            self._singstat, self._datagov = build_clients()
        return self._singstat, self._datagov

    def resolve(self, objects: list[FetchObject], as_of: date, *, refetch: bool = False,
                offline: bool = False) -> dict[str, ObjectState]:
        states: dict[str, ObjectState] = {}
        for obj in objects:
            states[obj.object_key] = self._resolve_one(obj, as_of, refetch=refetch, offline=offline)
        return states

    def _resolve_one(self, obj: FetchObject, as_of: date, *, refetch: bool, offline: bool) -> ObjectState:
        if obj.source == "curated":
            return self._resolve_curated(obj, as_of)
        existing = self.store.select(obj.object_key, as_of, obj.accepts)
        if existing is not None and not refetch:
            return ObjectState(obj, existing)
        if offline:
            return ObjectState(obj, existing, error=None if existing else "offline and no stored vintage")

        try:
            self.log(f"fetching {obj.object_key}")
            created = self._download(obj)
        except (FetchError, DataGovError, SingStatError, OSError) as exc:
            # An upstream failure degrades to the stored vintage. Anything else is a
            # bug in this code and should stop the run rather than pass as an outage.
            error = f"{type(exc).__name__}: {exc}"
            self.log(f"  failed: {error}")
            fallback = existing or self._any_usable(obj)
            return ObjectState(obj, fallback, downloaded=False, degraded=fallback is not None, error=error)
        selection = self.store.select(obj.object_key, as_of, obj.accepts)
        return ObjectState(obj, selection, downloaded=True, created=created)

    def _resolve_curated(self, obj: FetchObject, as_of: date) -> ObjectState:
        """The local event list is snapshotted on every run (deduplicated by
        hash), so an edit becomes a new vintage with its own date."""
        try:
            content = Path(obj.upstream_id).read_bytes()
            _, created = self.store.put(obj.object_key, data=content, extension=obj.extension, metadata=None,
                                        request={"source": "curated", "path": obj.upstream_id},
                                        upstream=curated.summarise(content))
        except (OSError, curated.CuratedError) as exc:
            fallback = self.store.select(obj.object_key, as_of) or self._any_usable(obj)
            return ObjectState(obj, fallback, degraded=fallback is not None, error=f"{type(exc).__name__}: {exc}")
        return ObjectState(obj, self.store.select(obj.object_key, as_of), created=created)

    def _any_usable(self, obj: FetchObject) -> Selection | None:
        usable = [v for v in self.store.vintages(obj.object_key) if obj.accepts(v)]
        return Selection(usable[-1], after_as_of=True) if usable else None

    def _download(self, obj: FetchObject) -> bool:
        singstat, datagov = self._clients()
        if obj.source == "singstat":
            payload = singstat.fetch(obj.upstream_id, list(obj.series))
        else:
            payload = datagov.fetch(obj.upstream_id)
        _, created = self.store.put(
            obj.object_key,
            data=payload.data,
            extension=obj.extension,
            metadata=payload.metadata,
            request=payload.request,
            upstream=payload.upstream,
        )
        self.log(f"  {'new vintage' if created else 'unchanged since last vintage'}")
        return created
