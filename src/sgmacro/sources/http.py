"""Rate-limited HTTP client with retry and backoff.

data.gov.sg's download API allows 5 requests per minute and answers the sixth
with HTTP 429, so throttling is a correctness requirement rather than
politeness: an unthrottled fetch silently loses half the registry.

Retry policy:
  429              honour ``Retry-After`` when present, else exponential backoff
  500/502/503/504  exponential backoff (transient upstream trouble)
  other 4xx        fail immediately; a bad request will not fix itself
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

USER_AGENT = "sgmacro/2.1.0 (Singapore housing data workflow)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


_QUERY = re.compile(r"\?[^\s'\"<>()]+")


def without_query(text: str) -> str:
    """Drop every query string from ``text``, absolute URL or not.

    A presigned download URL carries temporary credentials in its query, and
    error text is persisted (validation table, reports, manifest). urllib3 quotes
    only the path and query ("Max retries exceeded with url: /x.csv?X-Amz-...")
    so a pattern anchored on "https://" is not enough. Cut where the error is
    raised rather than filtered at every place that might write it.
    """
    return _QUERY.sub("?[query removed]", text)


class FetchError(RuntimeError):
    """A request that could not be completed after exhausting retries."""

    def __init__(self, url: str, status: int | None, detail: str) -> None:
        self.url = without_query(url)
        self.status = status
        self.detail = without_query(detail)
        super().__init__(f"{self.url} -> status={status}: {self.detail}")


@dataclass
class RateLimiter:
    """Minimum-interval spacing, one instance per upstream host."""

    requests_per_min: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last: float | None = field(default=None, init=False)

    @property
    def min_interval(self) -> float:
        return 60.0 / self.requests_per_min

    def wait(self) -> float:
        slept = 0.0
        if self._last is not None:
            elapsed = self.clock() - self._last
            if elapsed < self.min_interval:
                slept = self.min_interval - elapsed
                self.sleep(slept)
        self._last = self.clock()
        return slept


class HttpClient:
    """Thin wrapper over ``requests`` adding throttling, retry and a call log."""

    def __init__(
        self,
        limiter: RateLimiter,
        *,
        max_attempts: int = 4,
        backoff_base: float = 5.0,
        timeout: float = 60.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limiter = limiter
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.timeout = timeout
        self._sleep = sleep
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.calls = 0

    def get(self, url: str, *, params: dict | None = None, accept: str | None = None) -> requests.Response:
        headers = {"Accept": accept} if accept else None
        status: int | None = None
        detail = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            self.limiter.wait()
            self.calls += 1
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                status, detail = None, f"{type(exc).__name__}: {exc}"
                if attempt < self.max_attempts:
                    self._sleep(self.backoff_base * attempt)
                continue
            status = response.status_code
            if response.ok:
                return response
            detail = response.text[:200]
            if status in RETRY_STATUS and attempt < self.max_attempts:
                self._sleep(self._retry_delay(response, attempt))
                continue
            break
        raise FetchError(url, status, detail)

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(1.0, float(header))
            except ValueError:
                pass
        return self.backoff_base * attempt
