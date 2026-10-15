"""Retrieval: throttling, retries, the SingStat silent failures, degradation."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
import requests

from sgmacro.fetch import FetchObject, Fetcher
from sgmacro.sources.datagovsg import DataGovClient, DataGovError
from sgmacro.sources.http import FetchError, HttpClient, RateLimiter
from sgmacro.sources.singstat import SingStatClient, SingStatError, check_tabledata
from sgmacro.store import SnapshotStore
from tests.conftest import fixture_bytes, singstat_metadata, singstat_payload


class FakeResponse:
    def __init__(self, status=200, body=b"{}", headers=None):
        self.status_code = status
        self.content = body
        self.text = body.decode()
        self.headers = headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 300


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(responses, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return HttpClient(RateLimiter(6000, sleep=lambda s: None), session=FakeSession(responses),
                      sleep=sleeps.append, backoff_base=1.0)


def test_429_honours_retry_after_then_succeeds():
    sleeps = []
    c = client([FakeResponse(429, b"slow down", {"Retry-After": "7"}), FakeResponse(200, b"ok")], sleeps)
    assert c.get("https://x").content == b"ok"
    assert sleeps == [7.0]


def test_client_error_fails_immediately():
    c = client([FakeResponse(404, b"missing"), FakeResponse(200)])
    with pytest.raises(FetchError) as info:
        c.get("https://x")
    assert info.value.status == 404 and c.calls == 1


def test_network_errors_are_retried_then_raised():
    c = client([requests.ConnectionError("down")] * 4)
    with pytest.raises(FetchError, match="ConnectionError"):
        c.get("https://x")


def test_rate_limiter_spaces_requests():
    clock = iter([0.0, 1.0, 12.0])
    slept = []
    limiter = RateLimiter(5, clock=lambda: next(clock), sleep=slept.append)
    limiter.wait()
    limiter.wait()
    assert slept == [11.0]  # 12 s interval at 5/min, 1 s already elapsed


def test_singstat_truncation_is_detected():
    """Real response: CPI requested without a series filter came back 5000 points
    short of the Accommodation series (fixture trimmed from it)."""
    with pytest.raises(SingStatError, match="silent truncation"):
        check_tabledata("M213751", ["1", "1.03.1"], fixture_bytes("singstat_truncated.json"))


def test_singstat_empty_response_is_a_failure():
    with pytest.raises(SingStatError, match="zero rows"):
        check_tabledata("M213751", ["1"], fixture_bytes("singstat_empty.json"))


def test_singstat_datapoint_budget_is_a_failure():
    payload = singstat_payload({"1": [(f"{1600 + i // 12} Jan", 1.0) for i in range(5000)]})
    with pytest.raises(SingStatError, match="budget"):
        check_tabledata("M1", ["1"], payload)


def test_singstat_complete_response_passes():
    check_tabledata("M213751", ["1", "1.03.1"], fixture_bytes("singstat_rows.json"))


def test_datagov_application_error_with_http_200():
    api = client([FakeResponse(200, json.dumps({"code": 24, "errorMsg": "rate limited"}).encode())])
    dg = DataGovClient(api, api, api)
    with pytest.raises(FetchError, match="code=24"):
        dg.download("d_x")


def test_datagov_metadata_drops_personal_email():
    body = json.dumps({"code": 0, "data": {"name": "X", "contactEmails": ["someone@gov.sg"]}}).encode()
    meta_client = client([FakeResponse(200, body)])
    dg = DataGovClient(meta_client, meta_client, meta_client)
    assert b"someone@gov.sg" not in dg.metadata("d_x")


def test_datagov_poll_without_url_is_an_error():
    ok = json.dumps({"code": 0, "data": {}}).encode()
    pending = json.dumps({"code": 0, "data": {"status": "PENDING"}}).encode()
    api = client([FakeResponse(200, ok)] + [FakeResponse(200, pending)] * 4)
    with pytest.raises(DataGovError, match="never produced a URL"):
        DataGovClient(api, api, api).download("d_x", max_polls=4)


class ExplodingSingStat:
    def fetch(self, *a, **k):
        raise FetchError("https://tablebuilder", 503, "maintenance")


class WorkingSingStat:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    def fetch(self, table, series):
        from sgmacro.sources.singstat import SingStatPayload
        self.calls += 1
        return SingStatPayload(self.data, singstat_metadata({"1": "Index"}),
                               {"source": "singstat", "table_id": table, "series": series}, {"last_updated": "x"})


def test_failed_refetch_degrades_to_the_stored_vintage(tmp_path):
    store = SnapshotStore(tmp_path)
    store.put("singstat/M1", data=b"old", extension="json", metadata=None,
              request={"source": "singstat", "table_id": "M1", "series": ["1"]}, upstream={},
              retrieved_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
    fetcher = Fetcher(store, singstat=ExplodingSingStat(), datagov=object())
    obj = FetchObject("singstat/M1", "singstat", "M1", ("1",))
    state = fetcher.resolve([obj], date(2026, 10, 2), refetch=True)["singstat/M1"]
    assert state.degraded and "maintenance" in state.error
    assert state.vintage.data_bytes() == b"old"


def test_failed_fetch_without_any_vintage_reports_no_data(tmp_path):
    fetcher = Fetcher(SnapshotStore(tmp_path), singstat=ExplodingSingStat(), datagov=object())
    obj = FetchObject("singstat/M1", "singstat", "M1", ("1",))
    state = fetcher.resolve([obj], date(2026, 10, 2))["singstat/M1"]
    assert state.vintage is None and not state.degraded and state.error


def test_repeated_request_does_not_touch_the_network(tmp_path):
    working = WorkingSingStat(singstat_payload({"1": [("2026 1Q", 1)]}))
    fetcher = Fetcher(SnapshotStore(tmp_path), singstat=working, datagov=object())
    obj = FetchObject("singstat/M1", "singstat", "M1", ("1",))
    first = fetcher.resolve([obj], date(2026, 10, 2))["singstat/M1"]
    second = fetcher.resolve([obj], date(2026, 10, 2))["singstat/M1"]
    assert working.calls == 1 and first.downloaded and not second.downloaded
    third = fetcher.resolve([obj], date(2026, 10, 2), refetch=True)["singstat/M1"]
    assert working.calls == 2 and third.downloaded and not third.created  # same bytes: no new vintage


def test_vintage_missing_a_newly_registered_series_triggers_a_fetch(tmp_path):
    store = SnapshotStore(tmp_path)
    store.put("singstat/M1", data=b"x", extension="json", metadata=None,
              request={"source": "singstat", "table_id": "M1", "series": ["1"]}, upstream={},
              retrieved_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
    working = WorkingSingStat(singstat_payload({"1": [("2026 1Q", 1)], "2": [("2026 1Q", 2)]}))
    fetcher = Fetcher(store, singstat=working, datagov=object())
    obj = FetchObject("singstat/M1", "singstat", "M1", ("1", "2"))
    fetcher.resolve([obj], date(2026, 10, 2))
    assert working.calls == 1


def test_failed_presigned_download_never_carries_its_signature():
    url = "https://bucket.s3.amazonaws.com/blob.csv?X-Amz-Credential=AKIA123&X-Amz-Signature=deadbeef"

    class Refused:
        headers = {}

        def __init__(self):
            self.status_code, self.ok, self.text = 403, False, f"<Error>denied for {url}</Error>"

    class Session:
        headers = {}

        def get(self, *args, **kwargs):
            return Refused()

    client = HttpClient(RateLimiter(6000, sleep=lambda s: None), session=Session(), max_attempts=1)
    with pytest.raises(FetchError) as caught:
        client.get(url)
    text = f"{caught.value} {caught.value.url} {caught.value.detail}"
    assert "X-Amz" not in text and "deadbeef" not in text and "blob.csv" in text


@pytest.mark.parametrize("payload", [
    {"Data": [{"records": {}}]},                       # Data is a list
    {"Data": {"records": {"row": "x"}}},               # series list is a string
])
def test_a_malformed_metadata_response_is_an_upstream_error(payload):
    class Answer:
        content = json.dumps(payload).encode()

    class Http:
        def get(self, *args, **kwargs):
            return Answer()

    with pytest.raises(SingStatError, match="metadata"):
        SingStatClient(Http()).metadata("M1")


def test_malformed_rows_are_an_upstream_error():
    with pytest.raises(SingStatError, match="not a list of objects"):
        check_tabledata("M1", ["1"], json.dumps({"Data": {"row": ["x"]}}).encode())
    with pytest.raises(SingStatError, match="not a list of objects"):
        check_tabledata("M1", ["1"], json.dumps({"Data": {"row": [{"seriesNo": "1", "columns": "x"}]}}).encode())
    for cell in ("x", None):                         # a cell that is not an object
        bad = {"Data": {"row": [{"seriesNo": "1", "columns": [cell]}]}}
        with pytest.raises(SingStatError, match="columns are objects"):
            check_tabledata("M1", ["1"], json.dumps(bad).encode())


def test_a_relative_url_in_a_connection_error_loses_its_query_too():
    """urllib3 quotes only the path: "Max retries exceeded with url: /x.csv?X-Amz-...". """
    from sgmacro.sources.http import without_query

    text = ("HTTPSConnectionPool(host='bucket.s3.amazonaws.com', port=443): Max retries exceeded with url: "
            "/blob.csv?X-Amz-Credential=DUMMY_REVIEW&X-Amz-Signature=DUMMY_REVIEW (Caused by NewConnectionError)")
    cleaned = without_query(text)
    assert "X-Amz" not in cleaned and "DUMMY_REVIEW" not in cleaned and "/blob.csv?[query removed]" in cleaned
