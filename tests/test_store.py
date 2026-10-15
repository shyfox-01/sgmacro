"""Snapshot store: immutability, deduplication, point-in-time selection."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from sgmacro.store import SnapshotStore, assert_no_credentials


def put(store, data, when, key="singstat/M1", request=None):
    return store.put(key, data=data, extension="json", metadata=b"{}", request=request or {"series": ["1"]},
                     upstream={"last_updated": "x"}, retrieved_at=when)


def test_identical_bytes_are_deduplicated(tmp_path):
    store = SnapshotStore(tmp_path)
    first, created = put(store, b"abc", datetime(2026, 9, 1, tzinfo=timezone.utc))
    again, created_again = put(store, b"abc", datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert created and not created_again
    assert again.vintage_id == first.vintage_id
    assert len(store.vintages("singstat/M1")) == 1


def test_changed_metadata_over_the_same_data_is_kept(tmp_path):
    """Same numbers, new publisher metadata (here a changed unit): the store must not
    hand back the old vintage, or the contract check would read the old unit."""
    store = SnapshotStore(tmp_path)
    put(store, b"abc", datetime(2026, 9, 1, tzinfo=timezone.utc))
    newer, created = store.put("singstat/M1", data=b"abc", extension="json", metadata=b"{}",
                               request={"series": ["1"]}, upstream={"last_updated": "x", "unit": "Thousand"},
                               retrieved_at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert created and newer.upstream["unit"] == "Thousand"
    assert store.select("singstat/M1", date(2026, 10, 2)).vintage.vintage_id == newer.vintage_id


def test_changed_bytes_create_a_new_vintage(tmp_path):
    store = SnapshotStore(tmp_path)
    put(store, b"v1", datetime(2026, 9, 1, tzinfo=timezone.utc))
    second, created = put(store, b"v2", datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert created
    assert store.previous(second).data_bytes() == b"v1"


def test_point_in_time_selection(tmp_path):
    store = SnapshotStore(tmp_path)
    put(store, b"sept", datetime(2026, 9, 1, 2, tzinfo=timezone.utc))
    put(store, b"oct", datetime(2026, 10, 1, 2, tzinfo=timezone.utc))
    assert store.select("singstat/M1", date(2026, 9, 15)).vintage.data_bytes() == b"sept"
    assert store.select("singstat/M1", date(2026, 10, 2)).vintage.data_bytes() == b"oct"
    early = store.select("singstat/M1", date(2026, 1, 1))
    assert early.after_as_of and early.vintage.data_bytes() == b"sept"


def test_selection_uses_singapore_dates(tmp_path):
    """Retrieved 17:00 UTC on 30 Sep is 1 Oct in Singapore."""
    store = SnapshotStore(tmp_path)
    put(store, b"late", datetime(2026, 9, 30, 17, tzinfo=timezone.utc))
    assert store.select("singstat/M1", date(2026, 9, 30)).after_as_of
    assert not store.select("singstat/M1", date(2026, 10, 1)).after_as_of


def test_accept_filter_skips_vintages_missing_series(tmp_path):
    store = SnapshotStore(tmp_path)
    put(store, b"one", datetime(2026, 9, 1, tzinfo=timezone.utc), request={"series": ["1"]})
    assert store.select("singstat/M1", date(2026, 10, 1), lambda v: "2" in v.request["series"]) is None


def test_credentials_are_never_written(tmp_path):
    store = SnapshotStore(tmp_path)
    with pytest.raises(ValueError, match="X-Amz-Signature"):
        store.put("datagovsg/d", data=b"x", extension="csv", metadata=None,
                  request={"url": "https://s3/blob?X-Amz-Signature=abc"}, upstream={})
    with pytest.raises(ValueError):
        assert_no_credentials('{"api_key": "secret"}')
