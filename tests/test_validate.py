"""Validation: the checks that decide what is fit to report."""

from __future__ import annotations

from datetime import date

import pytest

from sgmacro.parse import extract
from sgmacro.registry import Frequency
from sgmacro.validate import Severity, SourceView, Status, validate_series
from tests.conftest import (fixture_bytes, make_spec, quarters, rate_spec, singstat_metadata, singstat_payload,
                            state_for, store_vintage)

AS_OF = date(2026, 10, 2)


def view_of(tmp_path, spec, content, *, upstream=None, extension="csv", key=None, **state_kw):
    vintage = store_vintage(tmp_path, key or spec.source.object_key, content, extension=extension,
                            upstream=upstream or {})
    extracted = extract(spec.source, content, spec.frequency, vintage.upstream)
    return SourceView(spec.source, state_for(vintage, **state_kw), extracted)


def checks(result, severity=None):
    return [f.check for f in result.findings if severity is None or f.severity is severity]


def test_healthy_series_passes(tmp_path):
    spec = make_spec()
    result = validate_series(spec, view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv")), as_of=AS_OF)
    assert result.status is Status.OK and result.latest[0].label == "2026-Q2"


def test_publication_lag_gate_withholds_unpublished_quarters(tmp_path):
    spec = make_spec()
    result = validate_series(spec, view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv")),
                             as_of=date(2026, 7, 15))
    assert result.latest[0].label == "2026-Q1"
    assert result.withheld == ["2026-Q2"]
    assert result.gated["period"].iloc[-1] == "2026-Q1"


def test_gate_counts_from_mid_year_reference_date(tmp_path):
    """Modelling annual periods as ending 31 December withheld the end-June 2026
    population estimate published on 25 September; the reference month fixes that."""
    content = singstat_payload({"5": [("2024", 100), ("2025", 103), ("2026", 106)]}, uom="Number")
    fields = dict(id="pop", source={"source": "singstat", "table_id": "M810001", "series": "5",
                                    "unit_of_measure": "Number"},
                  unit_kind="level", frequency="annual", transforms=["yoy"], headline="yoy",
                  publication_lag_days=90, plausible_range=[1, 1000])
    with_ref = make_spec(**fields, reference_month=6)
    without = make_spec(**fields)
    as_december = validate_series(without, view_of(tmp_path, without, content, extension="json"), as_of=AS_OF)
    as_june = validate_series(with_ref, view_of(tmp_path / "b", with_ref, content, extension="json"), as_of=AS_OF)
    assert as_december.latest[0].label == "2025" and as_december.withheld == ["2026"]
    assert as_june.latest[0].label == "2026" and as_june.withheld == []


def test_series_discontinued_inside_a_live_dataset_is_rejected(tmp_path):
    """The URA rental file is refreshed quarterly; its all-residential slice stopped in 2021."""
    spec = make_spec(id="slice")
    content = ("quarter,property_type,index\n" + "".join(
        f"{p},All Residential,{100 + i}\n" for i, (p, _) in enumerate(quarters(2016, [0] * 22)))).encode()
    upstream = {"coverage_end": "2026-06-30T08:00:00+08:00", "last_updated": "2026-07-27T09:56:07+08:00"}
    result = validate_series(spec, view_of(tmp_path, spec, content, upstream=upstream), as_of=AS_OF)
    assert result.verdict == "rejected:discontinued"
    assert "coverage" in checks(result, Severity.WARN)


def test_stale_data_behind_a_fresh_update_stamp(tmp_path):
    spec = make_spec(source={"source": "datagovsg", "dataset_id": "d_loans", "layout": "tidy",
                             "value_column": "utilised"}, unit_kind="level", plausible_range=[1, 10**7])
    upstream = {"last_updated": "2024-06-06T14:29:16+08:00"}
    result = validate_series(spec, view_of(tmp_path, spec, fixture_bytes("tidy_quarter_stale.csv"),
                                           upstream=upstream), as_of=AS_OF)
    assert result.verdict == "rejected:discontinued"


def test_one_or_two_releases_late_is_a_warning_not_a_rejection(tmp_path):
    spec = make_spec()
    content = b"quarter,property_type,index\n" + b"".join(
        f"{p},All Residential,{100 + i}\n".encode() for i, (p, _) in enumerate(quarters(2020, [0] * 24)))
    # latest 2025-Q4; by 2026-10-02 both 2026-Q1 and 2026-Q2 were due
    result = validate_series(spec, view_of(tmp_path, spec, content), as_of=AS_OF)
    assert result.status is Status.OK_WITH_WARNINGS and "staleness" in checks(result, Severity.WARN)


def test_unit_contract_mismatch_rejects(tmp_path):
    """The first registry declared non-resident population in thousands; SingStat says 'Number'."""
    spec = make_spec(id="pop", source={"source": "singstat", "table_id": "M810001", "series": "5",
                                       "unit_of_measure": "Thousand"},
                     unit_kind="level", frequency="annual", reference_month=6, transforms=["yoy"],
                     headline="yoy", plausible_range=[1, 10**7])
    content = singstat_payload({"5": [("2025", 1906660), ("2026", 1976999)]}, uom="Number")
    upstream = {"frequency": "Annual", "series": {"5": {"unit_of_measure": "Number"}}}
    result = validate_series(spec, view_of(tmp_path, spec, content, upstream=upstream, extension="json"),
                             as_of=AS_OF)
    assert result.verdict == "rejected:contract_mismatch"
    assert "'Number'" in result.findings[-1].message


def test_seasonal_adjustment_contract(tmp_path):
    spec = make_spec(source={"source": "singstat", "table_id": "M014811", "series": "1",
                             "unit_of_measure": "Million Dollars"},
                     unit_kind="level", seasonal_adjustment="seasonally_adjusted", plausible_range=[1, 10**7])
    content = singstat_payload({"1": quarters(2020, list(range(100, 126)), label=lambda y, q: f"{y} {q}Q")},
                               uom="Million Dollars")
    upstream = {"frequency": "Quarterly", "adjustment": "Non-seasonally Adjusted"}
    result = validate_series(spec, view_of(tmp_path, spec, content, upstream=upstream, extension="json"),
                             as_of=AS_OF)
    assert result.verdict == "rejected:contract_mismatch"


def test_frequency_contract(tmp_path):
    spec = make_spec(source={"source": "singstat", "table_id": "M1", "series": "1", "unit_of_measure": "Index"})
    content = singstat_payload({"1": quarters(2024, [1, 2, 3], label=lambda y, q: f"{y} {q}Q")})
    result = validate_series(spec, view_of(tmp_path, spec, content, upstream={"frequency": "Monthly"},
                                           extension="json"), as_of=AS_OF)
    assert result.verdict == "rejected:contract_mismatch"


def test_declared_lag_shorter_than_observed_warns(tmp_path):
    spec = make_spec(publication_lag_days=20)
    upstream = {"last_updated": "2026-07-27T09:27:44+08:00"}  # 27 days after 2026-06-30
    result = validate_series(spec, view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv"), upstream=upstream),
                             as_of=AS_OF)
    assert "declared_lag" in checks(result, Severity.WARN)
    assert result.observed_lag_days == 27


def test_out_of_band_values_warn(tmp_path):
    spec = make_spec(plausible_range=[1, 100])
    result = validate_series(spec, view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv")), as_of=AS_OF)
    assert "range" in checks(result, Severity.WARN)


def test_unusual_recent_move_warns(tmp_path):
    spec = make_spec(id="calm")
    values = [100 + 0.5 * i for i in range(40)] + [180]
    content = b"quarter,property_type,index\n" + b"".join(
        f"{p},All Residential,{v}\n".encode() for p, v in quarters(2016, values))
    result = validate_series(spec, view_of(tmp_path, spec, content), as_of=date(2026, 7, 31))
    assert "unusual_move" in checks(result, Severity.WARN)


def test_empty_trailing_observation_warns(tmp_path):
    spec = make_spec()
    content = fixture_bytes("tidy_quarter.csv").replace(b"2026-Q1,All Residential,218.3", b"2026-Q1,All Residential,na")
    result = validate_series(spec, view_of(tmp_path, spec, content), as_of=AS_OF)
    assert "gaps" in checks(result, Severity.WARN)


def test_revisions_against_previous_vintage(tmp_path):
    spec = make_spec()
    current = fixture_bytes("tidy_quarter.csv").replace(b"\r\n", b"\n")
    old = extract(spec.source, current.replace(b"213.2", b"213.0").replace(
        b"2026-Q2,All Residential,219.4\n", b""), Frequency.QUARTERLY)
    result = validate_series(spec, view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv")),
                             as_of=AS_OF, previous=old)
    assert result.revisions["revised_periods"] == 1
    assert result.revisions["new_periods"] == ["2026-Q2"]
    assert result.revisions["max_abs_revision"] == pytest.approx(0.2)


def test_cross_source_agreement_and_freshness(tmp_path):
    spec = rate_spec(cross_check={"ref": {"source": "datagovsg", "dataset_id": "d_rates", "layout": "wide",
                                          "row_label": "Compounded Singapore Overnight Rate Average (SORA) - 3 Month"},
                                  "tolerance": 0.00005})
    months = [("2026 Jul", 1.1354), ("2026 Jun", 1.0825), ("2026 Aug", 1.1863)]
    primary_content = singstat_payload({"23": sorted(months, key=lambda m: m[0])}, uom="Per Cent Per Annum")
    primary = view_of(tmp_path, spec, primary_content, extension="json",
                      upstream={"frequency": "Monthly", "series": {"23": {"unit_of_measure": "Per Cent Per Annum"}}})
    cross_vintage = store_vintage(tmp_path, "datagovsg/d_rates", fixture_bytes("wide_month.csv"), extension="csv")
    cross = SourceView(spec.cross_check.ref, state_for(cross_vintage),
                       extract(spec.cross_check.ref, fixture_bytes("wide_month.csv"), spec.frequency))
    result = validate_series(spec, primary, as_of=AS_OF, cross=cross)
    assert result.cross_check["mismatches"] == 0 and result.cross_check["overlap_periods"] == 2
    assert result.cross_check["primary_lead_periods"] == 1


def test_cross_source_mismatch_warns(tmp_path):
    spec = make_spec(cross_check={"ref": {"source": "singstat", "table_id": "M212261", "series": "1",
                                          "unit_of_measure": "Index"}, "tolerance": 0.05})
    primary = view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv"))
    content = singstat_payload({"1": [("2026 1Q", 218.3), ("2026 2Q", 221.0)]})
    vintage = store_vintage(tmp_path, "singstat/M212261", content)
    cross = SourceView(spec.cross_check.ref, state_for(vintage), extract(spec.cross_check.ref, content, spec.frequency))
    result = validate_series(spec, primary, as_of=AS_OF, cross=cross)
    assert result.cross_check["mismatches"] == 1
    assert "cross_source" in checks(result, Severity.WARN)


def test_degraded_retrieval_and_after_as_of_vintage_warn(tmp_path):
    spec = make_spec()
    view = view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv"), degraded=True, error="503", after_as_of=True)
    result = validate_series(spec, view, as_of=AS_OF)
    assert {"retrieval", "vintage"} <= set(checks(result, Severity.WARN))
    assert result.reportable


def test_no_snapshot_and_parse_failure_are_rejections(tmp_path):
    spec = make_spec()
    assert validate_series(spec, SourceView(spec.source, None), as_of=AS_OF).verdict == "rejected:no_data"
    vintage = store_vintage(tmp_path, spec.source.object_key, b"x", extension="csv")
    broken = SourceView(spec.source, state_for(vintage), None, "columns absent")
    assert validate_series(spec, broken, as_of=AS_OF).verdict == "rejected:parse_error"


def test_nothing_published_by_the_as_of_date(tmp_path):
    spec = make_spec()
    result = validate_series(spec, view_of(tmp_path, spec, fixture_bytes("tidy_quarter.csv")), as_of=date(2024, 3, 1))
    assert result.verdict == "rejected:nothing_admitted"
