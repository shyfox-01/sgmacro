"""The committed sample report is well-formed and complete."""

from __future__ import annotations

import re

import pytest

from sgmacro.registry import Role
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def report_dir(committed_run):
    """The committed sample, found by the run id the offline replay produces, so other
    runs for the same date beside it in reports/ do not matter."""
    return ROOT / "reports" / committed_run[1].run_id


@pytest.fixture(scope="module")
def report(report_dir):
    return (report_dir / "report.md").read_text(encoding="utf-8")


def test_sections_present(report):
    for heading in ("## 1. Summary", "## 2. Signal dashboard", "## 3. Target indices", "## 4. Predictors",
                    "## 5. Lead-lag diagnostics", "## 6. Data quality and validation",
                    "## 7. Provenance and reproduction"):
        assert heading in report


def test_tables_have_consistent_widths(report):
    lines = report.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|( ?-+ ?\|)+$", lines[i + 1]):
            width = lines[i].count("|")
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                cells = re.sub(r"`[^`]*`", "", lines[j])   # pipes inside code spans are literal
                assert cells.count("|") == width, (i + 1, lines[j])
                j += 1
            i = j
        else:
            i += 1


def test_no_rendering_artefacts(report):
    for bad in ("nan", "None", "-0.0", "+0.0", "{{", "}}", "{%"):
        assert not re.search(rf"(?<![\w.]){re.escape(bad)}(?![\w])", report), bad


def test_every_predictor_reports_the_required_fields(registry, report):
    for spec in registry.by_role(Role.PREDICTOR):
        block = report.split(f"### {spec.short_label}\n", 1)[1].split("\n### ", 1)[0]
        # The task's required fields: definition, unit, observation and update frequency,
        # source identifier, latest observation and as-of date, changes, why, lags, limitations.
        for field in ("**Source:**", "**Definition:**", "**Unit:**", "**Frequency:**", "**Updates:**",
                      "**Latest:**", ", as of 2026-10-02", "**Publication lag:**", "**Changes:**",
                      "**Why it may be informative:**", "**Lags:**", "**Limitations:**"):
            assert field in block, (spec.id, field)


def test_report_stays_concise(report):
    """Detail lives in the processed tables, the manifest and the HTML report."""
    assert len(report.splitlines()) < 400


def test_charts_exist_and_carry_no_timestamp(report_dir):
    for name in ("targets.svg", "predictors.svg"):
        svg = (report_dir / "charts" / name).read_text(encoding="utf-8")
        assert svg.startswith("<?xml") and "<dc:date>" not in svg
