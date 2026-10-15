"""End-to-end, offline, on the committed snapshot store and response cache."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sgmacro.cli import app
from sgmacro.pipeline import analyse, fetch_stage, run
from sgmacro.registry import Role, load_registry
from sgmacro.trace import verify_fact
from tests.conftest import AS_OF, REGISTRY_PATH, ROOT, STORE_ROOT, config_in


def test_offline_run_reproduces_the_committed_report(committed_run):
    tmp, result = committed_run
    for name in ("report.md", "report.html", "charts/targets.svg", "charts/predictors.svg"):
        produced = (tmp / "reports" / result.run_id / name).read_bytes()
        assert produced == (ROOT / "reports" / result.run_id / name).read_bytes(), name
    for name in ("facts.json", "observations.csv"):
        assert (tmp / "processed" / result.run_id / name).read_bytes() == \
            (ROOT / "data" / "processed" / result.run_id / name).read_bytes()


def test_repeated_request_is_byte_identical(committed_run, tmp_path):
    tmp, first = committed_run
    second = run(config_in(tmp_path), log=lambda m: None)
    assert second.run_id == first.run_id
    rid = first.run_id
    for rel in (f"reports/{rid}/report.md", f"reports/{rid}/report.html", f"reports/{rid}/charts/predictors.svg",
                f"runs/{rid}/manifest.json", f"processed/{rid}/facts.json"):
        assert (tmp / rel).read_bytes() == (tmp_path / rel).read_bytes(), rel


def test_every_reported_figure_recomputes_from_raw_bytes(committed_run):
    """Second implementation: re-hash each snapshot, re-read each cell with the
    minimal reader in sgmacro.trace, recompute with the formula for the fact's kind."""
    tmp, result = committed_run
    facts = json.loads((tmp / "processed" / result.run_id / "facts.json").read_text(encoding="utf-8"))
    specs = {s.id: s for s in result.registry.series}
    checked = 0
    for fact in facts:
        if not fact["inputs"]:
            continue
        ok, checks, recomputed = verify_fact(fact, specs, STORE_ROOT)
        assert ok, (fact["fact_id"], fact["label"], recomputed, fact["value"])
        checked += 1
    assert checked == sum(f["kind"] in ("level", "change", "real_change") for f in facts) >= 45


def test_every_fact_id_in_the_report_exists(committed_run):
    tmp, result = committed_run
    report = (tmp / "reports" / result.run_id / "report.md").read_text(encoding="utf-8")
    ids = set(re.findall(r"F:[0-9a-f]{8}", report))
    known = {f.fact_id for f in result.facts}
    assert ids and ids <= known


def test_html_report_is_self_contained_and_every_figure_opens(committed_run):
    """Every clickable figure in the HTML report has its lineage in the embedded
    payload, the payload is strict JSON, and nothing is loaded from elsewhere."""
    tmp, result = committed_run
    html = (tmp / "reports" / result.run_id / "report.html").read_text(encoding="utf-8")
    blob = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', html, re.S).group(1)
    payload = json.loads(blob, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    clickable = set(re.findall(r'data-fact="(F:[0-9a-f]{8})"', html))
    assert len(clickable) > 100 and clickable <= set(payload["facts"])
    assert set(payload["charts"]["predictors"]) == {s.id for s in result.registry.predictors}
    assert not re.search(r'<script[^>]+src=|<link[^>]+href=|@import', html)
    assert f"reports/{result.run_id}/report.html" in result.outputs


def test_negative_controls_fail_exactly_as_declared(committed_run):
    _, result = committed_run
    controls = result.by_role(Role.QUARANTINE)
    assert len(controls) == 3 and all(o.result.as_expected for o in controls)
    assert all(o.result.reportable for o in result.outcomes.values() if o.spec.role is not Role.QUARANTINE)


def test_cached_summary_passes_the_checker(committed_run):
    _, result = committed_run
    assert result.summary.from_cache and result.summary.source == "ollama:qwen3.5:27b"
    for market in result.summary.markets.values():
        assert market.source != "deterministic" and len(market.accepted) >= 2


def test_a_live_draft_and_its_cached_replay_write_the_same_report(tmp_path, monkeypatch):
    """Cold cache (the model is called) and warm cache must produce the same bytes
    and run id; only the execution log says which one happened. The resolver reports
    why another backend was skipped, as the real one does; that too is execution
    detail and must not reach the report or the manifest."""
    cached = json.loads(next((ROOT / "llm_cache").glob("*.json")).read_text(encoding="utf-8"))

    class Replay:
        name, model = "ollama", "qwen3.5:27b"

        def complete_json(self, system, prompt, schema):
            return cached["response"]

    monkeypatch.setattr("sgmacro.llm.summary.resolve_backend",
                        lambda p, **kw: (Replay(), ["anthropic: no ANTHROPIC_API_KEY in the environment"]))
    cache = tmp_path / "cache"
    cold = run(config_in(tmp_path / "a", llm="auto", llm_cache=cache), log=lambda m: None)
    warm = run(config_in(tmp_path / "b", llm="auto", llm_cache=cache), log=lambda m: None)
    assert not cold.summary.from_cache and warm.summary.from_cache
    assert cold.run_id == warm.run_id
    rid = cold.run_id
    for rel in (f"reports/{rid}/report.md", f"reports/{rid}/report.html", f"runs/{rid}/manifest.json"):
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes(), rel
    log = json.loads((tmp_path / "b" / "runs" / warm.run_id / "executions.jsonl").read_text(encoding="utf-8"))
    assert log["summary_from_cache"] is True
    cold_log = json.loads((tmp_path / "a" / "runs" / rid / "executions.jsonl").read_text(encoding="utf-8"))
    assert any("ANTHROPIC_API_KEY" in n for n in cold_log["summary_notes"])


def test_report_renders_at_the_edges_instead_of_crashing(committed_run, tmp_path):
    """An as-of date before the charts' window, a second source with no overlapping
    periods, and a run with no usable predictor each produce a report, not a crash."""
    from types import SimpleNamespace

    from sgmacro.report.charts import predictors_chart
    from sgmacro.report.render import series_context

    early = run(config_in(tmp_path / "early", as_of=date(2004, 1, 1), llm="off"), log=lambda m: None)
    assert (tmp_path / "early" / "reports" / early.run_id / "report.html").exists()

    _, result = committed_run
    outcome = result.outcomes["ura_ppi"]
    saved = outcome.result.cross_check
    outcome.result.cross_check = {**saved, "overlap_periods": 0, "mismatches": 0, "max_abs_diff": None}
    try:
        ctx = series_context(result, outcome)
    finally:
        outcome.result.cross_check = saved
    assert ctx["cross_ok"] is False and "no overlapping periods" in ctx["cross_check"]

    predictors_chart(SimpleNamespace(reportable=lambda role: []), tmp_path / "none.svg")
    assert (tmp_path / "none.svg").stat().st_size > 0
    assert early.run_id.startswith("2004-01-01_")


def test_a_different_summary_gets_a_different_run_id_and_its_own_report(committed_run, tmp_path):
    """Two runs for one date (cached LLM summary, then --llm off) must not overwrite
    each other: each manifest keeps pointing at the bytes it hashed."""
    _, result = committed_run
    first = run(config_in(tmp_path, llm="cache"), log=lambda m: None)
    off = run(config_in(tmp_path, llm="off"), log=lambda m: None)
    assert off.run_id != first.run_id == result.run_id
    for r in (first, off):
        manifest = json.loads((tmp_path / "runs" / r.run_id / "manifest.json").read_text(encoding="utf-8"))
        for rel, digest in manifest["outputs"].items():
            if rel.startswith("reports/"):
                path = tmp_path / rel
                assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, rel


def test_template_summaries_with_different_notes_get_different_run_ids(tmp_path, monkeypatch):
    """--llm off and a model answer that is valid JSON of the wrong shape both end on
    the template, but the second report carries a note saying why: different content,
    so a different run id. Replaying the same unusable answer from the cache keeps its id."""
    class Garbled:
        name, model = "ollama", "qwen3.5:27b"

        def complete_json(self, system, prompt, schema):
            return "[1]"

    off = run(config_in(tmp_path / "off", llm="off"), log=lambda m: None)
    monkeypatch.setattr("sgmacro.llm.summary.resolve_backend", lambda p, **kw: (Garbled(), []))
    cache = tmp_path / "cache"
    garbled = run(config_in(tmp_path / "g1", llm="auto", llm_cache=cache), log=lambda m: None)
    replay = run(config_in(tmp_path / "g2", llm="auto", llm_cache=cache), log=lambda m: None)
    assert garbled.summary.source == "deterministic" and garbled.summary.notes
    assert garbled.run_id != off.run_id and replay.run_id == garbled.run_id and replay.summary.from_cache


def test_fetch_errors_shown_in_the_report_are_part_of_the_run_id(committed_run):
    """Same snapshots, two different fetch failures (both falling back to the stored
    vintage): the reports show different errors, so the runs must not share an id.
    The same failure again gives the same id, and a failure with no vintage counts."""
    from dataclasses import replace as copy_state

    from sgmacro.pipeline import run_identity

    _, result = committed_run
    states = dict(result.states)
    key = next(k for k in states if k.startswith("singstat/"))

    def with_error(error, vintage=True):
        s = states[key]
        return {**states, key: copy_state(s, degraded=vintage, error=error, selection=s.selection if vintage else None)}

    ids = [run_identity(AS_OF, result.registry, with_error(e)) for e in ("HTTP 429", "HTTP 503", "HTTP 429")]
    assert ids[0] != ids[1] and ids[0] == ids[2]
    assert run_identity(AS_OF, result.registry, with_error("HTTP 503", vintage=False)) not in ids


def test_run_says_where_it_saved_everything(committed_run, monkeypatch):
    from typer.testing import CliRunner

    from sgmacro.cli import app

    _, result = committed_run
    monkeypatch.setattr("sgmacro.cli.run", lambda config, log: result)
    out = CliRunner().invoke(app, ["run", "--as-of", "2026-10-02", "--offline", "--llm", "off"])
    assert out.exit_code == 0, out.output
    rid = result.run_id
    for line in (f"report:           reports/{rid}/report.md", f"processed tables: ", f"/{rid}/",
                 "raw snapshots:    ", "0 new vintage(s), 20 reused", f"run record:       ", "/manifest.json"):
        assert line in out.output, line


def test_run_shows_skipped_backends_only_when_no_model_drafted(tmp_path, monkeypatch):
    """Live: Ollama drafted the summary, yet the run printed why Claude and the
    OpenAI-compatible service were skipped, which read like errors. Through the real
    summary, pipeline and CLI: the skipped backend stays in the run record, and is
    shown only when no model drafted."""
    cached = json.loads(next((ROOT / "llm_cache").glob("*.json")).read_text(encoding="utf-8"))

    class Replay:
        name, model = "ollama", "qwen3.5:27b"

        def complete_json(self, system, prompt, schema):
            return cached["response"]

    skipped = ["anthropic: no ANTHROPIC_API_KEY in the environment"]
    for case, backend, shown in (("drafted", Replay(), False), ("template", None, True)):
        out_dir = tmp_path / case
        monkeypatch.setattr("sgmacro.llm.summary.resolve_backend", lambda p, b=backend, **kw: (b, skipped))
        monkeypatch.setattr("sgmacro.cli.run", lambda config, log, d=out_dir: run(
            config_in(d, llm=config.llm, llm_cache=d / "cache"), log=log))
        out = CliRunner().invoke(app, ["run", "--as-of", "2026-10-02", "--offline"])
        assert out.exit_code == 0, out.output
        assert ("ANTHROPIC_API_KEY" in out.output) is shown, case
        assert ("summary: ollama:qwen3.5:27b" in out.output) is not shown, case
        record = json.loads(next((out_dir / "runs").glob("*/executions.jsonl")).read_text(encoding="utf-8"))
        assert skipped[0] in record["summary_notes"], case


def test_a_past_report_lists_only_policy_rounds_announced_by_then(tmp_path):
    from sgmacro.report.render import build_context

    registry = load_registry(REGISTRY_PATH)
    config = config_in(tmp_path, as_of=date(2020, 10, 2))
    result = analyse(config, registry, fetch_stage(config, registry, lambda m: None))
    policy = next(p for p in build_context(result)["predictors"] if p["id"] == "policy_tightening")
    assert policy["latest_period"] == "2020 Q3"
    assert policy["events"] and all(e["date"] <= "2020-10-02" for e in policy["events"])
    assert policy["events_total"] == 14          # the rounds announced 2005-2018


def test_trace_accepts_a_change_left_undefined_by_a_series_break(tmp_path):
    """In 1995 the vacancy ratio's year-on-year change spans the 1995-Q1 break in the
    stock count, so the pipeline leaves it undefined; trace must agree, not recompute it."""
    registry = load_registry(REGISTRY_PATH)
    config = config_in(tmp_path, as_of=date(1995, 8, 1), llm="off")
    result = analyse(config, registry, fetch_stage(config, registry, lambda m: None))
    vacancy = result.outcomes["vacancy_nonlanded"]
    fact = result.facts.get(vacancy.facts["yoy_pp"]).to_json()
    assert fact["value"] is None and "series break" in fact["note"]
    ok, checks, recomputed = verify_fact(fact, {s.id: s for s in registry.series}, STORE_ROOT)
    assert ok and recomputed is None and all(c.ok for c in checks)


def test_cross_checks_agree_and_rates_are_fresher_on_singstat(committed_run):
    _, result = committed_run
    for sid in ("ura_ppi", "ura_rri_nonlanded", "hdb_rpi", "sora_3m", "sgs_10y", "vacancy_nonlanded"):
        assert result.outcomes[sid].result.cross_check["mismatches"] == 0, sid
    assert result.outcomes["sora_3m"].result.cross_check["primary_lead_periods"] == 1


def test_manifest_records_vintages_and_no_wall_clock(committed_run):
    tmp, result = committed_run
    manifest = json.loads((tmp / "runs" / result.run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["as_of"] == AS_OF.isoformat() and len(manifest["vintages"]) == 20
    text = json.dumps({k: v for k, v in manifest.items() if k != "vintages"})
    assert "started_at" not in text and "finished_at" not in text
    log = (tmp / "runs" / result.run_id / "executions.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[0])["downloads"] == 0


def test_earlier_as_of_sees_only_what_was_published_then():
    registry = load_registry(REGISTRY_PATH)
    config = config_in(Path("unused"), as_of=date(2026, 7, 15))
    result = analyse(config, registry, fetch_stage(config, registry, lambda m: None))
    ppi = result.outcomes["ura_ppi"].result
    assert ppi.latest[0].label == "2026-Q1" and "2026-Q2" in ppi.withheld
    pop = result.outcomes["nonresident_population"].result
    assert pop.latest[0].label == "2025"   # the end-June 2026 estimate appeared on 25 Sep
    # every snapshot postdates this as-of date, which the validator must say
    assert any(f.check == "vintage" for f in ppi.findings)


def test_llm_off_falls_back_to_the_template(tmp_path):
    result = run(config_in(tmp_path, llm="off"), log=lambda m: None)
    assert result.summary.source == "deterministic"
    assert all(m.fallback_text for m in result.summary.markets.values())


def test_cli_sources_validate_and_trace(committed_run, monkeypatch):
    tmp, result = committed_run
    monkeypatch.chdir(ROOT)
    runner = CliRunner()
    assert "ura_ppi" in runner.invoke(app, ["sources"]).output
    validated = runner.invoke(app, ["validate", "--as-of", AS_OF.isoformat()])
    assert validated.exit_code == 0 and "rejected:contract_mismatch" in validated.output
    fact = result.outcomes["sora_3m"].facts["headline"]
    traced = runner.invoke(app, ["trace", fact, "--run", result.run_id])
    assert traced.exit_code == 0 and "VERIFIED" in traced.output, traced.output
    # --as-of works only while that date has a single run; it never guesses between runs
    assert runner.invoke(app, ["trace", fact, "--as-of", AS_OF.isoformat()]).exit_code == 0
    # a later registry must not be used to judge an older run's facts
    edited = tmp / "edited.yaml"
    edited.write_text(REGISTRY_PATH.read_text(encoding="utf-8").replace(
        "publication_lag_days: 31", "publication_lag_days: 32", 1), encoding="utf-8")
    stale = runner.invoke(app, ["trace", fact, "--run", result.run_id, "--registry", str(edited)])
    assert stale.exit_code == 2 and "trace it with that registry version" in stale.output
    assert runner.invoke(app, ["run", "--as-of", "2999-01-01"]).exit_code != 0


def test_credentials_in_error_text_stop_the_run_before_anything_is_written(committed_run, tmp_path):
    from sgmacro.report.render import write_outputs
    from sgmacro.validate import Finding, Severity

    _, result = committed_run
    outcome = result.outcomes["cpi_all_items"]
    leaked = Finding("fetch", Severity.WARN, "download failed: /x.csv?X-Amz-Signature=DUMMY_REVIEW")
    outcome.result.findings.append(leaked)
    try:
        moved = replace(result, config=config_in(tmp_path))
        with pytest.raises(ValueError, match="X-Amz-Signature"):
            write_outputs(moved)
    finally:
        outcome.result.findings.remove(leaked)
    assert not (tmp_path / "reports").exists() and not (tmp_path / "processed").exists()


def test_the_markdown_report_says_why_a_used_series_was_rejected(committed_run):
    from sgmacro.report.render import render_markdown
    from sgmacro.validate import Finding, Severity, Status

    _, result = committed_run
    r = result.outcomes["cpi_accommodation"].result
    saved = (r.status, r.reason, list(r.findings))
    r.status, r.reason = Status.REJECTED, "no_data"
    r.findings.append(Finding("fetch", Severity.ERROR, "UNIQUE_REVIEW_REASON: upstream returned nothing"))
    try:
        text = render_markdown(result)
    finally:
        r.status, r.reason, r.findings[:] = saved[0], saved[1], saved[2]
    assert "CPI accommodation (rejected:no_data: UNIQUE_REVIEW_REASON" in text
