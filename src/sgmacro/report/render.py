"""Render the run: Markdown report, charts, processed tables and run manifest.

The template receives display strings only, each produced from a :class:`Fact`
(or from the registry), so the report cannot print a number that has no fact id.
No wall-clock time is rendered, which keeps repeated runs byte-identical.
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined

from sgmacro.diagnostics import LAGS
from sgmacro.facts import ordinal_suffix
from sgmacro.periods import parse_period
from sgmacro.registry import Adjustment, CuratedRef, Market, Role, SingStatRef, Transform
from sgmacro.report.charts import predictors_chart, targets_chart

READING_LABEL = {"supportive": "supportive", "headwind": "headwind", "neutral": "neutral",
                 "ambiguous": "ambiguous", "unavailable": "n/a"}
EFFECT_TEXT = {"positive": "a rise lifts", "negative": "a rise weighs on", "ambiguous": "no clear sign for"}
VERB = {"up": "rose", "down": "fell", "flat": "was little changed"}


def _period(result, series_id: str, label: str | None) -> str:
    if not label:
        return ""
    return parse_period(label, result.registry.get(series_id).frequency).display


def _magnitude(display: str) -> str:
    return display.lstrip("+-")


def series_context(result, outcome) -> dict[str, Any]:
    spec, r = outcome.spec, outcome.result
    ref = spec.source
    ctx: dict[str, Any] = {
        "id": spec.id, "label": spec.label, "short": spec.short_label, "role": spec.role.value,
        "publisher": spec.publisher,
        "source_name": ("SingStat Table Builder" if isinstance(ref, SingStatRef)
                        else "a hand-maintained list of official announcements" if isinstance(ref, CuratedRef)
                        else "data.gov.sg"),
        # Curated lists live in the repository; link relative to reports/<as-of>/.
        "link": f"../../{ref.link}" if isinstance(ref, CuratedRef) else ref.link,
        "identifier": ref.identifier,
        "definition": " ".join(spec.definition.split()), "unit": spec.unit,
        "unit_kind": spec.unit_kind.value, "frequency": spec.frequency.value,
        "adjustment": ("seasonal adjustment not applicable" if spec.seasonal_adjustment is Adjustment.NOT_APPLICABLE
                       else spec.seasonal_adjustment.value.replace("_", " ")),
        "update_frequency": spec.update_frequency, "lag": spec.publication_lag_days,
        "reference_month": spec.reference_month,
        "observed_lag": r.observed_lag_days, "verdict": r.verdict,
        "vintage": r.vintage.vintage_id if r.vintage else "-",
        "upstream_updated": (r.vintage.upstream.get("last_updated") if r.vintage else None) or "-",
        "rationale": " ".join((spec.rationale or "").split()),
        "limitations": " ".join((spec.limitations or "").split()),
        "channel": spec.channel, "warnings": [f.message for f in r.findings if f.severity.value == "warn"],
        "withheld": r.withheld[-3:],
        "expected_status": spec.expected_status or "–",
        "comparability": ([" ".join(spec.history_note.split())] if spec.history_note else [])
        + [f"series break at {_period(result, spec.id, item.period)}: {item.note}; changes spanning it are "
           "excluded from changes, percentiles and correlations" for item in spec.breaks],
        "errors": [f.message for f in r.findings if f.severity.value == "error"],
        "reference_note": (f" (counted from the end-{calendar.month_name[spec.reference_month]} reference date)"
                           if spec.reference_month else ""),
    }
    if r.latest is not None:
        ctx["latest_period"] = _period(result, spec.id, r.latest[0].label)
    if outcome.changes is None:
        return ctx
    facts = result.facts
    level = facts.get(outcome.facts["level"])
    head = facts.get(outcome.facts["headline"])
    pct = facts.get(outcome.facts["percentile"])
    ctx.update({
        "latest_period": _period(result, spec.id, level.period), "level": level.display,
        "level_id": level.fact_id, "headline": head.display, "headline_id": head.fact_id,
        "headline_human": outcome.changes.headline.human,
        "percentile": pct.display, "percentile_id": pct.fact_id,
        "history_start": _period(result, spec.id, outcome.changes.history_start),
        "changes": [],
    })
    for change in outcome.changes.changes:
        fact = facts.get(outcome.facts[change.transform.value])
        base = change.inputs[0] if change.inputs else None
        end = _period(result, spec.id, change.period.label)
        if len(change.inputs) == 2:
            basis = (f"{_period(result, spec.id, base.period)} = {_fmt_input(base.value, spec.decimals)} -> "
                     f"{end} = {_fmt_input(change.inputs[-1].value, spec.decimals)}")
        elif change.transform.value == "t4q_sum":
            per_quarter = ", ".join(f"{_period(result, spec.id, i.period)}: {_fmt_input(i.value, 0)}"
                                    for i in change.inputs)
            basis = f"net rounds per quarter, {per_quarter}"
        else:
            basis = f"sum of 4 quarters to {end} vs prior 4"
        unit = {"pct": "%", "pp": "pp", "bp": "bp", "count": "rounds"}[change.unit]
        ctx["changes"].append({"name": f"{change.human} ({unit})", "value": fact.display, "basis": basis,
                               "id": fact.fact_id, "note": change.note})
    if "real_yoy" in outcome.facts:
        real = facts.get(outcome.facts["real_yoy"])
        ctx["real"], ctx["real_id"] = real.display, real.fact_id
        ctx["changes"].append({"name": "real year-on-year change, CPI-deflated (%)", "value": real.display,
                               "basis": "nominal YoY deflated by quarterly-average CPI all items",
                               "id": real.fact_id, "note": real.note})
    for key in ("qoq", "yoy"):
        if key in outcome.facts:
            ctx[key] = facts.get(outcome.facts[key]).display
            ctx[f"{key}_id"] = outcome.facts[key]
    # signals and the deterministic reading sentence
    ctx["signals"] = {s.market.value: READING_LABEL[s.reading] for s in outcome.signals}
    ctx["signal_ids"] = {s.market.value: outcome.facts.get(f"signal_{s.market.value}") for s in outcome.signals}
    if spec.role is Role.PREDICTOR:
        # A trailing sum is a total over four quarters, not a move between two dates.
        is_total = outcome.changes.headline.transform is Transform.T4Q_SUM
        if is_total:
            noun = {"up": "positive total", "down": "negative total"}.get(head.direction or "", "near-zero total")
        else:
            noun = {"up": "rise", "down": "fall"}.get(head.direction or "", "small move")
        effects = "; ".join(f"on {s.market.value} it is {s.effect.value} ({EFFECT_TEXT[s.effect.value]} the market), "
                            f"so this {noun} reads **{s.reading}**" for s in outcome.signals)
        if is_total:
            ctx["reading"] = (f"{spec.short_label}: {head.display} ({outcome.changes.headline.human}) to "
                              f"{ctx['latest_period']}, of which {level.display} in the latest quarter; the "
                              f"{pct.display} percentile of its history since {ctx['history_start']}. "
                              f"Declared effect: {effects}.")
        else:
            verb = VERB.get(head.direction or "flat", "changed")
            moved = "" if head.direction == "flat" else f" {_magnitude(head.display)}"
            ctx["reading"] = (f"{spec.short_label} {verb}{moved} ({outcome.changes.headline.human}) to "
                              f"{level.display} ({spec.unit}) in {ctx['latest_period']}, the {pct.display} "
                              f"percentile of its history since {ctx['history_start']}. Declared effect: {effects}.")
        lo, hi = spec.expected_lead_quarters  # type: ignore[misc]
        ctx["lead"] = f"{lo}-{hi} quarters" if lo != hi else f"{lo} quarters"
        ctx["evidence"] = _evidence(result, spec.id)
    if isinstance(ref, CuratedRef) and r.vintage is not None:
        from sgmacro.curated import load_policy_events

        listed = load_policy_events(r.vintage.data_bytes())
        # Only rounds announced by the as-of date: a report for 2020 must not list 2023.
        known = [e for e in listed.events if e.announced <= result.config.as_of]
        ctx["events"] = [{"date": e.announced.isoformat(), "direction": e.direction,
                          "instruments": ", ".join(e.instruments), "summary": e.summary, "source": e.source}
                         for e in reversed(known[-6:])]
        ctx["events_total"] = len(known)
        ctx["events_reviewed"] = listed.reviewed_on.isoformat()
    if r.cross_check and r.cross_check.get("available"):
        cc = r.cross_check
        ctx["cross_ok"] = cc["overlap_periods"] > 0 and cc["mismatches"] == 0
        if cc["overlap_periods"]:
            ctx["cross_check"] = (f"{cc['source']} ({cc['identifier']}): {cc['overlap_periods']} periods compared, "
                                  f"{cc['mismatches']} mismatches, max |difference| {cc['max_abs_diff']:.4g}")
        else:
            ctx["cross_check"] = f"{cc['source']} ({cc['identifier']}): no overlapping periods to compare"
    return ctx


def _fmt_input(value: float | None, decimals: int) -> str:
    """An input observation as published: the series' decimals, more if needed (max four)."""
    if value is None:
        return "n/a"
    for places in range(decimals, 5):
        text = f"{value:,.{places}f}"
        if abs(float(text.replace(",", "")) - value) < 1e-9:
            return text
    return f"{value:,.4f}"


def _evidence(result, predictor_id: str) -> list[dict[str, Any]]:
    """Per target: the verdict inside the declared lead window, and the strongest
    correlation outside it when that tells a different story."""
    from sgmacro.diagnostics import MIN_PAIRS
    from sgmacro.facts import fact_id

    rows = []
    for d in result.diagnostics:
        if d.predictor_id != predictor_id:
            continue
        target = result.registry.get(d.target_id).short_label
        if d.best is None:
            pairs = max((c.pairs for c in d.correlations if d.in_window(c.lag)), default=0)
            text = f"{target}: too few overlapping quarters to measure ({pairs}, minimum {MIN_PAIRS})"
            rows.append({"text": text, "id": None, "outside": None})
            continue
        text = f"{target} {d.best.r:+.2f} at *k*={d.best.lag} ({d.agreement}, {d.best.pairs} pairs)"
        out = d.strongest_outside
        rows.append({"text": text, "id": fact_id(d.predictor_id, d.target_id, "corr", d.best.lag),
                     "outside": None if out is None else f"{target} {out.r:+.2f} at *k*={out.lag}"})
    return rows


def cited_figures(result, ids: list[str]) -> list[dict[str, Any]]:
    """The figures a summary claim cites, written out by code: the claim's text has
    no numbers of its own. A signal is shown with the move it was read from, cited
    or not, as one item: "3M compounded SORA -38 bp, Aug 2026 (supportive)"."""
    facts = result.facts
    shown_with_signal = {facts.get(i).depends_on[0] for i in ids if facts.get(i).kind == "signal"}
    out = []
    for i in ids:
        fact = facts.get(i)
        if i in shown_with_signal:
            continue
        if fact.kind == "signal":
            short = result.registry.get(fact.series_id).short_label
            head = facts.get(fact.depends_on[0])
            text = f"{short} {head.display}, {_period(result, fact.series_id, head.period)}"
            out.append({"id": i, "text": text, "reading": fact.display, "title": fact.label})
        else:
            period = _period(result, fact.series_id, fact.period)
            out.append({"id": i, "text": f"{fact.label} {fact.display}" + (f", {period}" if period else ""),
                        "reading": None, "title": fact.label})
    return out


def diagnostics_view(result) -> dict[str, Any]:
    """Lead-lag results per predictor and target: the verdict inside the declared
    window and any stronger correlation outside it. Both report formats lay out
    this one view."""
    from sgmacro.facts import fact_id

    targets = [o.spec for o in result.reportable(Role.TARGET)]
    rows = []
    for outcome in result.reportable(Role.PREDICTOR):
        found = {d.target_id: d for d in result.diagnostics if d.predictor_id == outcome.spec.id}
        cells = []
        for target in targets:
            d = found.get(target.id)
            best = d.best if d else None
            out = d.strongest_outside if d else None
            cells.append({
                "target": target.short_label.split(" (")[0],
                "r": None if best is None else best.r, "lag": None if best is None else best.lag,
                "agreement": d.agreement if d else "insufficient data",
                "fact": None if best is None else fact_id(outcome.spec.id, target.id, "corr", best.lag),
                "text": "insufficient data" if best is None else f"{best.r:+.2f} (k={best.lag}) {d.agreement}",
                "outside": None if out is None else f"{out.r:+.2f} (k={out.lag})",
            })
        lo, hi = outcome.spec.expected_lead_quarters  # type: ignore[misc]
        rows.append({"predictor": outcome.spec.short_label, "lead": f"{lo}-{hi}" if lo != hi else f"{lo}",
                     "cells": cells,
                     "outside": "; ".join(f"{c['target']} {c['outside']}" for c in cells if c["outside"]) or "–"})
    return {"targets": [t.short_label for t in targets], "rows": rows}


def build_context(result) -> dict[str, Any]:
    targets = [series_context(result, o) for o in result.reportable(Role.TARGET)]
    predictors = [series_context(result, o) for o in result.reportable(Role.PREDICTOR)]
    every = [series_context(result, o) for o in result.outcomes.values()]
    summary = result.summary
    summary_ctx = None
    if summary is not None:
        markets = []
        for market in Market:
            ms = summary.markets[market.value]
            markets.append({
                "name": market.value.capitalize(),
                "source": ms.source,
                "claims": [{"text": c.text, "figures": cited_figures(result, [summary.resolve(a) for a in c.facts])}
                           for c in ms.accepted]
                if ms.source != "deterministic" else [],
                "fallback": ms.fallback_text,
                "accepted": len(ms.accepted), "total": len(ms.claims),
            })
        summary_ctx = {
            "source": summary.source, "model": summary.model, "markets": markets, "notes": summary.notes,
            "audit": [{"market": c.market, "text": c.text, "ok": c.ok, "violations": c.violations,
                       "stance": c.stance}
                      for ms in summary.markets.values() for c in ms.claims],
        }
    diag = diagnostics_view(result)

    cross = []
    revisions = []
    for o in result.outcomes.values():
        cc = o.result.cross_check
        if cc:
            cross.append({"series": o.spec.short_label, **{k: cc.get(k) for k in (
                "source", "identifier", "overlap_periods", "mismatches", "primary_latest", "cross_latest")},
                "max_abs_diff": "n/a" if cc.get("max_abs_diff") is None else f"{cc['max_abs_diff']:.4g}",
                "available": cc.get("available"), "reason": cc.get("reason")})
        if o.result.revisions:
            revisions.append({"series": o.spec.short_label, **o.result.revisions})

    vintages = []
    for key, state in sorted(result.states.items()):
        v = state.vintage
        vintages.append({"object": key, "vintage": v.vintage_id if v else "-",
                         "retrieved": v.retrieved_at.strftime("%Y-%m-%d %H:%M UTC") if v else "-",
                         "upstream": (v.upstream.get("last_updated") if v else None) or "-",
                         "sha": v.data_sha256[:12] if v else "-",
                         "flags": ", ".join(f for f, on in (("degraded", state.degraded),
                                                            ("after as-of", bool(state.selection and state.selection.after_as_of)))
                                            if on) or ""})
    controls = [o for o in result.outcomes.values() if o.spec.role is Role.QUARANTINE]
    return {
        "as_of": result.config.as_of.isoformat(),
        "run_id": result.run_id,
        "registry_version": result.registry.version,
        "fingerprint": result.registry.fingerprint()[:12],
        "pipeline_version": __import__("sgmacro").__version__,
        "targets": targets, "predictors": predictors, "every": every,
        "n_facts": len(result.facts),
        "controls_ok": sum(1 for o in controls if o.result.as_expected),
        "controls_total": len(controls),
        "unexpected_rejections": [o.spec.id for o in result.outcomes.values()
                                  if o.spec.role is not Role.QUARANTINE and not o.result.reportable],
        "summary": summary_ctx,
        "diagnostics": diag["rows"], "diag_targets": diag["targets"],
        "lags": list(LAGS),
        "cross": cross, "revisions": revisions, "vintages": vintages,
        "selection_rules": result.registry.selection_rules,
        "ordinal": ordinal_suffix,
    }


def render_markdown(result) -> str:
    env = Environment(loader=PackageLoader("sgmacro", "templates"), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    text = env.get_template("report.md.j2").render(**build_context(result))
    # Collapse runs of blank lines left by conditional blocks.
    lines, blank = [], 0
    for line in text.splitlines():
        blank = blank + 1 if not line.strip() else 0
        if blank <= 1:
            lines.append(line.rstrip())
    return "\n".join(lines).strip() + "\n"


def write_outputs(result) -> None:
    """Render every output in memory, check it, then write.

    Each run has its own report directory (``reports/<run id>/``), so two runs for
    the same date, say with and without the LLM summary, never overwrite each
    other and every manifest keeps pointing at the bytes it hashed.
    """
    from sgmacro.pipeline import file_digest, manifest, processed_tables
    from sgmacro.report.html import render_html
    from sgmacro.store import assert_no_credentials

    config = result.config
    report_rel = f"reports/{result.run_id}"
    report_dir = config.reports_root / result.run_id
    processed_dir = config.processed_root / result.run_id
    run_dir = config.runs_root / result.run_id

    tables = processed_tables(result)
    markdown = render_markdown(result)
    html = render_html(result, build_context(result))
    # Error text from a failed download reaches the validation table and both
    # reports; refuse before anything is written, not halfway through.
    for text in (*tables.values(), markdown, html):
        assert_no_credentials(text)

    for d in (report_dir, processed_dir, run_dir):
        d.mkdir(parents=True, exist_ok=True)
    targets_chart(result, report_dir / "charts" / "targets.svg")
    predictors_chart(result, report_dir / "charts" / "predictors.svg")
    # Manifest keys are logical paths, independent of where the roots point, so the
    # same request run into two different directories records the same manifest.
    outputs: dict[str, str] = {}
    for name, content in tables.items():
        (processed_dir / name).write_text(content, encoding="utf-8", newline="\n")
        outputs[f"data/processed/{result.run_id}/{name}"] = file_digest(content)
    (report_dir / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    outputs[f"{report_rel}/report.md"] = file_digest(markdown)
    (report_dir / "report.html").write_text(html, encoding="utf-8", newline="\n")
    outputs[f"{report_rel}/report.html"] = file_digest(html)
    for chart in ("targets.svg", "predictors.svg"):
        outputs[f"{report_rel}/charts/{chart}"] = file_digest((report_dir / "charts" / chart).read_bytes())
    result.outputs = outputs
    blob = json.dumps(manifest(result), indent=2, ensure_ascii=False) + "\n"
    assert_no_credentials(blob)
    (run_dir / "manifest.json").write_text(blob, encoding="utf-8", newline="\n")
