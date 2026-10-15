"""Interactive HTML rendering of the report: one self-contained file.

Same content and the same facts as the Markdown report, laid out for reading:
headline cards, a signal heatmap, hoverable charts, collapsible detail, and a
lineage panel that opens on any figure (its formula, inputs, raw-file locator and
content hash). No external scripts or fonts, so it opens offline, and no
wall-clock time is embedded, so repeated runs produce the same bytes.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from markupsafe import Markup, escape

from sgmacro.periods import Period
from sgmacro.registry import Role, Transform
from sgmacro.transform import change_history

START_YEAR = 2005
SUFFIX = {"pct": "%", "pp": " pp", "bp": " bp", "count": " rounds"}


def _decimal_year(period: Period) -> float:
    return round(period.year + (period.index - 0.5) / period.freq.periods_per_year, 4)


def _points(spec, history) -> list[list[Any]]:
    out = []
    for ordinal, value in history.items():
        period = Period.from_ordinal(spec.frequency, int(ordinal))
        if period.year < START_YEAR:
            continue
        v = None if value != value or value is None else round(float(value), 4)
        out.append([_decimal_year(period), v, period.display])
    return out


def chart_data(result) -> dict[str, Any]:
    targets = []
    for slot, outcome in enumerate(result.reportable(Role.TARGET), start=1):
        spec = outcome.spec
        history = change_history(spec, Transform.YOY, outcome.result.gated_series())
        targets.append({"id": spec.id, "name": spec.short_label, "slot": slot, "points": _points(spec, history)})
    predictors = {}
    for outcome in result.reportable(Role.PREDICTOR):
        spec, changes = outcome.spec, outcome.changes
        if changes is None:
            continue
        predictors[spec.id] = {
            "name": spec.short_label,
            "unit": SUFFIX[spec.headline.change_unit],
            "band": spec.neutral_band or 0,
            "points": _points(spec, changes.headline_history),
        }
    return {"targets": targets, "predictors": predictors}


def facts_data(result) -> dict[str, Any]:
    out = {}
    for fact in result.facts:
        if fact.kind == "correlation":
            out[fact.fact_id] = {"label": fact.label, "display": fact.display, "formula": fact.formula,
                                 "note": f"{fact.extra.get('pairs')} overlapping quarters; computed from "
                                         "data/processed/<run>/observations.csv", "inputs": [], "depends_on": []}
            continue
        out[fact.fact_id] = {
            "label": fact.label, "period": fact.period, "display": fact.display, "value": fact.value,
            "formula": fact.formula, "note": fact.note, "depends_on": fact.depends_on,
            "inputs": [{"series": i.series_id, "period": i.period, "value": i.value, "locator": i.locator,
                        "file": f"data/snapshots/{i.object_key}/{i.vintage_id}/data."
                                f"{ {'singstat': 'json', 'curated': 'yaml'}.get(i.object_key.split('/')[0], 'csv') }",
                        "sha256": i.data_sha256} for i in fact.inputs],
        }
    return out


def _corr_class(r: float | None) -> str:
    if r is None or math.isnan(r):
        return "corr-na"
    level = 0 if abs(r) < 0.2 else 1 if abs(r) < 0.4 else 2 if abs(r) < 0.6 else 3
    if level == 0:
        return "corr-0"
    return f"corr-{'pos' if r > 0 else 'neg'}{level}"


def diagnostics_matrix(result) -> dict[str, Any]:
    """The shared diagnostics view, with a colour class per cell for the heatmap."""
    from sgmacro.report.render import diagnostics_view

    view = diagnostics_view(result)
    for row in view["rows"]:
        for cell in row["cells"]:
            cell["cls"] = _corr_class(cell["r"])
            cell["short"] = "n/a" if cell["r"] is None else f"{cell['r']:+.2f}"
    return view


def _finite(obj: Any) -> Any:
    """JSON.parse rejects NaN and Infinity; send them as null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def inline_markup(text: str) -> Markup:
    """The report's two Markdown emphasis forms, **bold** and *italic*, as HTML."""
    html = str(escape(text))
    html = re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", html)
    html = re.sub(r"[*](.+?)[*]", r"<i>\1</i>", html)
    return Markup(html)


def render_html(result, context: dict[str, Any]) -> str:
    env = Environment(loader=PackageLoader("sgmacro", "templates"), undefined=StrictUndefined,
                      autoescape=True, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.filters["md_bold"] = inline_markup
    facts = facts_data(result)
    payload = _finite({"facts": facts, "charts": chart_data(result), "as_of": context["as_of"],
                       "run_id": context["run_id"]})
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # The JSON sits inside <script type="application/json">: neutralise "</" so a
    # string can never close the tag.
    blob = blob.replace("</", "<\\/")
    html = env.get_template("report.html.j2").render(
        **context, data_json=Markup(blob), diag=diagnostics_matrix(result))
    return "\n".join(line.rstrip() for line in html.splitlines()).strip() + "\n"
