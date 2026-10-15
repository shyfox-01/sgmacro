"""The executive summary: LLM-drafted, mechanically checked, deterministic fallback.

What the model sees is a *fact sheet*, not data: short aliases (F1, F2, ... and
S1, S2, ...) for figures and signal readings that code has already computed and
formatted. It writes structured claims citing those aliases; :mod:`claims` checks
each one; rejected claims are dropped and recorded; a market with fewer than two
surviving claims falls back to a template written from the same facts.

So the LLM contributes ordering, emphasis and plain-English connective tissue.
It never contributes a number, a direction or an economic sign: those are copied
from, or checked against, computed values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from sgmacro.credentials import redact
from sgmacro.facts import Fact
from sgmacro.llm.backends import BackendUnavailable, ResponseCache, request_key, resolve_backend
from sgmacro.llm.claims import ClaimCheck, check_summary
from sgmacro.periods import parse_period
from sgmacro.registry import Market, Role

MIN_CLAIMS = 2

SYSTEM = """You draft the executive summary of a data report on Singapore residential property for a quantitative research team.

You receive a fact sheet. Facts F1, F2, ... are figures computed by code. Signals S1, S2, ... state, for one predictor and one market, whether its latest move is "supportive" (tends to push that market up), a "headwind" (tends to push it down), "neutral" (barely moved) or "ambiguous" (theory gives no clear sign), under the economic mechanism shown.

For each market write:
- `lead`: one or two sentences on that market's target indices, citing their F ids.
- `points`: one entry per reading group listed in that market's `reading_groups_present`, in that order; omit groups that are not listed. Each point names the predictors in its group, cites their S ids and the F id of each predictor's move, and says in plain words how they bear on the market.

Rules, all mandatory:
1. Use only the fact sheet. List in `facts` every id a sentence relies on.
2. Write no numbers at all: no figures, percentages, basis points, index levels, counts or dates. The report prints every cited fact's value, with its series name, next to your sentence, so cite the F id instead of quoting it. Tenors inside names ("10-year", "3-month") are fine.
3. Describe movement exactly as the cited fact's `direction` says: "up" rose, "down" fell, "flat" barely changed. When you name a series, cite its F id.
4. A point cites only signals of its own market whose reading equals the point's `reading`. Never mention a predictor from another group in that point, and never call a neutral or ambiguous signal support or a headwind.
5. Signals apply declared hypotheses to the latest data. They are not forecasts: do not predict future prices or rents, and do not claim causality or statistical significance.
6. Style: one or two plain sentences per entry, no markdown, and never write ids in the text (they go in `facts` only). Explain the mechanism briefly in words."""

_SENTENCE = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["text", "facts"],
    "additionalProperties": False,
}
_POINT = {
    "type": "object",
    "properties": {
        "reading": {"type": "string", "enum": ["supportive", "headwind", "neutral", "ambiguous"]},
        "text": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reading", "text", "facts"],
    "additionalProperties": False,
}
_MARKET = {
    "type": "object",
    "properties": {"lead": _SENTENCE, "points": {"type": "array", "items": _POINT}},
    "required": ["lead", "points"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {"sales": _MARKET, "rents": _MARKET},
    "required": ["sales", "rents"],
    "additionalProperties": False,
}


def to_claims(payload: Any) -> dict[str, list[dict[str, Any]]]:
    """Flatten the structured output into claims with an explicit stance.

    The payload comes from a model or from the cache, so its shape is checked here,
    once; anything that does not match the schema raises ValueError and the caller
    uses the template. Downstream code can then rely on the types.
    """
    if not isinstance(payload, dict):
        raise ValueError("the response is not a JSON object")
    out: dict[str, list[dict[str, Any]]] = {}
    for market in ("sales", "rents"):
        block = payload.get(market)
        if not isinstance(block, dict) or not isinstance(block.get("points"), list):
            raise ValueError(f"{market!r} lacks a lead and a list of points")
        claims = [_claim(block.get("lead"), "context", f"{market}.lead")]
        for i, point in enumerate(block["points"]):
            reading = point.get("reading") if isinstance(point, dict) else None
            claims.append(_claim(point, reading, f"{market}.points[{i}]"))
        out[market] = claims
    return out


def _claim(entry: Any, stance: Any, where: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{where} is not an object")
    text, facts = entry.get("text"), entry.get("facts")
    if not isinstance(text, str) or not isinstance(stance, str):
        raise ValueError(f"{where} lacks a text or a reading")
    if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
        raise ValueError(f"{where}.facts is not a list of ids")
    return {"text": text, "facts": facts, "stance": stance}


@dataclass
class MarketSummary:
    market: str
    source: str
    claims: list[ClaimCheck] = field(default_factory=list)
    fallback_text: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.ok]

    @property
    def rejected(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.ok]


@dataclass
class Summary:
    source: str
    """Who wrote the text: ``backend:model``, ``mixed`` or ``deterministic``. Where
    this run read it from (cache or a live call) is execution detail, kept out of
    the report so a cold and a warm run produce the same bytes."""
    model: str | None
    prompt_sha256: str
    markets: dict[str, MarketSummary]
    aliases: dict[str, str]
    notes: list[str] = field(default_factory=list)
    """About the content (an unusable response, a market on the template). These
    follow from the stored response, so a cold run and its replay agree."""
    response_sha256: str | None = None
    from_cache: bool = False
    execution_notes: list[str] = field(default_factory=list)
    """How this run obtained (or failed to obtain) a response: failed calls, an
    ignored cache entry. Execution log only, never the report."""
    skipped_backends: list[str] = field(default_factory=list)
    """Why ``auto`` passed over the backends before the one it used (or all of
    them). Execution log only; shown on screen only when no model drafted."""

    @property
    def identity(self) -> str:
        """A hash of the summary exactly as the report and manifest record it (source,
        model, notes, every claim and its verdict), for the run identity. Two template
        summaries with different notes (--llm off vs an unusable response) differ;
        a cold run and its cached replay, which differ only in execution notes, do not."""
        return sha256(json.dumps(self.to_json(), sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def resolve(self, alias: str) -> str:
        return self.aliases.get(alias, alias)

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "notes": self.notes,
            "markets": {
                m: {"source": s.source,
                    "accepted": len(s.accepted), "rejected": len(s.rejected),
                    "claims": [{**c.to_json(), "facts": [self.resolve(f) for f in c.facts]} for c in s.claims]}
                for m, s in self.markets.items()
            },
        }


# ---------------------------------------------------------------------------
# fact sheet
# ---------------------------------------------------------------------------

def _period_display(fact: Fact, freq_of: dict[str, Any]) -> str:
    return parse_period(fact.period, freq_of[fact.series_id]).display if fact.period else ""


@dataclass
class FactSheetView:
    """The fact sheet in two shapes: nested by market for the prompt, flat by
    alias for the checker, plus alias -> real fact id for rendering."""

    prompt_view: dict[str, Any]
    flat: dict[str, dict[str, Any]]
    aliases: dict[str, str]


def build_sheet(result: Any) -> FactSheetView:
    registry = result.registry
    freq_of = {s.id: s.frequency for s in registry.series}
    flat: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    by_fact: dict[str, str] = {}
    counter = {"F": 0, "S": 0}

    def alias_for(prefix: str, fact: Fact, entry: dict[str, Any]) -> str:
        if fact.fact_id in by_fact:
            return by_fact[fact.fact_id]
        counter[prefix] += 1
        alias = f"{prefix}{counter[prefix]}"
        aliases[alias], by_fact[fact.fact_id], flat[alias] = fact.fact_id, alias, entry
        return alias

    def figure(spec: Any, fact: Fact, metric: str) -> dict[str, Any]:
        entry = {"kind": fact.kind, "series": spec.short_label, "metric": metric,
                 "period": _period_display(fact, freq_of), "value": fact.display}
        if fact.direction:
            entry["direction"] = fact.direction
        return {"id": alias_for("F", fact, entry), **entry}

    view: dict[str, Any] = {"as_of": result.config.as_of.isoformat()}
    for market in Market:
        targets, signals = [], []
        for outcome in result.reportable(Role.TARGET):
            if outcome.spec.market is not market:
                continue
            spec, ids = outcome.spec, outcome.facts
            item = {"index": spec.short_label,
                    "level": figure(spec, result.facts.get(ids["level"]), f"index level ({spec.unit})"),
                    "change": figure(spec, result.facts.get(ids["headline"]), "year-on-year change")}
            if "real_yoy" in ids:
                item["real_change"] = figure(spec, result.facts.get(ids["real_yoy"]),
                                             "real (CPI-deflated) year-on-year change")
            targets.append(item)
        for outcome in result.reportable(Role.PREDICTOR):
            spec, ids = outcome.spec, outcome.facts
            for signal in outcome.signals:
                if signal.market is not market:
                    continue
                head = result.facts.get(ids["headline"])
                fact = result.facts.get(ids[f"signal_{market.value}"])
                entry = {"kind": "signal", "series": spec.short_label, "market": market.value,
                         "reading": signal.reading}
                s_alias = alias_for("S", fact, entry)
                move = figure(spec, head, head.label.split(", ", 1)[-1])
                # A sentence that quotes the move is a sentence about this predictor,
                # so the checker can hold it to the predictor's reading even when the
                # model cites the figure rather than the signal.
                flat[move["id"]].setdefault("signals", {})[market.value] = s_alias
                # And the other way: a sentence citing only the signal is still held to
                # the direction of the move the signal was read from. (Checker's view
                # only; the prompt already shows each signal's move.)
                flat[s_alias]["move"] = move["id"]
                signals.append({
                    "id": s_alias,
                    "predictor": spec.short_label,
                    "reading": signal.reading,
                    "move": move,
                    "mechanism": f"{spec.channel}; declared effect on {market.value}: {signal.effect.value}",
                    "typical_lead_quarters": f"{spec.expected_lead_quarters[0]}-{spec.expected_lead_quarters[1]}",
                })
        readings = [r for r in ("supportive", "headwind", "neutral", "ambiguous")
                    if any(s["reading"] == r for s in signals)]
        view[market.value] = {"targets": targets, "reading_groups_present": readings, "signals": signals}
    return FactSheetView(view, flat, aliases)


def build_prompt(sheet: FactSheetView) -> str:
    return ("Fact sheet (JSON). Each market lists its target indices and its predictor signals; "
            "a signal's `move` is the figure the reading is based on.\n"
            f"{json.dumps(sheet.prompt_view, indent=1, ensure_ascii=False)}\n\n"
            "Write the executive summary as specified. Every claim about a predictor must cite "
            "that predictor's signal id (S...) for the market and, if it quotes the move, the "
            "move's id (F...).")


# ---------------------------------------------------------------------------
# deterministic fallback
# ---------------------------------------------------------------------------

def deterministic(result: Any, market: Market) -> list[str]:
    """Template summary for one market, written only from facts."""
    lines: list[str] = []
    for target in result.reportable(Role.TARGET):
        if target.spec.market is not market:
            continue
        f = target.facts
        head = result.facts.get(f["headline"])
        real = result.facts.get(f["real_yoy"]).display if "real_yoy" in f else "n/a"
        lines.append(f"{target.spec.short_label}: {head.display} year on year in {head.period} "
                     f"(real, CPI-deflated: {real}).")
    groups: dict[str, list[str]] = {"supportive": [], "headwind": [], "neutral": [], "ambiguous": []}
    for outcome in result.reportable(Role.PREDICTOR):
        for signal in outcome.signals:
            if signal.market is market and signal.reading in groups:
                head = result.facts.get(outcome.facts["headline"])
                groups[signal.reading].append(f"{outcome.spec.short_label} ({head.display})")
    names = {"supportive": "Supportive", "headwind": "Headwinds", "neutral": "Little changed",
             "ambiguous": "Ambiguous by design"}
    for reading, items in groups.items():
        if items:
            lines.append(f"{names[reading]}: {', '.join(items)}.")
    return lines


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def summarise(result: Any, *, backend_preference: str = "auto", cache_dir: Path = Path("llm_cache"),
              log: Callable[[str], None] = print, backend: Any = None, offline: bool = False) -> Summary:
    """Draft, check and assemble the summary. ``backend`` bypasses resolution
    (used by the model-comparison script); responses are cached either way. Any
    failure to get a usable response, including an unavailable model, falls back
    to the template and is recorded in ``notes``."""
    view = build_sheet(result)
    sheet, aliases = view.flat, view.aliases
    prompt = build_prompt(view)
    key = request_key(SYSTEM, prompt, SCHEMA)
    cache = ResponseCache(cache_dir)
    notes: list[str] = []
    execution: list[str] = []
    skipped: list[str] = []
    raw: str | None = None
    source, model, from_cache = "deterministic", None, False

    if backend_preference != "off":
        try:
            cached = cache.get(key)
        except ValueError as exc:
            cached = None
            execution.append(redact(f"cached response ignored: {exc}"))
        if cached is not None:
            raw, model = cached["response"], cached.get("model")
            source, from_cache = f"{cached.get('backend')}:{model}", True
        else:
            if backend is None:
                try:
                    backend, skipped = resolve_backend(backend_preference, offline=offline)
                except BackendUnavailable as exc:
                    execution.append(f"--llm {backend_preference} unavailable: {redact(str(exc))}")
                    log(f"warning: --llm {backend_preference} unavailable ({redact(str(exc))}); "
                        "using the template summary")
            if backend is not None:
                log(f"drafting summary with {backend.name}:{backend.model}")
                try:
                    raw = backend.complete_json(SYSTEM, prompt, SCHEMA)
                    if not raw.strip():
                        # Nothing to check or replay: a failure, so it is not cached.
                        raise ValueError("the model returned an empty response")
                    model = backend.model
                    source = f"{backend.name}:{model}"
                    cache.put(key, backend=backend.name, model=model, system=SYSTEM, prompt=prompt, response=raw)
                except Exception as exc:  # noqa: BLE001 - any backend failure falls back
                    raw, model, source = None, None, "deterministic"
                    execution.append(redact(f"{backend.name} failed: {type(exc).__name__}: {exc}"))
            elif backend_preference == "cache":
                execution.append("no cached response for this fact sheet")

    checks: dict[str, list[ClaimCheck]] = {"sales": [], "rents": []}
    if raw is not None:
        try:
            checks = check_summary(to_claims(json.loads(raw)), sheet)
        except json.JSONDecodeError as exc:
            notes.append(f"model output was not valid JSON: {exc}")
            raw = None
        except ValueError as exc:
            notes.append(f"model output does not match the schema: {exc}")
            raw = None

    markets: dict[str, MarketSummary] = {}
    for market in Market:
        claims = checks.get(market.value, [])
        accepted = [c for c in claims if c.ok]
        if len(accepted) >= MIN_CLAIMS:
            markets[market.value] = MarketSummary(market.value, source, claims)
        else:
            if claims:
                notes.append(f"{market.value}: only {len(accepted)} of {len(claims)} claims passed; "
                             "using the deterministic summary")
            markets[market.value] = MarketSummary(market.value, "deterministic", claims,
                                                  fallback_text=deterministic(result, market))
    overall = source if all(m.source == source for m in markets.values()) else (
        "mixed" if any(m.source != "deterministic" for m in markets.values()) else "deterministic")
    response_sha = sha256(raw.encode("utf-8")).hexdigest() if raw is not None and overall != "deterministic" else None
    return Summary(overall, model, key, markets, aliases, notes, response_sha, from_cache, execution, skipped)
