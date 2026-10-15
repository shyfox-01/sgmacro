"""Mechanical checks on LLM-written claims before they reach the report.

The model writes words, never figures: every figure a claim relies on is cited by
fact id, and the report prints those figures, with their series names, next to
the sentence. So a number cannot be mistyped, re-unitised or pinned on the wrong
series, by construction. What the text still asserts is checked:

citations    every cited id exists, and at least one is cited
no figures   the text contains no numbers (tenors inside names, "10-year",
             "3M", are allowed); a figure belongs in the citations
naming       a clause that names a series from the fact sheet must cite it, so
             "home prices rose" cannot lean on a bank-loan figure
direction    rise/fall wording agrees with the computed direction: for the whole
             claim when its cited moves share one direction, and clause by
             clause when a clause names exactly one cited series
stance       a claim citing signal facts must take exactly their reading, for the
             market it is filed under; a stance without a signal is rejected
implication  "supports / lifts" vs "weighs on / drags" wording must agree with
             the stance, so prose cannot call a rate rise supportive of prices
hedging      no forecasts ("will rise"), causal proofs or significance claims

Series are recognised only by names taken from the fact sheet (the label without
its parenthetical, plus any acronym in it: SORA, SGS, GDP, CPI, HDB). Where the
wording is ambiguous the checks abstain rather than guess, because a checker that
rejects correct prose gets switched off. What passes is consistent with the
computed figures and readings; it is not proven to be good economics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Any digit run that is not a tenor inside a name ("10Y", "3M", "4Q", "10-year", "3-month").
FIGURE = re.compile(r"(?<![A-Za-z\d.])\d[\d,.]*(?![\d,.]*(?:[YMQ]\b|-(?:year|month|quarter)s?\b))")

RISE = frozenset("""rose rise rises rising risen increase increased increases increasing higher up upward
gained gain gains climbed climbing grew grown growth growing accelerated accelerating expanded
expanding firmed firmer jumped surged strengthened picked""".split())
FALL = frozenset("""fell fall falls falling fallen decrease decreased decreases decreasing lower down
downward declined decline declines declining dropped drop eased easing softened softer slowed slowing
contracted contracting shrank shrinking cooled cooling moderated moderating weakened""".split())
CLAUSE = re.compile(r"[;,.!?]|\b(?:while|whereas|but|though|although|and)\b")

SUPPORTIVE_WORDS = ("supportive", "support", "supports", "supporting", "tailwind", "boost", "boosts",
                    "lift", "lifts", "upward pressure", "push up", "pushes up", "underpin", "underpins")
HEADWIND_WORDS = ("headwind", "headwinds", "weigh", "weighs", "weighing", "drag", "drags", "dampen",
                  "dampens", "downward pressure", "push down", "pushes down", "restrain", "restrains")
FORBIDDEN = ("will rise", "will fall", "will increase", "will decrease", "will push", "will lead",
             "forecast", "predict", "guarantee", "statistically significant", "proves", "proof",
             "causes ", "caused by")

STANCES = ("supportive", "headwind", "neutral", "ambiguous", "context")


@dataclass
class ClaimCheck:
    text: str
    market: str
    facts: list[str]
    stance: str
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_json(self) -> dict[str, Any]:
        return {"market": self.market, "text": self.text, "facts": self.facts, "stance": self.stance,
                "accepted": self.ok, "violations": self.violations}


def figures(text: str) -> list[str]:
    return FIGURE.findall(text)


def series_names(label: str) -> list[str]:
    """How a sentence may name a series: its label without the parenthetical, and
    any acronym in the label."""
    base = re.sub(r"\s*\(.*?\)", "", label).strip().lower()
    return [base, *{w.lower() for w in re.findall(r"\b[A-Z]{3,}\b", label)}]


def name_index(sheet: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Name -> series, keeping only names that identify one series ("CPI" and "URA"
    each belong to two series, so they name neither)."""
    owners: dict[str, set[str]] = {}
    for series in {f["series"] for f in sheet.values()}:
        for name in series_names(series):
            owners.setdefault(name, set()).add(series)
    return {name: next(iter(s)) for name, s in owners.items() if len(s) == 1}


def _named(clause: str, names: dict[str, str]) -> set[str]:
    return {series for name, series in names.items() if re.search(rf"\b{re.escape(name)}\b", clause)}


def _polarity(text: str) -> int | None:
    words = set(re.findall(r"[a-z]+", text.lower()))
    rising, falling = bool(words & RISE), bool(words & FALL)
    if rising == falling:
        return None
    return 1 if rising else -1


def _implication(text: str) -> str | None:
    lowered = f" {text.lower()} "
    up = any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in SUPPORTIVE_WORDS)
    down = any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in HEADWIND_WORDS)
    if up == down:
        return None
    return "supportive" if up else "headwind"


def check_claim(claim: dict[str, Any], market: str, sheet: dict[str, dict[str, Any]]) -> ClaimCheck:
    text = claim["text"].strip()
    cited = list(claim["facts"])
    stance = claim["stance"].strip().lower()
    result = ClaimCheck(text, market, cited, stance)

    if not text:
        result.violations.append("empty claim")
        return result
    if len(text) > 420:
        result.violations.append("claim longer than two sentences' worth of text")
    if stance not in STANCES:
        result.violations.append(f"unknown stance {stance!r}")
    unknown = [c for c in cited if c not in sheet]
    if unknown:
        result.violations.append(f"cites unknown fact ids {unknown}")
    if not cited:
        result.violations.append("cites no facts")
    facts = [sheet[c] for c in cited if c in sheet]
    # A cited signal stands for the move it was read from, so wording about that
    # predictor is checked against the move's direction even if only the signal is cited.
    implied_moves = [sheet[f["move"]] for f in facts if f.get("move") in sheet and sheet[f["move"]] not in facts]
    evidence = facts + implied_moves

    # figures ---------------------------------------------------------------
    found = figures(text)
    if found:
        result.violations.append(f"the text contains figures {sorted(set(found))}; cite the fact instead, "
                                 "and the report prints its value")

    # naming and clause-level direction ---------------------------------------
    cited_series = {f["series"] for f in facts}
    moves: dict[str, set[str]] = {}
    for f in evidence:
        if f.get("direction") in ("up", "down"):
            moves.setdefault(f["series"], set()).add(f["direction"])
    names = name_index(sheet)
    for clause in CLAUSE.split(text.lower()):
        named = _named(clause, names)
        for series in sorted(named - cited_series):
            result.violations.append(f"names {series!r} but cites none of its figures")
        named_cited = named & cited_series
        polarity = _polarity(clause)
        if len(named_cited) == 1 and polarity is not None:
            series = next(iter(named_cited))
            if moves.get(series) and len(moves[series]) == 1:
                expected = 1 if moves[series] == {"up"} else -1
                if polarity != expected:
                    result.violations.append(
                        f"says {series} {'rose' if polarity > 0 else 'fell'} but its cited change is "
                        f"{'up' if expected > 0 else 'down'}")

    # direction over the whole claim -----------------------------------------
    directions = {f.get("direction") for f in evidence if f.get("direction") in ("up", "down")}
    polarity = _polarity(text)
    if len(directions) == 1 and polarity is not None:
        expected = 1 if directions == {"up"} else -1
        if polarity != expected:
            result.violations.append(
                f"wording says {'rise' if polarity > 0 else 'fall'} but the cited change is "
                f"{'up' if expected > 0 else 'down'}")

    # stance vs signals ---------------------------------------------------
    # Signals cited directly, plus the signal of every predictor whose move is
    # cited: a sentence about a predictor is held to that predictor's reading.
    signals = [f for f in facts if f.get("kind") == "signal"]
    for fact in facts:
        implied = fact.get("signals")
        if implied is None:
            continue
        if market not in implied:
            result.violations.append(f"cites {fact.get('series')!r}, which has no {market} signal")
        elif implied[market] in sheet and sheet[implied[market]] not in signals:
            signals.append(sheet[implied[market]])
    if signals:
        readings = {s["reading"] for s in signals}
        markets = {s["market"] for s in signals}
        if markets != {market}:
            result.violations.append(f"cites signals for {sorted(markets)} under the {market} summary")
        if len(readings) > 1:
            result.violations.append(f"one claim mixes signal readings {sorted(readings)}")
        elif stance != next(iter(readings)):
            result.violations.append(f"stance {stance!r} contradicts the computed reading "
                                     f"{next(iter(readings))!r}")
    elif stance in ("supportive", "headwind", "neutral", "ambiguous"):
        result.violations.append(f"stance {stance!r} asserted without citing a signal")

    implied = _implication(text)
    if implied is not None and implied != stance:
        result.violations.append(f"wording implies {implied!r} but the claim's stance is {stance!r}")

    lowered = text.lower()
    hits = [w.strip() for w in FORBIDDEN if w in lowered]
    if hits:
        result.violations.append(f"forecast or causal language {hits}")
    return result


def check_summary(claims: dict[str, list[dict[str, Any]]],
                  sheet: dict[str, dict[str, Any]]) -> dict[str, list[ClaimCheck]]:
    """Check claims already flattened (and shape-checked) by ``summary.to_claims``."""
    return {market: [check_claim(c, market, sheet) for c in claims.get(market, [])]
            for market in ("sales", "rents")}
