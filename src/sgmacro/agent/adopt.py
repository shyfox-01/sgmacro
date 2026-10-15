"""Adopting a verified candidate into the registry, after a person says yes.

The verifier's draft leaves the judgement fields as TODO. Adoption fills them
with explicit, conservative defaults and says so in the entry:

* publication lag: the observed lag (update stamp minus period end) plus five
  days, or a frequency default when no stamp exists;
* neutral band: 1% for percentage changes, 0.1 pp for rates;
* lead window: around the lag at which the data agreed with the declared
  effect, else 0-4 quarters.

The entry is inserted as text above the negative-control block, so the rest of
the hand-written file (comments included) is untouched, and the new file must
load and validate before it replaces the old one. A candidate that replaces a
broken series also retires the old entry: it becomes a quarantined entry that
is expected to keep failing, so the report says why it is gone.

The next ``sgmacro run`` sees an object with no stored snapshot and downloads it.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from sgmacro.agent.tools import result_identity
from sgmacro.periods import parse_period
from sgmacro.registry import Frequency, load_registry

DEFAULT_LAG = {"monthly": 35, "quarterly": 60, "annual": 120}
NEGATIVE_CONTROLS_MARKER = "  # Negative controls"


class AdoptError(RuntimeError):
    pass


def load_results(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "verification.json"
    if not path.exists():
        raise AdoptError(f"no verification.json in {session_dir}; run `sgmacro verify {session_dir}` first")
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def complete_entry(result: dict[str, Any], existing_ids: set[str], session_dir: Path, today: date,
                   reference_month: int | None = None) -> dict[str, Any]:
    draft = dict(result.get("draft") or {})
    if not draft:
        raise AdoptError(f"candidate {result.get('label')!r} has no draft (verdict {result.get('verdict')})")
    observed = result.get("observed") or {}
    freq = draft["frequency"]

    base_id = draft["id"]
    new_id, n = base_id, 2
    while new_id in existing_ids:
        new_id, n = f"{base_id}_{n}", n + 1
    draft["id"] = new_id
    draft["short_label"] = (draft.get("short_label") or draft["label"])[:48]

    observed_lag = observed.get("observed_lag_days")
    remeasured = False
    # Keyed on the frequency, not on the draft having the field: a draft written
    # before the verifier asked for the month has no such field at all.
    if freq == "annual":
        month = reference_month if reference_month is not None else draft.get("reference_month")
        if not isinstance(month, int):
            raise AdoptError("an annual series needs its reference month (e.g. 6 for end-June estimates); "
                             "pass --reference-month")
        if month != observed.get("reference_month"):
            # Verification measured the lag from another reference date; measure it
            # again from the same update stamp, so lag and gate use one date.
            stamp, latest = observed.get("update_stamp"), observed.get("latest")
            observed_lag = ((date.fromisoformat(stamp) - parse_period(latest, Frequency.ANNUAL)
                             .reference_date(month)).days if stamp and latest else None)
            remeasured = True
        draft["reference_month"] = month
    lag = int(observed_lag) + 5 if isinstance(observed_lag, int) and observed_lag >= 0 else DEFAULT_LAG[freq]
    draft["publication_lag_days"] = lag
    draft["neutral_band"] = 0.1 if draft["unit_kind"] == "rate" else 1.0
    agreeing = [r["best_lag_quarters"] for r in observed.get("relationship") or []
                if r.get("agreement") == "agrees" and r.get("best_lag_quarters") is not None]
    if agreeing:
        lo, hi = max(0, min(agreeing) - 1), max(agreeing) + 1
    else:
        lo, hi = 0, 4
    draft["expected_lead_quarters"] = [lo, hi]
    title = re.sub(r"^TODO \(publisher title: |\)$", "", draft.get("definition") or "")
    draft["definition"] = f"{draft['label']}, as published by {draft.get('publisher')} ({title})."
    draft["update_frequency"] = f"{Frequency(freq).value.capitalize()} (publisher schedule not yet recorded)"
    draft["limitations"] = (
        f"Adopted from discovery session {session_dir.name} on {today.isoformat()}. Publication lag "
        f"({lag} days), neutral band and lead window were filled automatically from the verifier's "
        "observations and should be reviewed."
        + (" The lead-lag check ran before the reference month was given, so it assumed end-December; "
           "re-run `sgmacro verify` to measure it from the stated month." if remeasured else ""))
    for key, value in list(draft.items()):
        if isinstance(value, str) and value.startswith("TODO"):
            raise AdoptError(f"field {key!r} still has no value")
    return draft


def _entry_text(entry: dict[str, Any], comment: str) -> str:
    dumped = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=100)
    body = "".join(f"  {line}\n" if line else "\n" for line in dumped.splitlines())
    return f"  # {comment}\n{body}\n"


def insert_entry(text: str, entry: dict[str, Any], comment: str) -> str:
    block = _entry_text(entry, comment)
    marker = text.find(NEGATIVE_CONTROLS_MARKER)
    if marker == -1:
        return text.rstrip("\n") + "\n\n" + block
    # Insert above the "# ====" rule that precedes the negative-control header.
    rule = text.rfind("  # ====", 0, marker)
    at = rule if rule != -1 else marker
    return text[:at] + block + text[at:]


def retire_entry(text: str, series_id: str, expected_status: str) -> str:
    start = text.find(f"  - id: {series_id}\n")
    if start == -1:
        raise AdoptError(f"no entry {series_id!r} in the registry text")
    nxt = text.find("\n  - id: ", start + 1)
    end = len(text) if nxt == -1 else nxt
    block = text[start:end]
    new_block, count = re.subn(r"\n    role: (predictor|target)\n",
                               f"\n    role: quarantine\n    expected_status: {expected_status}\n", block, count=1)
    if count != 1:
        raise AdoptError(f"could not find the role of {series_id!r}")
    return text[:start] + new_block + text[end:]


def _defaults(existing: Any, new: dict[str, Any]) -> dict[str, Any]:
    """Fill fields the draft omits with the existing ref's defaults, for comparison."""
    dumped = existing.model_dump()
    return {k: v for k, v in dumped.items() if k not in new}


def adopt(session_dir: Path, index: int, registry_path: Path, *, today: date,
          retired_status: str | None = None, reference_month: int | None = None) -> str:
    """Write candidate ``index`` (1-based) into the registry. Returns the new id."""
    results = load_results(session_dir)
    if not 1 <= index <= len(results):
        raise AdoptError(f"candidate {index} does not exist (session has {len(results)})")
    result = results[index - 1]
    if result.get("verdict") == "reject":
        raise AdoptError(f"candidate {index} ({result.get('label')}) was rejected by the verifier")
    registry = load_registry(registry_path)
    entry = complete_entry(result, {s.id for s in registry.series}, session_dir, today, reference_month)
    replaces = str((result.get("proposal") or {}).get("replaces") or "").strip()     # as the verifier read it
    for spec in registry.series:
        if spec.id != replaces and spec.source.model_dump() == entry["source"] | _defaults(spec.source, entry["source"]):
            raise AdoptError(f"{result.get('label')!r} is already in the registry as {spec.id!r}")

    text = registry_path.read_text(encoding="utf-8")
    new_text = insert_entry(text, entry, f"Adopted from discovery session {session_dir.name} "
                                        f"({today.isoformat()}); auto-filled fields noted under limitations.")
    if replaces and retired_status:
        new_text = retire_entry(new_text, replaces, retired_status)

    candidate = registry_path.with_suffix(".candidate.yaml")
    candidate.write_text(new_text, encoding="utf-8", newline="\n")
    try:
        load_registry(candidate)
    except Exception as exc:
        candidate.unlink()
        raise AdoptError(f"the adopted entry does not validate, registry unchanged: {exc}") from exc
    candidate.replace(registry_path)
    record = session_dir / "decisions.jsonl"
    with record.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"candidate": index, "label": result.get("label"), "decision": "adopted",
                             "series_id": entry["id"], "retired": replaces if replaces and retired_status else None,
                             "series": list(result_identity(result)), "on": today.isoformat(), "at": _now()}) + "\n")
    return entry["id"]


def record_decline(session_dir: Path, index: int, label: str | None, today: date,
                   series: tuple[str, ...] | None = None) -> None:
    """Record that a person declined candidate ``index``, with the series identity
    confirmed at the time (see :func:`sgmacro.agent.tools.result_identity`)."""
    row = {"candidate": index, "label": label, "decision": "declined", "on": today.isoformat(), "at": _now()}
    if series:
        row["series"] = list(series)
    with (session_dir / "decisions.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")


def _now() -> str:
    """Decision time, so a later re-verification of the session is ordered after it."""
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
