"""The source-discovery agent: an LLM with read-only tools, under a step budget.

Why an agent here and nowhere else in the workflow: finding candidate predictors
is open-ended search over ~4,600 data.gov.sg datasets and hundreds of SingStat
tables with noisy titles ("population" matches motor-vehicle population). That is
judgement work a model does well and a keyword list does badly. Everything after
it, the numbers, is deterministic code.

Trust boundary: the agent can only *propose*. Proposals go to the deterministic
verifier (:mod:`sgmacro.agent.verify`) and then to a person, who is asked about
each candidate; only on "yes" does :mod:`sgmacro.agent.adopt` validate the entry
and write it into ``config/sources.yaml``. The report run never calls the agent,
so an LLM's non-determinism cannot change a report.

Every session is recorded in full (messages, tool calls, tool results) under
``discovery/<session>/`` so a reviewer can see exactly what the agent looked at
before proposing anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sgmacro.agent.tools import TOOL_SPECS, Toolbox, describe_candidate, result_identity
from sgmacro.credentials import redact, redact_keys
from sgmacro.llm.backends import Backend, ToolResult, Turn

SYSTEM = """You are a data-sourcing agent for a quantitative research team. You find publicly available Singapore statistical series that could help predict changes in Singapore residential sale prices and rents.

Work method:
1. Call list_registry first to see what is already covered.
2. Search with short keywords of one or two words ("cement", "work permit"), with search_singstat and search_datagovsg. Titles are noisy; inspect before trusting.
3. For every candidate, call describe_* and preview_* to confirm it exists, is still updated (its latest period should be recent), has a clear unit, and has many years of history. Prefer seasonally adjusted data and the original publishing agency.
4. As soon as a candidate passes step 3, propose it with propose_series; do not save proposals for the end. State the economic channel, the markets it bears on, the expected sign of its effect on each market and the mechanism. Avoid duplicates of registry series unless you propose a replacement (set `replaces`).
5. If two or three searches on a theme find nothing suitable, drop that theme and move on. Proposing fewer series than the brief asks for, or none, is an acceptable result.
6. Never repeat a call you have already made: its result will not change.
7. When you have proposed what you can (at most the number the brief asks for), call finish with a short summary of what you proposed and what you could not find.

Never invent table ids, series numbers or dataset ids; copy them from tool results."""

#: Consecutive steps made only of repeated calls before the session is stopped.
MAX_REPEAT_STEPS = 3
#: How much of an earlier result a repeated call gets back while the model still
#: has the whole result in its recent history.
REPEAT_RESULT_CHARS = 1500
#: Times an identical call that failed is run again; a transient upstream error
#: may clear, a bad argument will not.
MAX_ATTEMPTS = 2
#: Share of the step budget after which every step reminds the model to wrap up.
WRAP_UP_AT = 2 / 3
#: Shares of a backend's known context window (``context_tokens``) at which the
#: model is reminded to wrap up, and at which the session ends before the next
#: request, leaving room for the reply instead of having it cut off.
CONTEXT_WRAP_UP, CONTEXT_STOP = 0.6, 0.75
#: Characters per token for what the server has not counted yet; errs towards
#: counting high (JSON and numbers take more tokens than prose).
CHARS_PER_TOKEN = 3.0


@dataclass
class Session:
    brief: str
    backend: str
    model: str
    started_at: str
    events: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    finished: str | None = None
    stop_reason: str = ""
    model_failed: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"brief": self.brief, "backend": self.backend, "model": self.model,
                "started_at": self.started_at,
                "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "stop_reason": self.stop_reason, "model_failed": self.model_failed, "agent_summary": self.finished,
                "proposals": self.proposals, "events": self.events}


def run_agent(brief: str, backend: Backend, toolbox: Toolbox, *, max_steps: int = 40,
              log: Callable[[str], None] = print) -> Session:
    session = Session(brief, backend.name, backend.model,
                      datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    chat = backend.session(SYSTEM, TOOL_SPECS)
    turn = _call_model(session, log, lambda: chat.send_user(brief))
    # Small local models can fall into re-issuing the same calls until the context
    # fills. A call is identified by the arguments its tool actually uses (extra
    # fields it ignores do not make it new). A repeat is answered from the record,
    # not run again; a failure is retried once; a result the model was later sent
    # shortened (see the Ollama session) is given back whole, as a recovery and not
    # a repeat; and a run of steps made only of repeats ends the session.
    made: dict[tuple[str, str], dict[str, Any]] = {}
    sent = 0                                       # tool results sent so far
    keep_recent = getattr(chat, "KEEP_RECENT", None)
    short_chars = getattr(chat, "SHORT_CHARS", 0)
    repeat_steps = 0
    # A local model's context is fixed (Ollama's num_ctx); a long session would
    # otherwise end with the reply cut off. Cloud backends report no limit.
    window = getattr(backend, "context_tokens", None)
    # Characters not yet counted by the server; the whole conversation until a
    # backend reports real token counts.
    pending = len(SYSTEM) + len(json.dumps([t.__dict__ for t in TOOL_SPECS])) + len(brief)
    counted = 0
    for step in range(1, max_steps + 1):
        if turn is None:
            break
        session.events.append({"step": step, "type": "assistant", "text": turn.text,
                               "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
                               "stop_reason": turn.stop_reason, "context_used": turn.context_used})
        if turn.context_used:
            counted, pending = turn.context_used, 0
        else:
            pending += len(turn.text) + len(json.dumps([c.arguments for c in turn.tool_calls]))
        if turn.text.strip():
            log(f"[{step}] agent: {redact(turn.text.strip())[:160]}")
        if not turn.tool_calls:
            session.stop_reason = "no tool call (agent stopped)"
            break
        results = []
        fresh = 0
        for call in turn.tool_calls:
            effective = getattr(toolbox, "effective_arguments", None)
            used = effective(call.name, call.arguments)[0] if effective else call.arguments
            key = (call.name, json.dumps(used, sort_keys=True, ensure_ascii=False))
            earlier = made.get(key)
            if earlier is None or (earlier["is_error"] and earlier["attempts"] < MAX_ATTEMPTS):
                text, is_error = toolbox.call(call.name, call.arguments)
                made[key] = {"step": step, "text": text, "is_error": is_error, "sent_at": sent,
                             "attempts": (earlier or {}).get("attempts", 0) + 1}
                fresh += 1
            elif keep_recent and earlier["sent_at"] < sent - keep_recent and len(earlier["text"]) > short_chars:
                text, is_error = (f"[earlier result from step {earlier['step']}, shown again in full; not fetched "
                                  f"again]\n{earlier['text']}"), earlier["is_error"]
                earlier["sent_at"] = sent                  # recent again: a second re-read is a repeat
                fresh += 1
            else:
                # A small model often re-reads a series just before proposing it; a bare
                # refusal stalled it, so the result comes back with a nudge.
                text, is_error = (f"[repeated call: same result as step {earlier['step']}, not fetched again]\n"
                                  f"{earlier['text'][:REPEAT_RESULT_CHARS]}\n[If this is suitable, propose it now "
                                  "with propose_series; otherwise try something else or call finish.]"),\
                    earlier["is_error"]
            sent += 1
            log(f"[{step}] {redact(call.name)}({redact(json.dumps(call.arguments, ensure_ascii=False))[:120]}) "
                f"-> {'ERROR ' if is_error else ''}{redact(text).splitlines()[0][:100] if text else ''}")
            session.events.append({"step": step, "type": "tool_result", "name": call.name,
                                   "arguments": call.arguments, "is_error": is_error, "result": text})
            results.append(ToolResult(call, text, is_error))
            pending += len(text)
        if toolbox.finished is not None:
            session.stop_reason = "finish tool called"
            break
        repeat_steps = 0 if fresh else repeat_steps + 1
        if repeat_steps >= MAX_REPEAT_STEPS:
            session.stop_reason = f"stopped: the model only repeated earlier calls for {repeat_steps} steps"
            log(session.stop_reason)
            break
        if step == max_steps:
            # Stop before asking the model again: every model call made is recorded,
            # and the budget counts model calls.
            session.stop_reason = f"step budget of {max_steps} exhausted"
            break
        estimate = counted + pending / CHARS_PER_TOKEN
        if window and estimate >= window * CONTEXT_STOP:
            session.stop_reason = (f"stopped: the conversation is near the model's context window "
                                   f"(about {estimate:,.0f} of {window:,} tokens)")
            log(session.stop_reason)
            break
        near_limit = bool(window) and estimate >= window * CONTEXT_WRAP_UP
        if step >= max_steps * WRAP_UP_AT or near_limit:
            left = "the conversation is close to the model's context limit" if near_limit else \
                f"{max_steps - step} model call(s) left"
            note = (f"\n\n[workflow note: {left}. Propose the candidates you have "
                    "previewed now, then call finish.]")
            last = results[-1]
            results[-1] = ToolResult(last.call, last.content + note, last.is_error)
            session.events.append({"step": step, "type": "workflow_note", "text": note.strip()})
        turn = _call_model(session, log, lambda: chat.send_tool_results(results))
    session.proposals = list(toolbox.proposals)
    session.finished = toolbox.finished
    return session


def _call_model(session: Session, log: Callable[[str], None], call: Callable[[], Turn]) -> Turn | None:
    """One model request. A failure (server down, timeout, a model without tool
    support, a refused request) ends the session instead of the program, so the
    steps already taken are still saved and their proposals still verified."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - any backend failure ends the session the same way
        session.model_failed = True
        session.stop_reason = "model call failed: " + redact(f"{type(exc).__name__}: {exc}")[:300]
        log(session.stop_reason)
        return None


def save_session(session: Session, root: Path, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / f"{stamp}_{label}"
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(session.to_json(), indent=2, ensure_ascii=False) + "\n"
    (directory / "session.json").write_text(redact_keys(text), encoding="utf-8", newline="\n")
    return directory


#: At most this many turned-down candidates are listed in a brief.
MAX_TURNED_DOWN = 30
#: Verifier failures about the series itself (stale, unreadable, missing, empty),
#: which a rerun would meet again. A failure to fetch it, or a wrong frequency or
#: filter in the proposal, says nothing lasting about the series.
LASTING_FAILURES = frozenset({"freshness", "layout", "data", "exists", "parse"})


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """The complete object lines of an append-only log, and how many lines were
    skipped. Each line is decoded on its own, so a half-written last line (cut
    inside a multi-byte character, too) or a non-object is skipped, not the log."""
    try:
        data = path.read_bytes()
    except OSError:
        return [], 0
    rows, skipped = [], 0
    for raw in data.splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except ValueError:                         # includes UnicodeDecodeError
            row = None
        if isinstance(row, dict):
            rows.append(row)
        else:
            skipped += 1
    return rows, skipped


def _identity(value: Any) -> tuple[str, ...] | None:
    """A series identity stored with a decision, if it has the shape
    :func:`sgmacro.agent.tools.candidate_key` gives one: three parts for SingStat,
    four for data.gov.sg. Anything else is not trusted over the verification."""
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return None
    if (len(value), value[:1]) in ((3, ["singstat"]), (4, ["datagovsg"])):
        return tuple(value)
    return None


def _day(session: Path) -> str:
    """A session's creation date, for records that carry no time of their own."""
    name = session.name
    return f"{name[:4]}-{name[4:6]}-{name[6:8]}" if name[:8].isdigit() else ""


def turned_down(root: Path, problems: list[str] | None = None) -> dict[tuple[str, ...], tuple[str, str]]:
    """Candidates earlier sessions under ``root`` turned down, keyed by series (see
    :func:`sgmacro.agent.tools.candidate_key`); the value is ``(label and source, who turned it down
    and why)``. Without this a rerun, which sees the same registry, would propose
    the same candidates again.

    Evidence is applied in the order it happened, by the times the records carry
    (``verified_at``/``verified_on`` for a verification, which `sgmacro verify`
    rewrites when it re-checks an old session; ``at``/``on`` for a decision), not
    by session folder. A person's decline lasts until a person adopts the series. A
    verifier rejection lasts only for a lasting failure (stale, unreadable, missing,
    empty) and is lifted by a later verification that passes (accept or review).
    A decision is applied to the series identity stored with it when it was made;
    older decisions without one fall back to the session's current verification.
    Unreadable records are skipped entry by entry, so one bad entry neither stops a
    new session nor drops the others; each skip is reported in ``problems``."""
    problems = problems if problems is not None else []
    events: list[tuple[str, str, int, str, tuple[str, ...], str, Any]] = []
    for session in sorted(p for p in root.glob("*") if p.is_dir()):
        verification = _read_json(session / "verification.json")
        results = verification.get("results") if isinstance(verification, dict) else None
        if not isinstance(results, list):
            if (session / "verification.json").exists():
                problems.append(f"{session.name}/verification.json is unreadable")
            continue
        checked = str(verification.get("verified_at") or verification.get("verified_on") or _day(session))
        candidates: dict[int, tuple[tuple[str, ...], str]] = {}
        bad_results = 0
        for i, result in enumerate(results, 1):
            if not isinstance(result, dict) or not isinstance(result.get("proposal"), dict):
                bad_results += 1
                continue
            proposal = result["proposal"]
            key = result_identity(result)
            label = f"{proposal.get('label')} ({describe_candidate(proposal)})"
            candidates[i] = (key, label)
            checks = result.get("checks") if isinstance(result.get("checks"), list) else []
            lasting = [c for c in checks if isinstance(c, dict)
                       and c.get("severity") == "fail" and c.get("check") in LASTING_FAILURES]
            if result.get("verdict") == "reject" and lasting:
                events.append((checked, session.name, 0, "rejected", key, label, lasting[0].get("message")))
            elif result.get("verdict") in ("accept", "review"):
                events.append((checked, session.name, 0, "passed", key, label, None))
        if bad_results:
            problems.append(f"{session.name}/verification.json: {bad_results} result(s) skipped")
        rows, skipped = _read_jsonl(session / "decisions.jsonl")
        bad_series = 0
        for line, d in enumerate(rows, 1):
            n = d.get("candidate")
            known = None if isinstance(n, bool) or not isinstance(n, int) else candidates.get(n)
            stored = _identity(d.get("series"))
            if d.get("series") is not None and stored is None:
                bad_series += 1                    # not trusted; the verification's identity is used
            if d.get("decision") not in ("declined", "adopted") or (stored is None and known is None):
                skipped += 1                       # a malformed entry: skip it, keep the rest
                continue
            key = stored or known[0]
            label = known[1] if known else f"{d.get('label')} ({' '.join(key[1:3])})"
            events.append((str(d.get("at") or d.get("on") or checked), session.name, line, d["decision"], key,
                           label, d.get("on")))
        if skipped:
            problems.append(f"{session.name}/decisions.jsonl: {skipped} line(s) skipped")
        if bad_series:
            problems.append(f"{session.name}/decisions.jsonl: {bad_series} malformed series field(s) ignored")
    state: dict[tuple[str, ...], tuple[str, str, str]] = {}
    for when, _, _, kind, key, label, detail in sorted(events, key=lambda e: e[:3]):
        held = state.get(key)
        if kind == "adopted":
            state.pop(key, None)
        elif kind == "declined":
            state[key] = (label, f"declined by a person on {detail or when[:10]}", "person")
        elif kind == "rejected" and (held is None or held[2] != "person"):
            state[key] = (label, f"rejected by the verifier ({detail})", "verifier")
        elif kind == "passed":
            # A wide table's row is one series whatever the filters; a record from
            # before the layout was stored keyed it with them, so lift those too.
            same = [k for k in state if k == key or (key[-1:] == ("wide",) and k[:3] == key[:3])]
            for k in same:
                if state[k][2] == "verifier":
                    state.pop(k)
    return {key: (label, why) for key, (label, why, _) in state.items()}


def with_turned_down(brief: str, found: dict[tuple[str, ...], tuple[str, str]]) -> str:
    if not found:
        return brief
    lines = [f"- {label}: {why[:160]}" for label, why in list(found.values())[-MAX_TURNED_DOWN:]]
    return brief + "\n\nAlready reviewed and turned down; do not propose these again:\n" + "\n".join(lines)


#: The most proposals each built-in brief allows (a free-text brief has no limit).
BRIEF_LIMITS = {"predictors": 4, "repair": 2}

BRIEFS = {
    "predictors": (
        "Find up to four additional publicly available Singapore series that could help predict changes "
        "in residential sale prices or rents and are NOT already in the registry. Prefer channels the "
        "registry covers thinly (for example household wealth, foreign workforce or construction costs), "
        "but only where suitable current data exist; propose fewer if they do not."),
    "repair": (
        "The registry series `{series_id}` failed validation: {reason}. Find the best currently "
        "published replacement measuring the same concept, verify it is up to date, and propose it with "
        "`replaces` set to `{series_id}`. Propose at most two candidates."),
}
