"""Command-line interface.

    sgmacro run --as-of 2026-10-02              full workflow -> reports/<run id>/report.md
    sgmacro validate --as-of 2026-10-02         verdict per series, no report
    sgmacro trace F:1a2b3c4d --run <run id>     re-derive one figure from the raw bytes
    sgmacro discover --brief predictors         agent proposes new series (+ verification)
    sgmacro verify discovery/<session>          re-run verification of a session's proposals
    sgmacro accept discovery/<session> 2        add candidate 2 to the registry (no prompt)
    sgmacro catalogue                           refresh the data.gov.sg catalogue snapshot
    sgmacro sources                             what the registry declares
    sgmacro models                              which LLM backends this machine can use
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer

from sgmacro.pipeline import RunConfig, analyse, fetch_stage, run
from sgmacro.registry import Role, load_registry
from sgmacro.validate import Severity

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
SGT = timezone(timedelta(hours=8))

AS_OF = typer.Option(..., "--as-of", help="Reporting date (YYYY-MM-DD); gates which observations are visible")
REGISTRY = typer.Option(Path("config/sources.yaml"), "--registry", help="Source registry")


def _as_of(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"--as-of must be YYYY-MM-DD, got {value!r}") from exc
    if parsed > datetime.now(SGT).date():
        raise typer.BadParameter("--as-of cannot be in the future: nothing published after today exists yet")
    return parsed


LLM_CHOICES = ("auto", "cache", "anthropic", "openai", "ollama", "off")
AGENT_BACKENDS = ("auto", "anthropic", "openai", "ollama")


def _echo(message: str) -> None:
    typer.echo(message, err=True)


@app.command("run")
def run_cmd(
    as_of: str = AS_OF,
    refetch: bool = typer.Option(False, "--refetch", help="Download every object again (a new vintage only if "
                                 "the bytes or the publisher's metadata changed)"),
    offline: bool = typer.Option(False, "--offline", help="No downloads and no remote model call (a local model "
                                 "may still draft; use --llm cache or off for a strict replay)"),
    llm: str = typer.Option("auto", "--llm", help="auto | cache | anthropic | openai | ollama | off "
                            "(see `sgmacro models`)"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero on any warning or unexpected verdict (CI)"),
    registry: Path = REGISTRY,
) -> None:
    """Run the full workflow and write the report, processed tables and run record."""
    if llm not in LLM_CHOICES:
        raise typer.BadParameter(f"--llm must be one of {', '.join(LLM_CHOICES)}, got {llm!r}")
    if offline and llm == "anthropic":
        raise typer.BadParameter("--offline never calls the Claude API; use --llm cache, ollama or off")
    config = RunConfig(as_of=_as_of(as_of), registry_path=registry, llm=llm, refetch=refetch, offline=offline)
    result = run(config, log=_echo)
    targets, predictors = result.reportable(Role.TARGET), result.reportable(Role.PREDICTOR)
    controls = result.by_role(Role.QUARANTINE)
    _echo(f"run {result.run_id}: {len(targets)} targets, {len(predictors)} predictors, "
          f"{sum(o.result.as_expected for o in controls)}/{len(controls)} negative controls as expected, "
          f"{len(result.facts)} facts")
    _echo(f"summary: {result.summary.source}{' (from cache)' if result.summary.from_cache else ''}")
    # Backends passed over matter only when none drafted the summary; otherwise
    # they stay in the run record, so a working setup does not print what looks like errors.
    skipped = result.summary.skipped_backends if result.summary.source == "deterministic" else []
    for note in result.summary.notes + skipped + result.summary.execution_notes:
        _echo(f"  summary note: {note}")
    model_failed = any(" failed: " in n for n in result.summary.execution_notes)
    if (result.summary.source == "deterministic" and not result.summary.notes and not model_failed
            and llm not in ("cache", "off")):
        _echo("  (no model was available to draft the summary; `sgmacro models` shows how to connect one)")
    states = result.states.values()
    new, degraded = sum(s.created for s in states), sum(s.degraded for s in states)
    reused = sum(1 for s in states if s.vintage is not None and not s.created)
    config, rid = result.config, result.run_id
    _echo("saved:")
    for path in sorted(result.outputs):
        if path.endswith("report.md"):
            _echo(f"  report:           {path} (and report.html, charts/)")
    _echo(f"  processed tables: {(config.processed_root / rid).as_posix()}/")
    _echo(f"  raw snapshots:    {config.store_root.as_posix()}/ ({new} new vintage(s), {reused} reused"
          + (f", {degraded} degraded: fetch failed, stored vintage used" if degraded else "") + ")")
    _echo(f"  run record:       {(config.runs_root / rid).as_posix()}/manifest.json and executions.jsonl")
    warnings = [(o.spec.id, f.message) for o in result.outcomes.values() if o.result.reportable
                for f in o.result.by_severity(Severity.WARN)]
    unexpected = [o.spec.id for o in result.outcomes.values()
                  if (o.spec.role is Role.QUARANTINE and not o.result.as_expected)
                  or (o.spec.role is not Role.QUARANTINE and not o.result.reportable)]
    for sid, message in warnings:
        _echo(f"  warn {sid}: {message[:160]}")
    for sid in unexpected:
        _echo(f"  UNEXPECTED verdict for {sid}")
    if strict and (warnings or unexpected):
        raise typer.Exit(code=1)


@app.command()
def validate(as_of: str = AS_OF, registry: Path = REGISTRY,
             offline: bool = typer.Option(True, "--offline/--online")) -> None:
    """Print the verdict and findings for every series (no report written)."""
    config = RunConfig(as_of=_as_of(as_of), registry_path=registry, offline=offline)
    reg = load_registry(registry)
    result = analyse(config, reg, fetch_stage(config, reg, _echo))
    for o in result.outcomes.values():
        r = o.result
        latest = f"{r.latest[0].label}={r.latest[1]:g}" if r.latest else "-"
        flag = "" if r.as_expected in (None, True) else "  [UNEXPECTED]"
        typer.echo(f"{o.spec.id:32} {o.spec.role.value:10} {r.verdict:30} {latest}{flag}")
        for f in r.findings:
            if f.severity is not Severity.INFO:
                typer.echo(f"    {f.severity.value:5} {f.check}: {f.message[:150]}")


@app.command()
def trace(fact: str = typer.Argument(..., help="Fact id, e.g. F:1a2b3c4d"),
          run_id: Optional[str] = typer.Option(None, "--run", help="Run id, as printed in the report header"),
          as_of: Optional[str] = typer.Option(None, "--as-of", help="Instead of --run, when that date has one run"),
          registry: Path = REGISTRY) -> None:
    """Re-derive one reported figure from the raw snapshot bytes and check it."""
    from sgmacro.trace import verify_fact

    config = RunConfig(as_of=_as_of(as_of) if as_of else date.today(), registry_path=registry, offline=True)
    reg = load_registry(registry)
    # A fact id names a figure (series, period, kind), not its value, so the run
    # that reported it must be named too: a later run may hold a revised value.
    if run_id is None:
        if as_of is None:
            raise typer.BadParameter("give --run <run id> (from the report header), or --as-of if that date has one run")
        runs = sorted(p.parent.name for p in config.processed_root.glob(f"{as_of}_*/facts.json"))
        if len(runs) != 1:
            typer.echo(f"{len(runs)} processed runs for {as_of}{': ' + ', '.join(runs) if runs else ''}; "
                       "name one with --run", err=True)
            raise typer.Exit(code=2)
        run_id = runs[0]
    facts_path = config.processed_root / run_id / "facts.json"
    if not facts_path.exists():
        typer.echo(f"no processed run {run_id}; run `sgmacro run --as-of <date>` first", err=True)
        raise typer.Exit(code=2)
    # The raw locators and formulas are read through the registry's series definitions,
    # so an old run must be traced with the registry it was produced with.
    manifest_path = config.runs_root / run_id / "manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["registry"]["fingerprint"]
        if recorded != reg.fingerprint():
            typer.echo(f"run {run_id} was produced with registry {recorded[:12]}, but {registry} is "
                       f"{reg.fingerprint()[:12]}; trace it with that registry version, e.g. "
                       f"`git show <commit>:config/sources.yaml > old.yaml` and `--registry old.yaml`", err=True)
            raise typer.Exit(code=2)
    else:
        typer.echo(f"note: no manifest for run {run_id}, so the registry version is not checked", err=True)
    record = {f["fact_id"]: f for f in json.loads(facts_path.read_text(encoding="utf-8"))}.get(fact)
    if record is None:
        typer.echo(f"{fact} is not a fact of run {run_id}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({k: record[k] for k in ("fact_id", "label", "period", "value", "unit", "display",
                                                  "formula")}, indent=2, ensure_ascii=False))
    if not record["inputs"]:
        if record["kind"] == "correlation":
            typer.echo(f"not re-read from single cells: a correlation over {record['extra'].get('pairs')} quarterly "
                       f"pairs of two change histories, recomputed by the pipeline from "
                       f"data/processed/{run_id}/observations.csv")
        else:
            typer.echo(f"computed from facts {record['depends_on']}; trace those to reach the raw cells. "
                       f"{record.get('note') or ''}".rstrip())
        return
    ok, checks, recomputed = verify_fact(record, {s.id: s for s in reg.series}, config.store_root)
    for c in checks:
        typer.echo(f"  {c.period:9} recorded={c.recorded!s:>14} reread={c.reread!s:>14} "
                   f"sha256={'ok' if c.hash_ok else 'MISMATCH'}  {c.locator}  [{c.file}]")
    shown = "None" if recomputed is None else f"{recomputed:.10g}"
    typer.echo(f"recomputed {shown} vs reported {record['value']!r}: {'VERIFIED' if ok else 'MISMATCH'}")
    if not ok:
        raise typer.Exit(code=1)


PROMPT = typer.Option(True, "--prompt/--no-prompt",
                      help="Ask, candidate by candidate, whether to add it to the registry")


@app.command()
def discover(
    brief: str = typer.Option("predictors", "--brief", help="predictors | repair | free text"),
    series: Optional[str] = typer.Option(None, "--series", help="Series id to repair (with --brief repair)"),
    backend: str = typer.Option("auto", "--backend", help="auto | anthropic | openai | ollama "
                                "(see `sgmacro models`)"),
    max_steps: int = typer.Option(40, "--max-steps"),
    verify_proposals: bool = typer.Option(True, "--verify/--no-verify"),
    prompt: bool = PROMPT,
    registry: Path = REGISTRY,
    out: Path = typer.Option(Path("discovery"), "--out"),
) -> None:
    """Agent proposes series; each is verified, then you decide whether to add it."""
    from sgmacro.agent.discover import BRIEF_LIMITS, BRIEFS, run_agent, save_session, turned_down, with_turned_down
    from sgmacro.agent.tools import Toolbox
    from sgmacro.fetch import build_clients
    from sgmacro.llm.backends import SETUP_HELP, BackendUnavailable, resolve_backend

    if backend not in AGENT_BACKENDS:
        raise typer.BadParameter(f"--backend must be one of {', '.join(AGENT_BACKENDS)}, got {backend!r}")
    reg = load_registry(registry)
    if brief == "repair":
        if not series:
            raise typer.BadParameter("--brief repair needs --series")
        result = _analysis(reg, registry)
        if series not in result.outcomes:
            raise typer.BadParameter(f"no series {series!r} in the registry")
        outcome = result.outcomes[series]
        reason = "; ".join(f.message for f in outcome.result.findings if f.severity is Severity.ERROR) or "unknown"
        text, label = BRIEFS["repair"].format(series_id=series, reason=reason), f"repair_{series}"
    elif brief in BRIEFS:
        text, label = BRIEFS[brief], brief
    else:
        text, label = brief, "custom"
    try:
        live, reasons = resolve_backend(backend)
    except BackendUnavailable as exc:
        live, reasons = None, str(exc).split("; ")
    if live is None:
        _echo("The discovery agent needs a model, and none is available here:")
        for reason in reasons:
            _echo(f"  {reason}")
        _echo("\n" + SETUP_HELP + "\nWithout a model, the recorded sessions can still be re-checked: "
              "`sgmacro verify discovery/<session> --no-prompt`.")
        raise typer.Exit(code=2)
    _echo(f"agent model: {live.name}:{live.model}")
    problems: list[str] = []
    earlier = turned_down(out, problems)
    if problems:
        _echo("note: damaged history entries were skipped: " + "; ".join(problems[:3])
              + (f" (and {len(problems) - 3} more)" if len(problems) > 3 else ""))
    if earlier:
        _echo(f"{len(earlier)} candidate(s) turned down in earlier sessions under {out.as_posix()}/ "
              "will not be proposed again")
    text = with_turned_down(text, earlier)
    singstat, datagov = build_clients()
    toolbox = Toolbox(reg, singstat, turned_down={key: why for key, (_, why) in earlier.items()},
                      max_proposals=BRIEF_LIMITS.get(brief))
    session = run_agent(text, live, toolbox, max_steps=max_steps, log=_echo)
    directory = save_session(session, out, label)
    _echo(f"session saved to {directory} ({len(session.proposals)} proposals, {session.stop_reason})")
    if verify_proposals and session.proposals:
        _verify_dir(directory, reg, registry, singstat, datagov, prompt=prompt)
    if session.model_failed:
        raise typer.Exit(code=1)


@app.command()
def verify(session_dir: Path = typer.Argument(..., help="discovery/<session> directory"),
           prompt: bool = PROMPT, registry: Path = REGISTRY) -> None:
    """Re-run verification of a recorded session's proposals (then ask, unless --no-prompt)."""
    from sgmacro.fetch import build_clients

    singstat, datagov = build_clients()
    _verify_dir(session_dir, load_registry(registry), registry, singstat, datagov, prompt=prompt)


@app.command()
def accept(session_dir: Path = typer.Argument(..., help="discovery/<session> directory"),
           candidate: int = typer.Argument(..., help="Candidate number, as listed by verify"),
           reference_month: Optional[int] = typer.Option(
               None, "--reference-month", min=1, max=12,
               help="Annual series only: the month the figure refers to (6 for end-June)"),
           registry: Path = REGISTRY) -> None:
    """Add a verified candidate to the registry without the interactive prompt."""
    _adopt(session_dir, candidate, registry, load_registry(registry), reference_month)


def _analysis(reg, registry: Path):
    config = RunConfig(as_of=datetime.now(SGT).date(), registry_path=registry, offline=True)
    return analyse(config, reg, fetch_stage(config, reg, lambda m: None))


def _adopt(session_dir: Path, index: int, registry: Path, reg, reference_month: int | None = None) -> None:
    from sgmacro.agent.adopt import AdoptError, adopt, load_results

    results = load_results(session_dir)
    result = results[index - 1] if 1 <= index <= len(results) else {}
    replaces = str((result.get("proposal") or {}).get("replaces") or "").strip()
    retired = None
    old_spec = next((s for s in reg.series if s.id == replaces), None) if replaces else None
    if old_spec is not None and old_spec.role is Role.QUARANTINE:
        # Retiring turns an active entry into a quarantined one; this one already
        # is (a negative control, say), so it keeps its expected status.
        _echo(f"  note: {replaces} is already quarantined ({old_spec.expected_status}); it is left as it is")
    elif old_spec is not None:
        old = _analysis(reg, registry).outcomes[replaces].result
        retired = None if old.reportable else old.verdict
        if retired is None:
            _echo(f"  note: {replaces} still validates, so it is kept alongside the new entry")
    try:
        new_id = adopt(session_dir, index, registry, today=datetime.now(SGT).date(), retired_status=retired,
                       reference_month=reference_month)
    except AdoptError as exc:
        typer.echo(f"not added: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _echo(f"  added {new_id} to {registry}" + (f"; {replaces} retired ({retired})" if retired else "")
          + ". The next `sgmacro run` downloads it.")


def _verify_dir(directory: Path, reg, registry: Path, singstat, datagov, *, prompt: bool) -> None:
    import yaml

    from sgmacro.agent.adopt import record_decline
    from sgmacro.agent.tools import result_identity
    from sgmacro.agent.verify import known_titles, verify_proposal

    session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    today = datetime.now(SGT).date()
    titles = known_titles(reg, Path("data/snapshots"))
    analysis = _analysis(reg, registry)
    targets = [(o.spec, o.result.gated_series()) for o in analysis.reportable(Role.TARGET)]
    for p in session["proposals"]:
        month = p.get("reference_month")
        if prompt and p.get("frequency") == "annual" and not (isinstance(month, int) and 1 <= month <= 12):
            # Asked before verifying, so the lag and lead-lag checks use the right date.
            p["reference_month"] = typer.prompt(
                f"Annual candidate {p.get('label')!r}: which month does each figure refer to (1-12, 6 = end-June)?",
                value_proc=_month)
    results = [verify_proposal(p, reg, today, singstat=singstat, datagov=datagov, titles=titles, targets=targets)
               for p in session["proposals"]]
    (directory / "verification.json").write_text(
        json.dumps({"verified_on": today.isoformat(), "verified_at": datetime.now(SGT).isoformat(timespec="seconds"),
                    "results": [r.to_json() for r in results]},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    drafts = [r.draft for r in results if r.draft is not None and r.verdict != "reject"]
    header = ("# DRAFT registry entries from an agent session (TODO fields are filled with defaults\n"
              "# by `sgmacro accept`, which also validates the entry before writing it).\n")
    (directory / "registry_draft.yaml").write_text(
        header + (yaml.safe_dump({"series": drafts}, sort_keys=False, allow_unicode=True) if drafts else "# none\n"),
        encoding="utf-8", newline="\n")
    for i, r in enumerate(results, 1):
        _echo(_describe(i, len(results), r))
    _echo(f"  saved: {(directory / 'verification.json').as_posix()} and registry_draft.yaml")
    if not prompt:
        _echo("  (no prompt: adopt later with `sgmacro accept <session> <number>`)")
        return
    for i, r in enumerate(results, 1):
        if r.verdict == "reject":
            continue
        if typer.confirm(f"Add candidate {i} ({r.proposal.get('label')}) to the registry?", default=False):
            month = None
            if r.draft and r.draft.get("frequency") == "annual" and not isinstance(r.draft.get("reference_month"),
                                                                                   int):
                month = typer.prompt("  Annual series: which month does each figure refer to (1-12, 6 = end-June)?",
                                     value_proc=_month)
            _adopt(directory, i, registry, load_registry(registry), month)
        else:
            record_decline(directory, i, r.proposal.get("label"), today, series=result_identity(r.to_json()))
    if any(r.verdict != "reject" for r in results):
        _echo(f"  decisions recorded in {(directory / 'decisions.jsonl').as_posix()}")


def _month(value: str) -> int:
    """Prompt converter: a month number, asked again until it is one."""
    month = int(value) if value.strip().isdigit() else 0
    if not 1 <= month <= 12:
        raise typer.BadParameter("enter a month number from 1 to 12")
    return month


def _describe(i: int, n: int, r) -> str:
    p, o = r.proposal, r.observed
    where = (f"SingStat {p.get('table_id')} series {p.get('series_no')}" if p.get("source") == "singstat"
             else f"data.gov.sg {p.get('dataset_id')}")
    lines = [f"[{i}/{n}] {r.verdict.upper():7} {p.get('label')}  ({where})"]
    if o.get("latest"):
        lines.append(f"        {p.get('frequency')}, {o.get('unit') or p.get('unit')}, {o.get('first')} to "
                     f"{o.get('latest')} ({o.get('observations')} observations)")
    lines.append(f"        agent's reasoning: {p.get('rationale')}")
    for c in r.checks:
        if c.severity != "pass" or c.name == "relationship":
            lines.append(f"        {c.severity}: {c.name}: {c.message}")
    return "\n".join(lines)


@app.command()
def catalogue(out: Path = typer.Option(Path("data/catalogue/datagovsg_catalogue.json.gz"), "--out")) -> None:
    """Snapshot the data.gov.sg dataset catalogue for the agent's search tool."""
    from sgmacro.sources.datagovsg import crawl_catalogue
    from sgmacro.sources.http import HttpClient, RateLimiter

    records = crawl_catalogue(HttpClient(RateLimiter(240)))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
               "count": len(records), "datasets": sorted(records, key=lambda r: r["dataset_id"] or "")}
    with gzip.open(out, "wt", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    _echo(f"{len(records)} datasets -> {out}")


@app.command()
def models() -> None:
    """Which LLM backends this machine can use, and how to set up the others.

    Free to run: the Claude API is not contacted, and the other backends are
    only asked to list their models."""
    from sgmacro.llm.backends import SETUP_HELP, backend_status, unused_keys

    rows = backend_status()
    for name, ok, detail in rows:
        typer.echo(f"{name:10} {'ready' if ok else 'not ready':10} {detail}")
    for hint in unused_keys():
        typer.echo(f"note: {hint}")
    first = next(((name, detail) for name, ok, detail in rows if ok), None)
    typer.echo(f"\n`auto` would use: {first[0]} ({first[1].split(' ')[0]})" if first else
               "\nNo backend is ready. Reports still work: the summary comes from the response cache or a "
               "template, and `discover` is the only command that needs a model.")
    typer.echo("Used by `discover` (required) and by `run` (to draft the summary when no cached one fits).\n")
    typer.echo(SETUP_HELP)


@app.command()
def sources(registry: Path = REGISTRY) -> None:
    """List what the registry declares."""
    reg = load_registry(registry)
    for spec in reg.series:
        typer.echo(f"{spec.id:32} {spec.role.value:10} {spec.frequency.value:9} {spec.unit_kind.value:6} "
                   f"{spec.source.identifier}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
