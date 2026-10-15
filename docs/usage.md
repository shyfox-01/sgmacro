# Command reference

Run commands from the repository root after installing the package in a virtual
environment. Use `.venv/bin/sgmacro` on macOS/Linux or
`.\.venv\Scripts\sgmacro.exe` in PowerShell if the environment is not active.

## Reports

```bash
sgmacro run --as-of 2026-10-02 --offline --llm cache
sgmacro run --as-of YYYY-MM-DD --refetch --llm off
```

Replace `YYYY-MM-DD` with a reporting date. Future dates are rejected.

| Option | Meaning |
| --- | --- |
| `--refetch` | Download source objects again; save a new vintage only if bytes or publisher metadata changed |
| `--offline` | Use stored data and prohibit remote model calls; a local model may still run |
| `--llm cache` | Use a matching cached summary, otherwise a template |
| `--llm off` | Always use the template summary |
| `--llm auto` | Use a cached response or resolve an available backend |
| `--strict` | Exit nonzero on warnings or unexpected validation verdicts |
| `--registry PATH` | Use another registry file |

For a replay without model inference, combine `--offline` with
`--llm cache` or `--llm off`. Turning off the summary changes the included
report's content and therefore its run ID.

The included snapshots were retrieved on 2 October 2026. A report for an earlier
date applies release-lag gates but warns when no historical vintage is available.
It is not a complete point-in-time backtest.

Network access is needed to fetch current data. Requests to data.gov.sg are
rate-limited and retried with backoff. Source outages can lead to a saved-vintage
fallback, which the report labels as degraded.

## Validation and tracing

```bash
sgmacro sources
sgmacro validate --as-of 2026-10-02
sgmacro trace F:331a4bda --run 2026-10-02_d26e13cf
```

Validation is offline by default; `--online` allows retrieval. It prints each
series' verdict and findings without generating a report.

Tracing checks raw-file hashes and re-reads source cells for input-backed facts.
A run ID identifies the exact report. `--as-of DATE` also works if exactly one
saved run matches that date.

The HTML report contains the figure lineage and works as a local file without
external scripts or a web server.

## Discovery

```bash
sgmacro models
sgmacro discover --brief predictors
```

Discovery requires a model with tool support. See [model setup](models.md).

A session saves messages, tool results, proposals, verification results and a
registry draft under `discovery/<session>/`. The built-in predictor brief allows
at most four proposals; a run can finish with fewer.

Each proposal is checked for data quality, duplication and its stated relationship
to the targets. Weak or contrary correlations are warnings for review, not proof
that a variable is useless. The command prompts before adding a candidate.

To revisit a recorded session:

```bash
sgmacro verify discovery/SESSION --no-prompt
sgmacro accept discovery/SESSION N
```

Replace `SESSION` with a session directory and `N` with the displayed candidate
number. `accept` changes the registry. Annual candidates must specify a reference
month; the command supports `--reference-month M`.

Re-verification can change a proposal's status as source data changes. Previously
declined candidates and lasting verification failures inform later sessions.

`sgmacro catalogue` refreshes the local data.gov.sg catalogue used for search.
The repository includes a catalogue snapshot, so this refresh is optional.

## Repair

```bash
sgmacro discover --brief repair --series mas_housing_loans_discontinued
```

The brief looks for another source for the same concept, with at most two
proposals. A person reviews the meaning and approves a replacement. Repair does
not run automatically when a report encounters a failure.

If no suitable replacement is accepted, the existing failure remains visible.
Neither missing data nor mismatched definitions are silently patched.

## Outputs

| Path | Purpose |
| --- | --- |
| `config/sources.yaml` | Series definitions and validation settings |
| `config/policy_events.yaml` | Manually maintained policy events |
| `data/snapshots/` | Raw source bytes and vintage metadata |
| `data/processed/<run-id>/` | Tidy observations, facts, signals and diagnostics |
| `reports/<run-id>/` | Markdown, HTML and SVG output |
| `runs/<run-id>/` | Input/output manifest and append-only invocation log |
| `llm_cache/` | Summary responses keyed by the request content |
| `discovery/` | Recorded proposals and review results |

A repeated request can reuse its run directory; each invocation adds an execution
record. Raw snapshots are retained. Keep your own backups before making local
registry or data changes.

## Docker

A Dockerfile is provided. Its build installs dependencies and runs the offline
test suite. The default command reads the included snapshots and summary cache.

```bash
docker build -t sgmacro .
docker run --rm sgmacro
```

Mount an output directory if you want to retain generated files after the
container exits. Docker execution has not been verified for this publication;
the documented Python environment is the tested path.
