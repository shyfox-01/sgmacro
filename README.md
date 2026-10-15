# sgmacro

[![Tests](https://github.com/shyfox-01/sgmacro/actions/workflows/tests.yml/badge.svg)](https://github.com/shyfox-01/sgmacro/actions/workflows/tests.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[中文说明](README.zh-CN.md)

Singapore housing data, with source checks and a report you can audit.

sgmacro collects housing and macroeconomic series from SingStat and data.gov.sg,
checks their definitions and freshness, and produces CSV/JSON tables and an
analytical report. Every reported figure has a source reference and a calculation
record.

The included dataset covers three housing indices and thirteen candidate
predictors. It is a starting point for housing research; the project does not fit
a forecasting model.

[Sample report](reports/2026-10-02_d26e13cf/report.md) ·
[Design and data choices](docs/design.md) ·
[Command reference](docs/usage.md)

![Housing indices in the included sample](reports/2026-10-02_d26e13cf/charts/targets.svg)

## Run the sample

Python 3.12 is the tested environment. The package requires Python 3.12 or newer.

```bash
git clone https://github.com/shyfox-01/sgmacro.git
cd sgmacro
```

**Windows / PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" -c constraints.txt
.\.venv\Scripts\sgmacro.exe run --as-of 2026-10-02 --offline --llm cache
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" -c constraints.txt
.venv/bin/sgmacro run --as-of 2026-10-02 --offline --llm cache
```

Once dependencies are installed, this example needs no network access, API key or
running model. It reads the bundled snapshots and cached summary.

Open [report.html](reports/2026-10-02_d26e13cf/report.html) locally in a browser
to inspect a figure's source, period and calculation. GitHub displays the
[Markdown report](reports/2026-10-02_d26e13cf/report.md) directly.

The output includes:

| Location | Contents |
| --- | --- |
| `reports/<run-id>/` | Markdown and HTML reports, plus SVG charts |
| `data/processed/<run-id>/` | Observations, series definitions, validation findings, signals and diagnostics |
| `data/snapshots/` | Source responses, publisher metadata, retrieval times and file hashes |
| `runs/<run-id>/` | Manifest and execution log |

Run IDs depend on the inputs and report content. Replaying the same request uses
the same directory; different report content gets a different ID.

## Data coverage

| Target | Coverage |
| --- | --- |
| URA private residential property price index | Private sales, all residential |
| URA private residential rental index | Non-landed private rents, whole island |
| HDB resale price index | HDB resale transactions |

The thirteen predictors cover financing costs, bond yields, housing credit, GDP,
unemployment, inflation, transactions, inventory, vacancy, construction, population
and housing policy. The [registry](config/sources.yaml) records each series'
economic rationale, source identifier, unit, publication lag and expected effect.

Most inputs come from public APIs. Housing policy comes from a
[curated list of official announcements](config/policy_events.yaml). It counts
tightening and easing rounds and needs manual updates.

The sample uses data retrieved on **2 October 2026**. It is a saved example,
not a continuously updated feed. Source attribution and data terms are in
[NOTICE.md](NOTICE.md).

## Workflow

```text
source registry
  -> fetch and store snapshots
  -> parse and validate each series
  -> calculate changes, signals and lead-lag diagnostics
  -> assemble facts and source references
  -> write tables, charts and reports
```

Validation checks the selected series itself, including units, frequency,
missing observations, duplicate periods and stale data. Publication lags and
stored snapshot dates determine what a run can use for its reporting date.

Three negative controls stay in the registry: a discontinued rental series, an
outdated housing-loan series, and a population series with a deliberately wrong
unit. The tests expect all three to be rejected.

Two optional workflows use a language model:

- **Discovery and repair.** An agent searches for new predictors or replacement
  sources. Code verifies each proposal; a person decides whether to add it to
  the registry.
- **Report summary.** A model drafts statements against computed facts.
  A checker rejects unsupported references, numerical text and inconsistent
  directions. The report uses a template if too few claims pass.

The regular data pipeline does not invoke the discovery agent.
[Model setup](docs/models.md) explains the supported backends and their limits.

## Common commands

The examples below assume the virtual environment is active. Otherwise use
`.venv/bin/sgmacro` or `.\.venv\Scripts\sgmacro.exe`.

```bash
# Download current source data for a reporting date
sgmacro run --as-of YYYY-MM-DD --refetch --llm off

# Inspect validation results without generating a report
sgmacro validate --as-of 2026-10-02

# Recalculate a sample figure from its saved source cells
sgmacro trace F:331a4bda --run 2026-10-02_d26e13cf

# Propose new predictors; requires a model
sgmacro discover --brief predictors

# Look for a replacement source for a failed series
sgmacro discover --brief repair --series mas_housing_loans_discontinued
```

Replace `YYYY-MM-DD` with the intended date. Details, including proposal
approval and offline behavior, are in the [command reference](docs/usage.md).

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python -m pytest`.

The offline suite covers parsing, unit-aware calculations, publication dates,
source failures, proposal verification, adoption, model-response checks and
report replay. It also recomputes the sample's 49 input-backed facts directly
from raw snapshots and compares regenerated reports with the included files.

Recorded local-model runs and a small summary comparison are available in
[discovery/](discovery/) and [docs/evidence/](docs/evidence/).
Hosted model backends were tested with simulated servers, not paid live calls.

## Limits

- Correlations and signals describe the data. Their predictive value has not
  been established with out-of-sample forecasts.
- The bundled snapshots share one retrieval date. Earlier reporting dates can
  still contain later revisions; those runs carry a warning.
- Publication lags and signal thresholds are declared settings. HDB rents and
  regional housing targets are outside the current coverage.
- Policy-round counts do not measure the intensity of individual measures.
- Passing a text checker establishes consistency with declared rules, not the
  truth of an economic explanation.

## License

Project code and documentation are available under the [MIT License](LICENSE).
Source data remains subject to its publishers' terms; see [NOTICE.md](NOTICE.md).
