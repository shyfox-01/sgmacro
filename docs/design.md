# Design and data choices

sgmacro prepares a dataset for studying Singapore residential prices and rents.
The registry describes what each series measures; the pipeline downloads,
validates and transforms it; the report makes the calculations inspectable.

The included run has three targets, thirteen predictors and three negative
controls. It uses nineteen API objects and one curated policy file. Of its 340
facts, 49 are levels or changes that can be recomputed directly from raw cells.
The remaining 291 are derived percentiles, signals and correlations.

## Targets and predictors

The targets are the URA private residential property price index, URA non-landed
private rental index and HDB resale price index. Each is read from its agency's
publication on data.gov.sg and checked against the copy on SingStat.

One series represents each selected channel. This keeps the initial registry
small enough to inspect. It does not establish that the proxies are optimal or
that the channel coverage is complete.

| Channel | Series | Reason for inclusion |
| --- | --- | --- |
| Mortgage financing cost | 3-month compounded SORA | A reference rate for floating-rate mortgages |
| Discount rate | 10-year Singapore Government Securities yield | A long-term rate relevant to property valuation |
| Housing credit | Bank housing and bridging loans | Measures the stock of financing for housing |
| Income and activity | Seasonally adjusted real GDP | A broad proxy for economic activity |
| Labour market | Seasonally adjusted unemployment rate | A proxy for employment conditions and payment capacity |
| Realised housing costs | CPI accommodation | A monthly measure of housing costs, with different coverage from new private leases |
| Deflator | CPI all items | Used to compare nominal and inflation-adjusted changes |
| Market activity | Private residential transactions, trailing four quarters | Measures transaction volume |
| Developer inventory | Unsold units with planning approval | Measures available developer inventory |
| Supply slack | Non-landed private residential vacancy rate | Measures unused housing stock |
| Forward supply | Non-landed units under construction | A proxy for future completions |
| Rental demand | Non-resident population, end-June | An imperfect proxy that also includes people outside the private rental market |
| Housing policy | Net tightening rounds over four quarters | A simple record of announced tightening and easing measures |

The [registry](../config/sources.yaml) records expected effects on sales and rents
separately, expected lead windows, source alternatives and known limitations.
An ambiguous effect remains ambiguous in the report.

## Source selection

SingStat supplies rates, bank loans, GDP, unemployment, CPI, population, vacancy
and construction data. data.gov.sg supplies the primary housing indices,
transactions and inventory, along with copies used for cross-checks.

In the saved sample, the SingStat rate series has one more month than the
data.gov.sg copy; overlapping observations agree. This is an observation about
that snapshot, not a guarantee about which source will always be fresher.

The policy file lists twenty announcements with dates, instruments and official
links. A tightening round contributes +1 and an easing round -1. The file needs
manual updates, and its weights do not represent policy intensity.

A second publication can expose extraction or update differences. It is not an
independent measurement when both copies originate with the same agency.

## Registry and storage

Each registry entry specifies its identity, unit, frequency, source selector and
publication lag. Predictors also specify a rationale, market effects and expected
lead window. Pydantic validates these fields; duplicate YAML keys are rejected.

The store keeps source bytes and publisher metadata with retrieval timestamps and
SHA-256 hashes. Raw snapshots bypass Git line-ending conversion so that cloning
the repository does not invalidate those hashes.

Processed outputs are CSV and JSON. Plain files make this dataset easy to
inspect and replay without operating a database.

A run ID incorporates the reporting date, registry fingerprint, selected
vintages, fallback state and summary content. The manifest records input and
output hashes. Execution timestamps belong to an append-only log rather than
the reproducible manifest.

## Reporting dates

Two checks apply:

1. An observation's reference date plus its declared publication lag must be no
   later than the reporting date.
2. Snapshot selection uses the newest stored vintage retrieved by that date.

If no snapshot predates the reporting date, the pipeline uses the oldest stored
vintage and warns that revisions may leak into the result. The bundled archive
has one retrieval date, so it cannot reconstruct a complete historical
information set.

Annual series declare their reference month. For example, an end-June population
estimate is not treated as a December observation.

## Validation and failure handling

| Condition | Handling |
| --- | --- |
| Missing input | Leave the derived change undefined and record why |
| Incomplete sum | Refuse a total with missing or duplicate components |
| Wrong unit or frequency | Reject a mismatch against publisher metadata |
| Missing columns or ambiguous selectors | Stop that series |
| Stale data | Check the last observation of the selected series |
| Fetch failure | Retry within limits, then use a saved snapshot with a degraded warning; exclude the series if none exists |
| Revised observations | Compare the new vintage with the previously stored one |
| Declared series break | Leave changes spanning the break undefined |
| Truncated or empty API response | Check completeness rather than accepting HTTP 200 as sufficient |

The three negative controls exercise these checks with actual source structures:

- `ura_rri_all_discontinued`: the selected all-residential rental slice stops
  at 2021-Q2 even though the containing dataset was updated later.
- `mas_housing_loans_discontinued`: observations stop at 2015-Q3 despite a
  more recent metadata timestamp.
- `nonresident_population_unit_trap`: the declaration says thousands of
  people while the publisher reports individual persons.

They are quarantined entries and do not become normal predictors.

## Calculations and diagnostics

Indices and levels use percentage changes. Rates and ratios use percentage-point
or basis-point changes. Missing inputs and declared breaks are preserved in the
outputs.

A signal combines the latest change with a declared economic effect. It is a
reading of that hypothesis, not a fitted forecast.

Diagnostics compare a predictor's headline change at quarter t with target
year-on-year growth at t+k, for k = 0, 1, 2, 4, 6, 8 and 12 quarters. A pair needs
at least twenty aligned observations. Monthly series use the last month of each
quarter; annual values enter their reference quarter rather than being filled
across all four quarters.

For registered predictors, the agreement verdict uses correlations inside the
declared lead window. Stronger associations outside that window are shown
separately. New candidates have no predeclared window; their diagnostics are
exploratory, and proposed defaults need review.

These diagnostics have no significance tests or multiple-testing correction.
Overlapping year-on-year changes, autocorrelation and correlated predictors
limit what can be inferred.

## Discovery and repair

The agent uses read-only tools to search sources, inspect metadata and preview
data. It proposes a source definition and an economic explanation.

Verification checks identity, availability, schema, units, freshness and duplicate
series. Short history or weak and contradictory correlation evidence can require
review; correlation alone does not reject a candidate.

The user approves each addition. Accepted entries pass registry validation before
an atomic write. The report pipeline does not run this discovery loop.

Repair searches for a replacement source for an existing concept. The reviewer
must decide whether the replacement measures the same thing. A failed active
entry is retired after adoption; an already quarantined entry stays quarantined.
An original entry that still validates can remain alongside the replacement.
The workflow does not automatically splice histories or change the economic
definition.

The built-in briefs allow at most four predictor proposals or two repair
proposals. They can return fewer. Repeated calls reuse results; persistent
non-progress ends the session. Step and context limits preserve partial proposals
for verification. See [model setup](models.md).

## Summary generation and traceability

A model receives computed facts and signal readings, not raw tables. It drafts
statements that cite fact IDs. Code renders the figures beside the statements.

The checker rejects numerical text, unknown references, uncited named series,
directions that contradict the data, inconsistent market stances and forecasts.
Fewer than two accepted claims for a market triggers a template summary.

The checker enforces consistency with declared rules. It does not establish
economic causality or validate the model's explanation.

`sgmacro trace` re-hashes snapshots and independently reads the source cells
behind levels and changes. For derived percentiles, signals and correlations,
the lineage identifies the calculation and its inputs.

The cached summary makes the bundled example reproducible without a model.
A fresh model response may differ and receives its own report identity.

## Evaluation and next steps

Offline tests cover unit calculations, parsing, negative controls, date handling,
provider protocols, agent stopping and adoption, and report replay. The sample's
49 raw-input-backed facts are independently recomputed. A small
[summary comparison](evidence/summary_models.md) records actual local-model
outputs; it is not a general model benchmark.

The next research steps are to accumulate vintages, replace policy counts with
instrument-level measures, and compare forecasting baselines out of sample.
Broader agent evaluation would use fixed briefs, reviewed answers and repeated
runs across models.
