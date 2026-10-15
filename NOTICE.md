# Sources and attribution

[中文说明](NOTICE.zh-CN.md)

The MIT license covers this project's code and documentation. It does not replace
the terms governing upstream datasets, model weights or third-party dependencies.

## Public datasets

This repository contains information from datasets listed in
[the source inventory](docs/data-sources.md), retrieved on 2 October 2026 from
[data.gov.sg](https://data.gov.sg/) and
[SingStat Table Builder](https://tablebuilder.singstat.gov.sg/).
The data is made available under the
[Singapore Open Data Licence](https://data.gov.sg/open-data-licence).
SingStat's [Terms of Use](https://www.singstat.gov.sg/terms-of-use) also apply.

The original publishers include the Urban Redevelopment Authority, Housing &
Development Board, Monetary Authority of Singapore, Ministry of Manpower and
Singapore Department of Statistics. Each saved vintage identifies its source,
publisher, retrieval time and file hashes.

Raw snapshots and the catalogue are saved examples rather than a current feed.
Processed tables, calculations and commentary are produced by this project and
should not be attributed to the data publishers. No government agency endorses
this project.

Fixtures in `tests/fixtures/` contain trimmed source responses or synthetic
examples. Recorded model sessions and summary responses are included for
inspection and replay; no model weights are distributed.

## Policy events

`config/policy_events.yaml` records dates, instruments and references to official
policy announcements. It is a manually maintained dataset. Linked announcements
remain subject to their publishers' terms.

## Development tools

Claude and ChatGPT / Codex were used during development. Model-generated code and
text remain subject to the checks and limitations described in the documentation.
