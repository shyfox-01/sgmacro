"""Read-only tools the discovery agent may call.

Every tool is deterministic code over the public APIs (or the local catalogue
snapshot). The agent decides *what to look at*; it never reads data any other
way, never writes to the registry, and its proposals are only recorded here for
the verifier and a human to judge.
"""

from __future__ import annotations

import gzip
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from sgmacro.agent.verify import _registry_duplicate
from sgmacro.llm.backends import ToolSpec
from sgmacro.registry import Registry, SingStatRef
from sgmacro.sources.singstat import SingStatClient

CATALOGUE_PATH = Path("data/catalogue/datagovsg_catalogue.json.gz")
DATASTORE_SEARCH = "https://data.gov.sg/api/action/datastore_search"
MAX_RESULT_CHARS = 6000

CHANNELS = ["Mortgage financing cost", "Long-horizon discount rate", "Mortgage credit", "Income and activity",
            "Labour market slack", "General price level", "Realised housing costs", "Market activity",
            "Developer inventory", "Supply slack", "Forward supply", "Rental demand", "Housing policy",
            "Household wealth", "Other"]

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Short human name of the series"},
        "source": {"type": "string", "enum": ["singstat", "datagovsg"]},
        "table_id": {"type": "string", "description": "SingStat table id, e.g. M212261 (singstat only)"},
        "series_no": {"type": "string", "description": "SingStat seriesNo, e.g. 1.2 (singstat only)"},
        "dataset_id": {"type": "string", "description": "data.gov.sg dataset id (datagovsg only)"},
        "value_column": {"type": "string", "description": "datagovsg only: the numeric column holding the value; "
                         "for a wide table (first column DataSeries, one row per series) the series' row label from "
                         "that column, e.g. 'Cement', never 'DataSeries' itself"},
        "filters": {"type": "string", "description": "datagovsg only: JSON object of column=value filters, or {}"},
        "frequency": {"type": "string", "enum": ["monthly", "quarterly", "annual"]},
        "unit": {"type": "string", "description": "unit of measure as the publisher states it"},
        "channel": {"type": "string", "enum": CHANNELS},
        "markets": {"type": "array", "items": {"type": "string", "enum": ["sales", "rents"]}},
        "effect_on_sales": {"type": "string", "enum": ["positive", "negative", "ambiguous", "not_applicable"]},
        "effect_on_rents": {"type": "string", "enum": ["positive", "negative", "ambiguous", "not_applicable"]},
        "rationale": {"type": "string", "description": "mechanism linking the series to prices or rents"},
        "replaces": {"type": "string", "description": "registry series id this would replace, or empty"},
        "reference_month": {"type": "integer", "description": "annual series only: the month (1-12) each figure "
                            "refers to, e.g. 6 for an end-June estimate; 0 if not annual or not stated"},
    },
    "required": ["label", "source", "table_id", "series_no", "dataset_id", "value_column", "filters",
                 "frequency", "unit", "channel", "markets", "effect_on_sales", "effect_on_rents",
                 "rationale", "replaces", "reference_month"],
    "additionalProperties": False,
}


def candidate_key(proposal: dict[str, Any], layout: str | None = None) -> tuple[str, ...]:
    """The series a proposal selects, by the rules the extractor applies, not by its
    label or wording: a wide data.gov.sg table is read by row label alone, without
    its surrounding blanks (filters play no part); a tidy one by its exact column
    name, with filter values compared as text (1 and "1" select the same rows).
    Source identifiers are read as :func:`sgmacro.agent.verify.verify_proposal`
    reads them. ``layout`` is known once a candidate has been verified."""
    if proposal.get("source") == "singstat":
        return ("singstat", str(proposal.get("table_id") or "").strip().upper(),
                str(proposal.get("series_no") or "").strip())
    dataset = str(proposal.get("dataset_id") or "").strip()
    column = str(proposal.get("value_column") or "")
    if layout == "wide":
        return ("datagovsg", dataset, column.strip(), "wide")
    filters = proposal.get("filters") or "{}"
    try:
        parsed = json.loads(filters) if isinstance(filters, str) else dict(filters)
        text = json.dumps({str(k): str(v) for k, v in parsed.items()}, sort_keys=True)
    except (TypeError, ValueError, AttributeError):
        text = str(filters)
    return ("datagovsg", dataset, column, text)


def result_identity(result: dict[str, Any]) -> tuple[str, ...]:
    """The series a verification result is about: its proposal, read with the
    table layout the verifier found (recorded before any failure, else from the
    draft). Stored with a person's decision, so a later re-verification that
    fails before it sees the table cannot change what was decided about."""
    observed = result.get("observed") if isinstance(result.get("observed"), dict) else {}
    draft = result.get("draft") if isinstance(result.get("draft"), dict) else {}
    source = draft.get("source") if isinstance(draft.get("source"), dict) else {}
    proposal = result.get("proposal") if isinstance(result.get("proposal"), dict) else {}
    return candidate_key(proposal, observed.get("layout") or source.get("layout"))


def describe_candidate(proposal: dict[str, Any]) -> str:
    if proposal.get("source") == "singstat":
        return f"SingStat {proposal.get('table_id')} series {proposal.get('series_no')}"
    return f"data.gov.sg {proposal.get('dataset_id')} column {proposal.get('value_column')}"


def _strict(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


TOOL_SPECS = [
    ToolSpec("search_singstat", "Keyword search over SingStat Table Builder time-series table titles. "
             "Use short keywords (one to three words). Returns table ids and titles.",
             _strict({"keyword": {"type": "string"}})),
    ToolSpec("describe_singstat_table", "Metadata of one SingStat table: frequency, seasonal adjustment, "
             "publisher, last update, period range, and its series (seriesNo, label, unit).",
             _strict({"table_id": {"type": "string"}})),
    ToolSpec("preview_singstat_series", "First period, observation count and the latest six observations "
             "of one SingStat series.",
             _strict({"table_id": {"type": "string"}, "series_no": {"type": "string"}})),
    ToolSpec("search_datagovsg", "Keyword search over the data.gov.sg dataset catalogue (names and "
             "descriptions). Returns dataset ids, names, publishers and last-update dates.",
             _strict({"query": {"type": "string"}})),
    ToolSpec("describe_datagovsg_dataset", "Columns, coverage, publisher, last update and the most recent "
             "rows of one data.gov.sg dataset.",
             _strict({"dataset_id": {"type": "string"}})),
    ToolSpec("list_registry", "The series already in the registry (id, label, channel, source).", _strict({})),
    ToolSpec("propose_series", "Record one candidate predictor for deterministic verification and human "
             "review. Only propose series you have described and previewed.", PROPOSAL_SCHEMA),
    ToolSpec("finish", "End the session with a short summary of what was proposed and why.",
             _strict({"summary": {"type": "string"}})),
]


@dataclass
class Toolbox:
    registry: Registry
    singstat: SingStatClient
    http_get: Callable[..., Any] = requests.get
    catalogue_path: Path = CATALOGUE_PATH
    proposals: list[dict[str, Any]] = field(default_factory=list)
    finished: str | None = None
    #: Series turned down in earlier sessions (key -> who turned it down and why).
    turned_down: dict[tuple[str, ...], str] = field(default_factory=dict)
    #: The most proposals the brief allows (None: no limit). Live, a model asked for
    #: "up to four" proposed six; the prompt asks, this enforces.
    max_proposals: int | None = None
    _catalogue: list[dict[str, Any]] | None = None

    # -- dispatch ----------------------------------------------------------
    def effective_arguments(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """The arguments a tool actually uses, and the extra ones it would ignore.
        Local models sometimes add fields a tool does not take (a summary plus a
        list for finish); they are dropped rather than failing the call, and two
        calls that differ only in such fields are the same call."""
        handler = getattr(self, f"tool_{name}", None)
        if handler is None or not isinstance(arguments, dict):
            return arguments, []
        params = inspect.signature(handler).parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return arguments, []
        ignored = sorted(k for k in arguments if k not in params)
        return {k: v for k, v in arguments.items() if k not in ignored}, ignored

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool. Returns ``(text, is_error)``; never raises."""
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return f"unknown tool {name!r}", True
        arguments, ignored = self.effective_arguments(name, arguments)
        try:
            text = handler(**arguments)
        except TypeError as exc:
            return f"bad arguments for {name}: {exc}", True
        except Exception as exc:  # noqa: BLE001 - surface upstream failures to the agent
            return f"{name} failed: {type(exc).__name__}: {exc}", True
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n... [truncated at {MAX_RESULT_CHARS} characters]"
        if ignored:
            text += f"\n(ignored argument(s) this tool does not take: {', '.join(ignored)})"
        return text, False

    # -- SingStat ----------------------------------------------------------
    def tool_search_singstat(self, keyword: str) -> str:
        hits = self.singstat.search(keyword)
        if not hits:
            return f"no time-series tables match {keyword!r}; try a shorter or different keyword"
        lines = [f"{h['table_id']} | {h['title']}" for h in hits[:25]]
        more = f"\n({len(hits) - 25} more not shown)" if len(hits) > 25 else ""
        return "\n".join(lines) + more

    def tool_describe_singstat_table(self, table_id: str) -> str:
        records = json.loads(self.singstat.metadata(table_id))["Data"]["records"]
        rows = records.get("row") or []
        head = (f"{records.get('title')}\nfrequency: {records.get('frequency')} | adjustment: "
                f"{records.get('adjustmentType') or 'n/a'} | publisher: {records.get('dataSource')}\n"
                f"last updated: {records.get('dataLastUpdated')} | periods: {records.get('startPeriod')} to "
                f"{records.get('endPeriod')}\nseries (seriesNo | label | unit):")
        body = [f"{r.get('seriesNo')} | {r.get('rowText')} | {r.get('uoM')}" for r in rows[:60]]
        more = [f"({len(rows) - 60} more series not shown)"] if len(rows) > 60 else []
        return "\n".join([head, *body, *more])

    def tool_preview_singstat_series(self, table_id: str, series_no: str) -> str:
        payload = json.loads(self.singstat.tabledata(table_id, [series_no]))
        row = payload["Data"]["row"][0]
        cols = row.get("columns") or []
        latest = ", ".join(f"{c['key']}={c['value']}" for c in cols[-6:])
        return (f"{table_id} series {series_no}: {row.get('rowText')} [{row.get('uoM')}]\n"
                f"first period {cols[0]['key'] if cols else '-'}, {len(cols)} observations\nlatest: {latest}")

    # -- data.gov.sg -------------------------------------------------------
    def catalogue(self) -> list[dict[str, Any]]:
        if self._catalogue is None:
            if not self.catalogue_path.exists():
                raise FileNotFoundError(f"no catalogue snapshot at {self.catalogue_path}; "
                                        "run `sgmacro catalogue` first")
            with gzip.open(self.catalogue_path, "rt", encoding="utf-8") as fh:
                self._catalogue = json.load(fh)["datasets"]
        return self._catalogue

    def tool_search_datagovsg(self, query: str) -> str:
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1]
        if not terms:
            return "empty query"
        scored = []
        for item in self.catalogue():
            name = (item.get("name") or "").lower()
            desc = (item.get("description") or "").lower()
            score = sum(3 * name.count(t) + desc.count(t) for t in terms)
            if all(t in name or t in desc for t in terms):
                score += 10
            if score:
                scored.append((score, item))
        scored.sort(key=lambda s: (-s[0], s[1].get("name") or ""))
        if not scored:
            return f"no datasets match {query!r}"
        return "\n".join(
            f"{i['dataset_id']} | {i['name']} | {i.get('publisher')} | updated {(i.get('last_updated') or '')[:10]}"
            for _, i in scored[:15])

    def tool_describe_datagovsg_dataset(self, dataset_id: str) -> str:
        meta = requests.get(f"https://api-production.data.gov.sg/v2/public/api/datasets/{dataset_id}/metadata",
                            timeout=30, headers={"Accept": "application/json"}).json().get("data") or {}
        columns = meta.get("columnMetadata") or {}
        names = [(columns.get("map") or {}).get(c, c) for c in columns.get("order") or []]
        # A wide table holds one series per row, labelled in its DataSeries column.
        # Live, a model proposed "DataSeries" itself three times; list the labels.
        wide = bool(names) and names[0] == "DataSeries"
        preview = self.http_get(DATASTORE_SEARCH, params={"resource_id": dataset_id, "limit": 40 if wide else 3,
                                                          "sort": "_id desc"}, timeout=30).json()
        records = (preview.get("result") or {}).get("records") or []
        rows = []
        # A wide table (one dataset has 332 columns) would fill the model's context
        # with a few rows; show the columns listed above and count the rest.
        for r in records[:3]:
            cells = [(k, v) for k, v in r.items() if k != "_id"]
            extra = f" (+{len(cells) - 15} more columns)" if len(cells) > 15 else ""
            rows.append(json.dumps(dict(cells[:15]), ensure_ascii=False) + extra)
        text = (f"{meta.get('name')}\npublisher: {meta.get('managedBy')} | last updated: {meta.get('lastUpdatedAt')}"
                f"\ncoverage: {meta.get('coverageStart')} to {meta.get('coverageEnd')}\n"
                f"columns ({len(names)}): {names[:15]}{' ...' if len(names) > 15 else ''}\n"
                "most recent rows:\n" + "\n".join(rows))
        if wide:
            labels = [str(r["DataSeries"]) for r in records if r.get("DataSeries")]
            text += ("\nlayout: wide (one row per series, one column per period). To propose one of these series, "
                     f"set value_column to its DataSeries row label{', e.g. ' + repr(labels[0]) if labels else ''}. "
                     f"Row labels (first {len(labels)} rows returned, at most 40): {labels}")
        return text

    # -- registry and proposals ---------------------------------------------
    def tool_list_registry(self) -> str:
        lines = []
        for spec in self.registry.series:
            ref = spec.source
            where = (ref.identifier if isinstance(ref, SingStatRef)
                     else f"data.gov.sg {ref.identifier}" if ref.source == "datagovsg"
                     else f"curated list {ref.identifier}")
            lines.append(f"{spec.id} | {spec.role.value} | {spec.label} | channel: {spec.channel or '-'} | {where}")
        return "\n".join(lines)

    def tool_propose_series(self, **proposal: Any) -> str:
        if proposal.get("source") == "singstat" and not (proposal.get("table_id") and proposal.get("series_no")):
            raise ValueError("a singstat proposal needs table_id and series_no")
        if proposal.get("source") == "datagovsg" and not (proposal.get("dataset_id") and proposal.get("value_column")):
            raise ValueError("a datagovsg proposal needs dataset_id and value_column")
        if proposal.get("source") == "datagovsg" and str(proposal.get("value_column")).strip() == "DataSeries":
            raise ValueError("DataSeries is the label column of a wide table: set value_column to the row label of "
                             "the series you mean (describe_datagovsg_dataset lists them)")
        if not proposal.get("markets"):
            raise ValueError("state at least one market")
        if self.max_proposals is not None and len(self.proposals) >= self.max_proposals:
            raise ValueError(f"the brief allows at most {self.max_proposals} proposal(s) and all are made; "
                             "call finish")
        # The verifier rejects a duplicate anyway; refusing it here leaves the slot
        # free. Live, a 9B model proposed registry series once the registry listing
        # had been shortened out of its context.
        duplicate = _registry_duplicate(self.registry, proposal)
        if duplicate and duplicate != (proposal.get("replaces") or "").strip():
            raise ValueError(f"{describe_candidate(proposal)} is already in the registry as {duplicate!r}; "
                             "propose a series the registry does not have (list_registry shows them)")
        why = self.turned_down.get(candidate_key(proposal))
        if why is None and proposal.get("source") == "datagovsg":
            why = self.turned_down.get(candidate_key(proposal, "wide"))     # a wide table's row, any filters
        if why:
            raise ValueError(f"{describe_candidate(proposal)} was already {why}; propose a different series")
        self.proposals.append(proposal)
        return f"recorded proposal #{len(self.proposals)}: {proposal.get('label')}"

    def tool_finish(self, summary: str) -> str:
        self.finished = summary
        return "session finished"
