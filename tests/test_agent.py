"""Discovery agent loop, tools and verifier, driven by a scripted fake model."""

from __future__ import annotations

import gzip
import json
from datetime import date

from sgmacro.agent.discover import run_agent
from sgmacro.agent.tools import TOOL_SPECS, Toolbox
from sgmacro.agent.verify import verify_proposal
from sgmacro.llm.backends import ToolCall, Turn
from tests.conftest import ROOT as ROOT_DIR, singstat_metadata, singstat_payload


class FakeSingStat:
    def __init__(self, metadata: dict[str, bytes], data: dict[str, bytes]):
        self._metadata, self._data = metadata, data
        self.searches = []

    def search(self, keyword):
        self.searches.append(keyword)
        return [{"table_id": "M016082", "title": "Personal Disposable Income, Quarterly, Seasonally Adjusted",
                 "topic": "National Accounts"}]

    def metadata(self, table_id):
        if table_id not in self._metadata:
            raise RuntimeError(f"{table_id} not found")
        return self._metadata[table_id]

    def tabledata(self, table_id, series):
        return self._data[table_id]


def quarterly(start_year, n, base=100.0):
    out = []
    for i in range(n):
        out.append((f"{start_year + i // 4} {i % 4 + 1}Q", base + i))
    return out


PDI_META = singstat_metadata({"1": "Million Dollars"}, adjustment="Seasonally Adjusted",
                             title="Personal Disposable Income At Current Prices, Quarterly, Seasonally Adjusted")
PDI_DATA = singstat_payload({"1": quarterly(2005, 86)}, uom="Million Dollars")      # to 2026 Q2
STALE_DATA = singstat_payload({"1": quarterly(2005, 60)}, uom="Million Dollars")    # to 2019 Q4


class ScriptedSession:
    def __init__(self, turns):
        self.turns = list(turns)
        self.received = []

    def send_user(self, text):
        self.received.append(("user", text))
        return self.turns.pop(0)

    def send_tool_results(self, results):
        self.received.append(("tools", [(r.call.name, r.is_error) for r in results]))
        return self.turns.pop(0)


class ScriptedBackend:
    name, model = "scripted", "fake-1"

    def __init__(self, turns):
        self.chat = ScriptedSession(turns)

    def session(self, system, tools):
        assert {t.name for t in tools} == {t.name for t in TOOL_SPECS}
        return self.chat


PROPOSAL = {"label": "Personal disposable income", "source": "singstat", "table_id": "M016082", "series_no": "1",
            "dataset_id": "", "value_column": "", "filters": "{}", "frequency": "quarterly",
            "unit": "Million Dollars", "channel": "Income and activity", "markets": ["sales", "rents"],
            "effect_on_sales": "positive", "effect_on_rents": "positive",
            "rationale": "Disposable income funds mortgage payments and rent.", "replaces": ""}


def toolbox(registry, tmp_path, data=None):
    catalogue = tmp_path / "cat.json.gz"
    with gzip.open(catalogue, "wt", encoding="utf-8") as fh:
        json.dump({"datasets": [
            {"dataset_id": "d_1", "name": "Foreign Workforce Numbers", "publisher": "MOM",
             "last_updated": "2024-06-06", "description": "stock of foreign workforce by pass type"},
            {"dataset_id": "d_2", "name": "Motor Vehicle Population", "publisher": "LTA",
             "last_updated": "2026-01-01", "description": "vehicles"}]}, fh)
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": data or PDI_DATA})
    return Toolbox(registry, fake, catalogue_path=catalogue)


def test_agent_loop_records_tools_proposals_and_errors(registry, tmp_path):
    turns = [
        Turn("Checking the registry.", [ToolCall("1", "list_registry", {})]),
        Turn("", [ToolCall("2", "search_singstat", {"keyword": "disposable income"}),
                  ToolCall("3", "search_datagovsg", {"query": "foreign workforce"})]),
        Turn("", [ToolCall("4", "describe_singstat_table", {"table_id": "M016082"}),
                  ToolCall("5", "preview_singstat_series", {"table_id": "M016082", "series_no": "1"})]),
        Turn("", [ToolCall("6", "propose_series", {**PROPOSAL, "markets": []}),          # invalid: no market
                  ToolCall("7", "no_such_tool", {})]),
        Turn("", [ToolCall("8", "propose_series", PROPOSAL)]),
        Turn("", [ToolCall("9", "finish", {"summary": "one proposal"})]),
    ]
    backend = ScriptedBackend(turns)
    box = toolbox(registry, tmp_path)
    session = run_agent("find predictors", backend, box, log=lambda m: None)
    assert session.stop_reason == "finish tool called"
    assert len(session.proposals) == 1 and session.finished == "one proposal"
    results = {e["name"]: e for e in session.events if e["type"] == "tool_result"}
    assert "ura_ppi" in results["list_registry"]["result"]
    assert results["search_datagovsg"]["result"].startswith("d_1 | Foreign Workforce")
    assert "2026 2Q" in results["preview_singstat_series"]["result"]
    errors = [e for e in session.events if e["type"] == "tool_result" and e["is_error"]]
    assert {e["name"] for e in errors} == {"propose_series", "no_such_tool"}
    assert ("tools", [("propose_series", True), ("no_such_tool", True)]) in backend.chat.received


def test_agent_step_budget_is_enforced(registry, tmp_path):
    turns = [Turn("", [ToolCall(str(i), "search_singstat", {"keyword": f"k{i}"})]) for i in range(10)]
    backend = ScriptedBackend(turns)
    session = run_agent("loop forever", backend, toolbox(registry, tmp_path), max_steps=3, log=lambda m: None)
    assert session.stop_reason == "step budget of 3 exhausted"
    model_calls = len(backend.chat.received)
    assert model_calls == 3 == sum(e["type"] == "assistant" for e in session.events)   # every reply recorded


class RecordingSession(ScriptedSession):
    """Also keeps the text of every tool result sent back to the model."""

    def __init__(self, turns):
        super().__init__(turns)
        self.contents = []

    def send_tool_results(self, results):
        self.contents.append([r.content for r in results])
        return super().send_tool_results(results)


def test_repeated_calls_are_answered_from_the_record_and_end_the_session(registry, tmp_path):
    """Seen live with qwen3.5:9b: the same two searches re-issued for nine steps until
    the context filled. A repeat is not run again but gets the earlier result back
    (a small model often re-reads a series just before proposing it), and three steps
    of nothing but repeats stop the session."""
    search = ToolCall("1", "search_singstat", {"keyword": "cement"})
    turns = [Turn("", [search]),
             Turn("", [search, ToolCall("2", "list_registry", {})]),     # one fresh call: not a repeat step
             Turn("", [search]),
             Turn("", [ToolCall("3", "search_singstat", {"keyword": "cement"})]),
             Turn("", [search]),
             Turn("", [ToolCall("4", "finish", {"summary": "never reached"})])]
    backend = ScriptedBackend([])
    backend.chat = RecordingSession(turns)
    box = toolbox(registry, tmp_path)
    session = run_agent("find predictors", backend, box, log=lambda m: None)
    assert session.stop_reason == "stopped: the model only repeated earlier calls for 3 steps"
    assert not session.model_failed and box.singstat.searches == ["cement"]    # searched once
    repeats = [e for e in session.events if e["type"] == "tool_result" and e["result"].startswith("[repeated call")]
    assert len(repeats) == 4 and not any(e["is_error"] for e in repeats)
    again = backend.chat.contents[1][0]
    assert "same result as step 1" in again and "M016082 | Personal Disposable Income" in again
    assert "propose it now with propose_series" in again


def test_the_model_is_told_to_wrap_up_near_the_budget(registry, tmp_path):
    turns = [Turn("", [ToolCall(str(i), "search_singstat", {"keyword": f"k{i}"})]) for i in range(6)]
    backend = ScriptedBackend([])
    backend.chat = RecordingSession(turns)
    session = run_agent("find predictors", backend, toolbox(registry, tmp_path), max_steps=6, log=lambda m: None)
    notes = [e for e in session.events if e["type"] == "workflow_note"]
    assert [e["step"] for e in notes] == [4, 5]                      # from two thirds of the budget on
    assert "2 model call(s) left" in backend.chat.contents[3][-1]
    assert "workflow note" not in backend.chat.contents[2][-1]


def test_a_long_session_ends_before_the_local_context_fills(registry, tmp_path):
    """Seen live: 29 steps of qwen3.5:9b reached ~23k tokens of history and the next
    reply was cut off at the 32k window. With a known window the session is told to
    wrap up, then ends before the request that would not fit; proposals are kept."""
    turns = [Turn("x" * 400, [ToolCall("1", "propose_series", PROPOSAL)])] + [
        Turn("y" * 400, [ToolCall(str(i), "search_singstat", {"keyword": f"k{i}"})]) for i in range(2, 30)]
    from sgmacro.agent.discover import CHARS_PER_TOKEN, SYSTEM

    backend = ScriptedBackend([])
    backend.chat = RecordingSession(turns)
    start = (len(SYSTEM) + len(json.dumps([t.__dict__ for t in TOOL_SPECS])) + len("find predictors")) / CHARS_PER_TOKEN
    backend.context_tokens = int(start / 0.5)  # the prompt alone fills half: a few steps fill the rest
    session = run_agent("find predictors", backend, toolbox(registry, tmp_path), log=lambda m: None)
    assert session.stop_reason.startswith("stopped: the conversation is near the model's context window")
    assert not session.model_failed and len(session.proposals) == 1
    notes = [e["text"] for e in session.events if e["type"] == "workflow_note"]
    assert notes and "close to the model's context limit" in notes[0]
    assert max(e["step"] for e in session.events) < 10


def test_the_context_check_uses_the_servers_token_count(registry, tmp_path):
    """Ollama reports the real prompt size; a character estimate undercounted a live
    session (72% estimated where the window was already full)."""
    turns = [Turn("", [ToolCall("1", "search_singstat", {"keyword": "a"})], context_used=10_000),
             Turn("", [ToolCall("2", "search_singstat", {"keyword": "b"})], context_used=20_000),
             Turn("", [ToolCall("3", "search_singstat", {"keyword": "c"})], context_used=25_000),
             Turn("", [ToolCall("4", "finish", {"summary": "never reached"})])]
    backend = ScriptedBackend([])
    backend.chat = RecordingSession(turns)
    backend.context_tokens = 32_768
    session = run_agent("find predictors", backend, toolbox(registry, tmp_path), log=lambda m: None)
    assert [e["step"] for e in session.events if e["type"] == "workflow_note"] == [2]   # 20k is past 60%
    assert session.stop_reason.startswith("stopped: the conversation is near")      # 25k is past 75%
    assert [e["context_used"] for e in session.events if e["type"] == "assistant"] == [10_000, 20_000, 25_000]


def test_extra_arguments_are_ignored_not_an_error(registry, tmp_path):
    box = toolbox(registry, tmp_path)
    text, is_error = box.call("finish", {"summary": "done", "proposals": ["x"]})
    assert not is_error and box.finished == "done"
    assert "ignored argument(s) this tool does not take: proposals" in text


def test_a_wide_table_names_its_row_labels_and_refuses_the_label_column(registry, tmp_path, monkeypatch):
    """Live, qwen3.5:9b proposed value_column "DataSeries" (the label column) for
    three wide tables, all rejected after the session. The description now lists
    the row labels, and such a proposal is refused at once so the model can fix it."""
    meta = {"data": {"name": "Construction Material Demand", "managedBy": "BCA",
                     "columnMetadata": {"order": ["DataSeries", "2026Jun", "2026May"], "map": {}}}}
    records = [{"_id": i, "DataSeries": label, "2026Jun": 1.0, "2026May": 2.0}
               for i, label in enumerate(["Cement", "Steel Reinforcement Bars", "Granite"])]
    asked = []
    monkeypatch.setattr("sgmacro.agent.tools.requests.get",
                        lambda *a, **kw: type("R", (), {"json": lambda self: meta})())
    box = toolbox(registry, tmp_path)
    box.http_get = lambda url, params=None, timeout=None: (asked.append(params), type(
        "R", (), {"json": lambda self: {"result": {"records": records}}})())[1]
    text, is_error = box.call("describe_datagovsg_dataset", {"dataset_id": "d_w"})
    assert not is_error and "layout: wide" in text and "e.g. 'Cement'" in text
    assert "'Steel Reinforcement Bars', 'Granite'" in text and asked[0]["limit"] == 40
    wide = {**PROPOSAL, "source": "datagovsg", "table_id": "", "series_no": "", "dataset_id": "d_w",
            "value_column": "DataSeries", "filters": "{}"}
    text, is_error = box.call("propose_series", wide)
    assert is_error and "row label" in text and box.proposals == []
    _, is_error = box.call("propose_series", {**wide, "value_column": "Cement"})
    assert not is_error and len(box.proposals) == 1


def test_proposals_beyond_the_briefs_limit_are_refused(registry, tmp_path):
    from sgmacro.agent.discover import BRIEF_LIMITS

    assert BRIEF_LIMITS == {"predictors": 4, "repair": 2}
    box = toolbox(registry, tmp_path)
    box.max_proposals = 1
    assert not box.call("propose_series", PROPOSAL)[1]
    text, is_error = box.call("propose_series", {**PROPOSAL, "table_id": "M222222"})
    assert is_error and "at most 1 proposal(s)" in text and "call finish" in text and len(box.proposals) == 1


def test_a_wide_data_gov_sg_table_is_described_briefly(registry, tmp_path, monkeypatch):
    columns = [f"col{i}" for i in range(332)]
    meta = {"data": {"name": "Construction Material Market Prices, Monthly", "managedBy": "SingStat",
                     "columnMetadata": {"order": columns, "map": {}}}}
    record = {"_id": 1, **{c: i for i, c in enumerate(columns)}}
    monkeypatch.setattr("sgmacro.agent.tools.requests.get",
                        lambda *a, **kw: type("R", (), {"json": lambda self: meta})())
    box = toolbox(registry, tmp_path)
    box.http_get = lambda *a, **kw: type("R", (), {"json": lambda self: {"result": {"records": [record] * 6}}})()
    text, is_error = box.call("describe_datagovsg_dataset", {"dataset_id": "d_x"})
    assert not is_error and text.count("(+317 more columns)") == 3      # three rows, fifteen columns each
    assert "columns (332)" in text and len(text) < 2500


def write_session(root, name, results, decisions=()):
    session = root / name
    session.mkdir(parents=True)
    (session / "verification.json").write_text(json.dumps({"results": results}), encoding="utf-8")
    if decisions:
        (session / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions), encoding="utf-8")
    return session


def test_candidates_turned_down_earlier_are_listed_and_refused(registry, tmp_path):
    """A rerun sees the same registry, so without this it proposes what a person
    already declined (or the verifier rejected) all over again."""
    from sgmacro.agent.discover import turned_down, with_turned_down

    cement = {**PROPOSAL, "label": "Cement price", "table_id": "M211251", "series_no": "1"}
    steel = {**PROPOSAL, "label": "Steel price", "table_id": "M211251", "series_no": "2"}
    deciles = {**PROPOSAL, "label": "Income deciles", "source": "datagovsg", "table_id": "", "series_no": "",
               "dataset_id": "d_e1", "value_column": "Dollar", "filters": '{"Dollar": "2025"}'}
    root = tmp_path / "discovery"
    write_session(root, "20261004T000000Z_predictors",
                  [{"verdict": "review", "proposal": cement}, {"verdict": "review", "proposal": steel},
                   {"verdict": "reject", "proposal": deciles,
                    "checks": [{"check": "layout", "severity": "fail", "message": "no period column"}]},
                   {"verdict": "accept", "proposal": PROPOSAL}],
                  [{"candidate": 1, "decision": "declined", "on": "2026-10-04"},
                   {"candidate": 2, "decision": "declined", "on": "2026-10-04"},
                   {"candidate": 2, "decision": "adopted", "on": "2026-10-04"}])     # changed their mind on 2
    (root / "20261004T010000Z_broken").mkdir()
    (root / "20261004T010000Z_broken" / "verification.json").write_text("{not json", encoding="utf-8")
    found = turned_down(root)
    assert {label.split(" (")[0]: why for label, why in found.values()} == {
        "Cement price": "declined by a person on 2026-10-04",
        "Income deciles": "rejected by the verifier (no period column)"}
    brief = with_turned_down("Find predictors.", found)
    assert "do not propose these again" in brief and "SingStat M211251 series 1" in brief
    assert with_turned_down("Find predictors.", {}) == "Find predictors."

    box = toolbox(registry, tmp_path)
    box.turned_down = {key: why for key, (_, why) in found.items()}
    text, is_error = box.call("propose_series", {**cement, "label": "Bulk cement, renamed"})
    assert is_error and "already declined by a person" in text and box.proposals == []
    _, is_error = box.call("propose_series", steel)                      # adopted, not turned down
    assert not is_error and len(box.proposals) == 1


# --- repeats, retries, recovery and the turned-down history -------------

class OllamaScript:
    """requests stand-in for a local Ollama that answers with scripted tool calls
    and records every request, so the real _OllamaSession (and its shortening of
    older results) runs under run_agent."""

    RequestException = __import__("requests").RequestException

    def __init__(self, calls):
        self.calls = list(calls)
        self.payloads = []

    def get(self, url, timeout=None):
        return type("R", (), {"raise_for_status": lambda self: None,
                              "json": lambda self: {"models": [{"name": "qwen3.5:9b"}]}})()

    def post(self, url, json=None, timeout=None):
        self.payloads.append(__import__("copy").deepcopy(json))      # the request as sent, not a live view
        name, args = self.calls.pop(0)
        body = {"message": {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": name, "arguments": args}}]}, "done_reason": "stop"}
        return type("R", (), {"status_code": 200, "json": lambda self: body})()


def test_a_shortened_result_can_be_read_again_in_full(registry, tmp_path, monkeypatch):
    """Older results are sent shortened with a note to call the tool
    again; that call must return the whole result (not its first 1,500 characters)
    and count as recovery, not as a repeat."""
    from sgmacro.llm import backends

    rows = [(str(i), f"Series number {i} with a long descriptive label") for i in range(1, 60)]
    meta = singstat_metadata({n: "Index" for n, _ in rows}, title="A wide SingStat table")
    record = json.loads(meta)
    for row, (n, label) in zip(record["Data"]["records"]["row"], rows):
        row["rowText"] = label
    record["Data"]["records"]["row"][-1]["rowText"] = "MARKER-SERIES at the very end"
    script = [("describe_singstat_table", {"table_id": "M999999"})] + \
        [("search_singstat", {"keyword": f"k{i}"}) for i in range(7)] + \
        [("describe_singstat_table", {"table_id": "M999999"}), ("finish", {"summary": "done"})]
    fake = OllamaScript(script)
    monkeypatch.setattr(backends, "requests", fake)
    box = toolbox(registry, tmp_path)
    box.singstat._metadata["M999999"] = json.dumps(record).encode()
    session = run_agent("find predictors", backends.OllamaBackend(model="qwen3.5:9b"), box, log=lambda m: None)
    first = next(e["result"] for e in session.events if e["type"] == "tool_result")
    assert first.index("MARKER-SERIES") > 1500                     # beyond what a repeat returns
    shortened = [m["content"] for m in fake.payloads[8]["messages"] if m["role"] == "tool"][0]
    assert "MARKER-SERIES" not in shortened and "call the tool again" in shortened
    reread = [e for e in session.events if e["type"] == "tool_result"][8]
    assert reread["result"].startswith("[earlier result from step 1, shown again in full")
    assert "MARKER-SERIES" in [m["content"] for m in fake.payloads[9]["messages"] if m["role"] == "tool"][-1]
    assert session.stop_reason == "finish tool called" and not reread["is_error"]


def test_calls_differing_only_in_ignored_fields_are_one_call(registry, tmp_path):
    """A tool drops fields it does not take, so the repeat check must
    key on the arguments it actually uses."""
    turns = [Turn("", [ToolCall(str(i), "search_singstat", {"keyword": "cement", "limit": i})]) for i in range(5)]
    backend = ScriptedBackend(turns)
    box = toolbox(registry, tmp_path)
    session = run_agent("find predictors", backend, box, max_steps=5, log=lambda m: None)
    assert box.singstat.searches == ["cement"]                      # fetched once
    assert session.stop_reason == "stopped: the model only repeated earlier calls for 3 steps"
    turns = [Turn("", [ToolCall("1", "search_singstat", {"keyword": "cement", "limit": 1})]),
             Turn("", [ToolCall("2", "search_singstat", {"keyword": "steel", "limit": 1})]),
             Turn("", [ToolCall("3", "finish", {"summary": "done"})])]
    box = toolbox(registry, tmp_path)
    run_agent("find predictors", ScriptedBackend(turns), box, log=lambda m: None)
    assert box.singstat.searches == ["cement", "steel"]             # a real change is a new call


def test_a_failed_call_is_retried_once_and_replayed_as_an_error(registry, tmp_path):
    """A failure was cached and then replayed as a success."""
    class Flaky(FakeSingStat):
        def __init__(self, failures):
            super().__init__({}, {})
            self.failures, self.attempts = failures, 0

        def search(self, keyword):
            self.attempts += 1
            if self.failures:
                self.failures -= 1
                raise RuntimeError("HTTP 503 temporarily unavailable")
            return super().search(keyword)

    def run(failures):
        box = toolbox(registry, tmp_path)
        box.singstat = Flaky(failures)
        turns = [Turn("", [ToolCall(str(i), "search_singstat", {"keyword": "rent"})]) for i in range(5)]
        session = run_agent("find", ScriptedBackend(turns), box, max_steps=5, log=lambda m: None)
        flags = [e["is_error"] for e in session.events if e["type"] == "tool_result"]
        return box.singstat.attempts, flags

    attempts, flags = run(failures=1)                                # transient: the retry succeeds
    assert attempts == 2 and flags[:3] == [True, False, False]
    attempts, flags = run(failures=99)                               # persistent: no endless retries
    assert attempts == 2 and all(flags)


def verified(proposal, verdict, fail=None):
    checks = [{"check": fail[0], "severity": "fail", "message": fail[1]}] if fail else []
    return {"verdict": verdict, "proposal": proposal, "checks": checks}


def test_only_lasting_rejections_persist_and_later_evidence_wins(tmp_path):
    """A failure to fetch is not a judgement on the series, and a later
    successful verification lifts an earlier verifier rejection; a person's
    decline stays until a person adopts the series."""
    from sgmacro.agent.discover import turned_down

    pdi = {**PROPOSAL}
    stale = {**PROPOSAL, "table_id": "M222222"}
    declined = {**PROPOSAL, "table_id": "M333333"}
    root = tmp_path / "discovery"
    write_session(root, "20261001T000000Z_a", [
        verified(pdi, "reject", ("resolve", "RuntimeError: HTTP 503 temporarily unavailable")),
        verified(stale, "reject", ("freshness", "latest observation 2019-Q4 is 27 periods behind")),
        verified(declined, "review")], [{"candidate": 3, "decision": "declined", "on": "2026-10-01"}])
    assert {k[1] for k in turned_down(root)} == {"M222222", "M333333"}   # the 503 is not lasting
    write_session(root, "20261002T000000Z_b", [verified(stale, "accept"), verified(declined, "accept")])
    assert {k[1] for k in turned_down(root)} == {"M333333"}              # verifier ban lifted, person's kept
    write_session(root, "20261003T000000Z_c", [verified(declined, "accept")],
                  [{"candidate": 1, "decision": "adopted", "on": "2026-10-03"}])
    assert turned_down(root) == {}


def test_equivalent_selections_of_one_series_share_an_identity(registry, tmp_path):
    """The identity follows the extractor: tidy filter values compare as
    text; a wide table is read by row label, whatever the filters say."""
    from sgmacro.agent.discover import turned_down
    from sgmacro.agent.tools import candidate_key

    tidy = {**PROPOSAL, "source": "datagovsg", "table_id": "", "series_no": "", "dataset_id": "d_t",
            "value_column": "value", "filters": '{"group": 1}'}
    assert candidate_key(tidy) == candidate_key({**tidy, "filters": '{"group": "1"}'})
    assert candidate_key(tidy) != candidate_key({**tidy, "filters": '{"group": "2"}'})
    wide = {**tidy, "dataset_id": "d_w", "value_column": "Housing loans", "filters": "{}"}
    root = tmp_path / "discovery"
    write_session(root, "20261001T000000Z_a",
                  [{**verified(wide, "review"), "draft": {"source": {"source": "datagovsg", "layout": "wide"}}},
                   verified(tidy, "review")],
                  [{"candidate": 1, "decision": "declined", "on": "2026-10-01"},
                   {"candidate": 2, "decision": "declined", "on": "2026-10-01"}])
    box = toolbox(registry, tmp_path)
    box.turned_down = {key: why for key, (_, why) in turned_down(root).items()}
    for again in ({**wide, "filters": '{"DataSeries": "Housing loans"}'}, {**tidy, "filters": '{"group": "1"}'}):
        text, is_error = box.call("propose_series", again)
        assert is_error and "already declined" in text
    _, is_error = box.call("propose_series", {**tidy, "filters": '{"group": "2"}'})
    assert not is_error and len(box.proposals) == 1                  # another group is another series


def test_a_damaged_history_entry_neither_drops_the_rest_nor_stops_discovery(tmp_path):
    """A half-written last line or a non-object line used to drop the
    session's other decisions, or crash discover before it started."""
    from sgmacro.agent.discover import turned_down

    root = tmp_path / "discovery"
    session = write_session(root, "20261001T000000Z_a", [verified(PROPOSAL, "review"),
                                                         verified({**PROPOSAL, "table_id": "M2"}, "review")],
                            [{"candidate": 1, "decision": "declined", "on": "2026-10-01"}])
    with (session / "decisions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('null\n["a list"]\n{"candidate": 2, ')                # junk, then a half-written line
    (root / "20261002T000000Z_b").mkdir()
    (root / "20261002T000000Z_b" / "verification.json").write_text('{"results": {"not": "a list"}}', encoding="utf-8")
    (root / "20261003T000000Z_c").mkdir()
    (root / "20261003T000000Z_c" / "verification.json").write_text(
        '{"results": [null, 3, {"proposal": 5}, {"proposal": {"source": "singstat"}, "verdict": "reject", "checks": 7}]}',
        encoding="utf-8")
    found = turned_down(root)
    assert [k[1] for k in found] == ["M016082"]                        # the complete decline survives


# --- event order, the verifier's own records, damaged fields ------------

class FakeDataGov:
    """data.gov.sg stand-in for the verifier: metadata and a CSV download per dataset."""

    def __init__(self, csv):
        self.csv = csv

    def metadata(self, dataset_id):
        return json.dumps({"name": f"Table {dataset_id}", "managedBy": "SingStat", "lastUpdatedAt": "2026-07-24",
                           "columnMetadata": {"order": [], "map": {}}}).encode()

    def download(self, dataset_id):
        return self.csv[dataset_id].encode()


def wide_csv(rows, quarters=80):
    """A wide quarterly table, newest column first, ending 2026 2Q."""
    labels = []
    year, quarter = 2026, 2
    for _ in range(quarters):
        labels.append(f"{year} {quarter}Q")
        year, quarter = (year, quarter - 1) if quarter > 1 else (year - 1, 4)
    lines = ["DataSeries," + ",".join(labels)]
    for name, filled in rows.items():
        lines.append(name + "," + ",".join(str(100 + i) if filled else "" for i in range(quarters)))
    return "\n".join(lines) + "\n"


def wide_proposal(**kw):
    return {**PROPOSAL, "source": "datagovsg", "table_id": "", "series_no": "", "dataset_id": "d_w",
            "value_column": "Housing loans", "filters": "{}", "frequency": "quarterly", **kw}


def as_session(root, name, results, verified_at, decisions=()):
    session = root / name
    session.mkdir(parents=True)
    (session / "verification.json").write_text(json.dumps({"verified_at": verified_at, "results": results}),
                                               encoding="utf-8")
    if decisions:
        (session / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions), encoding="utf-8")


def test_a_wrong_frequency_is_correctable_not_a_lasting_rejection(registry, tmp_path):
    """With the real verifier: a quarterly table proposed as monthly used
    to fail as `parse` (lasting) and block the corrected proposal. It is now a
    frequency failure, which a corrected proposal is free to fix."""
    from sgmacro.agent.discover import turned_down

    datagov = FakeDataGov({"d_w": wide_csv({"Housing loans": True})})
    wrong = verify_proposal(wide_proposal(frequency="monthly"), registry, date(2026, 10, 4),
                            singstat=FakeSingStat({}, {}), datagov=datagov)
    assert wrong.verdict == "reject" and [c.name for c in wrong.checks if c.severity == "fail"] == ["frequency"]
    root = tmp_path / "discovery"
    as_session(root, "20261004T000000Z_a", [wrong.to_json()], "2026-10-04T09:00:00+08:00")
    assert turned_down(root) == {}
    right = verify_proposal(wide_proposal(), registry, date(2026, 10, 4), singstat=FakeSingStat({}, {}),
                            datagov=datagov)
    assert right.verdict in ("accept", "review")


def test_a_wide_rows_identity_survives_a_rejection_without_a_draft(registry, tmp_path):
    """With the real verifier: a rejected wide row has no draft, so its
    layout used to be lost; an equivalent re-proposal got through and a later pass
    did not lift the ban. The layout is now recorded before any failure."""
    from sgmacro.agent.discover import turned_down

    empty = verify_proposal(wide_proposal(), registry, date(2026, 10, 4), singstat=FakeSingStat({}, {}),
                            datagov=FakeDataGov({"d_w": wide_csv({"Housing loans": False, "Other": True})}))
    assert empty.verdict == "reject" and empty.draft is None and empty.observed["layout"] == "wide"
    root = tmp_path / "discovery"
    as_session(root, "20261003T000000Z_a", [empty.to_json()], "2026-10-03T09:00:00+08:00")
    box = toolbox(registry, tmp_path)
    box.turned_down = {key: why for key, (_, why) in turned_down(root).items()}
    text, is_error = box.call("propose_series", wide_proposal(filters='{"DataSeries": "Housing loans"}'))
    assert is_error and "rejected by the verifier" in text
    later = verify_proposal(wide_proposal(), registry, date(2026, 10, 4), singstat=FakeSingStat({}, {}),
                            datagov=FakeDataGov({"d_w": wide_csv({"Housing loans": True})}))
    assert later.verdict in ("accept", "review")
    as_session(root, "20261004T000000Z_b", [later.to_json()], "2026-10-04T09:00:00+08:00")
    assert turned_down(root) == {}


def test_history_is_ordered_by_when_things_happened_not_by_folder(tmp_path):
    """`sgmacro verify` rewrites an old session's verification, and a
    decision can come later; the folder name is not the order of events."""
    from sgmacro.agent.discover import turned_down

    stale = {**PROPOSAL, "table_id": "M222222"}
    person = {**PROPOSAL, "table_id": "M333333"}
    root = tmp_path / "discovery"
    as_session(root, "20261001T000000Z_old", [verified(stale, "accept"), verified(person, "review")],
               "2026-10-04T10:00:00+08:00",                                   # re-verified last
               [{"candidate": 2, "decision": "adopted", "on": "2026-10-04", "at": "2026-10-04T11:00:00+08:00"}])
    as_session(root, "20261002T000000Z_new",
               [verified(stale, "reject", ("freshness", "stale on Oct 2")), verified(person, "review")],
               "2026-10-02T09:00:00+08:00",
               [{"candidate": 2, "decision": "declined", "on": "2026-10-03", "at": "2026-10-03T09:00:00+08:00"}])
    assert turned_down(root) == {}            # the later pass and the later adoption win
    as_session(root, "20261005T000000Z_after", [verified(person, "accept")], "2026-10-05T09:00:00+08:00",
               [{"candidate": 1, "decision": "declined", "on": "2026-10-05", "at": "2026-10-05T09:30:00+08:00"}])
    as_session(root, "20261006T000000Z_check", [verified(person, "accept")], "2026-10-06T09:00:00+08:00")
    assert [k[1] for k in turned_down(root)] == ["M333333"]   # a pass does not undo a person's decline


def test_bad_fields_and_a_cut_multibyte_line_are_skipped(tmp_path):
    """An unhashable candidate field crashed the scan, and a last line cut
    inside a multi-byte character made the whole log undecodable."""
    from sgmacro.agent.discover import turned_down

    root = tmp_path / "discovery"
    session = write_session(root, "20261001T000000Z_a", [verified(PROPOSAL, "review"),
                                                         verified({**PROPOSAL, "table_id": "M2"}, "review")],
                            [{"candidate": 1, "decision": "declined", "on": "2026-10-01"},
                             {"candidate": [], "decision": "declined"}, {"candidate": {}, "decision": "adopted"},
                             {"candidate": True, "decision": "adopted"}, {"candidate": "1", "decision": "adopted"}])
    with (session / "decisions.jsonl").open("ab") as fh:
        fh.write('{"candidate": 2, "label": "住宅'.encode("utf-8")[:-1])  # cut inside the last character
    assert [k[1] for k in turned_down(root)] == ["M016082"]


def test_new_records_carry_the_time_they_were_made(registry, tmp_path, monkeypatch):
    session, reg_copy = session_with(tmp_path, registry)
    monkeypatch.setattr("sgmacro.fetch.build_clients",
                        lambda: (FakeSingStat({"M016082": PDI_META}, {"M016082": PDI_DATA}), None))
    monkeypatch.chdir(ROOT_DIR)
    assert CliRunner().invoke(app, ["verify", str(session), "--registry", str(reg_copy)], input="n\n").exit_code == 0
    assert "T" in json.loads((session / "verification.json").read_text(encoding="utf-8"))["verified_at"]
    decision = json.loads((session / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert decision["decision"] == "declined" and decision["at"].endswith("+08:00")


# --- quoted CSV headers, decisions that keep their identity -------------

def test_a_quoted_header_still_reports_a_wrong_frequency(registry, tmp_path):
    """The frequency check split the header on commas, so a quoted
    period label ("2026 2Q") was not recognised and the mismatch failed as a
    lasting parse error. It now reads the header as CSV, like the extractor."""
    from sgmacro.agent.discover import turned_down

    quoted = "\n".join('"' + '","'.join(row.split(",")) + '"' for row in wide_csv({"Housing loans": True}).splitlines())
    datagov = FakeDataGov({"d_w": quoted + "\n"})
    wrong = verify_proposal(wide_proposal(frequency="monthly"), registry, date(2026, 10, 4),
                            singstat=FakeSingStat({}, {}), datagov=datagov)
    assert [c.name for c in wrong.checks if c.severity == "fail"] == ["frequency"]
    root = tmp_path / "discovery"
    as_session(root, "20261004T000000Z_a", [wrong.to_json()], "2026-10-04T09:00:00+08:00")
    box = toolbox(registry, tmp_path)
    box.turned_down = {key: why for key, (_, why) in turned_down(root).items()}
    right = verify_proposal(wide_proposal(), registry, date(2026, 10, 4), singstat=FakeSingStat({}, {}),
                            datagov=datagov)
    assert right.verdict in ("accept", "review") and not box.call("propose_series", wide_proposal())[1]
    # data that genuinely cannot be read is still a lasting rejection
    bad = wide_csv({"Housing loans": True}).replace("Housing loans,100,", "Housing loans,see note,")
    unreadable = verify_proposal(wide_proposal(), registry, date(2026, 10, 4), singstat=FakeSingStat({}, {}),
                                 datagov=FakeDataGov({"d_w": bad}))
    assert [c.name for c in unreadable.checks if c.severity == "fail"] == ["parse"]
    as_session(root, "20261004T010000Z_b", [unreadable.to_json()], "2026-10-04T10:00:00+08:00")
    assert len(turned_down(root)) == 1


def test_a_declined_wide_row_stays_declined_after_a_failed_reverification(registry, tmp_path, monkeypatch):
    """Through the CLI twice: a person declines a wide row; re-verifying
    the same session then fails to download (503), and the rewritten result has
    no layout. The decision keeps the identity confirmed when it was made."""
    from sgmacro.agent.discover import turned_down

    class Down(FakeDataGov):
        def download(self, dataset_id):
            raise ConnectionError("HTTP 503 temporarily unavailable")

    root = tmp_path / "discovery"
    session = root / "20261004T000000Z_predictors"
    session.mkdir(parents=True)
    (session / "session.json").write_text(json.dumps({"proposals": [wide_proposal()]}), encoding="utf-8")
    reg_copy = tmp_path / "sources.yaml"
    shutil.copy(REGISTRY_PATH, reg_copy)
    monkeypatch.chdir(ROOT_DIR)
    tables = {"d_w": wide_csv({"Housing loans": True, "Other": True})}
    monkeypatch.setattr("sgmacro.fetch.build_clients", lambda: (FakeSingStat({}, {}), FakeDataGov(tables)))
    first = CliRunner().invoke(app, ["verify", str(session), "--registry", str(reg_copy)], input="n\n")
    assert first.exit_code == 0, first.output
    monkeypatch.setattr("sgmacro.fetch.build_clients", lambda: (FakeSingStat({}, {}), Down(tables)))
    second = CliRunner().invoke(app, ["verify", str(session), "--registry", str(reg_copy), "--no-prompt"])
    assert second.exit_code == 0 and "resolve" in second.output
    box = toolbox(registry, tmp_path)
    box.turned_down = {key: why for key, (_, why) in turned_down(root).items()}
    text, is_error = box.call("propose_series", wide_proposal(filters='{"DataSeries": "Housing loans"}'))
    assert is_error and "declined by a person" in text
    _, is_error = box.call("propose_series", wide_proposal(value_column="Other"))   # another row is unaffected
    assert not is_error


def test_older_records_without_times_or_layouts_still_merge(tmp_path):
    """Guards for formats written before this round: date-only times order by date
    (then folder and line), and a ban keyed without a layout is lifted by a later
    pass of the same wide row."""
    from sgmacro.agent.discover import turned_down

    root = tmp_path / "discovery"
    stale = {**PROPOSAL, "table_id": "M222222"}
    for name, verified_on, verdict in (("20261001T000000Z_a", "2026-10-03", "accept"),
                                       ("20261002T000000Z_b", "2026-10-02", "reject")):
        session = root / name
        session.mkdir(parents=True)
        result = verified(stale, verdict, ("freshness", "stale") if verdict == "reject" else None)
        (session / "verification.json").write_text(json.dumps({"verified_on": verified_on, "results": [result]}),
                                                   encoding="utf-8")
    assert turned_down(root) == {}                     # the accept of Oct 3 is later than the reject of Oct 2
    legacy = wide_proposal(value_column="Housing loans")
    as_session(root, "20261003T000000Z_c", [verified(legacy, "reject", ("data", "no populated observations"))],
               "2026-10-03T09:00:00+08:00")            # no layout recorded: keyed with its filters
    assert len(turned_down(root)) == 1
    as_session(root, "20261004T000000Z_d", [{**verified(legacy, "review"), "observed": {"layout": "wide"}}],
               "2026-10-04T09:00:00+08:00")
    assert turned_down(root) == {}


def test_skipped_history_records_are_reported(tmp_path):
    from sgmacro.agent.discover import turned_down

    root = tmp_path / "discovery"
    write_session(root, "20261001T000000Z_a", [verified(PROPOSAL, "review")],
                  [{"candidate": 1, "decision": "declined", "on": "2026-10-01"}, {"candidate": [], "decision": "x"}])
    with (root / "20261001T000000Z_a" / "decisions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    (root / "20261002T000000Z_b").mkdir()
    (root / "20261002T000000Z_b" / "verification.json").write_text("{broken", encoding="utf-8")
    problems = []
    assert len(turned_down(root, problems)) == 1
    assert problems == ["20261001T000000Z_a/decisions.jsonl: 2 line(s) skipped",
                        "20261002T000000Z_b/verification.json is unreadable"]


def test_a_malformed_series_field_does_not_override_the_verified_identity(tmp_path):
    """Any list of strings was trusted as a stored identity, so an edited
    or damaged one could replace the identity the verification still gives."""
    from sgmacro.agent.discover import turned_down

    for bad in (["singstat"], ["not_a_source", "M016082", "1"], ["singstat", "M016082", "1", "extra"]):
        root = tmp_path / f"discovery_{len(bad)}_{bad[0]}"
        write_session(root, "20261001T000000Z_a", [verified(PROPOSAL, "review")],
                      [{"candidate": 1, "decision": "declined", "on": "2026-10-01", "series": bad},
                       {"candidate": 9, "decision": "declined", "on": "2026-10-01", "series": bad}])
        problems = []
        found = turned_down(root, problems)
        assert list(found) == [("singstat", "M016082", "1")], bad      # the verified identity, still declined
        assert problems == ["20261001T000000Z_a/decisions.jsonl: 1 line(s) skipped",
                            "20261001T000000Z_a/decisions.jsonl: 2 malformed series field(s) ignored"]


def test_skipped_verification_results_are_reported(tmp_path):
    """Results that are not candidate records were skipped silently."""
    from sgmacro.agent.discover import turned_down

    root = tmp_path / "discovery"
    write_session(root, "20261001T000000Z_a", [None, {"proposal": 7}, verified(PROPOSAL, "review")],
                  [{"candidate": 3, "decision": "declined", "on": "2026-10-01"}])
    problems = []
    assert list(turned_down(root, problems)) == [("singstat", "M016082", "1")]
    assert problems == ["20261001T000000Z_a/verification.json: 2 result(s) skipped"]


class FailingSession(ScriptedSession):
    """Answers the first request, then fails the way a crashed server would."""

    def send_tool_results(self, results):
        raise RuntimeError("HTTP 500: model runner stopped; header was Bearer sk-live-0123456789abcdefghij")


def test_a_failed_model_call_ends_the_session_and_keeps_its_steps(registry, tmp_path):
    backend = ScriptedBackend([])
    backend.chat = FailingSession([Turn("Checking.", [ToolCall("1", "list_registry", {}),
                                                      ToolCall("2", "propose_series", PROPOSAL)])])
    lines = []
    session = run_agent("find predictors", backend, toolbox(registry, tmp_path), log=lines.append)
    assert session.stop_reason.startswith("model call failed: RuntimeError: HTTP 500")
    assert session.model_failed and session.to_json()["model_failed"] is True
    assert "sk-live" not in session.stop_reason and "[key redacted]" in session.stop_reason
    assert len(session.proposals) == 1                            # the step before the failure counts
    assert [e["type"] for e in session.events] == ["assistant", "tool_result", "tool_result"]
    assert any("model call failed" in line for line in lines)


def test_discover_without_a_model_explains_how_to_connect_one(monkeypatch):
    from sgmacro.llm.backends import BackendUnavailable

    def nothing(preference):
        if preference == "ollama":
            raise BackendUnavailable("ollama: no Ollama server at http://localhost:11434")
        return None, ["anthropic: no ANTHROPIC_API_KEY in the environment", "openai: not configured"]

    monkeypatch.setattr("sgmacro.llm.backends.resolve_backend", nothing)
    monkeypatch.chdir(ROOT_DIR)
    for args in (["discover"], ["discover", "--backend", "ollama"]):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 2, result.output
        assert "SGMACRO_OPENAI_BASE_URL" in result.output and "ollama pull" in result.output
        assert "Traceback" not in result.output
    assert CliRunner().invoke(app, ["discover", "--backend", "gpt"]).exit_code == 2


def test_verifier_accepts_a_current_seasonally_adjusted_series(registry):
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": PDI_DATA})
    v = verify_proposal(PROPOSAL, registry, date(2026, 10, 2), singstat=fake)
    assert v.verdict == "accept", [c.message for c in v.checks]
    assert v.observed["latest"] == "2026-Q2"
    assert v.draft["source"] == {"source": "singstat", "table_id": "M016082", "series": "1",
                                 "unit_of_measure": "Million Dollars"}
    assert v.draft["publication_lag_days"].startswith("TODO")


def test_verifier_rejects_stale_series(registry):
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": STALE_DATA})
    v = verify_proposal(PROPOSAL, registry, date(2026, 10, 2), singstat=fake)
    assert v.verdict == "reject" and any(c.name == "freshness" and c.severity == "fail" for c in v.checks)


def test_verifier_rejects_wrong_frequency_and_missing_series(registry):
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": PDI_DATA})
    assert verify_proposal({**PROPOSAL, "frequency": "monthly"}, registry, date(2026, 10, 2),
                           singstat=fake).verdict == "reject"
    assert verify_proposal({**PROPOSAL, "series_no": "9"}, registry, date(2026, 10, 2),
                           singstat=fake).verdict == "reject"


def test_verifier_rejects_unknown_table(registry):
    fake = FakeSingStat({}, {})
    v = verify_proposal({**PROPOSAL, "table_id": "M999999"}, registry, date(2026, 10, 2), singstat=fake)
    assert v.verdict == "reject" and v.checks[-1].name == "resolve"


def test_verifier_rejects_a_registry_duplicate(registry):
    meta = singstat_metadata({"23": "Per Cent Per Annum"}, frequency="Monthly")
    fake = FakeSingStat({"M700071": meta}, {"M700071": singstat_payload({"23": [("2026 Aug", 1.2)]})})
    dup = {**PROPOSAL, "table_id": "M700071", "series_no": "23", "frequency": "monthly", "unit": "Per Cent Per Annum"}
    v = verify_proposal(dup, registry, date(2026, 10, 2), singstat=fake)
    assert any(c.name == "duplicate" and "sora_3m" in c.message for c in v.checks)
    assert v.verdict == "reject"


def test_duplicates_are_judged_by_series_not_by_file(registry):
    from sgmacro.agent.verify import _registry_duplicate

    rents = {"source": "datagovsg", "dataset_id": "d_8e4c50283fb7052a391dfb746a05c853", "value_column": "index"}
    same = {**rents, "filters": json.dumps({"property_type": "Non-Landed", "locality": "Whole Island"})}
    other = {**rents, "filters": json.dumps({"property_type": "Landed", "locality": "Whole Island"})}
    assert _registry_duplicate(registry, same) == "ura_rri_nonlanded"
    assert _registry_duplicate(registry, other) is None          # another property type in the same file
    component = {"source": "singstat", "table_id": "M400841", "series_no": "2.2"}
    assert _registry_duplicate(registry, component) is None      # one input of the vacancy ratio


def test_verifier_flags_a_mirror_of_a_registry_table(registry):
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": PDI_DATA})
    titles = {"personal disposable income at current prices, quarterly, seasonally adjusted": "some_series"}
    v = verify_proposal(PROPOSAL, registry, date(2026, 10, 2), singstat=fake, titles=titles)
    assert v.verdict == "review" and any("some_series" in c.message for c in v.checks)


# ---------------------------------------------------------------------------
# relationship check, adoption and the confirmation prompt
# ---------------------------------------------------------------------------

import math  # noqa: E402
import shutil  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from sgmacro.agent.adopt import AdoptError, adopt, insert_entry, retire_entry  # noqa: E402
from sgmacro.cli import app  # noqa: E402
from sgmacro.registry import load_registry  # noqa: E402
from tests.conftest import REGISTRY_PATH, make_spec  # noqa: E402


def wave(n, start, sign=1.0):
    """Quarterly level series whose YoY change is a smooth wave."""
    out = []
    level = 100.0
    for i in range(n):
        level *= 1 + sign * 0.01 * math.sin(i / 3.0)
        out.append((f"{start + i // 4} {i % 4 + 1}Q", round(level, 4)))
    return out


def target_values(n, start):
    return pd.Series([v for _, v in wave(n, start)], index=range(start * 4, start * 4 + n))


def test_relationship_check_reports_agreement(registry):
    data = singstat_payload({"1": wave(86, 2005)}, uom="Million Dollars")
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": data})
    target = make_spec(id="t", short_label="Target")
    v = verify_proposal(PROPOSAL, registry, date(2026, 10, 2), singstat=fake, targets=[(target, target_values(86, 2005))])
    check = next(c for c in v.checks if c.name == "relationship")
    assert check.severity == "pass" and "consistent with the declared effect" in check.message
    assert v.observed["relationship"][0]["agreement"] == "agrees"


def test_relationship_check_warns_when_data_contradict_the_claim(registry):
    data = singstat_payload({"1": wave(86, 2005, sign=-1.0)}, uom="Million Dollars")
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": data})
    target = make_spec(id="t", short_label="Target")
    v = verify_proposal(PROPOSAL, registry, date(2026, 10, 2), singstat=fake, targets=[(target, target_values(86, 2005))])
    check = next(c for c in v.checks if c.name == "relationship")
    assert check.severity == "warn" and "contradict" in check.message
    assert v.verdict == "review"   # informs the person; never rejects on its own


def test_relationship_check_warns_on_mixed_evidence(registry):
    """Live: an income series agreed with HDB prices but contradicted private prices
    and rents, and the check still passed (one agreement was enough), so the
    verdict was ACCEPT under a message that listed two disagreements."""
    data = singstat_payload({"1": wave(86, 2005)}, uom="Million Dollars")
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": data})
    same, opposite = make_spec(id="t1", short_label="Same"), make_spec(id="t2", short_label="Opposite")
    mirrored = 200 - target_values(86, 2005)
    v = verify_proposal(PROPOSAL, registry, date(2026, 10, 2), singstat=fake,
                        targets=[(same, target_values(86, 2005)), (opposite, mirrored)])
    check = next(c for c in v.checks if c.name == "relationship")
    assert [r["agreement"] for r in v.observed["relationship"]] == ["agrees", "disagrees"]
    assert check.severity == "warn" and check.message.startswith("mixed evidence")
    assert v.verdict == "review"


def test_a_registry_series_is_refused_when_proposed(registry, tmp_path):
    """Live: late in a session the agent proposed two series the registry already
    has (its listing had been shortened out of the context). The verifier rejected
    them, but they used up proposal slots; the tool now refuses them at once."""
    box = toolbox(registry, tmp_path)
    box.max_proposals = 1
    sora = {**PROPOSAL, "label": "SORA 3-Month", "table_id": "M700071", "series_no": "23"}
    text, is_error = box.call("propose_series", sora)
    assert is_error and "already in the registry as 'sora_3m'" in text and not box.proposals
    _, is_error = box.call("propose_series", {**sora, "replaces": "sora_3m"})     # a replacement may name it
    assert not is_error and len(box.proposals) == 1


def test_a_registry_series_is_refused_however_its_selector_is_written(registry, tmp_path):
    """The duplicate check compared selectors verbatim, while the extractor
    strips a wide row label and reads a SingStat series number as text. The same
    series written with blanks, or as a JSON number, passed as new (and the verifier
    let it through too), so four such variants used up every slot."""
    from sgmacro.agent.verify import _registry_duplicate

    row = "Compounded Singapore Overnight Rate Average (SORA) - 3 Month"
    sora_wide = wide_proposal(dataset_id="d_5fe5a4bb4a1ecc4d8a56a095832e2b24", value_column=row)
    sora = {**PROPOSAL, "table_id": "M700071", "series_no": "23"}
    ppi = wide_proposal(dataset_id="d_97f8a2e995022d311c6c68cfda6d034c", value_column="index",
                        filters='{"property_type": "All Residential"}')
    variants = [{**sora_wide, "value_column": " " * n + row + " " * n} for n in range(1, 5)]
    variants += [{**sora, "series_no": 23}, {**sora, "table_id": " m700071 "}, {**sora, "replaces": "sgs_10y"}, ppi]
    box = toolbox(registry, tmp_path)
    box.max_proposals = 4
    for proposal in variants:
        text, is_error = box.call("propose_series", proposal)
        assert is_error and "already in the registry as" in text, proposal
    assert not box.proposals
    assert {_registry_duplicate(registry, p) for p in variants} == {"sora_3m", "ura_ppi"}
    # the verifier reads the same helper: a blank-padded row is a duplicate there too
    datagov = FakeDataGov({"d_5fe5a4bb4a1ecc4d8a56a095832e2b24": wide_csv({row: True})})
    v = verify_proposal(variants[1], registry, date(2026, 10, 4), singstat=FakeSingStat({}, {}), datagov=datagov)
    assert any(c.name == "duplicate" and c.severity == "fail" for c in v.checks) and v.verdict == "reject"
    # every slot is still free for series the registry does not have
    allowed = [{**sora_wide, "value_column": row.replace("3 Month", "6 Month")}, {**sora, "series_no": "24"},
               {**ppi, "filters": '{"property_type": "Landed"}'}, {**sora, "replaces": "sora_3m"}]
    for proposal in allowed:
        text, is_error = box.call("propose_series", proposal)
        assert not is_error, text
    assert len(box.proposals) == 4


def test_registry_variants_in_one_model_turn_leave_every_slot_free(registry, tmp_path):
    """Through the agent loop: four padded spellings of a
    registry row sent in one model turn are all refused, and the next turn's four
    new proposals are all recorded."""
    row = "Compounded Singapore Overnight Rate Average (SORA) - 3 Month"
    sora_wide = wide_proposal(dataset_id="d_5fe5a4bb4a1ecc4d8a56a095832e2b24", frequency="monthly")
    padded = [ToolCall(f"p{n}", "propose_series", {**sora_wide, "value_column": " " * n + row + " " * n})
              for n in range(1, 5)]
    fresh = [ToolCall(f"n{i}", "propose_series", {**PROPOSAL, "series_no": str(i)}) for i in range(1, 5)]
    backend = ScriptedBackend([Turn("", padded), Turn("", fresh), Turn("", [ToolCall("f", "finish", {"summary": "4"})])])
    box = toolbox(registry, tmp_path)
    box.max_proposals = 4
    session = run_agent("find predictors", backend, box, log=lambda m: None)
    assert session.stop_reason == "finish tool called" and len(session.proposals) == 4
    assert backend.chat.received[1:3] == [("tools", [("propose_series", True)] * 4),
                                          ("tools", [("propose_series", False)] * 4)]


def test_tidy_columns_differing_only_in_blanks_stay_different_series(registry, tmp_path):
    """The history identity stripped a tidy column name, while
    the extractor matches tidy columns exactly, so declining `index` also blocked a
    different, readable column ` index `. Only a wide row label is stripped."""
    from sgmacro.agent.adopt import record_decline
    from sgmacro.agent.discover import turned_down
    from sgmacro.agent.tools import candidate_key, result_identity

    quarters = [(y, q) for y in range(2006, 2027) for q in range(1, 5)][2:-2]          # 2006-Q3 .. 2026-Q2
    table = "quarter,index, index \n" + "".join(f"{y}-Q{q},{100 + i},{1000 + i}\n" for i, (y, q) in enumerate(quarters))
    datagov = FakeDataGov({"d_two": table})
    plain = wide_proposal(dataset_id="d_two", value_column="index")
    padded = {**plain, "value_column": " index "}
    results = [verify_proposal(p, registry, date(2026, 10, 4), singstat=FakeSingStat({}, {}), datagov=datagov)
               for p in (plain, padded)]
    assert all(r.verdict != "reject" and r.observed["layout"] == "tidy" for r in results)
    assert [r.observed["latest_value"] for r in results] == [179.0, 1079.0]           # two different series
    keys = [result_identity(r.to_json()) for r in results]
    assert keys == [candidate_key(plain, "tidy"), candidate_key(padded, "tidy")] and keys[0] != keys[1]
    root = tmp_path / "discovery"
    as_session(root, "20261004T000000Z_a", [r.to_json() for r in results], "2026-10-04T09:00:00+08:00")
    record_decline(root / "20261004T000000Z_a", 1, "index", date(2026, 10, 4), series=keys[0])
    box = toolbox(registry, tmp_path)
    box.turned_down = {key: why for key, (_, why) in turned_down(root).items()}
    assert box.call("propose_series", plain)[1]                                        # the declined column
    assert not box.call("propose_series", padded)[1]                                   # the other one is free
    # a wide row label is still one series however it is padded
    assert len({candidate_key({**plain, "value_column": " " * n + "Housing loans"}, "wide") for n in range(4)}) == 1


def test_a_padded_series_number_selects_the_same_series_and_leaves_no_ban(registry, tmp_path):
    """The identity stripped a SingStat series number but the
    verifier looked it up as written, so ` 1 ` failed `exists` (a lasting failure)
    under the identity of `1`, and the correct proposal was then refused. The
    verifier now reads source identifiers as the identity does."""
    from sgmacro.agent.discover import turned_down

    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": PDI_DATA})
    plain = verify_proposal(PROPOSAL, registry, date(2026, 10, 2), singstat=fake)
    root = tmp_path / "discovery"
    for n, variant in enumerate(({"series_no": " 1 "}, {"series_no": 1}, {"table_id": " m016082 "})):
        v = verify_proposal({**PROPOSAL, **variant}, registry, date(2026, 10, 2), singstat=fake)
        assert v.verdict == plain.verdict != "reject" and v.observed == plain.observed, variant
        assert v.draft["source"] == plain.draft["source"], variant                    # the draft names M016082 / "1"
        as_session(root, f"20261002T00000{n}Z_a", [v.to_json()], "2026-10-02T09:00:00+08:00")
    box = toolbox(registry, tmp_path)
    box.turned_down = {key: why for key, (_, why) in turned_down(root).items()}
    assert not box.turned_down and not box.call("propose_series", PROPOSAL)[1]


def test_a_repair_of_a_quarantined_entry_is_adopted_without_retiring_it_again(tmp_path, monkeypatch):
    """The shipped repair session targets
    mas_housing_loans_discontinued, a negative control already quarantined, and
    accepting its candidate tried to retire it again ("could not find the role").
    The new entry is now added and the quarantined one keeps its expected status."""
    from sgmacro.registry import Role

    session = tmp_path / "repair"
    shutil.copytree(ROOT_DIR / "discovery" / "20261002T092208Z_repair_mas_housing_loans_discontinued", session)
    reg_copy = tmp_path / "sources.yaml"
    shutil.copy(REGISTRY_PATH, reg_copy)
    monkeypatch.chdir(ROOT_DIR)
    out = CliRunner().invoke(app, ["accept", str(session), "1", "--registry", str(reg_copy)])
    assert out.exit_code == 0, out.output
    assert "already quarantined" in out.output and " retired (" not in out.output
    registry = load_registry(reg_copy)
    old = next(s for s in registry.series if s.id == "mas_housing_loans_discontinued")
    assert old.role is Role.QUARANTINE and old.expected_status == "rejected:discontinued"
    assert len(registry.series) == len(load_registry(REGISTRY_PATH).series) + 1
    decision = json.loads((session / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert decision["decision"] == "adopted" and decision["retired"] is None


def test_a_padded_replaces_means_the_same_registry_entry_when_adopted(registry, tmp_path):
    """The tool and the verifier read `replaces` without blanks
    and let a replacement of sora_3m through, but adoption compared it as written
    and refused the same candidate as a duplicate of sora_3m."""
    months = [f"{y} {m}" for y in range(2005, 2027) for m in
              ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")][9:-4]
    fake = FakeSingStat({"M700071": singstat_metadata({"23": "Per Cent Per Annum"}, frequency="Monthly")},
                        {"M700071": singstat_payload({"23": [(m, 2.0) for m in months]}, uom="Per Cent Per Annum",
                                                     frequency="Monthly")})
    sora = {**PROPOSAL, "label": "SORA 3M", "table_id": "M700071", "series_no": "23", "frequency": "monthly",
            "unit": "Per Cent Per Annum", "channel": "Mortgage financing cost", "replaces": " sora_3m "}
    v = verify_proposal(sora, registry, date(2026, 10, 4), singstat=fake)
    assert v.proposal["replaces"] == "sora_3m" and v.verdict != "reject"
    assert not any(c.name == "duplicate" for c in v.checks)
    old_format = {**v.to_json(), "proposal": {**v.to_json()["proposal"], "replaces": " sora_3m "}}
    for name, result in (("now", v.to_json()), ("older", old_format)):    # as recorded now, and by earlier versions
        session = tmp_path / name
        session.mkdir()
        (session / "verification.json").write_text(json.dumps({"results": [result]}), encoding="utf-8")
        reg_copy = tmp_path / f"{name}.yaml"
        shutil.copy(REGISTRY_PATH, reg_copy)
        new_id = adopt(session, 1, reg_copy, today=date(2026, 10, 4))
        ids = {s.id for s in load_registry(reg_copy).series}
        assert new_id in ids and "sora_3m" in ids, name                 # kept alongside: it still validates


def session_with(tmp_path, registry, proposal=PROPOSAL, data=PDI_DATA):
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": data})
    v = verify_proposal(proposal, registry, date(2026, 10, 2), singstat=fake)
    v.observed["observed_lag_days"] = 42
    session = tmp_path / "20261002T000000Z_predictors"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"proposals": [proposal]}), encoding="utf-8")
    (session / "verification.json").write_text(json.dumps({"results": [v.to_json()]}), encoding="utf-8")
    reg_copy = tmp_path / "sources.yaml"
    shutil.copy(REGISTRY_PATH, reg_copy)
    return session, reg_copy


def test_adopt_writes_a_valid_entry_above_the_negative_controls(registry, tmp_path):
    session, reg_copy = session_with(tmp_path, registry)
    new_id = adopt(session, 1, reg_copy, today=date(2026, 10, 2))
    updated = load_registry(reg_copy)
    spec = updated.get(new_id)
    assert spec.role.value == "predictor" and spec.publication_lag_days == 47   # 42 observed + 5
    assert spec.source.table_id == "M016082" and spec.neutral_band == 1.0
    text = reg_copy.read_text(encoding="utf-8")
    assert text.index(f"- id: {new_id}") < text.index("# Negative controls")
    assert "# Adopted from discovery session" in text and "# Source registry" in text   # comments kept
    assert json.loads((session / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])["decision"] == "adopted"
    with pytest.raises(AdoptError, match="already in the registry"):
        adopt(session, 1, reg_copy, today=date(2026, 10, 2))


def test_adopting_a_replacement_retires_the_broken_entry(registry, tmp_path):
    proposal = {**PROPOSAL, "replaces": "bank_housing_loans"}
    session, reg_copy = session_with(tmp_path, registry, proposal)
    new_id = adopt(session, 1, reg_copy, today=date(2026, 10, 2), retired_status="rejected:discontinued")
    updated = load_registry(reg_copy)
    assert updated.get("bank_housing_loans").role.value == "quarantine"
    assert updated.get("bank_housing_loans").expected_status == "rejected:discontinued"
    assert updated.get(new_id).role.value == "predictor"


def test_rejected_candidates_cannot_be_adopted(registry, tmp_path):
    session, reg_copy = session_with(tmp_path, registry, data=STALE_DATA)
    with pytest.raises(AdoptError, match="rejected"):
        adopt(session, 1, reg_copy, today=date(2026, 10, 2))


def test_invalid_entry_leaves_the_registry_untouched(tmp_path):
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    broken = insert_entry(text, {"id": "broken", "label": "x"}, "test")
    assert "- id: broken" in broken
    with pytest.raises(AdoptError):
        retire_entry(text, "no_such_series", "rejected:x")


def test_accept_command_and_prompt(registry, tmp_path, monkeypatch):
    session, reg_copy = session_with(tmp_path, registry)
    runner = CliRunner()
    monkeypatch.chdir(ROOT_DIR)
    result = runner.invoke(app, ["accept", str(session), "1", "--registry", str(reg_copy)])
    assert result.exit_code == 0, result.output
    assert "personal_disposable_income" in reg_copy.read_text(encoding="utf-8")
    again = runner.invoke(app, ["accept", str(session), "9", "--registry", str(reg_copy)])
    assert again.exit_code != 0


def test_verify_prompts_per_candidate_and_records_decisions(registry, tmp_path, monkeypatch):
    session, reg_copy = session_with(tmp_path, registry)
    fake = FakeSingStat({"M016082": PDI_META}, {"M016082": PDI_DATA})
    monkeypatch.setattr("sgmacro.fetch.build_clients", lambda: (fake, None))
    monkeypatch.chdir(ROOT_DIR)
    runner = CliRunner()
    declined = runner.invoke(app, ["verify", str(session), "--registry", str(reg_copy)], input="n\n")
    assert declined.exit_code == 0, declined.output
    assert "relationship" in declined.output and "personal_disposable_income" not in reg_copy.read_text(encoding="utf-8")
    accepted = runner.invoke(app, ["verify", str(session), "--registry", str(reg_copy)], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert "personal_disposable_income" in reg_copy.read_text(encoding="utf-8")
    decisions = [json.loads(x)["decision"] for x in (session / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert decisions == ["declined", "adopted"]


def test_verify_asks_for_an_annual_candidates_month_and_adopts_it(registry, tmp_path, monkeypatch):
    """The interactive path for an annual candidate, end to end through the CLI: the
    month is asked before verifying (an invalid answer is asked again), then adopted."""
    meta = singstat_metadata({"1": "Number"}, frequency="Annual", title="Employed Residents, Annual")
    data = singstat_payload({"1": [(str(y), 1000 + y) for y in range(1990, 2026)]}, uom="Number",
                            frequency="Annual")
    proposal = {**PROPOSAL, "label": "Employed residents", "table_id": "M555555", "frequency": "annual",
                "unit": "Number"}
    session = tmp_path / "20261002T000000Z_predictors"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"proposals": [proposal]}), encoding="utf-8")
    reg_copy = tmp_path / "sources.yaml"
    shutil.copy(REGISTRY_PATH, reg_copy)
    monkeypatch.setattr("sgmacro.fetch.build_clients",
                        lambda: (FakeSingStat({"M555555": meta}, {"M555555": data}), None))
    monkeypatch.chdir(ROOT_DIR)
    result = CliRunner().invoke(app, ["verify", str(session), "--registry", str(reg_copy)], input="13\n6\ny\n")
    assert result.exit_code == 0, result.output
    assert "month number from 1 to 12" in result.output              # 13 was asked again
    assert "saved: " in result.output and "decisions recorded in" in result.output
    assert load_registry(reg_copy).get("employed_residents").reference_month == 6


def test_a_percent_sign_unit_is_a_rate_changed_in_points(registry):
    from sgmacro.agent.verify import draft_entry
    from sgmacro.registry import Frequency

    draft = draft_entry({**PROPOSAL, "unit": "%"}, Frequency.QUARTERLY, {"unit": "%"}, {"min": 1.0, "max": 4.0})
    assert draft["unit_kind"] == "rate" and draft["transforms"] == ["qoq_pp", "yoy_pp"]


def test_an_annual_candidate_needs_its_reference_month(registry, tmp_path):
    meta = singstat_metadata({"1": "Number"}, frequency="Annual", title="Employed Residents, Annual")
    data = singstat_payload({"1": [(str(y), 1000 + y) for y in range(1990, 2026)]}, uom="Number",
                            frequency="Annual")
    fake = FakeSingStat({"M555555": meta}, {"M555555": data})
    proposal = {**PROPOSAL, "label": "Employed residents", "table_id": "M555555", "frequency": "annual",
                "unit": "Number"}
    v = verify_proposal(proposal, registry, date(2026, 10, 2), singstat=fake)
    assert v.draft["reference_month"].startswith("TODO")
    session = tmp_path / "20261002T000000Z_predictors"
    session.mkdir()
    (session / "verification.json").write_text(json.dumps({"results": [v.to_json()]}), encoding="utf-8")
    reg_copy = tmp_path / "sources.yaml"
    shutil.copy(REGISTRY_PATH, reg_copy)
    with pytest.raises(AdoptError, match="reference month"):
        adopt(session, 1, reg_copy, today=date(2026, 10, 2))
    new_id = adopt(session, 1, reg_copy, today=date(2026, 10, 2), reference_month=6)
    spec = load_registry(reg_copy).get(new_id)
    assert spec.reference_month == 6
    # The lag is measured from the same end-June date the gate will use: update stamp
    # 2026-07-24 minus 2025-06-30 is 389 days, plus the 5-day margin. Measured from
    # end-December it would have been 210 days, and the gate would open six months early.
    assert spec.publication_lag_days == 394 and "re-run `sgmacro verify`" in spec.limitations


def test_a_draft_recorded_before_the_month_was_asked_for_still_needs_it(registry, tmp_path):
    """Sessions verified before the reference-month check have annual drafts with
    no such field; adopting one must still require the month."""
    meta = singstat_metadata({"1": "Number"}, frequency="Annual", title="Employed Residents, Annual")
    data = singstat_payload({"1": [(str(y), 1000 + y) for y in range(1990, 2026)]}, uom="Number",
                            frequency="Annual")
    proposal = {**PROPOSAL, "label": "Employed residents", "table_id": "M555555", "frequency": "annual",
                "unit": "Number"}
    v = verify_proposal(proposal, registry, date(2026, 10, 2), singstat=FakeSingStat({"M555555": meta},
                                                                                     {"M555555": data}))
    old = v.to_json()
    del old["draft"]["reference_month"]
    session = tmp_path / "20261002T000000Z_predictors"
    session.mkdir()
    (session / "verification.json").write_text(json.dumps({"results": [old]}), encoding="utf-8")
    reg_copy = tmp_path / "sources.yaml"
    shutil.copy(REGISTRY_PATH, reg_copy)
    with pytest.raises(AdoptError, match="reference month"):
        adopt(session, 1, reg_copy, today=date(2026, 10, 2))
    new_id = adopt(session, 1, reg_copy, today=date(2026, 10, 2), reference_month=6)
    assert load_registry(reg_copy).get(new_id).reference_month == 6


def test_an_annual_candidate_sits_in_the_quarter_of_its_reference_date():
    from sgmacro.agent.verify import candidate_changes
    from sgmacro.parse import extract
    from sgmacro.registry import Frequency, SingStatRef

    data = singstat_payload({"1": [(str(y), 100.0 + y - 2000) for y in range(2000, 2026)]}, uom="Number",
                            frequency="Annual")
    extracted = extract(SingStatRef(table_id="M1", series="1", unit_of_measure="Number"), data, Frequency.ANNUAL)
    june = candidate_changes(extracted, Frequency.ANNUAL, "Number", reference_month=6)
    december = candidate_changes(extracted, Frequency.ANNUAL, "Number")
    assert june.index.max() % 4 == 1 and december.index.max() % 4 == 3     # Q2 vs Q4 of 2025


@pytest.mark.parametrize("credential", ["sk-test-0123456789abcdefghijklmnop",
                                        "ordinary-looking-test-secret", "Bearer abcdefgh12345678"])
@pytest.mark.parametrize("near_cutoff", [False, True])
def test_agent_logs_and_saved_session_redact_before_truncating(tmp_path, monkeypatch, credential, near_cutoff):
    from sgmacro.agent.discover import save_session
    from sgmacro.store import assert_no_credentials

    monkeypatch.setenv("SGMACRO_OPENAI_API_KEY", "ordinary-looking-test-secret")
    argument = ("x" * 107 + " " if near_cutoff else "") + credential
    result = ("x" * 93 + " " if near_cutoff else "") + credential
    text = ("x" * 149 + " " if near_cutoff else "") + credential

    class EchoToolbox:
        finished = None
        proposals = []

        def call(self, name, arguments):
            assert arguments["query"] == argument        # logging must not alter the tool's input
            return result, False

    backend = ScriptedBackend([Turn(text, [ToolCall("c1", "echo", {"query": argument})]), Turn("Finished.")])
    lines = []
    session = run_agent("find predictors", backend, EchoToolbox(), log=lines.append)
    assert len(lines) == 3 and not session.model_failed
    assert credential[:10] not in "\n".join(lines)         # no leaked prefix at the display cutoff
    assert_no_credentials("\n".join(lines))
    saved = save_session(session, tmp_path, "test")
    blob = (saved / "session.json").read_text(encoding="utf-8")
    assert_no_credentials(blob)
    assert "[key redacted]" in blob and len(json.loads(blob)["events"]) == 3


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize("ending", ["empty", "length", "truncated_tools", "normal"])
def test_discover_exit_code_and_saved_steps_follow_model_outcome(tmp_path, monkeypatch, provider, ending):
    from sgmacro.llm import backends
    from tests.test_backends import FakeHTTP, FakeRequests, reply

    calls = [{"id": "c1", "type": "function", "function": {"name": "list_registry", "arguments": "{}"}},
             {"id": "c2", "type": "function", "function": {
                 "name": "propose_series", "arguments": json.dumps(PROPOSAL)}}]
    first = {"role": "assistant", "content": "Checking.", "tool_calls": calls}
    last = {"role": "assistant", "content": "Finished." if ending == "normal" else ""}
    finish = "length" if ending in ("length", "truncated_tools") else "stop"
    if ending == "truncated_tools":
        last["tool_calls"] = [{"function": {"name": "finish", "arguments": '{"summary":'}}]
    if provider == "openai":
        monkeypatch.setenv("SGMACRO_OPENAI_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("SGMACRO_OPENAI_MODEL", "m1")
        fake = FakeHTTP(replies=[reply(first, "tool_calls"), reply(last, finish)])
    else:
        monkeypatch.setenv("SGMACRO_OLLAMA_URL", "http://localhost:11434")
        monkeypatch.setenv("SGMACRO_OLLAMA_MODEL", backends.OLLAMA_MODEL)
        fake = FakeRequests([backends.OLLAMA_MODEL], [
            {"message": first, "done_reason": "stop"}, {"message": last, "done_reason": finish}])
    monkeypatch.setattr(backends, "requests", fake)
    monkeypatch.chdir(ROOT_DIR)
    result = CliRunner().invoke(app, ["discover", "--backend", provider, "--no-verify", "--no-prompt",
                                      "--out", str(tmp_path)])
    assert result.exit_code == (0 if ending == "normal" else 1), result.output
    files = list(tmp_path.glob("*/session.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["model_failed"] is (ending != "normal")
    assert len(saved["proposals"]) == 1
    assert [e["name"] for e in saved["events"] if e["type"] == "tool_result"] == ["list_registry", "propose_series"]
    if ending != "normal":
        assert "no tool call" not in saved["stop_reason"]
        assert ("finish_reason=length" if finish == "length" else "no content") in saved["stop_reason"]
