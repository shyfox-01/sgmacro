"""The LLM claim checker, on the failure modes actually observed from local models
and those found in review."""

from __future__ import annotations

import pytest

from sgmacro.llm.claims import check_summary
from sgmacro.llm.summary import to_claims

SHEET = {
    "F1": {"kind": "change", "series": "Private home prices (URA PPI)", "period": "2026 Q2", "value": "+2.9%",
           "direction": "up"},
    "F2": {"kind": "level", "series": "Private home prices (URA PPI)", "period": "2026 Q2", "value": "219.4"},
    "F7": {"kind": "change", "series": "3M compounded SORA", "period": "Aug 2026", "value": "-38 bp",
           "direction": "down", "signals": {"sales": "S1", "rents": "S2"}},
    "F8": {"kind": "change", "series": "SGS 10Y yield", "period": "Aug 2026", "value": "+52 bp",
           "direction": "up", "signals": {"sales": "S3", "rents": "S4"}},
    "F9": {"kind": "change", "series": "Bank housing loans", "period": "Aug 2026", "value": "+7.0%",
           "direction": "up", "signals": {"sales": "S5"}},
    "F10": {"kind": "change", "series": "CPI all items", "period": "Aug 2026", "value": "+2.3%",
            "direction": "up", "signals": {"sales": "S6"}},
    "F11": {"kind": "change", "series": "CPI accommodation", "period": "Aug 2026", "value": "+0.8%",
            "direction": "up", "signals": {"sales": "S7"}},
    "S1": {"kind": "signal", "series": "3M compounded SORA", "market": "sales", "reading": "supportive", "move": "F7"},
    "S2": {"kind": "signal", "series": "3M compounded SORA", "market": "rents", "reading": "ambiguous", "move": "F7"},
    "S3": {"kind": "signal", "series": "SGS 10Y yield", "market": "sales", "reading": "headwind", "move": "F8"},
    "S4": {"kind": "signal", "series": "SGS 10Y yield", "market": "rents", "reading": "ambiguous", "move": "F8"},
    "S5": {"kind": "signal", "series": "Bank housing loans", "market": "sales", "reading": "supportive", "move": "F9"},
    "S6": {"kind": "signal", "series": "CPI all items", "market": "sales", "reading": "ambiguous", "move": "F10"},
    "S7": {"kind": "signal", "series": "CPI accommodation", "market": "sales", "reading": "supportive",
           "move": "F11"},
}


def check(text, facts, stance, market="sales"):
    return check_summary({market: [{"text": text, "facts": facts, "stance": stance}]}, SHEET)[market][0]


def test_well_formed_claim_passes():
    c = check("The 3M compounded SORA fell, which supports sales through cheaper mortgages.", ["S1", "F7"],
              "supportive")
    assert c.ok, c.violations


def test_target_context_claim_passes():
    assert check("Private home prices rose year on year.", ["F1", "F2"], "context").ok


@pytest.mark.parametrize("text", [
    "Private home prices rose 2.9% in the quarter.",      # even a correctly copied figure
    "Private home prices rose 999PERCENT.",                # glued to letters, so no "unit" to hide behind
    "Private home prices rose in 2026.",                   # dates too: the cited period is printed
])
def test_the_text_carries_no_figures(text):
    c = check(text, ["F1"], "context")
    assert any("contains figures" in v for v in c.violations)


def test_tenors_inside_names_are_not_figures():
    assert check("The SGS 10Y yield rose, a headwind for sales.", ["S3", "F8"], "headwind").ok
    assert check("The 10-year yield rose, a headwind for sales.", ["S3", "F8"], "headwind").ok


def test_citing_only_the_signal_still_checks_the_move():
    """Review case: the signal alone satisfied the naming and stance checks, and the
    wrong direction slipped through because a signal carries no direction itself."""
    c = check("The 3M compounded SORA rose, supporting sales through cheaper mortgages.", ["S1"], "supportive")
    assert any("rose" in v and "down" in v for v in c.violations)
    assert check("The 3M compounded SORA fell, supporting sales through cheaper mortgages.", ["S1"],
                 "supportive").ok


def test_a_named_series_must_be_cited():
    """Review case: "home prices rose, supporting sales" leaning on the bank-loan figure."""
    c = check("Private home prices rose, supporting sales.", ["F9", "S5"], "supportive")
    assert any("names 'Private home prices (URA PPI)' but cites none" in v for v in c.violations)


def test_mixed_directions_are_checked_clause_by_clause():
    """Review case: the moves swapped between two series; the whole-claim check
    abstains (one up, one down), the clause check does not."""
    c = check("SORA rose while the SGS yield fell.", ["F7", "F8", "S2", "S4"], "ambiguous", market="rents")
    assert any("says 3M compounded SORA rose" in v for v in c.violations)
    assert any("says SGS 10Y yield fell" in v for v in c.violations)
    assert check("SORA fell while the SGS yield rose.", ["F7", "F8", "S2", "S4"], "ambiguous", market="rents").ok


def test_a_sentence_end_closes_a_clause():
    """Observed from qwen3.5:27b: "...real price growth also advanced. HDB resale prices held
    steady..." was read as one clause, so "growth" was pinned on HDB. Correct claim."""
    sheet_text = ("Private home prices rose while real growth also advanced. HDB resale prices held steady but "
                  "declined in real terms.")
    hdb = {"kind": "real_change", "series": "HDB resale prices (HDB RPI)", "period": "2026 Q2", "value": "-1.9%",
           "direction": "down"}
    sheet = {**SHEET, "F20": hdb}
    c = check_summary({"sales": [{"text": sheet_text, "facts": ["F1", "F20"], "stance": "context"}]}, sheet)
    assert c["sales"][0].ok, c["sales"][0].violations


def test_an_acronym_shared_by_two_series_names_neither():
    """'CPI' belongs to CPI all items and CPI accommodation, so it is not read as
    naming the one that is not cited."""
    assert check("CPI accommodation rose, which supports sales.", ["F11", "S7"], "supportive").ok


def test_direction_contradicting_the_data_is_rejected():
    c = check("The 3M compounded SORA rose, which supports sales.", ["S1", "F7"], "supportive")
    assert any("rose" in v and "down" in v for v in c.violations)


def test_observed_qwen_error_rate_cut_called_a_headwind_is_rejected():
    """qwen3.5:9b wrote: 'SORA fell by 38 bp, raising mortgage financing costs that weigh on sales'
    and filed it under headwinds."""
    c = check("3M compounded SORA fell, raising mortgage financing costs that weigh on sales.",
              ["F7", "F8"], "headwind")
    assert any("mixes signal readings" in v or "contradicts" in v for v in c.violations)


def test_observed_qwen_error_rate_effect_on_rents_is_ambiguous_not_a_headwind():
    c = check("SGS 10Y yields rose, acting as a headwind for rents.", ["F8"], "headwind", market="rents")
    assert any("contradicts the computed reading 'ambiguous'" in v for v in c.violations)


def test_predictor_without_a_signal_in_this_market_is_rejected():
    c = check("Bank housing loans rose, supporting rents.", ["F9"], "supportive", market="rents")
    assert any("no rents signal" in v for v in c.violations)


def test_stance_without_any_signal_is_rejected():
    c = check("Prices are supported.", ["F1"], "supportive")
    assert any("without citing a signal" in v for v in c.violations)


def test_supportive_wording_under_a_headwind_stance_is_rejected():
    c = check("The 10-year yield rose, which supports sales.", ["S3", "F8"], "headwind")
    assert any("wording implies 'supportive'" in v for v in c.violations)


def test_forecast_language_is_rejected():
    c = check("Prices will rise next year as SORA fell.", ["S1", "F7"], "supportive")
    assert any("forecast" in v for v in c.violations)


def test_unknown_ids_and_signal_from_other_market():
    c = check("SORA fell.", ["S9", "S2", "F7"], "ambiguous")
    assert any("unknown fact ids" in v for v in c.violations)
    assert any("'rents'" in v and "under the sales summary" in v for v in c.violations)


def test_structured_output_is_flattened_with_explicit_stances():
    payload = {"sales": {"lead": {"text": "t", "facts": ["F1"]},
                         "points": [{"reading": "supportive", "text": "u", "facts": ["S1"]}]},
               "rents": {"lead": {"text": "v", "facts": []}, "points": []}}
    claims = to_claims(payload)
    assert [c["stance"] for c in claims["sales"]] == ["context", "supportive"]
    assert len(claims["rents"]) == 1


@pytest.mark.parametrize("payload", [
    [1],
    {"sales": {"points": 1}, "rents": {}},
    {"sales": {"lead": {"text": "t", "facts": 5}, "points": []}, "rents": {"lead": {"text": "v", "facts": []},
                                                                           "points": []}},
    {"sales": {"lead": "t", "points": []}, "rents": {"lead": {"text": "v", "facts": []}, "points": []}},
])
def test_valid_json_of_the_wrong_shape_is_refused(payload):
    with pytest.raises(ValueError):
        to_claims(payload)
