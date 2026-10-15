"""The registry is the contract; a bad entry must fail at load time."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sgmacro.registry import Effect, Role, Transform, load_registry
from tests.conftest import PREDICTOR_FIELDS, REGISTRY_PATH, make_spec, rate_spec


def test_committed_registry_loads(registry):
    assert len(registry.targets) == 3
    assert len(registry.predictors) == 13
    assert {s.expected_status for s in registry.quarantine} == {"rejected:discontinued",
                                                               "rejected:contract_mismatch"}


def test_every_market_has_a_target_and_predictors(registry):
    for market in ("sales", "rents"):
        assert any(t.market.value == market for t in registry.targets)
        assert sum(market in [m.value for m in p.markets] for p in registry.predictors) >= 5


def test_percentage_change_of_a_rate_is_rejected():
    with pytest.raises(ValidationError, match="never percentage changes"):
        rate_spec(transforms=["yoy"], headline="yoy")


def test_ratio_cannot_take_basis_points():
    with pytest.raises(ValidationError, match="forbids"):
        make_spec(unit_kind="ratio", transforms=["yoy_bp"], headline="yoy_bp")


def test_month_on_month_on_a_quarterly_series_is_rejected():
    with pytest.raises(ValidationError, match="frequency=quarterly forbids"):
        make_spec(transforms=["mom", "yoy"])


def test_trailing_four_quarter_change_needs_quarterly_data():
    with pytest.raises(ValidationError, match="t4q_yoy"):
        make_spec(unit_kind="level", frequency="monthly", transforms=["t4q_yoy"], headline="t4q_yoy")


def test_headline_must_be_computed():
    with pytest.raises(ValidationError, match="headline"):
        make_spec(transforms=["qoq"], headline="yoy")


def test_predictor_must_declare_effect_for_each_market():
    fields = {**PREDICTOR_FIELDS, "effects": {"sales": "negative"}}
    with pytest.raises(ValidationError, match="no expected effect on it"):
        make_spec(**fields)


def test_reference_month_only_for_annual():
    with pytest.raises(ValidationError, match="annual"):
        make_spec(reference_month=6)


def test_singstat_ref_requires_unit_of_measure():
    with pytest.raises(ValidationError, match="unit_of_measure"):
        make_spec(source={"source": "singstat", "table_id": "M1", "series": "1"})


def test_sum_aggregate_requires_rows_per_period():
    with pytest.raises(ValidationError, match="rows_per_period"):
        make_spec(source={"source": "datagovsg", "dataset_id": "d", "layout": "tidy",
                          "value_column": "units", "aggregate": "sum"})


def test_quarantine_requires_expected_status():
    with pytest.raises(ValidationError, match="expected_status"):
        make_spec(role="quarantine", markets=[])


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    text = REGISTRY_PATH.read_text(encoding="utf-8").replace(
        "    publication_lag_days: 21\n", "    publication_lag_days: 21\n    publication_lag_days: 5\n", 1)
    path = tmp_path / "sources.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        load_registry(path)


def test_fingerprint_is_stable_and_sensitive(registry):
    again = load_registry(REGISTRY_PATH)
    assert registry.fingerprint() == again.fingerprint()
    changed = again.model_copy(deep=True)
    changed.series[0].publication_lag_days += 1
    assert changed.fingerprint() != registry.fingerprint()


def test_transform_units():
    assert Transform.YOY_BP.change_unit == "bp"
    assert Transform.QOQ_PP.change_unit == "pp"
    assert Transform.T4Q_YOY.step == "yoy"
    assert Effect.NEGATIVE.sign == -1 and Effect.AMBIGUOUS.sign == 0


def test_plausible_bands_contain_the_admitted_history(registry):
    """A band that warns on real history would be ignored in practice; this guards the
    bands against the committed snapshots (the run reports no range warnings)."""
    from datetime import date

    from sgmacro.pipeline import RunConfig, analyse, fetch_stage

    config = RunConfig(as_of=date(2026, 10, 2), offline=True)
    result = analyse(config, registry, fetch_stage(config, registry, lambda m: None))
    offenders = [(o.spec.id, f.message) for o in result.outcomes.values() if o.spec.role is not Role.QUARANTINE
                 for f in o.result.findings if f.check == "range"]
    assert offenders == []
