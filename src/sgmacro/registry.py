"""Typed loader for ``config/sources.yaml``, the contract for the whole workflow.

The registry declares what each series *is* (definition, unit, frequency,
seasonal adjustment, publication lag, reference date) and what may be *done*
with it (which change calculations are meaningful, which market it bears on,
and in which direction economic theory says it should push that market).

Invariants are enforced here, at load time, so a bad configuration fails before
it can produce a plausible-looking wrong number:

1. ``unit_kind`` constrains ``transforms``. A rate or a ratio may only use
   percentage-point or basis-point changes. SORA moving from 1.00% to 1.50% is
   +50 bp, never +50%.
2. ``frequency`` constrains ``transforms``. A quarterly series cannot carry a
   month-on-month change, and "a year ago" is resolved from the frequency rather
   than hardcoded.
3. Every predictor declares its expected effect on each market it is claimed to
   bear on. That declaration is what the deterministic signal layer and the LLM
   claim checker hold generated text to, so it cannot be left implicit.
4. Aggregations declare how many rows make a complete period, so a partially
   published quarter is detected instead of being summed into a low total.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class UnitKind(str, Enum):
    """What kind of quantity a series carries. Drives change arithmetic."""

    INDEX = "index"
    LEVEL = "level"
    RATE = "rate"
    RATIO = "ratio"
    COUNT = "count"
    """A count of events per period (policy rounds); summed, never a percentage."""

    @property
    def is_difference_based(self) -> bool:
        return self in (UnitKind.RATE, UnitKind.RATIO, UnitKind.COUNT)


class Frequency(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"

    @property
    def periods_per_year(self) -> int:
        return {"monthly": 12, "quarterly": 4, "annual": 1}[self.value]

    @property
    def upstream_label(self) -> str:
        """How SingStat metadata spells this frequency."""
        return {"monthly": "Monthly", "quarterly": "Quarterly", "annual": "Annual"}[self.value]


class Adjustment(str, Enum):
    SA = "seasonally_adjusted"
    NSA = "not_seasonally_adjusted"
    NOT_APPLICABLE = "not_applicable"


class Role(str, Enum):
    TARGET = "target"
    PREDICTOR = "predictor"
    QUARANTINE = "quarantine"


class Market(str, Enum):
    SALES = "sales"
    RENTS = "rents"


class Effect(str, Enum):
    """Expected sign of a predictor's influence on a market, from theory."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    AMBIGUOUS = "ambiguous"

    @property
    def sign(self) -> int:
        return {"positive": 1, "negative": -1, "ambiguous": 0}[self.value]


class Transform(str, Enum):
    """Change calculations.

    Bare names are ratio-based percentage changes; ``_pp`` and ``_bp`` are
    differences. ``t4q_yoy`` compares the sum of the latest four quarters with the
    four before, which is the right reading for a noisy quarterly *flow* such as
    transaction counts. ``t4q_sum`` is the sum of the latest four quarters itself,
    used for event counts (net policy rounds in the past year).
    """

    MOM = "mom"
    QOQ = "qoq"
    YOY = "yoy"
    MOM_PP = "mom_pp"
    QOQ_PP = "qoq_pp"
    YOY_PP = "yoy_pp"
    MOM_BP = "mom_bp"
    YOY_BP = "yoy_bp"
    T4Q_YOY = "t4q_yoy"
    T4Q_SUM = "t4q_sum"

    @property
    def step(self) -> str:
        return "yoy" if self in (Transform.T4Q_YOY, Transform.T4Q_SUM) else self.value.split("_")[0]

    @property
    def change_unit(self) -> str:
        if self is Transform.T4Q_SUM:
            return "count"
        if self.value.endswith("_pp"):
            return "pp"
        if self.value.endswith("_bp"):
            return "bp"
        return "pct"


_ALLOWED_BY_UNIT: dict[UnitKind, set[Transform]] = {
    UnitKind.INDEX: {Transform.MOM, Transform.QOQ, Transform.YOY},
    UnitKind.LEVEL: {Transform.MOM, Transform.QOQ, Transform.YOY, Transform.T4Q_YOY},
    UnitKind.RATE: {Transform.MOM_PP, Transform.QOQ_PP, Transform.YOY_PP,
                    Transform.MOM_BP, Transform.YOY_BP},
    UnitKind.RATIO: {Transform.MOM_PP, Transform.QOQ_PP, Transform.YOY_PP},
    UnitKind.COUNT: {Transform.T4Q_SUM},
}

_ALLOWED_STEPS: dict[Frequency, set[str]] = {
    Frequency.MONTHLY: {"mom", "yoy"},
    Frequency.QUARTERLY: {"qoq", "yoy"},
    Frequency.ANNUAL: {"yoy"},
}


# ---------------------------------------------------------------------------
# Source references: where one series physically lives upstream
# ---------------------------------------------------------------------------

class _Ref(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SingStatRef(_Ref):
    """A series in a SingStat Table Builder table, addressed by ``seriesNo``.

    ``seriesNo`` rather than the row label: labels repeat inside a table (the
    pipeline table has two rows called "Under Construction"), series numbers do
    not.
    """

    source: Literal["singstat"] = "singstat"
    table_id: str
    series: str | None = None
    ratio: tuple[str, str] | None = None
    """``(numerator, denominator)`` series numbers; the value is their ratio in percent."""
    unit_of_measure: str | None = None
    """The ``uoM`` SingStat's metadata must report for the series (or both ratio
    inputs). Cross-checked on every run, so a units mistake in this file is caught
    by the publisher's own metadata instead of by luck."""

    @model_validator(mode="after")
    def _one_address(self) -> SingStatRef:
        if (self.series is None) == (self.ratio is None):
            raise ValueError(f"{self.table_id}: set exactly one of `series` or `ratio`")
        return self

    @property
    def object_key(self) -> str:
        return f"singstat/{self.table_id}"

    @property
    def series_numbers(self) -> tuple[str, ...]:
        return self.ratio if self.ratio else (self.series,)  # type: ignore[return-value]

    @property
    def link(self) -> str:
        return f"https://tablebuilder.singstat.gov.sg/table/TS/{self.table_id}"

    @property
    def identifier(self) -> str:
        if self.ratio:
            return f"{self.table_id} series {self.ratio[0]} / {self.ratio[1]}"
        return f"{self.table_id} series {self.series}"


class DataGovRef(_Ref):
    """A series inside a data.gov.sg CSV dataset.

    ``tidy`` layouts have one row per observation and dimension columns, so a
    series is a filter over rows. ``wide`` layouts have one row per series and
    one column per period, so a series is a row label.
    """

    source: Literal["datagovsg"] = "datagovsg"
    dataset_id: str
    layout: Literal["tidy", "wide"]
    period_column: str = "quarter"
    value_column: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    aggregate: Literal["single", "sum"] = "single"
    rows_per_period: int | None = None
    row_label: str | None = None
    ratio_labels: tuple[str, str] | None = None

    @model_validator(mode="after")
    def _layout_fields(self) -> DataGovRef:
        if self.layout == "tidy":
            if not self.value_column:
                raise ValueError(f"{self.dataset_id}: tidy layout requires value_column")
            if self.row_label or self.ratio_labels:
                raise ValueError(f"{self.dataset_id}: row labels apply only to wide layout")
            if self.aggregate == "sum" and not self.rows_per_period:
                raise ValueError(
                    f"{self.dataset_id}: aggregate=sum requires rows_per_period, so a "
                    "partially published period is detected rather than summed"
                )
        else:
            if (self.row_label is None) == (self.ratio_labels is None):
                raise ValueError(f"{self.dataset_id}: wide layout needs row_label or ratio_labels")
            if self.filters or self.value_column or self.aggregate != "single":
                raise ValueError(f"{self.dataset_id}: filters/value_column/aggregate are tidy-only")
        return self

    @property
    def object_key(self) -> str:
        return f"datagovsg/{self.dataset_id}"

    @property
    def link(self) -> str:
        return f"https://data.gov.sg/datasets/{self.dataset_id}/view"

    @property
    def identifier(self) -> str:
        if self.layout == "wide":
            what = self.row_label or " / ".join(self.ratio_labels or ())
            return f"{self.dataset_id} row '{what}'"
        if self.filters:
            cond = ", ".join(f"{k}={v}" for k, v in self.filters.items())
            return f"{self.dataset_id} [{cond}]"
        return self.dataset_id


class CuratedRef(_Ref):
    """A hand-maintained event list in the repository (no API exists).

    It is snapshotted and hashed like a download, gated by each event's
    announcement date, and traced back to its events, so it is held to the same
    standard as the published series, but a person keeps it up to date.
    """

    source: Literal["curated"] = "curated"
    path: str

    @property
    def object_key(self) -> str:
        return f"curated/{Path(self.path).stem}"

    @property
    def link(self) -> str:
        return self.path

    @property
    def identifier(self) -> str:
        return self.path


SourceRef = Annotated[Union[SingStatRef, DataGovRef, CuratedRef], Field(discriminator="source")]


class CrossCheck(BaseModel):
    """A second publication of the same series, used for reconciliation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: SourceRef
    tolerance: float = Field(gt=0)
    """Largest absolute difference treated as rounding, in the series' unit."""
    note: str | None = None


class SeriesBreak(BaseModel):
    """A point where the publisher changed coverage or definition, so levels on
    either side are not comparable. Changes whose window spans it are not
    computed, and drop out of percentiles and correlations."""

    model_config = ConfigDict(extra="forbid")

    period: str
    """First period on the new basis, in the series' own frequency."""
    note: str


class RejectedAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    reason: str


class MarketEffects(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sales: Effect | None = None
    rents: Effect | None = None

    def for_market(self, market: Market) -> Effect | None:
        return getattr(self, market.value)


# ---------------------------------------------------------------------------
# One series
# ---------------------------------------------------------------------------

class SeriesSpec(BaseModel):
    """One declared series. Every report field comes from here or from a
    computed value traced back to here."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    short_label: str
    role: Role
    publisher: str
    source: SourceRef
    cross_check: CrossCheck | None = None

    definition: str
    unit: str
    unit_kind: UnitKind
    frequency: Frequency
    seasonal_adjustment: Adjustment
    update_frequency: str
    publication_lag_days: int = Field(ge=0)
    reference_month: int | None = Field(default=None, ge=1, le=12)
    """For annual series measured at a point inside the year (end-June for
    Singapore population estimates). The as-of gate counts the publication lag
    from that date rather than from 31 December."""
    plausible_range: tuple[float, float]
    transforms: list[Transform]
    headline: Transform
    decimals: int = Field(default=1, ge=0, le=4)

    markets: list[Market] = Field(default_factory=list)
    channel: str | None = None
    effects: MarketEffects = Field(default_factory=MarketEffects)
    neutral_band: float | None = Field(default=None, ge=0)
    """Headline changes within +/- this band are read as flat, in headline units.
    Declared rather than estimated so a reviewer can see and argue with it."""
    expected_lead_quarters: tuple[int, int] | None = None
    rationale: str | None = None
    limitations: str | None = None
    history_note: str | None = None
    breaks: list[SeriesBreak] = Field(default_factory=list)
    rejected_alternatives: list[RejectedAlternative] = Field(default_factory=list)
    expected_status: str | None = None

    # -- invariants --------------------------------------------------------
    @model_validator(mode="after")
    def _transforms_match_units(self) -> SeriesSpec:
        allowed = _ALLOWED_BY_UNIT[self.unit_kind]
        bad = [t.value for t in self.transforms if t not in allowed]
        if bad:
            raise ValueError(
                f"{self.id}: unit_kind={self.unit_kind.value} forbids {bad}; allowed "
                f"{sorted(t.value for t in allowed)}. Rates and ratios take "
                "percentage-point or basis-point changes, never percentage changes."
            )
        return self

    @model_validator(mode="after")
    def _transforms_match_frequency(self) -> SeriesSpec:
        steps = _ALLOWED_STEPS[self.frequency]
        bad = [t.value for t in self.transforms if t.step not in steps]
        for quarterly_only in (Transform.T4Q_YOY, Transform.T4Q_SUM):
            if quarterly_only in self.transforms and self.frequency is not Frequency.QUARTERLY:
                bad.append(quarterly_only.value)
        if bad:
            raise ValueError(f"{self.id}: frequency={self.frequency.value} forbids {sorted(set(bad))}")
        return self

    @model_validator(mode="after")
    def _headline_is_computed(self) -> SeriesSpec:
        if self.headline not in self.transforms:
            raise ValueError(f"{self.id}: headline {self.headline.value} is not among its transforms")
        return self

    @model_validator(mode="after")
    def _breaks_parse(self) -> SeriesSpec:
        if self.breaks:
            from sgmacro.periods import parse_period  # local: periods imports this module

            for item in self.breaks:
                parse_period(item.period, self.frequency)
        return self

    @model_validator(mode="after")
    def _reference_month_only_for_annual(self) -> SeriesSpec:
        if self.reference_month is not None and self.frequency is not Frequency.ANNUAL:
            raise ValueError(f"{self.id}: reference_month applies only to annual series")
        return self

    @model_validator(mode="after")
    def _plausible_range_ordered(self) -> SeriesSpec:
        low, high = self.plausible_range
        if low >= high:
            raise ValueError(f"{self.id}: plausible_range {self.plausible_range} is not increasing")
        return self

    @model_validator(mode="after")
    def _role_requirements(self) -> SeriesSpec:
        if self.role is Role.PREDICTOR:
            missing = [f for f in ("channel", "rationale", "limitations", "neutral_band",
                                   "expected_lead_quarters") if getattr(self, f) is None]
            if missing or not self.markets:
                raise ValueError(f"{self.id}: a predictor must declare {missing or ['markets']}")
            for market in self.markets:
                if self.effects.for_market(market) is None:
                    raise ValueError(
                        f"{self.id}: declares market {market.value} but no expected effect on it"
                    )
            lo, hi = self.expected_lead_quarters  # type: ignore[misc]
            if not 0 <= lo <= hi:
                raise ValueError(f"{self.id}: expected_lead_quarters must be 0 <= lo <= hi")
        elif self.role is Role.TARGET:
            if len(self.markets) != 1:
                raise ValueError(f"{self.id}: a target belongs to exactly one market")
            if self.effects.sales or self.effects.rents:
                raise ValueError(f"{self.id}: targets carry no expected effect")
        elif self.role is Role.QUARANTINE and not self.expected_status:
            raise ValueError(f"{self.id}: quarantined series must declare expected_status")
        return self

    @model_validator(mode="after")
    def _unit_of_measure_for_singstat(self) -> SeriesSpec:
        refs = [self.source] + ([self.cross_check.ref] if self.cross_check else [])
        for ref in refs:
            if isinstance(ref, SingStatRef) and not ref.unit_of_measure:
                raise ValueError(
                    f"{self.id}: SingStat refs must declare unit_of_measure so the "
                    "publisher's metadata can be checked against it"
                )
        return self

    # -- helpers -------------------------------------------------------------
    @property
    def market(self) -> Market | None:
        """The market a target series measures."""
        return self.markets[0] if self.role is Role.TARGET else None

    def refs(self) -> list[SourceRef]:
        return [self.source] + ([self.cross_check.ref] if self.cross_check else [])


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    timezone: str
    selection_rules: list[str]
    series: list[SeriesSpec]

    @model_validator(mode="after")
    def _unique_ids(self) -> Registry:
        seen: set[str] = set()
        for spec in self.series:
            if spec.id in seen:
                raise ValueError(f"duplicate series id {spec.id!r}")
            seen.add(spec.id)
        return self

    def by_role(self, role: Role) -> list[SeriesSpec]:
        return [s for s in self.series if s.role is role]

    @property
    def targets(self) -> list[SeriesSpec]:
        return self.by_role(Role.TARGET)

    @property
    def predictors(self) -> list[SeriesSpec]:
        return self.by_role(Role.PREDICTOR)

    @property
    def quarantine(self) -> list[SeriesSpec]:
        return self.by_role(Role.QUARANTINE)

    def get(self, series_id: str) -> SeriesSpec:
        for spec in self.series:
            if spec.id == series_id:
                return spec
        raise KeyError(f"no series {series_id!r} in registry")

    def fingerprint(self) -> str:
        """Stable hash of the registry contents, part of every run identity."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


DEFAULT_REGISTRY_PATH = Path("config/sources.yaml")


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML silently keeps the last of two identical keys; a registry edit that
    duplicates ``publication_lag_days`` would then change behaviour invisibly."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> Registry:
    """Load and fully validate the registry. Raises with every problem listed."""
    raw: dict[str, Any] = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # noqa: S506
    return Registry.model_validate(raw)
