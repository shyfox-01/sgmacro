"""Static SVG charts for the Markdown report.

Two figures, both plotting computed change histories (never raw levels, whose
scales are incomparable):

* ``targets.svg``: year-on-year change of the three target indices, one axis,
  categorical slots 1-3 of the reference palette (the three that stay
  colour-blind-safe in every pairing), legend plus direct end labels.
* ``predictors.svg``: small multiples, one panel per predictor in its own unit,
  the declared neutral band shaded and the latest reading marked and labelled.

Output is byte-stable: a fixed SVG hash salt, no creation date, glyphs as paths,
so a repeated run leaves the files unchanged.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sgmacro.periods import Period  # noqa: E402
from sgmacro.registry import Frequency  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
START_YEAR = 2005
UNIT_SUFFIX = {"pct": "%", "pp": " pp", "bp": " bp", "count": " rounds"}

plt.rcParams.update({
    "svg.hashsalt": "sgmacro",
    "svg.fonttype": "path",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.titlecolor": TEXT,
})


def _decimal_years(freq: Frequency, ordinals) -> list[float]:
    out = []
    for o in ordinals:
        p = Period.from_ordinal(freq, int(o))
        out.append(p.year + (p.index - 0.5) / freq.periods_per_year)
    return out


YEAR_TICKS = list(range(START_YEAR, 2031, 5))


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xticks(YEAR_TICKS, [str(y) for y in YEAR_TICKS])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", facecolor=SURFACE, metadata={"Date": None})
    # Written as bytes with LF endings so the file hash is the same on every platform.
    path.write_bytes(buffer.getvalue().replace("\r\n", "\n").encode("utf-8"))
    plt.close(fig)


def targets_chart(result, path: Path) -> None:
    from sgmacro.registry import Role, Transform
    from sgmacro.transform import change_history

    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ends = []
    for colour, outcome in zip(SERIES, result.reportable(Role.TARGET)):
        spec = outcome.spec
        history = change_history(spec, Transform.YOY, outcome.result.gated_series()).dropna()
        history = history[history.index >= START_YEAR * spec.frequency.periods_per_year]
        if history.empty:  # e.g. an as-of date before the chart window opens
            continue
        x = _decimal_years(spec.frequency, history.index)
        ax.plot(x, history.values, color=colour, linewidth=1.6, label=spec.short_label, solid_capstyle="round")
        ax.plot([x[-1]], [history.values[-1]], "o", color=colour, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ends.append([x[-1], float(history.values[-1]), spec.short_label.split(" (")[0], colour])
    # Direct end labels in text ink, nudged apart so they never collide.
    ends.sort(key=lambda e: e[1])
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < span * 0.045:
            ends[i][1] = ends[i - 1][1] + span * 0.045
    for x_end, y_end, label, colour in ends:
        ax.annotate(label, (x_end, y_end), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXT)
    ax.set_xlim(START_YEAR, ends[0][0] + 3.2 if ends else None)
    ax.set_ylabel("year-on-year change, %")
    ax.set_title("Target indices: year-on-year change (%)", loc="left", fontsize=10)
    if ends:
        ax.legend(loc="upper left", frameon=False, fontsize=8, ncols=3)
    else:
        ax.text(0.5, 0.5, f"No target history since {START_YEAR} for this as-of date.", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=TEXT)
    fig.tight_layout()
    _save(fig, path)


def predictors_chart(result, path: Path) -> None:
    from sgmacro.registry import Role

    outcomes = [o for o in result.reportable(Role.PREDICTOR) if o.changes is not None]
    if not outcomes:
        fig, ax = plt.subplots(figsize=(9.6, 1.2), dpi=100)
        fig.patch.set_facecolor(SURFACE)
        ax.set_axis_off()
        ax.text(0.01, 0.5, "No predictor passed validation for this as-of date.", fontsize=10, color=TEXT,
                va="center")
        _save(fig, path)
        return
    cols = 3
    rows = math.ceil(len(outcomes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9.6, 2.15 * rows), dpi=100, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[len(outcomes):]:
        ax.set_visible(False)
    for ax, outcome in zip(axes.flat, outcomes):
        spec, changes = outcome.spec, outcome.changes
        history = changes.headline_history
        history = history[history.index >= START_YEAR * spec.frequency.periods_per_year].dropna()
        _style(ax)
        if history.empty:
            continue
        x = _decimal_years(spec.frequency, history.index)
        band = spec.neutral_band or 0
        if band:
            ax.axhspan(-band, band, color=BAND, linewidth=0, zorder=0)
        ax.plot(x, history.values, color=SERIES[0], linewidth=1.3, solid_capstyle="round")
        ax.plot([x[-1]], [history.values[-1]], "o", color=SERIES[0], markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        head = result.facts.get(outcome.facts["headline"])
        ax.annotate(head.display, (x[-1], history.values[-1]), xytext=(-4, 7), textcoords="offset points",
                    ha="right", fontsize=7.5, color=TEXT)
        unit = UNIT_SUFFIX[spec.headline.change_unit].strip()
        ax.set_title(f"{spec.short_label}\n{changes.headline.human} ({unit})", loc="left", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(START_YEAR, x[-1] + 0.8)
    fig.suptitle("Predictors: headline change history (shaded band = declared neutral zone)",
                 x=0.01, ha="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    _save(fig, path)
