"""
Create crisis_vs_normal_performance.png
Grouped bar charts comparing mean GMVP Sharpe under crisis vs non-crisis periods.

Output:
  results/figs_regime_similarity/crisis_vs_normal.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "regime_covariance" / "backtest.csv"
OUT_PATH = PROJECT_ROOT / "results" / "figs_regime_similarity" / "crisis_vs_normal.png"


@dataclass(frozen=True)
class BacktestRow:
    d: date
    model: Optional[float]
    mix: Optional[float]
    shrink: Optional[float]
    roll: Optional[float]


def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        y, m, dd = map(int, s.split("-"))
        return date(y, m, dd)
    except Exception:
        return None


def parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def mean_of_getter(rows: list[BacktestRow], predicate, getter) -> Optional[float]:
    total = 0.0
    count = 0
    for r in rows:
        if not predicate(r.d):
            continue
        v = getter(r)
        if v is None:
            continue
        total += v
        count += 1
    if count == 0:
        return None
    return total / count


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    # Local import so script fails with a clear message if matplotlib isn't installed.
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate the plot. "
            "Install it and re-run this script."
        ) from e

    rows: list[BacktestRow] = []
    with CSV_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {CSV_PATH}")
        for row in reader:
            d = parse_date(row.get("date", ""))
            if d is None:
                continue
            # Restrict to 2013-2021 as requested
            if d < date(2013, 1, 1) or d > date(2021, 12, 31):
                continue
            rows.append(
                BacktestRow(
                    d=d,
                    model=parse_float(row.get("model_gmvp_sharpe", "")),
                    mix=parse_float(row.get("mix_gmvp_sharpe", "")),
                    shrink=parse_float(row.get("shrink_gmvp_sharpe", "")),
                    roll=parse_float(row.get("roll_gmvp_sharpe", "")),
                )
            )

    # Period definitions (inclusive)
    china_start = date(2015, 8, 1)
    china_end = date(2016, 2, 29)
    q4_2018_start = date(2018, 10, 1)
    q4_2018_end = date(2019, 1, 31)
    covid_start = date(2020, 2, 1)
    covid_end = date(2020, 6, 30)

    def is_in_window(d: date, s: date, e: date) -> bool:
        return s <= d <= e

    def is_crisis(d: date) -> bool:
        return (
            is_in_window(d, china_start, china_end)
            or is_in_window(d, q4_2018_start, q4_2018_end)
            or is_in_window(d, covid_start, covid_end)
        )

    def is_non_crisis(d: date) -> bool:
        return (date(2013, 1, 1) <= d <= date(2021, 12, 31)) and (not is_crisis(d))

    # Compute mean Sharpe for each period and method
    periods = [
        ("Non-Crisis", is_non_crisis),
        ("China Selloff", lambda d: is_in_window(d, china_start, china_end)),
        ("Q4 2018 Selloff", lambda d: is_in_window(d, q4_2018_start, q4_2018_end)),
        ("COVID", lambda d: is_in_window(d, covid_start, covid_end)),
    ]

    methods = ["model", "mix", "shrink", "roll"]
    method_getters = {
        "model": lambda r: r.model,
        "mix": lambda r: r.mix,
        "shrink": lambda r: r.shrink,
        "roll": lambda r: r.roll,
    }
    method_colors = {"model": "#1f77b4", "mix": "#ff7f0e", "shrink": "#2ca02c", "roll": "#7f7f7f"}
    method_labels = {"model": "model", "mix": "mix", "shrink": "shrink", "roll": "roll"}

    # period_means[period_label][method] = mean or None
    period_means: dict[str, dict[str, Optional[float]]] = {}
    for plabel, pred in periods:
        period_means[plabel] = {}
        for m in methods:
            period_means[plabel][m] = mean_of_getter(rows, pred, method_getters[m])

    # Non-crisis mean for panel 2 and improvement annotations
    non_crisis_means = period_means["Non-Crisis"]

    # Global extrema for annotation placement
    all_means: list[float] = []
    for pl in period_means:
        for m in methods:
            v = period_means[pl][m]
            if v is not None:
                all_means.append(v)
    y_max = max(all_means) if all_means else 0.0
    y_min = min(all_means) if all_means else 0.0
    y_span = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

    def pct_improvement(base: float, compare: float) -> Optional[float]:
        # % improvement of base over compare.
        if compare is None or base is None:
            return None
        if compare == 0:
            return None
        return (base - compare) / abs(compare) * 100.0

    non_crisis_model = non_crisis_means["model"]

    # 2-panel figure: left wider, right narrower
    fig = Figure(figsize=(14, 5), dpi=200, facecolor="white")
    # Axes layout (manual coordinates in figure fractions)
    # Left panel: x=[0.08..0.64], Right panel: x=[0.66..0.96]
    # y=[0.12..0.88]
    ax1 = fig.add_axes([0.07, 0.12, 0.52, 0.78])
    ax2 = fig.add_axes([0.64, 0.12, 0.30, 0.78])

    # --- Panel 1: Grouped bars for each period ---
    group_labels = [p[0] for p in periods]  # Non-Crisis, China Selloff, Q4 2018 Selloff, COVID
    group_x = list(range(len(group_labels)))

    bar_width = 0.2
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    order = ["model", "mix", "shrink", "roll"]

    for i, m in enumerate(order):
        vals = []
        for pl in group_labels:
            vals.append(period_means[pl][m])

        xs = [x + offsets[i] for x in group_x]
        # Replace None with 0 for plotting; label will show None as "nan"
        plot_vals = [v if v is not None else 0.0 for v in vals]
        ax1.bar(
            xs,
            plot_vals,
            width=bar_width,
            color=method_colors[m],
            label=method_labels[m],
            edgecolor="black",
            linewidth=0.2,
        )

        # Label each bar with 1 decimal place
        y_offset = 0.018 * y_span
        for x, v in zip(xs, vals):
            if v is None:
                txt = "nan"
                y = 0.0
            else:
                txt = f"{v:.1f}"
                y = v
            ax1.text(
                x,
                y + (y_offset if y >= 0 else -y_offset),
                txt,
                ha="center",
                va="bottom" if v is not None and v >= 0 else "top",
                fontsize=9,
                rotation=0,
                clip_on=False,
            )

    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax1.set_xticks(group_x)
    ax1.set_xticklabels(group_labels)
    ax1.set_ylim(y_min - 0.15 * y_span, y_max + 0.25 * y_span)
    ax1.set_title("Mean GMVP Sharpe: Crisis vs Normal Periods")
    ax1.text(
        0.0,
        1.02,
        "(higher is better)",
        transform=ax1.transAxes,
        fontsize=11,
        va="bottom",
    )

    ax1.legend(title="Method", loc="best", frameon=True)

    # --- Panel 2: Non-crisis outperformance (model vs others) ---
    methods_short = ["model", "mix", "shrink", "roll"]
    x2 = list(range(len(methods_short)))
    vals2 = [non_crisis_means[m] if non_crisis_means[m] is not None else 0.0 for m in methods_short]
    colors2 = [method_colors[m] for m in methods_short]

    ax2.bar(
        x2,
        vals2,
        color=colors2,
        edgecolor="black",
        linewidth=0.2,
        width=0.6,
    )

    # Label model value and annotate % improvement
    for xi, m, v in zip(x2, methods_short, vals2):
        # Right panel: show ONLY the Sharpe value label on top of the bar.
        # (No percentage / improvement labels.)
        if non_crisis_means[m] is None:
            txt = "nan"
        else:
            txt = f"{non_crisis_means[m]:.1f}"

        # Offset labels by a small fraction of y-range to reduce collisions.
        y_offset = 0.015 * y_span
        ax2.text(
            xi,
            v + (y_offset if v >= 0 else -y_offset),
            txt,
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=10,
            clip_on=False,
        )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods_short)
    ax2.set_title("Non-Crisis Outperformance")
    ax2.set_ylim(y_min - 0.20 * y_span, y_max + 0.30 * y_span)

    # Final layout and save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Use Agg canvas to print PNG without relying on Pillow.
    canvas = FigureCanvas(fig)
    canvas.print_png(str(OUT_PATH))
    print(f"Saved plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()

