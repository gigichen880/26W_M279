"""
Data coverage heatmap: valid vs missing vs flagged outliers (100 stocks × 2007–2021).
Output: slides/visuals/data_coverage.png (6×4 in, 300 dpi).
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = REPO / "data" / "processed" / "returns_universe_100.parquet"


def _volatile(ts: pd.Timestamp) -> bool:
    return (pd.Timestamp("2020-03-01") <= ts <= pd.Timestamp("2020-06-30")) or (
        pd.Timestamp("2008-09-01") <= ts <= pd.Timestamp("2009-06-30")
    )


def _bad_cell_mask_fast(vals: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Vectorized proxy for pipeline outlier flags (>|500%|, suspicious multi-stock >200%)."""
    absv = np.abs(vals)
    bad = (absv > 5.0) & np.isfinite(vals)
    T = vals.shape[0]
    vol = np.array([_volatile(pd.Timestamp(d)) for d in dates])
    n_big = np.nansum(absv > 2.0, axis=1)
    for t in range(T):
        if vol[t] or n_big[t] < 2:
            continue
        row = vals[t]
        bad[t] |= (absv[t] > 2.0) & np.isfinite(row)
    return bad


def _load_cellwise_bad_mask(returns_df: pd.DataFrame) -> np.ndarray:
    p = REPO / "scripts" / "data_validation" / "cellwise_clean_returns.py"
    spec = importlib.util.spec_from_file_location("cellwise_clean", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bad, _ = mod.build_bad_cell_mask(returns_df)
    return bad


def _synthetic_returns(n_dates: int = 3655, n_stocks: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2007-06-27", periods=n_dates, freq="B")
    if len(dates) != n_dates:
        dates = pd.bdate_range("2007-06-27", "2021-12-31")
    R = rng.normal(0, 0.015, (len(dates), n_stocks))
    df = pd.DataFrame(R, index=dates, columns=[f"S{i:03d}" for i in range(n_stocks)])
    # IPO-style missing early columns
    for j in range(n_stocks):
        drop = rng.integers(0, 800)
        df.iloc[:drop, j] = np.nan
    # Random gaps
    m = rng.random(df.shape) < 0.012
    df = df.mask(m)
    # A few extreme "outliers" for red cells
    for _ in range(120):
        t, j = rng.integers(0, len(df)), rng.integers(0, n_stocks)
        if np.isfinite(df.iloc[t, j]):
            df.iloc[t, j] = 6.0 * np.sign(rng.choice([-1, 1]))
    return df


def _load_returns(parquet_path: Path | None = None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    parquet_path = Path(parquet_path or DEFAULT_PARQUET)
    if parquet_path.is_file():
        df = pd.read_parquet(parquet_path)
        if df.shape[1] > df.shape[0]:
            df = df.T
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.sort_index(axis=1).sort_index()
    else:
        df = _synthetic_returns()

    df.index = pd.to_datetime(df.index)
    df = df.loc[(df.index >= pd.Timestamp("2007-01-01")) & (df.index <= pd.Timestamp("2021-12-31"))]
    dates = pd.DatetimeIndex(df.index)
    return df, dates


def create_data_coverage_fixed(parquet_path: Path | None = None, out_path: Path | None = None) -> Path:
    """
    Presentation-ready data coverage visual with:
      - Top panel: % missing over time with 30% threshold
      - Bottom panel: stock/date heatmap with crisis overlays
    """
    df, dates = _load_returns(parquet_path)

    # Restrict to first 100 stocks for readability
    if df.shape[1] > 100:
        df = df.iloc[:, :100]
    tickers = df.columns.tolist()
    n_t, n_s = df.shape[0], df.shape[1]

    vals = df.values.astype(float)
    nan_m = np.isnan(vals)

    # Outlier mask (fast proxy or full cellwise if requested)
    if os.environ.get("COVERAGE_HEATMAP_FULL_CELLWISE", "").lower() in ("1", "true", "yes"):
        try:
            bad_m = _load_cellwise_bad_mask(df)
        except Exception:
            bad_m = _bad_cell_mask_fast(vals, dates)
    else:
        bad_m = _bad_cell_mask_fast(vals, dates)
    bad_m = bad_m & ~nan_m

    # Coverage codes: 0 = missing, 1 = valid, 2 = outlier
    codes = np.full_like(vals, 1, dtype=int)
    codes[nan_m] = 0
    codes[bad_m] = 2

    # % missing per date (top panel)
    pct_missing = nan_m.mean(axis=1) * 100.0

    # Create figure: wider for time axis
    fig = plt.figure(figsize=(10, 6), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.10, left=0.08, right=0.98, top=0.92, bottom=0.16)

    # ── Top: % missing ─────────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(dates, pct_missing, color="#D2691E", linewidth=2.5, label="% missing")

    ax_top.axhline(30.0, color="red", linestyle="--", linewidth=2.0, label="30% threshold (marginal)", zorder=1)
    upper = max(50.0, float(pct_missing.max()) * 1.2)
    ax_top.fill_between(dates, 30.0, upper, color="red", alpha=0.08, zorder=0)

    ax_top.set_ylabel("% Missing", fontsize=11, fontweight="bold")
    ax_top.set_ylim(0.0, upper)
    ax_top.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax_top.legend(loc="upper left", fontsize=9, frameon=True)
    ax_top.tick_params(axis="x", labelbottom=False)

    # ── Bottom: heatmap ───────────────────────────────────────────
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # Map codes → RGB
    rgb = np.ones((n_s, n_t, 3), dtype=float)
    from matplotlib.colors import to_rgb

    rgb[codes.T == 0] = to_rgb("#FFFFFF")   # missing
    rgb[codes.T == 1] = to_rgb("#90EE90")   # valid
    rgb[codes.T == 2] = to_rgb("#FFB6C1")   # outlier

    x0 = mdates.date2num(dates[0])
    x1 = mdates.date2num(dates[-1])

    # Crisis bands behind heatmap
    crisis_periods = [
        ("2008-09-01", "2009-03-31", "GFC"),
        ("2020-02-01", "2020-05-31", "COVID"),
    ]
    for start, end, label in crisis_periods:
        s = mdates.date2num(pd.to_datetime(start))
        e = mdates.date2num(pd.to_datetime(end))
        ax_bottom.axvspan(s, e, color="#9370DB", alpha=0.20, zorder=0)
        mid = (s + e) / 2.0
        ax_bottom.text(
            mid,
            -3.0,
            label,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#4B0082",
            clip_on=False,
        )

    ax_bottom.imshow(
        rgb,
        origin="upper",
        aspect="auto",
        extent=[x0, x1, n_s, 0],
        interpolation="nearest",
        zorder=1,
    )

    ax_bottom.set_ylabel("Stock Index", fontsize=12, fontweight="bold")
    ax_bottom.set_xlabel("Date", fontsize=12, fontweight="bold")

    ax_bottom.xaxis_date()
    ax_bottom.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bottom.tick_params(axis="both", labelsize=9)

    # Y ticks: 1..100 stepped
    y_ticks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100][: n_s]
    ax_bottom.set_yticks(y_ticks)
    ax_bottom.set_yticklabels(y_ticks)

    # Figure title
    fig.suptitle(
        "Data Coverage: Valid, Missing, and Flagged Cells",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Legend: only include items that actually appear
    from matplotlib.lines import Line2D

    legend_items = [
        mpatches.Patch(facecolor="#90EE90", edgecolor="black", label="Valid data"),
        mpatches.Patch(facecolor="#FFFFFF", edgecolor="black", label="Missing"),
        mpatches.Patch(facecolor="#9370DB", alpha=0.20, edgecolor="none", label="Crisis periods (GFC, COVID)"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="30% missing threshold"),
    ]
    if (codes == 2).any():
        legend_items.insert(
            1,
            mpatches.Patch(facecolor="#FFB6C1", edgecolor="black", label="Flagged outlier"),
        )

    ax_bottom.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(legend_items),
        fontsize=9,
        frameon=True,
    )

    out_path = Path(out_path or (Path(__file__).parent / "data_coverage_fixed.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    create_data_coverage_fixed()
