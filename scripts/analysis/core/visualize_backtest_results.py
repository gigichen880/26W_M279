# Moved implementation from scripts/analysis/visualize_backtest_results.py

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Ensure project root on path when run as script (e.g. python scripts/analysis/core/...)
_root = Path(__file__).resolve().parents[2]
if _root.name == "scripts":
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from scripts.config_utils import load_yaml, deep_update, parse_overrides


def _ensure_dir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


SAVED_FILES: list[str] = []


def _savefig(path: str):
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=200)
    plt.close()
    SAVED_FILES.append(path)


def _read_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _metric_cols(df: pd.DataFrame, metric: str, methods: List[str]):
    out = {}
    for m in methods:
        c = f"{m}_{metric}"
        if c in df.columns:
            out[m] = c
    return out


def plot_equity_curves(df, outdir, methods):
    """Single-panel plot: GMVP cumulative wealth for selected methods (see config `equity_methods`)."""
    cols = _metric_cols(df, "gmvp_cumret", methods)
    if not cols:
        return
    plt.figure(figsize=(12, 5))
    for m, c in cols.items():
        r = df[c].fillna(0.0)
        eq = np.cumprod(1 + r)
        plt.plot(eq.index, eq.values, label=m)
    plt.title("GMVP Equity Curves")
    plt.ylabel("Cumulative wealth")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(os.path.join(outdir, "equity_curves_gmvp.png"))


def plot_method_overlays(df, outdir, methods, metrics):
    metrics = [m for m in metrics if m not in {"gmvp_sharpe", "turnover_l1"}]
    if not metrics:
        return
    n = len(metrics)
    fig = plt.figure(figsize=(12, 2.5 * n))
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(n, 1, i + 1)
        cols = _metric_cols(df, metric, methods)
        for m, c in cols.items():
            ax.plot(df[c], label=m, alpha=0.9 if m in {"model", "mix"} else 0.6)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    _savefig(os.path.join(outdir, "method_overlays.png"))


def plot_rolling_median(df, outdir, methods, metrics, window):
    metrics = [m for m in metrics if m not in {"gmvp_sharpe", "turnover_l1"}]
    if not metrics:
        return
    fig = plt.figure(figsize=(12, 2.5 * len(metrics)))
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(len(metrics), 1, i + 1)
        cols = _metric_cols(df, metric, methods)
        for m, c in cols.items():
            r = df[c].rolling(window).median()
            ax.plot(r, label=m, alpha=0.9 if m in {"model", "mix"} else 0.6)
        ax.set_title(f"{metric} rolling median")
        ax.legend()
        ax.grid(alpha=0.3)
    _savefig(os.path.join(outdir, f"rolling_median_{window}d.png"))


def plot_gmvp_sharpe(df, outdir, methods, roll_window: int = 21):
    cols = _metric_cols(df, "gmvp_sharpe", methods)
    if not cols:
        return
    all_vals = []
    for c in cols.values():
        v = pd.to_numeric(df[c], errors="coerce")
        all_vals.append(v.values)
    all_arr = np.concatenate(all_vals)
    finite = all_arr[np.isfinite(all_arr)]
    if finite.size > 0:
        q1 = np.percentile(finite, 5)
        q99 = np.percentile(finite, 95)
    else:
        q1, q99 = -1.0, 1.0
    plt.figure(figsize=(12, 4))
    for m, c in cols.items():
        s = pd.to_numeric(df[c], errors="coerce")
        plt.plot(s.index, s.values, alpha=0.35, linewidth=0.8)
        r = s.rolling(roll_window, min_periods=1).mean()
        plt.plot(r.index, r.values, label=m, alpha=0.95, linewidth=1.2)
    plt.title("GMVP Sharpe over time (thick line = 21d rolling mean)")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.ylim(q1, q99)
    _savefig(os.path.join(outdir, "gmvp_sharpe_timeseries.png"))


def plot_gmvp_sharpe_distribution(df, outdir, methods):
    """Overlay KDE of per-horizon GMVP Sharpe for model vs baselines."""
    cols = _metric_cols(df, "gmvp_sharpe", methods)
    if not cols:
        return
    all_vals = []
    series = {}
    for m, c in cols.items():
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        s = s[np.isfinite(s)]
        if s.size > 1:
            series[m] = s.values
            all_vals.extend(s.tolist())
    if not series or not all_vals:
        return
    all_arr = np.array(all_vals)
    x_min = np.percentile(all_arr, 1)
    x_max = np.percentile(all_arr, 99)
    x_grid = np.linspace(x_min, x_max, 200)
    plt.figure(figsize=(10, 5))
    for m in methods:
        if m not in series:
            continue
        vals = series[m]
        if vals.size < 2:
            continue
        color = "#1f77b4" if m in ("model", "mix") else "#7f7f7f"
        try:
            kde = scipy_stats.gaussian_kde(vals)
            density = kde(x_grid)
            density = np.maximum(density, 0)
            plt.fill_between(x_grid, density, alpha=0.35, color=color)
            plt.plot(x_grid, density, label=m, color=color, linewidth=1.5)
        except Exception:
            plt.hist(vals, bins=min(40, max(10, vals.size // 5)), density=True, alpha=0.4, label=m, color=color, edgecolor="black")
    plt.axvline(0, color="black", linestyle="--", alpha=0.6)
    plt.xlabel("GMVP Sharpe (per horizon)")
    plt.ylabel("Density")
    plt.title("Distribution of GMVP Sharpe: model vs baselines (over evaluation dates)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.xlim(x_min, x_max)
    _savefig(os.path.join(outdir, "gmvp_sharpe_distribution.png"))


def plot_turnover_l1(df, outdir, methods, roll_window: int = 21):
    cols = _metric_cols(df, "turnover_l1", methods)
    if not cols:
        return
    plt.figure(figsize=(12, 4))
    for m, c in cols.items():
        s = pd.to_numeric(df[c], errors="coerce")
        plt.plot(s.index, s.values, alpha=0.35, linewidth=0.8)
        r = s.rolling(roll_window, min_periods=1).mean()
        plt.plot(r.index, r.values, label=m, alpha=0.95, linewidth=1.2)
    plt.title("GMVP turnover (L1) over time (thick line = 21d rolling mean)")
    plt.ylabel("turnover L1")
    plt.grid(alpha=0.3)
    plt.legend()
    _savefig(os.path.join(outdir, "gmvp_turnover_l1_timeseries.png"))


def plot_covariance_error_timeseries(df, outdir, methods, error_metric: str = "fro"):
    cols = _metric_cols(df, error_metric, methods)
    if not cols:
        return
    plt.figure(figsize=(12, 4))
    for m, c in cols.items():
        plt.plot(df[c], label=m, alpha=0.9 if m in {"model", "mix"} else 0.6)
    plt.title(f"Forecast error ({error_metric}) over time")
    plt.ylabel(error_metric)
    plt.legend()
    plt.grid(alpha=0.3)
    all_vals = np.concatenate([pd.to_numeric(df[c], errors="coerce").values for c in cols.values()])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size > 0:
        plt.ylim(0, np.percentile(finite, 98))
    _savefig(os.path.join(outdir, f"covariance_error_{error_metric}_timeseries.png"))


def plot_cumulative_advantage(
    df, outdir, ref: str, methods: List[str], metric: str = "fro", *, higher_is_better: bool = False
):
    """
    Cumulative advantage: cumsum of (ref - baseline) when higher_is_better else (baseline - ref).
    Positive curve = reference is better than that baseline on average over time.
    """
    ref_col = f"{ref}_{metric}"
    if ref_col not in df.columns:
        return
    ref_s = pd.to_numeric(df[ref_col], errors="coerce")
    baselines = [m for m in methods if m in ("roll", "pers", "shrink")]
    plt.figure(figsize=(12, 4))
    for m in baselines:
        c = f"{m}_{metric}"
        if c not in df.columns:
            continue
        base_s = pd.to_numeric(df[c], errors="coerce")
        diff = (ref_s - base_s) if higher_is_better else (base_s - ref_s)
        cum = diff.cumsum()
        plt.plot(cum.index, cum.values, label=f"{ref} vs {m}")
    plt.axhline(0, color="black", linestyle="--", alpha=0.7)
    plt.title(f"Cumulative advantage: {ref} vs baselines ({metric}, positive = {ref} better)")
    plt.ylabel("Cumulative difference")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(os.path.join(outdir, f"cumulative_advantage_{ref}_{metric}.png"))


# Key report metrics to show: (section, metric) -> (display name, higher_is_better)
_REPORT_KEY_METRICS = [
    ("cov_error_mean", "fro", "Frobenius error", False),
    ("gmvp_mean", "gmvp_sharpe", "GMVP Sharpe", True),
    ("gmvp_mean", "gmvp_var", "GMVP variance", False),
    ("gmvp_mean", "gmvp_vol", "GMVP vol", False),
    ("gmvp_mean", "turnover_l1", "Turnover L1", False),
]


def plot_report_summary(report_path: str, outdir: str) -> None:
    """
    Visualize report.csv: key metrics only, with (↑)/(↓) and best bar in green, worst in red.
    Saves report_summary.png in outdir.
    """
    r = pd.read_csv(report_path)
    if r.empty or "metric" not in r.columns or "method" not in r.columns or "value" not in r.columns:
        return
    piv = r.pivot_table(index=["section", "metric"], columns="method", values="value", aggfunc="first")
    if piv.empty:
        return
    methods_order = [m for m in ["model", "mix", "roll", "pers", "shrink"] if m in piv.columns]
    if not methods_order:
        methods_order = piv.columns.tolist()
    piv = piv.reindex(columns=methods_order)
    rows = []
    for section, metric, title, higher_is_better in _REPORT_KEY_METRICS:
        key = (section, metric)
        if key not in piv.index:
            continue
        rows.append((title, higher_is_better, piv.loc[key]))
    if not rows:
        return
    n = len(rows)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, (ax, (title, higher_is_better, row)) in enumerate(zip(axes, rows)):
        vals = np.array([float(row.get(m, np.nan)) for m in methods_order])
        valid = np.isfinite(vals)
        if valid.any():
            best_idx = np.nanargmax(vals) if higher_is_better else np.nanargmin(vals)
            worst_idx = np.nanargmin(vals) if higher_is_better else np.nanargmax(vals)
        else:
            best_idx = worst_idx = 0
        colors = []
        for j in range(len(methods_order)):
            if j == best_idx:
                colors.append("#2e7d32")
            elif j == worst_idx:
                colors.append("#c62828")
            else:
                colors.append("#9e9e9e")
        direction = "↑" if higher_is_better else "↓"
        ax.bar(methods_order, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.set_title(f"{title} ({direction})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
    for j in range(len(rows), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Report: key metrics by method (green = best, red = worst)", fontsize=12, fontweight="bold", y=1.00)
    outpath = os.path.join(outdir, "report_summary.png")
    _savefig(outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/viz_regime_covariance.yaml")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    overrides = parse_overrides(args.set)
    cfg = deep_update(cfg, overrides)

    csv_path = cfg["inputs"]["csv"]
    outdir = cfg["outputs"]["outdir"]
    _ensure_dir(outdir)

    df = _read_results(csv_path)
    methods = cfg["plot"]["methods"]
    equity_methods = cfg["plot"].get("equity_methods", methods)
    metrics = cfg["plot"]["overlay_metrics"]
    roll_window = cfg["plot"]["roll_window"]

    plot_equity_curves(df, outdir, equity_methods)
    plot_method_overlays(df, outdir, methods, metrics)
    plot_rolling_median(df, outdir, methods, metrics, roll_window)
    plot_gmvp_sharpe(df, outdir, methods, roll_window=roll_window)
    plot_gmvp_sharpe_distribution(df, outdir, methods)
    plot_turnover_l1(df, outdir, methods, roll_window=roll_window)

    error_metric = cfg["plot"].get("error_metric", "fro")
    plot_covariance_error_timeseries(df, outdir, methods, error_metric=error_metric)

    # Cumulative advantage only for the error metric (fro / vol_mse)
    cum_metric = cfg["plot"].get("cumulative_advantage_metric", error_metric)
    for ref in ("model", "mix"):
        if ref in methods:
            plot_cumulative_advantage(df, outdir, ref=ref, methods=methods, metric=cum_metric)

    # Report summary: visualize report.csv (mean metrics by method) as bar charts
    report_path = os.path.join(os.path.dirname(os.path.dirname(outdir)), "report.csv")
    if os.path.isfile(report_path):
        plot_report_summary(report_path, outdir)

    print("\nSaved figures:")
    for f in SAVED_FILES:
        print("  ", f)


if __name__ == "__main__":
    main()

