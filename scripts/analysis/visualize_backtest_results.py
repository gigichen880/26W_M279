# scripts/analysis/viz_backtest_results.py

"""
Clean visualization for regime similarity backtests.

Produces:

1. model vs baselines win-rate grid
2. mix vs baselines win-rate grid
3. method overlays
4. rolling medians
5. equity curves

Usage:
python -m scripts.analysis.visualize_backtest_results \
  --config configs/viz_regime_similarity.yaml
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.config_utils import load_yaml, deep_update, parse_overrides


# -------------------------------------------------------
# helpers
# -------------------------------------------------------

def _ensure_dir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


SAVED_FILES = []

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


# -------------------------------------------------------
# winrate utilities
# -------------------------------------------------------

def _is_higher_better(metric: str, higher_is_better: set[str]):
    return metric in higher_is_better


def _win_series(ref, other, metric, higher_is_better):

    ok = ref.notna() & other.notna()

    w = pd.Series(np.nan, index=ref.index)

    if _is_higher_better(metric, higher_is_better):
        w.loc[ok] = (ref.loc[ok] > other.loc[ok]).astype(float)
    else:
        w.loc[ok] = (ref.loc[ok] < other.loc[ok]).astype(float)

    return w


def plot_winrate_colored(dates, winrate, ax, title):

    ax.plot(dates, winrate, linewidth=2, label="rolling winrate")

    ax.axhline(0.5, linestyle="--", color="black", alpha=0.7, label="50%")

    ax.fill_between(
        dates,
        0.5,
        winrate,
        where=(winrate > 0.5),
        color="green",
        alpha=0.25,
    )

    ax.fill_between(
        dates,
        0.5,
        winrate,
        where=(winrate <= 0.5),
        color="red",
        alpha=0.25,
    )

    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)

    ax.grid(alpha=0.3)


# -------------------------------------------------------
# equity curves
# -------------------------------------------------------

def plot_equity_curves(df, outdir, methods):

    cols = _metric_cols(df, "gmvp_cumret", methods)

    if not cols:
        return

    plt.figure(figsize=(12, 5))

    for m, c in cols.items():

        r = df[c].fillna(0.0)

        eq = np.cumprod(1 + r)

        plt.plot(eq.index, eq.values, label=m)

    plt.title("GMVP Equity Curves")

    plt.legend()

    plt.grid(alpha=0.3)

    _savefig(os.path.join(outdir, "equity_curves_gmvp.png"))


# -------------------------------------------------------
# overlay metrics
# -------------------------------------------------------

def plot_method_overlays(df, outdir, methods, metrics):

    # Optionally exclude metrics that get dedicated figures
    metrics = [m for m in metrics if m not in {"gmvp_sharpe", "turnover_l1"}]

    if not metrics:
        return

    n = len(metrics)

    fig = plt.figure(figsize=(12, 2.5 * n))

    for i, metric in enumerate(metrics):

        ax = fig.add_subplot(n, 1, i + 1)

        cols = _metric_cols(df, metric, methods)

        for m, c in cols.items():

            ax.plot(
                df[c],
                label=m,
                alpha=0.9 if m in {"model", "mix"} else 0.6
            )

        ax.set_title(metric)

        ax.grid(alpha=0.3)

        ax.legend()

    _savefig(os.path.join(outdir, "method_overlays.png"))


# -------------------------------------------------------
# rolling medians
# -------------------------------------------------------

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

            ax.plot(
                r,
                label=m,
                alpha=0.9 if m in {"model", "mix"} else 0.6
            )

        ax.set_title(f"{metric} rolling median")

        ax.legend()

        ax.grid(alpha=0.3)

    _savefig(os.path.join(outdir, f"rolling_median_{window}d.png"))


# -------------------------------------------------------
# dedicated GMVP Sharpe and turnover plots
# -------------------------------------------------------

def plot_gmvp_sharpe(df, outdir, methods):

    cols = _metric_cols(df, "gmvp_sharpe", methods)
    if not cols:
        return

    # Collect values for robust y-axis scaling
    all_vals = []
    for c in cols.values():
        v = pd.to_numeric(df[c], errors="coerce")
        all_vals.append(v.values)
    import numpy as np
    all_arr = np.concatenate(all_vals)
    finite = all_arr[np.isfinite(all_arr)]
    if finite.size > 0:
        q1 = np.percentile(finite, 5)
        q99 = np.percentile(finite, 95)
    else:
        q1, q99 = -1.0, 1.0

    plt.figure(figsize=(12, 4))
    for m, c in cols.items():
        plt.plot(df[c], label=m, alpha=0.9 if m in {"model", "mix"} else 0.6)
    plt.title("GMVP Sharpe over time")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.legend()
    # Clip y-axis to focus on the interesting range, ignoring extreme warm-up spikes
    plt.ylim(q1, q99)
    _savefig(os.path.join(outdir, "gmvp_sharpe_timeseries.png"))


def plot_turnover_l1(df, outdir, methods):

    cols = _metric_cols(df, "turnover_l1", methods)
    if not cols:
        return

    plt.figure(figsize=(12, 4))
    for m, c in cols.items():
        plt.plot(df[c], label=m, alpha=0.9 if m in {"model", "mix"} else 0.6)
    plt.title("GMVP turnover (L1) over time")
    plt.ylabel("turnover L1")
    plt.grid(alpha=0.3)
    plt.legend()
    _savefig(os.path.join(outdir, "gmvp_turnover_l1_timeseries.png"))


# -------------------------------------------------------
# winrate grid (key figure)
# -------------------------------------------------------

def plot_winrate_grid(
    df,
    outdir,
    ref,
    methods,
    metrics,
    win_window,
    higher_is_better
):

    baselines = [m for m in methods if m not in {"model", "mix"}]

    rows = len(metrics)
    cols = len(baselines)

    fig = plt.figure(figsize=(4 * cols, 3 * rows))

    plot_id = 1

    for i, metric in enumerate(metrics):

        ref_col = f"{ref}_{metric}"

        if ref_col not in df.columns:
            continue

        ref_s = df[ref_col]

        for j, m in enumerate(baselines):

            c = f"{m}_{metric}"

            if c not in df.columns:
                continue

            ax = fig.add_subplot(rows, cols, plot_id)

            plot_id += 1

            win = _win_series(ref_s, df[c], metric, higher_is_better)

            wr = win.rolling(win_window).mean()

            plot_winrate_colored(
                wr.index,
                wr.values,
                ax,
                f"{ref} vs {m} ({metric})"
            )

            if j == 0:
                ax.set_ylabel("win rate")

            if i == rows - 1:
                ax.set_xlabel("date")

    fig.suptitle(
        f"{win_window}d rolling win-rate: {ref} vs roll / pers / shrink",
        fontsize=16,
        y=0.995
    )
    _savefig(os.path.join(outdir, f"rolling_winrate_{ref}.png"))


# -------------------------------------------------------
# cumulative win counts (since start)
# -------------------------------------------------------

def plot_cumulative_wins(
    df,
    outdir,
    ref,
    methods,
    metrics,
    higher_is_better,
):
    """
    For each metric and baseline, plot cumulative number of dates up to t
    where ref beats the baseline on that metric.
    """

    baselines = [m for m in methods if m not in {"model", "mix"}]

    rows = len(metrics)
    cols = len(baselines)

    fig = plt.figure(figsize=(4 * cols, 3 * rows))

    plot_id = 1

    for i, metric in enumerate(metrics):

        ref_col = f"{ref}_{metric}"
        if ref_col not in df.columns:
            continue

        ref_s = df[ref_col]

        for j, m in enumerate(baselines):

            c = f"{m}_{metric}"
            if c not in df.columns:
                continue

            ax = fig.add_subplot(rows, cols, plot_id)
            plot_id += 1

            win = _win_series(ref_s, df[c], metric, higher_is_better)
            cum_wins = win.cumsum()

            ax.plot(cum_wins.index, cum_wins.values, linewidth=2)

            ax.set_title(f"{ref} vs {m} ({metric})", fontsize=9)
            ax.grid(alpha=0.3)

            if j == 0:
                ax.set_ylabel("# wins so far")
            if i == rows - 1:
                ax.set_xlabel("date")

    fig.suptitle(
        f"Cumulative wins: {ref} vs roll / pers / shrink",
        fontsize=16,
        y=0.995
    )
    _savefig(os.path.join(outdir, f"cumulative_wins_{ref}.png"))


# -------------------------------------------------------
# main
# -------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/viz_regime_similarity.yaml"
    )

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

    metrics = cfg["plot"]["overlay_metrics"]

    higher_is_better = set(cfg["plot"]["higher_is_better"])

    roll_window = cfg["plot"]["roll_window"]

    win_window = cfg["plot"]["winroll_window"]

    print("methods:", methods)

    plot_equity_curves(df, outdir, methods)

    plot_method_overlays(df, outdir, methods, metrics)

    plot_rolling_median(df, outdir, methods, metrics, roll_window)

    # dedicated Sharpe / turnover plots
    plot_gmvp_sharpe(df, outdir, methods)
    plot_turnover_l1(df, outdir, methods)

    # -------------------------
    # winrate figures
    # -------------------------

    if "model" in methods:

        plot_winrate_grid(
            df,
            outdir,
            ref="model",
            methods=methods,
            metrics=metrics,
            win_window=win_window,
            higher_is_better=higher_is_better
        )
        plot_cumulative_wins(
            df,
            outdir,
            ref="model",
            methods=methods,
            metrics=metrics,
            higher_is_better=higher_is_better,
        )

    if "mix" in methods:

        plot_winrate_grid(
            df,
            outdir,
            ref="mix",
            methods=methods,
            metrics=metrics,
            win_window=win_window,
            higher_is_better=higher_is_better
        )
        plot_cumulative_wins(
            df,
            outdir,
            ref="mix",
            methods=methods,
            metrics=metrics,
            higher_is_better=higher_is_better,
        )

    print("\nSaved figures:")

    for f in SAVED_FILES:
        print("  ", f)


if __name__ == "__main__":
    main()