# Moved implementation from scripts/analysis/visualize_backtest_results.py

from __future__ import annotations

import os
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_gmvp_sharpe(df, outdir, methods):
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
        plt.plot(df[c], label=m, alpha=0.9 if m in {"model", "mix"} else 0.6)
    plt.title("GMVP Sharpe over time")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.legend()
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


def plot_cumulative_advantage(df, outdir, ref: str, methods: List[str], metric: str = "fro"):
    ref_col = f"{ref}_{metric}"
    if ref_col not in df.columns:
        return
    ref_s = pd.to_numeric(df[ref_col], errors="coerce")
    baselines = [m for m in methods if m != ref]
    plt.figure(figsize=(12, 4))
    for m in baselines:
        c = f"{m}_{metric}"
        if c not in df.columns:
            continue
        base_s = pd.to_numeric(df[c], errors="coerce")
        diff = base_s - ref_s
        cum = diff.cumsum()
        plt.plot(cum.index, cum.values, label=f"{ref} vs {m}")
    plt.axhline(0, color="black", linestyle="--", alpha=0.7)
    plt.title(f"Cumulative advantage: {ref} vs baselines ({metric}, positive = {ref} better)")
    plt.ylabel("Cumulative difference")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(os.path.join(outdir, f"cumulative_advantage_{ref}_{metric}.png"))


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
    metrics = cfg["plot"]["overlay_metrics"]
    roll_window = cfg["plot"]["roll_window"]

    plot_equity_curves(df, outdir, methods)
    plot_method_overlays(df, outdir, methods, metrics)
    plot_rolling_median(df, outdir, methods, metrics, roll_window)
    plot_gmvp_sharpe(df, outdir, methods)
    plot_turnover_l1(df, outdir, methods)

    error_metric = cfg["plot"].get("error_metric", "fro")
    plot_covariance_error_timeseries(df, outdir, methods, error_metric=error_metric)

    cum_metric = cfg["plot"].get("cumulative_advantage_metric", error_metric)
    for ref in ("model", "mix"):
        if ref in methods:
            plot_cumulative_advantage(df, outdir, ref=ref, methods=methods, metric=cum_metric)

    print("\nSaved figures:")
    for f in SAVED_FILES:
        print("  ", f)


if __name__ == "__main__":
    main()

