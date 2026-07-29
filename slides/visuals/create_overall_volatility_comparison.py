"""
Create 3-panel grouped bar chart for a volatility Slide.

Reads per-date volatility backtest metrics from:
  results/regime_volatility/backtest.csv

Aggregates by method by taking the mean across all rows (dates).

Output:
  slides/visuals/overall_volatility_comparison.png (12×5 in, 300 dpi)
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_overall_volatility_chart(
    csv_path: str | Path | None = None,
    *,
    eval_start: str = "2013-01-01",
    eval_end: str = "2021-12-31",
) -> Path:
    methods = ["Roll", "Pers", "Shrink", "Mix", "Model"]
    prefixes = {
        "Roll": "roll",
        "Pers": "pers",
        "Shrink": "shrink",
        "Mix": "mix",
        "Model": "model",
    }

    # Colors (match create_overall_results.py)
    colors = {
        "Roll": "#95A5A6",
        "Pers": "#E74C3C",
        "Shrink": "#3498DB",
        "Mix": "#F39C12",
        "Model": "#2ECC71",
    }
    bar_colors = [colors[m] for m in methods]

    repo_root = Path(__file__).resolve().parents[2]
    if csv_path is None:
        csv_path = repo_root / "results" / "regime_volatility" / "backtest.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing volatility backtest CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        date_min_raw = df["date"].min()
        date_max_raw = df["date"].max()

        eval_start_dt = pd.to_datetime(eval_start)
        eval_end_dt = pd.to_datetime(eval_end)
        df = df.loc[(df["date"] >= eval_start_dt) & (df["date"] <= eval_end_dt)].copy()

        date_min = df["date"].min()
        date_max = df["date"].max()
        n_dates = int(df["date"].nunique())
    else:
        date_min = None
        date_max = None
        n_dates = int(len(df))

    # Primary metrics:
    # - Panel 1: vol_mse (lower is better)
    # - Panel 2: vol_rmse (lower is better)
    vol_mse_vals = []
    vol_rmse_vals = []
    for m in methods:
        p = prefixes[m]
        c_mse = f"{p}_vol_mse"
        c_rmse = f"{p}_vol_rmse"
        for c in (c_mse, c_rmse):
            if c not in df.columns:
                raise KeyError(f"Missing column '{c}' in {csv_path.name}")

        vol_mse_vals.append(pd.to_numeric(df[c_mse], errors="coerce").mean())
        vol_rmse_vals.append(pd.to_numeric(df[c_rmse], errors="coerce").mean())

    vol_mse_vals = np.asarray(vol_mse_vals, dtype=float)
    vol_rmse_vals = np.asarray(vol_rmse_vals, dtype=float)

    best_mse_idx = int(np.nanargmin(vol_mse_vals))
    best_rmse_idx = int(np.nanargmin(vol_rmse_vals))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.patch.set_facecolor("white")
    x = np.arange(len(methods))

    # Panel 1: Vol MSE (lower is better)
    ax = axes[0]
    bars = ax.bar(
        x,
        vol_mse_vals,
        color=bar_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
    )
    bars[best_mse_idx].set_edgecolor("gold")
    bars[best_mse_idx].set_linewidth(3)

    for bar, val in zip(bars, vol_mse_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0008 * max(1.0, np.nanmax(vol_mse_vals)),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Volatility MSE", fontsize=11)
    ax.set_title("Vol MSE\n(lower is better)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.0, float(np.nanmax(vol_mse_vals)) * 1.20)

    # Panel 2: Vol RMSE (lower is better)
    ax = axes[1]
    bars = ax.bar(
        x,
        vol_rmse_vals,
        color=bar_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
    )
    bars[best_rmse_idx].set_edgecolor("gold")
    bars[best_rmse_idx].set_linewidth(3)

    for bar, val in zip(bars, vol_rmse_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005 * max(1.0, float(np.nanmax(vol_rmse_vals))),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Volatility RMSE", fontsize=11)
    ax.set_title("Vol RMSE\n(lower is better)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.0, float(np.nanmax(vol_rmse_vals)) * 1.20)

    if date_min is not None and date_max is not None:
        title = f"Volatility Performance Comparison: All Methods ({date_min.year}-{date_max.year}, {n_dates} dates)"
    else:
        title = f"Volatility Performance Comparison: All Methods ({n_dates} rows)"

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "overall_volatility_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {out_path}")
    return Path(out_path)


if __name__ == "__main__":
    create_overall_volatility_chart()

