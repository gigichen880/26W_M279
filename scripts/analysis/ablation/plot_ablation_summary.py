"""
Plot ablation visuals from an existing ablation_summary.csv (no reruns).

Outputs one figure per ablation axis (e.g. embedder / knn / transition):
  - model-only comparison across that axis' choices
  - grouped bars per metric (each metric is one "bar group")

Usage:
  MPLBACKEND=Agg python -m scripts.analysis.ablation.plot_ablation_summary \\
    --input results/regime_covariance/ablation/ablation_summary.csv \\
    --outdir results/regime_covariance/figs/ablation \\
    --target covariance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_axis_choice_bars_by_metric(
    *,
    sub: pd.DataFrame,
    axis_name: str,
    outpath: Path,
    target_type: str,
    primary_metric: str,
) -> None:
    import matplotlib.pyplot as plt

    if sub.empty:
        return
    sub = sub.copy().sort_values("choice")
    choices = sub["choice"].astype(str).tolist()

    if target_type == "volatility":
        metrics = [
            ("model_mean", f"Mean {primary_metric} (↓)"),
        ]
    else:
        metrics = [
            ("model_gmvp_sharpe_mean", "GMVP Sharpe (↑)"),
            ("model_turnover_l1_mean", "Turnover L1 (↓)"),
            ("model_mean", f"{primary_metric.upper()} (↓)"),
            ("model_stein_mean", "Stein (↓)"),
            ("model_logeuc_mean", "LogEuc (↓)"),
        ]

    metric_cols = [(c, t) for c, t in metrics if c in sub.columns]
    if not metric_cols:
        return

    C = len(choices)

    # Subplots per metric (independent y-scale)
    M = len(metric_cols)
    fig, axs = plt.subplots(1, M, figsize=(3.4 * M, 3.6), sharey=False)
    if M == 1:
        axs = [axs]

    group_w = 0.72
    bar_w = group_w / max(C, 1)
    offsets = (np.arange(C) - (C - 1) / 2.0) * bar_w
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for j, (col, title) in enumerate(metric_cols):
        ax = axs[j]
        for i, choice in enumerate(choices):
            row = sub.iloc[i]
            y = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            ax.bar(0 + offsets[i], y, width=bar_w * 0.95, label=choice, color=colors[i % len(colors)], alpha=0.9)

        # Baseline segments for Sharpe only
        if col == "model_gmvp_sharpe_mean":
            for b, bcol, colr in [
                ("roll", "roll_gmvp_sharpe_mean", "#8e8e8e"),
                ("shrink", "shrink_gmvp_sharpe_mean", "#6d6d6d"),
                ("pers", "pers_gmvp_sharpe_mean", "#3a3a3a"),
            ]:
                if bcol in sub.columns:
                    yb = float(pd.to_numeric(sub[bcol], errors="coerce").iloc[0])
                    if np.isfinite(yb):
                        ax.hlines(yb, -group_w / 2, group_w / 2, colors=colr, linestyles="--", linewidth=1.2, label=b)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels([""])
        ax.grid(alpha=0.25, axis="y")

    # One shared legend for choices (and baselines if present)
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), bbox_to_anchor=(0.5, 1.18), fontsize=9)
    fig.suptitle(f"{axis_name}: model-only comparison across choices", y=1.05, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ablation visuals from ablation_summary.csv (no reruns).")
    ap.add_argument("--input", default="results/regime_covariance/ablation/ablation_summary.csv")
    ap.add_argument("--outdir", default="results/regime_covariance/figs/ablation")
    ap.add_argument("--target", default="auto", choices=("auto", "covariance", "volatility"))
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Missing input: {inp}")
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    df = pd.read_csv(inp)
    # Support both one_at_a_time-only summaries and mixed summaries
    if "mode" in df.columns:
        df_oaat = df[df["mode"].astype(str) == "one_at_a_time"].copy()
    else:
        df_oaat = df.copy()

    target = str(args.target).lower()
    if target == "auto":
        target = "volatility" if any(c.startswith("model_vol_") for c in df.columns) else "covariance"

    primary_metric = "vol_mse" if target == "volatility" else "fro"

    # Create per-axis figures
    if {"axis", "choice"}.issubset(set(df_oaat.columns)):
        for axis_name in sorted(df_oaat["axis"].dropna().unique().tolist()):
            sub = df_oaat[df_oaat["axis"] == axis_name].copy()
            if sub.empty:
                continue
            outpath = outdir / f"axis_pairwise_bars_{axis_name}.png"
            _plot_axis_choice_bars_by_metric(
                sub=sub,
                axis_name=str(axis_name),
                outpath=outpath,
                target_type=target,
                primary_metric=primary_metric,
            )

    print(f"Saved ablation figures to: {outdir}")


if __name__ == "__main__":
    main()

