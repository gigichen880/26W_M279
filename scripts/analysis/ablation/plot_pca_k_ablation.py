"""
Plot pca_k slice from ablation_summary.csv (model metrics only).

Usage:
  python -m scripts.analysis.ablation.plot_pca_k_ablation
  python -m scripts.analysis.ablation.plot_pca_k_ablation \\
      --summary results/regime_covariance/ablation_pca_k/ablation_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pca_k_summary(summary_csv: str, out_png: str) -> None:
    df = pd.read_csv(summary_csv)
    sub = df[(df["axis"] == "pca_k") & (df["mode"] == "one_at_a_time")].copy()
    if sub.empty:
        raise ValueError("No pca_k rows in summary (check axis name and CSV path).")

    sub["choice"] = sub["choice"].astype(str)
    sub["_k"] = pd.to_numeric(
        sub["choice"].str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    sub = sub.sort_values("_k")

    labels = sub["choice"].tolist()
    x = np.arange(len(labels))

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    panels = [
        (axs[0, 0], "model_mean", "Mean Frobenius error (↓ better)", True),
        (axs[0, 1], "model_gmvp_sharpe_mean", "Mean GMVP Sharpe (↑ better)", False),
        (axs[1, 0], "model_stein_mean", "Mean Stein loss (↓ better)", True),
        (axs[1, 1], "model_turnover_l1_mean", "Mean GMVP turnover L1 (↓ often better)", True),
    ]
    for ax, col, title, lower_better in panels:
        y = pd.to_numeric(sub[col], errors="coerce").values
        ax.bar(x, y, color="steelblue", alpha=0.9, width=0.72)
        if lower_better:
            j = int(np.nanargmin(y))
            ax.axvline(j, color="crimson", linestyle="--", alpha=0.5, linewidth=1)
        else:
            finite = np.where(np.isfinite(y))[0]
            if len(finite):
                j = finite[int(np.nanargmax(y[finite]))]
                ax.axvline(j, color="crimson", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("PCA dimension ablation (model only)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pca_k ablation from ablation_summary.csv")
    ap.add_argument(
        "--summary",
        default=None,
        help="Path to ablation_summary.csv (default: results/.../ablation_pca_k/ablation_summary.csv)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: <summary_dir>/figs/pca_k_ablation_model_only.png)",
    )
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    csv_path = args.summary or os.path.join(
        root, "results", "regime_covariance", "ablation_pca_k", "ablation_summary.csv"
    )
    if not os.path.isfile(csv_path):
        print(f"Missing {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_png = args.out
    else:
        fig_dir = os.path.join(os.path.dirname(csv_path), "figs")
        os.makedirs(fig_dir, exist_ok=True)
        out_png = os.path.join(fig_dir, "pca_k_ablation_model_only.png")

    plot_pca_k_summary(csv_path, out_png)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
