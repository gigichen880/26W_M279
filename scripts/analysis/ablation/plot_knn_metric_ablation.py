"""
Plot knn_metric slice from ablation_summary.csv (model metrics only).

Usage:
  python -m scripts.analysis.ablation.plot_knn_metric_ablation
  python -m scripts.analysis.ablation.plot_knn_metric_ablation --summary path/to/ablation_summary.csv
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

ORDER = [
    "l2",
    "l1",
    "cosine",
    "angular",
    "chebyshev",
    "lp_p1",
    "lp_p2",
    "lp_p3",
    "lp_p4",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        default=None,
        help="Path to ablation_summary.csv (default: results/regime_covariance/ablation/ablation_summary.csv)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: .../ablation/_temp_knn_metric_figs/knn_metric_ablation_summary.png)",
    )
    args = ap.parse_args()

    # ablation/ -> analysis/ -> scripts/ -> repo root
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    csv_path = args.summary or os.path.join(
        root, "results", "regime_covariance", "ablation", "ablation_summary.csv"
    )
    if not os.path.isfile(csv_path):
        print(f"Missing {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    sub = df[(df["axis"] == "knn_metric") & (df["mode"] == "one_at_a_time")].copy()
    if sub.empty:
        print("No knn_metric rows in summary.", file=sys.stderr)
        sys.exit(1)

    sub["choice"] = sub["choice"].astype(str)
    order_idx = {c: i for i, c in enumerate(ORDER)}
    sub["_ord"] = sub["choice"].map(lambda x: order_idx.get(x, 99))
    sub = sub.sort_values("_ord")

    labels = sub["choice"].tolist()
    x = np.arange(len(labels))

    if args.out:
        out_png = args.out
    else:
        out_dir = os.path.join(os.path.dirname(csv_path), "_temp_knn_metric_figs")
        os.makedirs(out_dir, exist_ok=True)
        # Default: same basename as before, model-only bars
        out_png = os.path.join(out_dir, "knn_metric_ablation_summary.png")

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
            best_i = int(np.nanargmin(y))
        else:
            best_i = int(np.nanargmax(y))
        if np.isfinite(y[best_i]):
            ax.text(best_i, y[best_i], "★", ha="center", va="bottom", fontsize=11, color="navy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    r_sh = float(sub["roll_gmvp_sharpe_mean"].iloc[0])
    if np.isfinite(r_sh):
        axs[0, 1].axhline(r_sh, color="#888", linestyle="--", linewidth=1, label="roll (ref)")
        axs[0, 1].legend(loc="best", fontsize=8)

    fig.suptitle("Ablation: KNN distance — model only (one-at-a-time)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
