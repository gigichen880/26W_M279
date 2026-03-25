"""
Plot regime_clustering slice from ablation_summary.csv (model metrics only).

Usage:
  python -m scripts.analysis.ablation.plot_regime_clustering_ablation
  python -m scripts.analysis.ablation.plot_regime_clustering_ablation \\
      --summary results/regime_covariance/ablation_regime_clustering/ablation_summary.csv
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


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def plot_regime_clustering_summary(
    csv_path: str,
    out_png: str,
    *,
    order: list[str] | None = None,
) -> None:
    if order is None:
        from similarity_forecast.regime_clustering import implemented_regime_clustering_names

        order = implemented_regime_clustering_names()

    df = pd.read_csv(csv_path)
    sub = df[(df["axis"] == "regime_clustering") & (df["mode"] == "one_at_a_time")].copy()
    if sub.empty:
        raise ValueError("No regime_clustering rows in summary (check axis name and CSV path).")

    sub["choice"] = sub["choice"].astype(str)
    order_idx = {c: i for i, c in enumerate(order)}
    sub["_ord"] = sub["choice"].map(lambda x: order_idx.get(x, 99))
    sub = sub.sort_values("_ord")

    labels = sub["choice"].tolist()
    x = np.arange(len(labels))

    d = os.path.dirname(out_png)
    if d:
        os.makedirs(d, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        (axs[0, 0], "model_mean", "Mean Frobenius error (↓ better)", True),
        (axs[0, 1], "model_gmvp_sharpe_mean", "Mean GMVP Sharpe (↑ better)", False),
        (axs[1, 0], "model_stein_mean", "Mean Stein loss (↓ better)", True),
        (axs[1, 1], "model_turnover_l1_mean", "Mean GMVP turnover L1 (↓ often better)", True),
    ]
    for ax, col, title, lower_better in panels:
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        y = pd.to_numeric(sub[col], errors="coerce").values
        ax.bar(x, y, color="darkgreen", alpha=0.85, width=0.72)
        finite = np.isfinite(y)
        if finite.any():
            yi = y[finite]
            if lower_better:
                best_rel = int(np.nanargmin(yi))
                best_i = int(np.flatnonzero(finite)[best_rel])
            else:
                best_rel = int(np.nanargmax(yi))
                best_i = int(np.flatnonzero(finite)[best_rel])
            if np.isfinite(y[best_i]):
                ax.text(best_i, y[best_i], "★", ha="center", va="bottom", fontsize=11, color="navy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    if "roll_gmvp_sharpe_mean" in sub.columns:
        r_sh = float(pd.to_numeric(sub["roll_gmvp_sharpe_mean"], errors="coerce").iloc[0])
        if np.isfinite(r_sh):
            axs[0, 1].axhline(r_sh, color="#888", linestyle="--", linewidth=1, label="roll (ref)")
            axs[0, 1].legend(loc="best", fontsize=8)

    fig.suptitle("Ablation: regime clustering (model only, one-at-a-time)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot regime_clustering ablation from summary CSV")
    ap.add_argument("--summary", default=None, help="Path to ablation_summary.csv")
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: <summary_dir>/figs/regime_clustering_model_only.png)",
    )
    args = ap.parse_args()

    root = _repo_root()
    csv_path = args.summary or os.path.join(
        root,
        "results",
        "regime_covariance",
        "ablation_regime_clustering",
        "ablation_summary.csv",
    )
    if not os.path.isfile(csv_path):
        print(f"Missing {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_png = args.out
    else:
        fig_dir = os.path.join(os.path.dirname(csv_path), "figs")
        os.makedirs(fig_dir, exist_ok=True)
        out_png = os.path.join(fig_dir, "regime_clustering_model_only.png")

    plot_regime_clustering_summary(csv_path, out_png)
    print(out_png)


if __name__ == "__main__":
    main()
