"""
Model-only 4-panel figures for each one-at-a-time ablation axis (Fro, GMVP Sharpe, Stein, turnover).

Writes under <summary_dir>/figs/:
  axis_<axis_name>_model_only.png

Used by run_ablation when plots=summary|all, or run standalone:

  python -m scripts.analysis.ablation.plot_oat_ablation_axes --summary path/to/ablation_summary.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sort_subframe(sub: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by numeric choice if possible, else alphabetically."""
    sub = sub.copy()
    sub["choice"] = sub["choice"].astype(str)
    nums = []
    for c in sub["choice"]:
        m = re.match(r"^[-+]?[0-9]*\.?[0-9]+", str(c).strip())
        nums.append(float(m.group(0)) if m else np.nan)
    sub["_num"] = nums
    if sub["_num"].notna().all():
        sub = sub.sort_values("_num")
    else:
        sub = sub.sort_values("choice")
    return sub.drop(columns=["_num"], errors="ignore")


def plot_oat_axes_from_summary(summary_csv: str, figs_dir: str | None = None) -> list[str]:
    """
    One 4-panel PNG per axis (mode one_at_a_time only).
    Returns list of written paths.
    """
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(summary_csv)
    df = pd.read_csv(summary_csv)
    oat = df[df.get("mode") == "one_at_a_time"].copy()
    if oat.empty or "axis" not in oat.columns:
        return []

    figs_dir = figs_dir or os.path.join(os.path.dirname(summary_csv), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    written: list[str] = []
    axes_names = sorted(oat["axis"].dropna().unique().tolist())

    panels = [
        ("model_mean", "Mean Frobenius error (↓ better)", True),
        ("model_gmvp_sharpe_mean", "Mean GMVP Sharpe (↑ better)", False),
        ("model_stein_mean", "Mean Stein loss (↓ better)", True),
        ("model_turnover_l1_mean", "Mean GMVP turnover L1 (↓ better)", True),
    ]

    for axis_name in axes_names:
        sub = oat[oat["axis"] == axis_name].copy()
        if sub.empty:
            continue
        sub = _sort_subframe(sub)
        labels = sub["choice"].tolist()
        x = np.arange(len(labels))

        fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
        flat = axs.ravel()
        for ax, (col, title, lower_better) in zip(flat, panels):
            if col not in sub.columns:
                ax.set_visible(False)
                continue
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

        # Roll Sharpe reference on Sharpe panel (same across choices for OAT)
        if "roll_gmvp_sharpe_mean" in sub.columns and np.isfinite(sub["roll_gmvp_sharpe_mean"].iloc[0]):
            r_sh = float(sub["roll_gmvp_sharpe_mean"].iloc[0])
            axs[0, 1].axhline(r_sh, color="#888", linestyle="--", linewidth=1, label="roll (ref)")
            axs[0, 1].legend(loc="best", fontsize=8)

        safe = str(axis_name).replace("/", "_").replace(" ", "_")
        out_png = os.path.join(figs_dir, f"axis_{safe}_model_only.png")
        fig.suptitle(f"Ablation: {axis_name} — model only (one-at-a-time)", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(out_png)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-only 4-panel plots per OAT ablation axis")
    ap.add_argument("--summary", default=None, help="ablation_summary.csv path")
    ap.add_argument(
        "--figs-dir",
        default=None,
        help="Output directory for PNGs (default: <summary_dir>/figs)",
    )
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    summary = args.summary or os.path.join(
        root, "results", "regime_covariance", "ablation_phase1_covariance", "ablation_summary.csv"
    )
    paths = plot_oat_axes_from_summary(summary, figs_dir=args.figs_dir)
    if not paths:
        print("No one_at_a_time rows to plot (grid-only summary or empty).", file=sys.stderr)
        sys.exit(0)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
