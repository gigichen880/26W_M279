"""
Pareto frontier for joint grid ablations (covariance backtest summary).

Reads ablation_summary.csv from run_ablation in mode: grid. Uses **model** metrics:
  - maximize model_gmvp_sharpe_mean
  - minimize model_gmvp_var_mean
  - minimize model_turnover_l1_mean

Writes:
  - pareto_frontier.csv  (nondominated rows + is_pareto column on full table optional)
  - 2D/3D scatter figures highlighting the Pareto set

Usage:
  python -m scripts.analysis.ablation.pareto_gmvp_report \\
      --summary results/regime_covariance/ablation_joint_gmvp/ablation_summary.csv
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

COL_SHARPE = "model_gmvp_sharpe_mean"
COL_VAR = "model_gmvp_var_mean"
COL_TURN = "model_turnover_l1_mean"
COL_FRO = "model_mean"


def _to_matrix_for_pareto(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Higher row value = better for all columns (maximize)."""
    s = pd.to_numeric(df[COL_SHARPE], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[COL_VAR], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df[COL_TURN], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(s) & np.isfinite(v) & np.isfinite(t)
    Y = np.column_stack([s, -v, -t]).astype(float)
    return Y, valid


def pareto_efficient_mask(Y: np.ndarray) -> np.ndarray:
    """
    Y shape (n, m), all objectives **maximize**. Row i is Pareto-efficient if no j
    weakly dominates i with a strict improvement in at least one coordinate.
    """
    n = Y.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if np.any(np.isnan(Y[i])):
            is_eff[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if np.any(np.isnan(Y[j])):
                continue
            if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                is_eff[i] = False
                break
    return is_eff


def run_report(summary_csv: str, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)
    grid = df[df.get("mode") == "grid"].copy()
    if grid.empty:
        raise ValueError("No rows with mode=='grid' in summary (use ablation mode: grid).")

    Y, valid = _to_matrix_for_pareto(grid)
    sub = grid.loc[valid].reset_index(drop=True)
    Y = Y[valid]
    mask = pareto_efficient_mask(Y)
    sub = sub.assign(is_pareto=mask)

    full_out = os.path.join(out_dir, "grid_with_pareto_flags.csv")
    # merge flags back to grid rows by run_tag
    flag_map = dict(zip(sub["run_tag"], sub["is_pareto"]))
    grid_flagged = grid.copy()
    grid_flagged["is_pareto"] = grid_flagged["run_tag"].map(flag_map).fillna(False)
    grid_flagged.to_csv(full_out, index=False)

    front = sub[sub["is_pareto"]].copy()
    front_path = os.path.join(out_dir, "pareto_frontier.csv")
    front.to_csv(front_path, index=False)

    # ---- plots ----
    s = pd.to_numeric(sub[COL_SHARPE], errors="coerce")
    v = pd.to_numeric(sub[COL_VAR], errors="coerce")
    t = pd.to_numeric(sub[COL_TURN], errors="coerce")

    # 2a: Sharpe vs turnover, color variance
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(t, s, c=v, cmap="viridis", alpha=0.85, edgecolors="k", linewidths=0.4, s=55)
    ax.scatter(t[mask], s[mask], s=120, facecolors="none", edgecolors="crimson", linewidths=2.0, label="Pareto")
    plt.colorbar(sc, ax=ax, label="model GMVP var (mean, ↓ better)")
    ax.set_xlabel("model turnover L1 (mean, ↓ better)")
    ax.set_ylabel("model GMVP Sharpe (mean, ↑ better)")
    ax.set_title("Joint grid: Sharpe vs turnover (color = variance)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_2d_sharpe_vs_turnover.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 2b: Sharpe vs variance, color turnover
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(v, s, c=t, cmap="plasma", alpha=0.85, edgecolors="k", linewidths=0.4, s=55)
    ax.scatter(v[mask], s[mask], s=120, facecolors="none", edgecolors="crimson", linewidths=2.0, label="Pareto")
    plt.colorbar(sc, ax=ax, label="model turnover L1 (mean)")
    ax.set_xlabel("model GMVP var (mean, ↓ better)")
    ax.set_ylabel("model GMVP Sharpe (mean, ↑ better)")
    ax.set_title("Joint grid: Sharpe vs variance (color = turnover)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_2d_sharpe_vs_var.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(t, v, s, c="steelblue", alpha=0.5, s=40, depthshade=True)
    ax.scatter(t[mask], v[mask], s[mask], c="crimson", s=80, marker="o", edgecolors="k", linewidths=0.5, label="Pareto")
    ax.set_xlabel("turnover L1 (↓)")
    ax.set_ylabel("GMVP var (↓)")
    ax.set_zlabel("GMVP Sharpe (↑)")
    ax.set_title("Pareto frontier (3 objectives, model)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_3d_sharpe_var_turnover.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Optional: Fro on Pareto points (diagnostic)
    if COL_FRO in sub.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        fro = pd.to_numeric(sub[COL_FRO], errors="coerce")
        ax.bar(range(len(sub)), fro, color=["crimson" if m else "steelblue" for m in mask], alpha=0.85)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["run_tag"].astype(str), rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("mean Fro (↓)")
        ax.set_title("Mean Fro by grid point (red = Pareto w.r.t Sharpe/var/turnover)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pareto_diagnostic_fro_by_run.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {full_out}")
    print(f"Wrote {front_path} ({len(front)} Pareto points)")
    print(f"Figures in {out_dir}")
    return front


def main() -> None:
    ap = argparse.ArgumentParser(description="Pareto report for joint GMVP grid ablation")
    ap.add_argument(
        "--summary",
        default=None,
        help="ablation_summary.csv from run_ablation (grid mode)",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <summary_dir>/pareto)",
    )
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    summary = args.summary or os.path.join(
        root, "results", "regime_covariance", "ablation_joint_gmvp", "ablation_summary.csv"
    )
    if not os.path.isfile(summary):
        print(f"Missing {summary}", file=sys.stderr)
        sys.exit(1)
    out_dir = args.outdir or os.path.join(os.path.dirname(summary), "pareto")
    run_report(summary, out_dir)


if __name__ == "__main__":
    main()
