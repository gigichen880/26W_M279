"""
Implementation moved from scripts/analysis/visualize_transition_matrix.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.analysis.utils.paths import resolve_backtest_path, resolve_figs_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"


def _default_paths(target: str):
    if target == "volatility":
        return resolve_backtest_path("regime_volatility"), resolve_figs_dir("regime_volatility")
    return resolve_backtest_path("regime_covariance"), resolve_figs_dir("regime_covariance")


def extract_transition_matrix(backtest_results_path=None):
    if backtest_results_path is None:
        backtest_results_path = resolve_backtest_path("regime_covariance")
    path = Path(backtest_results_path)
    if not path.exists():
        raise FileNotFoundError(f"Backtest file not found: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df = df.sort_values("date").reset_index(drop=True)
    regimes = df["regime_assigned"].values
    n_regimes = 4
    transition_counts = np.zeros((n_regimes, n_regimes))
    for i in range(len(regimes) - 1):
        a, b = regimes[i], regimes[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        from_regime = int(a)
        to_regime = int(b)
        if 0 <= from_regime < n_regimes and 0 <= to_regime < n_regimes:
            transition_counts[from_regime, to_regime] += 1
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.where(row_sums > 0, transition_counts / row_sums, 0.0)
    transition_matrix = np.nan_to_num(transition_matrix, nan=0.0)
    return transition_matrix


def plot_transition_matrix(transition_matrix, regime_names=None, save_path=None):
    if save_path is None:
        save_path = resolve_figs_dir("regime_covariance") / "transition_matrix_heatmap.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if regime_names is None:
        regime_names = ["Regime 0", "Regime 1", "Regime 2", "Regime 3"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Transition Probability"},
        ax=ax,
    )
    ax.set_xlabel("To Regime (t)", fontsize=12, fontweight="bold")
    ax.set_ylabel("From Regime (t-1)", fontsize=12, fontweight="bold")
    ax.set_title("Regime Transition Matrix (Markov Chain)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticklabels(regime_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(regime_names, rotation=0, fontsize=10)
    for i in range(4):
        rect = plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="red", linewidth=2.5)
        ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Transition matrix from backtest (cov or vol)")
    ap.add_argument("--input", default=None, help="Backtest parquet/csv")
    ap.add_argument("--target", choices=("auto", "covariance", "volatility"), default="auto")
    args = ap.parse_args()

    if args.input is not None:
        backtest_path = Path(args.input)
        target = args.target if args.target != "auto" else ("volatility" if "volatility" in str(args.input) else "covariance")
        _, figs_dir = _default_paths(target)
    else:
        target = args.target if args.target != "auto" else "covariance"
        backtest_path, figs_dir = _default_paths(target)
        if not backtest_path.exists() and args.target == "auto":
            backtest_path, figs_dir = _default_paths("volatility")
    figs_dir.mkdir(parents=True, exist_ok=True)

    transition_matrix = extract_transition_matrix(backtest_path)
    regime_names = ["Calm Bull", "Moderate Bull", "Normal", "High Stress"]
    plot_transition_matrix(transition_matrix, regime_names, save_path=figs_dir / "transition_matrix_heatmap.png")

    df_matrix = pd.DataFrame(
        transition_matrix,
        index=[f"From_{name}" for name in regime_names],
        columns=[f"To_{name}" for name in regime_names],
    )
    csv_name = "transition_matrix_volatility.csv" if "volatility" in str(backtest_path) else "transition_matrix.csv"
    out_dir = RESULTS_DIR / ("regime_volatility" if "volatility" in str(backtest_path) else "regime_covariance")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_matrix.to_csv(out_dir / csv_name)


if __name__ == "__main__":
    main()

