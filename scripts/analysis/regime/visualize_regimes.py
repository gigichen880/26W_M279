"""
Implementation moved from scripts/analysis/visualize_regimes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.analysis.utils.paths import resolve_backtest_path, resolve_figs_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"


def _default_backtest_and_outdir(target: str):
    if target == "volatility":
        p = resolve_backtest_path("regime_volatility")
        return p, resolve_figs_dir("regime_volatility") / "regime"
    p = resolve_backtest_path("regime_covariance")
    return p, resolve_figs_dir("regime_covariance") / "regime"


def _detect_target_from_path(backtest_path: str | Path) -> str:
    return "volatility" if "volatility" in str(backtest_path).lower() else "covariance"


DEFAULT_BACKTEST = resolve_backtest_path("regime_covariance")
DEFAULT_OUTDIR = resolve_figs_dir("regime_covariance") / "regime"


def load_regime_data(backtest_file: str | Path = DEFAULT_BACKTEST) -> pd.DataFrame:
    path = Path(backtest_file)
    if not path.exists():
        raise FileNotFoundError(f"Backtest file not found: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def identify_crisis_periods() -> list[dict]:
    return [
        {"start": "2007-12-01", "end": "2009-06-30", "label": "Financial Crisis"},
        {"start": "2011-08-01", "end": "2011-10-31", "label": "EU Debt Crisis"},
        {"start": "2015-08-01", "end": "2016-02-29", "label": "China Selloff"},
        {"start": "2018-10-01", "end": "2018-12-31", "label": "Q4 Selloff"},
        {"start": "2020-02-20", "end": "2020-04-30", "label": "COVID Crash"},
        {"start": "2020-09-01", "end": "2020-10-31", "label": "Election Vol"},
    ]


def plot_regime_timeline(df: pd.DataFrame, K: int = 4, save_path: str | Path | None = None) -> tuple:
    if save_path is None:
        save_path = DEFAULT_OUTDIR / "regime_timeline.png"
    save_path = Path(save_path)

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = {0: "#E74C3C", 1: "#3498DB", 2: "#9B59B6", 3: "#95A5A6", 4: "#F39C12", 5: "#2ECC71"}

    for k in range(K):
        mask = df["regime_assigned"] == k
        ax.scatter(df.loc[mask, "date"], df.loc[mask, "regime_assigned"], c=colors.get(k, f"C{k}"), s=20, alpha=0.7, label=f"Regime {k}")

    for period in identify_crisis_periods():
        start = pd.to_datetime(period["start"])
        end = pd.to_datetime(period["end"])
        if start <= df["date"].max() and end >= df["date"].min():
            ax.axvspan(start, end, alpha=0.15, color="gray", zorder=0)
            mid_date = start + (end - start) / 2
            if df["date"].min() <= mid_date <= df["date"].max():
                ax.text(mid_date, K - 0.3, period["label"], ha="center", va="top", fontsize=8, style="italic", color="darkgray")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Regime", fontsize=12)
    ax.set_title("GMM Regime Assignments Over Time (2012-2021)", fontsize=14, fontweight="bold")
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"Regime {k}" for k in range(K)])
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="upper right", ncol=min(K, 5), fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved regime timeline to {save_path}")
    return fig, ax


def plot_regime_probabilities_stacked(df: pd.DataFrame, K: int = 4, save_path: str | Path | None = None) -> tuple:
    if save_path is None:
        save_path = DEFAULT_OUTDIR / "regime_probs_stacked.png"
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(16, 6))
    prob_cols = [f"regime_prob_{k}" for k in range(K)]
    regime_probs = df[prob_cols].values.T
    colors = ["#E74C3C", "#3498DB", "#9B59B6", "#95A5A6", "#F39C12", "#2ECC71"][:K]
    ax.stackplot(df["date"], *regime_probs, labels=[f"Regime {k}" for k in range(K)], colors=colors, alpha=0.7)
    for period in identify_crisis_periods():
        start = pd.to_datetime(period["start"])
        end = pd.to_datetime(period["end"])
        if start <= df["date"].max() and end >= df["date"].min():
            ax.axvspan(start, end, alpha=0.1, color="black", zorder=0)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Regime Probability", fontsize=12)
    ax.set_title("Filtered Regime Probabilities Over Time (αₜ)", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right", ncol=min(K, 5), fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved stacked probabilities to {save_path}")
    return fig, ax


def plot_regime_filtered_vs_raw(df: pd.DataFrame, K: int = 4, save_path: str | Path | None = None) -> tuple:
    if save_path is None:
        save_path = DEFAULT_OUTDIR / "regime_filtering_effect.png"
    save_path = Path(save_path)
    fig, axes = plt.subplots(K, 1, figsize=(16, 10), sharex=True)
    for k in range(K):
        ax = axes[k]
        raw_col = f"regime_raw_{k}"
        prob_col = f"regime_prob_{k}"
        if raw_col in df.columns:
            ax.plot(df["date"], df[raw_col], label="Raw GMM π(k)", alpha=0.4, linewidth=1, color="lightgray")
        ax.plot(df["date"], df[prob_col], label="Filtered α(k)", linewidth=2, color=f"C{k}")
        for period in identify_crisis_periods():
            start = pd.to_datetime(period["start"])
            end = pd.to_datetime(period["end"])
            if start <= df["date"].max() and end >= df["date"].min():
                ax.axvspan(start, end, alpha=0.1, color="gray", zorder=0)
        ax.set_ylabel(f"Regime {k}\nProbability", fontsize=10)
        ax.set_ylim([0, 1])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date", fontsize=12)
    fig.suptitle("Effect of Markov Filtering on Regime Probabilities", fontsize=14, fontweight="bold", y=0.995)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved filtering comparison to {save_path}")
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize regime assignments from backtest results (cov or vol).")
    parser.add_argument("--target", choices=("auto", "covariance", "volatility"), default="auto", help="Target type; auto = infer from --backtest path")
    parser.add_argument("--backtest", type=str, default=None, help="Path to backtest parquet/csv (default: from --target)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: from --target)")
    parser.add_argument("--K", type=int, default=4, help="Number of regimes")
    args = parser.parse_args()

    target = args.target
    if target == "auto" and args.backtest is not None:
        target = _detect_target_from_path(args.backtest)
    elif target == "auto":
        target = "covariance"

    if args.backtest is None or args.outdir is None:
        default_backtest, default_outdir = _default_backtest_and_outdir(target)
        if args.backtest is None:
            args.backtest = str(default_backtest)
        if args.outdir is None:
            args.outdir = str(default_outdir)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_regime_data(args.backtest)
    K = args.K

    if "regime_assigned" not in df.columns:
        print("❌ ERROR: 'regime_assigned' column not found in backtest results!")
        print("Re-run: python run_backtest.py --config configs/regime_covariance.yaml (or configs/regime_volatility.yaml)")
        return

    plot_regime_timeline(df, K=K, save_path=outdir / "regime_timeline.png")
    plot_regime_probabilities_stacked(df, K=K, save_path=outdir / "regime_probs_stacked.png")
    if "regime_raw_0" in df.columns:
        plot_regime_filtered_vs_raw(df, K=K, save_path=outdir / "regime_filtering_effect.png")


if __name__ == "__main__":
    main()

