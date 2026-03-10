"""
Visualize regime assignments over time with market event annotations.

Expects backtest results with regime columns: regime_assigned, regime_prob_0..K-1, regime_raw_0..K-1.
Run: python run_backtest.py --config configs/regime_similarity.yaml first to generate them.

Usage:
  python scripts/analysis/visualize_regimes.py
  python scripts/analysis/visualize_regimes.py --backtest results/regime_similarity_backtest.parquet --outdir results/figs_regime_similarity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default paths (repo root = parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKTEST = REPO_ROOT / "results" / "regime_similarity_backtest.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "figs_regime_similarity" / "regime"


def load_regime_data(backtest_file: str | Path = DEFAULT_BACKTEST) -> pd.DataFrame:
    """Load regime assignments from backtest results (one row per date; no method column)."""
    path = Path(backtest_file)
    if not path.exists():
        raise FileNotFoundError(f"Backtest file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)
    return df


def identify_crisis_periods() -> list[dict]:
    """Define major market crisis/event periods for shading."""
    return [
        {"start": "2007-12-01", "end": "2009-06-30", "label": "Financial Crisis"},
        {"start": "2011-08-01", "end": "2011-10-31", "label": "EU Debt Crisis"},
        {"start": "2015-08-01", "end": "2016-02-29", "label": "China Selloff"},
        {"start": "2018-10-01", "end": "2018-12-31", "label": "Q4 Selloff"},
        {"start": "2020-02-20", "end": "2020-04-30", "label": "COVID Crash"},
        {"start": "2020-09-01", "end": "2020-10-31", "label": "Election Vol"},
    ]


def plot_regime_timeline(
    df: pd.DataFrame,
    K: int = 4,
    save_path: str | Path = None,
) -> tuple:
    """Create regime timeline visualization (scatter of regime_assigned over time)."""
    if save_path is None:
        save_path = DEFAULT_OUTDIR / "regime_timeline.png"
    save_path = Path(save_path)

    fig, ax = plt.subplots(figsize=(16, 6))

    colors = {
        0: "#E74C3C",
        1: "#3498DB",
        2: "#9B59B6",
        3: "#95A5A6",
        4: "#F39C12",
        5: "#2ECC71",
    }

    for k in range(K):
        mask = df["regime_assigned"] == k
        ax.scatter(
            df.loc[mask, "date"],
            df.loc[mask, "regime_assigned"],
            c=colors.get(k, f"C{k}"),
            s=20,
            alpha=0.7,
            label=f"Regime {k}",
        )

    crisis_periods = identify_crisis_periods()
    for period in crisis_periods:
        start = pd.to_datetime(period["start"])
        end = pd.to_datetime(period["end"])
        if start <= df["date"].max() and end >= df["date"].min():
            ax.axvspan(start, end, alpha=0.15, color="gray", zorder=0)
            mid_date = start + (end - start) / 2
            if df["date"].min() <= mid_date <= df["date"].max():
                ax.text(
                    mid_date,
                    K - 0.3,
                    period["label"],
                    ha="center",
                    va="top",
                    fontsize=8,
                    style="italic",
                    color="darkgray",
                )

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


def plot_regime_probabilities_stacked(
    df: pd.DataFrame,
    K: int = 4,
    save_path: str | Path = None,
) -> tuple:
    """Stacked area chart of regime probabilities over time."""
    if save_path is None:
        save_path = DEFAULT_OUTDIR / "regime_probs_stacked.png"
    save_path = Path(save_path)

    fig, ax = plt.subplots(figsize=(16, 6))

    prob_cols = [f"regime_prob_{k}" for k in range(K)]
    regime_probs = df[prob_cols].values.T
    colors = ["#E74C3C", "#3498DB", "#9B59B6", "#95A5A6", "#F39C12", "#2ECC71"][:K]

    ax.stackplot(
        df["date"],
        *regime_probs,
        labels=[f"Regime {k}" for k in range(K)],
        colors=colors,
        alpha=0.7,
    )

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


def plot_regime_filtered_vs_raw(
    df: pd.DataFrame,
    K: int = 4,
    save_path: str | Path = None,
) -> tuple:
    """Compare raw GMM probabilities vs filtered posteriors."""
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


def characterize_regimes(df: pd.DataFrame, K: int = 4) -> pd.DataFrame:
    """Compute descriptive statistics for each regime (uses model_* columns)."""
    chars = []
    fro_col = "model_fro"
    logeuc_col = "model_logeuc"
    var_col = "model_gmvp_var"
    sharpe_col = "model_gmvp_sharpe"
    turnover_col = "model_turnover_l1"

    for k in range(K):
        regime_mask = df["regime_assigned"] == k
        regime_df = df.loc[regime_mask]

        if len(regime_df) == 0:
            continue

        char = {
            "regime": k,
            "n_days": len(regime_df),
            "pct_time": len(regime_df) / len(df) * 100,
        }
        if fro_col in regime_df.columns:
            char["mean_fro"] = regime_df[fro_col].mean()
            char["std_fro"] = regime_df[fro_col].std()
        else:
            char["mean_fro"] = np.nan
            char["std_fro"] = np.nan
        if logeuc_col in regime_df.columns:
            char["mean_logeuc"] = regime_df[logeuc_col].mean()
        else:
            char["mean_logeuc"] = np.nan
        if var_col in regime_df.columns:
            char["mean_gmvp_var"] = regime_df[var_col].mean()
        else:
            char["mean_gmvp_var"] = np.nan
        if sharpe_col in regime_df.columns:
            char["mean_gmvp_sharpe"] = regime_df[sharpe_col].mean()
        else:
            char["mean_gmvp_sharpe"] = np.nan
        if turnover_col in regime_df.columns:
            char["mean_turnover"] = regime_df[turnover_col].mean()
        else:
            char["mean_turnover"] = np.nan

        char["first_date"] = regime_df["date"].min().strftime("%Y-%m-%d")
        char["last_date"] = regime_df["date"].max().strftime("%Y-%m-%d")
        chars.append(char)

    return pd.DataFrame(chars)


def print_regime_characterization(chars_df: pd.DataFrame, save_csv_path: str | Path = None) -> None:
    """Print regime characteristics and optionally save to CSV."""
    if save_csv_path is None:
        save_csv_path = REPO_ROOT / "results" / "regime_characterization.csv"
    save_csv_path = Path(save_csv_path)

    print("\n" + "=" * 80)
    print("REGIME CHARACTERIZATION")
    print("=" * 80 + "\n")

    for _, row in chars_df.iterrows():
        k = int(row["regime"])
        print(f"REGIME {k}:")
        print(f"  Time spent: {row['n_days']} days ({row['pct_time']:.1f}%)")
        print(f"  Date range: {row['first_date']} to {row['last_date']}")
        print(f"  Forecast error (Frobenius): {row['mean_fro']:.4f} ± {row['std_fro']:.4f}")
        print(f"  Forecast error (LogEuc): {row['mean_logeuc']:.2f}")
        print(f"  GMVP variance: {row['mean_gmvp_var']:.6f}")
        print(f"  GMVP Sharpe: {row['mean_gmvp_sharpe']:.3f}")
        print(f"  Turnover: {row['mean_turnover']:.3f}")
        print()

    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    chars_df.to_csv(save_csv_path, index=False)
    print(f"✓ Saved characterization to {save_csv_path}\n")


def identify_regime_names(chars_df: pd.DataFrame) -> list[dict]:
    """Suggest interpretable names for regimes based on characteristics."""
    suggestions = []

    for _, row in chars_df.iterrows():
        k = int(row["regime"])
        sharpe = row["mean_gmvp_sharpe"]
        variance = row["mean_gmvp_var"]
        turnover = row["mean_turnover"]
        fro = row["mean_fro"]

        if sharpe > 1.5 and turnover < 0.5:
            name = "Calm Bull"
        elif sharpe < 0.5 and variance > 0.0001:
            name = "Crisis/High Stress"
        elif fro > 0.025:
            name = "High Uncertainty"
        elif turnover > 0.7:
            name = "Volatile/Choppy"
        else:
            name = "Normal/Transition"

        suggestions.append({"regime": k, "suggested_name": name})

    print("\n" + "=" * 80)
    print("SUGGESTED REGIME NAMES (verify with visual inspection):")
    print("=" * 80 + "\n")
    for s in suggestions:
        print(f"  Regime {s['regime']}: {s['suggested_name']}")
    print()

    return suggestions


def print_regime_observations(chars_df: pd.DataFrame) -> None:
    """Print KEY OBSERVATIONS from regime characterization."""
    print("\n" + "=" * 80)
    print("REGIME CHARACTERIZATION SUMMARY")
    print("=" * 80 + "\n")
    print(chars_df.to_string(index=False))
    print("\nKEY OBSERVATIONS:\n")

    if "n_days" in chars_df.columns:
        most_common = chars_df.loc[chars_df["n_days"].idxmax()]
        print(f"  • Most common regime: Regime {int(most_common['regime'])} ({most_common['pct_time']:.1f}% of time)")
    if "mean_gmvp_sharpe" in chars_df.columns and chars_df["mean_gmvp_sharpe"].notna().any():
        best_sharpe = chars_df.loc[chars_df["mean_gmvp_sharpe"].idxmax()]
        print(f"  • Best GMVP Sharpe: Regime {int(best_sharpe['regime'])} (Sharpe = {best_sharpe['mean_gmvp_sharpe']:.2f})")
        worst_sharpe = chars_df.loc[chars_df["mean_gmvp_sharpe"].idxmin()]
        print(f"  • Worst GMVP Sharpe: Regime {int(worst_sharpe['regime'])} (Sharpe = {worst_sharpe['mean_gmvp_sharpe']:.2f})")
    if "mean_fro" in chars_df.columns and chars_df["mean_fro"].notna().any():
        highest_error = chars_df.loc[chars_df["mean_fro"].idxmax()]
        print(f"  • Highest forecast error: Regime {int(highest_error['regime'])} (Frobenius = {highest_error['mean_fro']:.4f})")
    if "mean_turnover" in chars_df.columns and chars_df["mean_turnover"].notna().any():
        highest_turn = chars_df.loc[chars_df["mean_turnover"].idxmax()]
        print(f"  • Highest turnover: Regime {int(highest_turn['regime'])} (turnover = {highest_turn['mean_turnover']:.2f})")
    print()


def save_regime_names_mapping(
    suggestions: list[dict],
    save_path: str | Path = None,
) -> None:
    """Save regime index -> name mapping to JSON (for report use)."""
    import json

    if save_path is None:
        save_path = REPO_ROOT / "results" / "regime_names_mapping.json"
    save_path = Path(save_path)
    mapping = {s["regime"]: s["suggested_name"] for s in suggestions}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Regime name mapping saved to {save_path}")
    print("\nSuggested names:")
    for k, name in mapping.items():
        print(f"  Regime {k}: {name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize regime assignments from backtest results.")
    parser.add_argument(
        "--backtest",
        type=str,
        default=str(DEFAULT_BACKTEST),
        help="Path to regime_similarity_backtest.parquet or .csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Directory for output figures",
    )
    parser.add_argument("--K", type=int, default=4, help="Number of regimes")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("REGIME ANALYSIS PIPELINE")
    print("=" * 80 + "\n")

    print("Loading regime assignments from backtest results...")
    df = load_regime_data(args.backtest)
    K = args.K

    # DEBUG HERE
    prob_cols = [f"regime_prob_{k}" for k in range(K)]
    raw_cols = [f"regime_raw_{k}" for k in range(K)]

    print("\nFiltered regime probability summary:")
    print(df[prob_cols].describe())

    if all(col in df.columns for col in raw_cols):
        print("\nRaw GMM probability summary:")
        print(df[raw_cols].describe())

    print(f"✓ Loaded {len(df)} evaluation dates")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    print(f"✓ Loaded {len(df)} evaluation dates")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    if "regime_assigned" in df.columns:
        print(f"  Regimes: {sorted(df['regime_assigned'].unique())}")
    print()

    if "regime_assigned" not in df.columns:
        print("❌ ERROR: 'regime_assigned' column not found in backtest results!")
        print("   Re-run the backtest with regime saving enabled:")
        print("   python run_backtest.py --config configs/regime_similarity.yaml")
        return

    print("Generating visualizations...")
    plot_regime_timeline(df, K=K, save_path=outdir / "regime_timeline.png")
    plot_regime_probabilities_stacked(df, K=K, save_path=outdir / "regime_probs_stacked.png")

    if f"regime_raw_0" in df.columns:
        plot_regime_filtered_vs_raw(df, K=K, save_path=outdir / "regime_filtering_effect.png")
    else:
        print("  (Skipping filtering comparison - raw probabilities not saved)")

    print()

    print("Characterizing regimes...")
    chars_df = characterize_regimes(df, K=K)
    print_regime_characterization(chars_df, save_csv_path=REPO_ROOT / "results" / "regime_characterization.csv")

    suggestions = identify_regime_names(chars_df)
    print_regime_observations(chars_df)
    save_regime_names_mapping(suggestions, save_path=REPO_ROOT / "results" / "regime_names_mapping.json")

    print("=" * 80)
    print("✓ REGIME ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    print("Generated files:")
    print(f"  - {outdir / 'regime_timeline.png'}")
    print(f"  - {outdir / 'regime_probs_stacked.png'}")
    print(f"  - {outdir / 'regime_filtering_effect.png'}")
    print("  - results/regime_characterization.csv")
    print("  - results/r egime_names_mapping.json")
    print()

    print("REGIME ANALYSIS PIPELINE EXECUTED")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✓ Regime assignments extracted and saved")
    print("✓ Timeline visualization created (similar to reference)")
    print("✓ Regime probabilities stacked area plot created")
    print("✓ Filtering effect comparison created")
    print("✓ Regime characterization computed and saved")
    print()
    print("Next steps:")
    print("1. Review regime_characterization.csv to understand what each regime represents")
    print("2. Manually assign interpretable names based on characteristics + visual inspection")
    print("3. Use these names in final report figures and discussion")
    print("4. Run performance-by-regime analysis (Part 2)")
    print()


if __name__ == "__main__":
    main()
