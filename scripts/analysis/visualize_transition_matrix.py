"""
Visualize regime transition matrix (works for both covariance and volatility backtests).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def _default_paths(target: str):
    if target == "volatility":
        p = RESULTS_DIR / "regime_volatility_backtest.parquet"
        if not p.exists():
            p = RESULTS_DIR / "regime_volatility_backtest.csv"
        return p, RESULTS_DIR / "figs_regime_volatility"
    p = RESULTS_DIR / "regime_covariance_backtest.parquet"
    if not p.exists():
        p = RESULTS_DIR / "regime_covariance_backtest.csv"
    return p, RESULTS_DIR / "figs_regime_covariance"


def extract_transition_matrix(backtest_results_path=None):
    """
    Extract transition matrix from trained model.

    The transition matrix A is stored in the RegimeModel after training.
    We need to reconstruct it from the regime assignments.

    Alternative: If stored in config or model checkpoint, load from there.
    For now, estimate from observed transitions in backtest results.
    """
    if backtest_results_path is None:
        backtest_results_path = RESULTS_DIR / "regime_covariance_backtest.parquet"
    path = Path(backtest_results_path)
    if not path.exists():
        raise FileNotFoundError(f"Backtest file not found: {path}")

    df = pd.read_parquet(path)

    # Backtest has one row per date; regime_assigned is the model's regime
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df = df.sort_values("date").reset_index(drop=True)

    # Get regime assignments
    regimes = df["regime_assigned"].values

    # Count transitions
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

    # Normalize to get probabilities (row sums to 1)
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.where(row_sums > 0, transition_counts / row_sums, 0.0)

    # Handle any NaN (if a regime never appears, though unlikely)
    transition_matrix = np.nan_to_num(transition_matrix, nan=0.0)

    return transition_matrix


def plot_transition_matrix(
    transition_matrix,
    regime_names=None,
    save_path=None,
):
    """
    Create annotated heatmap of transition matrix.

    Args:
        transition_matrix: 4×4 numpy array of transition probabilities
        regime_names: Optional list of regime names
        save_path: Where to save figure
    """
    if save_path is None:
        save_path = RESULTS_DIR / "figs_regime_covariance" / "transition_matrix_heatmap.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if regime_names is None:
        regime_names = [
            "Regime 0\n(Calm Bull)",
            "Regime 1\n(Moderate Bull)",
            "Regime 2\n(Normal)",
            "Regime 3\n(High Stress)",
        ]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        transition_matrix,
        annot=True,  # Show values
        fmt=".3f",  # 3 decimal places
        cmap="Blues",  # Color scheme
        vmin=0,
        vmax=1,
        square=True,  # Square cells
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Transition Probability"},
        ax=ax,
    )

    # Labels
    ax.set_xlabel("To Regime (t)", fontsize=12, fontweight="bold")
    ax.set_ylabel("From Regime (t-1)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Regime Transition Matrix (Markov Chain)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set tick labels
    ax.set_xticklabels(regime_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(regime_names, rotation=0, fontsize=10)

    # Highlight diagonal (persistence)
    for i in range(4):
        # Add rectangle around diagonal elements
        rect = plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="red", linewidth=2.5)
        ax.add_patch(rect)

    # Add text annotation
    diagonal_mean = np.diag(transition_matrix).mean()
    off_diagonal_sum = transition_matrix.sum() - np.diag(transition_matrix).sum()
    off_diagonal_mean = off_diagonal_sum / 12 if off_diagonal_sum > 0 else 0.0

    annotation_text = (
        f"Diagonal (persistence): {diagonal_mean:.3f} avg\n"
        f"Off-diagonal (switching): {off_diagonal_mean:.3f} avg"
    )

    ax.text(
        0.02,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved transition matrix heatmap: {save_path}")


def print_transition_analysis(transition_matrix, regime_names=None):
    """Print interpretation of transition matrix."""

    if regime_names is None:
        regime_names = ["Calm Bull", "Moderate Bull", "Normal", "High Stress"]

    print("\n" + "=" * 80)
    print("TRANSITION MATRIX ANALYSIS")
    print("=" * 80 + "\n")

    # Diagonal (persistence)
    diagonal = np.diag(transition_matrix)
    print("PERSISTENCE (Diagonal Elements):")
    for i, (name, prob) in enumerate(zip(regime_names, diagonal)):
        print(f"  Regime {i} ({name}): {prob:.3f} ({prob*100:.1f}% stay in same regime)")

    print(f"\n  Average persistence: {diagonal.mean():.3f}")

    # Most persistent regime
    most_persistent_idx = diagonal.argmax()
    print(
        f"  Most persistent: Regime {most_persistent_idx} ({regime_names[most_persistent_idx]}) - {diagonal[most_persistent_idx]:.3f}"
    )

    # Least persistent regime
    least_persistent_idx = diagonal.argmin()
    print(
        f"  Least persistent: Regime {least_persistent_idx} ({regime_names[least_persistent_idx]}) - {diagonal[least_persistent_idx]:.3f}"
    )

    # Common transitions (off-diagonal)
    print("\n\nCOMMON TRANSITIONS (Off-Diagonal > 0.15):")
    for i in range(4):
        for j in range(4):
            if i != j and transition_matrix[i, j] > 0.15:
                print(f"  {regime_names[i]} → {regime_names[j]}: {transition_matrix[i, j]:.3f}")

    # Transition symmetry
    print("\n\nTRANSITION SYMMETRY:")
    for i in range(4):
        for j in range(i + 1, 4):
            forward = transition_matrix[i, j]
            reverse = transition_matrix[j, i]
            if forward > 0.1 or reverse > 0.1:
                print(
                    f"  {regime_names[i]} ⇄ {regime_names[j]}: "
                    f"forward={forward:.3f}, reverse={reverse:.3f}"
                )

    print()


def main():
    """Generate transition matrix visualization and analysis (cov or vol backtest)."""
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

    print("\n" + "=" * 80)
    print("TRANSITION MATRIX VISUALIZATION")
    print("=" * 80 + "\n")

    print("Extracting transition matrix from backtest results...")
    transition_matrix = extract_transition_matrix(backtest_path)

    print(f"✓ Extracted {transition_matrix.shape[0]}×{transition_matrix.shape[1]} matrix")
    print("\nTransition Matrix (A):")
    print(transition_matrix)
    print()

    regime_names = ["Calm Bull", "Moderate Bull", "Normal", "High Stress"]
    print_transition_analysis(transition_matrix, regime_names)

    print("Creating heatmap...")
    plot_transition_matrix(transition_matrix, regime_names, save_path=figs_dir / "transition_matrix_heatmap.png")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_matrix = pd.DataFrame(
        transition_matrix,
        index=[f"From_{name}" for name in regime_names],
        columns=[f"To_{name}" for name in regime_names],
    )
    csv_name = "transition_matrix_volatility.csv" if "volatility" in str(backtest_path) else "transition_matrix.csv"
    csv_path = results_dir / csv_name
    df_matrix.to_csv(csv_path)
    print(f"✓ Saved matrix to {csv_path}")

    print("\n" + "=" * 80)
    print("✓ TRANSITION MATRIX ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  • Average persistence: {np.diag(transition_matrix).mean():.3f}")
    off_sum = transition_matrix.sum() - np.diag(transition_matrix).sum()
    print(f"  • Average switching: {off_sum / 12:.3f}")
    print("\nInterpretation:")
    print("  High diagonal values (>0.5) indicate regime persistence")
    print("  Low off-diagonal values indicate stable regimes with infrequent transitions")
    print()


if __name__ == "__main__":
    main()
