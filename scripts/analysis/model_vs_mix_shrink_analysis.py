"""
Deep analysis: When does Model beat Mix/Shrink?
Analyzes temporal patterns, regime-specific performance, and metric-specific wins.

Backtest is wide format: one row per date, columns model_fro, mix_fro, shrink_fro, etc.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figs_regime_similarity"
DEFAULT_BACKTEST = RESULTS_DIR / "regime_similarity_backtest.parquet"
DEFAULT_BACKTEST_CSV = RESULTS_DIR / "regime_similarity_backtest.csv"

# Optional regime labels (adjust for your K)
REGIME_NAMES = {
    0: "Calm Bull",
    1: "Moderate Bull",
    2: "Normal",
    3: "High Stress",
    4: "Regime 4",
    5: "Regime 5",
}


def load_data(path=None):
    """Load backtest results (wide format: one row per date)."""
    path = path or (DEFAULT_BACKTEST if DEFAULT_BACKTEST.exists() else DEFAULT_BACKTEST_CSV)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Backtest not found: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def analyze_win_rates(df):
    """
    Compute win rates: % of dates where Model beats Mix/Shrink.
    Wide format: df has model_*, mix_*, shrink_* columns on same row.
    Returns two DataFrames with date + model/mix/shrink cols for downstream use.
    """
    print("\n" + "=" * 100)
    print("WIN RATES: Model vs Mix/Shrink")
    print("=" * 100 + "\n")

    cols_mix = ["date", "model_fro", "model_gmvp_sharpe", "model_gmvp_var", "model_turnover_l1",
                "mix_fro", "mix_gmvp_sharpe", "mix_gmvp_var", "mix_turnover_l1"]
    cols_shrink = ["date", "model_fro", "model_gmvp_sharpe", "model_gmvp_var", "model_turnover_l1",
                   "shrink_fro", "shrink_gmvp_sharpe", "shrink_gmvp_var", "shrink_turnover_l1"]
    merged_mix = df[[c for c in cols_mix if c in df.columns]].copy()
    merged_shrink = df[[c for c in cols_shrink if c in df.columns]].copy()

    metrics = {
        "fro": "lower",
        "gmvp_sharpe": "higher",
        "gmvp_var": "lower",
        "turnover_l1": "lower",
    }

    print("MODEL vs MIX:")
    print("-" * 100)
    for metric, direction in metrics.items():
        model_col = f"model_{metric}"
        mix_col = f"mix_{metric}"
        if model_col not in merged_mix.columns or mix_col not in merged_mix.columns:
            continue
        a = pd.to_numeric(merged_mix[model_col], errors="coerce")
        b = pd.to_numeric(merged_mix[mix_col], errors="coerce")
        valid = a.notna() & b.notna()
        if direction == "lower":
            wins = (a[valid] < b[valid]).sum()
        else:
            wins = (a[valid] > b[valid]).sum()
        total = valid.sum()
        win_pct = wins / total * 100 if total else 0
        print(f"  {metric:<15}: {int(wins):3d}/{int(total)} dates ({win_pct:5.1f}%)")
    print()

    print("MODEL vs SHRINK:")
    print("-" * 100)
    for metric, direction in metrics.items():
        model_col = f"model_{metric}"
        shrink_col = f"shrink_{metric}"
        if model_col not in merged_shrink.columns or shrink_col not in merged_shrink.columns:
            continue
        a = pd.to_numeric(merged_shrink[model_col], errors="coerce")
        b = pd.to_numeric(merged_shrink[shrink_col], errors="coerce")
        valid = a.notna() & b.notna()
        if direction == "lower":
            wins = (a[valid] < b[valid]).sum()
        else:
            wins = (a[valid] > b[valid]).sum()
        total = valid.sum()
        win_pct = wins / total * 100 if total else 0
        print(f"  {metric:<15}: {int(wins):3d}/{int(total)} dates ({win_pct:5.1f}%)")
    print()
    return merged_mix, merged_shrink


def analyze_by_regime(df):
    """
    Compare Model vs Mix/Shrink performance by regime.
    Wide format: regime_assigned and model_*, mix_*, shrink_* on same row.
    """
    print("\n" + "=" * 100)
    print("PERFORMANCE BY REGIME: Model vs Mix/Shrink")
    print("=" * 100 + "\n")

    if "regime_assigned" not in df.columns:
        print("  No regime_assigned column in backtest. Skip by-regime analysis.")
        return

    regime_ids = sorted(pd.Series(df["regime_assigned"].dropna().unique()).astype(int).tolist())
    for regime_id in regime_ids:
        mask = df["regime_assigned"] == regime_id
        regime_df = df.loc[mask]
        name = REGIME_NAMES.get(regime_id, f"Regime {regime_id}")

        print(f"\nREGIME {regime_id} ({name}):")
        print(f"  N = {len(regime_df)} dates")
        print("-" * 100)

        def mn(c):
            return pd.to_numeric(regime_df[c], errors="coerce").mean()

        print("  Frobenius error:")
        print(f"    Model:  {mn('model_fro'):.4f}")
        print(f"    Mix:    {mn('mix_fro'):.4f}")
        print(f"    Shrink: {mn('shrink_fro'):.4f}")
        print("  GMVP Sharpe:")
        print(f"    Model:  {mn('model_gmvp_sharpe'):.3f}")
        print(f"    Mix:    {mn('mix_gmvp_sharpe'):.3f}")
        print(f"    Shrink: {mn('shrink_gmvp_sharpe'):.3f}")
        print("  Turnover:")
        print(f"    Model:  {mn('model_turnover_l1'):.3f}")
        print(f"    Mix:    {mn('mix_turnover_l1'):.3f}")
        print(f"    Shrink: {mn('shrink_turnover_l1'):.3f}")
    print()


def analyze_temporal_patterns(merged_mix, merged_shrink):
    """
    Rolling win rate over time: when does Model win?
    """
    print("\n" + "=" * 100)
    print("TEMPORAL PATTERNS: When Does Model Win?")
    print("=" * 100 + "\n")

    merged_mix = merged_mix.sort_values("date").copy()
    merged_shrink = merged_shrink.sort_values("date").copy()

    merged_mix["win_fro"] = merged_mix["model_fro"] < merged_mix["mix_fro"]
    merged_mix["win_sharpe"] = merged_mix["model_gmvp_sharpe"] > merged_mix["mix_gmvp_sharpe"]
    merged_shrink["win_fro"] = merged_shrink["model_fro"] < merged_shrink["shrink_fro"]
    merged_shrink["win_sharpe"] = merged_shrink["model_gmvp_sharpe"] > merged_shrink["shrink_gmvp_sharpe"]

    window = 63
    merged_mix["rolling_win_fro_mix"] = merged_mix["win_fro"].rolling(window, min_periods=20).mean()
    merged_mix["rolling_win_sharpe_mix"] = merged_mix["win_sharpe"].rolling(window, min_periods=20).mean()
    merged_shrink["rolling_win_fro_shrink"] = merged_shrink["win_fro"].rolling(window, min_periods=20).mean()
    merged_shrink["rolling_win_sharpe_shrink"] = merged_shrink["win_sharpe"].rolling(window, min_periods=20).mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Model vs Mix - Frobenius
    ax = axes[0, 0]
    ax.plot(merged_mix["date"], merged_mix["rolling_win_fro_mix"], linewidth=2, color="steelblue")
    ax.axhline(y=0.5, color="red", linestyle="--", label="50% (neutral)")
    ax.fill_between(
        merged_mix["date"], 0.5, merged_mix["rolling_win_fro_mix"],
        where=(merged_mix["rolling_win_fro_mix"] > 0.5), alpha=0.3, color="green"
    )
    ax.fill_between(
        merged_mix["date"], 0.5, merged_mix["rolling_win_fro_mix"],
        where=(merged_mix["rolling_win_fro_mix"] <= 0.5), alpha=0.3, color="red"
    )
    ax.set_ylabel("Win rate (Model < Mix)", fontsize=11)
    ax.set_title("Model vs Mix: Frobenius win rate (63d rolling)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()

    # Model vs Mix - Sharpe
    ax = axes[0, 1]
    ax.plot(merged_mix["date"], merged_mix["rolling_win_sharpe_mix"], linewidth=2, color="steelblue")
    ax.axhline(y=0.5, color="red", linestyle="--", label="50% (neutral)")
    ax.fill_between(
        merged_mix["date"], 0.5, merged_mix["rolling_win_sharpe_mix"],
        where=(merged_mix["rolling_win_sharpe_mix"] > 0.5), alpha=0.3, color="green"
    )
    ax.fill_between(
        merged_mix["date"], 0.5, merged_mix["rolling_win_sharpe_mix"],
        where=(merged_mix["rolling_win_sharpe_mix"] <= 0.5), alpha=0.3, color="red"
    )
    ax.set_ylabel("Win rate (Model > Mix)", fontsize=11)
    ax.set_title("Model vs Mix: GMVP Sharpe win rate (63d rolling)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()

    # Model vs Shrink - Frobenius
    ax = axes[1, 0]
    ax.plot(merged_shrink["date"], merged_shrink["rolling_win_fro_shrink"], linewidth=2, color="darkorange")
    ax.axhline(y=0.5, color="red", linestyle="--", label="50% (neutral)")
    ax.fill_between(
        merged_shrink["date"], 0.5, merged_shrink["rolling_win_fro_shrink"],
        where=(merged_shrink["rolling_win_fro_shrink"] > 0.5), alpha=0.3, color="green"
    )
    ax.fill_between(
        merged_shrink["date"], 0.5, merged_shrink["rolling_win_fro_shrink"],
        where=(merged_shrink["rolling_win_fro_shrink"] <= 0.5), alpha=0.3, color="red"
    )
    ax.set_ylabel("Win rate (Model < Shrink)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Model vs Shrink: Frobenius win rate (63d rolling)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()

    # Model vs Shrink - Sharpe
    ax = axes[1, 1]
    ax.plot(merged_shrink["date"], merged_shrink["rolling_win_sharpe_shrink"], linewidth=2, color="darkorange")
    ax.axhline(y=0.5, color="red", linestyle="--", label="50% (neutral)")
    ax.fill_between(
        merged_shrink["date"], 0.5, merged_shrink["rolling_win_sharpe_shrink"],
        where=(merged_shrink["rolling_win_sharpe_shrink"] > 0.5), alpha=0.3, color="green"
    )
    ax.fill_between(
        merged_shrink["date"], 0.5, merged_shrink["rolling_win_sharpe_shrink"],
        where=(merged_shrink["rolling_win_sharpe_shrink"] <= 0.5), alpha=0.3, color="red"
    )
    ax.set_ylabel("Win rate (Model > Shrink)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Model vs Shrink: GMVP Sharpe win rate (63d rolling)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Temporal win rates: Model vs Mix/Shrink\nGreen = Model winning, Red = Model losing",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGS_DIR / "model_vs_mix_shrink_temporal.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}\n")


def print_key_insights(merged_mix, merged_shrink):
    """Print key insights about when Model wins."""
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100 + "\n")

    def pct_win(a, b, lower_better):
        valid = a.notna() & b.notna()
        if lower_better:
            return (a[valid] < b[valid]).mean() * 100
        return (a[valid] > b[valid]).mean() * 100

    fro_win_mix = pct_win(merged_mix["model_fro"], merged_mix["mix_fro"], True)
    sharpe_win_mix = pct_win(merged_mix["model_gmvp_sharpe"], merged_mix["mix_gmvp_sharpe"], False)
    fro_win_shrink = pct_win(merged_shrink["model_fro"], merged_shrink["shrink_fro"], True)
    sharpe_win_shrink = pct_win(merged_shrink["model_gmvp_sharpe"], merged_shrink["shrink_gmvp_sharpe"], False)

    print("MODEL vs MIX:")
    print(f"  • Forecast accuracy (Frobenius): Model wins {fro_win_mix:.1f}% of dates")
    print(f"  • Portfolio Sharpe: Model wins {sharpe_win_mix:.1f}% of dates")
    if fro_win_mix > 50:
        print("  → Model has better FORECASTS than Mix")
    else:
        print("  → Mix has better FORECASTS than Model")
    if sharpe_win_mix > 50:
        print("  → Model has better PORTFOLIOS than Mix")
    else:
        print("  → Mix has better PORTFOLIOS than Model")
    print()

    print("MODEL vs SHRINK:")
    print(f"  • Forecast accuracy (Frobenius): Model wins {fro_win_shrink:.1f}% of dates")
    print(f"  • Portfolio Sharpe: Model wins {sharpe_win_shrink:.1f}% of dates")
    if fro_win_shrink > 50:
        print("  → Model has better FORECASTS than Shrink")
    else:
        print("  → Shrink has better FORECASTS than Model")
    if sharpe_win_shrink > 50:
        print("  → Model has better PORTFOLIOS than Shrink")
    else:
        print("  → Shrink has better PORTFOLIOS than Model")
    print()

    print("SUMMARY:")
    print("-" * 100)
    if fro_win_mix < 50 and fro_win_shrink < 50:
        print("  Model does NOT have best forecast accuracy vs Mix/Shrink")
        print("  → Mix and Shrink's regularization can yield more accurate forecasts")
    elif fro_win_mix >= 50 or fro_win_shrink >= 50:
        print("  Model is competitive or better on forecast accuracy in some comparisons")
    if sharpe_win_mix < 50 and sharpe_win_shrink < 50:
        print("  Model does NOT have best portfolio performance vs Mix/Shrink")
        print("  → Mix and Shrink's lower turnover can yield better net returns")
    elif sharpe_win_mix >= 50 or sharpe_win_shrink >= 50:
        print("  Model is competitive or better on Sharpe in some comparisons")
    print()


def main():
    """Run complete Model vs Mix/Shrink analysis."""
    print("\n" + "=" * 100)
    print("MODEL vs MIX/SHRINK: COMPREHENSIVE ANALYSIS")
    print("=" * 100)

    df = load_data()
    merged_mix, merged_shrink = analyze_win_rates(df)
    analyze_by_regime(df)
    analyze_temporal_patterns(merged_mix, merged_shrink)
    print_key_insights(merged_mix, merged_shrink)

    print("=" * 100)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 100)
    print("\nGenerated:")
    print("  - results/figs_regime_similarity/model_vs_mix_shrink_temporal.png")
    print()


if __name__ == "__main__":
    main()
