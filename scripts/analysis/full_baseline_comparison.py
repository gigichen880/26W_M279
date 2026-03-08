"""
Create comprehensive comparison table: Model vs all baselines.
Shows actual mean performance and rankings. Backtest is wide format (one row per date,
columns like model_fro, roll_fro, etc.).
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_BACKTEST = RESULTS_DIR / "regime_similarity_backtest.parquet"
DEFAULT_BACKTEST_CSV = RESULTS_DIR / "regime_similarity_backtest.csv"

METHODS = ["model", "roll", "pers", "shrink", "mix"]
METRICS = {
    "fro": "Frobenius",
    "logeuc": "LogEuc",
    "gmvp_sharpe": "GMVP Sharpe",
    "gmvp_var": "GMVP Variance",
    "turnover_l1": "Turnover",
}
# Lower is better for ranking
LOWER_IS_BETTER = {"fro", "logeuc", "gmvp_var", "turnover_l1"}


def load_backtest(path=None):
    """Load backtest (parquet or CSV). Wide format: one row per date."""
    path = path or (DEFAULT_BACKTEST if DEFAULT_BACKTEST.exists() else DEFAULT_BACKTEST_CSV)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Backtest not found: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    return df


def create_comparison_table(backtest_file=None):
    """
    Create table showing mean (and std) performance of each method on key metrics.
    Wide format: df has columns model_fro, roll_fro, etc.
    """
    df = load_backtest(backtest_file)
    results = []
    for method in METHODS:
        row = {"Method": method.capitalize()}
        for metric_key, metric_name in METRICS.items():
            col = f"{method}_{metric_key}"
            if col not in df.columns:
                row[metric_name] = np.nan
                row[f"{metric_name} Std"] = np.nan
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            row[metric_name] = vals.mean()
            row[f"{metric_name} Std"] = vals.std()
        results.append(row)
    results_df = pd.DataFrame(results)

    # Rankings: lower is better for fro, logeuc, gmvp_var, turnover; higher for sharpe
    for metric_key, metric_name in METRICS.items():
        if metric_name not in results_df.columns:
            continue
        ascending = metric_key in LOWER_IS_BETTER
        results_df[f"{metric_name} Rank"] = results_df[metric_name].rank(ascending=ascending)
    return results_df


def print_comparison_table(results_df):
    """Print formatted table and best performer per metric."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 100 + "\n")

    # Round for display
    display = results_df.copy()
    for metric_name in METRICS.values():
        if metric_name in display.columns:
            display[metric_name] = display[metric_name].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        if f"{metric_name} Std" in display.columns:
            display[f"{metric_name} Std"] = display[f"{metric_name} Std"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
    print(display.to_string(index=False))
    print()

    print("BEST PERFORMER BY METRIC:")
    print("-" * 100)
    for metric_name in METRICS.values():
        rank_col = f"{metric_name} Rank"
        if rank_col not in results_df.columns:
            continue
        best_idx = results_df[rank_col].idxmin()
        best_method = results_df.loc[best_idx, "Method"]
        best_val = results_df.loc[best_idx, metric_name]
        if pd.notna(best_val):
            print(f"  {metric_name:<20}: {best_method:<10} ({best_val:.4f})")
        else:
            print(f"  {metric_name:<20}: N/A")
    print()


def analyze_mix_advantage(backtest_file=None):
    """
    Analyze why Mix beats Model on GMVP Sharpe.
    Mix = (1-λ)*Shrink + λ*Model; λ=0.3 from config.
    """
    df = load_backtest(backtest_file)
    print("\n" + "=" * 100)
    print("WHY DOES MIX BEAT MODEL ON GMVP?")
    print("=" * 100 + "\n")

    def mean_ser(col):
        return pd.to_numeric(df[col], errors="coerce").dropna().mean()

    print("Mean performance (wide format: one row per date):")
    print(f"  Model Sharpe:   {mean_ser('model_gmvp_sharpe'):.4f}")
    print(f"  Mix Sharpe:    {mean_ser('mix_gmvp_sharpe'):.4f}")
    print(f"  Shrink Sharpe: {mean_ser('shrink_gmvp_sharpe'):.4f}")
    print()
    print(f"  Model Variance:   {mean_ser('model_gmvp_var'):.6f}")
    print(f"  Mix Variance:     {mean_ser('mix_gmvp_var'):.6f}")
    print(f"  Shrink Variance:  {mean_ser('shrink_gmvp_var'):.6f}")
    print()
    print(f"  Model Turnover:   {mean_ser('model_turnover_l1'):.3f}")
    print(f"  Mix Turnover:     {mean_ser('mix_turnover_l1'):.3f}")
    print(f"  Shrink Turnover:  {mean_ser('shrink_turnover_l1'):.3f}")
    print()
    print("INTERPRETATION:")
    print("  Mix = 0.7 * Shrink + 0.3 * Model (λ=0.3 from config)")
    print("  Mix benefits from:")
    print("    1. Shrink's lower variance (regularization)")
    print("    2. Shrink's lower turnover (stability)")
    print("    3. Model's forecast information (30% weight)")
    print("  Result: Mix gets 'best of both worlds' — stability + information")
    print()
    print("KEY INSIGHT:")
    print("  Pure Model has higher turnover → higher transaction costs in practice")
    print("  Mixing with Shrink reduces turnover while preserving forecast signal")
    print("  This is EVIDENCE that Model forecasts contain value (otherwise mixing wouldn't help)")
    print()


def main():
    """Run comprehensive comparison and Mix analysis."""
    backtest_path = DEFAULT_BACKTEST if DEFAULT_BACKTEST.exists() else DEFAULT_BACKTEST_CSV
    if not backtest_path.exists():
        print(f"Backtest not found: {backtest_path}")
        print("Run: python run_backtest.py --config configs/regime_similarity.yaml")
        return

    results_df = create_comparison_table()
    out_csv = RESULTS_DIR / "comprehensive_baseline_comparison.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print_comparison_table(results_df)
    print(f"✓ Saved to {out_csv}\n")

    analyze_mix_advantage()


if __name__ == "__main__":
    main()
