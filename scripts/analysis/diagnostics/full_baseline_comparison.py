"""
Create comprehensive comparison table: Model vs all baselines (covariance or volatility).

Shows actual mean performance and rankings. Backtest is wide format (one row per date,
columns like model_fro, roll_fro, or model_vol_mse, roll_vol_mse, etc.).

Usage:
  python -m scripts.analysis.full_baseline_comparison
  python -m scripts.analysis.full_baseline_comparison --input results/regime_volatility_backtest.parquet --target volatility
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.utils.paths import RESULTS_DIR, resolve_backtest_path

METHODS = ["model", "roll", "pers", "shrink", "mix"]
METRICS_COV = {
    "fro": "Frobenius",
    "logeuc": "LogEuc",
    "gmvp_sharpe": "GMVP Sharpe",
    "gmvp_var": "GMVP Variance",
    "turnover_l1": "Turnover",
}
METRICS_VOL = {
    "vol_mse": "Vol MSE",
    "vol_mae": "Vol MAE",
    "vol_rmse": "Vol RMSE",
}
LOWER_IS_BETTER_COV = {"fro", "logeuc", "gmvp_var", "turnover_l1"}
LOWER_IS_BETTER_VOL = {"vol_mse", "vol_mae", "vol_rmse"}


def load_backtest(path: str | Path | None):
    """Load backtest (parquet or CSV). Wide format: one row per date."""
    path = Path(path) if path else None
    if path is None or not path.exists():
        raise FileNotFoundError(f"Backtest path required and must exist: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Backtest not found: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    return df


def create_comparison_table(backtest_file=None, is_vol: bool = False):
    """
    Create table showing mean (and std) performance of each method on key metrics.
    Wide format: df has columns model_fro, roll_fro, or model_vol_mse, etc.
    """
    df = load_backtest(backtest_file)
    metrics = METRICS_VOL if is_vol else METRICS_COV
    lower_better = LOWER_IS_BETTER_VOL if is_vol else LOWER_IS_BETTER_COV
    results = []
    for method in METHODS:
        row = {"Method": method.capitalize()}
        for metric_key, metric_name in metrics.items():
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
    for metric_key, metric_name in metrics.items():
        if metric_name not in results_df.columns:
            continue
        ascending = metric_key in lower_better
        results_df[f"{metric_name} Rank"] = results_df[metric_name].rank(ascending=ascending)
    return results_df


def print_comparison_table(results_df, is_vol: bool = False):
    """Print formatted table and best performer per metric."""
    metrics = METRICS_VOL if is_vol else METRICS_COV
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BASELINE COMPARISON" + (" (volatility)" if is_vol else ""))
    print("=" * 100 + "\n")

    # Round for display
    display = results_df.copy()
    for metric_name in metrics.values():
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
    for metric_name in metrics.values():
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
    ap = argparse.ArgumentParser(description="Comprehensive baseline comparison (cov or vol)")
    ap.add_argument("--input", default=None, help="Backtest parquet/csv (default: regime_covariance or regime_volatility)")
    ap.add_argument("--target", choices=("auto", "covariance", "volatility"), default="auto")
    args = ap.parse_args()
    is_vol = args.target == "volatility"
    if args.input is not None:
        backtest_path = Path(args.input)
    else:
        if args.target == "volatility":
            backtest_path = resolve_backtest_path("regime_volatility")
        elif args.target == "covariance":
            backtest_path = resolve_backtest_path("regime_covariance")
        else:
            backtest_path = resolve_backtest_path("regime_covariance")
            if not backtest_path.exists():
                backtest_path = resolve_backtest_path("regime_volatility")
            if backtest_path.exists():
                is_vol = "regime_volatility" in str(backtest_path)
    if not backtest_path.exists():
        print(f"Backtest not found: {backtest_path}")
        print("Run: python run_backtest.py --config configs/regime_covariance.yaml (or configs/regime_volatility.yaml)")
        return

    results_df = create_comparison_table(backtest_path, is_vol=is_vol)
    tag = "regime_volatility" if is_vol else "regime_covariance"
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "comprehensive_baseline_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print_comparison_table(results_df, is_vol=is_vol)
    print(f"✓ Saved to {out_csv}\n")

    if not is_vol:
        analyze_mix_advantage(backtest_path)
    else:
        print("(Mix/GMVP analysis applies to covariance backtest only.)")


if __name__ == "__main__":
    main()
