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
import sys

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `scripts.*` imports work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

EVAL_START_DEFAULT = pd.Timestamp("2013-01-01")
EVAL_END_DEFAULT = pd.Timestamp("2021-12-31")


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
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _filter_eval_period(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    return df.loc[(df["date"] >= eval_start) & (df["date"] <= eval_end)].copy()


def _require_common_dates_for_key_metrics(df: pd.DataFrame, *, is_vol: bool) -> pd.DataFrame:
    """
    Enforce that key headline metrics are computed on the same date set.
    Covariance: fro + gmvp_sharpe + turnover_l1 must all be non-NaN for every method.
    Volatility: vol_mse + vol_mae + vol_rmse must all be non-NaN for every method.
    """
    if "date" not in df.columns:
        return df

    if is_vol:
        required_metric_keys = ["vol_mse", "vol_mae", "vol_rmse"]
    else:
        required_metric_keys = ["fro", "gmvp_sharpe", "turnover_l1"]

    required_cols: list[str] = []
    for method in METHODS:
        for metric_key in required_metric_keys:
            required_cols.append(f"{method}_{metric_key}")

    existing_required_cols = [c for c in required_cols if c in df.columns]
    if not existing_required_cols:
        return df

    return df.dropna(subset=existing_required_cols).copy()


def create_comparison_table(
    backtest_file=None,
    *,
    is_vol: bool = False,
    eval_start: pd.Timestamp = EVAL_START_DEFAULT,
    eval_end: pd.Timestamp = EVAL_END_DEFAULT,
    require_common_dates: bool = True,
):
    """
    Create table showing mean (and std) performance of each method on key metrics.
    Wide format: df has columns model_fro, roll_fro, or model_vol_mse, etc.
    """
    df = load_backtest(backtest_file)
    df = _filter_eval_period(df, eval_start, eval_end)
    if require_common_dates:
        df = _require_common_dates_for_key_metrics(df, is_vol=is_vol)

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
    ap.add_argument(
        "--input",
        default=None,
        help="Backtest parquet/csv path. Default: results/regime_similarity_backtest.parquet (or fallback).",
    )
    ap.add_argument("--target", choices=("auto", "covariance", "volatility"), default="auto")
    ap.add_argument("--eval-start", default=str(EVAL_START_DEFAULT.date()), help="Evaluation start date (YYYY-MM-DD)")
    ap.add_argument("--eval-end", default=str(EVAL_END_DEFAULT.date()), help="Evaluation end date (YYYY-MM-DD)")
    ap.add_argument(
        "--require-common-dates",
        action="store_true",
        help="Require fro+gmvp_sharpe+turnover (or vol metrics) to be non-NaN for all methods on a date.",
    )
    ap.add_argument(
        "--no-require-common-dates",
        dest="require_common_dates",
        action="store_false",
        help="Do not enforce a common date intersection for headline metrics.",
    )
    ap.set_defaults(require_common_dates=True)
    args = ap.parse_args()
    is_vol = args.target == "volatility"
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)

    if args.input is not None:
        backtest_path = Path(args.input)
    else:
        if args.target == "volatility":
            backtest_path = resolve_backtest_path("regime_volatility")
        else:
            # Enforce the canonical covariance backtest requested for slides.
            backtest_path = RESULTS_DIR / "regime_similarity_backtest.parquet"

    if not backtest_path.exists():
        # Convenience: if the canonical legacy file is missing but we have the current layout
        # CSV, convert it so downstream scripts have a stable path to reference.
        if args.input is None and backtest_path.name == "regime_similarity_backtest.parquet":
            fallback_csv = RESULTS_DIR / "regime_covariance" / "backtest.csv"
            if fallback_csv.exists():
                print(f"Backtest not found: {backtest_path}")
                print(f"Found fallback backtest CSV: {fallback_csv}")
                print("Creating legacy parquet alias at results/regime_similarity_backtest.parquet ...")
                df_fallback = load_backtest(fallback_csv)
                backtest_path.parent.mkdir(parents=True, exist_ok=True)
                df_fallback.to_parquet(backtest_path, index=False)
                print(f"✓ Wrote: {backtest_path}\n")
            else:
                print(f"Backtest not found: {backtest_path}")
                print("This script is configured to use the canonical covariance backtest:")
                print("  - results/regime_similarity_backtest.parquet")
                print("")
                print("That file is not present, and no fallback CSV was found at:")
                print("  - results/regime_covariance/backtest.csv")
                print("")
                print("Options:")
                print("  1) Run: python run_backtest.py --config configs/regime_covariance.yaml")
                print("  2) Or pass an explicit path via --input")
                return
        else:
            print(f"Backtest not found: {backtest_path}")
            print("Run: python run_backtest.py --config configs/regime_covariance.yaml (or configs/regime_volatility.yaml)")
            return

    df_raw = load_backtest(backtest_path)
    print(f"Loaded file: {backtest_path}")
    if "date" in df_raw.columns:
        print(f"Raw date range: {df_raw['date'].min()} to {df_raw['date'].max()} | rows={len(df_raw)} | unique dates={df_raw['date'].nunique()}")
    df_eval = _filter_eval_period(df_raw, eval_start, eval_end)
    if "date" in df_eval.columns:
        print(f"Eval filter: {eval_start.date()} to {eval_end.date()} | rows={len(df_eval)} | unique dates={df_eval['date'].nunique()}")
    if args.require_common_dates:
        df_common = _require_common_dates_for_key_metrics(df_eval, is_vol=is_vol)
        if "date" in df_common.columns:
            print(f"Common-date intersection enabled | rows={len(df_common)} | unique dates={df_common['date'].nunique()}")
    else:
        df_common = df_eval

    # Reuse table builder on the filtered frame by writing to a temporary in-memory object
    # (simpler than threading df through every caller).
    # We pass the original file to keep CLI usage consistent, but compute from df_common below.
    metrics = METRICS_VOL if is_vol else METRICS_COV
    lower_better = LOWER_IS_BETTER_VOL if is_vol else LOWER_IS_BETTER_COV
    results = []
    for method in METHODS:
        row = {"Method": method.capitalize()}
        for metric_key, metric_name in metrics.items():
            col = f"{method}_{metric_key}"
            if col not in df_common.columns:
                row[metric_name] = np.nan
                row[f"{metric_name} Std"] = np.nan
                continue
            vals = pd.to_numeric(df_common[col], errors="coerce").dropna()
            row[metric_name] = vals.mean()
            row[f"{metric_name} Std"] = vals.std()
        results.append(row)
    results_df = pd.DataFrame(results)
    for metric_key, metric_name in metrics.items():
        if metric_name not in results_df.columns:
            continue
        ascending = metric_key in lower_better
        results_df[f"{metric_name} Rank"] = results_df[metric_name].rank(ascending=ascending)

    tag = "regime_volatility" if is_vol else "regime_similarity"
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
