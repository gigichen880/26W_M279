"""
Statistical comparison of Model vs all baselines using paired t-tests.
Tests whether performance differences are statistically significant.

Backtest data is in wide format: one row per date, columns like model_fro, roll_fro, etc.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_BACKTEST = RESULTS_DIR / "regime_similarity_backtest.parquet"
DEFAULT_BACKTEST_CSV = RESULTS_DIR / "regime_similarity_backtest.csv"


def load_backtest_results(path=None):
    """Load backtest results (parquet or CSV)."""
    if path is None:
        path = DEFAULT_BACKTEST if DEFAULT_BACKTEST.exists() else DEFAULT_BACKTEST_CSV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Backtest not found: {path}. Run: python run_backtest.py --config configs/regime_similarity.yaml"
        )
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def paired_t_test_wide(df, metric_name, method_a="model", method_b="roll"):
    """
    Perform paired t-test between two methods on a given metric.
    Assumes wide format: columns method_a_metric, method_b_metric.

    Returns:
        dict with t_statistic, p_value, mean_diff, mean_a, mean_b, n_pairs
    """
    col_a = f"{method_a}_{metric_name}"
    col_b = f"{method_b}_{metric_name}"
    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"Missing columns: {col_a} or {col_b}")

    vals_a = pd.to_numeric(df[col_a], errors="coerce").values
    vals_b = pd.to_numeric(df[col_b], errors="coerce").values

    valid_mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
    vals_a = vals_a[valid_mask]
    vals_b = vals_b[valid_mask]

    if len(vals_a) < 2:
        return {
            "t_statistic": np.nan,
            "p_value": np.nan,
            "mean_diff": np.nan,
            "mean_a": np.nan,
            "mean_b": np.nan,
            "n_pairs": len(vals_a),
        }

    t_stat, p_value = stats.ttest_rel(vals_a, vals_b)
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_diff": float(np.mean(vals_a) - np.mean(vals_b)),
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "n_pairs": len(vals_a),
    }


def compare_all_baselines(
    df,
    metrics=("fro", "logeuc", "gmvp_sharpe", "gmvp_var", "turnover_l1"),
    baselines=("roll", "pers", "shrink", "mix"),
):
    """
    Compare Model vs all baselines on multiple metrics (wide-format backtest).

    Returns DataFrame with statistical test results.
    """
    results = []
    for baseline in baselines:
        for metric in metrics:
            col_model = f"model_{metric}"
            col_base = f"{baseline}_{metric}"
            if col_model not in df.columns or col_base not in df.columns:
                continue
            try:
                test_result = paired_t_test_wide(df, metric, method_a="model", method_b=baseline)
                results.append({
                    "baseline": baseline,
                    "metric": metric,
                    "model_mean": test_result["mean_a"],
                    "baseline_mean": test_result["mean_b"],
                    "mean_diff": test_result["mean_diff"],
                    "t_statistic": test_result["t_statistic"],
                    "p_value": test_result["p_value"],
                    "n_pairs": test_result["n_pairs"],
                    "significant_5pct": test_result["p_value"] < 0.05 if not np.isnan(test_result["p_value"]) else False,
                    "significant_1pct": test_result["p_value"] < 0.01 if not np.isnan(test_result["p_value"]) else False,
                })
            except Exception as e:
                print(f"Error comparing {baseline} on {metric}: {e}")
    return pd.DataFrame(results)


def print_comparison_table(results_df):
    """Print formatted comparison table."""
    if results_df.empty:
        print("No comparison results to display.")
        return

    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISON: Model vs Baselines (Paired t-tests)")
    print("=" * 100 + "\n")

    metrics_display = {
        "fro": "Frobenius Error",
        "logeuc": "Log-Euclidean Distance",
        "gmvp_sharpe": "GMVP Sharpe Ratio",
        "gmvp_var": "GMVP Variance",
        "turnover_l1": "Turnover (L1)",
    }

    for metric in results_df["metric"].unique():
        metric_df = results_df[results_df["metric"] == metric]
        print(f"\n{metrics_display.get(metric, metric)}:")
        print("-" * 100)
        print(
            f"{'Baseline':<12} {'Model Mean':<12} {'Base Mean':<12} {'Diff':<12} "
            f"{'t-stat':<10} {'p-value':<10} {'Sig?':<8}"
        )
        print("-" * 100)
        for _, row in metric_df.iterrows():
            sig_marker = "***" if row["significant_1pct"] else ("**" if row["significant_5pct"] else "")
            print(
                f"{row['baseline']:<12} "
                f"{row['model_mean']:<12.4f} "
                f"{row['baseline_mean']:<12.4f} "
                f"{row['mean_diff']:<+12.4f} "
                f"{row['t_statistic']:<+10.3f} "
                f"{row['p_value']:<10.4f} "
                f"{sig_marker:<8}"
            )
        print()

    print("Significance levels: *** p<0.01, ** p<0.05")
    print("Positive difference = Model better (for Sharpe)")
    print("Negative difference = Model better (for Fro, LogEuc, Var, Turnover)")
    print()


def main():
    """Run statistical comparison."""
    print("\n" + "=" * 100)
    print("LOADING BACKTEST RESULTS FOR STATISTICAL COMPARISON")
    print("=" * 100 + "\n")

    df = load_backtest_results()
    print(f"Loaded {len(df)} rows")
    methods = [c.split("_", 1)[0] for c in df.columns if "_" in c and c not in ("regime_assigned",)]
    methods = sorted(set(m for m in methods if m in ("model", "roll", "pers", "shrink", "mix")))
    print(f"  Methods present: {methods}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    results_df = compare_all_baselines(df)
    print_comparison_table(results_df)

    out_path = RESULTS_DIR / "statistical_comparison.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}\n")

    n_total = len(results_df)
    if n_total > 0:
        n_sig5 = results_df["significant_5pct"].sum()
        n_sig1 = results_df["significant_1pct"].sum()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Total comparisons: {n_total}")
        print(f"Significant at 5% level: {n_sig5} ({n_sig5 / n_total * 100:.1f}%)")
        print(f"Significant at 1% level: {n_sig1} ({n_sig1 / n_total * 100:.1f}%)")
        print()


if __name__ == "__main__":
    main()
