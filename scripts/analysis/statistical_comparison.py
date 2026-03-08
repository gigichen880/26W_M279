"""
Statistical comparison of Model vs all baselines using paired t-tests.
Tests whether performance differences are statistically significant.

Backtest data is in wide format: one row per date, columns like model_fro, roll_fro, etc.
"""
from pathlib import Path

import matplotlib.pyplot as plt
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


# Metrics where lower is better (error/loss); flip sign so positive = Model better
_LOWER_IS_BETTER = {"fro", "logeuc", "kl", "stein", "nll", "gmvp_var", "gmvp_vol", "turnover_l1"}


def normalize_difference(metric_name: str, raw_diff: float) -> float:
    """
    Normalize difference so positive always means Model is better.
    For lower-is-better metrics: flip sign. For higher-is-better: keep sign.
    """
    if metric_name in _LOWER_IS_BETTER:
        return -raw_diff
    return raw_diff


METRIC_LABELS_PLOT = {
    "fro": "Frobenius Error\n(Model advantage = lower error)",
    "logeuc": "Log-Euclidean Distance\n(Model advantage = lower distance)",
    "gmvp_sharpe": "GMVP Sharpe Ratio\n(Model advantage = higher Sharpe)",
    "gmvp_var": "GMVP Variance\n(Model advantage = lower variance)",
    "turnover_l1": "Turnover (L1)\n(Model advantage = lower turnover)",
}

KEY_METRICS_PLOT = ["fro", "logeuc", "gmvp_sharpe", "gmvp_var", "turnover_l1"]


def plot_statistical_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar chart of Model vs baselines with normalized differences.
    Positive bars ALWAYS mean Model is better. Green = Model better, Red = Model worse.
    """
    if results_df.empty:
        return
    # Ensure booleans (CSV may have read them as strings)
    for c in ("significant_5pct", "significant_1pct"):
        if c in results_df.columns:
            results_df[c] = results_df[c].replace({"True": True, "False": False}).fillna(False).astype(bool)

    results_df = results_df.copy()
    results_df["normalized_diff"] = results_df.apply(
        lambda row: normalize_difference(row["metric"], row["mean_diff"]), axis=1
    )
    df_plot = results_df[results_df["metric"].isin(KEY_METRICS_PLOT)].copy()

    if df_plot.empty:
        return

    fig, axes = plt.subplots(len(KEY_METRICS_PLOT), 1, figsize=(10, 12))
    if len(KEY_METRICS_PLOT) == 1:
        axes = [axes]
    baselines = ["roll", "pers", "shrink", "mix"]

    for idx, metric in enumerate(KEY_METRICS_PLOT):
        ax = axes[idx]
        metric_df = df_plot[df_plot["metric"] == metric].copy()
        metric_df = metric_df.set_index("baseline").reindex(baselines).reset_index()
        metric_df = metric_df.dropna(subset=["normalized_diff"])
        if metric_df.empty:
            ax.set_title(METRIC_LABELS_PLOT.get(metric, metric), fontsize=11, fontweight="bold")
            ax.axvline(x=0, color="black", linewidth=1)
            continue

        y_labels = metric_df["baseline"].str.capitalize().tolist()
        normalized = metric_df["normalized_diff"].values
        colors = ["#2e7d32" if x > 0 else "#c62828" for x in normalized]
        bars = ax.barh(y_labels, normalized, color=colors, alpha=0.7, edgecolor="black")

        for bar, (_, row) in zip(bars, metric_df.iterrows()):
            marker = "***" if row.get("significant_1pct") else ("**" if row.get("significant_5pct") else "")
            if marker:
                x_pos = bar.get_width()
                y_pos = bar.get_y() + bar.get_height() / 2
                ax.text(
                    x_pos, y_pos, f" {marker}",
                    va="center", ha="left" if x_pos > 0 else "right",
                    fontsize=10, fontweight="bold",
                )

        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlabel("Model Advantage (positive = Model better)", fontsize=10)
        ax.set_title(METRIC_LABELS_PLOT.get(metric, metric), fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric in ("fro", "gmvp_var"):
            ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    fig.suptitle(
        "Statistical Comparison: Model vs Baselines (Paired t-tests)\nGreen = Model Better, Red = Model Worse",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


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

    csv_path = RESULTS_DIR / "statistical_comparison.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    fig_path = RESULTS_DIR / "figs_regime_similarity" / "statistical_comparison.png"
    plot_statistical_comparison(results_df, fig_path)
    print(f"Saved figure to {fig_path}\n")

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
    import argparse
    parser = argparse.ArgumentParser(description="Statistical comparison: Model vs baselines (paired t-tests)")
    parser.add_argument("--plot-only", type=str, default=None, metavar="CSV",
                        help="Only generate figure from existing results/statistical_comparison.csv")
    args = parser.parse_args()
    if args.plot_only:
        csv_path = Path(args.plot_only)
        if not csv_path.exists():
            csv_path = RESULTS_DIR / "statistical_comparison.csv"
        df = pd.read_csv(csv_path)
        fig_path = RESULTS_DIR / "figs_regime_similarity" / "statistical_comparison.png"
        plot_statistical_comparison(df, fig_path)
        print(f"Saved figure to {fig_path}")
    else:
        main()
