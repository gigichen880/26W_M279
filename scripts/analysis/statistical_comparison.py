"""
Statistical comparison of methods using paired t-tests.

Generates:
1) Model vs baselines (excluding mix)
2) Mix vs baselines (excluding model)

Backtest data is in wide format: one row per date,
columns like model_fro, roll_fro, mix_fro, etc.

Usage:
python scripts/analysis/statistical_comparison.py
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_backtest_results(path=None):
    """Load backtest results (parquet or CSV)."""

    if path is None:
        path = DEFAULT_BACKTEST if DEFAULT_BACKTEST.exists() else DEFAULT_BACKTEST_CSV

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Backtest not found: {path}. "
            f"Run: python run_backtest.py --config configs/regime_similarity.yaml"
        )

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"])

    return df


# =============================================================================
# STATISTICAL TEST
# =============================================================================

def paired_t_test_wide(df, metric_name, method_a="model", method_b="roll"):
    """
    Perform paired t-test between two methods on a given metric.
    Assumes wide format columns: method_metric
    """

    col_a = f"{method_a}_{metric_name}"
    col_b = f"{method_b}_{metric_name}"

    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"Missing columns: {col_a} or {col_b}")

    vals_a = pd.to_numeric(df[col_a], errors="coerce").values
    vals_b = pd.to_numeric(df[col_b], errors="coerce").values

    valid = ~(np.isnan(vals_a) | np.isnan(vals_b))

    vals_a = vals_a[valid]
    vals_b = vals_b[valid]

    if len(vals_a) < 2:
        return dict(
            t_statistic=np.nan,
            p_value=np.nan,
            mean_diff=np.nan,
            mean_a=np.nan,
            mean_b=np.nan,
            n_pairs=len(vals_a),
        )

    t_stat, p_value = stats.ttest_rel(vals_a, vals_b)

    return dict(
        t_statistic=t_stat,
        p_value=p_value,
        mean_diff=float(np.mean(vals_a) - np.mean(vals_b)),
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        n_pairs=len(vals_a),
    )


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def compare_vs_reference(
    df,
    reference="model",
    baselines=("roll", "pers", "shrink"),
    metrics=("fro", "logeuc", "gmvp_sharpe", "gmvp_var", "turnover_l1"),
):
    """
    Compare reference method vs baselines.
    """

    results = []

    for baseline in baselines:

        for metric in metrics:

            col_ref = f"{reference}_{metric}"
            col_base = f"{baseline}_{metric}"

            if col_ref not in df.columns or col_base not in df.columns:
                continue

            try:

                test = paired_t_test_wide(
                    df,
                    metric,
                    method_a=reference,
                    method_b=baseline,
                )

                results.append(
                    dict(
                        reference=reference,
                        baseline=baseline,
                        metric=metric,
                        ref_mean=test["mean_a"],
                        baseline_mean=test["mean_b"],
                        mean_diff=test["mean_diff"],
                        t_statistic=test["t_statistic"],
                        p_value=test["p_value"],
                        n_pairs=test["n_pairs"],
                        significant_5pct=test["p_value"] < 0.05
                        if not np.isnan(test["p_value"])
                        else False,
                        significant_1pct=test["p_value"] < 0.01
                        if not np.isnan(test["p_value"])
                        else False,
                    )
                )

            except Exception as e:
                print(f"Error comparing {baseline} on {metric}: {e}")

    return pd.DataFrame(results)


# =============================================================================
# METRIC NORMALIZATION
# =============================================================================

_LOWER_IS_BETTER = {
    "fro",
    "logeuc",
    "kl",
    "stein",
    "nll",
    "gmvp_var",
    "gmvp_vol",
    "turnover_l1",
}


def normalize_difference(metric_name: str, raw_diff: float) -> float:
    """
    Normalize difference so positive always means reference method better.
    """

    if metric_name in _LOWER_IS_BETTER:
        return -raw_diff

    return raw_diff


METRIC_LABELS_PLOT = {
    "fro": "Frobenius Error\n(lower is better)",
    "logeuc": "Log-Euclidean Distance\n(lower is better)",
    "gmvp_sharpe": "GMVP Sharpe\n(higher is better)",
    "gmvp_var": "GMVP Variance\n(lower is better)",
    "turnover_l1": "Turnover L1\n(lower is better)",
}

KEY_METRICS_PLOT = [
    "fro",
    "logeuc",
    "gmvp_sharpe",
    "gmvp_var",
    "turnover_l1",
]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_statistical_comparison(results_df: pd.DataFrame, out_path: Path):
    """
    Horizontal bar plot showing reference advantage.
    """

    if results_df.empty:
        return

    reference = results_df["reference"].iloc[0]

    results_df = results_df.copy()

    results_df["normalized_diff"] = results_df.apply(
        lambda r: normalize_difference(r["metric"], r["mean_diff"]), axis=1
    )

    df_plot = results_df[results_df["metric"].isin(KEY_METRICS_PLOT)]

    fig, axes = plt.subplots(len(KEY_METRICS_PLOT), 1, figsize=(10, 12))

    if len(KEY_METRICS_PLOT) == 1:
        axes = [axes]

    baselines = sorted(df_plot["baseline"].unique())

    for idx, metric in enumerate(KEY_METRICS_PLOT):

        ax = axes[idx]

        metric_df = df_plot[df_plot["metric"] == metric]

        metric_df = metric_df.set_index("baseline").reindex(baselines).reset_index()

        metric_df = metric_df.dropna(subset=["normalized_diff"])

        if metric_df.empty:
            continue

        y_labels = metric_df["baseline"].str.capitalize()

        normalized = metric_df["normalized_diff"].values

        colors = ["#2e7d32" if x > 0 else "#c62828" for x in normalized]

        bars = ax.barh(y_labels, normalized, color=colors, alpha=0.7, edgecolor="black")

        for bar, (_, row) in zip(bars, metric_df.iterrows()):

            marker = "***" if row["significant_1pct"] else ("**" if row["significant_5pct"] else "")

            if marker:

                x_pos = bar.get_width()
                y_pos = bar.get_y() + bar.get_height() / 2

                ax.text(
                    x_pos,
                    y_pos,
                    f" {marker}",
                    va="center",
                    ha="left" if x_pos > 0 else "right",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.axvline(0, color="black", linewidth=1)

        ax.set_title(METRIC_LABELS_PLOT.get(metric, metric), fontsize=11, fontweight="bold")

        ax.set_xlabel(f"{reference.capitalize()} Advantage")

        ax.grid(axis="x", alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Statistical Comparison: {reference.capitalize()} vs Baselines\n"
        "Green = Reference Better, Red = Reference Worse",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n" + "=" * 100)
    print("LOADING BACKTEST RESULTS")
    print("=" * 100 + "\n")

    df = load_backtest_results()

    print(f"Loaded {len(df)} rows")

    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    # ---------------------------------------------------
    # Model vs baselines (exclude mix)
    # ---------------------------------------------------

    model_results = compare_vs_reference(
        df,
        reference="model",
        baselines=("roll", "pers", "shrink"),
    )

    fig_path_model = RESULTS_DIR / "figs_regime_similarity" / "statistical_comparison" / "model_vs_baselines.png"

    plot_statistical_comparison(model_results, fig_path_model)

    print(f"Saved figure: {fig_path_model}")

    # ---------------------------------------------------
    # Mix vs baselines (exclude model)
    # ---------------------------------------------------

    mix_results = compare_vs_reference(
        df,
        reference="mix",
        baselines=("roll", "pers", "shrink"),
    )

    fig_path_mix = RESULTS_DIR / "figs_regime_similarity" / "statistical_comparison" / "mix_vs_baselines.png"

    plot_statistical_comparison(mix_results, fig_path_mix)

    print(f"Saved figure: {fig_path_mix}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    main()