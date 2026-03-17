"""
Implementation moved from scripts/analysis/visualize_statistical_comparison.py
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.analysis.utils.paths import resolve_backtest_path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_BACKTEST_COV = resolve_backtest_path("regime_covariance")
DEFAULT_BACKTEST_VOL = resolve_backtest_path("regime_volatility")


def _cov_config():
    return {
        "metrics": ("fro", "stein", "kl", "gmvp_sharpe", "gmvp_var", "turnover_l1"),
        "key_metrics_plot": ["fro", "stein", "kl", "gmvp_sharpe", "gmvp_var", "turnover_l1"],
        "lower_is_better": {"fro", "logeuc", "kl", "stein", "nll", "gmvp_var", "gmvp_vol", "turnover_l1"},
        "metric_labels": {
            "fro": "Frobenius Error\n(lower is better)",
            "stein": "Stein Loss\n(lower is better)",
            "kl": "Gaussian KL Divergence\n(lower is better)",
            "gmvp_sharpe": "GMVP Sharpe\n(higher is better)",
            "gmvp_var": "GMVP Variance\n(lower is better)",
            "turnover_l1": "Turnover L1\n(lower is better)",
        },
        "out_subdir": "regime_covariance/figs/statistical_comparison",
    }


def _vol_config():
    return {
        "metrics": ("vol_mse", "vol_mae", "vol_rmse"),
        "key_metrics_plot": ["vol_mse", "vol_mae", "vol_rmse"],
        "lower_is_better": {"vol_mse", "vol_mae", "vol_rmse"},
        "metric_labels": {
            "vol_mse": "Vol MSE (log-vol)\n(lower is better)",
            "vol_mae": "Vol MAE (log-vol)\n(lower is better)",
            "vol_rmse": "Vol RMSE (log-vol)\n(lower is better)",
        },
        "out_subdir": "regime_volatility/figs/statistical_comparison",
    }


def get_target_config(target: str, df: pd.DataFrame):
    if target == "auto":
        target = "volatility" if "model_vol_mse" in df.columns else "covariance"
    if target == "volatility":
        return _vol_config()
    return _cov_config()


def load_backtest_results(path=None):
    if path is None:
        for p in (DEFAULT_BACKTEST_COV, resolve_backtest_path("regime_covariance")):
            if Path(p).exists():
                path = p
                break
        else:
            path = DEFAULT_BACKTEST_COV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Backtest not found: {path}. "
            "Run: python run_backtest.py --config configs/regime_covariance.yaml or configs/regime_volatility.yaml"
        )
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def paired_t_test_wide(df, metric_name, method_a="model", method_b="roll"):
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
        return dict(t_statistic=np.nan, p_value=np.nan, mean_diff=np.nan, mean_a=np.nan, mean_b=np.nan, n_pairs=len(vals_a))
    t_stat, p_value = stats.ttest_rel(vals_a, vals_b)
    return dict(
        t_statistic=t_stat,
        p_value=p_value,
        mean_diff=float(np.mean(vals_a) - np.mean(vals_b)),
        mean_a=float(np.mean(vals_a)),
        mean_b=float(np.mean(vals_b)),
        n_pairs=len(vals_a),
    )


def compare_vs_reference(df, reference="model", baselines=("roll", "pers", "shrink"), metrics=()):
    results = []
    for baseline in baselines:
        for metric in metrics:
            col_ref = f"{reference}_{metric}"
            col_base = f"{baseline}_{metric}"
            if col_ref not in df.columns or col_base not in df.columns:
                continue
            test = paired_t_test_wide(df, metric, method_a=reference, method_b=baseline)
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
                    significant_5pct=test["p_value"] < 0.05 if not np.isnan(test["p_value"]) else False,
                    significant_1pct=test["p_value"] < 0.01 if not np.isnan(test["p_value"]) else False,
                )
            )
    return pd.DataFrame(results)


def normalize_difference(metric_name: str, raw_diff: float, lower_is_better: set) -> float:
    return -raw_diff if metric_name in lower_is_better else raw_diff


def plot_statistical_comparison(results_df: pd.DataFrame, out_path: Path, *, key_metrics_plot: list, metric_labels: dict, lower_is_better: set):
    if results_df.empty:
        return
    reference = results_df["reference"].iloc[0]
    results_df = results_df.copy()
    results_df["normalized_diff"] = results_df.apply(lambda r: normalize_difference(r["metric"], r["mean_diff"], lower_is_better), axis=1)
    df_plot = results_df[results_df["metric"].isin(key_metrics_plot)]
    if df_plot.empty:
        return
    n_metrics = len(key_metrics_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, max(6, 2 * n_metrics)))
    if n_metrics == 1:
        axes = [axes]
    baselines = sorted(df_plot["baseline"].unique())
    for idx, metric in enumerate(key_metrics_plot):
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
                ax.text(x_pos, y_pos, f" {marker}", va="center", ha="left" if x_pos > 0 else "right", fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{reference.capitalize()} Advantage")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        f"Statistical Comparison: {reference.capitalize()} vs Baselines\nGreen = Better than Baselines, Red = Worse than Baselines",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Statistical comparison (paired t-tests) for cov or vol backtest")
    ap.add_argument("--input", default=None, help="Backtest parquet or CSV (default: auto-detect cov/vol)")
    ap.add_argument("--target", default="auto", choices=("auto", "covariance", "volatility"), help="Target type for metrics; auto = infer from columns")
    args = ap.parse_args()
    df = load_backtest_results(path=args.input)
    config = get_target_config(args.target, df)
    out_dir = RESULTS_DIR / config["out_subdir"]
    metrics = config["metrics"]
    key_metrics = config["key_metrics_plot"]
    labels = config["metric_labels"]
    lower = config["lower_is_better"]
    model_results = compare_vs_reference(df, reference="model", baselines=("roll", "pers", "shrink"), metrics=metrics)
    plot_statistical_comparison(model_results, out_dir / "model_vs_baselines.png", key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)
    mix_results = compare_vs_reference(df, reference="mix", baselines=("roll", "pers", "shrink"), metrics=metrics)
    plot_statistical_comparison(mix_results, out_dir / "mix_vs_baselines.png", key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)


if __name__ == "__main__":
    main()

