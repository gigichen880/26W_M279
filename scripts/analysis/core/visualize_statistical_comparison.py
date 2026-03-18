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
        "metrics": ("fro", "stein", "kl", "gmvp_mean", "gmvp_sharpe", "gmvp_var", "turnover_l1"),
        "key_metrics_plot": ["fro", "stein", "kl", "gmvp_mean", "gmvp_sharpe", "gmvp_var", "turnover_l1"],
        "lower_is_better": {"fro", "logeuc", "kl", "stein", "nll", "gmvp_var", "gmvp_vol", "turnover_l1"},
        "metric_labels": {
            "fro": "Frobenius Error\n(lower is better)",
            "stein": "Stein Loss\n(lower is better)",
            "kl": "Gaussian KL Divergence\n(lower is better)",
            "gmvp_mean": "GMVP Mean Return\n(higher is better)",
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


def _sig_marker(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def plot_statistical_comparison(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    out_path: Path,
    *,
    key_metrics_plot: list,
    metric_labels: dict,
    lower_is_better: set,
):
    """Boxplots of paired difference (ref − baseline), normalized so positive = ref better. One box per baseline; t-statistic annotated."""
    if results_df.empty or raw_df.empty:
        return
    reference = results_df["reference"].iloc[0]
    baselines = ["shrink", "roll", "pers"]
    df_plot = results_df[results_df["metric"].isin(key_metrics_plot)].copy()
    if df_plot.empty:
        return
    n_metrics = len(key_metrics_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, max(5, 2 * n_metrics)))
    if n_metrics == 1:
        axes = [axes]
    ref_label = reference.capitalize()
    color_better = "#2e7d32"
    color_worse = "#c62828"
    for idx, metric in enumerate(key_metrics_plot):
        ax = axes[idx]
        col_ref = f"{reference}_{metric}"
        if col_ref not in raw_df.columns:
            continue
        ref_vals = pd.to_numeric(raw_df[col_ref], errors="coerce")
        higher_better = metric not in lower_is_better
        diffs_list = []
        positions = []
        present_baselines = []
        for baseline in baselines:
            col_base = f"{baseline}_{metric}"
            if col_base not in raw_df.columns:
                continue
            base_vals = pd.to_numeric(raw_df[col_base], errors="coerce")
            valid = np.isfinite(ref_vals) & np.isfinite(base_vals)
            r = ref_vals.loc[valid].values
            b = base_vals.loc[valid].values
            if len(r) < 2:
                continue
            raw_diff = r - b
            diff = raw_diff if higher_better else -raw_diff
            g = len(present_baselines)
            diffs_list.append(diff)
            positions.append(g)
            present_baselines.append(baseline)
        if not diffs_list:
            continue
        bp = ax.boxplot(
            diffs_list,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            zorder=1,
            vert=False,
        )
        for i, patch in enumerate(bp["boxes"]):
            med = np.median(diffs_list[i])
            patch.set_facecolor(color_better if med > 0 else color_worse)
            patch.set_alpha(0.85)
        for whisker in bp["whiskers"]:
            whisker.set_color("black")
        for cap in bp["caps"]:
            cap.set_color("black")
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels([f"vs {b.capitalize()}" for b in present_baselines], fontsize=10)
        ax.set_xlabel("Difference (positive = " + ref_label + " better)", fontsize=10)
        metric_title = metric_labels.get(metric, metric).split("\n")[0]
        ax.set_title(metric_title, fontsize=10, fontweight="bold")
        all_diffs = np.concatenate(diffs_list)
        x_min, x_max = np.nanpercentile(all_diffs, [1, 99])
        x_span = max(abs(x_min), abs(x_max), 1e-12) * 1.15
        ax.set_xlim(-x_span, x_span)
        for g, baseline in enumerate(present_baselines):
            row = df_plot[(df_plot["metric"] == metric) & (df_plot["baseline"] == baseline)]
            if row.empty:
                continue
            row = row.iloc[0]
            t_val = row.get("t_statistic", np.nan)
            p_val = row.get("p_value", np.nan)
            sig = _sig_marker(p_val)
            t_str = f"t = {t_val:.2f}{sig}" if np.isfinite(t_val) else "t = —"
            ax.text(x_span * 0.97, g, " " + t_str, ha="right", va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_better, alpha=0.85, edgecolor="black", label=ref_label + " better (median diff > 0)"),
        Patch(facecolor=color_worse, alpha=0.85, edgecolor="black", label=ref_label + " worse (median diff < 0)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), fontsize=9)
    fig.suptitle(
        f"Statistical Comparison: {ref_label} vs Baselines (paired difference; positive = {ref_label} better)",
        fontsize=12,
        fontweight="bold",
        y=0.96,
    )
    fig.text(0.5, 0.01, "Paired t-test on daily outcomes. t = test statistic; *** p<0.01, ** p<0.05, * p<0.10.", ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
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
    plot_statistical_comparison(model_results, df, out_dir / "model_vs_baselines.png", key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)
    mix_results = compare_vs_reference(df, reference="mix", baselines=("roll", "pers", "shrink"), metrics=metrics)
    plot_statistical_comparison(mix_results, df, out_dir / "mix_vs_baselines.png", key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)


if __name__ == "__main__":
    main()

