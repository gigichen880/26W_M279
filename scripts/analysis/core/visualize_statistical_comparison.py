"""
Paired statistical comparison vs baselines (roll, pers, shrink).

Writes under results/<tag>/figs/statistical_comparison/:
  - model_vs_baselines.png, mix_vs_baselines.png — boxplots of daily paired differences
  - model_vs_baselines_meanbars.png, mix_vs_baselines_meanbars.png — horizontal bars of
    mean paired advantage (+ = reference better), with paired t-stat and significance
  - paired_comparison_summary_model.csv, paired_comparison_summary_mix.csv — median Δ̃ and
    Wilcoxon p per metric×baseline (same construction as the boxplots; cite these, not report.csv)
  - forecast_correlation.png (+ .csv) — correlation of model/baseline metric series

Headless runs: `MPLBACKEND=Agg python -m scripts.analysis.core.visualize_statistical_comparison ...`
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.analysis.utils.backtest_io import read_backtest_table
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
        "metrics": ("vol_mse", "vol_mae", "vol_rmse", "vol_qlike", "vol_r2"),
        "key_metrics_plot": ["vol_mse", "vol_mae", "vol_rmse", "vol_qlike", "vol_r2"],
        "lower_is_better": {"vol_mse", "vol_mae", "vol_rmse", "vol_qlike"},
        "metric_labels": {
            "vol_mse": "Vol MSE (log-vol)\n(lower is better)",
            "vol_mae": "Vol MAE (log-vol)\n(lower is better)",
            "vol_rmse": "Vol RMSE (log-vol)\n(lower is better)",
            "vol_qlike": "QLIKE (Quasi-Likelihood)\n(lower is better)",
            "vol_r2": "R² (variance explained)\n(higher is better)",
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
    df = read_backtest_table(path)
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

def plot_mean_advantage_bars(
    results_df: pd.DataFrame,
    out_path: Path,
    *,
    key_metrics_plot: list,
    metric_labels: dict,
    lower_is_better: set,
) -> None:
    """
    Old-style summary: per metric, horizontal bars of mean paired advantage.
    Advantage is normalized so positive always means reference is better.
    Annotate with t-statistic and significance marker from paired t-test.
    """
    if results_df.empty:
        return
    df_plot = results_df[results_df["metric"].isin(key_metrics_plot)].copy()
    if df_plot.empty:
        return
    reference = str(df_plot["reference"].iloc[0])
    baselines_order = ["shrink", "roll", "pers"]
    color_better = "#2e7d32"
    color_worse = "#c62828"

    n_metrics = len(key_metrics_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, max(5, 2.2 * n_metrics)))
    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(key_metrics_plot):
        ax = axes[i]
        sub = df_plot[df_plot["metric"] == metric].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        # enforce baseline order if present
        sub["baseline"] = sub["baseline"].astype(str)
        sub["baseline_rank"] = sub["baseline"].map({b: j for j, b in enumerate(baselines_order)}).fillna(999)
        sub = sub.sort_values("baseline_rank")

        # normalize mean diff so positive = reference better
        raw = pd.to_numeric(sub["mean_diff"], errors="coerce").values
        higher_better = metric not in lower_is_better
        adv = raw if higher_better else -raw

        y = np.arange(len(sub))
        colors = [color_better if (np.isfinite(a) and a > 0) else color_worse for a in adv]
        ax.barh(y, adv, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([b.capitalize() for b in sub["baseline"].tolist()])

        title = metric_labels.get(metric, metric)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"{reference.capitalize()} advantage (mean; + = better)")
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # annotate with t-stat and sig marker
        for yy, (_, r) in zip(y, sub.iterrows()):
            t = r.get("t_statistic", np.nan)
            p = r.get("p_value", np.nan)
            sig = _sig_marker(float(p)) if np.isfinite(p) else ""
            if np.isfinite(t):
                txt = f"t={float(t):.2f}{sig}"
                # place to the right of bar end (or slightly right of zero)
                x_txt = float(adv[yy]) if np.isfinite(adv[yy]) else 0.0
                pad = 0.01 * (np.nanmax(np.abs(adv)) if np.isfinite(np.nanmax(np.abs(adv))) else 1.0)
                ax.text(x_txt + (pad if x_txt >= 0 else -pad), yy, txt,
                        va="center", ha="left" if x_txt >= 0 else "right", fontsize=9)

    fig.suptitle(
        f"Statistical Comparison (mean advantage): {reference.capitalize()} vs baselines",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_wilcoxon_summary(
    raw_df: pd.DataFrame,
    *,
    reference: str,
    baselines: tuple[str, ...],
    key_metrics: list[str],
    lower_is_better: set,
) -> pd.DataFrame:
    """
    Same paired differences as `plot_statistical_comparison` (positive = reference better).
    Median advantage + Wilcoxon signed-rank p-value per (metric, baseline).
    """
    rows: list[dict] = []
    for metric in key_metrics:
        col_ref = f"{reference}_{metric}"
        if col_ref not in raw_df.columns:
            continue
        ref_vals = pd.to_numeric(raw_df[col_ref], errors="coerce")
        higher_better = metric not in lower_is_better
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
            med = float(np.median(diff))
            try:
                _, p = stats.wilcoxon(diff)
            except Exception:
                p = np.nan
            rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "baseline": baseline,
                    "n_pairs": int(len(r)),
                    "median_advantage": med,
                    "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
                    "sig": _sig_marker(float(p)) if np.isfinite(p) else "",
                }
            )
    return pd.DataFrame(rows)


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
        # Color by median difference (aligns with Wilcoxon test)
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
        # Annotate with median difference and Wilcoxon signed-rank test (median-based)
        for g, baseline in enumerate(present_baselines):
            diff = diffs_list[g]
            med = float(np.median(diff))
            try:
                w_stat, w_p = stats.wilcoxon(diff)
                sig = _sig_marker(w_p)
            except Exception:
                w_p, sig = np.nan, ""
            ann = f"Δ̃ = {med:.3f}{sig}" if np.isfinite(med) else "Δ̃ = —"
            ax.text(x_span * 0.97, g, " " + ann, ha="right", va="center", fontsize=9)
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
    fig.text(0.5, 0.01, "Box = distribution of daily differences. Color and Δ̃ = median. Wilcoxon signed-rank test; *** p<0.01, ** p<0.05, * p<0.10.", ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_forecast_correlation(
    raw_df: pd.DataFrame,
    out_dir: Path,
    *,
    target: str,
):
    """
    Correlation among model and baselines only (excludes mix). For cov: fro series; for vol: vol_mse.
    Saves forecast_correlation.png and forecast_correlation.csv.
    """
    methods = ["model", "roll", "pers", "shrink"]
    if target == "volatility":
        metric = "vol_mse"
        title_metric = "Vol MSE (log-vol)"
    else:
        metric = "fro"
        title_metric = "Frobenius error"
    cols = [f"{m}_{metric}" for m in methods]
    missing = [c for c in cols if c not in raw_df.columns]
    if missing:
        return
    block = raw_df[cols].copy()
    block = block.rename(columns={c: m for c, m in zip(cols, methods)})
    block = block.dropna(how="all")
    for m in methods:
        block[m] = pd.to_numeric(block[m], errors="coerce")
    block = block.dropna()
    if block.shape[0] < 3:
        return
    corr = block.corr()
    corr = corr.reindex(index=methods, columns=methods)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_dir / "forecast_correlation.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(len(methods)):
        for j in range(len(methods)):
            v = corr.iloc[i, j]
            txt = f"{v:.2f}" if np.isfinite(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels([m.capitalize() for m in methods], fontsize=11)
    ax.set_yticklabels([m.capitalize() for m in methods], fontsize=11)
    ax.set_title(f"Forecast correlation across dates ({title_metric})\nModel + baselines only (no mix)", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.02, "High correlation → forecasts move together → ensembling adds little diversification of error risk.", ha="center", fontsize=9, style="italic", wrap=True)
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(out_dir / "forecast_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Statistical comparison (paired t-tests) for cov or vol backtest")
    ap.add_argument("--input", default=None, help="Backtest parquet or CSV (default: auto-detect cov/vol)")
    ap.add_argument("--target", default="auto", choices=("auto", "covariance", "volatility"), help="Target type for metrics; auto = infer from columns")
    ap.add_argument(
        "--plots",
        nargs="+",
        choices=("all", "box", "meanbars", "correlation"),
        default=["all"],
        help="Figures to write (default: all). E.g. --plots meanbars or --plots box meanbars",
    )
    args = ap.parse_args()
    want = set(args.plots)
    if "all" in want:
        want = {"box", "meanbars", "correlation"}

    df = load_backtest_results(path=args.input)
    config = get_target_config(args.target, df)
    out_dir = RESULTS_DIR / config["out_subdir"]
    metrics = config["metrics"]
    key_metrics = config["key_metrics_plot"]
    labels = config["metric_labels"]
    lower = config["lower_is_better"]
    saved: list[Path] = []

    model_results = compare_vs_reference(df, reference="model", baselines=("roll", "pers", "shrink"), metrics=metrics)
    summ_model = paired_wilcoxon_summary(
        df,
        reference="model",
        baselines=("shrink", "roll", "pers"),
        key_metrics=key_metrics,
        lower_is_better=lower,
    )
    summ_mix = paired_wilcoxon_summary(
        df,
        reference="mix",
        baselines=("shrink", "roll", "pers"),
        key_metrics=key_metrics,
        lower_is_better=lower,
    )
    for name, sdf in (("paired_comparison_summary_model.csv", summ_model), ("paired_comparison_summary_mix.csv", summ_mix)):
        if not sdf.empty:
            p_csv = out_dir / name
            sdf.to_csv(p_csv, index=False)
            saved.append(p_csv)

    if "box" in want:
        p = out_dir / "model_vs_baselines.png"
        plot_statistical_comparison(model_results, df, p, key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)
        if p.exists():
            saved.append(p)
    if "meanbars" in want:
        p = out_dir / "model_vs_baselines_meanbars.png"
        plot_mean_advantage_bars(model_results, p, key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)
        if p.exists():
            saved.append(p)

    mix_results = compare_vs_reference(df, reference="mix", baselines=("roll", "pers", "shrink"), metrics=metrics)
    if "box" in want:
        p = out_dir / "mix_vs_baselines.png"
        plot_statistical_comparison(mix_results, df, p, key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)
        if p.exists():
            saved.append(p)
    if "meanbars" in want:
        p = out_dir / "mix_vs_baselines_meanbars.png"
        plot_mean_advantage_bars(mix_results, p, key_metrics_plot=key_metrics, metric_labels=labels, lower_is_better=lower)
        if p.exists():
            saved.append(p)

    target_resolved = "volatility" if "model_vol_mse" in df.columns else "covariance"
    if "correlation" in want:
        plot_forecast_correlation(df, out_dir, target=target_resolved)
        for name in ("forecast_correlation.png", "forecast_correlation.csv"):
            q = out_dir / name
            if q.exists():
                saved.append(q)

    for p in saved:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()

