# scripts/analysis/run_ablation.py
"""
Systematic ablation over high-level design choices in the similarity_forecast pipeline.

Evaluates one axis at a time: for each choice, runs the backtest and records summary
metrics (primary metric mean for model/mix). Use to isolate the effect of:
  - embedder (pca, corr_eig, vol_stats for vol)
  - transition_estimator (hard vs soft)
  - knn_metric (l2 vs l1)
  - regime_aggregation (soft vs hard)
  - regime_weighting (filtered vs raw_pi)

Usage:
  python -m scripts.analysis.run_ablation --config configs/ablation_covariance.yaml
  python -m scripts.analysis.run_ablation --config configs/ablation_volatility.yaml
"""

from __future__ import annotations

import os
import argparse
import copy
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from scripts.config_utils import load_yaml, deep_update
from run_backtest import run_backtest_from_config


def _set_dotted(cfg: dict, key: str, value: Any) -> dict:
    """Return a nested dict that sets cfg[key_path] = value. E.g. 'model.knn_metric' -> {'model': {'knn_metric': value}}."""
    parts = key.split(".")
    out: dict = {parts[-1]: value}
    for p in reversed(parts[:-1]):
        out = {p: out}
    return out


def _get_dotted(cfg: dict, key: str, default: Any = None) -> Any:
    """Get value at dotted path, e.g. 'model.knn_metric' -> cfg['model']['knn_metric']."""
    parts = key.split(".")
    cur = cfg
    for p in parts:
        cur = cur.get(p) if isinstance(cur, dict) else default
        if cur is None:
            return default
    return cur


PlotsMode = Literal["none", "summary", "per_run", "all"]


def run_ablation(
    cfg_path: str,
    out_dir: str | None = None,
    *,
    verbose: bool = False,
    plots: PlotsMode = "summary",
) -> pd.DataFrame:
    """
    Load ablation config, run one-at-a-time ablations, return summary table.

    Ablation config must have:
      base_config: path to base YAML (e.g. regime_covariance.yaml)
      axes: dict of axis_name -> { key: "dotted.path", choices: [v1, v2, ...] }
    """
    cfg = load_yaml(cfg_path)
    base_path = cfg["base_config"]
    if not os.path.isabs(base_path):
        base_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(cfg_path)), base_path)
        )
    base_cfg = load_yaml(base_path)
    axes_cfg = cfg["axes"]
    mode = str(cfg.get("mode", "one_at_a_time")).lower()
    tag = str(base_cfg.get("outputs", {}).get("tag", "regime_covariance"))
    out_dir = out_dir or cfg.get("outputs", {}).get("outdir") or os.path.join("results", tag, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    target_type = str(_get_dotted(base_cfg, "model.target", "covariance")).lower()
    if target_type == "volatility":
        primary_metric = "vol_mse"
        extra_metrics = ["vol_mae", "vol_rmse"]
        gmvp_metrics: list[str] = []
    else:
        primary_metric = "fro"
        extra_metrics = ["stein", "logeuc"]
        gmvp_metrics = ["gmvp_sharpe", "gmvp_var", "gmvp_vol", "turnover_l1"]

    rows: list[dict] = []
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    for axis_name, axis_spec in axes_cfg.items():
        key_path = axis_spec["key"]
        choices = list(axis_spec["choices"])
        base_value = _get_dotted(base_cfg, key_path)
        if base_value is not None and isinstance(base_value, str):
            base_value = base_value.lower()

        for choice in choices:
            choice_str = str(choice).lower() if isinstance(choice, str) else choice
            merged = deep_update(copy.deepcopy(base_cfg), _set_dotted({}, key_path, choice))
            run_tag = f"{axis_name}={choice_str}"
            print(f"  Running ablation: {run_tag} ...")
            try:
                results_df, _ = run_backtest_from_config(merged, verbose=verbose)
            except Exception as e:
                print(f"    Failed: {e}")
                rows.append({
                    "axis": axis_name,
                    "choice": choice_str,
                    "primary_metric": primary_metric,
                    "model_mean": None,
                    "mix_mean": None,
                    "run_tag": run_tag,
                    "error": str(e),
                })
                continue

            # Persist time series so we can visualize later (and avoid reruns)
            axis_dir = os.path.join(runs_dir, axis_name)
            os.makedirs(axis_dir, exist_ok=True)
            run_base = os.path.join(axis_dir, f"{choice_str}")
            try:
                results_df.to_parquet(run_base + ".parquet")
            except Exception:
                # Fallback: always ensure we save something readable.
                results_df.to_csv(run_base + ".csv", index=True)

            model_col = f"model_{primary_metric}"
            mix_col = f"mix_{primary_metric}"
            model_mean = float(results_df[model_col].mean()) if model_col in results_df.columns else None
            mix_mean = float(results_df[mix_col].mean()) if mix_col in results_df.columns else None

            row = {
                "axis": axis_name,
                "choice": choice_str,
                "primary_metric": primary_metric,
                "model_mean": model_mean,
                "mix_mean": mix_mean,
                "run_tag": run_tag,
            }
            for m in extra_metrics:
                for pref in ("model", "mix"):
                    c = f"{pref}_{m}"
                    if c in results_df.columns:
                        row[c + "_mean"] = float(results_df[c].mean())
            # Covariance-only: also record GMVP-related means (and preserve time series in parquet above)
            for m in gmvp_metrics:
                for pref in ("model", "mix", "roll", "pers", "shrink"):
                    c = f"{pref}_{m}"
                    if c in results_df.columns:
                        row[c + "_mean"] = float(pd.to_numeric(results_df[c], errors="coerce").mean())
            rows.append(row)

    summary = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    with open(os.path.join(out_dir, "ablation_config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ----------------------------
    # Visualization (optional)
    # ----------------------------
    if plots == "none":
        return summary

    figs_dir = os.path.join("results", tag, "figs", "ablation")
    os.makedirs(figs_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available; skipping ablation plots.)")
        return summary

    if plots in {"summary", "all"}:
        # Plot: summary bars by axis/choice
        try:
            axes = list(summary["axis"].unique())
            n = len(axes)
            fig, axs = plt.subplots(n, 1, figsize=(12, max(3, 2.6 * n)))
            if n == 1:
                axs = [axs]
            for i, ax in enumerate(axs):
                a = axes[i]
                sub = summary[summary["axis"] == a].copy()
                sub = sub.sort_values("choice")
                x = np.arange(len(sub))
                w = 0.35
                ax.bar(x - w / 2, sub["model_mean"].astype(float), w, label="model", color="steelblue", alpha=0.9)
                ax.bar(x + w / 2, sub["mix_mean"].astype(float), w, label="mix", color="gray", alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(sub["choice"].astype(str).tolist())
                ax.set_title(f"{a}: mean {primary_metric} (lower is better)")
                ax.grid(alpha=0.3, axis="y")
                if i == 0:
                    ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(figs_dir, "ablation_summary.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"(warn) Failed to plot ablation summary: {e}")

    # Plot: per-axis time series for key metrics using saved parquet/csv
    def _load_run_df(pbase: str) -> pd.DataFrame:
        if os.path.exists(pbase + ".parquet"):
            df = pd.read_parquet(pbase + ".parquet")
        else:
            df = pd.read_csv(pbase + ".csv")
        if df.index.name == "date" or "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df.replace([np.inf, -np.inf], np.nan)

    def _plot_timeseries(df: pd.DataFrame, metric: str, outpath: str, title: str) -> None:
        cols = [c for c in (f"model_{metric}", f"mix_{metric}", f"roll_{metric}") if c in df.columns]
        if not cols:
            return
        plt.figure(figsize=(12, 4))
        for c in cols:
            plt.plot(df[c], label=c.split("_", 1)[0], alpha=0.9 if c.startswith(("model", "mix")) else 0.6)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()

    def _plot_cum_adv(df: pd.DataFrame, ref: str, metric: str, outpath: str, title: str) -> None:
        ref_col = f"{ref}_{metric}"
        roll_col = f"roll_{metric}"
        if ref_col not in df.columns or roll_col not in df.columns:
            return
        ref_s = pd.to_numeric(df[ref_col], errors="coerce")
        roll_s = pd.to_numeric(df[roll_col], errors="coerce")
        # For error metrics: positive advantage = ref better = (roll - ref)
        diff = roll_s - ref_s
        plt.figure(figsize=(12, 4))
        plt.plot(diff.cumsum(), label=f"{ref} vs roll")
        plt.axhline(0, color="black", linestyle="--", alpha=0.7)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()

    def _plot_axis_overlay_timeseries(
        axis_name: str,
        dfs_by_choice: dict[str, pd.DataFrame],
        *,
        metric: str,
        outpath: str,
        title: str,
    ) -> None:
        if not dfs_by_choice:
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Panel 1: model
        any_model = False
        for choice, df in dfs_by_choice.items():
            c = f"model_{metric}"
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            ax1.plot(s.index, s.values, label=str(choice))
            any_model = True
        ax1.set_title("model")
        ax1.grid(alpha=0.3)
        if any_model:
            ax1.legend(loc="upper right", fontsize=8, ncol=2)

        # Panel 2: mix
        any_mix = False
        for choice, df in dfs_by_choice.items():
            c = f"mix_{metric}"
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            ax2.plot(s.index, s.values, label=str(choice))
            any_mix = True
        ax2.set_title("mix")
        ax2.grid(alpha=0.3)
        if any_mix:
            ax2.legend(loc="upper right", fontsize=8, ncol=2)

        fig.suptitle(title, y=0.98, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_axis_overlay_cumadv(
        dfs_by_choice: dict[str, pd.DataFrame],
        *,
        ref: str,
        metric: str,
        outpath: str,
        title: str,
    ) -> None:
        if not dfs_by_choice:
            return
        plt.figure(figsize=(12, 4))
        any_line = False
        for choice, df in dfs_by_choice.items():
            ref_col = f"{ref}_{metric}"
            roll_col = f"roll_{metric}"
            if ref_col not in df.columns or roll_col not in df.columns:
                continue
            ref_s = pd.to_numeric(df[ref_col], errors="coerce")
            roll_s = pd.to_numeric(df[roll_col], errors="coerce")
            diff = roll_s - ref_s
            plt.plot(diff.cumsum(), label=str(choice))
            any_line = True
        if not any_line:
            plt.close()
            return
        plt.axhline(0, color="black", linestyle="--", alpha=0.7)
        plt.title(title, fontsize=12, fontweight="bold")
        plt.grid(alpha=0.3)
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()

    if plots in {"per_run", "all"}:
        for axis_name, axis_spec in axes_cfg.items():
            axis_fig_dir = os.path.join(figs_dir, axis_name)
            dfs_by_choice: dict[str, pd.DataFrame] = {}
            for choice in axis_spec["choices"]:
                choice_str = str(choice).lower() if isinstance(choice, str) else str(choice)
                pbase = os.path.join(runs_dir, axis_name, choice_str)
                if not (os.path.exists(pbase + ".parquet") or os.path.exists(pbase + ".csv")):
                    continue
                try:
                    dfs_by_choice[choice_str] = _load_run_df(pbase)
                except Exception:
                    continue

            # Primary metric overlays (model+mix) and cum-adv overlays
            _plot_axis_overlay_timeseries(
                axis_name,
                dfs_by_choice,
                metric=primary_metric,
                outpath=os.path.join(axis_fig_dir, f"{primary_metric}_timeseries_overlay.png"),
                title=f"{axis_name}: {primary_metric} over time (choices overlaid)",
            )
            _plot_axis_overlay_cumadv(
                dfs_by_choice,
                ref="model",
                metric=primary_metric,
                outpath=os.path.join(axis_fig_dir, f"cumadv_model_{primary_metric}_overlay.png"),
                title=f"{axis_name}: cumulative advantage (model vs roll) on {primary_metric}",
            )
            _plot_axis_overlay_cumadv(
                dfs_by_choice,
                ref="mix",
                metric=primary_metric,
                outpath=os.path.join(axis_fig_dir, f"cumadv_mix_{primary_metric}_overlay.png"),
                title=f"{axis_name}: cumulative advantage (mix vs roll) on {primary_metric}",
            )

            # Covariance-only GMVP overlays
            if target_type != "volatility":
                for gm in ("gmvp_sharpe", "gmvp_var", "gmvp_vol", "turnover_l1"):
                    _plot_axis_overlay_timeseries(
                        axis_name,
                        dfs_by_choice,
                        metric=gm,
                        outpath=os.path.join(axis_fig_dir, f"{gm}_timeseries_overlay.png"),
                        title=f"{axis_name}: {gm} over time (choices overlaid)",
                    )

    return summary


def main():
    ap = argparse.ArgumentParser(description="Run ablation over pipeline design choices")
    ap.add_argument("--config", default="configs/ablation_covariance.yaml", help="Ablation config YAML")
    ap.add_argument("--outdir", default=None, help="Output directory for summary CSV")
    ap.add_argument("--verbose", action="store_true", help="Verbose backtest output")
    ap.add_argument(
        "--plots",
        choices=("none", "summary", "per_run", "all"),
        default="all",
        help="Plotting level: none (no plots), summary (bar chart), per_run (timeseries per axis/choice), all (both).",
    )
    args = ap.parse_args()
    run_ablation(args.config, out_dir=args.outdir, verbose=args.verbose, plots=args.plots)


if __name__ == "__main__":
    main()
