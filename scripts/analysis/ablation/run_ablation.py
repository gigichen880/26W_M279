# scripts/analysis/ablation/run_ablation.py
"""
Systematic ablation over high-level design choices in the similarity_forecast pipeline.

Evaluates design choices either:
  - one axis at a time (ablation): isolate the effect of a single knob, OR
  - full grid (grid mode): search for the best combined set of decisions.

Records summary metrics (primary metric mean for model/mix and GMVP metrics for covariance).

Use to isolate or tune:
  - embedder (pca, corr_eig, vol_stats for vol)
  - transition_estimator (hard vs soft)
  - knn_metric (scalars or dict choices to set model.knn_metric + model.knn_lp_p together)
  - regime_aggregation (soft vs hard)
  - regime_weighting (filtered vs raw_pi)

Dict choice example (YAML):

  knn_metric:
    key: model.knn_metric
    choices:
      - l2
      - label: lp_p3
        model.knn_metric: lp
        model.knn_lp_p: 3

Usage:
  python -m scripts.analysis.ablation.run_ablation --config configs/ablation_covariance.yaml
  python -m scripts.analysis.ablation.run_ablation --config configs/ablation_volatility.yaml
"""

from __future__ import annotations

import os
import argparse
import copy
import itertools
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


def _ablation_choice_str(axis_spec: dict, choice: Any) -> str:
    """Tag for summary tables, run_tag, and output filenames."""
    if isinstance(choice, dict):
        label = choice.get("label")
        if label is not None:
            return str(label)
        updates = {k: v for k, v in choice.items() if k != "label"}
        return "_".join(
            f"{str(k).replace('.', '_')}_{v}" for k, v in sorted(updates.items())
        )
    return str(choice).lower() if isinstance(choice, str) else str(choice)


def _safe_filename_tag(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("|", "_")
    )


def _merge_ablation_choice(merged: dict, *, axis_spec: dict, choice: Any) -> tuple[dict, str]:
    """
    Merge one axis choice into config. Scalar choices use axis_spec['key'].
    Dict choices set each dotted path (excluding key 'label') and should use label: for short tags.
    """
    choice_str = _ablation_choice_str(axis_spec, choice)
    if isinstance(choice, dict):
        updates = {k: v for k, v in choice.items() if k != "label"}
        for dotted, val in updates.items():
            if not isinstance(dotted, str):
                raise ValueError(
                    f"Ablation dict choice keys must be dotted config paths (str), got {dotted!r}"
                )
            if "." not in dotted:
                raise ValueError(
                    f"Ablation dict choice key must be a dotted path (e.g. model.knn_metric), got {dotted!r}"
                )
            merged = deep_update(merged, _set_dotted({}, dotted, val))
    else:
        key_path = axis_spec.get("key")
        if not key_path:
            raise ValueError("Scalar ablation choice requires axis['key'] in the axis spec.")
        merged = deep_update(merged, _set_dotted({}, key_path, choice))
    return merged, choice_str


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
      Each choice may be a scalar (sets key) or a dict of dotted.path -> value, optional label: str.
    """
    cfg = load_yaml(cfg_path)
    base_path = cfg["base_config"]
    if not os.path.isabs(base_path):
        base_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(cfg_path)), base_path)
        )
    base_cfg = load_yaml(base_path)
    # Optional: apply config overrides (e.g., shorten date range for speed)
    overrides_cfg = cfg.get("overrides")
    if isinstance(overrides_cfg, dict) and overrides_cfg:
        base_cfg = deep_update(base_cfg, overrides_cfg)
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

    # Ranking objective (for grid mode)
    # Example: "model_gmvp_sharpe_mean" or "mix_gmvp_sharpe_mean"
    default_objective = "model_gmvp_sharpe_mean" if target_type != "volatility" else "model_vol_mse_mean"
    objective = str(cfg.get("objective", default_objective))

    rows: list[dict] = []
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    def _summarize_run(results_df: pd.DataFrame) -> dict:
        model_col = f"model_{primary_metric}"
        mix_col = f"mix_{primary_metric}"
        out = {
            "primary_metric": primary_metric,
            "model_mean": float(results_df[model_col].mean()) if model_col in results_df.columns else None,
            "mix_mean": float(results_df[mix_col].mean()) if mix_col in results_df.columns else None,
        }
        for m in extra_metrics:
            for pref in ("model", "mix"):
                c = f"{pref}_{m}"
                if c in results_df.columns:
                    out[c + "_mean"] = float(pd.to_numeric(results_df[c], errors="coerce").mean())
        for m in gmvp_metrics:
            for pref in ("model", "mix", "roll", "pers", "shrink"):
                c = f"{pref}_{m}"
                if c in results_df.columns:
                    out[c + "_mean"] = float(pd.to_numeric(results_df[c], errors="coerce").mean())
        return out

    if mode in {"grid", "full_grid"}:
        axis_items = list(axes_cfg.items())
        axis_names = [a for a, _ in axis_items]
        choices_lists = [list(spec["choices"]) for _, spec in axis_items]
        total = int(np.prod([len(c) for c in choices_lists])) if choices_lists else 0
        print(f"Running GRID over {len(axis_names)} axes ({total} combinations). Objective={objective}")

        for combo in itertools.product(*choices_lists):
            merged = copy.deepcopy(base_cfg)
            tags = []
            combo_dict = {}
            for (axis_name, axis_spec), choice in zip(axis_items, combo):
                merged, choice_str = _merge_ablation_choice(merged, axis_spec=axis_spec, choice=choice)
                combo_dict[axis_name] = choice_str
                tags.append(f"{axis_name}={choice_str}")
            run_tag = "|".join(tags)
            print(f"  Running grid: {run_tag} ...")
            try:
                results_df, _ = run_backtest_from_config(merged, verbose=verbose)
            except Exception as e:
                print(f"    Failed: {e}")
                row = {"mode": "grid", "run_tag": run_tag, "error": str(e)}
                row.update(combo_dict)
                rows.append(row)
                continue

            grid_dir = os.path.join(runs_dir, "__grid__")
            os.makedirs(grid_dir, exist_ok=True)
            safe_name = run_tag.replace("/", "_").replace(":", "_")
            run_base = os.path.join(grid_dir, safe_name)
            try:
                results_df.to_parquet(run_base + ".parquet")
            except Exception:
                results_df.to_csv(run_base + ".csv", index=True)

            row = {"mode": "grid", "run_tag": run_tag}
            row.update(combo_dict)
            row.update(_summarize_run(results_df))
            rows.append(row)
    else:
        # One-at-a-time ablation
        for axis_name, axis_spec in axes_cfg.items():
            choices = list(axis_spec["choices"])
            for choice in choices:
                merged, choice_str = _merge_ablation_choice(
                    copy.deepcopy(base_cfg), axis_spec=axis_spec, choice=choice
                )
                run_tag = f"{axis_name}={choice_str}"
                print(f"  Running ablation: {run_tag} ...")
                try:
                    results_df, _ = run_backtest_from_config(merged, verbose=verbose)
                except Exception as e:
                    print(f"    Failed: {e}")
                    rows.append({
                        "mode": "one_at_a_time",
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
                run_base = os.path.join(axis_dir, _safe_filename_tag(choice_str))
                try:
                    results_df.to_parquet(run_base + ".parquet")
                except Exception:
                    results_df.to_csv(run_base + ".csv", index=True)

                row = {
                    "mode": "one_at_a_time",
                    "axis": axis_name,
                    "choice": choice_str,
                    "run_tag": run_tag,
                }
                row.update(_summarize_run(results_df))
                rows.append(row)

    summary = pd.DataFrame(rows)

    # Rank grid runs by objective if present
    if mode in {"grid", "full_grid"} and (objective in summary.columns):
        obj = pd.to_numeric(summary[objective], errors="coerce")
        higher_is_better = "sharpe" in objective.lower() or "r2" in objective.lower() or "mean" in objective.lower() and "vol_mse" not in objective.lower()
        summary["_objective"] = obj
        summary = summary.sort_values("_objective", ascending=not higher_is_better, na_position="last")

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

    def _plot_axis_pairwise_bars(
        *,
        sub: pd.DataFrame,
        axis_name: str,
        outpath: str,
        target_type: str,
    ) -> None:
        """
        One figure per axis:
          - multiple subplots (key metrics)
          - grouped bars for model vs mix for each choice
          - for GMVP Sharpe, add horizontal lines for roll/shrink/pers baselines
        """
        if sub.empty:
            return
        sub = sub.copy()
        sub = sub.sort_values("choice")
        labels = sub["choice"].astype(str).tolist()
        x = np.arange(len(labels))
        w = 0.35

        if target_type == "volatility":
            panels = [
                ("model_mean", "mix_mean", f"Mean {primary_metric} (↓)"),
            ]
        else:
            panels = [
                ("model_gmvp_sharpe_mean", "mix_gmvp_sharpe_mean", "Mean GMVP Sharpe (↑)"),
                ("model_turnover_l1_mean", "mix_turnover_l1_mean", "Mean Turnover L1 (↓)"),
                ("model_mean", "mix_mean", f"Mean {primary_metric} (↓)"),
                ("model_stein_mean", "mix_stein_mean", "Mean Stein (↓)"),
            ]

        n = len(panels)
        fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharex=False)
        if n == 1:
            axs = [axs]

        for ax, (c_model, c_mix, title) in zip(axs, panels):
            if c_model not in sub.columns or c_mix not in sub.columns:
                ax.set_visible(False)
                continue
            y_model = pd.to_numeric(sub[c_model], errors="coerce").values
            y_mix = pd.to_numeric(sub[c_mix], errors="coerce").values
            ax.bar(x - w / 2, y_model, w, label="model", color="steelblue", alpha=0.9)
            ax.bar(x + w / 2, y_mix, w, label="mix", color="gray", alpha=0.7)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
            ax.grid(alpha=0.25, axis="y")
            # Baseline reference lines for GMVP Sharpe (constant across one-at-a-time runs)
            if c_model.endswith("gmvp_sharpe_mean"):
                for b, col, ls, colr in [
                    ("roll", "roll_gmvp_sharpe_mean", "--", "#8e8e8e"),
                    ("shrink", "shrink_gmvp_sharpe_mean", "--", "#6d6d6d"),
                    ("pers", "pers_gmvp_sharpe_mean", "--", "#3a3a3a"),
                ]:
                    if col in sub.columns:
                        yb = float(pd.to_numeric(sub[col], errors="coerce").iloc[0])
                        if np.isfinite(yb):
                            ax.axhline(yb, linestyle=ls, linewidth=1.0, color=colr, alpha=0.9, label=b)

        handles, labels_leg = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"Ablation axis: {axis_name} (grouped bars; model vs mix)", y=1.10, fontsize=12, fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=220, bbox_inches="tight")
        plt.close(fig)

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

        # Plot: one figure per axis with multiple key metrics (pairwise decision comparisons)
        try:
            if "axis" in summary.columns and "choice" in summary.columns:
                for axis_name in sorted([a for a in summary["axis"].dropna().unique().tolist()]):
                    sub = summary[(summary.get("mode") == "one_at_a_time") & (summary["axis"] == axis_name)].copy()
                    if sub.empty:
                        continue
                    outpath = os.path.join(figs_dir, f"axis_pairwise_bars_{axis_name}.png")
                    _plot_axis_pairwise_bars(sub=sub, axis_name=axis_name, outpath=outpath, target_type=target_type)
        except Exception as e:
            print(f"(warn) Failed to plot axis pairwise bars: {e}")

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
                choice_str = _safe_filename_tag(_ablation_choice_str(axis_spec, choice))
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
