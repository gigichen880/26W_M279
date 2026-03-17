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
from typing import Any

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


def run_ablation(cfg_path: str, out_dir: str | None = None, verbose: bool = False) -> pd.DataFrame:
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
    out_dir = out_dir or cfg.get("outputs", {}).get("outdir", "results/ablation")
    os.makedirs(out_dir, exist_ok=True)

    target_type = str(_get_dotted(base_cfg, "model.target", "covariance")).lower()
    if target_type == "volatility":
        primary_metric = "vol_mse"
        extra_metrics = ["vol_mae", "vol_rmse"]
    else:
        primary_metric = "fro"
        extra_metrics = ["stein", "logeuc"]

    rows: list[dict] = []

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
            rows.append(row)

    summary = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    with open(os.path.join(out_dir, "ablation_config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run ablation over pipeline design choices")
    ap.add_argument("--config", default="configs/ablation_covariance.yaml", help="Ablation config YAML")
    ap.add_argument("--outdir", default=None, help="Output directory for summary CSV")
    ap.add_argument("--verbose", action="store_true", help="Verbose backtest output")
    args = ap.parse_args()
    run_ablation(args.config, out_dir=args.outdir, verbose=args.verbose)


if __name__ == "__main__":
    main()
