#!/usr/bin/env python3
"""
Run walk-forward ablation over each implemented `model.regime_clustering` method.

Writes ablation_spec.yaml (generated from code), runs run_ablation, then a model-only figure.

Usage (from repo root — directory that contains the `scripts/` folder):
  python -m scripts.analysis.ablation.run_regime_clustering_ablation
  python -m scripts.analysis.ablation.run_regime_clustering_ablation --plots all --verbose
  python -m scripts.analysis.ablation.run_regime_clustering_ablation --plot-only
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from similarity_forecast.regime_clustering import implemented_regime_clustering_names

DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "regime_covariance", "ablation_regime_clustering")


def _default_params_for_clusterer(name: str) -> dict:
    if name == "spectral_knn":
        return {"n_neighbors": 10}
    if name == "signed_knn_spectral":
        return {"n_neighbors": 15}
    return {}


def build_regime_clustering_ablation_spec(
    *,
    repo_root: str = _REPO_ROOT,
    outdir: str = DEFAULT_OUTDIR,
) -> dict:
    choices: list[dict] = []
    for name in implemented_regime_clustering_names():
        choices.append(
            {
                "label": name,
                "model.regime_clustering": {
                    "name": name,
                    "params": _default_params_for_clusterer(name),
                },
            }
        )

    base_cfg = os.path.join(repo_root, "configs", "regime_covariance.yaml")

    return {
        "base_config": base_cfg,
        "mode": "one_at_a_time",
        "overrides": {
            "data": {
                "start_date": "2015-01-01",
                "end_date": "2021-12-31",
            },
            "backtest": {
                "stride": 5,
                "refit_mode": "days",
                "refit_every_days": 20,
            },
        },
        "axes": {
            "regime_clustering": {
                "key": "model.regime_clustering",
                "choices": choices,
            }
        },
        "outputs": {
            "outdir": outdir,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation sweep over regime clustering methods")
    ap.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Directory for ablation_summary.csv, runs/, ablation_spec.yaml",
    )
    ap.add_argument(
        "--plots",
        choices=("none", "summary", "per_run", "all"),
        default="all",
        help="Plots from run_ablation.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Only build model-only PNG from existing ablation_summary.csv",
    )
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    spec_path = os.path.join(outdir, "ablation_spec.yaml")

    if not args.plot_only:
        spec = build_regime_clustering_ablation_spec(repo_root=_REPO_ROOT, outdir=outdir)
        with open(spec_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        print(f"Wrote {spec_path}")

        from scripts.analysis.ablation.run_ablation import run_ablation

        run_ablation(spec_path, out_dir=outdir, verbose=args.verbose, plots=args.plots)

    summary_csv = os.path.join(outdir, "ablation_summary.csv")
    if not os.path.isfile(summary_csv):
        print(f"Missing {summary_csv}; run without --plot-only first.", file=sys.stderr)
        sys.exit(1)

    from scripts.analysis.ablation.plot_regime_clustering_ablation import plot_regime_clustering_summary

    fig_dir = os.path.join(outdir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, "regime_clustering_model_only.png")
    try:
        plot_regime_clustering_summary(summary_csv, out_png)
        print(f"Saved {out_png}")
    except ValueError as e:
        print(f"(warn) Could not plot regime clustering summary: {e}", file=sys.stderr)

    print("Done.")
    print(f"  Summary: {summary_csv}")


if __name__ == "__main__":
    main()
