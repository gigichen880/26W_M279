#!/usr/bin/env python3
"""
Run joint grid ablation (default: model.tau × backtest.k_neighbors) and Pareto GMVP report.

Usage (from repo root):
  python -m scripts.analysis.ablation.run_joint_gmvp_grid
  python -m scripts.analysis.ablation.run_joint_gmvp_grid --plots none --verbose
  python -m scripts.analysis.ablation.run_joint_gmvp_grid --pareto-only
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CFG = os.path.join(_REPO_ROOT, "configs", "ablation_joint_gmvp_grid.yaml")
DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "regime_covariance", "ablation_joint_gmvp")


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint grid (τ × k) + Pareto GMVP report")
    ap.add_argument("--config", default=DEFAULT_CFG, help="Grid ablation YAML")
    ap.add_argument("--outdir", default=None, help=f"Output dir (default: {DEFAULT_OUTDIR})")
    ap.add_argument(
        "--plots",
        choices=("none", "summary", "per_run", "all"),
        default="none",
        help="run_ablation plots (default: none — grid is many runs)",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--pareto-only",
        action="store_true",
        help="Only run pareto_gmvp_report on existing ablation_summary.csv",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        print(f"Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)

    outdir = os.path.abspath(args.outdir or DEFAULT_OUTDIR)
    os.makedirs(outdir, exist_ok=True)
    summary_csv = os.path.join(outdir, "ablation_summary.csv")

    if not args.pareto_only:
        with open(cfg_path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        spec = dict(spec)
        bc = spec.get("base_config", "regime_covariance.yaml")
        if isinstance(bc, str) and not os.path.isabs(bc):
            spec["base_config"] = os.path.join(_REPO_ROOT, "configs", os.path.basename(bc))
        spec["outputs"] = dict(spec.get("outputs") or {})
        spec["outputs"]["outdir"] = outdir
        spec_path = os.path.join(outdir, "ablation_spec.yaml")
        with open(spec_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        print(f"Wrote {spec_path}")

        from scripts.analysis.ablation.run_ablation import run_ablation

        run_ablation(spec_path, out_dir=outdir, verbose=args.verbose, plots=args.plots)

    if not os.path.isfile(summary_csv):
        print(f"Missing {summary_csv}", file=sys.stderr)
        sys.exit(1)

    from scripts.analysis.ablation.pareto_gmvp_report import run_report

    pareto_dir = os.path.join(outdir, "pareto")
    run_report(summary_csv, pareto_dir)
    print("Done.")


if __name__ == "__main__":
    main()
