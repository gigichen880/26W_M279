#!/usr/bin/env python3
"""
Run walk-forward ablation over embedder.pca_k (PCA embedding dimension D).

Uses configs/ablation_pca_k.yaml (merged with explicit outdir), then writes a model-only 4-panel figure.

Usage (from repo root):
  python -m scripts.analysis.ablation.run_pca_k_ablation
  python -m scripts.analysis.ablation.run_pca_k_ablation --plots all --verbose
  python -m scripts.analysis.ablation.run_pca_k_ablation --plot-only
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CFG = os.path.join(_REPO_ROOT, "configs", "ablation_pca_k.yaml")
DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "regime_covariance", "ablation_pca_k")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation sweep over embedder.pca_k")
    ap.add_argument(
        "--config",
        default=DEFAULT_CFG,
        help="Ablation YAML (default: configs/ablation_pca_k.yaml)",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
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

    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        print(f"Missing config {cfg_path}", file=sys.stderr)
        sys.exit(1)

    outdir = os.path.abspath(args.outdir or DEFAULT_OUTDIR)
    os.makedirs(outdir, exist_ok=True)
    spec_path = os.path.join(outdir, "ablation_spec.yaml")

    if not args.plot_only:
        with open(cfg_path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        spec = dict(spec)
        # run_ablation resolves base_config relative to this YAML's directory; spec lives under results/.
        bc = spec.get("base_config", "regime_covariance.yaml")
        if isinstance(bc, str) and not os.path.isabs(bc):
            spec["base_config"] = os.path.join(_REPO_ROOT, "configs", os.path.basename(bc))
        spec["outputs"] = dict(spec.get("outputs") or {})
        spec["outputs"]["outdir"] = outdir
        with open(spec_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        print(f"Wrote {spec_path}")

        from scripts.analysis.ablation.run_ablation import run_ablation

        run_ablation(spec_path, out_dir=outdir, verbose=args.verbose, plots=args.plots)

    summary_csv = os.path.join(outdir, "ablation_summary.csv")
    if not os.path.isfile(summary_csv):
        print(f"Missing {summary_csv}; run without --plot-only first.", file=sys.stderr)
        sys.exit(1)

    from scripts.analysis.ablation.plot_pca_k_ablation import plot_pca_k_summary

    fig_dir = os.path.join(outdir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, "pca_k_ablation_model_only.png")
    try:
        plot_pca_k_summary(summary_csv, out_png)
        print(f"Saved {out_png}")
    except ValueError as e:
        print(f"(warn) Could not plot pca_k summary: {e}", file=sys.stderr)

    print("Done.")
    print(f"  Summary: {summary_csv}")

    sub = pd.read_csv(summary_csv)
    sub = sub[(sub["axis"] == "pca_k") & (sub["mode"] == "one_at_a_time")].copy()
    if not sub.empty and "model_mean" in sub.columns:
        sub["_k"] = pd.to_numeric(
            sub["choice"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        sub = sub.sort_values("_k")
        mm = pd.to_numeric(sub["model_mean"], errors="coerce")
        if mm.notna().any():
            i = mm.idxmin()
            best_row = sub.loc[i]
            print(
                f"  Best model_mean Fro (ablation primary): pca_k={best_row['choice']} "
                f"(model_mean={float(best_row['model_mean']):.6g})"
            )


if __name__ == "__main__":
    main()
