#!/usr/bin/env python3
"""
Phased covariance ablation workflow (excludes K-regime / n_regimes sweep).

  Phase 0 — Joint τ × k only on the base config (similarity layer; cheap interaction probe).
            Config: configs/ablation_joint_gmvp_grid.yaml

  Phase 1 — All marginals in one YAML (one-at-a-time): embedder, pca_k, regime_clustering,
            knn_metric, k_neighbors, tau, transition, regime_aggregation, regime_weighting.
            Config: configs/ablation_phase1_covariance.yaml

  Phase 2 — Small Cartesian grid on a SHORTLIST (edit YAML after Phase 1): typically
            2 clusterers × 3 knn metrics × 2 embedders × τ × k; then Pareto on GMVP means.
            Config: configs/ablation_phase2_joint_shortlist.yaml

  Phase 3 — Coordinate descent (manual): fix winners from Phase 2 in regime_covariance.yaml,
            then re-run single-axis ablations or small grids; this script only prints steps.

Usage (repo root):
  python -m scripts.analysis.ablation.run_phased_covariance_ablation --phase 0
  python -m scripts.analysis.ablation.run_phased_covariance_ablation --phase 1
  python -m scripts.analysis.ablation.run_phased_covariance_ablation --phase 2 --plots none
  python -m scripts.analysis.ablation.run_phased_covariance_ablation --phase 3
  python -m scripts.analysis.ablation.run_phased_covariance_ablation --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CFG_PHASE0 = os.path.join(_REPO_ROOT, "configs", "ablation_joint_gmvp_grid.yaml")
CFG_PHASE1 = os.path.join(_REPO_ROOT, "configs", "ablation_phase1_covariance.yaml")
CFG_PHASE2 = os.path.join(_REPO_ROOT, "configs", "ablation_phase2_joint_shortlist.yaml")


def _resolve_spec_paths(cfg_path: str, outdir: str | None) -> tuple[str, str]:
    """Write resolved spec with absolute base_config and outputs.outdir; return (spec_path, outdir)."""
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    spec = dict(spec)
    bc = spec.get("base_config", "regime_covariance.yaml")
    if isinstance(bc, str) and not os.path.isabs(bc):
        spec["base_config"] = os.path.join(_REPO_ROOT, "configs", os.path.basename(bc))
    out = outdir or spec.get("outputs", {}).get("outdir")
    if not out:
        raise ValueError("outputs.outdir missing in ablation YAML")
    out = os.path.abspath(os.path.join(_REPO_ROOT, out) if not os.path.isabs(out) else out)
    spec["outputs"] = dict(spec.get("outputs") or {})
    spec["outputs"]["outdir"] = out
    os.makedirs(out, exist_ok=True)
    spec_path = os.path.join(out, "ablation_spec_resolved.yaml")
    with open(spec_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
    return spec_path, out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phased covariance ablation (no K-regime sweep)")
    ap.add_argument("--phase", type=int, choices=(0, 1, 2, 3), default=None, help="Which phase to run")
    ap.add_argument(
        "--plots",
        choices=("none", "summary", "per_run", "all"),
        default="summary",
        help="run_ablation plots (Phase 0/1/2); default summary includes bar charts + model-only OAT figures",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print planned actions and exit")
    args = ap.parse_args()

    if args.phase is None and not args.dry_run:
        ap.error("Pass --phase 0|1|2|3 or use --dry-run")

    if args.phase == 3:
        print(
            """
Phase 3 — coordinate descent (manual)
  1. Copy your Phase-2 Pareto pick (or single best) into configs/regime_covariance.yaml.
  2. Re-run ONE marginal axis at a time that you did not fully resolve jointly, e.g. trim
     configs/ablation_phase1_covariance.yaml to a single `axes:` entry, or add a tiny grid
     for one knob only.
  3. Alternate: fix (τ,k), sweep clustering only; fix clustering, sweep knn_metric only; etc.
  4. Full n_regimes (K) sweep stays separate: scripts.analysis.ablation.run_k_ablation
"""
        )
        return

    from scripts.analysis.ablation.run_ablation import run_ablation

    def _run_phase(cfg_file: str, *, pareto: bool = False) -> None:
        spec_path, out = _resolve_spec_paths(cfg_file, outdir=None)
        print(f"Spec: {spec_path}")
        print(f"Out:  {out}")
        if args.dry_run:
            return
        run_ablation(spec_path, out_dir=out, verbose=args.verbose, plots=args.plots)
        summary = os.path.join(out, "ablation_summary.csv")
        if pareto and os.path.isfile(summary):
            from scripts.analysis.ablation.pareto_gmvp_report import run_report

            pdir = os.path.join(out, "pareto")
            run_report(summary, pdir)

    if args.dry_run:
        print("Dry run — planned configs:")
        print(f"  Phase 0: {CFG_PHASE0}  (grid τ×k; pareto via run_joint_gmvp_grid or --phase 0)")
        print(f"  Phase 1: {CFG_PHASE1}  (one-at-a-time marginals)")
        print(f"  Phase 2: {CFG_PHASE2}  (shortlist grid + pareto)")
        print("  Phase 3: manual coordinate descent (see docstring)")
        return

    if args.phase == 0:
        _run_phase(CFG_PHASE0, pareto=True)
    elif args.phase == 1:
        _run_phase(CFG_PHASE1, pareto=False)
    elif args.phase == 2:
        _run_phase(CFG_PHASE2, pareto=True)


if __name__ == "__main__":
    main()
