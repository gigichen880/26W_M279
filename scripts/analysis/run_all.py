"""
Single entrypoint to generate core figures/tables (covariance or volatility).

This script only runs visualization/analysis scripts against an existing backtest.
It does NOT run `run_backtest.py`.

Usage:
  # Covariance (default)
  python -m scripts.analysis.run_all --target covariance

  # Volatility
  python -m scripts.analysis.run_all --target volatility
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.analysis.utils.paths import RESULTS_DIR, resolve_backtest_path
from scripts.analysis.utils.migrate_results import ensure_canonical_results

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _paths_for_target(target: str) -> dict[str, str]:
    if target == "volatility":
        return {
            "config": "configs/regime_volatility.yaml",
            "viz_config": "configs/viz_regime_volatility.yaml",
            "backtest": str(resolve_backtest_path("regime_volatility")),
        }
    return {
        "config": "configs/regime_covariance.yaml",
        "viz_config": "configs/viz_regime_covariance.yaml",
        "backtest": str(resolve_backtest_path("regime_covariance")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate figures/tables from existing backtest.")
    ap.add_argument("--target", choices=("covariance", "volatility"), default="covariance")
    ap.add_argument("--skip-regime", action="store_true", help="Skip regime visualizations/characterization.")
    ap.add_argument("--skip-stats", action="store_true", help="Skip statistical comparison.")
    ap.add_argument("--skip-viz", action="store_true", help="Skip time-series/backtest visualizations.")
    ap.add_argument("--skip-ablation-figs", action="store_true", help="Skip ablation-derived figures (requires prior ablation runs).")
    args = ap.parse_args()

    # If outputs are still in legacy layout, copy them into results/<tag>/...
    tag = "regime_volatility" if args.target == "volatility" else "regime_covariance"
    ensure_canonical_results(tag, target=args.target)

    paths = _paths_for_target(args.target)

    if not args.skip_viz:
        _run(["python", "-m", "scripts.analysis.core.visualize_backtest_results", "--config", paths["viz_config"]])

    if not args.skip_stats:
        _run(
            [
                "python",
                "-m",
                "scripts.analysis.core.visualize_statistical_comparison",
                "--input",
                paths["backtest"],
                "--target",
                args.target,
            ]
        )

    if not args.skip_regime:
        _run(["python", "-m", "scripts.analysis.regime.visualize_regimes", "--target", args.target])
        _run(["python", "-m", "scripts.analysis.regime.regime_characterization", "--target", args.target])
        _run(["python", "-m", "scripts.analysis.regime.visualize_transition_matrix", "--target", args.target])
        _run(["python", "-m", "scripts.analysis.regime.performance_by_regime", "--target", args.target])

    # Ablation-derived figures that live in the same figs folder (covariance only)
    if not args.skip_ablation_figs and args.target == "covariance":
        # K-ablation figure/table from existing ablation outputs
        ablation_dir = RESULTS_DIR / "ablation_k"
        if any((ablation_dir / f"report_k{k}.csv").exists() for k in (1, 2, 3, 4, 5, 6)):
            _run(["python", "-m", "scripts.analysis.ablation.analyze_k_ablation"])
        else:
            print(f"\n(skip) K-ablation analysis: missing reports in {ablation_dir}")

        # Crisis-vs-normal figure requires backtest_k*.parquet (optional)
        if (ablation_dir / "backtest_k1.parquet").exists() and (ablation_dir / "backtest_k4.parquet").exists():
            _run(["python", "-m", "scripts.analysis.misc.lag_analysis"])
        else:
            print(f"\n(skip) crisis-vs-normal figure: missing backtest_k*.parquet in {ablation_dir}")

    print("\n✓ Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)

