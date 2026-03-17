"""
Run K regime ablation study (covariance or volatility).

Tests K=1,2,3,4,5,6. K=1 is pure similarity (no regime weighting).

Usage:
  python scripts/analysis/run_k_ablation.py
  python scripts/analysis/run_k_ablation.py --config configs/regime_volatility.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "regime_covariance.yaml"
RESULTS_DIR = REPO_ROOT / "results"


def run_ablation_for_k(
    k: int,
    config_path: Path,
    output_dir: Path,
    tag: str,
) -> tuple[Path, Path] | None:
    """
    Run backtest with specified number of regimes K.

    Args:
        k: Number of regimes (1-6)
        config_path: Path to base config file
        output_dir: Directory to save ablation results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING K={k} ABLATION")
    print(f"{'='*80}\n")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["model"]["n_regimes"] = k
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = output_dir / f"config_k{k}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"Running backtest with K={k} regimes...")
    print(f"Config saved to: {temp_config_path}")

    cmd = ["python", "run_backtest.py", "--config", str(temp_config_path)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR running K={k}:")
        print(result.stderr)
        return None

    print(f"✓ K={k} backtest complete")

    source_parquet = RESULTS_DIR / f"{tag}_backtest.parquet"
    source_csv = RESULTS_DIR / ("regime_volatility_report.csv" if tag == "regime_volatility" else "regime_covariance_report.csv")

    dest_parquet = output_dir / f"backtest_k{k}.parquet"
    dest_report = output_dir / f"report_k{k}.csv"

    if source_parquet.exists():
        shutil.copy(source_parquet, dest_parquet)
        print(f"✓ Saved: {dest_parquet}")
    if source_csv.exists():
        shutil.copy(source_csv, dest_report)
        print(f"✓ Saved: {dest_report}")

    return (dest_parquet, dest_report)


def main() -> None:
    ap = argparse.ArgumentParser(description="K regime ablation (cov or vol)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Base config (regime_covariance or regime_volatility)")
    args = ap.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tag = str(cfg.get("outputs", {}).get("tag", "regime_covariance"))
    output_dir = RESULTS_DIR / f"ablation_k_{tag}"

    print("\n" + "=" * 80)
    print("K REGIME ABLATION STUDY" + (f" ({tag})" if tag != "regime_covariance" else ""))
    print("=" * 80 + "\n")

    K_values = [1, 2, 3, 4, 5, 6]
    results = {}

    for k in K_values:
        out = run_ablation_for_k(k, config_path=config_path, output_dir=output_dir, tag=tag)
        if out:
            results[k] = out
        print()

    print("\n" + "=" * 80)
    print("K ABLATION COMPLETE")
    print("=" * 80 + "\n")
    print("Results saved to: results/ablation_k/")
    print(f"Completed K values: {list(results.keys())}")
    print("\nNext: Run analyze_k_ablation.py to compare results")
    print()


if __name__ == "__main__":
    main()
