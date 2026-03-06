"""
Run K regime ablation study.

Tests K=1,2,3,4,5,6 to validate regime contribution.
K=1 is pure similarity (no regime weighting) - the key baseline.

Usage: python scripts/analysis/run_k_ablation.py
       (Run from repo root; each K runs a full backtest ~5-10 min, ~30-60 min total)
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "regime_similarity.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "ablation_k"
RESULTS_DIR = REPO_ROOT / "results"


def run_ablation_for_k(
    k: int,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path] | None:
    """
    Run backtest with specified number of regimes K.

    Args:
        k: Number of regimes (1-6)
        config_path: Path to base config file
        output_dir: Directory to save ablation results
    """
    try:
        import yaml
    except ImportError:
        print("Install PyYAML: pip install pyyaml")
        return None

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

    source_parquet = RESULTS_DIR / "regime_similarity_backtest.parquet"
    source_csv = RESULTS_DIR / "regime_similarity_report.csv"

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
    print("\n" + "=" * 80)
    print("K REGIME ABLATION STUDY")
    print("=" * 80 + "\n")

    if not DEFAULT_CONFIG.exists():
        print(f"Config not found: {DEFAULT_CONFIG}")
        return

    K_values = [1, 2, 3, 4, 5, 6]
    results = {}

    for k in K_values:
        out = run_ablation_for_k(k, config_path=DEFAULT_CONFIG, output_dir=DEFAULT_OUTPUT_DIR)
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
