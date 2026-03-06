"""
Verify that regime assignments were properly saved in backtest results.

Run after: python run_backtest.py --config configs/regime_similarity.yaml
Usage: python scripts/analysis/verify_regime_data.py [path/to/regime_similarity_backtest.parquet]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKTEST = REPO_ROOT / "results" / "regime_similarity_backtest.parquet"


def main() -> None:
    backtest_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKTEST
    backtest_path = Path(backtest_path)
    if not backtest_path.exists():
        print(f"File not found: {backtest_path}")
        sys.exit(1)

    if backtest_path.suffix == ".parquet":
        df = pd.read_parquet(backtest_path)
    else:
        df = pd.read_csv(backtest_path)

    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    print("=" * 80)
    print("REGIME DATA VERIFICATION")
    print("=" * 80)
    print()
    print("Available columns (first 25):", list(df.columns[:25]))
    print("... total columns:", len(df.columns))
    print()

    regime_cols = [c for c in df.columns if "regime" in c.lower()]
    print("Regime-related columns:", regime_cols)
    print()

    print("Total evaluation dates:", len(df))
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print()

    if "regime_assigned" not in df.columns:
        print("ERROR: 'regime_assigned' column not found!")
        print("Re-run backtest: python run_backtest.py --config configs/regime_similarity.yaml")
        sys.exit(1)

    print("Regime distribution:")
    regime_counts = df["regime_assigned"].value_counts().sort_index()
    for r, count in regime_counts.items():
        pct = count / len(df) * 100
        print(f"  Regime {int(r)}: {count} days ({pct:.1f}%)")
    print()

    n_missing = df["regime_assigned"].isna().sum()
    print("Missing regime assignments:", n_missing)
    print()

    prob_cols = [f"regime_prob_{k}" for k in range(4)]
    if all(c in df.columns for c in prob_cols):
        print("Sample regime probabilities (first 5 dates):")
        print(df[["date", "regime_assigned"] + prob_cols].head().to_string())
        print()
        prob_sums = df[prob_cols].sum(axis=1)
        print(f"Probability sums (should be ~1.0): min={prob_sums.min():.4f}, max={prob_sums.max():.4f}")
        print()

    print("Regime data looks good. Ready for visualization.")
    print("Run: python scripts/analysis/visualize_regimes.py")
    print()


if __name__ == "__main__":
    main()
