"""Load backtest outputs from parquet or CSV with safe fallbacks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def read_backtest_table(path: str | Path) -> pd.DataFrame:
    """
    Read a backtest table. If the path is .parquet and reading fails (corrupt file,
    truncated write, etc.), fall back to sibling backtest.csv when present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)

    if suf == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            alt = path.with_suffix(".csv")
            if alt.exists():
                print(f"(warn) Failed to read Parquet {path}: {e}; using {alt}", file=sys.stderr)
                return pd.read_csv(alt)
            raise

    return pd.read_csv(path)
