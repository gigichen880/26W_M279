# data/build_pvclcl_matrix.py
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


DATE_RE = re.compile(r"(\d{8})\.csv$", re.IGNORECASE)

def iter_csv_files(data_dir: Path) -> List[Path]:
    """
    Return all CSV files under data_dir/<year>/*.csv, sorted by date.
    """
    files = []
    for year_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        files.extend(sorted(year_dir.glob("*.csv")))
    # sort by parsed date if possible
    def key(p: Path) -> Tuple[int, str]:
        m = DATE_RE.search(p.name)
        return (int(m.group(1)) if m else 99999999, str(p))
    return sorted(files, key=key)

def parse_date_from_filename(p: Path) -> pd.Timestamp:
    m = DATE_RE.search(p.name)
    if not m:
        raise ValueError(f"Filename does not match YYYYMMDD.csv: {p}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")

def read_ticker_pvclcl(csv_path: Path) -> pd.Series:
    """
    Read a single daily CSV and return a Series:
      index = ticker
      values = pvCLCL
      name = date (set by caller)
    Handles:
      - extra unnamed first column
      - duplicate tickers within a file (keeps last non-null)
    """
    df = pd.read_csv(
        csv_path,
        usecols=["ticker", "pvCLCL"], 
        dtype={"ticker": "string"},
    )

    # Clean ticker
    df["ticker"] = df["ticker"].astype("string").str.strip()

    # pvCLCL to numeric (coerce bad values to NaN)
    df["pvCLCL"] = pd.to_numeric(df["pvCLCL"], errors="coerce")

    # If duplicates exist per day, keep the last non-null
    df = df.dropna(subset=["ticker"])
    s = df.groupby("ticker", sort=False)["pvCLCL"].last()

    return s

def build_matrix(
    data_dir: Path,
    years: Optional[List[int]] = None,
    limit_files: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the grand ticker x date matrix of pvCLCL.
    """
    all_files = iter_csv_files(data_dir)

    if years is not None:
        year_set = set(years)
        all_files = [p for p in all_files if p.parent.name.isdigit() and int(p.parent.name) in year_set]

    if limit_files is not None:
        all_files = all_files[:limit_files]

    series_list: List[pd.Series] = []
    dates: List[pd.Timestamp] = []

    total = len(all_files)
    if verbose:
        print(f"Found {total} CSV files under {data_dir}")

    for i, fp in enumerate(all_files, 1):
        dt = parse_date_from_filename(fp)
        try:
            s = read_ticker_pvclcl(fp)
        except Exception as e:
            raise RuntimeError(f"Failed reading {fp}: {e}") from e

        s.name = dt
        series_list.append(s)
        dates.append(dt)

        if verbose and (i % 250 == 0 or i == total):
            print(f"Processed {i}/{total}: {fp.name} ({len(s):,} tickers)")

    if not series_list:
        raise ValueError("No CSV files found to process.")

    # Outer-join on ticker index; columns are dates
    mat = pd.concat(series_list, axis=1, join="outer")

    # Ensure sorted columns by date
    mat = mat.reindex(sorted(mat.columns), axis=1)

    # Optional: sort tickers for stable output
    mat = mat.sort_index()

    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data", help="Root data directory (contains 2000/, ..., 2021/)")
    ap.add_argument("--out-parquet", type=str, default="pvCLCL_matrix.parquet", help="Output parquet path")
    ap.add_argument("--out-csv", type=str, default="", help="Optional output CSV path (empty to skip)")
    ap.add_argument("--years", type=str, default="", help="Optional comma-separated years, e.g. 2010,2011,2012")
    ap.add_argument("--limit-files", type=int, default=0, help="Debug: limit number of daily files processed")
    ap.add_argument("--no-verbose", action="store_true", help="Disable progress prints")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {data_dir}")

    years = None
    if args.years.strip():
        years = [int(x) for x in args.years.split(",") if x.strip()]

    limit_files = args.limit_files if args.limit_files > 0 else None
    verbose = not args.no_verbose

    mat = build_matrix(
        data_dir=data_dir,
        years=years,
        limit_files=limit_files,
        verbose=verbose,
    )

    # Save parquet
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    mat.to_parquet(out_parquet)

    if verbose:
        print(f"Saved parquet: {out_parquet}  shape={mat.shape}  missing={int(mat.isna().sum().sum()):,}")

    # Optional CSV
    if args.out_csv.strip():
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        mat.to_csv(out_csv)
        if verbose:
            print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
