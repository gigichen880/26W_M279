"""
Data quality reporting for returns DataFrames (NA analysis).
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def data_quality_report(returns_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate data quality report for returns DataFrame.

    Returns dict with:
    - overall_na_pct
    - na_by_stock (pd.Series)
    - na_by_date (pd.Series)
    - stocks_below_threshold (dict of stock -> na_pct for stocks with >20% NAs)
    """
    report: Dict[str, Any] = {}

    report["overall_na_pct"] = returns_df.isna().sum().sum() / (returns_df.size or 1)

    report["na_by_stock"] = returns_df.isna().sum(axis=0) / len(returns_df)

    report["na_by_date"] = returns_df.isna().sum(axis=1) / len(returns_df.columns)

    threshold = 0.2
    below = report["na_by_stock"][report["na_by_stock"] > threshold]
    report["stocks_below_threshold"] = (
        below.sort_values(ascending=False).to_dict()
        if isinstance(below, pd.Series) else {}
    )

    return report


def print_data_quality_report(returns_df: pd.DataFrame) -> None:
    """Print human-readable data quality summary."""
    report = data_quality_report(returns_df)

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Overall NA percentage: {report['overall_na_pct']:.2%}")
    print(f"\nStocks with >20% missing data: {len(report['stocks_below_threshold'])}")

    if report["stocks_below_threshold"]:
        print("\nProblem stocks:")
        for stock, na_pct in list(report["stocks_below_threshold"].items())[:10]:
            print(f"  {stock}: {na_pct:.1%} missing")

    na_by_stock = report["na_by_stock"]
    print(f"\nMean NA% by stock: {na_by_stock.mean():.2%}")
    print(f"Max NA% by stock: {na_by_stock.max():.2%}")
    print(f"Stocks with 0% NAs: {(na_by_stock == 0).sum()}")
    print("=" * 60)
