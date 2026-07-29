"""Helpers for analysis scripts: resolve regime_name_map next to backtest outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from similarity_forecast.regime_labels import infer_n_regimes, load_or_build_regime_name_map


def regime_name_map_path_for_backtest(backtest_path: str | Path) -> Path:
    return Path(backtest_path).resolve().parent / "regime_name_map.json"


def get_regime_name_map_for_backtest(backtest_path: str | Path, df: pd.DataFrame | None = None) -> dict[int, str]:
    jp = regime_name_map_path_for_backtest(backtest_path)
    if df is None:
        df = pd.read_parquet(backtest_path) if str(backtest_path).endswith(".parquet") else pd.read_csv(backtest_path)
    try:
        return load_or_build_regime_name_map(df, json_path=jp)
    except Exception:
        K = infer_n_regimes(df)
        return {k: f"Regime {k}" for k in range(K)}
