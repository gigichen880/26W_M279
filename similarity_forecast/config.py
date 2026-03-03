# similarity_forecast/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SimilarityConfig:
    lookback: int = 60
    horizon: int = 20
    k: int = 50

    # two-stage rerank
    use_two_stage: bool = False
    stage2_metric: str = "fro"
    stage2_M: int = 200

    # weighting
    weighting: Literal["rbf", "inv", "rank"] = "rbf"
    tau: float = 1.0  # for rbf

    # NA handling
    max_window_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8
    cov_min_periods_ratio: float = 0.5
    filter_high_na_stocks: bool = True
    high_na_threshold: float = 0.3
