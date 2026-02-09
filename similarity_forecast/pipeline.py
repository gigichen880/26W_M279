from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .embeddings import WindowEmbedder
from .similarity import ExactKNN
from .aggregation import Weighting, Aggregator
from .target_objects import TargetObject


@dataclass
class SimilarityForecaster:
    """
    Max-flex design:
      - Similarity is computed from embeddings of RAW windows (feature engineering inside embedder)
      - Forecast target is computed independently from future windows
    """
    embedder: WindowEmbedder
    target_object: TargetObject
    weighting: Weighting
    aggregator: Aggregator

    lookback: int
    horizon: int

    embeds_: Optional[NDArray[np.floating]] = None   # [T0, D]
    targets_: Optional[NDArray[np.floating]] = None  # [T0, ...]
    knn_: Optional[ExactKNN] = None

    def _build_windows(self, R: NDArray[np.floating]) -> List[Tuple[slice, slice]]:
        T = R.shape[0]
        L, H = self.lookback, self.horizon
        pairs = []
        for i in range(L - 1, T - H - 1):
            past = slice(i - L + 1, i + 1)
            fut = slice(i + 1, i + H + 1)
            pairs.append((past, fut))
        return pairs

    def fit(self, returns_df: pd.DataFrame) -> "SimilarityForecaster":
        R = returns_df.to_numpy(dtype=float)  # [T, N]
        pairs = self._build_windows(R)
        if not pairs:
            raise ValueError("Not enough rows for lookback/horizon.")

        embeds = []
        targets = []

        for past_sl, fut_sl in pairs:
            past = R[past_sl]     # [L, N]
            fut = R[fut_sl]       # [H, N]

            e = self.embedder.embed(past)              # [D]
            y = self.target_object.target(fut)         # target object (vector or matrix)

            embeds.append(e)
            targets.append(y)

        self.embeds_ = np.stack(embeds, axis=0)
        self.targets_ = np.stack(targets, axis=0)
        self.knn_ = ExactKNN(self.embeds_)
        return self

    def _check_fitted(self) -> None:
        if self.embeds_ is None or self.targets_ is None or self.knn_ is None:
            raise RuntimeError("Call fit() first.")

    def predict_at_anchor(self, anchor_pos: int, k: int = 50, exclude_self: bool = True) -> NDArray[np.floating]:
        self._check_fitted()
        assert self.embeds_ is not None and self.targets_ is not None and self.knn_ is not None

        e = self.embeds_[anchor_pos]
        exclude = anchor_pos if exclude_self else None
        idx, dist = self.knn_.query(e=e, k=k, exclude_index=exclude)

        w = self.weighting.weights(dist)
        y_hat = self.aggregator.aggregate(self.targets_[idx], w)
        return self.target_object.postprocess(y_hat)
