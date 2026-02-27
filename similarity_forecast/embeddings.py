from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np
from numpy.typing import NDArray

from .core import cov_from_returns, corr_from_cov


class WindowEmbedder(Protocol):
    """
    Maps a raw lookback window of returns to a fixed-D embedding vector.
    past_returns: shape [L, N]
    returns: shape [D]
    """
    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]: ...
    @property
    def dim(self) -> int: ...


@dataclass(frozen=True)
class CorrEigenEmbedder:
    """
    Compute correlation on the lookback window, then embed by top-k log eigenvalues.
    Handles NAs via pairwise-complete covariance; falls back to complete-case or zero if needed.
    """
    k: int
    ddof: int = 1
    eps: float = 1e-12
    min_periods_ratio: float = 0.5

    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        T, N = past_returns.shape
        min_periods = max(2, int(T * self.min_periods_ratio))
        try:
            Sigma = cov_from_returns(past_returns, ddof=self.ddof, min_frac=0.8)
            C = corr_from_cov(Sigma, eps=self.eps)
            if np.isnan(C).any():
                complete_mask = ~np.isnan(past_returns).all(axis=0)
                n_complete = complete_mask.sum()
                if n_complete < 10:
                    warnings.warn(
                        f"CorrEigenEmbedder: only {n_complete} stocks with data, using zero embedding",
                        UserWarning,
                        stacklevel=2,
                    )
                    return np.zeros(self.k, dtype=float)
                X_complete = past_returns[:, complete_mask]
                Sigma = cov_from_returns(X_complete, ddof=self.ddof, min_periods=min_periods)
                C = corr_from_cov(Sigma, eps=self.eps)
            w = np.linalg.eigvalsh(C)
            w = np.maximum(w, self.eps)[::-1][: self.k]
            return np.log(w)
        except Exception as e:
            warnings.warn(f"Embedding failed: {e}, using zeros", UserWarning, stacklevel=2)
            return np.zeros(self.k, dtype=float)

    @property
    def dim(self) -> int:
        return self.k


@dataclass(frozen=True)
class VolStatsEmbedder:
    """
    Simple feature engineering directly from window returns:
      - log vol quantiles
      - mean vol
    Output is fixed-d, independent of N.
    """
    ddof: int = 1
    eps: float = 1e-12
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        # vol per asset over window (NA-safe: nanstd/nanmean)
        v = np.nanstd(past_returns, axis=0, ddof=self.ddof)
        v = np.where(np.isnan(v), 0.0, v)
        v = np.sqrt(np.maximum(v * v, self.eps))
        lv = np.log(np.maximum(v, self.eps))

        feats = [float(np.nanmean(lv))]
        feats.extend(np.nanquantile(lv, self.quantiles).tolist())
        return np.array(feats, dtype=float)

    @property
    def dim(self) -> int:
        return 1 + len(self.quantiles)
