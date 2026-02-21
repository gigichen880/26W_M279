from __future__ import annotations

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
    """
    k: int
    ddof: int = 1
    eps: float = 1e-12

    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        Sigma = cov_from_returns(past_returns, ddof=self.ddof)     # [N, N]
        C = corr_from_cov(Sigma, eps=self.eps)                     # [N, N]
        w = np.linalg.eigvalsh(C)                                  # ascending
        w = np.maximum(w, self.eps)[::-1][: self.k]                # top-k
        return np.log(w)

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
        # vol per asset over window
        v = np.nanstd(past_returns, axis=0, ddof=self.ddof)
        v = np.sqrt(np.maximum(v * v, self.eps))  # just to be safe
        lv = np.log(np.maximum(v, self.eps))

        feats = [np.nanmean(lv)]
        feats.extend(np.nanquantile(lv, self.quantiles).tolist())
        return np.array(feats, dtype=float)

    @property
    def dim(self) -> int:
        return 1 + len(self.quantiles)
