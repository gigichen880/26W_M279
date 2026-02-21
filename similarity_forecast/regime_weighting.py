# similarity_forecast/regime_weighting.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RegimeAwareWeights:
    """
    Stage 5 helper:
      w_i^(k) ∝ kappa_i * PI[i,k]
    where:
      - kappa_i is the similarity weight for neighbor i (already computed from distance kernel)
      - PI[i,k] is neighbor's soft regime membership

    Returns:
      W: [K, M] normalized over i for each k
    """
    eps: float = 1e-12

    def compute(
        self,
        kappa: NDArray[np.floating],      # [M]
        PI_neighbors: NDArray[np.floating] # [M, K]
    ) -> NDArray[np.floating]:
        if kappa.ndim != 1:
            raise ValueError("kappa must be 1D [M]")
        if PI_neighbors.ndim != 2:
            raise ValueError("PI_neighbors must be 2D [M, K]")
        M = kappa.shape[0]
        if PI_neighbors.shape[0] != M:
            raise ValueError("PI_neighbors first dim must match kappa length")

        # [M, K] <- kappa[:,None] * PI_neighbors
        WK = (kappa[:, None] * PI_neighbors).astype(float)
        # normalize over neighbors per regime
        denom = np.maximum(WK.sum(axis=0, keepdims=True), self.eps)  # [1, K]
        WK = WK / denom
        # return [K, M]
        return WK.T