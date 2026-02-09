from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

from .utils import logm_spd, expm_sym, project_to_spd


class Weighting(Protocol):
    def weights(self, distances: NDArray[np.floating]) -> NDArray[np.floating]: ...


@dataclass(frozen=True)
class RBFWeighting:
    tau: float = 1.0
    eps: float = 1e-12

    def weights(self, distances: NDArray[np.floating]) -> NDArray[np.floating]:
        d2 = distances * distances
        w = np.exp(-d2 / max(self.tau, self.eps))
        s = w.sum()
        return w / max(s, self.eps)


@dataclass(frozen=True)
class InverseDistanceWeighting:
    eps: float = 1e-8

    def weights(self, distances: NDArray[np.floating]) -> NDArray[np.floating]:
        w = 1.0 / (distances + self.eps)
        return w / w.sum()


@dataclass(frozen=True)
class RankWeighting:
    alpha: float = 1.0  # w_i ∝ 1 / (rank^alpha)
    eps: float = 1e-12

    def weights(self, distances: NDArray[np.floating]) -> NDArray[np.floating]:
        # assumes distances already sorted ascending
        r = np.arange(1, distances.shape[0] + 1, dtype=float)
        w = 1.0 / np.power(r, self.alpha)
        return w / max(w.sum(), self.eps)


class Aggregator(Protocol):
    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]: ...


@dataclass(frozen=True)
class EuclideanMean:
    """
    Works for vectors or matrices (including SPD cov/corr).
    Weighted sum preserves SPD if inputs are SPD and weights >= 0.
    """
    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]:
        # targets: [k, ...]
        return np.tensordot(w, targets, axes=(0, 0))


@dataclass(frozen=True)
class LogEuclideanSPDMean:
    """
    SPD-aware mean: exp(sum_i w_i log(Sigma_i)).
    Uses eigen-based log/expm for symmetric matrices.
    """
    eps_spd: float = 1e-8

    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]:
        # targets: [k, N, N]
        S = np.zeros_like(targets[0])
        for i in range(targets.shape[0]):
            S += w[i] * logm_spd(project_to_spd(targets[i], eps=self.eps_spd))
        return project_to_spd(expm_sym(S), eps=self.eps_spd)
