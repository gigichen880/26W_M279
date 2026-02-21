# similarity_forecast/core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


# =========================
# Linear algebra / SPD utils
# =========================

def symmetrize(A: NDArray[np.floating]) -> NDArray[np.floating]:
    return 0.5 * (A + A.T)


def project_to_spd(A: NDArray[np.floating], eps: float = 1e-8) -> NDArray[np.floating]:
    """
    Projects a symmetric matrix to SPD by eigenvalue clipping.
    """
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def logm_spd(A: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    """
    Matrix logarithm for SPD matrices using eigen-decomposition.
    """
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T


def expm_sym(A: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Matrix exponential for symmetric matrices using eigen-decomposition.
    """
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T


def cov_from_returns(R: NDArray[np.floating], ddof: int = 1) -> NDArray[np.floating]:
    """
    R: [L, N] returns
    returns: [N, N] sample covariance
    """
    X = R - R.mean(axis=0, keepdims=True)
    denom = max(1, (X.shape[0] - ddof))
    return (X.T @ X) / denom


def corr_from_cov(Sigma: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    d = np.sqrt(np.maximum(np.diag(Sigma), eps))
    invd = 1.0 / d
    return (Sigma * invd[None, :]) * invd[:, None]


# =========================
# Neighbor search (Exact KNN)
# =========================

@dataclass
class ExactKNN:
    """
    Exact KNN in embedding space using L2 distance.
    Stores E: [T, d]
    """
    E: NDArray[np.floating]

    def query(
        self,
        e: NDArray[np.floating],
        k: int,
        exclude_index: Optional[int] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.floating]]:
        diff = self.E - e[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)

        if exclude_index is not None and 0 <= exclude_index < d2.shape[0]:
            d2[exclude_index] = np.inf

        k = min(k, d2.shape[0])
        idx = np.argpartition(d2, kth=k - 1)[:k]
        order = np.argsort(d2[idx])
        idx = idx[order]
        dist = np.sqrt(d2[idx])
        return idx.astype(np.int64), dist.astype(float)


# =========================
# Weighting schemes
# =========================

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
    alpha: float = 1.0
    eps: float = 1e-12

    def weights(self, distances: NDArray[np.floating]) -> NDArray[np.floating]:
        # assumes distances already sorted ascending
        r = np.arange(1, distances.shape[0] + 1, dtype=float)
        w = 1.0 / np.power(r, self.alpha)
        return w / max(w.sum(), self.eps)


# =========================
# Aggregators
# =========================

class Aggregator(Protocol):
    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]: ...


@dataclass(frozen=True)
class EuclideanMean:
    """
    Works for vectors or matrices (including SPD cov/corr).
    Weighted sum preserves SPD if inputs are SPD and weights >= 0.
    """
    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.tensordot(w, targets, axes=(0, 0))


@dataclass(frozen=True)
class LogEuclideanSPDMean:
    """
    SPD-aware mean: exp(sum_i w_i log(Sigma_i)).
    """
    eps_spd: float = 1e-8

    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]:
        S = np.zeros_like(targets[0])
        for i in range(targets.shape[0]):
            S += w[i] * logm_spd(project_to_spd(targets[i], eps=self.eps_spd))
        return project_to_spd(expm_sym(S), eps=self.eps_spd)