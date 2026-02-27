# similarity_forecast/core.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple
import numpy as np
import pandas as pd
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

import warnings
from typing import Optional, Tuple
from numpy.typing import NDArray
from typing import Tuple

def cov_from_returns_filtered(
    R: NDArray[np.floating],
    ddof: int = 1,
    min_periods: Optional[int] = None,
    min_frac: float = 0.8,
    ridge: float = 1e-8,
) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Window-wise filter assets then compute covariance on the kept subset.
    Returns: (cov_small, mask)
    """
    T, N = R.shape
    if min_periods is None:
        min_periods = max(2, int(np.ceil(min_frac * T)))

    obs = np.sum(~np.isnan(R), axis=0)
    mask = obs >= min_periods
    if mask.sum() < 2:
        raise ValueError(f"Not enough assets with >= {min_periods} obs (kept {mask.sum()} / {N}).")

    X = R[:, mask].astype(float)
    good_rows = ~np.isnan(X).any(axis=1)
    X = X[good_rows]
    T_eff = X.shape[0]
    if T_eff <= ddof:
        raise ValueError(f"Not enough complete rows after filtering (T_eff={T_eff}, ddof={ddof}).")

    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(1, T_eff - ddof)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=float)

    cov = project_to_spd(cov)
    return cov.astype(float), mask

def cov_from_returns(
    R: NDArray[np.floating],
    ddof: int = 1,
    min_periods: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Fixed-shape covariance from returns.

    R: (T, N) with NaNs allowed.
    Returns: (N, N) covariance (SPD-projected). Always NxN.
    """
    T, N = R.shape
    if min_periods is None:
        min_periods = max(2, T // 2)

    if not np.isnan(R).any():
        X = R - R.mean(axis=0, keepdims=True)
        denom = max(1, T - ddof)
        cov = (X.T @ X) / denom
    else:
        # pairwise-complete covariance, always NxN
        df = pd.DataFrame(R)
        cov = df.cov(min_periods=min_periods, ddof=ddof).to_numpy()

        if np.isnan(cov).any():
            n_valid = int((~np.isnan(cov)).sum())
            warnings.warn(
                f"Covariance has NaNs. Only {n_valid} / {N*N} entries valid. "
                f"Filling remaining with 0 then SPD-projecting. "
                f"(Try increasing min_periods or filtering assets/windows.)",
                UserWarning,
                stacklevel=2,
            )

        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    cov = project_to_spd(cov)
    return cov.astype(float)

def corr_from_cov(Sigma: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    d = np.sqrt(np.maximum(np.diag(Sigma), eps))
    invd = 1.0 / d
    return (Sigma * invd[None, :]) * invd[:, None]


def validate_window(
    returns_window: NDArray[np.floating],
    max_na_pct: float = 0.3,
    min_stocks_pct: float = 0.8,
) -> bool:
    """
    Check if a returns window has sufficient data quality.

    Args:
        returns_window: (T, N) returns for this window
        max_na_pct: maximum allowed fraction of NAs (default 0.3)
        min_stocks_pct: minimum fraction of stocks that must have data (default 0.8)

    Returns:
        True if window is valid, False otherwise
    """
    T, N = returns_window.shape
    na_pct = np.isnan(returns_window).sum() / (T * N)
    if na_pct > max_na_pct:
        return False
    stocks_with_data = (~np.isnan(returns_window).all(axis=0)).sum()
    if stocks_with_data / N < min_stocks_pct:
        return False
    times_with_data = (~np.isnan(returns_window).all(axis=1)).sum()
    if times_with_data / T < 0.5:
        return False
    return True


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