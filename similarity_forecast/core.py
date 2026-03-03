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

def impute_returns_window(
    R: NDArray[np.floating],
    *,
    fill_all_nan: float = 0.0,
) -> NDArray[np.floating]:
    """
    Impute NaNs/non-finite in a returns window (T, N) using per-asset mean over time.
    Assets with all-NaN over time are filled with fill_all_nan (default 0.0).

    This is lookback-local (no lookahead) and produces a fully finite window.
    """
    X = np.asarray(R, dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)

    # columns with at least one finite obs
    col_has = np.isfinite(X).any(axis=0)  # (N,)

    col_mean = np.full(X.shape[1], fill_all_nan, dtype=float)
    if np.any(col_has):
        col_mean[col_has] = np.nanmean(X[:, col_has], axis=0)

    nan_i, nan_j = np.where(np.isnan(X))
    if nan_i.size > 0:
        X[nan_i, nan_j] = col_mean[nan_j]

    return np.nan_to_num(X, nan=fill_all_nan, posinf=fill_all_nan, neginf=fill_all_nan)


def cov_from_returns_imputed(
    R: NDArray[np.floating],
    *,
    ddof: int = 1,
    ridge: float = 1e-8,
) -> NDArray[np.floating]:
    """
    Fixed-shape covariance from returns using a single NA policy:
      - impute window (per-asset mean; all-NaN -> 0)
      - demean
      - sample covariance
      - add ridge
      - SPD project
    """
    X = impute_returns_window(R, fill_all_nan=0.0)
    T, N = X.shape
    denom = max(1, T - ddof)

    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / denom
    if ridge > 0:
        cov = cov + float(ridge) * np.eye(N, dtype=float)

    return project_to_spd(cov).astype(float)

def cov_from_returns(
    R: NDArray[np.floating],
    ddof: int = 1,
    min_periods: Optional[int] = None,  # kept for API compatibility; no longer used
) -> NDArray[np.floating]:
    """
    Fixed-shape covariance from returns (T, N) with NaNs allowed.

    CONSISTENT POLICY:
      - windows are expected to be validated upstream (validate_window)
      - remaining NaNs are imputed by per-asset mean (all-NaN assets -> 0)
      - covariance is computed on the imputed window, ridge-added, SPD-projected
    """
    return cov_from_returns_imputed(R, ddof=ddof, ridge=1e-8)

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
    mask_bad = ~np.isfinite(returns_window)
    na_pct = mask_bad.sum() / (T*N)
    if na_pct > max_na_pct:
        return False
    
    stocks_with_data = (~mask_bad).any(axis=0).sum()
    if stocks_with_data / N < min_stocks_pct:
        return False
    
    times_with_data  = (~mask_bad).any(axis=1).sum()
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
    

@dataclass(frozen=True)
class ArithmeticSPDMean:
    """
    Simple weighted arithmetic mean of SPD matrices, then project to SPD.
    """
    eps_spd: float = 1e-8

    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]:
        S = np.tensordot(w, targets, axes=(0, 0))
        return project_to_spd((S + S.T) / 2.0, eps=self.eps_spd)