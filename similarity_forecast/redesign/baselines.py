"""Classical covariance baselines for redesign comparisons."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from similarity_forecast.core import cov_from_returns, project_to_spd, symmetrize


def rolling_cov(past: NDArray[np.floating], ddof: int = 1) -> NDArray[np.floating]:
    return cov_from_returns(past, ddof=ddof)


def persistence_cov(prev_realized: NDArray[np.floating]) -> NDArray[np.floating]:
    return project_to_spd(symmetrize(prev_realized), eps=1e-8)


def ewma_cov(past: NDArray[np.floating], lam: float = 0.94) -> NDArray[np.floating]:
    X = np.asarray(past, dtype=float)
    T, N = X.shape
    S = np.zeros((N, N), dtype=float)
    wsum = 0.0
    for t in range(T):
        w = (1.0 - lam) * (lam ** (T - 1 - t))
        r = X[t]
        r = np.where(np.isfinite(r), r, 0.0)
        S += w * np.outer(r, r)
        wsum += w
    S = S / max(wsum, 1e-12)
    return project_to_spd(symmetrize(S), eps=1e-8)


def ledoit_wolf_cov(past: NDArray[np.floating]) -> NDArray[np.floating]:
    from sklearn.covariance import LedoitWolf

    X = np.asarray(past, dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    return LedoitWolf().fit(X).covariance_


def oas_cov(past: NDArray[np.floating]) -> NDArray[np.floating]:
    from sklearn.covariance import OAS

    X = np.asarray(past, dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    return OAS().fit(X).covariance_


def diag_shrink_cov(past: NDArray[np.floating], gamma: float = 0.3) -> NDArray[np.floating]:
    S = rolling_cov(past)
    d = np.diag(np.diag(S))
    return project_to_spd((1.0 - gamma) * S + gamma * d, eps=1e-8)


BASELINE_FNS = {
    "rolling": rolling_cov,
    "ewma": ewma_cov,
    "ledoit_wolf": ledoit_wolf_cov,
    "oas": oas_cov,
    "shrink": diag_shrink_cov,
}
