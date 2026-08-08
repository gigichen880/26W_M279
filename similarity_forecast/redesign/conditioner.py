"""Common covariance conditioning operator C_η.

Applied identically to forecasts and realized targets for SPD metrics.
Portfolio construction uses C_η; realized-risk quadratic forms use raw Σ^real.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from similarity_forecast.core import symmetrize


@dataclass(frozen=True)
class ConditioningConfig:
    eta: float = 0.01
    """Relative eigen floor: ε_t = η * tr(Σ) / N."""
    diag_shrink: float = 0.0
    """Optional common diagonal shrink after floor: (1-γ)Σ + γ diag(Σ)."""
    abs_floor: float = 0.0
    """Optional absolute floor added after relative floor (usually 0)."""


def relative_eigen_floor(
    Sigma: NDArray[np.floating],
    *,
    eta: float = 0.01,
    abs_floor: float = 0.0,
) -> NDArray[np.floating]:
    S = symmetrize(np.asarray(Sigma, dtype=float))
    n = S.shape[0]
    tr = float(np.trace(S))
    eps = float(eta) * tr / max(n, 1) + float(abs_floor)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def apply_conditioner(
    Sigma: NDArray[np.floating],
    cfg: ConditioningConfig,
) -> NDArray[np.floating]:
    out = relative_eigen_floor(Sigma, eta=cfg.eta, abs_floor=cfg.abs_floor)
    g = float(cfg.diag_shrink)
    if g > 0.0:
        d = np.diag(np.diag(out))
        out = (1.0 - g) * out + g * d
        out = relative_eigen_floor(out, eta=0.0, abs_floor=1e-12)
    return out
