from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    Matrix logarithm for SPD matrices using eigen-decomposition (stable & fast).
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
    R: shape [L, N] returns for L times, N assets
    returns: shape [N, N] sample covariance
    """
    X = R - R.mean(axis=0, keepdims=True)
    denom = max(1, (X.shape[0] - ddof))
    return (X.T @ X) / denom


def corr_from_cov(Sigma: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    d = np.sqrt(np.maximum(np.diag(Sigma), eps))
    invd = 1.0 / d
    return (Sigma * invd[None, :]) * invd[:, None]
