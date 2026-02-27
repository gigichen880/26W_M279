import numpy as np
from numpy.typing import NDArray


def frobenius_error(S_hat: NDArray[np.floating], S_true: NDArray[np.floating]) -> float:
    """
    Frobenius norm error between forecast and realized covariance.

    Args:
        S_hat: (N, N) forecast covariance (SPD recommended)
        S_true: (N, N) realized covariance (SPD recommended)

    Returns:
        ||S_hat - S_true||_F  (scalar)
    """
    S_hat = np.asarray(S_hat, dtype=float)
    S_true = np.asarray(S_true, dtype=float)
    if S_hat.shape != S_true.shape:
        raise ValueError(f"Shape mismatch: S_hat {S_hat.shape} vs S_true {S_true.shape}")
    return float(np.linalg.norm(S_hat - S_true, ord="fro"))

import numpy as np

def gaussian_kl_divergence(S_true: np.ndarray, S_hat: np.ndarray, eps: float = 1e-10) -> float:
    """
    KL( N(0, S_true) || N(0, S_hat) ) = 0.5 * ( tr(S_hat^{-1} S_true) - n + logdet(S_hat) - logdet(S_true) )

    Uses slogdet (stable) and solve (stable). Assumes inputs are SPD; if near-singular, add eps*I.
    """
    S_true = np.asarray(S_true, dtype=float)
    S_hat = np.asarray(S_hat, dtype=float)
    n = S_true.shape[0]

    # Symmetrize
    S_true = (S_true + S_true.T) / 2.0
    S_hat = (S_hat + S_hat.T) / 2.0

    # Jitter for numerical stability
    I = np.eye(n)
    S_true_j = S_true + eps * I
    S_hat_j = S_hat + eps * I

    sign_t, logdet_t = np.linalg.slogdet(S_true_j)
    sign_h, logdet_h = np.linalg.slogdet(S_hat_j)
    if sign_t <= 0 or sign_h <= 0:
        raise ValueError("slogdet sign <= 0; matrices not SPD even after jitter.")

    # trace(S_hat^{-1} S_true)
    X = np.linalg.solve(S_hat_j, S_true_j)
    tr_term = float(np.trace(X))

    return 0.5 * (tr_term - n + (logdet_h - logdet_t))

def min_variance_weights(
    Sigma_hat: NDArray[np.floating],
    long_only: bool = False,
    ridge: float = 0.05,          # 5% ridge relative to avg variance
    eps_spd: float = 1e-12,
) -> NDArray[np.floating]:
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    N = Sigma_hat.shape[0]
    I = np.eye(N, dtype=float)

    # symmetrize
    S = (Sigma_hat + Sigma_hat.T) / 2.0

    # scale-aware ridge (huge difference vs eps=1e-10)
    avg_var = float(np.trace(S)) / N
    avg_var = max(avg_var, eps_spd)
    S = S + ridge * avg_var * I

    ones = np.ones(N, dtype=float)
    x = np.linalg.solve(S, ones)
    denom = float(ones @ x)
    if not np.isfinite(denom) or abs(denom) < 1e-20:
        raise ValueError("Degenerate min-var solution: 1^T Sigma^{-1} 1 is near zero.")

    w = x / denom

    if long_only:
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        w = (np.full(N, 1.0 / N) if s <= 0 else w / s)

    return w.astype(float)

def realized_portfolio_variance(
    w: NDArray[np.floating],
    Sigma_true: NDArray[np.floating],
) -> float:
    """
    Compute realized variance w^T Sigma_true w.

    Args:
        w: (N,) portfolio weights
        Sigma_true: (N, N) realized covariance

    Returns:
        realized variance (scalar)
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    Sigma_true = np.asarray(Sigma_true, dtype=float)
    if Sigma_true.shape != (w.shape[0], w.shape[0]):
        raise ValueError(f"Shape mismatch: w {w.shape} vs Sigma_true {Sigma_true.shape}")
    return float(w @ Sigma_true @ w)