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


def gaussian_kl_divergence(
    S_true: NDArray[np.floating],
    S_hat: NDArray[np.floating],
    eps: float = 1e-10,
) -> float:
    """
    Gaussian KL divergence D_KL(N(0, S_true) || N(0, S_hat)).

    Formula:
        0.5 * ( tr(S_hat^{-1} S_true) - log det(S_hat^{-1} S_true) - N )

    Args:
        S_true: (N, N) realized covariance (SPD)
        S_hat:  (N, N) forecast covariance (SPD)
        eps: small jitter added to diagonals for numerical stability

    Returns:
        KL divergence (scalar, >= 0 up to numerical error)
    """
    S_true = np.asarray(S_true, dtype=float)
    S_hat = np.asarray(S_hat, dtype=float)
    if S_true.shape != S_hat.shape:
        raise ValueError(f"Shape mismatch: S_true {S_true.shape} vs S_hat {S_hat.shape}")
    if S_true.ndim != 2 or S_true.shape[0] != S_true.shape[1]:
        raise ValueError(f"Expected square matrices, got {S_true.shape}")

    N = S_true.shape[0]
    I = np.eye(N, dtype=float)

    # Stabilize (helps if near-singular due to numerical issues)
    S_true_j = S_true + eps * I
    S_hat_j = S_hat + eps * I

    # Use solve instead of explicit inverse
    # M = S_hat^{-1} S_true
    M = np.linalg.solve(S_hat_j, S_true_j)

    tr_term = float(np.trace(M))

    # logdet(M) via slogdet for stability
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        # If numerical issues cause non-positive determinant, jitter more
        M = np.linalg.solve(S_hat_j + 10 * eps * I, S_true_j + 10 * eps * I)
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            raise ValueError("Non-positive determinant encountered in KL computation; matrices may not be SPD.")

    return float(0.5 * (tr_term - logdet - N))


def min_variance_weights(
    Sigma_hat: NDArray[np.floating],
    long_only: bool = False,
    eps: float = 1e-10,
) -> NDArray[np.floating]:
    """
    Compute minimum-variance portfolio weights under budget constraint 1^T w = 1.

    Unconstrained (allows shorting):
        w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)

    Args:
        Sigma_hat: (N, N) forecast covariance (SPD recommended)
        long_only: if True, projects negative weights to 0 and renormalizes (heuristic)
                  (proper long-only requires QP; this is a simple fallback)
        eps: diagonal jitter for stability

    Returns:
        w: (N,) weights summing to 1
    """
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    if Sigma_hat.ndim != 2 or Sigma_hat.shape[0] != Sigma_hat.shape[1]:
        raise ValueError(f"Expected square matrix, got {Sigma_hat.shape}")

    N = Sigma_hat.shape[0]
    I = np.eye(N, dtype=float)
    Sigma_j = Sigma_hat + eps * I

    ones = np.ones(N, dtype=float)
    x = np.linalg.solve(Sigma_j, ones)          # Sigma^{-1} 1
    denom = float(ones @ x)
    if not np.isfinite(denom) or abs(denom) < 1e-20:
        raise ValueError("Degenerate min-var solution: 1^T Sigma^{-1} 1 is near zero.")

    w = x / denom

    if long_only:
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0:
            # fallback to equal weight if projection kills everything
            w = np.full(N, 1.0 / N, dtype=float)
        else:
            w = w / s

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