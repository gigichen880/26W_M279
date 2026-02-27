# similarity_forecast/backtests.py
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

def frobenius_error(S_hat: NDArray[np.floating], S_true: NDArray[np.floating]) -> float:
    S_hat = np.asarray(S_hat, dtype=float)
    S_true = np.asarray(S_true, dtype=float)
    if S_hat.shape != S_true.shape:
        raise ValueError(f"Shape mismatch: S_hat {S_hat.shape} vs S_true {S_true.shape}")
    return float(np.linalg.norm(S_hat - S_true, ord="fro"))


def gaussian_kl_divergence(S_true: np.ndarray, S_hat: np.ndarray, eps: float = 1e-10) -> float:
    """
    KL( N(0, S_true) || N(0, S_hat) ) = 0.5 * ( tr(S_hat^{-1} S_true) - n + logdet(S_hat) - logdet(S_true) )
    """
    S_true = np.asarray(S_true, dtype=float)
    S_hat = np.asarray(S_hat, dtype=float)
    n = S_true.shape[0]

    S_true = (S_true + S_true.T) / 2.0
    S_hat = (S_hat + S_hat.T) / 2.0

    I = np.eye(n)
    S_true_j = S_true + eps * I
    S_hat_j = S_hat + eps * I

    sign_t, logdet_t = np.linalg.slogdet(S_true_j)
    sign_h, logdet_h = np.linalg.slogdet(S_hat_j)
    if sign_t <= 0 or sign_h <= 0:
        raise ValueError("slogdet sign <= 0; matrices not SPD even after jitter.")

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

    S = (Sigma_hat + Sigma_hat.T) / 2.0

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
    w = np.asarray(w, dtype=float).reshape(-1)
    Sigma_true = np.asarray(Sigma_true, dtype=float)
    if Sigma_true.shape != (w.shape[0], w.shape[0]):
        raise ValueError(f"Shape mismatch: w {w.shape} vs Sigma_true {Sigma_true.shape}")
    return float(w @ Sigma_true @ w)


def _sym(S: np.ndarray) -> np.ndarray:
    return (S + S.T) / 2.0


def spd_jitter(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = _sym(np.asarray(S, dtype=float))
    n = S.shape[0]
    return S + eps * np.eye(n, dtype=float)


def stein_loss(S_true: np.ndarray, S_hat: np.ndarray, eps: float = 1e-10) -> float:
    """
    Stein loss: tr(S_hat^{-1} S_true) - log det(S_hat^{-1} S_true) - n
    """
    S_true_j = spd_jitter(S_true, eps=eps)
    S_hat_j = spd_jitter(S_hat, eps=eps)
    n = S_true_j.shape[0]

    X = np.linalg.solve(S_hat_j, S_true_j)
    tr_term = float(np.trace(X))

    sign_x, logdet_x = np.linalg.slogdet(X)
    if sign_x <= 0:
        # fallback: compute via logdet(S_true)-logdet(S_hat)
        st, ldt = np.linalg.slogdet(S_true_j)
        sh, ldh = np.linalg.slogdet(S_hat_j)
        if st <= 0 or sh <= 0:
            raise ValueError("Non-SPD in stein_loss even after jitter.")
        logdet_x = float(ldt - ldh)

    return float(tr_term - logdet_x - n)


def gaussian_nll_future_window(
    Sigma_hat: np.ndarray,
    fut_returns: np.ndarray,  # (H, N)
    eps: float = 1e-10,
) -> float:
    """
    Gaussian NLL (up to additive constant):
      0.5 * ( logdet(Sigma_hat) + r^T Sigma_hat^{-1} r )
    averaged over days in the future window.

    Uses the realized returns directly, which is often more stable than comparing
    two noisy cov estimates.
    """
    S = spd_jitter(Sigma_hat, eps=eps)
    fut = np.asarray(fut_returns, dtype=float)
    H, N = fut.shape

    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("Sigma_hat not SPD in gaussian_nll_future_window.")

    # Solve S^{-1} r for each r
    # Compute quadratic forms efficiently: r^T S^{-1} r
    # Do one solve on transpose: solve(S, fut.T) -> (N, H)
    X = np.linalg.solve(S, fut.T)  # (N, H)
    quad = np.sum(fut.T * X, axis=0)  # (H,)

    nll = 0.5 * (logdet + quad)  # (H,)
    return float(np.mean(nll))


def log_euclidean_distance(S_true: np.ndarray, S_hat: np.ndarray, eps: float = 1e-10) -> float:
    """
    ||log(S_hat) - log(S_true)||_F  using eigendecomposition (SPD).
    """
    A = spd_jitter(S_true, eps=eps)
    B = spd_jitter(S_hat, eps=eps)

    wa, Qa = np.linalg.eigh(A)
    wb, Qb = np.linalg.eigh(B)
    wa = np.maximum(wa, eps)
    wb = np.maximum(wb, eps)

    logA = (Qa * np.log(wa)) @ Qa.T
    logB = (Qb * np.log(wb)) @ Qb.T
    return float(np.linalg.norm(logB - logA, ord="fro"))


def covariance_to_correlation(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    S = _sym(np.asarray(S, dtype=float))
    d = np.diag(S)
    d = np.maximum(d, eps)
    inv_sqrt = 1.0 / np.sqrt(d)
    C = (S * inv_sqrt[None, :]) * inv_sqrt[:, None]
    # clamp numeric drift
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C


def corr_offdiag_fro(S_true: np.ndarray, S_hat: np.ndarray) -> float:
    C_t = covariance_to_correlation(S_true)
    C_h = covariance_to_correlation(S_hat)
    off = ~np.eye(C_t.shape[0], dtype=bool)
    diff = (C_h - C_t)[off]
    return float(np.linalg.norm(diff, ord=2))


def corr_upper_spearman(S_true: np.ndarray, S_hat: np.ndarray) -> float:
    """
    Spearman correlation between upper-triangle correlation entries.
    Uses pandas rank to handle ties.
    """
    C_t = covariance_to_correlation(S_true)
    C_h = covariance_to_correlation(S_hat)

    iu = np.triu_indices(C_t.shape[0], k=1)
    vt = C_t[iu]
    vh = C_h[iu]

    s1 = pd.Series(vt).rank(method="average")
    s2 = pd.Series(vh).rank(method="average")
    c = s1.corr(s2, method="pearson")
    return float(c) if np.isfinite(c) else np.nan


def eigen_log_mse(S_true: np.ndarray, S_hat: np.ndarray, eps: float = 1e-10) -> float:
    """
    Mean squared error between log-eigenvalues (sorted).
    """
    A = spd_jitter(S_true, eps=eps)
    B = spd_jitter(S_hat, eps=eps)
    la = np.linalg.eigvalsh(A)
    lb = np.linalg.eigvalsh(B)
    la = np.maximum(la, eps)
    lb = np.maximum(lb, eps)
    return float(np.mean((np.log(lb) - np.log(la)) ** 2))


def condition_number(S: np.ndarray, eps: float = 1e-10) -> float:
    S = spd_jitter(S, eps=eps)
    lam = np.linalg.eigvalsh(S)
    lam = np.maximum(lam, eps)
    return float(np.max(lam) / np.min(lam))


def make_eval_portfolios(
    N: int,
    n_rand: int = 20,
    seed: int = 0,
    long_only: bool = True,
) -> np.ndarray:
    """
    Fixed portfolio set for evaluation.
    - Always includes equal-weight.
    - Adds n_rand random portfolios (Dirichlet if long_only).
    """
    rng = np.random.default_rng(seed)
    W = []

    # equal weight
    W.append(np.full(N, 1.0 / N))

    if long_only:
        for _ in range(n_rand):
            w = rng.dirichlet(alpha=np.ones(N))
            W.append(w)
    else:
        for _ in range(n_rand):
            w = rng.normal(size=N)
            w = w / (np.sum(np.abs(w)) + 1e-12)
            W.append(w)

    return np.asarray(W, dtype=float)  # (M, N)


def multi_portfolio_risk_errors(
    Sigma_hat: np.ndarray,
    Sigma_true: np.ndarray,
    W_eval: np.ndarray,  # (M, N)
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Compare predicted vs realized variances across a fixed set of portfolios.
    Returns MSE(var) and MSE(log var), plus mean abs log error (more interpretable).
    """
    Sh = _sym(np.asarray(Sigma_hat, dtype=float))
    St = _sym(np.asarray(Sigma_true, dtype=float))
    W = np.asarray(W_eval, dtype=float)

    pred = np.einsum("mi,ij,mj->m", W, Sh, W)
    real = np.einsum("mi,ij,mj->m", W, St, W)

    pred = np.maximum(pred, eps)
    real = np.maximum(real, eps)

    mse_var = float(np.mean((pred - real) ** 2))
    log_err = np.log(pred) - np.log(real)
    mse_log = float(np.mean(log_err ** 2))
    mae_log = float(np.mean(np.abs(log_err)))
    return {"port_mse_var": mse_var, "port_mse_logvar": mse_log, "port_mae_logvar": mae_log}


def weight_concentration_stats(w: np.ndarray) -> dict[str, float]:
    w = np.asarray(w, dtype=float).reshape(-1)
    hhi = float(np.sum(w ** 2))
    maxw = float(np.max(np.abs(w)))
    l1 = float(np.sum(np.abs(w)))
    return {"w_hhi": hhi, "w_max_abs": maxw, "w_l1": l1}