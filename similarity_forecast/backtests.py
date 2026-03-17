# similarity_forecast/backtests.py
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from .core import project_to_spd, cov_from_returns

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

def weight_concentration_stats(w: np.ndarray) -> dict[str, float]:
    w = np.asarray(w, dtype=float).reshape(-1)
    hhi = float(np.sum(w ** 2))
    maxw = float(np.max(np.abs(w)))
    l1 = float(np.sum(np.abs(w)))
    return {"w_hhi": hhi, "w_max_abs": maxw, "w_l1": l1}

def baseline_rolling_cov(past: np.ndarray, ddof: int = 1) -> np.ndarray:
    return cov_from_returns(past, ddof=ddof)


def baseline_persistence_realized_cov(R: np.ndarray, raw_anchor: int, horizon: int, ddof: int = 1) -> np.ndarray:
    """
    Uses the previous horizon window [t-H+1 ... t] as the forecast for [t+1 ... t+H].
    """
    prev = R[raw_anchor - horizon + 1 : raw_anchor + 1, :]
    return cov_from_returns(prev, ddof=ddof)

def baseline_shrink_to_diag(S: np.ndarray, gamma: float = 0.3) -> np.ndarray:
    D = np.diag(np.diag(S))
    return (1.0 - gamma) * S + gamma * D


# ---------- Realized volatility (log-vol) baselines and metrics ----------
def realized_log_vol_from_returns(
    R: NDArray[np.floating], ddof: int = 1, eps: float = 1e-12
) -> NDArray[np.floating]:
    """
    (T, N) returns -> (N,) log realized vol per asset.
    Uses sample cov then sqrt(diag) then log; matches VolTarget.target() convention.
    """
    S = cov_from_returns(R, ddof=ddof)
    v = np.sqrt(np.maximum(np.diag(S), eps))
    return np.log(v).astype(float)


def baseline_rolling_vol(past: NDArray[np.floating], ddof: int = 1, eps: float = 1e-12) -> NDArray[np.floating]:
    """Log realized vol from past window (lookback returns)."""
    return realized_log_vol_from_returns(past, ddof=ddof, eps=eps)


def baseline_persistence_vol(
    R: NDArray[np.floating], raw_anchor: int, horizon: int, ddof: int = 1, eps: float = 1e-12
) -> NDArray[np.floating]:
    """Log realized vol from previous horizon window [raw_anchor-horizon+1 .. raw_anchor+1]."""
    prev = R[raw_anchor - horizon + 1 : raw_anchor + 1, :]
    return realized_log_vol_from_returns(prev, ddof=ddof, eps=eps)


def baseline_shrink_vol(
    past: NDArray[np.floating], ddof: int = 1, gamma: float = 0.3, eps: float = 1e-12
) -> NDArray[np.floating]:
    """Log vol from shrink-to-diag covariance of past window (diag unchanged → same as rolling)."""
    S = cov_from_returns(past, ddof=ddof)
    S_shrink = baseline_shrink_to_diag(S, gamma=gamma)
    v = np.sqrt(np.maximum(np.diag(S_shrink), eps))
    return np.log(v).astype(float)


def baseline_shrink_vol_toward_cs_mean(
    past: NDArray[np.floating], ddof: int = 1, gamma: float = 0.3, eps: float = 1e-12
) -> NDArray[np.floating]:
    """
    Vol-specific shrinkage: shrink each asset's log-vol toward cross-sectional mean log-vol.
    (1 - gamma) * vol_roll + gamma * mean(vol_roll). Reduces cross-sectional dispersion.
    """
    vol_roll = realized_log_vol_from_returns(past, ddof=ddof, eps=eps)
    cs_mean = float(np.nanmean(vol_roll))
    return ((1.0 - gamma) * vol_roll + gamma * cs_mean).astype(float)


def eval_vol_metrics(
    vol_hat: NDArray[np.floating], vol_true: NDArray[np.floating]
) -> dict[str, float]:
    """
    Scalar metrics for log-vol prediction (vector vs vector).
    Lower is better for all.
    """
    vh = np.asarray(vol_hat, dtype=float).reshape(-1)
    vt = np.asarray(vol_true, dtype=float).reshape(-1)
    if vh.shape != vt.shape:
        raise ValueError(f"Shape mismatch: vol_hat {vh.shape} vs vol_true {vt.shape}")
    diff = vh - vt
    mse = float(np.nanmean(diff ** 2))
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    return {"vol_mse": mse, "vol_mae": mae, "vol_rmse": rmse}


def eval_all_metrics(
    Sigma_hat: np.ndarray,
    Sigma_true: np.ndarray,
    fut: np.ndarray,
    long_only: bool,
    W_eval: np.ndarray | None = None,
) -> dict:
    # base
    fro = frobenius_error(Sigma_hat, Sigma_true)

    # project SPD for geometry metrics
    S_hat = project_to_spd((Sigma_hat + Sigma_hat.T) / 2.0, eps=1e-8)
    S_true = project_to_spd((Sigma_true + Sigma_true.T) / 2.0, eps=1e-8)

    out = {"fro": fro}

    out["kl"] = gaussian_kl_divergence(S_true=S_true, S_hat=S_hat)

    try:
        out["nll"] = gaussian_nll_future_window(Sigma_hat=S_hat, fut_returns=fut, eps=1e-10)
    except Exception:
        out["nll"] = np.nan

    try:
        out["stein"] = stein_loss(S_true=S_true, S_hat=S_hat, eps=1e-10)
    except Exception:
        out["stein"] = np.nan

    try:
        out["logeuc"] = log_euclidean_distance(S_true=S_true, S_hat=S_hat, eps=1e-10)
    except Exception:
        out["logeuc"] = np.nan

    try:
        out["corr_offdiag_fro"] = corr_offdiag_fro(S_true=S_true, S_hat=S_hat)
    except Exception:
        out["corr_offdiag_fro"] = np.nan

    try:
        out["corr_spearman"] = corr_upper_spearman(S_true=S_true, S_hat=S_hat)
    except Exception:
        out["corr_spearman"] = np.nan

    try:
        out["eig_log_mse"] = eigen_log_mse(S_true=S_true, S_hat=S_hat, eps=1e-10)
    except Exception:
        out["eig_log_mse"] = np.nan

    try:
        cond_hat = condition_number(S_hat, eps=1e-10)
        cond_true = condition_number(S_true, eps=1e-10)
        out["cond_ratio"] = float(cond_hat / max(cond_true, 1e-12))
    except Exception:
        out["cond_ratio"] = np.nan
    return out

# GMVP Portfolio Testings
def _project_to_simplex(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Euclidean projection onto simplex {w >= 0, sum w = 1}.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        # fallback: uniform
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n

def gmvp_weights(
    Sigma: NDArray[np.floating],
    long_only: bool = False,
    ridge: float = 1e-8,
    max_iter: int = 500,
    step: float = 0.5,
    tol: float = 1e-10,
) -> NDArray[np.floating]:
    """
    GMVP weights for covariance Sigma.

    Unconstrained: closed form  w ∝ Sigma^{-1} 1
    Long-only: projected gradient descent on simplex.

    Notes:
      - Adds ridge to improve conditioning.
      - Assumes Sigma is SPD-ish (you already project_to_spd elsewhere).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    I = np.eye(n)

    S = Sigma + ridge * I
    ones = np.ones(n)

    if not long_only:
        # closed form
        try:
            x = np.linalg.solve(S, ones)
        except np.linalg.LinAlgError:
            x = np.linalg.pinv(S) @ ones
        denom = float(ones @ x)
        if not np.isfinite(denom) or abs(denom) < 1e-18:
            return np.ones(n) / n
        w = x / denom
        return w.astype(float)

    # long-only: projected gradient descent on simplex
    w = np.ones(n) / n
    prev_obj = np.inf
    for _ in range(max_iter):
        grad = 2.0 * (S @ w)
        w_new = _project_to_simplex(w - step * grad)

        obj = float(w_new @ S @ w_new)
        if not np.isfinite(obj):
            return np.ones(n) / n

        if abs(prev_obj - obj) < tol:
            w = w_new
            break

        # simple monotone step: if objective increases, shrink step
        if obj > prev_obj:
            step *= 0.5
            if step < 1e-12:
                break
            continue

        w = w_new
        prev_obj = obj

    return w.astype(float)

def hold_period_portfolio_stats(
    fut: NDArray[np.floating],     # (H, N) future returns
    w: NDArray[np.floating],       # (N,)
    ann_factor: float = 252.0,
) -> dict:
    """
    Compute realized performance over the horizon holding w fixed.

    Returns:
      - daily portfolio return series stats
      - cumulative return over horizon
      - realized variance/vol (per-day)
      - annualized sharpe (rough)
    """
    fut = np.asarray(fut, dtype=float)
    w = np.asarray(w, dtype=float)

    rp = fut @ w  # (H,)
    rp = rp[np.isfinite(rp)]
    if rp.size == 0:
        return {
            "gmvp_cumret": np.nan,
            "gmvp_mean": np.nan,
            "gmvp_vol": np.nan,
            "gmvp_var": np.nan,
            "gmvp_sharpe": np.nan,
        }

    cumret = float(np.prod(1.0 + rp) - 1.0)
    mu = float(np.mean(rp))
    var = float(np.var(rp, ddof=1)) if rp.size > 1 else 0.0
    vol = float(np.sqrt(var))
    sharpe = float((mu / (vol + 1e-12)) * np.sqrt(ann_factor)) if vol > 0 else np.nan

    return {
        "gmvp_cumret": cumret,
        "gmvp_mean": mu,
        "gmvp_vol": vol,
        "gmvp_var": var,
        "gmvp_sharpe": sharpe,
    }