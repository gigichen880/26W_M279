"""Metric families: raw-object, common-conditioned SPD, predictive, decision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from similarity_forecast.backtests import gmvp_weights, stein_loss, gaussian_kl_divergence
from similarity_forecast.core import logm_spd, project_to_spd, symmetrize
from similarity_forecast.redesign.conditioner import ConditioningConfig, apply_conditioner


def _corr_from_cov(S: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    S = symmetrize(S)
    d = np.sqrt(np.maximum(np.diag(S), eps))
    return S / np.outer(d, d)


def frobenius(A: NDArray[np.floating], B: NDArray[np.floating]) -> float:
    """Squared Frobenius ||A-B||_F^2 = sum_{ij}(A_ij-B_ij)^2. Prefer frobenius_norm for ||.||_F."""
    D = A - B
    return float(np.sum(D * D))


def frobenius_norm(A: NDArray[np.floating], B: NDArray[np.floating]) -> float:
    """Standard Frobenius norm ||A-B||_F."""
    D = np.asarray(A, dtype=float) - np.asarray(B, dtype=float)
    return float(np.linalg.norm(D, ord="fro"))


def log_euclidean(A: NDArray[np.floating], B: NDArray[np.floating], eps: float = 1e-12) -> float:
    La = logm_spd(project_to_spd(A, eps=eps), eps=eps)
    Lb = logm_spd(project_to_spd(B, eps=eps), eps=eps)
    return frobenius(La, Lb)


def corr_frobenius(A: NDArray[np.floating], B: NDArray[np.floating]) -> float:
    return frobenius(_corr_from_cov(A), _corr_from_cov(B))


def eigenvalue_l2(A: NDArray[np.floating], B: NDArray[np.floating]) -> float:
    wa = np.sort(np.linalg.eigvalsh(symmetrize(A)))
    wb = np.sort(np.linalg.eigvalsh(symmetrize(B)))
    return float(np.sum((wa - wb) ** 2))


def gaussian_nll_returns(
    returns: NDArray[np.floating],
    Sigma_hat: NDArray[np.floating],
    eps: float = 1e-12,
) -> float:
    """Mean per-day Gaussian NLL of future returns under N(0, Σ̂)."""
    R = np.asarray(returns, dtype=float)
    S = apply_conditioner(Sigma_hat, ConditioningConfig(eta=0.0, abs_floor=eps))
    # Use plain SPD projection for NLL when caller already conditioned; keep invertibility.
    S = project_to_spd(S, eps=eps)
    try:
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return float("nan")
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return float("nan")
    n = S.shape[0]
    vals = []
    for t in range(R.shape[0]):
        r = R[t]
        m = np.isfinite(r)
        if m.sum() < max(2, int(0.5 * n)):
            continue
        rr = np.where(m, r, 0.0)
        # Restrict to observed names via mask on quadratic form / approx full S
        q = float(rr @ inv @ rr)
        vals.append(0.5 * (logdet + q + n * np.log(2.0 * np.pi)))
    return float(np.mean(vals)) if vals else float("nan")


def conditioned_gmvp_variance_regret(
    Sigma_hat: NDArray[np.floating],
    Sigma_real: NDArray[np.floating],
    cfg: ConditioningConfig,
    *,
    long_only: bool = False,
    max_abs_weight: Optional[float] = None,
    max_l1: Optional[float] = None,
) -> dict:
    """
    Frozen selection metric (may be negative):
      R_raw = w' Σ^real w − w*' Σ^real w*
    with w=w(C_η(Σ̂)), w*=w(C_η(Σ^real)).

    Robustness metric (mathematically nonnegative):
      R_cond = w' C_η(Σ^real) w − w*' C_η(Σ^real) w*
    """
    Sf = apply_conditioner(Sigma_hat, cfg)
    Sr = apply_conditioner(Sigma_real, cfg)
    w = _portfolio_weights(Sf, long_only=long_only, max_abs_weight=max_abs_weight, max_l1=max_l1)
    w_star = _portfolio_weights(Sr, long_only=long_only, max_abs_weight=max_abs_weight, max_l1=max_l1)
    Sraw = symmetrize(np.asarray(Sigma_real, dtype=float))
    var = float(w @ Sraw @ w)
    var_star = float(w_star @ Sraw @ w_star)
    var_cond = float(w @ Sr @ w)
    var_star_cond = float(w_star @ Sr @ w_star)
    return {
        "gmvp_var": var,
        "oracle_var": var_star,
        "var_regret": var - var_star,  # frozen alias = R_raw
        "R_raw": var - var_star,
        "R_cond": var_cond - var_star_cond,
        "gmvp_var_cond": var_cond,
        "pred_var": float(w @ Sf @ w),
        "leverage_l1": float(np.sum(np.abs(w))),
        "max_abs_w": float(np.max(np.abs(w))),
        "hhi": float(np.sum(w * w)),
        "weights": w,
    }


def _portfolio_weights(
    S: NDArray[np.floating],
    *,
    long_only: bool,
    max_abs_weight: Optional[float],
    max_l1: Optional[float],
) -> NDArray[np.floating]:
    w = gmvp_weights(S, long_only=long_only)
    w = np.asarray(w, dtype=float)
    if max_abs_weight is not None and max_abs_weight > 0:
        c = float(max_abs_weight)
        w = np.clip(w, -c, c)
        s = w.sum()
        if abs(s) < 1e-12:
            w = np.ones_like(w) / w.size
        else:
            w = w / s
    if max_l1 is not None and max_l1 > 0:
        # Project onto {1'w=1, ||w||_1 <= Lmax} by mixing toward equal weight.
        # (Naive L1 scaling + re-sum-normalization undoes the leverage cap.)
        s = float(w.sum())
        if abs(s) > 1e-12:
            w = w / s
        L = float(np.sum(np.abs(w)))
        if L > max_l1:
            n = w.size
            eq = np.ones(n, dtype=float) / n
            lo, hi = 0.0, 1.0
            for _ in range(60):
                a = 0.5 * (lo + hi)
                wa = (1.0 - a) * w + a * eq
                if float(np.sum(np.abs(wa))) > max_l1:
                    lo = a
                else:
                    hi = a
            w = (1.0 - hi) * w + hi * eq
    return w


@dataclass
class ForecastEval:
    raw_frob: float
    raw_logeuc: float
    raw_corr_frob: float
    raw_eig_l2: float
    cond_stein: float
    cond_kl: float
    nll: float
    gmvp_var: float
    var_regret: float
    pred_var: float
    leverage_l1: float
    max_abs_w: float
    hhi: float


def evaluate_forecast(
    Sigma_hat: NDArray[np.floating],
    Sigma_real: NDArray[np.floating],
    fut_returns: NDArray[np.floating],
    cfg: ConditioningConfig,
    *,
    long_only: bool = False,
    max_abs_weight: Optional[float] = None,
    max_l1: Optional[float] = None,
) -> ForecastEval:
    Sf = apply_conditioner(Sigma_hat, cfg)
    Sr = apply_conditioner(Sigma_real, cfg)
    dec = conditioned_gmvp_variance_regret(
        Sigma_hat, Sigma_real, cfg,
        long_only=long_only, max_abs_weight=max_abs_weight, max_l1=max_l1,
    )
    return ForecastEval(
        raw_frob=frobenius(Sigma_hat, Sigma_real),
        raw_logeuc=log_euclidean(Sigma_hat, Sigma_real),
        raw_corr_frob=corr_frobenius(Sigma_hat, Sigma_real),
        raw_eig_l2=eigenvalue_l2(Sigma_hat, Sigma_real),
        cond_stein=float(stein_loss(Sr, Sf)),
        cond_kl=float(gaussian_kl_divergence(Sr, Sf)),
        nll=gaussian_nll_returns(fut_returns, Sf),
        gmvp_var=dec["gmvp_var"],
        var_regret=dec["var_regret"],
        pred_var=dec["pred_var"],
        leverage_l1=dec["leverage_l1"],
        max_abs_w=dec["max_abs_w"],
        hhi=dec["hhi"],
    )
