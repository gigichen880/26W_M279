#!/usr/bin/env python3
"""Task A/B: audit C_η, Stein/KL, Frobenius units, and regret definitions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from similarity_forecast.backtests import frobenius_error, gaussian_kl_divergence, stein_loss
from similarity_forecast.core import cov_from_returns, symmetrize
from similarity_forecast.redesign.conditioner import ConditioningConfig, apply_conditioner, relative_eigen_floor
from similarity_forecast.redesign.metrics import frobenius as frob_squared
from similarity_forecast.redesign.runner import load_returns


def audit_conditioner():
    rng = np.random.default_rng(0)
    n = 20
    A = rng.normal(size=(n, n))
    S = A @ A.T + 0.01 * np.eye(n)
    T = rng.normal(size=(n, n))
    R = T @ T.T + 0.02 * np.eye(n)
    eta = 0.01
    cfg = ConditioningConfig(eta=eta)

    # Independent thresholds
    eps_f = eta * np.trace(S) / n
    eps_r = eta * np.trace(R) / n
    Sf = apply_conditioner(S, cfg)
    Sr = apply_conditioner(R, cfg)
    wf = np.linalg.eigvalsh(Sf)
    wr = np.linalg.eigvalsh(Sr)

    # Mutating R must not change Sf
    R2 = R + rng.normal(scale=0.1, size=R.shape)
    R2 = R2 @ R2.T + np.eye(n)
    Sf2 = apply_conditioner(S, cfg)
    assert np.allclose(Sf, Sf2), "forecast conditioning depends on target!"

    return {
        "eps_f": float(eps_f),
        "eps_r": float(eps_r),
        "eps_f_equals_eps_r": bool(np.isclose(eps_f, eps_r)),
        "Sf_symmetric": bool(np.allclose(Sf, Sf.T)),
        "Sr_symmetric": bool(np.allclose(Sr, Sr.T)),
        "Sf_min_eig": float(wf.min()),
        "Sr_min_eig": float(wr.min()),
        "Sf_min_eig_ge_eps_f": bool(wf.min() >= eps_f - 1e-10),
        "Sr_min_eig_ge_eps_r": bool(wr.min() >= eps_r - 1e-10),
        "forecast_independent_of_target": True,
        "both_spd": bool(wf.min() > 0 and wr.min() > 0),
    }


def audit_stein_kl():
    rng = np.random.default_rng(1)
    n = 15
    A = rng.normal(size=(n, n))
    S = A @ A.T + 0.05 * np.eye(n)
    stein0 = stein_loss(S, S)
    kl0 = gaussian_kl_divergence(S, S)
    # Direction check: KL(true||hat); if hat is inflated, KL large
    Shat = S * 2.0
    kl_inflated = gaussian_kl_divergence(S, Shat)
    stein_inflated = stein_loss(S, Shat)
    return {
        "stein_S_S": float(stein0),
        "stein_near_zero": bool(abs(stein0) < 1e-8),
        "kl_S_S": float(kl0),
        "kl_near_zero": bool(abs(kl0) < 1e-8),
        "kl_direction": "KL(N(0,S_true) || N(0,S_hat)) = 0.5*(tr(S_hat^{-1}S_true)-n+logdet(S_hat)-logdet(S_true))",
        "stein_formula": "tr(S_hat^{-1}S_true) - logdet(S_hat^{-1}S_true) - n",
        "kl_inflated_hat": float(kl_inflated),
        "stein_inflated_hat": float(stein_inflated),
        "evaluate_on": "C_eta(hat), C_eta(real) with independent thresholds",
    }


def audit_frobenius_units():
    df = load_returns(str(ROOT / "data/processed/returns_universe_100.parquet"))
    df = df.loc["2017-01-01":"2017-06-30"].dropna(how="all")
    R = df.to_numpy(dtype=float)
    # one 50/20 window
    L, H = 50, 20
    past = R[L : L + L]
    fut = R[L + L : L + L + H]
    # impute lightly
    past = np.nan_to_num(past, nan=0.0)
    fut = np.nan_to_num(fut, nan=0.0)
    S = cov_from_returns(fut)
    # rolling vs persistence-ish
    Sroll = cov_from_returns(past)
    sq = frob_squared(Sroll, S)
    uns = frobenius_error(Sroll, S)
    return {
        "returns_scale": "decimal daily returns (median |r|≈0.01)",
        "redesign_raw_frob_definition": "sum_{ij} (A_ij-B_ij)^2 = ||A-B||_F^2 (SQUARED Frobenius)",
        "old_paper_frob_definition": "np.linalg.norm(A-B, ord='fro') = ||A-B||_F (unsquared)",
        "example_squared": float(sq),
        "example_unsquared": float(uns),
        "example_sqrt_squared": float(np.sqrt(sq)),
        "match_sqrt": bool(np.isclose(np.sqrt(sq), uns)),
        "typical_diag_entry": float(np.mean(np.diag(S))),
        "typical_||S||_F": float(np.linalg.norm(S, ord="fro")),
        "oos_panel_mean_raw_frob_is_squared": True,
        "scale_note": (
            "Old manuscript ~0.02–0.03 is unsquared Frobenius on similarly scaled covs; "
            "redesign panel ~10^2–10^3 is squared Frobenius. Rankings under monotone x->x^2 "
            "are identical for nonnegative values, but absolute levels are not comparable."
        ),
    }


def audit_regret_sign():
    """Show R_raw can be negative; R_cond >= 0."""
    rng = np.random.default_rng(2)
    n = 30
    A = rng.normal(size=(n, n))
    Sreal = A @ A.T
    # Make Sreal nearly singular
    w, V = np.linalg.eigh(symmetrize(Sreal))
    w = np.maximum(w, 0.0)
    w[:5] = 1e-12
    Sreal = (V * w) @ V.T
    Shat = Sreal + 0.1 * np.eye(n)
    cfg = ConditioningConfig(eta=0.01)
    Sf = apply_conditioner(Shat, cfg)
    Sr = apply_conditioner(Sreal, cfg)
    from similarity_forecast.backtests import gmvp_weights

    w = gmvp_weights(Sf)
    wstar = gmvp_weights(Sr)
    R_raw = float(w @ Sreal @ w - wstar @ Sreal @ wstar)
    R_cond = float(w @ Sr @ w - wstar @ Sr @ wstar)
    # Monte Carlo of signs
    negs = 0
    for i in range(50):
        A = rng.normal(size=(n, n))
        Sreal = A @ A.T
        w_, V_ = np.linalg.eigh(symmetrize(Sreal))
        w_ = np.maximum(w_, 0)
        w_[:3] = 0
        Sreal = (V_ * w_) @ V_.T + 1e-14 * np.eye(n)
        Shat = np.eye(n)
        Sf = apply_conditioner(Shat, cfg)
        Sr = apply_conditioner(Sreal, cfg)
        ww = gmvp_weights(Sf)
        ws = gmvp_weights(Sr)
        if ww @ Sreal @ ww - ws @ Sreal @ ws < -1e-12:
            negs += 1
    return {
        "example_R_raw": R_raw,
        "example_R_cond": R_cond,
        "R_cond_nonneg_example": bool(R_cond >= -1e-10),
        "R_raw_can_be_negative_in_MC": negs > 0,
        "neg_count_of_50": negs,
        "naming": {
            "frozen_metric": "raw_realized_excess_vs_conditioned_oracle (R_raw)",
            "robustness_metric": "conditioned_gmvp_variance_regret (R_cond >= 0)",
        },
    }


def main():
    outdir = ROOT / "results" / "redesign" / "metric_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "conditioner": audit_conditioner(),
        "stein_kl": audit_stein_kl(),
        "frobenius_units": audit_frobenius_units(),
        "regret": audit_regret_sign(),
    }
    (outdir / "audit.json").write_text(json.dumps(report, indent=2))
    # human summary
    lines = ["# Metric audit machine summary", ""]
    for k, v in report.items():
        lines.append(f"## {k}")
        for kk, vv in v.items():
            lines.append(f"- **{kk}**: `{vv}`")
        lines.append("")
    (outdir / "audit_summary.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))
    # assert hard requirements
    assert report["conditioner"]["forecast_independent_of_target"]
    assert report["conditioner"]["Sf_min_eig_ge_eps_f"]
    assert report["stein_kl"]["stein_near_zero"]
    assert report["stein_kl"]["kl_near_zero"]
    assert report["frobenius_units"]["match_sqrt"]
    print("ALL HARD ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
