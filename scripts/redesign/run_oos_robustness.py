#!/usr/bin/env python3
"""Pre-specified OOS robustness for the FROZEN D0 model (no architecture changes).

C1: η sensitivity (report only; official remains η=0.01)
C2: predictive NLL rankings
C3: constrained GMVP (gross leverage L_max)
D:  eigenstructure diagnostics

Does not retune or select a new finalist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from similarity_forecast.backtests import gaussian_kl_divergence, gmvp_weights, stein_loss
from similarity_forecast.core import validate_window
from similarity_forecast.redesign.baselines import BASELINE_FNS, persistence_cov
from similarity_forecast.redesign.conditioner import ConditioningConfig, apply_conditioner
from similarity_forecast.redesign.embeddings import build_embedder
from similarity_forecast.redesign.metrics import (
    conditioned_gmvp_variance_regret,
    frobenius,
    frobenius_norm,
    gaussian_nll_returns,
)
from similarity_forecast.redesign.runner import ProtocolConfig, load_returns
from similarity_forecast.redesign.similarity import (
    SimilarityForecaster,
    SimilarityLibrary,
    build_covariance_target,
)


# Pre-specified (disclosed) robustness settings — not selected on OOS
ETA_GRID = (0.001, 0.01, 0.05)
LMAX_GROSS = 2.0  # one constrained-GMVP specification


def eigen_diag(Shat, Sreal, cfg):
    Sf = apply_conditioner(Shat, cfg)
    Sr = apply_conditioner(Sreal, cfg)
    wh = np.sort(np.linalg.eigvalsh(Sf))[::-1]
    wr = np.sort(np.linalg.eigvalsh(Sr))[::-1]
    n = wh.size
    top = slice(0, max(1, n // 10))
    bot = slice(-max(1, n // 10), None)
    mid = slice(max(1, n // 10), -max(1, n // 10) if n > 20 else n)
    def band_err(a, b, sl):
        return float(np.mean((np.log(a[sl] + 1e-12) - np.log(b[sl] + 1e-12)) ** 2))
    return {
        "cond_number_hat": float(wh[0] / max(wh[-1], 1e-12)),
        "cond_number_real": float(wr[0] / max(wr[-1], 1e-12)),
        "min_eig_hat": float(wh[-1]),
        "min_eig_real": float(wr[-1]),
        "logeig_mse_top": band_err(wh, wr, top),
        "logeig_mse_mid": band_err(wh, wr, mid) if mid.start < mid.stop else np.nan,
        "logeig_mse_bot": band_err(wh, wr, bot),
        "inv_frob_sq": float(np.sum((np.linalg.inv(Sf) - np.linalg.inv(Sr)) ** 2)),
    }


def main():
    out_rob = ROOT / "results" / "redesign" / "robustness"
    out_eig = ROOT / "results" / "redesign" / "eigenstructure"
    out_rob.mkdir(parents=True, exist_ok=True)
    out_eig.mkdir(parents=True, exist_ok=True)

    proto = ProtocolConfig()
    df = load_returns(str(ROOT / "data/processed/returns_universe_100.parquet"))
    R = df.to_numpy(dtype=float)
    dates = df.index
    T = R.shape[0]
    L, H, g = proto.lookback, proto.horizon, proto.neighbor_gap

    anchors = [a for a in range(L, T - H, proto.stride) if a >= L + proto.burn_in]
    test_anchors = [a for a in anchors if dates[a] >= pd.Timestamp(proto.test_start)]

    forecaster = SimilarityForecaster(
        metric="euclidean", k_neighbors=20, aggregation="log_euclidean"
    )
    embedder = None
    library = None
    last_refit = None

    rows = []
    eig_rows = []

    def refit(a):
        nonlocal embedder, library, last_refit
        train = R[: a + 1]
        emb = build_embedder("pca_only", lookback=L, pca_k=8)
        emb.fit(train)
        lib_a, embeds, targets = [], [], []
        for t in range(L, a - H - g + 1, max(1, proto.stride // 2)):
            past, fut = train[t - L : t], train[t + 1 : t + H + 1]
            if past.shape[0] < L or fut.shape[0] < H:
                continue
            if not validate_window(past) or not validate_window(fut):
                continue
            try:
                z = emb.embed(past)
            except Exception:
                continue
            lib_a.append(t)
            embeds.append(z)
            targets.append(build_covariance_target(fut))
        embedder = emb
        library = SimilarityLibrary(
            anchors=np.asarray(lib_a, dtype=np.int64),
            embeds=np.vstack(embeds),
            targets=targets,
        )
        last_refit = dates[a]

    print(f"[robustness] OOS anchors={len(test_anchors)}", flush=True)
    for a in test_anchors:
        if embedder is None or library is None or (dates[a] - last_refit).days >= proto.refit_every_days:
            try:
                refit(a)
            except Exception:
                continue
        past = R[a - L : a]
        fut = R[a + 1 : a + H + 1]
        if not validate_window(past) or not validate_window(fut):
            continue
        try:
            z = embedder.embed(past)
            S_d0, _ = forecaster.predict(z, library, raw_anchor=a, horizon=H, neighbor_gap=g)
        except Exception:
            continue
        S_real = build_covariance_target(fut)
        methods = {"D0_pca8": S_d0}
        for name, fn in BASELINE_FNS.items():
            try:
                methods[name] = fn(past)
            except Exception:
                pass
        prev = R[max(0, a - H + 1) : a + 1]
        if prev.shape[0] == H:
            try:
                methods["persistence"] = persistence_cov(build_covariance_target(prev))
            except Exception:
                pass

        for mname, Sh in methods.items():
            for eta in ETA_GRID:
                cfg = ConditioningConfig(eta=eta)
                Sf = apply_conditioner(Sh, cfg)
                Sr = apply_conditioner(S_real, cfg)
                dec = conditioned_gmvp_variance_regret(Sh, S_real, cfg)
                # constrained
                dec_c = conditioned_gmvp_variance_regret(
                    Sh, S_real, cfg, max_l1=LMAX_GROSS
                )
                nll = gaussian_nll_returns(fut, Sf)
                row = {
                    "date": dates[a],
                    "method": mname,
                    "eta": eta,
                    "raw_frob_sq": frobenius(Sh, S_real),
                    "raw_frob": frobenius_norm(Sh, S_real),
                    "cond_stein": float(stein_loss(Sr, Sf)),
                    "cond_kl": float(gaussian_kl_divergence(Sr, Sf)),
                    "nll": nll,
                    "gmvp_var_raw": dec["gmvp_var"],
                    "R_raw": dec["R_raw"],
                    "R_cond": dec["R_cond"],
                    "leverage_l1": dec["leverage_l1"],
                    "max_abs_w": dec["max_abs_w"],
                    "hhi": dec["hhi"],
                    "gmvp_var_raw_Lmax": dec_c["gmvp_var"],
                    "R_raw_Lmax": dec_c["R_raw"],
                    "R_cond_Lmax": dec_c["R_cond"],
                    "leverage_l1_Lmax": dec_c["leverage_l1"],
                    "max_abs_w_Lmax": dec_c["max_abs_w"],
                }
                rows.append(row)
                if eta == 0.01:
                    ed = eigen_diag(Sh, S_real, cfg)
                    eig_rows.append({"date": dates[a], "method": mname, **ed, "leverage_l1": dec["leverage_l1"]})

    panel = pd.DataFrame(rows)
    panel.to_csv(out_rob / "oos_robustness_panel.csv", index=False)
    eig = pd.DataFrame(eig_rows)
    eig.to_csv(out_eig / "oos_eigenstructure_panel.csv", index=False)

    # Summaries at official eta
    off = panel[panel.eta == 0.01]
    summ = (
        off.groupby("method")[
            [
                "raw_frob",
                "raw_frob_sq",
                "cond_stein",
                "cond_kl",
                "nll",
                "gmvp_var_raw",
                "R_raw",
                "R_cond",
                "leverage_l1",
                "max_abs_w",
                "gmvp_var_raw_Lmax",
                "R_cond_Lmax",
                "leverage_l1_Lmax",
            ]
        ]
        .mean()
        .sort_values("R_raw")
    )
    summ.to_csv(out_rob / "oos_summary_eta0.01.csv")
    print("\n=== Official η=0.01 OOS summary ===")
    print(summ.to_string())

    # η sensitivity ranks
    rank_rows = []
    for eta in ETA_GRID:
        sub = panel[panel.eta == eta]
        for metric in ["cond_stein", "cond_kl", "nll", "gmvp_var_raw", "R_raw", "R_cond"]:
            order = sub.groupby("method")[metric].mean().sort_values()
            for rank, (m, v) in enumerate(order.items(), 1):
                rank_rows.append({"eta": eta, "metric": metric, "method": m, "value": v, "rank": rank})
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(out_rob / "eta_sensitivity_ranks.csv", index=False)

    # Fraction of dates with R_raw < 0
    frac_neg = off.groupby("method")["R_raw"].apply(lambda s: float(np.mean(s < -1e-12)))
    frac_neg.to_csv(out_rob / "R_raw_negative_fraction.csv")
    print("\nFraction R_raw < 0:\n", frac_neg)

    eig_sum = eig.groupby("method")[
        ["logeig_mse_top", "logeig_mse_mid", "logeig_mse_bot", "inv_frob_sq", "cond_number_hat", "leverage_l1"]
    ].mean()
    eig_sum.to_csv(out_eig / "oos_eigenstructure_summary.csv")
    print("\n=== Eigenstructure (η=0.01) ===")
    print(eig_sum.sort_values("logeig_mse_bot").to_string())

    meta = {
        "frozen_model": "D0_pca8",
        "official_eta": 0.01,
        "eta_grid": list(ETA_GRID),
        "Lmax_gross": LMAX_GROSS,
        "note": "Robustness only; no architecture reselection on OOS.",
    }
    (out_rob / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_rob} and {out_eig}")


if __name__ == "__main__":
    main()
