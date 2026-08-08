#!/usr/bin/env python3
"""Stage C: regime-model acceptance diagnostics on shortlisted representations.

Uses development fit (≤2012) and selection-window queries (2013–2016).
Rejects probabilistic models that are flat or unjustifiably near-deterministic.
Hard k-means is a control (max p=1 allowed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from similarity_forecast.core import validate_window
from similarity_forecast.redesign.embeddings import build_embedder
from similarity_forecast.redesign.geometry import membership_stats
from similarity_forecast.redesign.regimes import build_regime_model
from similarity_forecast.redesign.runner import load_returns


def collect(emb, R, dates, L, lo, hi, stride=5):
    Z = []
    for t in range(L, R.shape[0], stride):
        if dates[t] < pd.Timestamp(lo) or dates[t] > pd.Timestamp(hi):
            continue
        past = R[t - L : t]
        if not validate_window(past):
            continue
        try:
            Z.append(emb.embed(past))
        except Exception:
            continue
    return np.vstack(Z) if Z else np.zeros((0, 1))


def accept_probabilistic(mem: dict) -> tuple[bool, str]:
    mx = mem.get("mean_max_p", 0.0)
    h = mem.get("mean_norm_entropy", 1.0)
    if mx < 0.35 or h > 0.95:
        return False, "near_uniform"
    if mx > 0.98 and h < 0.05:
        return False, "near_deterministic_without_review"
    return True, "ok"


def main():
    outdir = ROOT / "results" / "redesign" / "stage_c_regime_diag"
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_returns(str(ROOT / "data/processed/returns_universe_100.parquet"))
    df = df.loc["2008-01-01":"2016-12-31"]
    R = df.to_numpy(dtype=float)
    dates = df.index
    L = 50
    fit_end = int(np.searchsorted(dates, pd.Timestamp("2012-12-31"), side="right") - 1)
    R_fit = R[: fit_end + 1]

    reps = [
        ("market_state", "market_state", 6),
        ("pca_only_8", "pca_only", 8),
    ]
    models = [
        ("fcm_k3", "fcm", {"n_states": 3}),
        ("fcm_k4", "fcm", {"n_states": 4}),
        ("gmm_diag_k3", "gmm", {"n_states": 3, "covariance_type": "diag"}),
        ("gmm_diag_k4", "gmm", {"n_states": 4, "covariance_type": "diag"}),
        ("kmeans_k4", "kmeans", {"n_states": 4}),
        ("gmm_transition_k3", "gmm_transition_forecast", {"n_states": 3}),
    ]
    # Try true HMM if available
    try:
        import hmmlearn  # noqa: F401
        models.append(("hmm_diag_k3", "hmm", {"n_states": 3, "covariance_type": "diag"}))
    except ImportError:
        pass

    rows = []
    for rtag, rname, k in reps:
        emb = build_embedder(rname, lookback=L, pca_k=k)
        if hasattr(emb, "fit"):
            try:
                emb.fit(R_fit, lookback=L)
            except TypeError:
                emb.fit(R_fit)
        Z_tr = collect(emb, R, dates, L, "2008-01-01", "2012-12-31")
        Z_q = collect(emb, R, dates, L, "2013-01-01", "2016-12-31")
        for mtag, mname, kw in models:
            print(f"[stage_c] {rtag} + {mtag}", flush=True)
            try:
                model = build_regime_model(mname, random_state=0, **kw)
                model.fit(Z_tr)
                P_tr = model.predict_proba(Z_tr)
                P_q = model.predict_proba(Z_q)
            except Exception as e:
                rows.append({"rep": rtag, "model": mtag, "error": str(e)})
                continue
            mem_tr = membership_stats(P_tr)
            mem_q = membership_stats(P_q)
            hard = mname in ("kmeans", "hard_kmeans")
            if hard:
                ok, reason = True, "hard_control"
            else:
                ok, reason = accept_probabilistic(mem_q)
            rows.append(
                {
                    "rep": rtag,
                    "model": mtag,
                    "accepted_query": ok,
                    "reason": reason,
                    **{f"train_{k}": v for k, v in mem_tr.items()},
                    **{f"query_{k}": v for k, v in mem_q.items()},
                }
            )

    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / "regime_acceptance.csv", index=False)
    print(tab[["rep", "model", "accepted_query", "reason", "query_mean_max_p", "query_mean_norm_entropy"]].to_string(index=False))
    accepted = tab[tab.get("accepted_query") == True]  # noqa: E712
    (outdir / "SHORTLIST.md").write_text(
        "# Stage C accepted probabilistic models (query 2013–2016)\n\n"
        + accepted.to_string(index=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
