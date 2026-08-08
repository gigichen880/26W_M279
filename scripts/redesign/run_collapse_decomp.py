#!/usr/bin/env python3
"""PC/SVD collapse decomposition with train vs query FCM diagnostics.

Compares:
  - pca_svd_legacy (D=240 unscaled SVD block)
  - pca_only k=48
  - pca_svd_standardized (D=240 jointly standardized)
  - pca_only k in {15,10,5}
  - market_state (6D)

Reports geometry + train/query membership stats. Uses methodology-development
window through 2012 for fit; queries from 2013–2016 (selection window) only
for predict diagnostics — no 2017+ peeking.
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
from similarity_forecast.redesign.geometry import (
    centroid_distance_contrast,
    fit_fcm_predict,
    membership_stats,
    pairwise_distance_stats,
)
from similarity_forecast.redesign.runner import load_returns


def collect_states(emb, R, dates, lookback, date_lo, date_hi, stride=5):
    Z, idxs = [], []
    for t in range(lookback, R.shape[0], stride):
        d = dates[t]
        if d < pd.Timestamp(date_lo) or d > pd.Timestamp(date_hi):
            continue
        past = R[t - lookback : t]
        if not validate_window(past):
            continue
        try:
            Z.append(emb.embed(past))
            idxs.append(t)
        except Exception:
            continue
    if not Z:
        return np.zeros((0, 1)), np.asarray([], dtype=int)
    return np.vstack(Z), np.asarray(idxs, dtype=int)


def main():
    outdir = ROOT / "results" / "redesign" / "collapse_decomp"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_returns(str(ROOT / "data/processed/returns_universe_100.parquet"))
    # Restrict load window for speed: through end of selection sample
    df = df.loc["2008-01-01":"2016-12-31"]
    R = df.to_numpy(dtype=float)
    dates = df.index
    L = 50

    # Fit on methodology-development sample only
    fit_end = int(np.searchsorted(dates, pd.Timestamp("2012-12-31"), side="right") - 1)
    fit_end = max(fit_end, L + 100)
    R_fit = R[: fit_end + 1]

    specs = [
        ("pca_svd_legacy_48", "pca_svd_legacy", 48),
        ("pca_only_48", "pca_only", 48),
        ("pca_svd_standardized_48", "pca_svd_standardized", 48),
        ("pca_only_15", "pca_only", 15),
        ("pca_only_10", "pca_only", 10),
        ("pca_only_5", "pca_only", 5),
        ("market_state_6", "market_state", 6),
    ]

    rows = []
    for tag, name, k in specs:
        print(f"[collapse] fitting {tag} ...", flush=True)
        emb = build_embedder(name, lookback=L, pca_k=k)
        if hasattr(emb, "fit"):
            try:
                emb.fit(R_fit, lookback=L)  # type: ignore
            except TypeError:
                emb.fit(R_fit)  # type: ignore

        Z_tr, _ = collect_states(emb, R, dates, L, "2008-01-01", "2012-12-31", stride=5)
        Z_q, _ = collect_states(emb, R, dates, L, "2013-01-01", "2016-12-31", stride=5)
        print(f"  train={Z_tr.shape} query={Z_q.shape} dim={Z_tr.shape[1] if Z_tr.size else '?'}", flush=True)

        geom_tr = pairwise_distance_stats(Z_tr) if Z_tr.size else {}
        geom_q = pairwise_distance_stats(Z_q) if Z_q.size else {}

        mem_tr = mem_q = cent_tr = cent_q = {}
        if Z_tr.shape[0] >= 40:
            PI_tr, PI_q, centers = fit_fcm_predict(Z_tr, Z_q, n_regimes=4, random_state=0)
            mem_tr = membership_stats(PI_tr)
            if PI_q.shape[0]:
                mem_q = membership_stats(PI_q)
                cent_q = centroid_distance_contrast(Z_q, centers)
            cent_tr = centroid_distance_contrast(Z_tr, centers)

        row = {
            "tag": tag,
            "embedder": name,
            "pca_k": k,
            "dim": int(Z_tr.shape[1]) if Z_tr.size else 0,
            "n_train": int(Z_tr.shape[0]),
            "n_query": int(Z_q.shape[0]),
            **{f"geom_train_{k}": v for k, v in geom_tr.items()},
            **{f"geom_query_{k}": v for k, v in geom_q.items()},
            **{f"fcm_train_{k}": v for k, v in mem_tr.items()},
            **{f"fcm_query_{k}": v for k, v in mem_q.items()},
            **{f"cent_train_{k}": v for k, v in cent_tr.items()},
            **{f"cent_query_{k}": v for k, v in cent_q.items()},
        }
        rows.append(row)
        print(
            f"  train max_p={mem_tr.get('mean_max_p', float('nan')):.3f} "
            f"query max_p={mem_q.get('mean_max_p', float('nan')):.3f} "
            f"query rel_contrast={geom_q.get('rel_contrast', float('nan')):.3f}",
            flush=True,
        )

    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / "collapse_decomp.csv", index=False)
    # Compact view
    cols = [
        "tag", "dim",
        "geom_train_cv", "geom_query_cv",
        "geom_train_rel_contrast", "geom_query_rel_contrast",
        "fcm_train_mean_max_p", "fcm_query_mean_max_p",
        "fcm_train_mean_norm_entropy", "fcm_query_mean_norm_entropy",
        "cent_train_mean_centroid_rel_contrast", "cent_query_mean_centroid_rel_contrast",
    ]
    cols = [c for c in cols if c in tab.columns]
    compact = tab[cols]
    compact.to_csv(outdir / "collapse_decomp_compact.csv", index=False)
    (outdir / "README.md").write_text(
        "# Collapse decomposition\n\n"
        "Train fit: 2008–2012. Query predict: 2013–2016. No 2017+ used.\n\n"
        "Memberships are from `cmeans_predict` for both train and query "
        "(same path as walk-forward queries).\n\n"
        "```\n" + compact.to_string(index=False) + "\n```\n"
    )
    print(compact.to_string(index=False))
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
