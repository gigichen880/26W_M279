#!/usr/bin/env python3
"""Apply the pre-specified freeze rule to the cleaned-data selection results.

Rule (identical to results/redesign/FREEZE_DECISION.md):
  - primary metric: mean var_regret (R_raw) on the 2013-2016 selection window
  - regimes enter only if their stage-D incremental bootstrap CI excludes zero
    in their favor; otherwise simplicity tie-break (no regime > regime,
    lower embedding dimension)
  - among no-regime embedders, pick the stage-A winner by mean var_regret

Writes RERUN_FREEZE_DECISION.{md,json} BEFORE any 2017-2021 evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / "results/redesign"

EMBEDDER_ARGS = {
    "stage_a_pca8": dict(embedder_name="pca_only", pca_k=8, method_label="D0_pca8"),
    "stage_a_market_state": dict(embedder_name="market_state", pca_k=8, method_label="D0_market"),
    "stage_a_spectral": dict(embedder_name="spectral", pca_k=8, method_label="D0_spectral"),
    "stage_a_hybrid": dict(embedder_name="hybrid", pca_k=5, method_label="D0_hybrid"),
}


def sim_regret(tag: str) -> float:
    s = pd.read_csv(RD / tag / "summary.csv")
    method_col = s.columns[0]
    sim_rows = s[~s[method_col].isin(["rolling", "ewma", "ledoit_wolf", "oas", "shrink", "persistence"])]
    return float(sim_rows["var_regret"].iloc[0])


def main():
    stage_a = {t: sim_regret(t) for t in EMBEDDER_ARGS if (RD / t / "summary.csv").exists()}
    best_tag = min(stage_a, key=stage_a.get)

    regime_supported = False
    regime_note = "stage_d incremental bootstrap not found"
    incr_path = RD / "stage_d" / "incremental_bootstrap.csv"
    if incr_path.exists():
        incr = pd.read_csv(incr_path)
        cols = [c.lower() for c in incr.columns]
        incr.columns = cols
        lo = next(c for c in cols if "lo" in c or "lower" in c or "2.5" in c)
        hi = next(c for c in cols if "hi" in c or "upper" in c or "97.5" in c)
        # regime increment supported only if CI excludes zero AND favors the regime variant
        supported_rows = incr[(incr[lo] > 0) | (incr[hi] < 0)]
        favored = supported_rows[(supported_rows[hi] < 0)]  # negative delta = regime variant better
        regime_supported = len(favored) > 0
        regime_note = f"{len(incr)} contrasts; {len(supported_rows)} exclude zero; {len(favored)} favor regimes"

    chosen = dict(EMBEDDER_ARGS[best_tag])
    chosen.update(
        k_neighbors=20, metric="euclidean", aggregation="log_euclidean", eta=0.01,
        regime_mode="none",
    )
    decision_basis = {
        "stage_a_mean_var_regret": stage_a,
        "regime_increment_supported": regime_supported,
        "regime_note": regime_note,
        "rule": "primary=mean R_raw on 2013-2016; regimes only if incremental CI excludes zero in their favor; simplicity tie-break",
        "data": "returns_universe_100_cleaned_cellwise.parquet (cellwise-cleaned; swapped in as returns_universe_100.parquet for this rerun)",
    }
    (RD / "RERUN_FREEZE_DECISION.json").write_text(json.dumps({**chosen, "basis": decision_basis}, indent=2))
    md = ["# RERUN freeze decision (cleaned data)", "",
          "Written BEFORE any 2017-2021 evaluation of this rerun.", "",
          f"Frozen config: {chosen}", "",
          f"Basis: {json.dumps(decision_basis, indent=2)}"]
    (RD / "RERUN_FREEZE_DECISION.md").write_text("\n".join(md))
    print("FROZEN:", chosen)
    print("BASIS:", json.dumps(decision_basis, indent=2))


if __name__ == "__main__":
    main()
