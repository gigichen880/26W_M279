#!/usr/bin/env python3
"""Stage D: D0 vs D2 (and optional D5) on selection window only.

Uses Stage B/C shortlists. No 2017+ evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from similarity_forecast.redesign.conditioner import ConditioningConfig
from similarity_forecast.redesign.runner import ExperimentConfig, ProtocolConfig, run_experiment


def run_one(tag, **kwargs):
    cfg = ExperimentConfig(
        tag=tag,
        parquet_path=str(ROOT / "data/processed/returns_universe_100.parquet"),
        protocol=ProtocolConfig(),
        conditioning=ConditioningConfig(eta=kwargs.pop("eta", 0.01)),
        eval_split="select",
        outdir=str(ROOT / "results/redesign"),
        baselines=("rolling", "ledoit_wolf") if kwargs.get("with_baselines", True) else (),
        **{k: v for k, v in kwargs.items() if k != "with_baselines"},
    )
    print(f"[stage_d] {tag}", flush=True)
    out = run_experiment(cfg)
    s = out.groupby("method")[["var_regret", "gmvp_var", "cond_stein"]].mean()
    print(s.sort_values("var_regret"))
    return out


def main():
    # Primary comparison: pca8 D0 (best Stage A) vs market_state D2 with accepted FCM
    jobs = [
        dict(
            tag="stage_d_D0_pca8",
            embedder_name="pca_only",
            pca_k=8,
            k_neighbors=20,
            aggregation="log_euclidean",
            regime_mode="none",
            method_label="D0_pca8",
        ),
        dict(
            tag="stage_d_D2_market_fcm3",
            embedder_name="market_state",
            pca_k=6,
            k_neighbors=20,
            aggregation="log_euclidean",
            regime_mode="overlap",
            regime_name="fcm",
            regime_n_states=3,
            method_label="D2_market_fcm3",
        ),
        dict(
            tag="stage_d_D0_market",
            embedder_name="market_state",
            pca_k=6,
            k_neighbors=20,
            aggregation="log_euclidean",
            regime_mode="none",
            method_label="D0_market",
        ),
        dict(
            tag="stage_d_D2_pca8_hmm3",
            embedder_name="pca_only",
            pca_k=8,
            k_neighbors=20,
            aggregation="log_euclidean",
            regime_mode="overlap",
            regime_name="hmm",
            regime_n_states=3,
            method_label="D2_pca8_hmm3",
        ),
        dict(
            tag="stage_d_D5_pca8_hmm3",
            embedder_name="pca_only",
            pca_k=8,
            k_neighbors=20,
            aggregation="log_euclidean",
            regime_mode="overlap_pred",
            regime_name="hmm",
            regime_n_states=3,
            method_label="D5_pca8_hmm3",
        ),
    ]
    summaries = []
    for job in jobs:
        out = run_one(**job)
        m = job["method_label"]
        sub = out[out["method"] == m]
        if len(sub):
            summaries.append(
                {
                    "tag": job["tag"],
                    "method": m,
                    "var_regret": sub["var_regret"].mean(),
                    "gmvp_var": sub["gmvp_var"].mean(),
                    "cond_stein": sub["cond_stein"].mean(),
                    "n": len(sub),
                }
            )
    tab = pd.DataFrame(summaries).sort_values("var_regret")
    outdir = ROOT / "results" / "redesign" / "stage_d"
    outdir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(outdir / "comparison.csv", index=False)
    print("\n=== Stage D comparison (selection) ===")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
