#!/usr/bin/env python3
"""Stage A: similarity-only vs classical baselines on the selection window (2013–2016).

Primary metric: mean conditioned GMVP variance regret.
No 2017–2021 evaluation here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from similarity_forecast.redesign.conditioner import ConditioningConfig
from similarity_forecast.redesign.runner import ExperimentConfig, ProtocolConfig, run_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", default="market_state")
    p.add_argument("--pca-k", type=int, default=8)
    p.add_argument("--k-neighbors", type=int, default=20)
    p.add_argument("--aggregation", default="log_euclidean", choices=["log_euclidean", "arithmetic"])
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan"])
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    tag = args.tag or f"stage_a_{args.embedder}_k{args.k_neighbors}_{args.aggregation}_eta{args.eta}"
    cfg = ExperimentConfig(
        tag=tag,
        parquet_path=str(ROOT / "data/processed/returns_universe_100.parquet"),
        protocol=ProtocolConfig(),
        embedder_name=args.embedder,
        pca_k=args.pca_k,
        k_neighbors=args.k_neighbors,
        metric=args.metric,
        aggregation=args.aggregation,
        conditioning=ConditioningConfig(eta=args.eta),
        eval_split="select",
        outdir=str(ROOT / "results/redesign"),
    )
    print(f"[stage_a] running {tag} on selection window 2013–2016 ...", flush=True)
    out = run_experiment(cfg)
    print(out.groupby("method")[["var_regret", "gmvp_var", "cond_stein"]].mean().sort_values("var_regret"))
    print(f"n_rows={len(out)} -> results/redesign/{tag}/")


if __name__ == "__main__":
    main()
