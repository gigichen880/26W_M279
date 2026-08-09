#!/usr/bin/env python3
"""Frozen OOS test run for the cleaned-data rerun.

Reads the freeze decision produced by freeze_decide.py (RERUN_FREEZE_DECISION.json)
and runs the frozen configuration once on the 2017-2021 test split.
Mirrors the committed oos_frozen_D0_pca8/config.json field-for-field except data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from similarity_forecast.redesign.conditioner import ConditioningConfig
from similarity_forecast.redesign.runner import ExperimentConfig, ProtocolConfig, run_experiment


def main():
    freeze = json.loads((ROOT / "results/redesign/RERUN_FREEZE_DECISION.json").read_text())
    cfg = ExperimentConfig(
        tag=f"oos_frozen_{freeze['method_label']}",
        parquet_path=str(ROOT / "data/processed/returns_universe_100.parquet"),
        protocol=ProtocolConfig(),
        embedder_name=freeze["embedder_name"],
        pca_k=freeze["pca_k"],
        k_neighbors=freeze["k_neighbors"],
        metric=freeze["metric"],
        aggregation=freeze["aggregation"],
        conditioning=ConditioningConfig(eta=freeze["eta"]),
        baselines=("rolling", "ewma", "ledoit_wolf", "oas", "shrink"),
        eval_split="test",
        outdir=str(ROOT / "results/redesign"),
        regime_mode=freeze.get("regime_mode", "none"),
        method_label=freeze["method_label"],
    )
    print(f"[frozen_oos] running {cfg.tag} on 2017-2021 test split ...", flush=True)
    out = run_experiment(cfg)
    print(out.groupby("method")[["var_regret", "gmvp_var", "cond_stein"]].mean().sort_values("var_regret"))
    print(f"n_rows={len(out)}")


if __name__ == "__main__":
    main()
