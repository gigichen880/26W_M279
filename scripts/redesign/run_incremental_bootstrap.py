#!/usr/bin/env python3
"""Block-bootstrap Δ variance regret for Stage D increments (selection panels)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def block_bootstrap_delta(a: np.ndarray, b: np.ndarray, *, block: int = 20, n_boot: int = 5000, seed: int = 0):
    """Paired moving-block bootstrap of mean(a)-mean(b)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(x)
    if n < block + 2:
        return float(np.mean(x)), (np.nan, np.nan), np.nan
    n_blocks = int(np.ceil(n / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        samp = np.concatenate([x[s : s + block] for s in starts])[:n]
        means.append(float(np.mean(samp)))
    means = np.asarray(means)
    lo, hi = np.quantile(means, [0.025, 0.975])
    # two-sided p: fraction of boots on other side of 0 from observed
    obs = float(np.mean(x))
    p = 2 * min(np.mean(means >= 0), np.mean(means <= 0))
    return obs, (float(lo), float(hi)), float(p)


def load_method(tag: str, method: str) -> pd.Series:
    path = ROOT / "results" / "redesign" / tag / "panel.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    sub = df[df["method"] == method].sort_values("date")
    return sub.set_index("date")["var_regret"]


def main():
    pairs = [
        ("D5 − D0 (pca8 HMM)", "stage_d_D5_pca8_hmm3", "D5_pca8_hmm3", "stage_d_D0_pca8", "D0_pca8"),
        ("D2 − D0 (pca8 HMM)", "stage_d_D2_pca8_hmm3", "D2_pca8_hmm3", "stage_d_D0_pca8", "D0_pca8"),
        ("D2 − D0 (market FCM)", "stage_d_D2_market_fcm3", "D2_market_fcm3", "stage_d_D0_market", "D0_market"),
        ("D0_pca8 − LW", "stage_d_D0_pca8", "D0_pca8", "stage_d_D0_pca8", "ledoit_wolf"),
        ("D5 − LW", "stage_d_D5_pca8_hmm3", "D5_pca8_hmm3", "stage_d_D5_pca8_hmm3", "ledoit_wolf"),
    ]
    rows = []
    for name, t1, m1, t2, m2 in pairs:
        s1 = load_method(t1, m1)
        s2 = load_method(t2, m2)
        idx = s1.index.intersection(s2.index)
        a, b = s1.loc[idx].to_numpy(), s2.loc[idx].to_numpy()
        delta, (lo, hi), p = block_bootstrap_delta(a, b, block=20, n_boot=5000)
        supported = (hi < 0) if delta < 0 else (lo > 0)  # CI excludes 0 in direction of improvement if delta<0 means a better
        # improvement = lower regret: negative delta means first better
        rows.append(
            {
                "increment": name,
                "delta_var_regret": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p,
                "supports_first_better": bool(hi < 0),
                "n": len(idx),
            }
        )
        print(f"{name}: Δ={delta:.6g} CI=[{lo:.6g},{hi:.6g}] p={p:.3f} n={len(idx)}")

    out = pd.DataFrame(rows)
    outdir = ROOT / "results" / "redesign" / "stage_d"
    out.to_csv(outdir / "incremental_bootstrap.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
