"""
Moving-block bootstrap on the held-out daily Sharpe difference (model - baseline).

Tests whether the out-of-sample GMVP Sharpe gap between the model and each
baseline is statistically distinguishable from zero, accounting for the
autocorrelation induced by overlapping 20-day holding windows (block length =
horizon). Reports the point difference, a bootstrap 95% CI, and a two-sided
p-value for H0: difference = 0.

Note: this is an inference (CI/p-value) on already-computed returns; it does NOT
select or tune anything on the held-out period.

Usage:
    python -m scripts.analysis.core.oos_significance --tag-dir results/oos_final \
        --split 2017-01-01 --block 20 --n-boot 20000
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

ANN = 252.0


def _tranched(tag_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "build_equity_curves.py")
    spec = importlib.util.spec_from_file_location("bec", path)
    bec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bec)
    d = pd.read_parquet(os.path.join(tag_dir, "gmvp_daily_returns.parquet"))
    d["date"] = pd.to_datetime(d["date"])
    return bec.tranched_daily_returns(d)


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan


def block_boot_diff(a: np.ndarray, b: np.ndarray, block: int, n_boot: int, seed: int = 12345):
    """Moving-block bootstrap of Sharpe(a) - Sharpe(b) on paired daily returns."""
    rng = np.random.default_rng(seed)
    n = len(a)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        diffs[i] = _sharpe(a[idx]) - _sharpe(b[idx])
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", default="results/oos_final")
    ap.add_argument("--split", default="2017-01-01")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    tr = _tranched(args.tag_dir)
    oos = tr[tr.index >= args.split]
    print(f"Held-out {oos.index.min().date()}..{oos.index.max().date()}  n_days={len(oos)}  "
          f"block={args.block}  n_boot={args.n_boot}")
    model = oos["model"].values
    print(f"\nmodel daily Sharpe = {_sharpe(model):.4f}")
    for base in ["pers", "shrink", "roll", "mix"]:
        if base not in oos.columns:
            continue
        b = oos[base].values
        mask = np.isfinite(model) & np.isfinite(b)
        diffs = block_boot_diff(model[mask], b[mask], args.block, args.n_boot)
        pt = _sharpe(model[mask]) - _sharpe(b[mask])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # two-sided p for H0: diff=0
        p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        verdict = "DISTINGUISHABLE" if (lo > 0 or hi < 0) else "tie (CI contains 0)"
        print(f"  model - {base:6}: Δ={pt:+.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
