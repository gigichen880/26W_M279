"""
Autocorrelation-robust significance via moving-block bootstrap (report 2.4 / B2).

The paper's Wilcoxon test operates on per-date paired differences
d_t = model_metric_t - baseline_metric_t. Those differences are autocorrelated
(overlapping rolling windows, 20-day horizon), so the Wilcoxon p-values are
optimistic. This script recomputes significance with a moving-block bootstrap,
which preserves short-range dependence, and reports p-values + 95% CIs
alongside the Wilcoxon p for comparison.

Run:
    .venv/bin/python -m scripts.analysis.core.block_bootstrap_significance
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKTEST = REPO_ROOT / "results" / "regime_covariance" / "backtest.csv"

SEED = 20260630
B = 20000                    # bootstrap resamples
BLOCK = 20                   # block length = forecast horizon h (near-independence gap)
BASELINES = ["roll", "shrink", "pers"]
# (metric, higher_is_better)
METRICS = [("gmvp_sharpe", True), ("fro", False)]


def moving_block_bootstrap_mean(d, block, n_boot, rng):
    """Distribution of the resample mean under moving-block bootstrap."""
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = d[idx].mean()
    return means


def analyze():
    df = pd.read_csv(BACKTEST)
    rng = np.random.default_rng(SEED)
    rows = []
    for metric, higher_better in METRICS:
        rcol = f"model_{metric}"
        for base in BASELINES:
            bcol = f"{base}_{metric}"
            sub = df[[rcol, bcol]].dropna()
            raw = sub[rcol].values - sub[bcol].values
            d = raw if higher_better else -raw     # positive = model better
            n = len(d)
            # Wilcoxon (matches paper)
            try:
                _, wp = stats.wilcoxon(d)
            except ValueError:
                wp = np.nan
            # moving-block bootstrap of the mean advantage
            boot = moving_block_bootstrap_mean(d, BLOCK, B, rng)
            # standard percentile-bootstrap two-sided p for H0: mean adv = 0
            p_boot = 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))
            p_boot = min(1.0, p_boot)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({
                "metric": metric, "baseline": base, "n": n,
                "mean_adv": round(float(d.mean()), 4),
                "median_adv": round(float(np.median(d)), 4),
                "wilcoxon_p": float(wp),
                "block_boot_p": round(float(p_boot), 4),
                "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
            })
    return pd.DataFrame(rows)


def main():
    out = analyze()
    pd.set_option("display.width", 160)
    print(f"Moving-block bootstrap: B={B}, block={BLOCK} days, seed={SEED}\n")
    print(out.to_string(index=False))
    outdir = REPO_ROOT / "results" / "regime_covariance" / "significance"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "block_bootstrap_significance.csv", index=False)
    print(f"\nSaved -> {outdir/'block_bootstrap_significance.csv'}")


if __name__ == "__main__":
    main()
