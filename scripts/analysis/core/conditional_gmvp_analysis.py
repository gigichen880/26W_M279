"""
Regime/condition-conditional GMVP performance on the honest daily harness.

Motivation (advisor suggestion, 2026-07-25): the model ties baselines overall
out-of-sample -- does it WIN in particular regimes or conditions?

Two strictly separated analyses:

1. DESCRIPTIVE (concurrent conditioning): slice daily tranched GMVP returns by
   what actually happened during the forecast window (future realized vol /
   avg corr from backtest.csv, crisis windows). Valid as characterization of
   WHEN the model helps; NOT tradeable (conditions on the future).

2. EX-ANTE (tradeable conditioning): condition only on information available at
   the anchor date -- filtered regime assignment, TRAILING cross-sectional vol
   and correlation (recomputed here from the returns matrix; note the
   realized_vol / avg_corr columns in backtest.csv are computed over the FUTURE
   horizon via compute_horizon_cross_sectional_stats(fut) and must NOT be used
   for this). Includes an honest hybrid test: switching/ensemble rules tuned on
   the tuning era only, evaluated held-out.

Usage:
  python -m scripts.analysis.core.conditional_gmvp_analysis \
      --tag-dir results/oos_final --split 2017-01-01
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from scripts.clean_data import clean_returns_matrix_at_load
from scripts.config_utils import load_yaml

ANN = 252.0
BLOCK = 20
METHODS = ["model", "mix", "shrink", "pers", "roll"]

CRISES = {
    "GFC 2008-09..2009-03": ("2008-09-01", "2009-03-31"),
    "US downgrade 2011": ("2011-08-01", "2011-12-31"),
    "2015-16 turbulence": ("2015-08-01", "2016-02-29"),
    "Volmageddon 2018-02": ("2018-02-01", "2018-03-31"),
    "Q4 2018 selloff": ("2018-10-01", "2018-12-31"),
    "COVID 2020-02..06": ("2020-02-01", "2020-06-30"),
}


def sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 40:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan")


def block_boot_p(a, b, n_boot=8000, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3 * BLOCK:
        return float("nan"), float("nan"), float("nan")
    nblocks = int(np.ceil(n / BLOCK))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n - BLOCK + 1, nblocks)
        idx = (st[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
        diffs[i] = sharpe(a[idx]) - sharpe(b[idx])
    p = float(2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return p, float(lo), float(hi)


def trailing_stats(past: np.ndarray) -> tuple[float, float]:
    """Same math as compute_horizon_cross_sectional_stats, on the PAST window."""
    col_std = np.nanstd(past, axis=0, ddof=1)
    vol = float(np.nanmean(col_std))
    c = pd.DataFrame(past).corr(numeric_only=True).values
    m = np.triu(np.ones(c.shape, dtype=bool), k=1)
    vals = c[m]
    vals = vals[np.isfinite(vals)]
    return vol, (float(np.mean(vals)) if vals.size else float("nan"))


def report(tr, mask, label, min_n=60):
    sl = tr[mask]
    if len(sl) < min_n:
        print(f"\n--- {label}: n={len(sl)} (skipped)")
        return
    stats = {m: sharpe(sl[m].dropna().values) for m in METHODS}
    p_pers, _, _ = block_boot_p(sl["model"].values, sl["pers"].values)
    print(f"\n--- {label} (n={len(sl)}, {sl.index.min().date()}..{sl.index.max().date()})")
    for m in METHODS:
        r = sl[m].dropna().values
        star = " <== best" if stats[m] == max(stats.values()) else ""
        print(f"    {m:7} sharpe={stats[m]:+.3f}  vol={r.std(ddof=1)*np.sqrt(ANN):.3f}  "
              f"tw={float(np.prod(1+r)):.3f}{star}")
    print(f"    model-pers block-bootstrap p={p_pers:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", default="results/oos_final")
    ap.add_argument("--split", default="2017-01-01")
    args = ap.parse_args()

    cfg = load_yaml(os.path.join(args.tag_dir, "config_used.yaml"))
    d = cfg["data"]

    md = pd.read_parquet(os.path.join(args.tag_dir, "gmvp_daily_returns.parquet"))
    md["date"] = pd.to_datetime(md["date"])
    bt = pd.read_csv(os.path.join(args.tag_dir, "backtest.csv"), parse_dates=["date"])
    with open(os.path.join(args.tag_dir, "regime_name_map.json")) as f:
        names = json.load(f)

    tr = md.groupby(["date", "method"])["ret"].mean().unstack()[METHODS].sort_index()
    OOS = pd.Timestamp(args.split)
    ins = np.asarray(tr.index < OOS)
    oos = np.asarray(tr.index >= OOS)

    # ---------- ex-ante conditions ----------
    rdf = clean_returns_matrix_at_load(
        parquet_path=d["parquet_path"], policy=d["policy"],
        q99_thresh=float(d["q99_thresh"]), max_thresh=float(d["max_thresh"]),
        min_non_nan_frac=float(d["min_non_nan_frac"]),
    ).T
    rdf.index = pd.to_datetime(rdf.index)
    rdf = rdf.sort_index()
    if d.get("start_date"):
        rdf = rdf.loc[pd.to_datetime(d["start_date"]):]
    if d.get("end_date"):
        rdf = rdf.loc[:pd.to_datetime(d["end_date"])]
    R = rdf.to_numpy(dtype=float)
    dates = rdf.index

    rows = []
    for _, r in bt.iterrows():
        a = int(r["raw_anchor"])
        if a >= len(dates) or dates[a] != r["date"] or a - 20 + 1 < 0:
            continue
        v, c = trailing_stats(R[a - 20 + 1: a + 1, :])
        rows.append({"date": r["date"], "tvol20": v, "tcorr20": c,
                     "dominant_regime": r["dominant_regime"]})
    anch = pd.DataFrame(rows).set_index("date").sort_index()
    cond = anch.reindex(tr.index.union(anch.index)).ffill().reindex(tr.index)

    print("=" * 72)
    print("PART 1: DESCRIPTIVE (concurrent conditions -- NOT tradeable)")
    print("=" * 72)
    report(tr, np.ones(len(tr), dtype=bool), "FULL sample")
    report(tr, oos, "HELD-OUT")
    crisis = np.zeros(len(tr), dtype=bool)
    for lbl, (lo, hi) in CRISES.items():
        m = np.asarray((tr.index >= lo) & (tr.index <= hi))
        crisis |= m
        report(tr, m, lbl, min_n=25)
    report(tr, crisis, "ALL crisis pooled")
    report(tr, ~crisis, "Non-crisis")

    print("\n" + "=" * 72)
    print("PART 2: EX-ANTE conditioning (tradeable information only)")
    print("=" * 72)
    for k in sorted(cond["dominant_regime"].dropna().unique()):
        m = np.asarray(cond["dominant_regime"] == k)
        report(tr, m & ins, f"TUNING regime {int(k)} [{names[str(int(k))]}]")
        report(tr, m & oos, f"HELD-OUT regime {int(k)} [{names[str(int(k))]}]")

    def gap(mask):
        sl = tr[mask]
        return sharpe(sl["model"].values) - sharpe(sl["pers"].values)

    # hybrid rules tuned strictly on the tuning era
    rules = {}
    gaps_in = {int(k): gap(np.asarray(cond.dominant_regime == k) & ins)
               for k in sorted(cond.dominant_regime.dropna().unique())}
    fav = [k for k, g in gaps_in.items() if g > 0]
    rules[f"regime in {fav}"] = np.asarray(cond.dominant_regime.isin(fav))
    for var in ["tvol20", "tcorr20"]:
        for qq in (1 / 3, 1 / 2, 2 / 3):
            thr = cond.loc[ins, var].quantile(qq)
            rules[f"{var}<=q{qq:.2f}"] = np.asarray(cond[var] <= thr)

    print("\nHybrid switch rules (model if rule true else pers), tuned on tuning era:")
    best_rule = max(rules, key=lambda n: sharpe(
        np.where(rules[n], tr["model"].values, tr["pers"].values)[ins]))
    for name, m in rules.items():
        hyb = np.where(m, tr["model"].values, tr["pers"].values)
        tag = "  <== chosen (tuning-era winner)" if name == best_rule else ""
        print(f"  {name:22} IN={sharpe(hyb[ins]):+.3f}  OOS={sharpe(hyb[oos]):+.3f}{tag}")
    hyb = np.where(rules[best_rule], tr["model"].values, tr["pers"].values)
    p, lo, hi = block_boot_p(hyb[oos], tr["pers"].values[oos])
    print(f"  Honest OOS test of chosen rule vs pers: "
          f"d={sharpe(hyb[oos]) - sharpe(tr['pers'].values[oos]):+.3f} p={p:.3f} CI[{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
