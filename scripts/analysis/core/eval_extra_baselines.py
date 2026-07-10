"""
Evaluate REAL covariance baselines (Ledoit-Wolf, OAS, EWMA) on the same
walk-forward harness as an existing backtest run, faithfully and post-hoc.

Motivation (issue #6): the built-in "shrink" baseline is a fixed gamma=0.3 blend
toward the diagonal -- a strawman versus the Ledoit-Wolf estimator the paper
cites. This script re-uses the harness's own anchors, data loading, imputation,
SPD floor (apply_floor_to=gmvp_only), GMVP weights, renormalized daily returns,
and matrix-loss metrics, so the new baselines are directly comparable to the
model and the existing baselines already in <tag-dir>/backtest.csv.

It does NOT re-run or tune the model; it adds comparators on fixed forecasts.

Outputs, per tuning / held-out / full slice:
  - mean matrix losses (fro, stein, kl, logeuc)
  - GMVP daily tranched Sharpe + terminal wealth
  - block-bootstrap significance of (model - baseline) held-out Sharpe

Usage (scai4):
  PYTHONNOUSERSITE=1 ~/m279iso/bin/python -m scripts.analysis.core.eval_extra_baselines \
      --tag-dir results/oos_final --split 2017-01-01
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

from scripts.config_utils import load_yaml
from scripts.clean_data import clean_returns_matrix_at_load
from similarity_forecast.core import project_to_spd
from similarity_forecast.target_objects import CovarianceTarget
from similarity_forecast.backtests import (
    eval_all_metrics, gmvp_weights, gmvp_daily_returns_renorm,
    baseline_ledoit_wolf, baseline_oas, baseline_ewma_cov, baseline_shrink_to_diag,
    baseline_rolling_cov,
)

ANN = 252.0
NEW_BASELINES = {
    "lw": lambda past, ddof: baseline_ledoit_wolf(past, ddof=ddof),
    "oas": lambda past, ddof: baseline_oas(past, ddof=ddof),
    "ewma94": lambda past, ddof: baseline_ewma_cov(past, lam=0.94, ddof=ddof),
}


def _spd_floor(S, eps, proj_eps=1e-8):
    eps = float(eps)
    if eps <= 0:
        return S
    S2 = (S + S.T) / 2.0 + eps * np.eye(S.shape[0])
    return project_to_spd(S2, eps=proj_eps)


def _sharpe(r):
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan


def _import(mod_path, name):
    spec = importlib.util.spec_from_file_location(name, mod_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", default="results/oos_final")
    ap.add_argument("--split", default="2017-01-01")
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    bec = _import(os.path.join(here, "build_equity_curves.py"), "bec")
    sig = _import(os.path.join(here, "oos_significance.py"), "sig")

    cfg = load_yaml(os.path.join(args.tag_dir, "config_used.yaml"))
    d, m, st = cfg["data"], cfg["model"], cfg["stability"]
    L, H, ddof = int(m["lookback"]), int(m["horizon"]), int(m["ddof"])
    floor_eps = float(st["floor_eps"])

    # Identical data loading + date slice to run_backtest, so raw_anchor aligns.
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

    bt = pd.read_csv(os.path.join(args.tag_dir, "backtest.csv"))
    bt["date"] = pd.to_datetime(bt["date"])
    anchors = bt[["date", "raw_anchor"]].dropna().astype({"raw_anchor": int})

    tgt = CovarianceTarget(ddof=ddof)
    daily_rows, mat_rows = [], []
    n_align_fail = 0
    for _, row in anchors.iterrows():
        a = int(row["raw_anchor"])
        if a < L or a + H + 1 > len(R):
            continue
        if dates[a] != row["date"]:        # alignment guard
            n_align_fail += 1
            continue
        past = R[a - L + 1: a + 1, :]
        fut = R[a + 1: a + H + 1, :]
        try:
            Sigma_true = tgt.target(fut)
        except Exception:
            continue
        fut_dates = dates[a + 1: a + H + 1]
        for name, fn in NEW_BASELINES.items():
            try:
                S = fn(past, ddof)
            except Exception:
                continue
            mm = eval_all_metrics(S, Sigma_true, fut, long_only=False)  # unfloored (gmvp_only)
            mat_rows.append({"date": row["date"], "method": name,
                             **{k: mm.get(k) for k in ("fro", "stein", "kl", "logeuc")}})
            w = gmvp_weights(_spd_floor(S, floor_eps))
            rp = gmvp_daily_returns_renorm(fut, w)
            for j in range(len(fut_dates)):
                daily_rows.append({"date": fut_dates[j], "raw_anchor": a,
                                   "method": name, "ret": float(rp[j])})
    if n_align_fail:
        print(f"[warn] {n_align_fail} anchors failed date alignment and were skipped")

    new_daily = pd.DataFrame(daily_rows)
    new_daily["date"] = pd.to_datetime(new_daily["date"])
    new_tr = bec.tranched_daily_returns(new_daily)

    # model daily returns from the run's own artifact
    md = pd.read_parquet(os.path.join(args.tag_dir, "gmvp_daily_returns.parquet"))
    md["date"] = pd.to_datetime(md["date"])
    model_tr = bec.tranched_daily_returns(md)[["model", "shrink", "pers", "roll"]]
    tr = model_tr.join(new_tr, how="outer").sort_index()

    mat = pd.DataFrame(mat_rows)

    def report(lo_hi, label):
        lo, hi = lo_hi
        sl = tr[(tr.index >= lo) & (tr.index <= hi)]
        print(f"\n===== {label}  ({sl.index.min().date()}..{sl.index.max().date()}, n={len(sl)}) =====")
        print("  GMVP daily Sharpe / terminal wealth:")
        for meth in ["model", "lw", "oas", "ewma94", "shrink", "pers", "roll"]:
            if meth not in sl.columns:
                continue
            r = sl[meth].dropna().values
            if r.size == 0:
                continue
            print(f"    {meth:8} sharpe={_sharpe(r):+.4f}  tw={np.prod(1+r):.3f}")
        ms = mat[(mat.date >= lo) & (mat.date <= hi)]
        if len(ms):
            print("  Matrix losses (mean, new baselines):")
            g = ms.groupby("method")[["fro", "stein", "kl", "logeuc"]].mean()
            print(g.to_string(float_format=lambda x: f"{x:.4f}"))

    full = (tr.index.min(), tr.index.max())
    report((full[0], pd.to_datetime(args.split) - pd.Timedelta(days=1)), "TUNING (in-sample)")
    report((pd.to_datetime(args.split), full[1]), "HELD-OUT (out-of-sample)")

    # Significance: model vs each new baseline, held-out
    oos = tr[tr.index >= args.split]
    print("\n===== Held-out significance: model - baseline (block=20) =====")
    model = oos["model"].values
    for base in ["lw", "oas", "ewma94", "shrink", "pers"]:
        if base not in oos.columns:
            continue
        b = oos[base].values
        mask = np.isfinite(model) & np.isfinite(b)
        diffs = sig.block_boot_diff(model[mask], b[mask], 20, args.n_boot)
        pt = _sharpe(model[mask]) - _sharpe(b[mask])
        p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        loq, hiq = np.percentile(diffs, [2.5, 97.5])
        verdict = "distinguishable" if (loq > 0 or hiq < 0) else "tie"
        print(f"  model - {base:7}: Δ={pt:+.4f} CI[{loq:+.3f},{hiq:+.3f}] p={p:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
