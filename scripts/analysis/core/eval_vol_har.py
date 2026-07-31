"""
Add a HAR baseline to the volatility-forecast comparison, faithfully and post-hoc.

Issue #6 (volatility target): the paper's realized-vol section cites Corsi (2009)
HAR but never compares to it. HAR is THE standard realized-vol baseline. This
script fits a pooled HAR(daily/weekly/monthly) on |returns|, walk-forward
(refit every `refit` days on an expanding history, no look-ahead), predicts the
H-day-ahead per-asset log realized vol, and scores it with the harness's own
`eval_vol_metrics` against the harness's own target
(`realized_log_vol_from_returns`). It then prints held-out model-vs-HAR-vs-
existing-baseline metrics read from <tag-dir>/backtest.csv.

Usage (scai4):
  PYTHONNOUSERSITE=1 ~/m279iso/bin/python -m scripts.analysis.core.eval_vol_har \
      --tag-dir results/vol_oos --split 2017-01-01
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from scripts.config_utils import load_yaml
from scripts.clean_data import clean_returns_matrix_at_load
from similarity_forecast.backtests import realized_log_vol_from_returns, eval_vol_metrics


def _load_returns(cfg):
    d = cfg["data"]
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
    return rdf.to_numpy(dtype=float), rdf.index


def _har_features(absr, t):
    """[1, daily, weekly(5), monthly(22)] mean-abs-return features at day t (per asset)."""
    vd = absr[t]
    vw = np.nanmean(absr[max(0, t - 4): t + 1], axis=0)
    vm = np.nanmean(absr[max(0, t - 21): t + 1], axis=0)
    return vd, vw, vm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", default="results/vol_oos")
    ap.add_argument("--split", default="2017-01-01")
    ap.add_argument("--refit", type=int, default=20, help="HAR refit cadence (days)")
    args = ap.parse_args()

    cfg = load_yaml(os.path.join(args.tag_dir, "config_used.yaml"))
    L = int(cfg["model"]["lookback"])
    H = int(cfg["model"]["horizon"])
    ddof = int(cfg["model"]["ddof"])
    R, dates = _load_returns(cfg)
    absr = np.abs(R)
    T = len(R)

    bt = pd.read_csv(os.path.join(args.tag_dir, "backtest.csv"))
    bt["date"] = pd.to_datetime(bt["date"])
    anchors = bt[["date", "raw_anchor"]].dropna().astype({"raw_anchor": int})

    # Precompute forward realized log-vol target for every usable day (harness convention).
    fwd = {}
    def target_at(t):
        if t not in fwd:
            try:
                fwd[t] = realized_log_vol_from_returns(R[t + 1: t + H + 1], ddof=ddof)
            except Exception:
                fwd[t] = None
        return fwd[t]

    # Walk-forward HAR: refit every `refit` days on expanding history whose targets
    # end at or before the anchor (no look-ahead: target uses [tau+1, tau+H]).
    rows = []
    beta = None
    last_fit = -10 ** 9
    train_days = list(range(22, T))
    for _, row in anchors.iterrows():
        a = int(row["raw_anchor"])
        if a >= T or dates[a] != row["date"]:
            continue
        if beta is None or (a - last_fit) >= args.refit:
            X, y = [], []
            for tau in train_days:
                if tau > a - H - 1:
                    break
                yt = target_at(tau)
                if yt is None:
                    continue
                vd, vw, vm = _har_features(absr, tau)
                feat = np.column_stack([np.ones_like(vd), vd, vw, vm])
                m = np.isfinite(feat).all(1) & np.isfinite(yt)
                if m.any():
                    X.append(feat[m]); y.append(yt[m])
            if X:
                X = np.vstack(X); y = np.concatenate(y)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                last_fit = a
        if beta is None:
            continue
        vd, vw, vm = _har_features(absr, a)
        pred = beta[0] + beta[1] * vd + beta[2] * vw + beta[3] * vm
        vt = target_at(a)
        if vt is None:
            continue
        mm = eval_vol_metrics(pred, vt)
        rows.append({"date": row["date"], **{f"har_{k}": v for k, v in mm.items()}})

    har = pd.DataFrame(rows)
    har["date"] = pd.to_datetime(har["date"])
    merged = bt.merge(har, on="date", how="left")

    def _pooled_r2(mse: pd.Series, r2: pd.Series) -> float:
        """SST-weighted / pooled R² from per-anchor MSE and R² (equal n per anchor).

        Per-anchor: R2 = 1 - SSE/SST with SSE = n*MSE. Recover SST ∝ MSE/(1-R2),
        then pooled R2 = 1 - sum(MSE) / sum(MSE/(1-R2)). See docs/VOL_METRIC_AUDIT.md.
        """
        m = mse.to_numpy(dtype=float)
        r = r2.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sst = m / (1.0 - r)
        ok = np.isfinite(m) & np.isfinite(sst) & (sst > 0)
        if not ok.any():
            return float("nan")
        return float(1.0 - m[ok].sum() / sst[ok].sum())

    def slice_report(df, label):
        # Common sample: every method on anchors where HAR is finite (avoids
        # comparing HAR's nan-skipped mean to baselines averaged over more dates).
        if "har_vol_mse" in df.columns:
            df = df.loc[df["har_vol_mse"].notna()].copy()
        print(f"\n===== {label}  (common-sample n={len(df)}) =====")
        methods = {"model": "model", "har": "har", "roll": "roll", "pers": "pers", "shrink": "shrink"}
        hdr = (
            f"{'method':8}{'vol_mse':>10}{'r2_mean':>10}{'r2_pooled':>12}{'vol_qlike':>12}"
        )
        print(hdr)
        print(
            "# r2_mean = unweighted mean of per-anchor R² (can disagree with MSE rank); "
            "# r2_pooled = SST-weighted / pooled R² (consistent with MSE rank)."
        )
        for label_m, pref in methods.items():
            mse = df.get(f"{pref}_vol_mse")
            r2 = df.get(f"{pref}_vol_r2")
            ql = df.get(f"{pref}_vol_qlike")
            if mse is None or mse.notna().sum() == 0:
                continue
            r2_mean = float(r2.mean()) if r2 is not None else float("nan")
            r2_pool = _pooled_r2(mse, r2) if r2 is not None else float("nan")
            ql_mean = float(ql.mean()) if ql is not None else float("nan")
            print(
                f"{label_m:8}{float(mse.mean()):10.4f}{r2_mean:10.4f}"
                f"{r2_pool:12.4f}{ql_mean:12.4f}"
            )

    split = pd.to_datetime(args.split)
    slice_report(merged[merged.date < split], "TUNING (in-sample)")
    slice_report(merged[merged.date >= split], "HELD-OUT (out-of-sample)")


if __name__ == "__main__":
    main()
