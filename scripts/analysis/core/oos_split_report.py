"""
Report honest daily GMVP Sharpe split into tuning vs held-out periods.

Consumes the per-day artifact `gmvp_daily_returns.parquet`, builds overlap-
corrected (tranched) daily returns per method, and reports terminal wealth,
whole-sample annualized daily Sharpe, and annualized return for a tuning slice
and a held-out slice. Used for issue #1 (out-of-sample tuning split): tune
2008-2016, hold out 2017-2021.

Usage:
    python -m scripts.analysis.core.oos_split_report --tag-dir results/oos_final \
        --split 2017-01-01
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

ANN = 252.0
METHODS = ["model", "pers", "mix", "shrink", "roll"]


def _load_tranched(tag_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "build_equity_curves.py")
    spec = importlib.util.spec_from_file_location("bec", path)
    bec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bec)
    d = pd.read_parquet(os.path.join(tag_dir, "gmvp_daily_returns.parquet"))
    d["date"] = pd.to_datetime(d["date"])
    return bec.tranched_daily_returns(d)


def summarize(sl: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for meth in METHODS:
        if meth not in sl.columns:
            continue
        r = sl[meth].dropna().values
        if r.size == 0:
            continue
        tw = float(np.prod(1.0 + r))
        mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
        sh = mu / sd * np.sqrt(ANN) if sd > 0 else float("nan")
        ar = tw ** (ANN / r.size) - 1.0
        rows.append({"method": meth, "terminal_wealth": tw, "daily_sharpe": sh,
                     "ann_return": ar, "n_days": r.size})
    df = pd.DataFrame(rows).set_index("method")
    lo, hi = sl.index.min().date(), sl.index.max().date()
    print(f"\n=== {label}  ({lo}..{hi}) ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", default="results/oos_final")
    ap.add_argument("--split", default="2017-01-01", help="first held-out date (inclusive)")
    args = ap.parse_args()
    tr = _load_tranched(args.tag_dir)
    tune = tr[tr.index < args.split]
    oos = tr[tr.index >= args.split]
    summarize(tune, "TUNING (in-sample)")
    summarize(oos, "HELD-OUT (TRUE OUT-OF-SAMPLE)")
    summarize(tr, "FULL SAMPLE")


if __name__ == "__main__":
    main()
