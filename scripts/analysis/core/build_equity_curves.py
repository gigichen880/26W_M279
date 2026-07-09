"""
Honest GMVP equity curves and whole-sample Sharpe.

The naive curve `cumprod(1 + gmvp_cumret)` over evaluation rows double-counts
returns: each row's `gmvp_cumret` spans the full H-day holding window, but rows
are spaced `stride` (< H) trading days apart, so every calendar period is
compounded ~H/stride times. This module builds curves that do not double-count:

  * tranched  -- overlapping-portfolio accounting. At each calendar date, average
                 the daily returns of all *active* H-day sleeves (anchors). This
                 is a 1/overlap-capital blend of staggered GMVP sleeves and uses
                 every anchor. Requires the per-day artifact `gmvp_daily_returns`.
  * nonoverlap -- pick a disjoint set of anchors >= H trading days apart and
                 concatenate their daily returns. Uses ~stride/H of the data but
                 needs no overlap modelling. Available from either the per-day
                 artifact or the per-anchor `*_gmvp_cumret` columns.

Whole-sample annualized Sharpe is computed from the resulting daily return series
(mu/sd * sqrt(252)) -- a standard strategy Sharpe, not a mean of per-window
Sharpes.

Usage:
    python -m scripts.analysis.core.build_equity_curves --tag-dir results/regime_covariance
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

METHOD_ORDER = ["model", "mix", "roll", "shrink", "pers"]
ANN = 252.0


def _read_horizon(tag_dir: str, default: int = 20) -> int:
    """Read forecast horizon H from config_used.yaml if present."""
    path = os.path.join(tag_dir, "config_used.yaml")
    if not os.path.exists(path):
        return default
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return int(cfg["model"]["horizon"])
    except Exception:
        return default


def tranched_daily_returns(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Overlapping-portfolio daily returns: mean over active sleeves per date."""
    out = {}
    for meth, g in daily_df.groupby("method"):
        out[meth] = g.groupby("date")["ret"].mean().sort_index()
    cols = [m for m in METHOD_ORDER if m in out] + [m for m in out if m not in METHOD_ORDER]
    return pd.DataFrame({m: out[m] for m in cols})


def _disjoint_anchors(anchors: list[int], horizon: int) -> list[int]:
    chosen, last = [], None
    for a in sorted(anchors):
        if last is None or a - last >= horizon:
            chosen.append(a)
            last = a
    return chosen


def nonoverlap_daily_returns(daily_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Concatenate daily returns from a disjoint (>= H apart) set of anchors."""
    out = {}
    for meth, g in daily_df.groupby("method"):
        chosen = _disjoint_anchors(list(g["raw_anchor"].unique()), horizon)
        sub = g[g["raw_anchor"].isin(chosen)].sort_values(["raw_anchor", "day_offset"])
        s = sub.set_index("date")["ret"]
        out[meth] = s[~s.index.duplicated(keep="first")].sort_index()
    cols = [m for m in METHOD_ORDER if m in out] + [m for m in out if m not in METHOD_ORDER]
    return pd.DataFrame({m: out[m] for m in cols})


def nonoverlap_from_cumret(bt_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fallback: stitch non-overlapping H-day cumrets when no per-day artifact exists.

    Returns a frame whose 'daily returns' are actually per-window compounded
    returns spaced H apart; terminal wealth is exact, but the Sharpe from this
    frame is a window-level Sharpe (annualized over H-day blocks), so callers
    should prefer the per-day artifact when available.
    """
    if "raw_anchor" not in bt_df.columns:
        raise ValueError("backtest frame lacks 'raw_anchor'; cannot stitch non-overlapping windows.")
    out = {}
    for m in METHOD_ORDER:
        col = f"{m}_gmvp_cumret"
        if col not in bt_df.columns:
            continue
        sub = bt_df[[col, "raw_anchor"]].dropna(subset=[col]).copy()
        sub = sub.sort_values("raw_anchor")
        chosen = _disjoint_anchors(list(sub["raw_anchor"].unique()), horizon)
        picked = sub[sub["raw_anchor"].isin(chosen)]
        out[m] = pd.Series(picked[col].values, index=picked.index)
    return pd.DataFrame(out)


def summarize(daily_ret_df: pd.DataFrame, per_period_days: float = 1.0) -> pd.DataFrame:
    """Terminal wealth and annualized Sharpe/return from a return series frame.

    per_period_days: trading days per row (1 for a daily series; H for a
    non-overlapping window-return series). Used to annualize correctly.
    """
    rows = []
    for m in daily_ret_df.columns:
        r = daily_ret_df[m].dropna().values
        if r.size == 0:
            continue
        tw = float(np.prod(1.0 + r))
        mu, sd = float(np.mean(r)), float(np.std(r, ddof=1)) if r.size > 1 else np.nan
        periods_per_year = ANN / per_period_days
        sharpe = float(mu / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else np.nan
        total_days = r.size * per_period_days
        ann_ret = float(tw ** (ANN / total_days) - 1.0) if total_days > 0 else np.nan
        rows.append({
            "method": m, "terminal_wealth": tw, "ann_sharpe": sharpe,
            "ann_return": ann_ret, "n_periods": r.size,
        })
    return pd.DataFrame(rows).set_index("method")


def plot_curves(daily_ret_df: pd.DataFrame, out_png: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    for m in daily_ret_df.columns:
        r = daily_ret_df[m].fillna(0.0)
        plt.plot(r.index, np.cumprod(1.0 + r.values), label=m)
    plt.title(title)
    plt.ylabel("Cumulative wealth (start = 1)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def build(tag_dir: str) -> dict:
    horizon = _read_horizon(tag_dir)
    daily_path = os.path.join(tag_dir, "gmvp_daily_returns.parquet")
    figs_dir = os.path.join(tag_dir, "figs", "raw_temporal")
    summaries = {}

    if os.path.exists(daily_path):
        daily_df = pd.read_parquet(daily_path)
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        tr = tranched_daily_returns(daily_df)
        no = nonoverlap_daily_returns(daily_df, horizon)
        summaries["tranched"] = summarize(tr, per_period_days=1.0)
        summaries["nonoverlap"] = summarize(no, per_period_days=1.0)
        plot_curves(tr, os.path.join(figs_dir, "equity_curves_gmvp.png"),
                    "GMVP equity curves (tranched, overlap-corrected)")
        plot_curves(no, os.path.join(figs_dir, "equity_curves_gmvp_nonoverlap.png"),
                    "GMVP equity curves (non-overlapping stitch)")
        source = "per-day artifact"
    else:
        # Fallback path from per-anchor cumrets (used until the backtest is re-run).
        bt_csv = os.path.join(tag_dir, "backtest.csv")
        bt = pd.read_csv(bt_csv)
        no = nonoverlap_from_cumret(bt, horizon)
        summaries["nonoverlap"] = summarize(no, per_period_days=float(horizon))
        source = "cumret fallback (no per-day artifact; re-run backtest for tranched curve)"

    out_csv = os.path.join(tag_dir, "equity_curve_summary.csv")
    combined = pd.concat({k: v for k, v in summaries.items()}, names=["construction", "method"])
    combined.to_csv(out_csv)
    return {"source": source, "horizon": horizon, "summary_csv": out_csv, "summaries": summaries}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", default="results/regime_covariance",
                    help="Results dir containing backtest.csv and (after re-run) gmvp_daily_returns.parquet")
    args = ap.parse_args()
    info = build(args.tag_dir)
    print(f"Source: {info['source']}  (H={info['horizon']})")
    for name, s in info["summaries"].items():
        print(f"\n[{name}]")
        print(s.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {info['summary_csv']}")


if __name__ == "__main__":
    main()
