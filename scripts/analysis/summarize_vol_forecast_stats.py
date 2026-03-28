"""
Summarize volatility forecast accuracy from a wide backtest CSV (per-date vol_mse, vol_r2, ...).

Volatility runs do **not** include GMVP Sharpe. For paper text, use:
  - Mean **vol MSE** (and % lower than rolling / shrinkage / persistence baselines)
  - Mean **vol R²** (higher = more variance of realized vol² explained by forecasts)
  - Optional **annualized IR** of daily MSE advantage (baseline_vol_mse − model_vol_mse): Sharpe-like skill
  - Optional **mean of rolling-median** vol MSE (matches rolling_median_21d.png smoothing)

Usage:
  python -m scripts.analysis.summarize_vol_forecast_stats \\
      --input results/regime_volatility/backtest.csv

  python -m scripts.analysis.summarize_vol_forecast_stats --full-sample
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("model", "mix", "roll", "pers", "shrink")
EVAL_START_DEFAULT = "2013-01-01"
EVAL_END_DEFAULT = "2021-12-31"


def _require_vol_common_dates(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"{m}_{k}" for m in METHODS for k in ("vol_mse", "vol_mae", "vol_rmse")]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df
    return df.dropna(subset=existing).copy()


def _rolling_median_mean(s: pd.Series, window: int) -> float:
    """Mean of rolling(window).median(); first window-1 rows NaN (matches `rolling_median_*d.png`)."""
    w = max(2, int(window))
    rm = pd.to_numeric(s, errors="coerce").rolling(w, min_periods=w).median()
    tail = rm.iloc[w - 1 :]
    if tail.empty:
        return float("nan")
    return float(tail.mean())


def _ann_ir(adv: np.ndarray) -> float:
    adv = adv[np.isfinite(adv)]
    if adv.size < 2:
        return float("nan")
    mu = float(np.mean(adv))
    sd = float(np.std(adv, ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return float("nan")
    return mu / sd * math.sqrt(252.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/regime_volatility/backtest.csv")
    ap.add_argument("--eval-start", default=EVAL_START_DEFAULT)
    ap.add_argument("--eval-end", default=EVAL_END_DEFAULT)
    ap.add_argument(
        "--full-sample",
        action="store_true",
        help="Use all dates (matches run_backtest report.csv means); ignore eval window.",
    )
    ap.add_argument("--rolling-window", type=int, default=21, help="For rolling-median summary.")
    ap.add_argument(
        "--baselines",
        nargs="+",
        default=["roll", "shrink", "pers"],
        help="Baselines to compare model against (default: roll shrink pers).",
    )
    args = ap.parse_args()

    baseline_labels = {
        "roll": "rolling",
        "shrink": "shrinkage diagonal",
        "pers": "persistence",
        "mix": "mix",
    }

    path = Path(args.input)
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    if not args.full_sample:
        t0 = pd.Timestamp(args.eval_start)
        t1 = pd.Timestamp(args.eval_end)
        df = df.loc[(df["date"] >= t0) & (df["date"] <= t1)]

    df = _require_vol_common_dates(df)
    n = len(df)
    if n == 0:
        print("No rows after filters.", file=sys.stderr)
        raise SystemExit(1)

    w = max(2, int(args.rolling_window))

    print("## Volatility forecast summary\n")
    print(f"- Input: `{path}`")
    print(f"- Rows: **{n}**" + (" (full sample)" if args.full_sample else f" (eval {args.eval_start} … {args.eval_end})"))
    print(
        "- Note: there is **no GMVP Sharpe** on volatility runs; use **vol MSE / R²** "
        "and optionally **annualized IR** of daily MSE advantage vs baselines.\n"
    )

    # Table: mean vol_mse, vol_r2, vol_rmse
    lines = ["| Method | Mean vol MSE | Mean vol R² | Mean vol RMSE |", "|--------|--------------|-------------|---------------|"]
    means: dict[str, dict[str, float]] = {}
    for m in METHODS:
        if f"{m}_vol_mse" not in df.columns:
            continue
        means[m] = {
            "vol_mse": float(pd.to_numeric(df[f"{m}_vol_mse"], errors="coerce").mean()),
            "vol_r2": float(pd.to_numeric(df.get(f"{m}_vol_r2"), errors="coerce").mean())
            if f"{m}_vol_r2" in df.columns
            else float("nan"),
            "vol_rmse": float(pd.to_numeric(df.get(f"{m}_vol_rmse"), errors="coerce").mean())
            if f"{m}_vol_rmse" in df.columns
            else float("nan"),
        }
        r2 = means[m]["vol_r2"]
        rm = means[m]["vol_rmse"]
        lines.append(
            f"| {m} | {means[m]['vol_mse']:.4f} | {r2:+.4f} | {rm:.4f} |"
            if math.isfinite(rm)
            else f"| {m} | {means[m]['vol_mse']:.4f} | {r2:+.4f} | — |"
        )
    print("\n".join(lines))

    m_mse = means.get("model", {}).get("vol_mse")
    print("\n### % lower mean vol MSE than baseline (model vs …)\n")
    for b in args.baselines:
        if b not in means or m_mse is None:
            continue
        b_mse = means[b]["vol_mse"]
        if b_mse and math.isfinite(b_mse):
            pct = 100.0 * (b_mse - m_mse) / b_mse
            print(f"- **{b}**: **{pct:.2f}%** lower MSE (model {m_mse:.4f} vs {b} {b_mse:.4f})")

    # Daily advantage IR
    if "model_vol_mse" in df.columns:
        m_series = pd.to_numeric(df["model_vol_mse"], errors="coerce").to_numpy()
        print("\n### Annualized IR of daily MSE advantage (baseline_mse − model_mse)\n")
        for b in args.baselines:
            col = f"{b}_vol_mse"
            if col not in df.columns:
                continue
            b_series = pd.to_numeric(df[col], errors="coerce").to_numpy()
            adv = b_series - m_series
            ir = _ann_ir(adv)
            print(f"- vs **{b}**: IR ≈ **{ir:.2f}** (higher = more consistent out-of-sample skill)")

    # Rolling median of vol_mse — mean of smoothed series (skip burn-in w-1)
    print(f"\n### Mean of **{w}-day rolling median** vol MSE (cf. `rolling_median_{w}d.png`)\n")
    for m in METHODS:
        col = f"{m}_vol_mse"
        if col not in df.columns:
            continue
        mu = _rolling_median_mean(df[col], w)
        if math.isfinite(mu):
            print(f"- **{m}**: {mu:.4f}")
    mm = _rolling_median_mean(df["model_vol_mse"], w)
    print("\n- Rolling-median vol MSE: model vs baselines (% lower MSE than baseline)\n")
    for b in args.baselines:
        if b not in means or b == "model":
            continue
        col = f"{b}_vol_mse"
        if col not in df.columns:
            continue
        bm = _rolling_median_mean(df[col], w)
        if math.isfinite(mm) and math.isfinite(bm) and bm != 0:
            pct = 100.0 * (bm - mm) / bm
            print(
                f"  - vs **{b}** ({baseline_labels.get(b, b)}): **{pct:.1f}%** "
                f"(model rolling-median MSE {mm:.4f} vs {b} {bm:.4f})"
            )

    # Paste-in sentence (all baselines in args.baselines)
    m_r2 = means.get("model", {}).get("vol_r2")
    m_rmse = means.get("model", {}).get("vol_rmse")
    bl_parts = []
    pct_parts = []
    for b in args.baselines:
        if b not in means or m_mse is None:
            continue
        b_mse = means[b]["vol_mse"]
        if b_mse and math.isfinite(b_mse):
            lbl = baseline_labels.get(b, b)
            bl_parts.append(f"**{b_mse:.3f}** ({lbl})")
            pct_parts.append(f"**{100.0 * (b_mse - m_mse) / b_mse:.1f}%** vs {lbl}")
    n_bl = len([b for b in args.baselines if b in means])
    if m_mse is not None and len(bl_parts) == n_bl and n_bl > 0:
        print("\n### Suggested paper sentence (edit tense/labels as needed)\n")
        vs_list = ", ".join(bl_parts)
        pct_str = "; ".join(pct_parts)
        print(
            f"> On mean squared volatility forecast error, the model achieves **{m_mse:.3f}** vs "
            f"{vs_list}, i.e. {pct_str} lower MSE."
        )
        r2_parts: list[str] = []
        rmse_parts: list[str] = []
        pct_rmse_parts: list[str] = []
        for b in args.baselines:
            if b not in means:
                continue
            r2b = means[b].get("vol_r2")
            rmb = means[b].get("vol_rmse")
            if r2b is not None and math.isfinite(r2b):
                r2_parts.append(f"**{r2b:.3f}** ({baseline_labels.get(b, b)})")
            if rmb is not None and math.isfinite(rmb) and m_rmse is not None and math.isfinite(m_rmse):
                rmse_parts.append(f"**{rmb:.3f}** ({baseline_labels.get(b, b)})")
                pct_rmse_parts.append(
                    f"**{100.0 * (rmb - m_rmse) / rmb:.1f}%** vs {baseline_labels.get(b, b)}"
                )
        if (
            m_r2 is not None
            and math.isfinite(m_r2)
            and len(r2_parts) == n_bl
            and m_r2 >= 0.0
        ):
            print(
                f"> For **variance prediction accuracy** (mean **vol R²**), the model reaches **{m_r2:.3f}** vs "
                f"{', '.join(r2_parts)}."
            )
        elif (
            m_rmse is not None
            and math.isfinite(m_rmse)
            and len(rmse_parts) == n_bl
            and len(pct_rmse_parts) == n_bl
        ):
            print(
                f"> *(Mean vol R² is negative or omitted on this slice; report **RMSE** instead.)* "
                f"Mean **vol RMSE** is **{m_rmse:.3f}** vs {', '.join(rmse_parts)}, "
                f"i.e. {', '.join(pct_rmse_parts)} lower error."
            )


if __name__ == "__main__":
    main()
