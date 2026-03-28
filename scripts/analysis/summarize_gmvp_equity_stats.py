"""
Summarize GMVP equity–relevant statistics from a covariance backtest CSV.

The equity plot uses cumulative wealth W_t = cumprod(1 + gmvp_cumret) per refit date.
This script reports:
  - Terminal wealth (final W) per method
  - Relative outperformance vs a baseline: (W_model / W_base - 1) × 100%
  - Mean horizon GMVP Sharpe (same average as report.csv / statistical comparison)

Usage:
  python -m scripts.analysis.summarize_gmvp_equity_stats \\
      --input results/regime_covariance/backtest.csv \\
      --baselines roll shrink pers

Optional Markdown table to stdout for paper paste-in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHODS_DEFAULT = ("model", "mix", "roll", "shrink", "pers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/regime_covariance/backtest.csv")
    ap.add_argument(
        "--baselines",
        nargs="+",
        default=["roll", "shrink", "pers"],
        help="Baselines for relative terminal wealth / Sharpe vs model",
    )
    ap.add_argument("--reference", default="model", help="Primary method (default model).")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    ref = args.reference
    rows = []
    for m in METHODS_DEFAULT:
        cr = f"{m}_gmvp_cumret"
        sh = f"{m}_gmvp_sharpe"
        if cr not in df.columns:
            continue
        r = pd.to_numeric(df[cr], errors="coerce").fillna(0.0).to_numpy()
        terminal_w = float(np.prod(1.0 + r))
        mean_sharpe = float(pd.to_numeric(df[sh], errors="coerce").mean()) if sh in df.columns else float("nan")
        rows.append(
            {
                "method": m,
                "terminal_wealth": terminal_w,
                "mean_gmvp_sharpe": mean_sharpe,
                "n_periods": int(len(df)),
            }
        )

    tab = pd.DataFrame(rows)
    if tab.empty:
        print("No gmvp_cumret columns found.", file=sys.stderr)
        raise SystemExit(1)

    ref_row = tab[tab["method"] == ref]
    if ref_row.empty:
        print(f"Reference {ref} not in data.", file=sys.stderr)
        raise SystemExit(1)

    W_ref = float(ref_row["terminal_wealth"].iloc[0])
    S_ref = float(ref_row["mean_gmvp_sharpe"].iloc[0])

    print("## GMVP summary (from chained horizon cumulative returns)\n")
    print(f"- Input: `{path}`")
    print(f"- Periods (rows): {int(ref_row['n_periods'].iloc[0])}")
    print(f"- Terminal wealth: wealth_T = ∏_t (1 + gmvp_cumret_t) per method (same construction as equity_curves_gmvp.png).\n")

    print("| Method | Terminal wealth | Mean GMVP Sharpe (horizon avg.) |")
    print("|--------|-----------------|--------------------------------|")
    for _, r in tab.iterrows():
        print(f"| {r['method']} | {r['terminal_wealth']:.4f} | {r['mean_gmvp_sharpe']:.4f} |")

    print("\n### Relative vs baselines (reference = **%s**)\n" % ref)
    print("| Baseline | Terminal wealth (%) | Mean Sharpe (%) |")
    print("|----------|----------------------|-------------------|")
    for b in args.baselines:
        br = tab[tab["method"] == b]
        if br.empty:
            continue
        W_b = float(br["terminal_wealth"].iloc[0])
        S_b = float(br["mean_gmvp_sharpe"].iloc[0])
        pct_w = 100.0 * (W_ref / W_b - 1.0) if W_b > 0 else float("nan")
        pct_s = 100.0 * (S_ref / S_b - 1.0) if np.isfinite(S_b) and S_b != 0 else float("nan")
        print(f"| {b} | **{pct_w:+.2f}%** vs baseline wealth | **{pct_s:+.2f}%** vs baseline Sharpe |")

    print(
        "\n*Interpretation:* **Terminal wealth %** is (W_ref/W_base − 1)×100. "
        "**Mean Sharpe %** is (mean horizon Sharpe_ref / mean horizon Sharpe_base − 1)×100. "
        "Horizon Sharpe matches the mean reported in `report.csv` and the statistical comparison plots."
    )


if __name__ == "__main__":
    main()
