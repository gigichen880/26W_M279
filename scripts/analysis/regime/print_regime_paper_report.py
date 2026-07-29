#!/usr/bin/env python3
"""
Load fresh backtest output and print regime diagnostics + validation + markdown skeleton.

Usage (from repo root):
  python -m scripts.analysis.regime.print_regime_paper_report
  python -m scripts.analysis.regime.print_regime_paper_report --input results/regime_covariance/backtest.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis.utils.paths import resolve_backtest_path
from similarity_forecast.regime_labels import (
    build_regime_name_map,
    compute_regime_diagnostics,
    resolve_dominant_regime_series,
)


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="backtest parquet/csv")
    args = ap.parse_args()

    path = Path(args.input) if args.input else resolve_backtest_path("regime_covariance")
    if not path.exists():
        print(f"Missing backtest: {path}")
        print("Run: python run_backtest.py --config configs/regime_covariance.yaml")
        return

    df = _load_df(path)
    need = ("dominant_regime", "realized_vol", "avg_corr", "market_ret", "regime_label")
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"ERROR: backtest missing columns {missing}. Re-run run_backtest.py (pipeline adds these).")
        return

    diag = compute_regime_diagnostics(df)
    rmap = build_regime_name_map(df)
    rs = diag["regime_stats"].set_index("regime_id")

    print("=== regime_stats ===")
    print(diag["regime_stats"].to_string(index=False))
    print("\n=== crisis_window_distributions (days per regime by window) ===")
    print(diag["crisis_window_distributions"].to_string())
    print("\n=== crisis_counts_total (days in any labeling-crisis window) ===")
    print(diag["crisis_counts_total"].to_string())
    print("\n=== final regime_name_map ===")
    print(rmap)

    # Named historical checks (informative; labeling uses DEFAULT_LABEL_CRISIS_WINDOWS)
    windows = [
        ("2008-09-01", "2009-03-31", "GFC (labeling window)"),
        ("2015-08-01", "2016-02-28", "2015 turbulence (labeling window)"),
        ("2020-02-01", "2020-06-30", "2020 COVID (labeling window)"),
    ]
    print("\n=== Dominant-regime share (%) in selected windows ===")
    dom = resolve_dominant_regime_series(df)
    for a, b, lab in windows:
        m = (df["date"] >= pd.Timestamp(a)) & (df["date"] <= pd.Timestamp(b))
        sub = dom[m & dom.notna()]
        vc = sub.value_counts(normalize=True).sort_index() * 100.0
        print(f"\n{lab} [{a} .. {b}]")
        for k, p in vc.items():
            print(f"  Regime {int(k)} ({rmap.get(int(k), '?')}): {float(p):.1f}%")

    # Consistency checks
    inv_hs = {v: k for k, v in rmap.items()}
    hs_id = inv_hs.get("High Stress")
    vol_id = int(rs["mean_realized_vol"].idxmax())
    corr_id = int(rs["mean_avg_corr"].idxmax())
    cc = diag["crisis_counts_total"]
    crisis_id = int(cc.idxmax()) if float(cc.sum()) > 0 else None

    print("\n=== Label sanity checks ===")
    print(f"Regime with highest mean realized_vol: {vol_id} ({rmap.get(vol_id)})")
    print(f"Regime with highest mean avg_corr:     {corr_id} ({rmap.get(corr_id)})")
    print(f"Regime with most crisis-window days:   {crisis_id} ({rmap.get(crisis_id) if crisis_id is not None else 'n/a'})")
    print(f"Assigned High Stress regime id:        {hs_id} ({rmap.get(hs_id) if hs_id is not None else 'n/a'})")

    flags = []
    if hs_id is not None:
        if hs_id != vol_id:
            flags.append("High Stress id != argmax mean vol (tie-break or composite rule may apply).")
        if hs_id != corr_id:
            flags.append("High Stress id != argmax mean corr.")
        if crisis_id is not None and hs_id != crisis_id:
            flags.append("High Stress id != regime with most days in union of labeling crisis windows.")
    if not flags:
        flags.append("No obvious inconsistencies (strict triple match or composite score aligns with High Stress).")
    print("\n=== flags ===")
    for f in flags:
        print(f"- {f}")

    # Markdown skeleton
    print("\n--- MARKDOWN (paste into doc) ---\n")
    print("## Final Regime Mapping")
    for k in sorted(rmap.keys()):
        print(f"- Regime {k} -> {rmap[k]}")
    print("\n## Evidence")
    print(f"- highest mean vol: Regime {vol_id} ({rmap.get(vol_id)})")
    print(f"- highest mean corr: Regime {corr_id} ({rmap.get(corr_id)})")
    low_id = int(rs["mean_realized_vol"].idxmin())
    print(f"- lowest mean vol: Regime {low_id} ({rmap.get(low_id)})")
    ret_id = int(rs["mean_market_ret"].idxmax())
    print(f"- strongest positive mean market_ret: Regime {ret_id} ({rmap.get(ret_id)})")
    print(f"- crisis-dominant regime(s) by day-count in union of labeling windows: Regime {crisis_id} ({rmap.get(crisis_id) if crisis_id is not None else 'n/a'})")


if __name__ == "__main__":
    main()
