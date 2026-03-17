"""
Investigate why Persistence (pers) method has extreme cumulative returns (>10) in 2021.

Covariance backtest only (uses GMVP columns). For volatility backtest use other analysis scripts.

Usage:
  python scripts/analysis/investigate_pers_extreme_returns.py
  python scripts/analysis/investigate_pers_extreme_returns.py --input results/regime_covariance_backtest.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description="Investigate pers GMVP extreme returns (covariance backtest only)")
    ap.add_argument("--input", default=None, help="Covariance backtest parquet/csv (default: regime_covariance_backtest)")
    args = ap.parse_args()
    if args.input:
        backtest_path = Path(args.input)
    else:
        backtest_path = RESULTS_DIR / "regime_covariance_backtest.parquet"
        if not backtest_path.exists():
            backtest_path = RESULTS_DIR / "regime_covariance_backtest.csv"
    if not backtest_path.exists():
        print("Backtest file not found. Expected one of:")
        print(f"  {RESULTS_DIR / 'regime_covariance_backtest.parquet'}")
        print(f"  {RESULTS_DIR / 'regime_covariance_backtest.csv'}")
        print("Run: python run_backtest.py --config configs/regime_covariance.yaml")
        return

    if backtest_path.suffix == ".parquet":
        df = pd.read_parquet(backtest_path)
    else:
        df = pd.read_csv(backtest_path)

    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "pers_gmvp_cumret" not in df.columns:
        print("This script requires a covariance backtest (columns pers_gmvp_cumret, roll_gmvp_cumret, etc.).")
        print("Volatility backtests do not have GMVP columns. Use --input with regime_covariance_backtest.parquet")
        return
    # Backtest has one row per date; columns are pers_gmvp_cumret, roll_gmvp_cumret, etc.
    pers_cumret = pd.to_numeric(df["pers_gmvp_cumret"], errors="coerce")
    roll_cumret = pd.to_numeric(df["roll_gmvp_cumret"], errors="coerce")
    pers_mean = pd.to_numeric(df["pers_gmvp_mean"], errors="coerce")
    roll_mean = pd.to_numeric(df["roll_gmvp_mean"], errors="coerce")

    print("=" * 80)
    print("PERSISTENCE (pers) EXTREME RETURNS INVESTIGATION")
    print("=" * 80)
    print()
    print(f"Loaded {len(df)} evaluation dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # 1–2. Pers cumret over time (summary stats)
    print("1. pers_gmvp_cumret summary:")
    print(f"   count: {pers_cumret.notna().sum()}, nulls: {pers_cumret.isna().sum()}")
    print(f"   min: {pers_cumret.min():.6f}, max: {pers_cumret.max():.6f}")
    print(f"   mean: {pers_cumret.mean():.6f}, std: {pers_cumret.std():.6f}")
    print()

    # 3–4. Equity curve = cumprod(1 + cumret); when does pers diverge?
    eq_pers = (1 + pers_cumret.fillna(0)).cumprod()
    eq_roll = (1 + roll_cumret.fillna(0)).cumprod()
    print("2. Cumulative equity (cumprod(1 + gmvp_cumret)):")
    print(f"   pers  final: {eq_pers.iloc[-1]:.4f}, max: {eq_pers.max():.4f}")
    print(f"   roll  final: {eq_roll.iloc[-1]:.4f}, max: {eq_roll.max():.4f}")
    print()

    # When does pers equity exceed 10?
    over_10 = np.where(eq_pers.values >= 10)[0]
    if len(over_10) > 0:
        first_over_10 = over_10[0]
        print(f"3. pers equity first reaches >= 10 at index {first_over_10}, date {df['date'].iloc[first_over_10].date()}")
    # When does pers start to diverge from roll (e.g. pers/roll > 2)?
    ratio_eq = eq_pers / np.maximum(eq_roll, 1e-12)
    div_idx = np.where(ratio_eq.values >= 2)[0]
    if len(div_idx) > 0:
        print(f"   pers/roll equity ratio first >= 2 at index {div_idx[0]}, date {df['date'].iloc[div_idx[0]].date()}")
    print()

    # 5. Are pers_gmvp_mean values realistic? (daily mean return in decimal: typically -0.02 to 0.02)
    print("4. pers_gmvp_mean (per-horizon daily mean return):")
    print(f"   min: {pers_mean.min():.6f}, max: {pers_mean.max():.6f}")
    print(f"   mean: {pers_mean.mean():.6f}, std: {pers_mean.std():.6f}")
    extreme_mean = (pers_mean.abs() > 0.05).sum()
    print(f"   count |mean| > 0.05 (5%): {extreme_mean}")
    if extreme_mean > 0:
        print("   Sample dates with |pers_gmvp_mean| > 0.05:")
        mask = pers_mean.abs() > 0.05
        sub = df.loc[mask, ["date", "pers_gmvp_mean", "pers_gmvp_cumret", "roll_gmvp_mean", "roll_gmvp_cumret"]].head(15)
        print(sub.to_string(index=False))
    print()

    # 6. Compare to other methods at same dates (2021)
    df_2021 = df[df["date"].dt.year == 2021]
    if len(df_2021) > 0:
        print("5. 2021 comparison (pers vs roll vs model):")
        for col in ["gmvp_cumret", "gmvp_mean", "gmvp_sharpe"]:
            p = pd.to_numeric(df_2021[f"pers_{col}"], errors="coerce")
            r = pd.to_numeric(df_2021[f"roll_{col}"], errors="coerce")
            m = pd.to_numeric(df_2021[f"model_{col}"], errors="coerce")
            print(f"   {col}: pers mean={p.mean():.6f}, roll mean={r.mean():.6f}, model mean={m.mean():.6f}")
            print(f"         pers max={p.max():.6f}, roll max={r.max():.6f}")
    print()

    # Suspicious dates: pers differs by >3x from roll (same-day horizon return ratio)
    ok = pers_cumret.notna() & roll_cumret.notna()
    r_pers = (1 + pers_cumret).clip(1e-8, None)
    r_roll = (1 + roll_cumret).clip(1e-8, None)
    ratio = r_pers / r_roll
    suspicious = ok & ((ratio > 3) | (ratio < 1 / 3))
    n_sus = suspicious.sum()
    print("6. Suspicious dates (pers (1+cumret) / roll (1+cumret) > 3 or < 1/3 on same day):")
    print(f"   count: {n_sus}")
    if n_sus > 0:
        sub = df.loc[suspicious, ["date", "pers_gmvp_cumret", "roll_gmvp_cumret", "pers_gmvp_mean", "roll_gmvp_mean"]].copy()
        sub["ratio"] = ratio[suspicious].values
        print(sub.to_string(index=False))
    else:
        print("   (none — pers does not exceed 3x roll on any single evaluation date)")
    # Cumulative equity ratio over time
    eq_ratio = eq_pers / np.maximum(eq_roll, 1e-12)
    print(f"   Cumulative equity ratio (pers/roll) at end: {eq_ratio.iloc[-1]:.2f}")
    print()

    # Also flag by absolute pers_cumret extreme
    print("7. Dates with |pers_gmvp_cumret| > 0.5 (extreme horizon return):")
    ext = pers_cumret.abs() > 0.5
    if ext.sum() > 0:
        sub = df.loc[ext, ["date", "pers_gmvp_cumret", "roll_gmvp_cumret", "pers_gmvp_mean", "roll_gmvp_mean"]]
        print(sub.to_string(index=False))
    else:
        print("   none")
    # Optional: plot gmvp_cumret over time (pers vs roll)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1, ax2 = axes
        ax1.plot(df["date"], pers_cumret, label="pers_gmvp_cumret", alpha=0.8)
        ax1.plot(df["date"], roll_cumret, label="roll_gmvp_cumret", alpha=0.8)
        ax1.set_ylabel("Horizon cumulative return")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Persistence vs Rolling: gmvp_cumret over time")
        ax2.plot(df["date"], eq_pers.values, label="pers equity")
        ax2.plot(df["date"], eq_roll.values, label="roll equity")
        ax2.set_ylabel("Cumulative equity")
        ax2.set_xlabel("Date")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Cumulative equity (cumprod(1+gmvp_cumret))")
        plt.tight_layout()
        out_path = RESULTS_DIR / "figs_regime_covariance" / "pers_vs_roll_investigation.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"8. Plot saved: {out_path}")
    except Exception as e:
        print(f"8. Plot skipped: {e}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
