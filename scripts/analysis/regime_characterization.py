"""
Generate quantitative regime characterization from backtest data.

For each regime (0-3): n_days, pct_time, mean_fro, mean_gmvp_sharpe, mean_turnover, crisis_overlap.
Saves results/regime_characterization.csv and prints a formatted table with suggested names.

Usage:
  python scripts/analysis/regime_characterization.py
  python scripts/analysis/regime_characterization.py --input results/regime_volatility_backtest.parquet --target volatility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def get_crisis_periods() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) for each crisis period."""
    return [
        (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30")),  # Financial Crisis
        (pd.Timestamp("2011-08-01"), pd.Timestamp("2011-10-31")),  # EU Debt
        (pd.Timestamp("2015-08-01"), pd.Timestamp("2016-02-29")),  # China Selloff
        (pd.Timestamp("2018-10-01"), pd.Timestamp("2018-12-31")),  # Q4 Selloff
        (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")),  # COVID Crash
        (pd.Timestamp("2020-09-01"), pd.Timestamp("2020-10-31")),  # Election Vol
    ]


def is_in_crisis(date: pd.Timestamp, periods: list) -> bool:
    """True if date falls within any crisis period."""
    for start, end in periods:
        if start <= date <= end:
            return True
    return False


def suggest_regime_name(row: pd.Series, is_vol: bool = False) -> str:
    """Suggest interpretable name from stats (including crisis_overlap)."""
    crisis_pct = row.get("crisis_overlap", 0) if "crisis_overlap" in row.index else 0
    if pd.notna(crisis_pct) and crisis_pct > 30:
        return "Crisis/High Stress"
    if not is_vol:
        sharpe = row.get("mean_gmvp_sharpe", np.nan)
        turnover = row.get("mean_turnover", np.nan)
        fro = row.get("mean_fro", np.nan)
        if pd.notna(sharpe) and sharpe > 1.5 and pd.notna(turnover) and turnover < 0.5:
            return "Calm Bull"
        if pd.notna(sharpe) and sharpe < 0.5:
            return "Stress/Low Sharpe"
        if pd.notna(fro) and fro > 0.025:
            return "High Uncertainty"
        if pd.notna(turnover) and turnover > 0.7:
            return "Volatile/Choppy"
    else:
        vol_mse = row.get("mean_vol_mse", np.nan)
        if pd.notna(vol_mse) and vol_mse > 0.05:
            return "High Vol MSE"
    return "Normal/Transition"


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime characterization (cov or vol)")
    ap.add_argument("--input", default=None, help="Backtest parquet/csv")
    ap.add_argument("--target", choices=("auto", "covariance", "volatility"), default="auto")
    args = ap.parse_args()

    if args.input is not None:
        backtest_path = Path(args.input)
    else:
        for name in ("regime_covariance_backtest.parquet", "regime_covariance_backtest.csv",
                     "regime_volatility_backtest.parquet", "regime_volatility_backtest.csv"):
            p = RESULTS_DIR / name
            if p.exists():
                backtest_path = p
                break
        else:
            backtest_path = RESULTS_DIR / "regime_covariance_backtest.parquet"
    if not backtest_path.exists():
        print("Backtest not found. Run: python run_backtest.py --config configs/regime_covariance.yaml or configs/regime_volatility.yaml")
        return

    df = pd.read_parquet(backtest_path) if backtest_path.suffix == ".parquet" else pd.read_csv(backtest_path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    if "regime_assigned" not in df.columns:
        print("No regime_assigned column. Re-run backtest with regime saving enabled.")
        return

    is_vol = args.target == "volatility" or (args.target == "auto" and "model_vol_mse" in df.columns)
    K = 4
    crisis_periods = get_crisis_periods()
    df["in_crisis"] = df["date"].apply(lambda d: is_in_crisis(d, crisis_periods))

    fro_col = "model_fro"
    sharpe_col = "model_gmvp_sharpe"
    turnover_col = "model_turnover_l1"
    vol_col = "model_vol_mse"

    rows = []
    for k in range(K):
        mask = df["regime_assigned"] == k
        regime_df = df.loc[mask]
        n_days = len(regime_df)
        if n_days == 0:
            continue
        n_crisis = regime_df["in_crisis"].sum()
        crisis_overlap = (n_crisis / n_days * 100) if n_days else 0.0
        row = {
            "regime": k,
            "n_days": n_days,
            "pct_time": n_days / len(df) * 100,
            "mean_fro": regime_df[fro_col].mean() if fro_col in df.columns else np.nan,
            "mean_gmvp_sharpe": regime_df[sharpe_col].mean() if sharpe_col in df.columns else np.nan,
            "mean_turnover": regime_df[turnover_col].mean() if turnover_col in df.columns else np.nan,
            "mean_vol_mse": regime_df[vol_col].mean() if vol_col in df.columns else np.nan,
            "crisis_overlap": round(crisis_overlap, 1),
        }
        rows.append(row)

    chars = pd.DataFrame(rows)
    chars["suggested_name"] = chars.apply(lambda r: suggest_regime_name(r, is_vol=is_vol), axis=1)

    out_name = "regime_characterization_volatility.csv" if is_vol else "regime_characterization.csv"
    out_path = RESULTS_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chars.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")

    print("=" * 100)
    print("REGIME CHARACTERIZATION (quantitative)" + (" [volatility]" if is_vol else " [covariance]"))
    print("=" * 100)
    print()
    disp = chars.copy()
    disp["pct_time"] = disp["pct_time"].apply(lambda x: f"{x:.1f}%")
    for c in ["mean_fro", "mean_gmvp_sharpe", "mean_turnover", "mean_vol_mse"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    disp["crisis_overlap"] = disp["crisis_overlap"].apply(lambda x: f"{x:.1f}%")
    cols = ["regime", "n_days", "pct_time"] + [c for c in ["mean_fro", "mean_vol_mse", "mean_gmvp_sharpe", "mean_turnover"] if c in disp.columns] + ["crisis_overlap", "suggested_name"]
    print(disp[[c for c in cols if c in disp.columns]].to_string(index=False))
    print()
    print("Suggested regime names (based on stats):")
    for _, r in chars.iterrows():
        print(f"  Regime {int(r['regime'])}: {r['suggested_name']}")
    print()


if __name__ == "__main__":
    main()
