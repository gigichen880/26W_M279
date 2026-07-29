"""
Implementation moved from scripts/analysis/regime_characterization.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.regime.regime_label_utils import get_regime_name_map_for_backtest
from scripts.analysis.utils.paths import resolve_backtest_path
from similarity_forecast.regime_labels import (
    compute_regime_diagnostics,
    infer_n_regimes,
    save_regime_name_map,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"


def get_crisis_periods() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30")),
        (pd.Timestamp("2011-08-01"), pd.Timestamp("2011-10-31")),
        (pd.Timestamp("2015-08-01"), pd.Timestamp("2016-02-29")),
        (pd.Timestamp("2018-10-01"), pd.Timestamp("2018-12-31")),
        (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")),
        (pd.Timestamp("2020-09-01"), pd.Timestamp("2020-10-31")),
    ]


def is_in_crisis(date: pd.Timestamp, periods: list) -> bool:
    for start, end in periods:
        if start <= date <= end:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime characterization (cov or vol)")
    ap.add_argument("--input", default=None, help="Backtest parquet/csv")
    ap.add_argument("--target", choices=("auto", "covariance", "volatility"), default="auto")
    args = ap.parse_args()

    if args.input is not None:
        backtest_path = Path(args.input)
    else:
        backtest_path = resolve_backtest_path("regime_covariance")
        if not backtest_path.exists():
            backtest_path = resolve_backtest_path("regime_volatility")
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
    K = infer_n_regimes(df)
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
    out_name = "regime_characterization_volatility.csv" if is_vol else "regime_characterization.csv"
    tag = "regime_volatility" if is_vol else "regime_covariance"
    out_path = RESULTS_DIR / tag / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rmap = get_regime_name_map_for_backtest(backtest_path, df=df)
        save_regime_name_map(out_path.parent / "regime_name_map.json", rmap)
        chars["regime_label"] = chars["regime"].map(lambda k: rmap.get(int(k), ""))
        diag = compute_regime_diagnostics(df)
        diag["regime_stats"].to_csv(out_path.parent / "regime_diagnostics_stats.csv", index=False)
        diag["crisis_window_distributions"].to_csv(out_path.parent / "regime_diagnostics_crisis_windows.csv")
    except Exception as exc:
        print(f"Note: data-driven regime labels unavailable ({exc})")

    chars.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

