"""
Generate quantitative regime characterization from backtest data.

For each regime (0-3): n_days, pct_time, mean_fro, mean_gmvp_sharpe, mean_turnover, crisis_overlap.
Saves results/regime_characterization.csv and prints a formatted table with suggested names.

Usage: python scripts/analysis/regime_characterization.py
"""

from __future__ import annotations

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


def suggest_regime_name(row: pd.Series) -> str:
    """Suggest interpretable name from stats (including crisis_overlap)."""
    sharpe = row.get("mean_gmvp_sharpe", np.nan)
    turnover = row.get("mean_turnover", np.nan)
    fro = row.get("mean_fro", np.nan)
    crisis_pct = row.get("crisis_overlap", 0) if "crisis_overlap" in row.index else 0

    if pd.notna(crisis_pct) and crisis_pct > 30:
        return "Crisis/High Stress"
    if pd.notna(sharpe) and sharpe > 1.5 and pd.notna(turnover) and turnover < 0.5:
        return "Calm Bull"
    if pd.notna(sharpe) and sharpe < 0.5:
        return "Stress/Low Sharpe"
    if pd.notna(fro) and fro > 0.025:
        return "High Uncertainty"
    if pd.notna(turnover) and turnover > 0.7:
        return "Volatile/Choppy"
    return "Normal/Transition"


def main() -> None:
    backtest_path = RESULTS_DIR / "regime_similarity_backtest.parquet"
    if not backtest_path.exists():
        backtest_path = RESULTS_DIR / "regime_similarity_backtest.csv"
    if not backtest_path.exists():
        print("Backtest not found. Run: python run_backtest.py --config configs/regime_similarity.yaml")
        return

    if backtest_path.suffix == ".parquet":
        df = pd.read_parquet(backtest_path)
    else:
        df = pd.read_csv(backtest_path)

    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "regime_assigned" not in df.columns:
        print("No regime_assigned column. Re-run backtest with regime saving enabled.")
        return

    K = 4
    crisis_periods = get_crisis_periods()
    df["in_crisis"] = df["date"].apply(lambda d: is_in_crisis(d, crisis_periods))

    fro_col = "model_fro"
    sharpe_col = "model_gmvp_sharpe"
    turnover_col = "model_turnover_l1"

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
            "crisis_overlap": round(crisis_overlap, 1),
        }
        rows.append(row)

    chars = pd.DataFrame(rows)
    chars["suggested_name"] = chars.apply(suggest_regime_name, axis=1)

    out_path = RESULTS_DIR / "regime_characterization.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chars.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")

    # Formatted table
    print("=" * 100)
    print("REGIME CHARACTERIZATION (quantitative)")
    print("=" * 100)
    print()
    # Build display DataFrame with nice formatting
    disp = chars.copy()
    disp["pct_time"] = disp["pct_time"].apply(lambda x: f"{x:.1f}%")
    disp["mean_fro"] = disp["mean_fro"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    disp["mean_gmvp_sharpe"] = disp["mean_gmvp_sharpe"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    disp["mean_turnover"] = disp["mean_turnover"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    disp["crisis_overlap"] = disp["crisis_overlap"].apply(lambda x: f"{x:.1f}%")
    print(disp[["regime", "n_days", "pct_time", "mean_fro", "mean_gmvp_sharpe", "mean_turnover", "crisis_overlap", "suggested_name"]].to_string(index=False))
    print()
    print("Suggested regime names (based on stats):")
    for _, r in chars.iterrows():
        print(f"  Regime {int(r['regime'])}: {r['suggested_name']}")
    print()


if __name__ == "__main__":
    main()
