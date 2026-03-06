"""
Compute Model vs Roll performance in each regime.

For each regime k: mean Fro (model vs roll), mean Sharpe (model vs roll), win rate (fraction of days
where model_fro < roll_fro). Prints table and saves results/figs_regime_similarity/performance_by_regime.png.

Usage: python scripts/analysis/performance_by_regime.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figs_regime_similarity"


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

    if "regime_assigned" not in df.columns:
        print("No regime_assigned column. Re-run backtest with regime saving.")
        return

    fro_model = pd.to_numeric(df["model_fro"], errors="coerce")
    fro_roll = pd.to_numeric(df["roll_fro"], errors="coerce")
    sharpe_model = pd.to_numeric(df["model_gmvp_sharpe"], errors="coerce")
    sharpe_roll = pd.to_numeric(df["roll_gmvp_sharpe"], errors="coerce")
    regime = df["regime_assigned"]

    K = 4
    rows = []
    for k in range(K):
        mask = regime == k
        n_days = mask.sum()
        if n_days == 0:
            continue
        fm = fro_model.loc[mask].mean()
        fr = fro_roll.loc[mask].mean()
        sm = sharpe_model.loc[mask].mean()
        sr = sharpe_roll.loc[mask].mean()
        # Win% = fraction of days where model has lower Fro (model better)
        sub_fm = fro_model.loc[mask]
        sub_fr = fro_roll.loc[mask]
        valid = sub_fm.notna() & sub_fr.notna()
        win_pct = (sub_fm[valid] < sub_fr[valid]).mean() * 100 if valid.sum() > 0 else np.nan
        rows.append({
            "Regime": k,
            "N_days": n_days,
            "Fro_Model": fm,
            "Fro_Roll": fr,
            "Sharpe_Model": sm,
            "Sharpe_Roll": sr,
            "Win%": win_pct,
        })

    table = pd.DataFrame(rows)

    # Print table
    print("=" * 80)
    print("MODEL vs ROLL BY REGIME")
    print("=" * 80)
    print()
    print(table.to_string(index=False))
    print()

    # Save figure
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        regimes = table["Regime"].astype(int)
        x = np.arange(len(regimes))
        w = 0.35

        # Fro: Model vs Roll
        ax = axes[0]
        ax.bar(x - w / 2, table["Fro_Model"], w, label="Model", color="steelblue", alpha=0.9)
        ax.bar(x + w / 2, table["Fro_Roll"], w, label="Roll", color="gray", alpha=0.7)
        ax.set_ylabel("Mean Frobenius error")
        ax.set_xlabel("Regime")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Regime {r}" for r in regimes])
        ax.legend(loc="best")
        ax.set_title("Covariance error (lower is better)")
        ax.grid(True, alpha=0.3, axis="y")

        # Sharpe: Model vs Roll
        ax = axes[1]
        ax.bar(x - w / 2, table["Sharpe_Model"], w, label="Model", color="steelblue", alpha=0.9)
        ax.bar(x + w / 2, table["Sharpe_Roll"], w, label="Roll", color="gray", alpha=0.7)
        ax.set_ylabel("Mean GMVP Sharpe")
        ax.set_xlabel("Regime")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Regime {r}" for r in regimes])
        ax.legend(loc="best")
        ax.set_title("Portfolio Sharpe (higher is better)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.5)

        # Win% (model beats roll on Fro)
        ax = axes[2]
        bars = ax.bar(x, table["Win%"], color="steelblue", alpha=0.9)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50%")
        ax.set_ylabel("Win % (model Fro < roll Fro)")
        ax.set_xlabel("Regime")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Regime {r}" for r in regimes])
        ax.set_ylim(0, 100)
        ax.set_title("Win rate vs Roll")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Model vs Roll performance by regime", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = FIG_DIR / "performance_by_regime.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Figure not saved: {e}")

    # Optionally save table to CSV
    table_path = RESULTS_DIR / "performance_by_regime.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()
