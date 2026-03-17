"""
Implementation moved from scripts/analysis/performance_by_regime.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.utils.paths import resolve_backtest_path, resolve_figs_dir


def _detect_target(df: pd.DataFrame) -> str:
    return "volatility" if "model_vol_mse" in df.columns else "covariance"


def main() -> None:
    ap = argparse.ArgumentParser(description="Model vs Roll by regime (cov or vol)")
    ap.add_argument("--input", default=None, help="Backtest parquet/csv (default: auto-detect cov/vol)")
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
        print("No regime_assigned column. Re-run backtest with regime saving.")
        return

    target = args.target
    if target == "auto":
        target = _detect_target(df)
    is_vol = target == "volatility"

    if is_vol:
        if "model_vol_mse" not in df.columns or "roll_vol_mse" not in df.columns:
            print(
                "Volatility backtest missing model_vol_mse/roll_vol_mse. "
                "Re-run backtest: python run_backtest.py --config configs/regime_volatility.yaml"
            )
            return
        err_model = pd.to_numeric(df["model_vol_mse"], errors="coerce")
        err_roll = pd.to_numeric(df["roll_vol_mse"], errors="coerce")
        err_label = "Vol MSE"
    else:
        err_model = pd.to_numeric(df["model_fro"], errors="coerce")
        err_roll = pd.to_numeric(df["roll_fro"], errors="coerce")
        err_label = "Fro"
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
        em = err_model.loc[mask].mean()
        er = err_roll.loc[mask].mean()
        sub_em = err_model.loc[mask]
        sub_er = err_roll.loc[mask]
        valid = sub_em.notna() & sub_er.notna()
        win_pct = (sub_em[valid] < sub_er[valid]).mean() * 100 if valid.sum() > 0 else np.nan
        row = {"Regime": k, "N_days": n_days, "Err_Model": em, "Err_Roll": er, "Win%": win_pct}
        if not is_vol:
            row["Sharpe_Model"] = sharpe_model.loc[mask].mean()
            row["Sharpe_Roll"] = sharpe_roll.loc[mask].mean()
        rows.append(row)

    table = pd.DataFrame(rows)
    fig_dir = resolve_figs_dir("regime_volatility" if is_vol else "regime_covariance")
    print(table.to_string(index=False))

    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        n_panels = 2 if is_vol else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 2:
            axes = list(axes)
        regimes = table["Regime"].astype(int)
        x = np.arange(len(regimes))
        w = 0.35

        ax = axes[0]
        ax.bar(x - w / 2, table["Err_Model"], w, label="Model", color="steelblue", alpha=0.9)
        ax.bar(x + w / 2, table["Err_Roll"], w, label="Roll", color="gray", alpha=0.7)
        ax.set_ylabel(f"Mean {err_label}")
        ax.set_xlabel("Regime")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Regime {r}" for r in regimes])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        if not is_vol:
            ax = axes[1]
            ax.bar(x - w / 2, table["Sharpe_Model"], w, label="Model", color="steelblue", alpha=0.9)
            ax.bar(x + w / 2, table["Sharpe_Roll"], w, label="Roll", color="gray", alpha=0.7)
            ax.set_ylabel("Mean GMVP Sharpe")
            ax.set_xlabel("Regime")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Regime {r}" for r in regimes])
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(0, color="black", linewidth=0.5)
            ax = axes[2]
        else:
            ax = axes[1]

        ax.bar(x, table["Win%"], color="steelblue", alpha=0.9)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50%")
        ax.set_ylabel(f"Win % (model {err_label} < roll)")
        ax.set_xlabel("Regime")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Regime {r}" for r in regimes])
        ax.set_ylim(0, 100)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out_path = fig_dir / "performance_by_regime.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Figure not saved: {e}")

    table_path = fig_dir / "performance_by_regime.csv"
    table.to_csv(table_path, index=False)


if __name__ == "__main__":
    main()

