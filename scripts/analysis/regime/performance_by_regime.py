"""
Implementation moved from scripts/analysis/performance_by_regime.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.utils import resolve_backtest_path, resolve_figs_dir


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
        var_model = pd.to_numeric(df.get("model_gmvp_var"), errors="coerce")
        var_roll = pd.to_numeric(df.get("roll_gmvp_var"), errors="coerce")
        turn_model = pd.to_numeric(df.get("model_turnover_l1"), errors="coerce")
        turn_roll = pd.to_numeric(df.get("roll_turnover_l1"), errors="coerce")

    regime = df["regime_assigned"]
    # Infer number of regimes from observed labels (falls back to 4)
    try:
        K = int(np.nanmax(pd.to_numeric(regime, errors="coerce"))) + 1
        if not np.isfinite(K) or K <= 0:
            K = 4
    except Exception:
        K = 4

    def _win_pct(model_s: pd.Series, roll_s: pd.Series, *, higher_is_better: bool) -> float:
        valid = model_s.notna() & roll_s.notna()
        if valid.sum() == 0:
            return float("nan")
        if higher_is_better:
            return float((model_s[valid] > roll_s[valid]).mean() * 100.0)
        return float((model_s[valid] < roll_s[valid]).mean() * 100.0)

    rows = []
    for k in range(K):
        mask = regime == k
        n_days = int(mask.sum())
        if n_days == 0:
            continue
        row = {
            "Regime": int(k),
            "N_days": n_days,
            "Err_Model": float(err_model.loc[mask].mean()),
            "Err_Roll": float(err_roll.loc[mask].mean()),
            "Win%_Err": _win_pct(err_model.loc[mask], err_roll.loc[mask], higher_is_better=False),
        }
        if not is_vol:
            row["Sharpe_Model"] = float(sharpe_model.loc[mask].mean())
            row["Sharpe_Roll"] = float(sharpe_roll.loc[mask].mean())
            row["Win%_Sharpe"] = _win_pct(sharpe_model.loc[mask], sharpe_roll.loc[mask], higher_is_better=True)
            if "model_gmvp_var" in df.columns and "roll_gmvp_var" in df.columns:
                row["Var_Model"] = float(var_model.loc[mask].mean())
                row["Var_Roll"] = float(var_roll.loc[mask].mean())
                row["Win%_Var"] = _win_pct(var_model.loc[mask], var_roll.loc[mask], higher_is_better=False)
            if "model_turnover_l1" in df.columns and "roll_turnover_l1" in df.columns:
                row["Turnover_Model"] = float(turn_model.loc[mask].mean())
                row["Turnover_Roll"] = float(turn_roll.loc[mask].mean())
                row["Win%_Turnover"] = _win_pct(turn_model.loc[mask], turn_roll.loc[mask], higher_is_better=False)
        rows.append(row)

    table = pd.DataFrame(rows)
    fig_dir = resolve_figs_dir("regime_volatility" if is_vol else "regime_covariance")
    print(table.to_string(index=False))

    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        regimes = table["Regime"].astype(int)
        x = np.arange(len(regimes))
        w = 0.35

        def _plot_triptych(metric_prefix: str, *, ylabel: str, win_col: str, win_ylabel: str, out_name: str, add_zero_line: bool = False):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax0, ax1, ax2 = axes

            # Panel 1: error (kept consistent across all outputs)
            ax0.bar(x - w / 2, table["Err_Model"], w, label="Model", color="steelblue", alpha=0.9)
            ax0.bar(x + w / 2, table["Err_Roll"], w, label="Roll", color="gray", alpha=0.7)
            ax0.set_ylabel(f"Mean {err_label}")
            ax0.set_xlabel("Regime")
            ax0.set_xticks(x)
            ax0.set_xticklabels([f"Regime {r}" for r in regimes])
            ax0.legend(loc="best")
            ax0.grid(True, alpha=0.3, axis="y")

            # Panel 2: chosen metric (model vs roll)
            m_col = f"{metric_prefix}_Model"
            r_col = f"{metric_prefix}_Roll"
            ax1.bar(x - w / 2, table[m_col], w, label="Model", color="steelblue", alpha=0.9)
            ax1.bar(x + w / 2, table[r_col], w, label="Roll", color="gray", alpha=0.7)
            ax1.set_ylabel(ylabel)
            ax1.set_xlabel("Regime")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Regime {r}" for r in regimes])
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3, axis="y")
            if add_zero_line:
                ax1.axhline(0, color="black", linewidth=0.5)

            # Panel 3: win% for that metric
            ax2.bar(x, table[win_col], color="steelblue", alpha=0.9)
            ax2.axhline(50, color="gray", linestyle="--", linewidth=1, label="50%")
            ax2.set_ylabel(win_ylabel)
            ax2.set_xlabel("Regime")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Regime {r}" for r in regimes])
            ax2.set_ylim(0, 100)
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            out_path = fig_dir / out_name
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

        # Existing figure (Fro + Sharpe + win% on Fro) kept for backwards compatibility
        if is_vol:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            ax0, ax1 = axes
            ax0.bar(x - w / 2, table["Err_Model"], w, label="Model", color="steelblue", alpha=0.9)
            ax0.bar(x + w / 2, table["Err_Roll"], w, label="Roll", color="gray", alpha=0.7)
            ax0.set_ylabel(f"Mean {err_label}")
            ax0.set_xlabel("Regime")
            ax0.set_xticks(x)
            ax0.set_xticklabels([f"Regime {r}" for r in regimes])
            ax0.legend(loc="best")
            ax0.grid(True, alpha=0.3, axis="y")

            ax1.bar(x, table["Win%_Err"], color="steelblue", alpha=0.9)
            ax1.axhline(50, color="gray", linestyle="--", linewidth=1, label="50%")
            ax1.set_ylabel(f"Win % (model {err_label} < roll)")
            ax1.set_xlabel("Regime")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Regime {r}" for r in regimes])
            ax1.set_ylim(0, 100)
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            out_path = fig_dir / "performance_by_regime.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            # Back-compat: Fro + Sharpe + win% on Fro
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax0, ax1, ax2 = axes
            ax0.bar(x - w / 2, table["Err_Model"], w, label="Model", color="steelblue", alpha=0.9)
            ax0.bar(x + w / 2, table["Err_Roll"], w, label="Roll", color="gray", alpha=0.7)
            ax0.set_ylabel(f"Mean {err_label}")
            ax0.set_xlabel("Regime")
            ax0.set_xticks(x)
            ax0.set_xticklabels([f"Regime {r}" for r in regimes])
            ax0.legend(loc="best")
            ax0.grid(True, alpha=0.3, axis="y")

            ax1.bar(x - w / 2, table["Sharpe_Model"], w, label="Model", color="steelblue", alpha=0.9)
            ax1.bar(x + w / 2, table["Sharpe_Roll"], w, label="Roll", color="gray", alpha=0.7)
            ax1.set_ylabel("Mean GMVP Sharpe")
            ax1.set_xlabel("Regime")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Regime {r}" for r in regimes])
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3, axis="y")
            ax1.axhline(0, color="black", linewidth=0.5)

            ax2.bar(x, table["Win%_Err"], color="steelblue", alpha=0.9)
            ax2.axhline(50, color="gray", linestyle="--", linewidth=1, label="50%")
            ax2.set_ylabel(f"Win % (model {err_label} < roll)")
            ax2.set_xlabel("Regime")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Regime {r}" for r in regimes])
            ax2.set_ylim(0, 100)
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            out_path = fig_dir / "performance_by_regime.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

            # New figures requested: GMVP Sharpe / variance / turnover (with their own win%)
            if "Win%_Sharpe" in table.columns:
                _plot_triptych(
                    "Sharpe",
                    ylabel="Mean GMVP Sharpe",
                    win_col="Win%_Sharpe",
                    win_ylabel="Win % (model sharpe > roll)",
                    out_name="performance_by_regime_gmvp_sharpe.png",
                    add_zero_line=True,
                )
            if "Win%_Var" in table.columns:
                _plot_triptych(
                    "Var",
                    ylabel="Mean GMVP Variance",
                    win_col="Win%_Var",
                    win_ylabel="Win % (model var < roll)",
                    out_name="performance_by_regime_gmvp_var.png",
                    add_zero_line=False,
                )
            if "Win%_Turnover" in table.columns:
                _plot_triptych(
                    "Turnover",
                    ylabel="Mean Turnover L1",
                    win_col="Win%_Turnover",
                    win_ylabel="Win % (model turnover < roll)",
                    out_name="performance_by_regime_turnover_l1.png",
                    add_zero_line=False,
                )
    except Exception as e:
        print(f"Figure not saved: {e}")

    table_path = fig_dir / "performance_by_regime.csv"
    table.to_csv(table_path, index=False)


if __name__ == "__main__":
    main()

