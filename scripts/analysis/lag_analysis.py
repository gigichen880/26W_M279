"""
Lag analysis: when does K=4 detect crisis regimes, and turnover by period.

Uses backtest parquet (one row per date; model_* and roll_* columns; no 'method' column).
Run from repo root: python scripts/analysis/lag_analysis.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        df = df.reset_index()
        if "date" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "date"})
    return df


def main():
    # Load backtest data
    df_k1 = pd.read_parquet(REPO_ROOT / "results/ablation_k/backtest_k1.parquet")
    df_k4 = pd.read_parquet(REPO_ROOT / "results/ablation_k/backtest_k4.parquet")
    df_k1 = _ensure_date_column(df_k1)
    df_k4 = _ensure_date_column(df_k4)

    # Crisis flag (same definition as baseline_k1_test)
    crisis_dates = [
        ("2015-08-01", "2016-02-29"),
        ("2018-10-01", "2018-12-31"),
        ("2020-02-20", "2020-04-30"),
    ]

    def in_crisis(date):
        for start, end in crisis_dates:
            if start <= str(date)[:10] <= end:
                return True
        return False

    df_k1["is_crisis"] = df_k1["date"].apply(in_crisis)
    df_k4["is_crisis"] = df_k4["date"].apply(in_crisis)

    # ----- Lag analysis: when does K=4 assign crisis regime (3)? -----
    crisis_periods = [
        ("2015-08-01", "2015-09-30", "China Selloff"),
        ("2018-10-01", "2018-12-31", "Q4 2018"),
        ("2020-02-20", "2020-03-31", "COVID Crash"),
    ]

    for start, end, label in crisis_periods:
        period = df_k4[(df_k4["date"].astype(str).str[:10] >= start) & (df_k4["date"].astype(str).str[:10] <= end)]
        crisis_detected = period.loc[period["regime_assigned"] == 3, "date"].min()
        if pd.isna(crisis_detected):
            print(f"{label}: Crisis regime NEVER detected during period!")
        else:
            start_ts = pd.Timestamp(start)
            crisis_ts = pd.to_datetime(crisis_detected)
            lag_days = (crisis_ts - start_ts).days
            print(f"{label}: Crisis detected on {crisis_detected}, lag = {lag_days} days")
        print(f"  Regime assignments: {period['regime_assigned'].value_counts().to_dict()}")
        print()

    # ----- Figure: crisis vs normal Sharpe (Roll, K=1, K=4) -----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        methods = ["Roll", "K=1\n(similarity)", "K=4\n(regimes)"]
        colors = ["gray", "steelblue", "coral"]

        roll_crisis = df_k1.loc[df_k1["is_crisis"], "roll_gmvp_sharpe"].mean()
        roll_normal = df_k1.loc[~df_k1["is_crisis"], "roll_gmvp_sharpe"].mean()
        k1_crisis = df_k1.loc[df_k1["is_crisis"], "model_gmvp_sharpe"].mean()
        k1_normal = df_k1.loc[~df_k1["is_crisis"], "model_gmvp_sharpe"].mean()
        k4_crisis = df_k4.loc[df_k4["is_crisis"], "model_gmvp_sharpe"].mean()
        k4_normal = df_k4.loc[~df_k4["is_crisis"], "model_gmvp_sharpe"].mean()

        crisis_sharpe = [roll_crisis, k1_crisis, k4_crisis]
        normal_sharpe = [roll_normal, k1_normal, k4_normal]

        ax1.bar(methods, crisis_sharpe, color=colors, alpha=0.7, edgecolor="black")
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax1.set_ylabel("Sharpe Ratio", fontsize=12)
        ax1.set_title("Crisis Periods (~20% of data)", fontsize=13, fontweight="bold")
        ax1.set_ylim([-0.3, 0.1])
        ax1.grid(axis="y", alpha=0.3)
        for i, v in enumerate(crisis_sharpe):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

        ax2.bar(methods, normal_sharpe, color=colors, alpha=0.7, edgecolor="black")
        ax2.set_ylabel("Sharpe Ratio", fontsize=12)
        ax2.set_title("Normal Periods (~80% of data)", fontsize=13, fontweight="bold")
        ax2.set_ylim([0.7, 1.4])
        ax2.grid(axis="y", alpha=0.3)
        for i, v in enumerate(normal_sharpe):
            ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

        fig.suptitle("K Ablation: Performance by Market Regime", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        outpath = REPO_ROOT / "results/figs_regime_similarity/ablation_crisis_vs_normal.png"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outpath}")

    # ----- Turnover by period (one row per date; no 'method' column) -----
    k1_crisis_turn = df_k1.loc[df_k1["is_crisis"], "model_turnover_l1"].mean()
    k1_normal_turn = df_k1.loc[~df_k1["is_crisis"], "model_turnover_l1"].mean()
    k4_crisis_turn = df_k4.loc[df_k4["is_crisis"], "model_turnover_l1"].mean()
    k4_normal_turn = df_k4.loc[~df_k4["is_crisis"], "model_turnover_l1"].mean()

    print("Turnover by period:")
    print(f"K=1: Crisis {k1_crisis_turn:.3f}, Normal {k1_normal_turn:.3f}")
    print(f"K=4: Crisis {k4_crisis_turn:.3f}, Normal {k4_normal_turn:.3f}")


if __name__ == "__main__":
    main()
