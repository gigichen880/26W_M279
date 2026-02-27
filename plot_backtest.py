# scripts/plot_backtest_results.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    required = {"raw_anchor", "fro", "kl", "pred_var", "real_var"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def safe_log10(x: pd.Series, eps: float = 1e-30) -> pd.Series:
    return np.log10(np.maximum(x.astype(float).to_numpy(), eps))


def rolling_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(5, w // 5)).mean()


def plot_time_series(df: pd.DataFrame, outdir: str, roll: int = 20) -> None:
    # Frobenius
    plt.figure(figsize=(11, 4))
    plt.plot(df.index, df["fro"], linewidth=1)
    plt.plot(df.index, rolling_mean(df["fro"], roll), linewidth=2)
    plt.title(f"Frobenius Error Over Time (raw + {roll}d rolling mean)")
    plt.xlabel("Date")
    plt.ylabel("||Σ̂ − Σ||_F")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ts_fro.png"), dpi=150)
    plt.close()

    # KL (log scale)
    plt.figure(figsize=(11, 4))
    plt.plot(df.index, df["kl"], linewidth=1)
    plt.plot(df.index, rolling_mean(df["kl"], roll), linewidth=2)
    plt.yscale("log")
    plt.title(f"Gaussian KL Over Time (log y, raw + {roll}d rolling mean)")
    plt.xlabel("Date")
    plt.ylabel("D_KL(Σ || Σ̂)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ts_kl_log.png"), dpi=150)
    plt.close()

    # Pred vs Real variance (two scales can differ a lot, so log y)
    plt.figure(figsize=(11, 4))
    plt.plot(df.index, df["pred_var"], linewidth=1, label="pred_var")
    plt.plot(df.index, df["real_var"], linewidth=1, label="real_var")
    plt.yscale("log")
    plt.title("Min-Var Portfolio Variance: Predicted vs Realized (log y)")
    plt.xlabel("Date")
    plt.ylabel("Variance (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ts_pred_vs_real_var_log.png"), dpi=150)
    plt.close()

    # Real/Pred ratio
    ratio = df["real_var"] / df["pred_var"].replace(0.0, np.nan)
    plt.figure(figsize=(11, 4))
    plt.plot(df.index, ratio, linewidth=1)
    plt.plot(df.index, rolling_mean(ratio, roll), linewidth=2)
    plt.yscale("log")
    plt.title(f"Variance Calibration: real_var / pred_var (log y, + {roll}d roll)")
    plt.xlabel("Date")
    plt.ylabel("real / pred (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ts_real_over_pred_log.png"), dpi=150)
    plt.close()


def plot_distributions(df: pd.DataFrame, outdir: str) -> None:
    # Histograms (log-x for KL often helps)
    plt.figure(figsize=(7, 4))
    plt.hist(df["fro"].to_numpy(), bins=60)
    plt.title("Distribution of Frobenius Error")
    plt.xlabel("||Σ̂ − Σ||_F")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_fro.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(df["kl"].to_numpy(), bins=80)
    plt.yscale("log")
    plt.title("Distribution of KL (log count)")
    plt.xlabel("KL")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_kl_logcount.png"), dpi=150)
    plt.close()

    # log10 KL histogram (often more readable than raw KL)
    plt.figure(figsize=(7, 4))
    plt.hist(safe_log10(df["kl"]), bins=60)
    plt.title("Distribution of log10(KL)")
    plt.xlabel("log10(KL)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_log10_kl.png"), dpi=150)
    plt.close()

    # log10 ratio histogram
    ratio = df["real_var"] / df["pred_var"].replace(0.0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7, 4))
    plt.hist(safe_log10(ratio), bins=60)
    plt.title("Distribution of log10(real_var / pred_var)")
    plt.xlabel("log10(real/pred)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_log10_real_over_pred.png"), dpi=150)
    plt.close()


def plot_calibration(df: pd.DataFrame, outdir: str) -> None:
    # Scatter: predicted vs realized variance (log-log)
    x = df["pred_var"].replace(0.0, np.nan).to_numpy()
    y = df["real_var"].replace(0.0, np.nan).to_numpy()
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Calibration: Predicted vs Realized Variance (log-log)")
    plt.xlabel("pred_var")
    plt.ylabel("real_var")

    # Add y=x line (in log scale)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_pred_vs_real_var_loglog.png"), dpi=150)
    plt.close()

    # Scatter: Frobenius vs KL (both heavy-tailed; log y for KL)
    plt.figure(figsize=(6, 5))
    plt.scatter(df["fro"].to_numpy(), df["kl"].to_numpy(), s=10)
    plt.yscale("log")
    plt.title("Frobenius vs KL (KL log scale)")
    plt.xlabel("Frobenius")
    plt.ylabel("KL (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_fro_vs_kl.png"), dpi=150)
    plt.close()


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    ratio = (df["real_var"] / df["pred_var"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(
        {
            "fro": df["fro"],
            "kl": df["kl"],
            "pred_var": df["pred_var"],
            "real_var": df["real_var"],
            "real_over_pred": ratio,
            "log10_kl": safe_log10(df["kl"]),
            "log10_real_over_pred": safe_log10(ratio.dropna()),
        }
    )
    # robust-ish percentiles
    summary = out.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    return summary


def main(
    csv_path: str = "results/regime_similarity_backtest.csv",
    outdir: str = "results/plots_regime_similarity",
    roll: int = 20,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    df = load_results(csv_path)

    # Print quick diagnostics
    print("Loaded:", df.shape, "| date range:", df.index.min().date(), "->", df.index.max().date())
    print("Missing fraction per column:")
    print((df.isna().mean()).sort_values(ascending=False))

    plot_time_series(df, outdir=outdir, roll=roll)
    plot_distributions(df, outdir=outdir)
    plot_calibration(df, outdir=outdir)

    summary = make_summary_table(df)
    summary.to_csv(os.path.join(outdir, "summary_table.csv"))
    print("\nSaved plots to:", outdir)
    print("Saved summary:", os.path.join(outdir, "summary_table.csv"))


if __name__ == "__main__":
    main()