# scripts/viz_backtest_results.py
# Visualize results/regime_similarity_backtest.csv
#
# Usage:
#   python scripts/viz_backtest_results.py \
#       --csv results/regime_similarity_backtest.csv \
#       --outdir results/figs_regime_similarity \
#       --start 2018-01-01 --end 2020-05-31
#
# Notes:
# - Uses matplotlib only (no seaborn).
# - Produces time-series, rolling summaries, distributions, and key scatter plots.

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def _maybe_slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    if start:
        out = out.loc[pd.to_datetime(start) :]
    if end:
        out = out.loc[: pd.to_datetime(end)]
    return out


def _ensure_dir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_timeseries(df: pd.DataFrame, outdir: str) -> None:
    # Prefer log scale for heavy-tailed metrics
    panels = [
        ("fro", "Frobenius error"),
        ("kl", "KL divergence"),
        ("nll", "Gaussian NLL (avg over H days)"),
        ("logeuc", "Log-Euclidean distance"),
        ("corr_spearman", "Corr upper-tri Spearman"),
        ("corr_offdiag_fro", "Corr off-diagonal Fro"),
        ("cond_ratio", "Condition number ratio (hat/true)"),
        ("turnover_l1", "Min-var turnover L1"),
        ("port_mse_logvar", "Multi-port MSE log(var)"),
    ]
    panels = [(c, t) for (c, t) in panels if c in df.columns]

    if not panels:
        print("[warn] No known metric columns found for time series plot.")
        return

    n = len(panels)
    fig = plt.figure(figsize=(12, 2.2 * n))
    for i, (col, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(n, 1, i)
        s = df[col].astype(float)

        ax.plot(s.index, s.values)
        ax.set_title(title)

        # Heavy-tailed: show log10 where appropriate (only if positive)
        if col in {"kl", "nll", "corr_offdiag_fro", "logeuc", "port_mse_logvar"}:
            if np.all(np.isfinite(s.values)) and np.nanmin(s.values) > 0:
                ax.set_yscale("log")

        ax.grid(True, alpha=0.3)
    _savefig(os.path.join(outdir, "timeseries_metrics.png"))


def plot_rolling(df: pd.DataFrame, outdir: str, window: int = 21) -> None:
    cols = [c for c in ["fro", "kl", "nll", "logeuc", "corr_spearman", "turnover_l1"] if c in df.columns]
    if not cols:
        return

    roll = df[cols].rolling(window, min_periods=max(5, window // 3)).median()

    fig = plt.figure(figsize=(12, 2.2 * len(cols)))
    for i, c in enumerate(cols, start=1):
        ax = fig.add_subplot(len(cols), 1, i)
        ax.plot(df.index, df[c].values, alpha=0.25, label=c)
        ax.plot(roll.index, roll[c].values, linewidth=2.0, label=f"{window}d rolling median")
        ax.set_title(f"{c}: raw + {window}d rolling median")
        if c in {"kl", "nll", "logeuc"} and np.nanmin(df[c].values) > 0:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    _savefig(os.path.join(outdir, f"rolling_median_{window}d.png"))


def plot_distributions(df: pd.DataFrame, outdir: str) -> None:
    cols = [c for c in ["fro", "kl", "nll", "logeuc", "corr_spearman", "turnover_l1", "w_hhi", "w_max_abs"] if c in df.columns]
    if not cols:
        return

    n = len(cols)
    fig = plt.figure(figsize=(12, 2.6 * n))
    for i, c in enumerate(cols, start=1):
        ax = fig.add_subplot(n, 1, i)
        x = df[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if x.size == 0:
            ax.set_title(f"{c} (empty)")
            continue

        ax.hist(x, bins=60, density=False)
        ax.set_title(f"Distribution: {c} (n={x.size})")
        ax.grid(True, alpha=0.3)

        # helpful vertical lines
        q = np.quantile(x, [0.5, 0.9, 0.99])
        for qq in q:
            ax.axvline(qq, linestyle="--", alpha=0.6)

    _savefig(os.path.join(outdir, "distributions.png"))


def plot_scatter(df: pd.DataFrame, outdir: str) -> None:
    pairs = [
        ("logeuc", "nll"),
        ("kl", "nll"),
        ("corr_spearman", "fro"),
        ("corr_offdiag_fro", "fro"),
        ("cond_ratio", "turnover_l1"),
        ("port_mse_logvar", "turnover_l1"),
    ]
    pairs = [(x, y) for (x, y) in pairs if x in df.columns and y in df.columns]
    if not pairs:
        return

    for xcol, ycol in pairs:
        sub = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 10:
            continue

        plt.figure(figsize=(8.5, 6))
        plt.scatter(sub[xcol].values, sub[ycol].values, s=12, alpha=0.6)
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(f"{ycol} vs {xcol} (n={len(sub)})")
        plt.grid(True, alpha=0.3)

        # log axes when it makes sense
        if ycol in {"kl", "nll", "logeuc", "port_mse_logvar"} and np.nanmin(sub[ycol].values) > 0:
            plt.yscale("log")
        if xcol in {"kl", "nll", "logeuc", "port_mse_logvar"} and np.nanmin(sub[xcol].values) > 0:
            plt.xscale("log")

        _savefig(os.path.join(outdir, f"scatter_{ycol}_vs_{xcol}.png"))


def plot_worst_dates(df: pd.DataFrame, outdir: str, metric: str, topk: int = 10) -> None:
    if metric not in df.columns:
        return
    s = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return

    worst = s.sort_values(ascending=False).head(topk)

    plt.figure(figsize=(10, 4))
    plt.bar([d.strftime("%Y-%m-%d") for d in worst.index], worst.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Worst {topk} dates by {metric}")
    plt.grid(True, axis="y", alpha=0.3)
    _savefig(os.path.join(outdir, f"worst_{topk}_{metric}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/regime_similarity_backtest.csv")
    ap.add_argument("--outdir", type=str, default="results/figs_regime_similarity")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--roll", type=int, default=21)
    args = ap.parse_args()

    df = _read_results(args.csv)
    df = _maybe_slice(df, args.start, args.end)
    _ensure_dir(args.outdir)

    # Basic sanity print
    print("Loaded:", args.csv)
    print("Date range:", df.index.min(), "->", df.index.max())
    print("Columns:", list(df.columns))
    print(df.describe(include="all").T.head(20))

    plot_timeseries(df, args.outdir)
    plot_rolling(df, args.outdir, window=args.roll)
    plot_distributions(df, args.outdir)
    plot_scatter(df, args.outdir)

    # Worst-date bar charts for key metrics
    for m in ["kl", "nll", "logeuc", "corr_offdiag_fro", "port_mse_logvar", "turnover_l1"]:
        plot_worst_dates(df, args.outdir, metric=m, topk=10)

    print(f"\nSaved figures to: {args.outdir}")
    print("Key files:")
    print("  timeseries_metrics.png")
    print(f"  rolling_median_{args.roll}d.png")
    print("  distributions.png")
    print("  scatter_*.png")
    print("  worst_10_*.png")


if __name__ == "__main__":
    main()