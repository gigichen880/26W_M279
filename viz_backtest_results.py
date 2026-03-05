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
# - Produces time-series, rolling summaries, distributions, scatter plots,
#   and portfolio calibration diagnostics.

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _metric_cols(df: pd.DataFrame, metric: str) -> dict:
    """
    Return available columns for a given metric across methods.
    Expected naming: model_{metric}, pers_{metric}, roll_{metric}, shrink_{metric}.
    """
    out = {}
    for m in ["model", "pers", "roll", "shrink"]:
        c = f"{m}_{metric}"
        if c in df.columns:
            out[m] = c
    return out


def _is_higher_better(metric: str) -> bool:
    # Only corr_spearman is higher-is-better in your current suite.
    return metric == "corr_spearman"


def _skill_series(model: pd.Series, base: pd.Series, metric: str) -> pd.Series:
    """
    Skill per date:
      - losses: ratio model/base (good < 1)
      - corr_spearman: diff model-base (good > 0)
    """
    model = model.astype(float).replace([np.inf, -np.inf], np.nan)
    base = base.astype(float).replace([np.inf, -np.inf], np.nan)

    if _is_higher_better(metric):
        return model - base
    else:
        return model / np.maximum(base, 1e-12)


def _win_series(model: pd.Series, base: pd.Series, metric: str) -> pd.Series:
    """
    Win indicator per date:
      - losses: 1(model < base)
      - corr_spearman: 1(model > base)
    """
    model = model.astype(float).replace([np.inf, -np.inf], np.nan)
    base = base.astype(float).replace([np.inf, -np.inf], np.nan)
    ok = model.notna() & base.notna()

    win = pd.Series(np.nan, index=model.index)
    if _is_higher_better(metric):
        win.loc[ok] = (model.loc[ok] > base.loc[ok]).astype(float)
    else:
        win.loc[ok] = (model.loc[ok] < base.loc[ok]).astype(float)
    return win

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
        ("port_mae_logvar", "Multi-port MAE log(var)"),
        ("w_hhi", "GMVP weight concentration (HHI)"),
        ("w_max_abs", "GMVP max abs weight"),
    ]
    panels = [(c, t) for (c, t) in panels if c in df.columns]

    if not panels:
        print("[warn] No known metric columns found for time series plot.")
        return

    n = len(panels)
    fig = plt.figure(figsize=(12, 2.2 * n))
    for i, (col, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(n, 1, i)
        s = df[col].astype(float).replace([np.inf, -np.inf], np.nan)

        ax.plot(s.index, s.values)
        ax.set_title(title)

        # Heavy-tailed: show log scale where appropriate (only if strictly positive)
        if col in {"kl", "nll", "corr_offdiag_fro", "logeuc", "port_mse_logvar", "port_mae_logvar"}:
            vals = s.values
            if np.nanmin(vals) > 0:
                ax.set_yscale("log")

        ax.grid(True, alpha=0.3)
    _savefig(os.path.join(outdir, "timeseries_metrics.png"))


def plot_rolling(df: pd.DataFrame, outdir: str, window: int = 21) -> None:
    cols = [c for c in ["fro", "kl", "nll", "logeuc", "corr_spearman", "turnover_l1", "port_mse_logvar"] if c in df.columns]
    if not cols:
        return

    roll = df[cols].rolling(window, min_periods=max(5, window // 3)).median()

    fig = plt.figure(figsize=(12, 2.2 * len(cols)))
    for i, c in enumerate(cols, start=1):
        ax = fig.add_subplot(len(cols), 1, i)
        raw = df[c].replace([np.inf, -np.inf], np.nan).astype(float)
        ax.plot(df.index, raw.values, alpha=0.25, label=c)
        ax.plot(roll.index, roll[c].values, linewidth=2.0, label=f"{window}d rolling median")
        ax.set_title(f"{c}: raw + {window}d rolling median")
        if c in {"kl", "nll", "logeuc", "port_mse_logvar"} and np.nanmin(raw.values) > 0:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    _savefig(os.path.join(outdir, f"rolling_median_{window}d.png"))


def plot_distributions(df: pd.DataFrame, outdir: str) -> None:
    cols = [c for c in [
        "fro", "kl", "nll", "logeuc",
        "corr_spearman", "turnover_l1",
        "pred_var", "real_var",
        "port_mse_var", "port_mse_logvar", "port_mae_logvar",
        "w_hhi", "w_max_abs"
    ] if c in df.columns]
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

def plot_portfolio_calibration(df: pd.DataFrame, outdir: str) -> None:
    if "pred_var" not in df.columns or "real_var" not in df.columns:
        return

    sub = df[["pred_var", "real_var"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 10:
        return

    pred = sub["pred_var"].astype(float).values
    real = sub["real_var"].astype(float).values

    # Time series: predicted vs realized GMVP variance
    plt.figure(figsize=(12, 4))
    plt.plot(sub.index, pred, label="pred_var")
    plt.plot(sub.index, real, label="real_var")
    plt.title("GMVP variance: predicted vs realized")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    _savefig(os.path.join(outdir, "gmvp_pred_vs_real_timeseries.png"))

    # Risk ratio time series: real/pred
    ratio = real / np.maximum(pred, 1e-12)
    plt.figure(figsize=(12, 4))
    plt.plot(sub.index, ratio)
    plt.axhline(1.0, linestyle="--", alpha=0.7)
    plt.title("GMVP risk ratio: real_var / pred_var (1.0 = perfect)")
    plt.grid(True, alpha=0.3)
    _savefig(os.path.join(outdir, "gmvp_risk_ratio_timeseries.png"))

    # Calibration scatter: real vs pred (log-log if positive)
    plt.figure(figsize=(6.5, 6))
    plt.scatter(pred, real, s=12, alpha=0.6)
    lo = float(np.nanmin([pred.min(), real.min()]))
    hi = float(np.nanmax([pred.max(), real.max()]))
    plt.plot([lo, hi], [lo, hi], linestyle="--", alpha=0.7)
    plt.xlabel("pred_var")
    plt.ylabel("real_var")
    plt.title(f"GMVP calibration scatter (n={len(sub)})")
    plt.grid(True, alpha=0.3)
    if lo > 0:
        plt.xscale("log")
        plt.yscale("log")
    _savefig(os.path.join(outdir, "gmvp_calibration_scatter.png"))


def plot_portfolio_diagnostics(df: pd.DataFrame, outdir: str) -> None:
    cols = [c for c in ["port_mse_var", "port_mse_logvar", "port_mae_logvar", "turnover_l1", "w_hhi", "w_max_abs"] if c in df.columns]
    if not cols:
        return

    fig = plt.figure(figsize=(12, 2.2 * len(cols)))
    for i, c in enumerate(cols, start=1):
        ax = fig.add_subplot(len(cols), 1, i)
        s = df[c].replace([np.inf, -np.inf], np.nan).astype(float)
        ax.plot(s.index, s.values)
        ax.set_title(c)
        if c in {"port_mse_var", "port_mse_logvar", "port_mae_logvar"} and np.nanmin(s.values) > 0:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    _savefig(os.path.join(outdir, "portfolio_diagnostics_timeseries.png"))

def plot_method_overlays(df: pd.DataFrame, outdir: str) -> None:
    """
    For each metric, overlay model/pers/roll/shrink over time.
    Produces a single multi-panel figure.
    """
    metrics = [
        ("fro", "Frobenius error"),
        ("kl", "KL divergence"),
        ("nll", "Gaussian NLL"),
        ("logeuc", "Log-Euclidean distance"),
        ("corr_spearman", "Corr Spearman"),
        ("corr_offdiag_fro", "Corr offdiag Fro"),
        ("port_mse_logvar", "Multi-port MSE log(var)"),
        ("turnover_l1", "Turnover L1"),
    ]

    # only keep metrics that have model_ columns
    metrics = [(m, t) for (m, t) in metrics if f"model_{m}" in df.columns]
    if not metrics:
        print("[warn] No model_* columns found for method overlays.")
        return

    n = len(metrics)
    fig = plt.figure(figsize=(12, 2.4 * n))

    for i, (metric, title) in enumerate(metrics, start=1):
        ax = fig.add_subplot(n, 1, i)

        cols = _metric_cols(df, metric)
        for m, c in cols.items():
            s = df[c].astype(float).replace([np.inf, -np.inf], np.nan)
            ax.plot(s.index, s.values, label=m, alpha=0.9 if m == "model" else 0.6)

        ax.set_title(f"{title}: model vs baselines")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=4)

        # log-scale for heavy-tailed positive losses
        if metric in {"kl", "nll", "logeuc", "corr_offdiag_fro", "port_mse_logvar"}:
            # check positivity across all plotted series
            all_vals = []
            for c in cols.values():
                vv = df[c].astype(float).replace([np.inf, -np.inf], np.nan).values
                all_vals.append(vv)
            all_vals = np.concatenate([v for v in all_vals if v.size > 0])
            if all_vals.size > 0 and np.nanmin(all_vals) > 0:
                ax.set_yscale("log")

    _savefig(os.path.join(outdir, "method_overlays.png"))

def plot_skill_timeseries(df: pd.DataFrame, outdir: str) -> None:
    """
    Plot per-date skill of model vs each baseline for several metrics.
    Loss metrics: ratio model/base (<1 good)
    corr_spearman: diff model-base (>0 good)
    """
    metrics = ["fro", "kl", "nll", "logeuc", "corr_spearman", "corr_offdiag_fro", "port_mse_logvar"]
    metrics = [m for m in metrics if f"model_{m}" in df.columns]
    if not metrics:
        return

    fig = plt.figure(figsize=(12, 2.4 * len(metrics)))
    for i, metric in enumerate(metrics, start=1):
        ax = fig.add_subplot(len(metrics), 1, i)

        cols = _metric_cols(df, metric)
        if "model" not in cols:
            continue

        model_s = df[cols["model"]]
        for b in ["pers", "roll", "shrink"]:
            if b not in cols:
                continue
            skill = _skill_series(model_s, df[cols[b]], metric)
            ax.plot(skill.index, skill.values, label=f"vs {b}", alpha=0.85)

        # reference line
        if _is_higher_better(metric):
            ax.axhline(0.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill (diff): model - baseline for {metric}")
        else:
            ax.axhline(1.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill (ratio): model / baseline for {metric}  (<1 better)")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=3)

        # for ratio metrics, log scale can help if all positive
        if not _is_higher_better(metric):
            vals = ax.lines[0].get_ydata()
            if np.size(vals) and np.nanmin(vals) > 0:
                ax.set_yscale("log")

    _savefig(os.path.join(outdir, "skill_timeseries.png"))

def plot_rolling_winrates(df: pd.DataFrame, outdir: str, window: int = 63) -> None:
    """
    Rolling win-rate of model vs baselines.
    Win definition:
      - losses: model < baseline
      - corr_spearman: model > baseline
    """
    metrics = ["fro", "kl", "nll", "logeuc", "corr_spearman", "port_mse_logvar"]
    metrics = [m for m in metrics if f"model_{m}" in df.columns]
    if not metrics:
        return

    fig = plt.figure(figsize=(12, 2.4 * len(metrics)))
    for i, metric in enumerate(metrics, start=1):
        ax = fig.add_subplot(len(metrics), 1, i)

        cols = _metric_cols(df, metric)
        if "model" not in cols:
            continue

        model_s = df[cols["model"]]
        for b in ["pers", "roll", "shrink"]:
            if b not in cols:
                continue
            win = _win_series(model_s, df[cols[b]], metric)
            wr = win.rolling(window, min_periods=max(10, window // 3)).mean()
            ax.plot(wr.index, wr.values, label=f"vs {b}")

        ax.axhline(0.5, linestyle="--", alpha=0.7)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{window}d rolling win-rate for {metric} (model vs baseline)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=3)

    _savefig(os.path.join(outdir, f"rolling_winrate_{window}d.png"))

def plot_skill_distributions(df: pd.DataFrame, outdir: str) -> None:
    metrics = ["fro", "kl", "nll", "logeuc", "corr_spearman", "port_mse_logvar"]
    metrics = [m for m in metrics if f"model_{m}" in df.columns]
    if not metrics:
        return

    fig = plt.figure(figsize=(12, 2.8 * len(metrics)))
    for i, metric in enumerate(metrics, start=1):
        ax = fig.add_subplot(len(metrics), 1, i)

        cols = _metric_cols(df, metric)
        if "model" not in cols:
            continue
        model_s = df[cols["model"]]

        for b in ["pers", "roll", "shrink"]:
            if b not in cols:
                continue
            skill = _skill_series(model_s, df[cols[b]], metric).replace([np.inf, -np.inf], np.nan).dropna()
            if len(skill) < 10:
                continue
            ax.hist(skill.values, bins=60, alpha=0.35, label=f"vs {b}")

        if _is_higher_better(metric):
            ax.axvline(0.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill distribution (diff): model - baseline for {metric}  (>0 better)")
        else:
            ax.axvline(1.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill distribution (ratio): model / baseline for {metric}  (<1 better)")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=3)

    _savefig(os.path.join(outdir, "skill_distributions.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/regime_similarity_backtest.csv")
    ap.add_argument("--outdir", type=str, default="results/figs_regime_similarity")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--roll", type=int, default=21)
    ap.add_argument("--winroll", type=int, default=63)
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

    # Portfolio visuals
    plot_portfolio_calibration(df, args.outdir)
    plot_portfolio_diagnostics(df, args.outdir)

    # # Worst-date bar charts for key metrics
    # for m in ["kl", "nll", "logeuc", "corr_offdiag_fro", "port_mse_logvar", "turnover_l1"]:
    #     plot_worst_dates(df, args.outdir, metric=m, topk=10)

    plot_method_overlays(df, args.outdir)
    plot_skill_timeseries(df, args.outdir)
    plot_rolling_winrates(df, args.outdir, window=args.winroll)
    plot_skill_distributions(df, args.outdir)

    print(f"\nSaved figures to: {args.outdir}")
    print("Key files:")
    print("  timeseries_metrics.png")
    print(f"  rolling_median_{args.roll}d.png")
    print("  distributions.png")
    print("  scatter_*.png")
    print("  worst_10_*.png")
    print("  gmvp_pred_vs_real_timeseries.png")
    print("  gmvp_risk_ratio_timeseries.png")
    print("  gmvp_calibration_scatter.png")
    print("  portfolio_diagnostics_timeseries.png")
    print("  method_overlays.png")
    print("  skill_timeseries.png")
    print(f"  rolling_winrate_{args.winroll}d.png")
    print("  skill_distributions.png")


if __name__ == "__main__":
    main()