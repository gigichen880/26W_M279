# scripts/viz_backtest_results.py
# Visualize results/regime_similarity_backtest.csv (full range, no slicing)
#
# Usage:
#   python scripts/viz_backtest_results.py \
#       --csv results/regime_similarity_backtest.csv \
#       --outdir results/figs_regime_similarity
#
# Notes:
# - matplotlib only (no seaborn)
# - Works with method-prefixed columns like: model_fro, mix_fro, shrink_fro, ...

from __future__ import annotations

import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _read_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # normalize inf
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _detect_methods(df: pd.DataFrame) -> List[str]:
    """
    Detect forecasting methods from columns of form '{method}_{metric}'.
    Filters out metadata prefixes like guardrail/raw/floor/apply/etc.
    """
    banned = {
        "guardrail", "raw", "apply", "floor", "mixlambda", "mix_lambda",
        "date", "stat"
    }

    candidates = {}
    for c in df.columns:
        if "_" not in c:
            continue
        m = c.split("_", 1)[0]
        if m in banned:
            continue
        candidates[m] = candidates.get(m, 0) + 1

    # keep only prefixes that appear "enough" times to be real methods
    # (prevents catching random metadata like 'samples' if it exists)
    methods = [m for m, cnt in candidates.items() if cnt >= 5]

    order = ["model", "mix", "roll", "pers", "shrink"]
    out = [m for m in order if m in methods]
    out += sorted([m for m in methods if m not in set(out)])
    return out

def _metric_cols(df: pd.DataFrame, metric: str, methods: List[str]) -> Dict[str, str]:
    out = {}
    for m in methods:
        c = f"{m}_{metric}"
        if c in df.columns:
            out[m] = c
    return out


def _is_higher_better(metric: str) -> bool:
    # Add other higher-is-better metrics here if you want
    return metric in {"corr_spearman", "gmvp_cumret", "gmvp_mean", "gmvp_sharpe"}


def _skill_series(model: pd.Series, base: pd.Series, metric: str) -> pd.Series:
    """
    Skill per date:
      - higher-is-better metrics: diff (model - base) (good > 0)
      - losses/risks: ratio (model / base) (good < 1)
    """
    model = model.astype(float)
    base = base.astype(float)
    ok = model.notna() & base.notna()

    s = pd.Series(np.nan, index=model.index)
    if _is_higher_better(metric):
        s.loc[ok] = model.loc[ok] - base.loc[ok]
    else:
        s.loc[ok] = model.loc[ok] / np.maximum(base.loc[ok], 1e-12)
    return s


def _win_series(model: pd.Series, base: pd.Series, metric: str) -> pd.Series:
    """
    Win indicator per date:
      - higher-is-better metrics: 1(model > base)
      - losses/risks: 1(model < base)
    """
    model = model.astype(float)
    base = base.astype(float)
    ok = model.notna() & base.notna()

    w = pd.Series(np.nan, index=model.index)
    if _is_higher_better(metric):
        w.loc[ok] = (model.loc[ok] > base.loc[ok]).astype(float)
    else:
        w.loc[ok] = (model.loc[ok] < base.loc[ok]).astype(float)
    return w


def _maybe_log_axis(ax: plt.Axes, values: np.ndarray) -> None:
    # Only set log scale if all positive (ignoring NaNs)
    v = values[np.isfinite(values)]
    if v.size and np.nanmin(v) > 0:
        ax.set_yscale("log")


# ----------------------------
# Plots
# ----------------------------
def plot_equity_curves(df: pd.DataFrame, outdir: str, methods: List[str]) -> None:
    """
    Build equity curves by chaining per-date hold-period cumret:
        equity_t = Π (1 + cumret_t)
    Uses columns: {method}_gmvp_cumret
    """
    cols = _metric_cols(df, "gmvp_cumret", methods)
    if not cols:
        print("[warn] No *_gmvp_cumret columns found; skipping equity curves.")
        return

    plt.figure(figsize=(12, 5))
    for m, c in cols.items():
        r = df[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        eq = np.cumprod(1.0 + r)
        plt.plot(df.index, eq, label=m, alpha=0.9 if m in {"mix", "model"} else 0.6)

    plt.title("GMVP Equity Curves (chained from per-date gmvp_cumret)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=min(5, len(cols)))
    _savefig(os.path.join(outdir, "equity_curves_gmvp.png"))


def plot_method_overlays(df: pd.DataFrame, outdir: str, methods: List[str]) -> None:
    """
    Overlay methods over time for key metrics.
    """
    metrics: List[Tuple[str, str]] = [
        ("fro", "Frobenius error"),
        ("kl", "KL divergence"),
        ("stein", "Stein loss"),
        ("logeuc", "Log-Euclidean distance"),
        ("corr_spearman", "Correlation Spearman"),
        ("corr_offdiag_fro", "Correlation offdiag Fro"),
        ("eig_log_mse", "Eigenvalue log-MSE"),
        ("cond_ratio", "Condition ratio"),
        ("gmvp_var", "GMVP realized variance"),
        ("gmvp_vol", "GMVP realized vol"),
        ("gmvp_sharpe", "GMVP Sharpe"),
        ("turnover_l1", "Turnover L1"),
        ("w_hhi", "Weight concentration (HHI)"),
        ("w_max_abs", "Max abs weight"),
        ("w_l1", "Weight L1 norm"),
    ]

    # keep only metrics that exist for at least 2 methods
    keep = []
    for metric, title in metrics:
        cols = _metric_cols(df, metric, methods)
        if len(cols) >= 2:
            keep.append((metric, title))

    if not keep:
        print("[warn] No overlay-able method metrics found.")
        return

    n = len(keep)
    fig = plt.figure(figsize=(12, 2.4 * n))

    heavy_pos = {"kl", "stein", "logeuc", "eig_log_mse"}

    for i, (metric, title) in enumerate(keep, start=1):
        ax = fig.add_subplot(n, 1, i)
        cols = _metric_cols(df, metric, methods)

        for m, c in cols.items():
            s = df[c].astype(float)
            ax.plot(s.index, s.values, label=m, alpha=0.9 if m in {"mix", "model"} else 0.6)

        ax.set_title(f"{title}: methods overlay")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=min(5, len(cols)))

        if metric in heavy_pos:
            all_vals = np.concatenate([
                df[c].astype(float).values for c in cols.values()
            ])
            _maybe_log_axis(ax, all_vals)

    _savefig(os.path.join(outdir, "method_overlays.png"))


def plot_rolling_median(df: pd.DataFrame, outdir: str, methods: List[str], window: int = 21) -> None:
    """
    Rolling median overlays for a few headline metrics (helps see regime phases).
    """
    headline = ["fro", "kl", "stein", "gmvp_var", "gmvp_sharpe", "turnover_l1"]
    keep = [m for m in headline if len(_metric_cols(df, m, methods)) >= 2]
    if not keep:
        return

    n = len(keep)
    fig = plt.figure(figsize=(12, 2.4 * n))
    heavy_pos = {"kl", "stein"}

    for i, metric in enumerate(keep, start=1):
        ax = fig.add_subplot(n, 1, i)
        cols = _metric_cols(df, metric, methods)

        for method, c in cols.items():
            s = df[c].astype(float)
            r = s.rolling(window, min_periods=max(10, window // 3)).median()
            ax.plot(r.index, r.values, label=method, alpha=0.9 if method in {"mix", "model"} else 0.6)

        ax.set_title(f"{metric}: {window}d rolling median")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=min(5, len(cols)))

        if metric in heavy_pos:
            all_vals = np.concatenate([df[c].astype(float).values for c in cols.values()])
            _maybe_log_axis(ax, all_vals)

    _savefig(os.path.join(outdir, f"rolling_median_{window}d.png"))

def plot_skill_vs_reference(
    df: pd.DataFrame,
    outdir: str,
    methods: List[str],
    ref: str,
    window: int = 63,
) -> None:
    """
    Skill + win-rate of a reference method vs all other methods.

    Skill definition:
      - higher-is-better metrics: diff (ref - other), good > 0
      - losses/risks: ratio (ref / other), good < 1

    Win definition:
      - higher-is-better metrics: ref > other
      - losses/risks: ref < other
    """
    if ref not in methods:
        print(f"[warn] ref method {ref!r} not in methods={methods}; skipping.")
        return

    others = [m for m in methods if m != ref]
    if not others:
        return

    # choose metrics that exist for ref and at least one other
    metrics = [
        "fro", "kl", "stein", "logeuc",
        "corr_spearman", "corr_offdiag_fro",
        "eig_log_mse", "cond_ratio",
        "gmvp_var", "gmvp_vol", "gmvp_sharpe", "gmvp_cumret",
        "turnover_l1", "w_hhi", "w_max_abs", "w_l1",
    ]

    keep_metrics = []
    for metric in metrics:
        ref_col = f"{ref}_{metric}"
        if ref_col not in df.columns:
            continue
        if any(f"{m}_{metric}" in df.columns for m in others):
            keep_metrics.append(metric)

    if not keep_metrics:
        print(f"[warn] No comparable metrics found for ref={ref}.")
        return

    # ----------------------------
    # Skill time series
    # ----------------------------
    n = len(keep_metrics)
    fig = plt.figure(figsize=(12, 2.4 * n))

    for i, metric in enumerate(keep_metrics, start=1):
        ax = fig.add_subplot(n, 1, i)

        ref_s = df[f"{ref}_{metric}"].astype(float)

        for m in others:
            c = f"{m}_{metric}"
            if c not in df.columns:
                continue
            skill = _skill_series(ref_s, df[c].astype(float), metric)
            ax.plot(skill.index, skill.values, label=f"vs {m}", alpha=0.85)

        ax.grid(True, alpha=0.3)

        if _is_higher_better(metric):
            ax.axhline(0.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill (diff): {ref} - other for {metric} (good > 0)")
        else:
            ax.axhline(1.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill (ratio): {ref} / other for {metric} (good < 1)")

            # log scale helps if positive
            ys = []
            for line in ax.lines:
                y = np.asarray(line.get_ydata(), dtype=float)
                if y.size:
                    ys.append(y)
            if ys:
                _maybe_log_axis(ax, np.concatenate(ys))

        ax.legend(loc="best", ncol=3)

    _savefig(os.path.join(outdir, f"skill_timeseries_ref_{ref}.png"))

    # ----------------------------
    # Rolling win-rate
    # ----------------------------
    fig = plt.figure(figsize=(12, 2.4 * n))
    for i, metric in enumerate(keep_metrics, start=1):
        ax = fig.add_subplot(n, 1, i)

        ref_s = df[f"{ref}_{metric}"].astype(float)

        for m in others:
            c = f"{m}_{metric}"
            if c not in df.columns:
                continue
            win = _win_series(ref_s, df[c].astype(float), metric)
            wr = win.rolling(window, min_periods=max(10, window // 3)).mean()
            ax.plot(wr.index, wr.values, label=f"vs {m}")

        ax.axhline(0.5, linestyle="--", alpha=0.7)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{window}d rolling win-rate: {ref} vs others for {metric}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=3)

    _savefig(os.path.join(outdir, f"rolling_winrate_ref_{ref}_{window}d.png"))

def plot_calibration(df: pd.DataFrame, outdir: str, methods: List[str]) -> None:
    """
    Optional calibration if you log per-date predicted vs realized GMVP variance.
    Looks for columns: {method}_pred_var and {method}_real_var
    """
    found_any = False
    for m in methods:
        p = f"{m}_pred_var"
        r = f"{m}_real_var"
        if p in df.columns and r in df.columns:
            found_any = True
            sub = df[[p, r]].astype(float).dropna()
            if len(sub) < 10:
                continue

            pred = sub[p].values
            real = sub[r].values

            # timeseries
            plt.figure(figsize=(12, 4))
            plt.plot(sub.index, pred, label=f"{m} pred_var")
            plt.plot(sub.index, real, label=f"{m} real_var")
            plt.title(f"{m}: pred_var vs real_var")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            _savefig(os.path.join(outdir, f"calib_{m}_pred_vs_real_timeseries.png"))

            # scatter
            plt.figure(figsize=(6.5, 6))
            plt.scatter(pred, real, s=12, alpha=0.6)
            lo = float(np.nanmin([pred.min(), real.min()]))
            hi = float(np.nanmax([pred.max(), real.max()]))
            plt.plot([lo, hi], [lo, hi], linestyle="--", alpha=0.7)
            plt.xlabel("pred_var")
            plt.ylabel("real_var")
            plt.title(f"{m}: calibration scatter (n={len(sub)})")
            plt.grid(True, alpha=0.3)
            if lo > 0:
                plt.xscale("log")
                plt.yscale("log")
            _savefig(os.path.join(outdir, f"calib_{m}_scatter.png"))

    if not found_any:
        print("[info] No {method}_pred_var/{method}_real_var columns found; skipping calibration plots.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/regime_similarity_backtest.csv")
    ap.add_argument("--outdir", type=str, default="results/figs_regime_similarity")
    ap.add_argument("--roll", type=int, default=21)
    ap.add_argument("--winroll", type=int, default=63)
    args = ap.parse_args()

    df = _read_results(args.csv)
    _ensure_dir(args.outdir)

    methods = _detect_methods(df)

    print("Loaded:", args.csv)
    print("Date range:", df.index.min(), "->", df.index.max())
    print("Detected methods:", methods)
    print("Num rows:", len(df))
    print("Num cols:", len(df.columns))

    plot_equity_curves(df, args.outdir, methods)
    plot_method_overlays(df, args.outdir, methods)
    plot_rolling_median(df, args.outdir, methods, window=args.roll)
    plot_skill_vs_reference(df, args.outdir, methods, ref="model", window=args.winroll)
    plot_skill_vs_reference(df, args.outdir, methods, ref="mix", window=args.winroll)
    plot_calibration(df, args.outdir, methods)

    print(f"\nSaved figures to: {args.outdir}")
    print("Key files:")
    print("  equity_curves_gmvp.png")
    print("  method_overlays.png")
    print(f"  rolling_median_{args.roll}d.png")
    print("  skill_timeseries_ref_model.png")
    print(f"  rolling_winrate_ref_model_{args.winroll}d.png")
    print("  skill_timeseries_ref_mix.png")
    print(f"  rolling_winrate_ref_mix_{args.winroll}d.png")


if __name__ == "__main__":
    main()