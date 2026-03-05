# scripts/viz_backtest_results.py
# YAML-driven visualization for results/*_backtest.csv (full range, no slicing)

'''
Usage:
  python scripts/viz_backtest_results.py --config configs/viz_regime_similarity.yaml

Override output folder + rolling window: 

  python scripts/viz_backtest_results.py --config configs/viz_regime_similarity.yaml \
    --set outputs.outdir="results/figs_regime_similarity_v2" \
    --set plot.roll_window=63 \
    --set plot.winroll_window=126
  
Force explicit methods (no auto-detect):
  python scripts/viz_backtest_results.py --config configs/viz_regime_similarity.yaml \
    --set plot.methods='["model","mix","roll","pers","shrink"]'

Turn on calibration plots:
  python scripts/viz_backtest_results.py --config configs/viz_regime_similarity.yaml \
    --set plot.calibration=true
'''


from __future__ import annotations

import os
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.config_utils import load_yaml, deep_update, parse_overrides


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
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _detect_methods(df: pd.DataFrame) -> List[str]:
    banned = {
        "guardrail", "raw", "apply", "floor", "mixlambda", "mix_lambda",
        "date", "stat"
    }

    counts: Dict[str, int] = {}
    for c in df.columns:
        if "_" not in c:
            continue
        m = c.split("_", 1)[0]
        if m in banned:
            continue
        counts[m] = counts.get(m, 0) + 1

    # heuristic: must appear at least 5 times
    methods = [m for m, cnt in counts.items() if cnt >= 5]

    # prefer canonical ordering
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


def _maybe_log_axis(ax: plt.Axes, values: np.ndarray) -> None:
    v = values[np.isfinite(values)]
    if v.size and np.nanmin(v) > 0:
        ax.set_yscale("log")


def _is_higher_better(metric: str, higher_is_better: set[str]) -> bool:
    return metric in higher_is_better


def _skill_series(ref: pd.Series, other: pd.Series, metric: str, higher_is_better: set[str]) -> pd.Series:
    ref = ref.astype(float)
    other = other.astype(float)
    ok = ref.notna() & other.notna()

    s = pd.Series(np.nan, index=ref.index)
    if _is_higher_better(metric, higher_is_better):
        s.loc[ok] = ref.loc[ok] - other.loc[ok]
    else:
        s.loc[ok] = ref.loc[ok] / np.maximum(other.loc[ok], 1e-12)
    return s


def _win_series(ref: pd.Series, other: pd.Series, metric: str, higher_is_better: set[str]) -> pd.Series:
    ref = ref.astype(float)
    other = other.astype(float)
    ok = ref.notna() & other.notna()

    w = pd.Series(np.nan, index=ref.index)
    if _is_higher_better(metric, higher_is_better):
        w.loc[ok] = (ref.loc[ok] > other.loc[ok]).astype(float)
    else:
        w.loc[ok] = (ref.loc[ok] < other.loc[ok]).astype(float)
    return w


def _fname(outdir: str, prefix: str, name: str) -> str:
    if prefix:
        return os.path.join(outdir, f"{prefix}{name}")
    return os.path.join(outdir, name)


# ----------------------------
# Plots
# ----------------------------
def plot_equity_curves(df: pd.DataFrame, outdir: str, prefix: str, methods: List[str]) -> None:
    cols = _metric_cols(df, "gmvp_cumret", methods)
    if not cols:
        print("[warn] No *_gmvp_cumret columns found; skipping equity curves.")
        return

    plt.figure(figsize=(12, 5))
    for m, c in cols.items():
        r = df[c].astype(float).fillna(0.0).values
        eq = np.cumprod(1.0 + r)
        plt.plot(df.index, eq, label=m, alpha=0.9 if m in {"mix", "model"} else 0.6)

    plt.title("GMVP Equity Curves (chained from per-date gmvp_cumret)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=min(5, len(cols)))
    _savefig(_fname(outdir, prefix, "equity_curves_gmvp.png"))


def plot_method_overlays(
    df: pd.DataFrame,
    outdir: str,
    prefix: str,
    methods: List[str],
    overlay_metrics: List[str],
    heavy_pos_metrics: set[str],
) -> None:
    keep: List[Tuple[str, str]] = []
    for metric in overlay_metrics:
        cols = _metric_cols(df, metric, methods)
        if len(cols) >= 2:
            keep.append((metric, metric))

    if not keep:
        print("[warn] No overlay-able method metrics found.")
        return

    n = len(keep)
    fig = plt.figure(figsize=(12, 2.4 * n))

    for i, (metric, title) in enumerate(keep, start=1):
        ax = fig.add_subplot(n, 1, i)
        cols = _metric_cols(df, metric, methods)

        for m, c in cols.items():
            s = df[c].astype(float)
            ax.plot(s.index, s.values, label=m, alpha=0.9 if m in {"mix", "model"} else 0.6)

        ax.set_title(f"{title}: methods overlay")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=min(5, len(cols)))

        if metric in heavy_pos_metrics:
            all_vals = np.concatenate([df[c].astype(float).values for c in cols.values()])
            _maybe_log_axis(ax, all_vals)

    _savefig(_fname(outdir, prefix, "method_overlays.png"))


def plot_rolling_median(
    df: pd.DataFrame,
    outdir: str,
    prefix: str,
    methods: List[str],
    window: int,
) -> None:
    headline = ["fro", "kl", "stein", "gmvp_var", "gmvp_sharpe", "turnover_l1"]
    keep = [m for m in headline if len(_metric_cols(df, m, methods)) >= 2]
    if not keep:
        return

    n = len(keep)
    fig = plt.figure(figsize=(12, 2.4 * n))

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

    _savefig(_fname(outdir, prefix, f"rolling_median_{window}d.png"))


def plot_skill_vs_reference(
    df: pd.DataFrame,
    outdir: str,
    prefix: str,
    methods: List[str],
    ref: str,
    win_window: int,
    overlay_metrics: List[str],
    higher_is_better: set[str],
) -> None:
    if ref not in methods:
        print(f"[warn] ref method {ref!r} not in methods={methods}; skipping.")
        return

    others = [m for m in methods if m != ref]
    if not others:
        return

    keep_metrics: List[str] = []
    for metric in overlay_metrics:
        ref_col = f"{ref}_{metric}"
        if ref_col not in df.columns:
            continue
        if any(f"{m}_{metric}" in df.columns for m in others):
            keep_metrics.append(metric)

    if not keep_metrics:
        print(f"[warn] No comparable metrics found for ref={ref}.")
        return

    # Skill series
    n = len(keep_metrics)
    fig = plt.figure(figsize=(12, 2.4 * n))

    for i, metric in enumerate(keep_metrics, start=1):
        ax = fig.add_subplot(n, 1, i)
        ref_s = df[f"{ref}_{metric}"].astype(float)

        for m in others:
            c = f"{m}_{metric}"
            if c not in df.columns:
                continue
            skill = _skill_series(ref_s, df[c].astype(float), metric, higher_is_better)
            ax.plot(skill.index, skill.values, label=f"vs {m}", alpha=0.85)

        ax.grid(True, alpha=0.3)

        if _is_higher_better(metric, higher_is_better):
            ax.axhline(0.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill (diff): {ref} - other for {metric} (good > 0)")
        else:
            ax.axhline(1.0, linestyle="--", alpha=0.7)
            ax.set_title(f"Skill (ratio): {ref} / other for {metric} (good < 1)")

            ys = []
            for line in ax.lines:
                y = np.asarray(line.get_ydata(), dtype=float)
                if y.size:
                    ys.append(y)
            if ys:
                _maybe_log_axis(ax, np.concatenate(ys))

        ax.legend(loc="best", ncol=3)

    _savefig(_fname(outdir, prefix, f"skill_timeseries_ref_{ref}.png"))

    # Rolling win-rate
    fig = plt.figure(figsize=(12, 2.4 * n))
    for i, metric in enumerate(keep_metrics, start=1):
        ax = fig.add_subplot(n, 1, i)
        ref_s = df[f"{ref}_{metric}"].astype(float)

        for m in others:
            c = f"{m}_{metric}"
            if c not in df.columns:
                continue
            win = _win_series(ref_s, df[c].astype(float), metric, higher_is_better)
            wr = win.rolling(win_window, min_periods=max(10, win_window // 3)).mean()
            ax.plot(wr.index, wr.values, label=f"vs {m}")

        ax.axhline(0.5, linestyle="--", alpha=0.7)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{win_window}d rolling win-rate: {ref} vs others for {metric}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=3)

    _savefig(_fname(outdir, prefix, f"rolling_winrate_ref_{ref}_{win_window}d.png"))


def plot_calibration(df: pd.DataFrame, outdir: str, prefix: str, methods: List[str]) -> None:
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

            plt.figure(figsize=(12, 4))
            plt.plot(sub.index, pred, label=f"{m} pred_var")
            plt.plot(sub.index, real, label=f"{m} real_var")
            plt.title(f"{m}: pred_var vs real_var")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            _savefig(_fname(outdir, prefix, f"calib_{m}_pred_vs_real_timeseries.png"))

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
            _savefig(_fname(outdir, prefix, f"calib_{m}_scatter.png"))

    if not found_any:
        print("[info] No {method}_pred_var/{method}_real_var columns found; skipping calibration plots.")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/viz_regime_similarity.yaml")
    ap.add_argument("--set", action="append", default=[], help="Override: section.key=value (repeatable)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    overrides = parse_overrides(args.set)
    cfg = deep_update(cfg, overrides)

    csv_path = cfg["inputs"]["csv"]
    outdir = cfg["outputs"]["outdir"]
    prefix = str(cfg["outputs"].get("prefix", ""))

    roll_window = int(cfg["plot"]["roll_window"])
    winroll_window = int(cfg["plot"]["winroll_window"])

    methods_cfg = cfg["plot"].get("methods", None)
    overlay_metrics = list(cfg["plot"].get("overlay_metrics", []))
    heavy_pos_metrics = set(cfg["plot"].get("heavy_pos_metrics", []))
    higher_is_better = set(cfg["plot"].get("higher_is_better", []))
    reference_methods = list(cfg["plot"].get("reference_methods", []))
    do_calib = bool(cfg["plot"].get("calibration", True))

    df = _read_results(csv_path)
    _ensure_dir(outdir)

    methods = list(methods_cfg) if methods_cfg is not None else _detect_methods(df)

    print("Loaded:", csv_path)
    print("Date range:", df.index.min(), "->", df.index.max())
    print("Methods:", methods)
    print("Num rows:", len(df))
    print("Num cols:", len(df.columns))

    plot_equity_curves(df, outdir, prefix, methods)
    plot_method_overlays(df, outdir, prefix, methods, overlay_metrics, heavy_pos_metrics)
    plot_rolling_median(df, outdir, prefix, methods, window=roll_window)

    for ref in reference_methods:
        plot_skill_vs_reference(
            df=df,
            outdir=outdir,
            prefix=prefix,
            methods=methods,
            ref=ref,
            win_window=winroll_window,
            overlay_metrics=overlay_metrics,
            higher_is_better=higher_is_better,
        )

    if do_calib:
        plot_calibration(df, outdir, prefix, methods)

    print(f"\nSaved figures to: {outdir}")
    print("Key files:")
    print(f"  {prefix}equity_curves_gmvp.png")
    print(f"  {prefix}method_overlays.png")
    print(f"  {prefix}rolling_median_{roll_window}d.png")
    for ref in reference_methods:
        print(f"  {prefix}skill_timeseries_ref_{ref}.png")
        print(f"  {prefix}rolling_winrate_ref_{ref}_{winroll_window}d.png")


if __name__ == "__main__":
    main()