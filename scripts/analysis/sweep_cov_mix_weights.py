"""
Grid search over mixing.cov_mix_weights (model / shrink / pers) for covariance backtests.

For each feasible triple that sums to 1, runs run_backtest_from_config (in memory, no per-run
file spam), records mean mix GMVP Sharpe and mean mix GMVP variance, and writes:
  - sweep_mix_weights.csv  (all runs)
  - sweep_mix_pareto.csv   (nondominated: higher Sharpe, lower variance)

Usage (repo root):
  # 3-way grid (model/shrink/pers); ~45 points at step 0.1 — long runtime
  python -m scripts.analysis.sweep_cov_mix_weights --config configs/regime_covariance.yaml --step 0.1

  # Fewer points: coarser step or cap (first N triples only)
  python -m scripts.analysis.sweep_cov_mix_weights --step 0.2 --max-runs 12

  # Faster 1D sweep: legacy mix_lambda only (model–shrink blend)
  python -m scripts.analysis.sweep_cov_mix_weights --two-way --step 0.1

  python -m scripts.analysis.sweep_cov_mix_weights --step 0.1 --plot
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.config_utils import load_yaml

from run_backtest import run_backtest_from_config


def _simplex_grid(step: float, min_w: float) -> list[tuple[float, float, float]]:
    """Grid on 2-simplex: w_model, w_shrink, w_pers >= min_w, sum = 1."""
    out: list[tuple[float, float, float]] = []
    w_m = float(min_w)
    while w_m <= 1.0 - 2.0 * min_w + 1e-12:
        w_s = float(min_w)
        while w_s <= 1.0 - w_m - min_w + 1e-12:
            w_p = 1.0 - w_m - w_s
            if w_p >= min_w - 1e-12:
                out.append((round(w_m, 6), round(w_s, 6), round(w_p, 6)))
            w_s += step
        w_m += step
    seen: set[tuple[float, float, float]] = set()
    dedup: list[tuple[float, float, float]] = []
    for t in out:
        key = (round(t[0], 4), round(t[1], 4), round(t[2], 4))
        if key not in seen:
            seen.add(key)
            dedup.append(t)
    return dedup


def _pareto_idx(sharpe: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Maximize sharpe, minimize var. Rows are Pareto-efficient if no other point dominates."""
    s = np.asarray(sharpe, dtype=float)
    v = np.asarray(var, dtype=float)
    Y = np.column_stack([s, -v])
    n = len(s)
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(Y[i]).all():
            ok[i] = False
            continue
        for j in range(n):
            if i == j or not ok[j]:
                continue
            if not np.isfinite(Y[j]).all():
                continue
            if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                ok[i] = False
                break
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep cov_mix_weights for GMVP Sharpe vs variance (mix method).")
    ap.add_argument("--config", default="configs/regime_covariance.yaml", help="Base YAML (covariance target).")
    ap.add_argument("--step", type=float, default=0.1, help="Grid step on simplex (smaller = more runs).")
    ap.add_argument("--min-weight", type=float, default=0.05, help="Minimum weight per component.")
    ap.add_argument(
        "--two-way",
        action="store_true",
        help="Sweep only mixing.mix_lambda (S_mix = (1-λ)S_shrink + λ S_model); ignores 3-way grid.",
    )
    ap.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Cap number of grid points (3-way only; takes first N after generation — for quick tests).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: results/<tag>/sweep_mix_weights from base config).",
    )
    ap.add_argument("--plot", action="store_true", help="Write sweep_mix_sharpe_vs_var.png in outdir.")
    args = ap.parse_args()

    base = load_yaml(os.path.join(_REPO, args.config))
    tag = str(base.get("outputs", {}).get("tag", "regime_covariance"))
    outdir = Path(args.outdir or (_REPO / "results" / tag / "sweep_mix_weights"))
    outdir.mkdir(parents=True, exist_ok=True)

    two_way = bool(args.two_way)
    if two_way:
        lam_step = float(args.step)
        triples = [(round(x, 6),) for x in np.arange(0.0, 1.0 + lam_step * 0.5, lam_step)]
    else:
        triples = _simplex_grid(step=float(args.step), min_w=float(args.min_weight))
        if args.max_runs is not None:
            triples = triples[: int(args.max_runs)]

    if not triples:
        print("No grid points; relax --min-weight or increase --step.", file=sys.stderr)
        raise SystemExit(1)

    # Quieter backtest
    if "backtest" in base and isinstance(base["backtest"], dict):
        base["backtest"]["verbose"] = False

    rows = []
    mode = "two-way mix_lambda" if two_way else "3-way cov_mix_weights"
    print(f"Grid ({mode}): {len(triples)} points. Full backtest each — may take a while.\n")

    for k, item in enumerate(triples):
        cfg = copy.deepcopy(base)
        cfg.setdefault("mixing", {})
        if two_way:
            lam = float(item[0])
            cfg["mixing"]["cov_mix_weights"] = None
            cfg["mixing"]["mix_lambda"] = lam
        else:
            w_m, w_s, w_p = item
            cfg["mixing"]["cov_mix_weights"] = {"model": w_m, "shrink": w_s, "pers": w_p}
        try:
            df, target = run_backtest_from_config(cfg, verbose=False)
        except Exception as e:
            lab = f"mix_lambda={item[0]}" if two_way else f"w={item}"
            print(f"(skip) {lab}: {e}", file=sys.stderr)
            continue
        if target != "covariance" or "mix_gmvp_sharpe" not in df.columns:
            print("This sweep is for covariance backtests with mix_* columns.", file=sys.stderr)
            raise SystemExit(2)
        ms = float(pd.to_numeric(df["mix_gmvp_sharpe"], errors="coerce").mean())
        mv = float(pd.to_numeric(df["mix_gmvp_var"], errors="coerce").mean())
        if two_way:
            lam = float(item[0])
            rows.append(
                {
                    "mix_lambda": lam,
                    "w_model": np.nan,
                    "w_shrink": np.nan,
                    "w_pers": np.nan,
                    "mix_gmvp_sharpe_mean": ms,
                    "mix_gmvp_var_mean": mv,
                }
            )
            print(f"  [{k+1}/{len(triples)}] mix_lambda={lam:.4f}  Sharpe={ms:.6f}  var={mv:.8f}")
        else:
            w_m, w_s, w_p = item
            rows.append(
                {
                    "mix_lambda": np.nan,
                    "w_model": w_m,
                    "w_shrink": w_s,
                    "w_pers": w_p,
                    "mix_gmvp_sharpe_mean": ms,
                    "mix_gmvp_var_mean": mv,
                }
            )
            print(f"  [{k+1}/{len(triples)}] model={w_m} shrink={w_s} pers={w_p}  Sharpe={ms:.6f}  var={mv:.8f}")

    res = pd.DataFrame(rows)
    if res.empty:
        print("No successful runs.", file=sys.stderr)
        raise SystemExit(1)

    res = res.sort_values(["mix_gmvp_sharpe_mean", "mix_gmvp_var_mean"], ascending=[False, True])
    csv_all = outdir / "sweep_mix_weights.csv"
    res.to_csv(csv_all, index=False)
    print(f"\nWrote {csv_all}")

    m = _pareto_idx(res["mix_gmvp_sharpe_mean"].values, res["mix_gmvp_var_mean"].values)
    par = res.loc[m].copy()
    par = par.sort_values(["mix_gmvp_sharpe_mean", "mix_gmvp_var_mean"], ascending=[False, True])
    csv_p = outdir / "sweep_mix_pareto.csv"
    par.to_csv(csv_p, index=False)
    print(f"Wrote {csv_p} ({len(par)} Pareto points)")

    # Suggest: knee — highest Sharpe on Pareto with var at or below median of Pareto
    if len(par) >= 1:
        med_var = float(par["mix_gmvp_var_mean"].median())
        below = par[par["mix_gmvp_var_mean"] <= med_var]
        pick = below.iloc[0] if len(below) else par.iloc[0]
        sug = outdir / "suggested_mix_weights.yaml"
        if two_way and pd.notna(pick.get("mix_lambda")):
            lines = [
                "# Picked from Pareto (2-way sweep): among var <= median(Pareto var), best Sharpe.",
                "# Under `mixing:` set mix_lambda and do NOT set cov_mix_weights (2-way blend).",
                f"mix_lambda: {float(pick['mix_lambda'])}",
                "",
                f"# mix_gmvp_sharpe_mean: {float(pick['mix_gmvp_sharpe_mean'])}",
                f"# mix_gmvp_var_mean: {float(pick['mix_gmvp_var_mean'])}",
            ]
        else:
            lines = [
                "# Picked from Pareto: among points with var <= median(Pareto var), take best Sharpe.",
                "# Paste under mixing: in your regime_covariance.yaml (adjust as you like).",
                "cov_mix_weights:",
                f"  model: {float(pick['w_model'])}",
                f"  shrink: {float(pick['w_shrink'])}",
                f"  pers: {float(pick['w_pers'])}",
                "",
                f"# mix_gmvp_sharpe_mean: {float(pick['mix_gmvp_sharpe_mean'])}",
                f"# mix_gmvp_var_mean: {float(pick['mix_gmvp_var_mean'])}",
            ]
        sug.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {sug}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5.5))
            ax.scatter(res["mix_gmvp_var_mean"], res["mix_gmvp_sharpe_mean"], c="steelblue", s=40, alpha=0.7, label="all")
            ax.scatter(par["mix_gmvp_var_mean"], par["mix_gmvp_sharpe_mean"], facecolors="none", edgecolors="crimson", s=120, linewidths=2, label="Pareto")
            ax.set_xlabel("mix GMVP var (mean, lower better)")
            ax.set_ylabel("mix GMVP Sharpe (mean, higher better)")
            ax.set_title("cov_mix_weights sweep: Sharpe vs variance (mix)")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            pfig = outdir / "sweep_mix_sharpe_vs_var.png"
            fig.savefig(pfig, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Wrote {pfig}")
        except Exception as e:
            print(f"(warn) --plot failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
