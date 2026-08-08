#!/usr/bin/env python3
"""Build paper tables/figures used in the ICAIF redesign manuscript.

Reads frozen OOS panels under results/redesign/ and writes:
  - results/redesign/paper_assets/tables/{oos_bootstrap,data_quality,rankcorr}.tex
  - results/redesign/paper_assets/figs/{frob_vs_stein,collapse_membership}.png
  - copies into paper/tables and paper/figs when those dirs exist (paper/ is gitignored)

Calendar-preserving moving-block bootstrap: blocks are drawn on the original
evaluation-anchor timeline; extreme-return filters are applied as masks inside
resampled blocks.

Usage:
  .venv/bin/python scripts/redesign/build_paper_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "results" / "redesign" / "robustness" / "oos_robustness_panel.csv"
RETURNS = ROOT / "data" / "processed" / "returns_universe_100.parquet"
COLLAPSE_K3 = ROOT / "results" / "redesign" / "collapse_decomp" / "collapse_decomp_k3.csv"
COLLAPSE_K4 = ROOT / "results" / "redesign" / "collapse_decomp" / "collapse_decomp_compact.csv"
OUT = ROOT / "results" / "redesign" / "paper_assets"
OUT_TAB = OUT / "tables"
OUT_FIG = OUT / "figs"

METHOD_LABEL = {
    "D0_pca8": "SIM",
    "ewma": "EWMA",
    "persistence": "Pers.",
    "ledoit_wolf": "LW",
    "oas": "OAS",
    "rolling": "Roll",
    "shrink": "Shrink",
}
SHORT = {
    "persistence": "Pers.",
    "rolling": "Roll.",
    "shrink": "Shrink",
    "ewma": "EWMA",
    "ledoit_wolf": "LW",
    "oas": "OAS",
    "D0_pca8": "SIM",
}


def _off_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    return panel[panel["eta"] == 0.01].copy()


def extreme_masks(dates: pd.DatetimeIndex, *, L: int = 50, H: int = 20, thr: float = 0.5):
    """Lookback = L days strictly before anchor; future = next H days after anchor."""
    rets = pd.read_parquet(RETURNS)
    rets.index = pd.to_datetime(rets.index)
    fut_clean, either_clean = [], []
    for a in dates:
        loc = rets.index.get_loc(a)
        past = rets.iloc[loc - L : loc].to_numpy()
        fut = rets.iloc[loc + 1 : loc + 1 + H].to_numpy()
        past_ext = bool(np.nanmax(np.abs(past)) > thr) if past.size else False
        fut_ext = bool(np.nanmax(np.abs(fut)) > thr) if fut.size else False
        fut_clean.append(not fut_ext)
        either_clean.append(not (past_ext or fut_ext))
    return np.asarray(fut_clean), np.asarray(either_clean)


def series(off: pd.DataFrame, dates, method: str, col: str) -> np.ndarray:
    s = off[off["method"] == method].copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date").loc[dates, col].to_numpy(dtype=float)


def mbb_calendar(a, b, mask, *, block=20, n_boot=5000, seed=0):
    """Paired MBB on original timeline; mean only over mask-True observations."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    est = float(d[mask].mean())
    boots = []
    max_start = n - block
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        vals = d[idx][mask[idx]]
        if len(vals):
            boots.append(float(vals.mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return est, float(lo), float(hi)


def sci(x: float) -> str:
    if abs(x) >= 1e3 or (abs(x) > 0 and abs(x) < 1e-2):
        ax = abs(x)
        exp = int(np.floor(np.log10(ax))) if ax > 0 else 0
        mant = x / (10**exp)
        return f"${mant:.2f}{{\\times}}10^{{{exp}}}$"
    return f"{x:.3f}"


def fmt_ci(lo: float, hi: float) -> str:
    # scale CI digits to match paper style
    scale = max(abs(lo), abs(hi), 1e-30)
    if scale >= 1e3:
        exp = int(np.floor(np.log10(scale)))
        return f"$[{lo/10**exp:.2f},\\,{hi/10**exp:.2f}]{{\\times}}10^{{{exp}}}$"
    if scale < 1e-2:
        exp = int(np.floor(np.log10(scale)))
        return f"$[{lo/10**exp:.1f},\\,{hi/10**exp:.1f}]{{\\times}}10^{{{exp}}}$"
    return f"$[{lo:.3f},\\,{hi:.3f}]$"


def write_oos_bootstrap(off: pd.DataFrame, dates, fut_clean, either_clean) -> Path:
    rows = []
    specs = [
        ("Full", np.ones(len(dates), dtype=bool), "SIM $-$ EWMA, $R^{\\mathrm{cond}}$", "D0_pca8", "ewma", "R_cond"),
        ("Full", np.ones(len(dates), dtype=bool), "SIM $-$ Persistence, $R^{\\mathrm{cond}}$", "D0_pca8", "persistence", "R_cond"),
        ("Full", np.ones(len(dates), dtype=bool), "SIM $-$ Ledoit--Wolf, Stein", "D0_pca8", "ledoit_wolf", "cond_stein"),
        ("Future $|r|\\le0.5$", fut_clean, "SIM $-$ EWMA, $R^{\\mathrm{cond}}$", "D0_pca8", "ewma", "R_cond"),
        ("Future $|r|\\le0.5$", fut_clean, "SIM $-$ Ledoit--Wolf, Stein", "D0_pca8", "ledoit_wolf", "cond_stein"),
        ("Lookback+future $|r|\\le0.5$", either_clean, "SIM $-$ Persistence, $R^{\\mathrm{cond}}$", "D0_pca8", "persistence", "R_cond"),
    ]
    csv_rows = []
    for panel, mask, contrast, m1, m2, col in specs:
        est, lo, hi = mbb_calendar(series(off, dates, m1, col), series(off, dates, m2, col), mask)
        covers = lo <= 0 <= hi
        rows.append((panel, contrast, est, lo, hi, covers))
        csv_rows.append(
            dict(panel=panel, contrast=contrast.replace("$", ""), est=est, lo=lo, hi=hi, covers0=covers, n=int(mask.sum()))
        )
    pd.DataFrame(csv_rows).to_csv(OUT / "oos_headline_bootstrap_calendar.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired moving-block bootstrap for headline frozen OOS contrasts (5000 resamples). Blocks of length 20 are taken on the \emph{original} evaluation-anchor timeline; extreme-return filters are applied as a mask inside resampled blocks (so block length retains calendar spacing). Positive estimates mean SIM is worse. On the full panel, mean $R^{\mathrm{cond}}$ intervals cover zero because means are heavy-tail dominated; future-clean windows support both decision and Stein contrasts.}",
        r"\label{tab:oosboot}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Panel & Contrast & Estimate & 95\% CI & Covers 0? \\",
        r"\midrule",
    ]
    for panel, contrast, est, lo, hi, covers in rows:
        # paper-style rounding for full-panel Rcond
        if "Stein" in contrast or abs(est) < 1e-3 or abs(est) >= 1e3:
            est_s = sci(est)
        else:
            est_s = f"{est:.3f}"
        lines.append(
            f"{panel} & {contrast} & {est_s} & {fmt_ci(lo, hi)} & {'Yes' if covers else 'No'} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ]
    path = OUT_TAB / "oos_bootstrap.tex"
    path.write_text("\n".join(lines))
    return path


def write_data_quality(off: pd.DataFrame, dates, fut_clean, either_clean) -> Path:
    def summarize(mask, name):
        sub = off[off["date"].isin(dates[mask])]
        g = sub.groupby("method").agg(
            frob_med=("raw_frob", "median"),
            frob_mean=("raw_frob", "mean"),
            stein=("cond_stein", "mean"),
            R=("R_cond", "mean"),
        )
        sim = g.loc["D0_pca8"]
        others = g.drop(index="D0_pca8")
        best_med_m = others["frob_med"].idxmin()
        max_mean_m = others["frob_mean"].idxmax()
        stein_rank = int((g["stein"] <= sim["stein"]).sum())
        r_rank = int((g["R"] <= sim["R"]).sum())
        n_methods = len(g)
        return dict(
            filter=name,
            n=int(mask.sum()),
            sim_med=float(sim["frob_med"]),
            best_med=float(others.loc[best_med_m, "frob_med"]),
            best_med_lab=SHORT[best_med_m],
            sim_mean=float(sim["frob_mean"]),
            max_mean=float(others.loc[max_mean_m, "frob_mean"]),
            max_mean_lab=SHORT[max_mean_m],
            stein_rank=stein_rank,
            r_rank=r_rank,
            n_methods=n_methods,
        )

    rows = [
        summarize(np.ones(len(dates), dtype=bool), "Full panel"),
        summarize(fut_clean, r"Future $|r|\le 0.5$"),
        summarize(either_clean, r"Lookback+future $|r|\le 0.5$"),
    ]
    pd.DataFrame(rows).to_csv(OUT / "data_quality_filter_summary.csv", index=False)

    def f3(x):
        return f"{x:.3f}" if x >= 0.01 else f"{x:.3f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-hoc extreme-return filters on the frozen OOS panel ($\eta=0.01$; models unchanged). Under future cleaning, SIM's mean Frobenius collapses while several classical means remain large; lookback+future cleaning normalizes means and removes SIM's median-Frobenius lead, but SIM stays last on Stein and $R^{\mathrm{cond}}$.}",
        r"\label{tab:dataqual}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Filter & $n$ & SIM med. & Best other med. & SIM mean & Max other mean & SIM ranks \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['filter']} & {r['n']} & {r['sim_med']:.3f} & {r['best_med']:.3f} ({r['best_med_lab']}) & "
            f"{r['sim_mean']:.3g} & {r['max_mean']:.3g} ({r['max_mean_lab']}) & "
            f"Stein/$R$: {r['stein_rank']}/{r['n_methods']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    path = OUT_TAB / "data_quality.tex"
    path.write_text("\n".join(lines))
    return path


def write_rankcorr(off: pd.DataFrame) -> Path:
    g = off.groupby("method").agg(
        frob_med=("raw_frob", "median"),
        stein=("cond_stein", "mean"),
        nll=("nll", "mean"),
        R_cond=("R_cond", "mean"),
    )
    corr = g.corr(method="spearman")
    corr.to_csv(OUT / "metric_rank_correlations_spearman.csv")
    labels = ["Frob.\\ med.", "Stein", "NLL", r"$R^{\mathrm{cond}}$"]
    keys = ["frob_med", "stein", "nll", "R_cond"]

    def cell(i, j):
        if j < i:
            return ""
        v = corr.loc[keys[i], keys[j]]
        if abs(v - 1.0) < 1e-12:
            return "1.00"
        if abs(v) < 5e-3:
            return "0.00"
        if v < 0:
            return f"$-{abs(v):.2f}$"
        return f"{v:.2f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Spearman rank correlations across seven methods (held-out, $\eta=0.01$), descriptive only ($n{=}7$). Stein and NLL are nearly interchangeable; neither matches classical decision rankings. (KL omitted: $\mathrm{Stein}=2\,\mathrm{KL}$.)}",
        r"\label{tab:rankcorr}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for i, lab in enumerate(labels):
        vals = [cell(i, j) for j in range(4)]
        # upper triangle style like paper
        row_vals = []
        for j, v in enumerate(vals):
            if j < i:
                row_vals.append("")
            else:
                row_vals.append(v)
        lines.append(lab + " & " + " & ".join(row_vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    path = OUT_TAB / "rankcorr.tex"
    path.write_text("\n".join(lines))
    return path


def fig_frob_vs_stein(off: pd.DataFrame) -> Path:
    agg = (
        off.groupby("method")
        .agg(frob_med=("raw_frob", "median"), stein=("cond_stein", "mean"), R_cond=("R_cond", "mean"))
        .reset_index()
    )
    agg["label"] = agg["method"].map(METHOD_LABEL)
    r = agg["R_cond"].to_numpy()
    sizes = 80 + 2200 * (r - r.min()) / (r.max() - r.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(agg.frob_med, agg.stein, s=sizes, c="#1f4e79", alpha=0.75, edgecolors="white", linewidths=0.6, zorder=3)
    for _, row in agg.iterrows():
        ax.annotate(row.label, (row.frob_med, row.stein), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel(r"Median raw Frobenius $\|\hat\Sigma-\Sigma^{real}\|_F$")
    ax.set_ylabel("Mean conditioned Stein (log scale)")
    ax.set_title("Matrix fit vs inverse-sensitive loss")
    for val, lab in [(r.min(), r"low $R^{cond}$"), (r.max(), r"high $R^{cond}$")]:
        ax.scatter(
            [],
            [],
            s=80 + 2200 * (val - r.min()) / (r.max() - r.min() + 1e-12),
            c="#1f4e79",
            alpha=0.75,
            edgecolors="white",
            label=lab,
        )
    ax.legend(frameon=False, fontsize=7, loc="upper right", title="marker size", title_fontsize=7)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    path = OUT_FIG / "frob_vs_stein.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def fig_collapse() -> Path:
    if COLLAPSE_K3.exists():
        df = pd.read_csv(COLLAPSE_K3)
        # columns: tag,dim,train,query,cent
        get = lambda tag: df[df.tag == tag].iloc[0]
        order = [
            ("pca_svd_legacy_48", "Unscaled\nPCA+SVD", "train", "query"),
            ("pca_only_48", "PCA-only\n(D=48)", "train", "query"),
            ("pca_svd_standardized_48", "PCA+SVD\nstandardized", "train", "query"),
            ("pca_only_15", "PCA-only\n(D=15)", "train", "query"),
            ("pca_only_10", "PCA-only\n(D=10)", "train", "query"),
            ("pca_only_5", "PCA-only\n(D=5)", "train", "query"),
            ("market_state_6", "Market-state\n(D=6)", "train", "query"),
        ]
        labs, tr, qu = [], [], []
        for tag, lab, tc, qc in order:
            r = get(tag)
            labs.append(lab)
            tr.append(float(r[tc]))
            qu.append(float(r[qc]))
        kline, klab = 1 / 3, r"$1/K$ (K=3)"
    else:
        df = pd.read_csv(COLLAPSE_K4)
        order = [
            ("pca_svd_legacy_48", "Unscaled\nPCA+SVD"),
            ("pca_only_48", "PCA-only\n(D=48)"),
            ("pca_svd_standardized_48", "PCA+SVD\nstandardized"),
            ("pca_only_15", "PCA-only\n(D=15)"),
            ("pca_only_10", "PCA-only\n(D=10)"),
            ("pca_only_5", "PCA-only\n(D=5)"),
            ("market_state_6", "Market-state\n(D=6)"),
        ]
        labs, tr, qu = [], [], []
        for tag, lab in order:
            r = df[df.tag == tag].iloc[0]
            labs.append(lab)
            tr.append(float(r.fcm_train_mean_max_p))
            qu.append(float(r.fcm_query_mean_max_p))
        kline, klab = 0.25, r"$1/K$ (K=4)"

    x = np.arange(len(labs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - w / 2, tr, width=w, label="Train", color="#1f4e79")
    ax.bar(x + w / 2, qu, width=w, label="Query", color="#7aa2c4")
    ax.axhline(kline, color="gray", ls="--", lw=0.8, label=klab)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Mean max membership")
    ax.set_title("FCM membership peaking by representation")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT_FIG / "collapse_membership.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def copy_into_paper():
    paper_tab = ROOT / "paper" / "tables"
    paper_fig = ROOT / "paper" / "figs"
    if paper_tab.is_dir():
        for p in OUT_TAB.glob("*.tex"):
            (paper_tab / p.name).write_text(p.read_text())
            print(f"copied {p.name} -> paper/tables/")
    if paper_fig.is_dir():
        for p in OUT_FIG.glob("*.png"):
            (paper_fig / p.name).write_bytes(p.read_bytes())
            print(f"copied {p.name} -> paper/figs/")


def main():
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    if not PANEL.exists():
        raise SystemExit(f"Missing {PANEL}; run scripts/redesign/run_oos_robustness.py first.")
    if not RETURNS.exists():
        raise SystemExit(f"Missing {RETURNS}")

    off = _off_panel()
    dates = pd.to_datetime(sorted(off[off.method == "D0_pca8"].date.unique()))
    fut_clean, either_clean = extreme_masks(dates)
    print(f"anchors={len(dates)} fut_clean={fut_clean.sum()} either_clean={either_clean.sum()}")

    p1 = write_oos_bootstrap(off, dates, fut_clean, either_clean)
    p2 = write_data_quality(off, dates, fut_clean, either_clean)
    p3 = write_rankcorr(off)
    f1 = fig_frob_vs_stein(off)
    f2 = fig_collapse()
    copy_into_paper()

    readme = OUT / "README.md"
    readme.write_text(
        "# Paper assets (redesign manuscript)\n\n"
        "Generated by `scripts/redesign/build_paper_assets.py` from frozen OOS panels.\n\n"
        "## Inputs\n"
        "- `results/redesign/robustness/oos_robustness_panel.csv`\n"
        "- `data/processed/returns_universe_100.parquet` (filter masks)\n"
        "- `results/redesign/collapse_decomp/collapse_decomp_k3.csv` (Fig 1; else K=4 compact)\n\n"
        "## Outputs\n"
        "- `tables/oos_bootstrap.tex`, `data_quality.tex`, `rankcorr.tex`\n"
        "- `figs/frob_vs_stein.png`, `collapse_membership.png`\n"
        "- CSV audits alongside this README\n"
    )
    print("wrote", p1, p2, p3, f1, f2, readme)


if __name__ == "__main__":
    main()
