"""
Bar charts for joint grid ablations: compare all grid runs on a few scalar metrics.

Reads ablation_summary.csv (mode == "grid"), optionally merges is_pareto from
pareto/grid_with_pareto_flags.csv, and writes one PNG per metric — default three:
  - Fro loss (model_mean when primary is fro)
  - model GMVP Sharpe (mean)
  - model GMVP variance (mean)

Runs are ordered identically on the x-axis in every figure (sorted by run_tag) so
you can align bars across plots. Each figure highlights the best bar (gold outline)
for that metric and prints the full winning run_tag and axis values in a text box.

Usage:
  python -m scripts.analysis.ablation.grid_ablation_metric_figs \\
      --summary results/regime_covariance/ablation_phase2_joint/ablation_summary.csv

  # 16 configs (l1 vs lp_p1 deduped) + compact labels for the paper:
  python -m scripts.analysis.ablation.grid_ablation_metric_figs \\
      --summary .../ablation_summary.csv --report
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (csv column, title, output filename stem, best is "min" or "max")
DEFAULT_METRICS: list[tuple[str, str, str, str]] = [
    ("model_mean", "Fro loss (model mean, ↓ better)", "grid_bar_fro", "min"),
    ("model_gmvp_sharpe_mean", "GMVP Sharpe (model mean, ↑ better)", "grid_bar_gmvp_sharpe", "max"),
    ("model_gmvp_var_mean", "GMVP variance (model mean, ↓ better)", "grid_bar_gmvp_var", "min"),
]

CONFIG_KEYS = ("regime_clustering", "knn_metric", "pca_k", "tau", "k_neighbors")


def _knn_equivalence_key(k: object) -> str:
    """l1 and lp_p1 (Lp with p=1) are the same distance; collapse for deduping."""
    s = str(k).strip().lower()
    if s in ("l1", "lp_p1"):
        return "l1"
    return s


def dedupe_l1_lp1(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate grid rows that only differ by l1 vs lp_p1 knn_metric (same effective metric).
    Keeps the row with knn_metric == l1 when both exist.
    """
    g = grid.copy()
    g["_knn_eq"] = g["knn_metric"].map(_knn_equivalence_key)
    g = g.sort_values(
        ["regime_clustering", "_knn_eq", "pca_k", "tau", "k_neighbors", "knn_metric"],
        ascending=[True, True, True, True, True, True],
    )
    # l1 sorts before lp_p1 → keep_first retains l1
    g = g.drop_duplicates(
        subset=["regime_clustering", "_knn_eq", "pca_k", "tau", "k_neighbors"],
        keep="first",
    )
    return g.drop(columns=["_knn_eq"]).reset_index(drop=True)


def recompute_pareto_flags(grid: pd.DataFrame) -> pd.DataFrame:
    """is_pareto from model Sharpe / var / turnover (same rule as pareto_gmvp_report)."""
    from scripts.analysis.ablation.pareto_gmvp_report import _to_matrix_for_pareto, pareto_efficient_mask

    g = grid.copy()
    Y, valid = _to_matrix_for_pareto(g)
    mask = np.zeros(len(g), dtype=bool)
    if valid.any():
        Yv = Y[valid]
        m = pareto_efficient_mask(Yv)
        mask[np.where(valid)[0]] = m
    g["is_pareto"] = mask
    return g


def _cluster_abbrev(rc: object) -> str:
    s = str(rc).lower()
    if "fuzzy" in s:
        return "FCM"
    if "agglomerative" in s or "ward" in s:
        return "Ward"
    return str(rc)[:8]


def compact_axis_label(row: pd.Series) -> str:
    """Short x tick: cluster | PCA D | τ | k (knn omitted after l1/lp dedupe)."""
    return (
        f"{_cluster_abbrev(row.get('regime_clustering'))} | "
        f"D{int(row['pca_k'])} | "
        f"τ={gfmt(row['tau'])} | "
        f"k={int(row['k_neighbors'])}"
    )


def gfmt(x: object) -> str:
    try:
        v = float(x)
        return str(int(v)) if v == int(v) else f"{v:g}"
    except (TypeError, ValueError):
        return str(x)


# Short titles for paper-style figures (y-axis / title only)
REPORT_TITLES: dict[str, str] = {
    "model_mean": "Fro loss",
    "model_gmvp_sharpe_mean": "GMVP Sharpe",
    "model_gmvp_var_mean": "GMVP variance",
}


def _best_row_index(y: np.ndarray, direction: str) -> int:
    if y.size == 0:
        return 0
    yi = y.astype(float)
    mask = np.isfinite(yi)
    if not np.any(mask):
        return 0
    yi = np.where(mask, yi, np.nan)
    return int(np.nanargmin(yi) if direction == "min" else np.nanargmax(yi))


def _format_best_annotation(row: pd.Series, col: str, value: float, direction: str) -> str:
    verb = "minimize" if direction == "min" else "maximize"
    lines = [
        f"Best on this plot ({verb} {col}): {value:.10g}",
        "",
        str(row.get("run_tag", "")),
    ]
    for k in CONFIG_KEYS:
        if k in row.index and pd.notna(row[k]):
            lines.append(f"  {k} = {row[k]}")
    return "\n".join(lines)


def _load_grid_df(summary_csv: str, pareto_flags_csv: str | None) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    grid = df[df.get("mode") == "grid"].copy()
    if grid.empty:
        raise ValueError("No rows with mode=='grid' in summary CSV.")
    if pareto_flags_csv and os.path.isfile(pareto_flags_csv):
        flags = pd.read_csv(pareto_flags_csv)
        if "run_tag" in flags.columns and "is_pareto" in flags.columns:
            m = flags[["run_tag", "is_pareto"]].drop_duplicates("run_tag")
            grid = grid.merge(m, on="run_tag", how="left")
    return grid


def _short_label(run_tag: str, max_len: int = 36) -> str:
    s = str(run_tag)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _safe_filename_stem(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:80]


def plot_metric_bars(
    grid: pd.DataFrame,
    col: str,
    title: str,
    out_path: str,
    *,
    direction: str,
    report_style: bool = False,
) -> None:
    if direction not in {"min", "max"}:
        raise ValueError(f'direction must be "min" or "max", got {direction!r}')

    sort_cols = (
        ["regime_clustering", "pca_k", "tau", "k_neighbors", "knn_metric"]
        if report_style
        else ["run_tag"]
    )
    sub = grid.sort_values(sort_cols).reset_index(drop=True)
    y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
    n = len(sub)
    x = np.arange(n)
    if report_style and all(c in sub.columns for c in CONFIG_KEYS):
        labels = [compact_axis_label(sub.iloc[i]) for i in range(n)]
    else:
        labels = [_short_label(t) for t in sub["run_tag"].astype(str)]

    best_i = _best_row_index(y, direction)
    best_row = sub.iloc[best_i]
    best_val = float(y[best_i])

    colors = np.full(n, "#4682b4", dtype=object)
    if "is_pareto" in sub.columns:
        m = sub["is_pareto"].fillna(False).astype(bool).to_numpy()
        colors = np.where(m, "#c0392b", "#4682b4")

    edgecolors = ["black"] * n
    edgewidths = [0.35] * n
    edgecolors[best_i] = "#f1c40f"
    edgewidths[best_i] = 3.0

    fig_w = max(10.0, 0.52 * n) if report_style else max(12.0, 0.42 * n)
    fig_h = 5.8 if report_style else 7.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.bar(
        x,
        y,
        color=colors.tolist(),
        edgecolor=edgecolors,
        linewidth=edgewidths,
        alpha=0.92,
    )
    rot = 35 if report_style else 72
    fs = 8 if report_style else 6
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=fs)

    title_show = REPORT_TITLES.get(col, title.split("(")[0].strip()) if report_style else title.split("(")[0].strip()
    if report_style:
        ax.set_title(title_show, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Mean (horizon average)", fontsize=11)
    else:
        ax.set_ylabel(title.split("(")[0].strip(), fontsize=10)

    ax.grid(alpha=0.3, axis="y")

    note = _format_best_annotation(best_row, col, best_val, direction)
    if report_style:
        note_lines = note.split("\n")
        note_short = "\n".join(
            [
                note_lines[0],
                "",
                f"{_cluster_abbrev(best_row.get('regime_clustering'))} | D{int(best_row['pca_k'])} | "
                f"τ={gfmt(best_row['tau'])} | k={int(best_row['k_neighbors'])}",
                f"knn: {best_row.get('knn_metric', '')}",
            ]
        )
        fig.text(
            0.01,
            0.02,
            note_short,
            transform=fig.transFigure,
            fontsize=8,
            verticalalignment="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffef5", edgecolor="#cccccc", alpha=0.95),
        )
        fig.subplots_adjust(bottom=0.28, top=0.92)
    else:
        fig.suptitle(
            title + "\n(gold outline = best bar on this plot)",
            fontsize=11,
            y=0.995,
        )
        fig.text(
            0.02,
            0.86,
            note,
            transform=fig.transFigure,
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffef5", edgecolor="#cccccc", alpha=0.95),
        )
        fig.subplots_adjust(top=0.58, bottom=0.22)

    if "is_pareto" in sub.columns and sub["is_pareto"].fillna(False).any():
        from matplotlib.patches import Patch

        lab_par = "Pareto frontier" if report_style else "Pareto (Sharpe / var / turnover)"
        lab_dom = "Dominated" if report_style else "Dominated"
        ax.legend(
            handles=[
                Patch(facecolor="#c0392b", edgecolor="black", label=lab_par),
                Patch(facecolor="#4682b4", edgecolor="black", label=lab_dom),
                Patch(facecolor="none", edgecolor="#f1c40f", linewidth=2.5, label="Best on plot"),
            ],
            loc="upper right",
            fontsize=8,
        )
    else:
        from matplotlib.patches import Patch

        ax.legend(
            handles=[Patch(facecolor="none", edgecolor="#f1c40f", linewidth=2.5, label="Best on plot")],
            loc="upper right",
            fontsize=8,
        )

    fig.savefig(out_path, dpi=260 if report_style else 220, bbox_inches="tight")
    plt.close(fig)


def run(
    summary_csv: str,
    out_dir: str,
    *,
    pareto_flags_csv: str | None,
    metric_specs: list[tuple[str, str, str, str]] | None,
    dedupe_l1_lp: bool = False,
    report_style: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if dedupe_l1_lp:
        df = pd.read_csv(summary_csv)
        grid = df[df.get("mode") == "grid"].copy()
        if grid.empty:
            raise ValueError("No rows with mode=='grid' in summary CSV.")
        n0 = len(grid)
        grid = dedupe_l1_lp1(grid)
        grid = recompute_pareto_flags(grid)
        grid.to_csv(os.path.join(out_dir, "deduped_grid_summary.csv"), index=False)
        print(f"Deduped l1≡lp(p=1): {n0} → {len(grid)} configurations (saved deduped_grid_summary.csv).")
    else:
        grid = _load_grid_df(summary_csv, pareto_flags_csv)
    specs = metric_specs or DEFAULT_METRICS
    written = []
    for spec in specs:
        col, descr, stem, direction = spec
        if col not in grid.columns:
            print(f"(skip) missing column: {col}", file=sys.stderr)
            continue
        suf = "_report" if report_style else ""
        path = os.path.join(out_dir, f"{stem}{suf}.png")
        plot_metric_bars(grid, col, descr, path, direction=direction, report_style=report_style)
        written.append(path)
    if not written:
        raise ValueError("No figures written; check metric column names vs CSV.")
    print(f"Wrote {len(written)} figure(s) under {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid ablation bar charts from ablation_summary.csv")
    ap.add_argument("--summary", required=True, help="ablation_summary.csv path")
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <summary_dir>/figs_grid_metrics)",
    )
    ap.add_argument(
        "--pareto-flags",
        default=None,
        help="grid_with_pareto_flags.csv (default: <summary_dir>/pareto/grid_with_pareto_flags.csv if exists)",
    )
    ap.add_argument(
        "--metrics",
        default=None,
        help='Optional: "col|title|stem|min" or "col|title|stem|max" per metric, semicolon-separated. '
        "Example: model_mean|Fro|fro|min;model_gmvp_sharpe_mean|Sharpe|sharpe|max",
    )
    ap.add_argument(
        "--dedupe-l1-lp",
        action="store_true",
        help="Collapse l1 vs lp_p1 (Lp with p=1) to one row each; recompute Pareto on the reduced grid.",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Paper-style figures: short titles, compact x-labels, legend at top; implies --dedupe-l1-lp.",
    )
    args = ap.parse_args()

    summary_csv = os.path.abspath(args.summary)
    sdir = os.path.dirname(summary_csv)
    dedupe = bool(args.dedupe_l1_lp or args.report)
    report_style = bool(args.report)
    if args.outdir:
        out_dir = args.outdir
    elif report_style:
        out_dir = os.path.join(sdir, "figs_grid_metrics_report")
    else:
        out_dir = os.path.join(sdir, "figs_grid_metrics")
    pareto = args.pareto_flags
    if pareto is None:
        cand = os.path.join(sdir, "pareto", "grid_with_pareto_flags.csv")
        pareto = cand if os.path.isfile(cand) else None

    metric_specs: list[tuple[str, str, str, str]] | None = None
    if args.metrics:
        metric_specs = []
        for part in args.metrics.split(";"):
            part = part.strip()
            if not part:
                continue
            bits = [b.strip() for b in part.split("|")]
            if len(bits) == 3:
                col, title, stem = bits
                direction = "max"
            elif len(bits) == 4:
                col, title, stem, direction = bits
                if direction not in {"min", "max"}:
                    raise ValueError(f'4th field must be min or max, got: "{direction}"')
            else:
                raise ValueError(f'Expected col|title|stem or col|title|stem|min|max, got: "{part}"')
            metric_specs.append((col, title, _safe_filename_stem(stem), direction))

    run(
        summary_csv,
        out_dir,
        pareto_flags_csv=pareto,
        metric_specs=metric_specs,
        dedupe_l1_lp=dedupe,
        report_style=report_style,
    )


if __name__ == "__main__":
    main()
