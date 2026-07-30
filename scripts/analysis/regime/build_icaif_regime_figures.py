"""
ICAIF Track B: regenerate regime timeline + transition heatmap with numbered labels.

Uses the locked paper numbering in
`results/regime_characterization/paper_regime_numbering.json` (Regimes 1–4 only;
no narrative names on the figures).

Source: results/oos_final/backtest.csv (honest harness).

Writes:
  results/regime_characterization/figs/regime_timeline.png
  results/regime_characterization/figs/transition_matrix_heatmap.png
  paper/figs/regime/regime_timeline.png          (Overleaf drop; gitignored)
  paper/figs/regime/transition_matrix_heatmap.png

Usage (from repo root):
  .venv/bin/python -m scripts.analysis.regime.build_icaif_regime_figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Headless / sandbox-safe backend before pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]

CRISES = [
    ("GFC", "2008-09-01", "2009-03-31"),
    ("COVID", "2020-02-01", "2020-06-30"),
]

# Stable, print-friendly colors keyed by paper regime 1..4
PAPER_COLORS = {
    1: "#2ca02c",
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#d62728",
}


def _load_numbering(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_backtest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    dom_col = "dominant_regime" if "dominant_regime" in df.columns else "regime_assigned"
    cluster = pd.to_numeric(df[dom_col], errors="coerce")
    out = df.copy()
    out["cluster_id"] = cluster
    return out.dropna(subset=["cluster_id"]).reset_index(drop=True)


def _paper_regime_series(cluster: pd.Series, cluster_to_paper: dict[int, int]) -> pd.Series:
    return cluster.map(lambda c: cluster_to_paper[int(c)])


def fig_timeline(df: pd.DataFrame, cluster_to_paper: dict[int, int], save_paths: list[Path]) -> None:
    d = df.copy()
    d["paper_regime"] = _paper_regime_series(d["cluster_id"], cluster_to_paper).astype(int)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    for paper in (1, 2, 3, 4):
        sub = d[d["paper_regime"] == paper]
        ax.scatter(
            sub["date"],
            sub["paper_regime"],
            s=14,
            color=PAPER_COLORS[paper],
            label=f"Regime {paper}",
            alpha=0.75,
            zorder=2,
        )

    ymax = 4.55
    for name, s, e in CRISES:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        ax.axvspan(s_ts, e_ts, color="black", alpha=0.08, zorder=0)
        ax.text(
            s_ts + (e_ts - s_ts) / 2,
            ymax,
            name,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
            color="dimgray",
        )

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels([f"Regime {k}" for k in (1, 2, 3, 4)], fontsize=10)
    ax.set_ylim(0.5, 4.7)
    ax.set_xlim(d["date"].min(), d["date"].max())
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Regime", fontsize=11)
    ax.set_title("Regime Assignments Over Time", fontsize=13, fontweight="bold", pad=18)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_transition(df: pd.DataFrame, cluster_to_paper: dict[int, int], save_paths: list[Path]) -> np.ndarray:
    paper = _paper_regime_series(df["cluster_id"], cluster_to_paper).astype(int).to_numpy()
    # Count transitions in paper-regime space; order rows/cols as Regime 1..4
    order = [1, 2, 3, 4]
    idx = {r: i for i, r in enumerate(order)}
    C = np.zeros((4, 4), dtype=float)
    for a, b in zip(paper[:-1], paper[1:]):
        C[idx[int(a)], idx[int(b)]] += 1.0
    row = C.sum(axis=1, keepdims=True)
    A = np.divide(C, row, out=np.zeros_like(C), where=row > 0)

    names = [f"Regime {k}" for k in order]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        A,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Transition probability"},
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_xlabel("To regime (t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("From regime (t−1)", fontsize=11, fontweight="bold")
    ax.set_title("Regime Transition Matrix", fontsize=13, fontweight="bold", pad=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    for i in range(4):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="#d62728", linewidth=2.0))
    fig.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return A


def main() -> None:
    ap = argparse.ArgumentParser(description="ICAIF numbered regime figures (Track B)")
    ap.add_argument(
        "--input",
        default=str(REPO_ROOT / "results" / "oos_final" / "backtest.csv"),
    )
    ap.add_argument(
        "--numbering",
        default=str(REPO_ROOT / "results" / "regime_characterization" / "paper_regime_numbering.json"),
    )
    args = ap.parse_args()

    numbering = _load_numbering(Path(args.numbering))
    cluster_to_paper = {int(k): int(v) for k, v in numbering["paper_regime_by_cluster_id"].items()}

    df = _load_backtest(Path(args.input))
    results_figs = REPO_ROOT / "results" / "regime_characterization" / "figs"
    paper_figs = REPO_ROOT / "paper" / "figs" / "regime"

    fig_timeline(
        df,
        cluster_to_paper,
        [results_figs / "regime_timeline.png", paper_figs / "regime_timeline.png"],
    )
    A = fig_transition(
        df,
        cluster_to_paper,
        [results_figs / "transition_matrix_heatmap.png", paper_figs / "transition_matrix_heatmap.png"],
    )

    # Persist transition matrix CSV in paper order for sanity checks
    names = [f"Regime_{k}" for k in (1, 2, 3, 4)]
    pd.DataFrame(A, index=[f"From_{n}" for n in names], columns=[f"To_{n}" for n in names]).to_csv(
        REPO_ROOT / "results" / "regime_characterization" / "transition_matrix_paper.csv"
    )

    print("Wrote numbered regime timeline + transition heatmap.")
    print("Self-transition (persistence) by paper regime:")
    for i, k in enumerate((1, 2, 3, 4)):
        print(f"  Regime {k}: {A[i, i]:.3f}")


if __name__ == "__main__":
    main()
