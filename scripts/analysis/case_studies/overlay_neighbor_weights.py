"""
Overlay neighbor-weight scatter plots for two anchors (no reruns).

This is a "second-order attention" visualization: compare where two anchors put weight.

Two alignment modes:
  - lag: x-axis = lag_days (days before anchor). Best for comparing patterns independent of calendar shift.
  - shift_date: x-axis = neighbor_date shifted so the two anchors coincide (calendar-style overlay).

Usage:
  MPLBACKEND=Agg python -m scripts.analysis.case_studies.overlay_neighbor_weights \\
    --a results/regime_covariance/case_studies/neighbors_20140724.csv \\
    --b results/regime_covariance/case_studies/neighbors_20140814.csv \\
    --align lag

Outputs a PNG next to the inputs unless --out is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    df["neighbor_date"] = pd.to_datetime(df["neighbor_date"])
    for c in ["lag_days", "dist_embedding", "kappa", "total_weight"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _default_out(a: Path, b: Path, align: str) -> Path:
    outdir = a.parent
    a_tag = a.stem.replace("neighbors_", "")
    b_tag = b.stem.replace("neighbors_", "")
    return outdir / f"neighbors_overlay_{a_tag}_vs_{b_tag}_{align}.png"


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay neighbor weights for two anchors.")
    ap.add_argument("--a", required=True, help="Path to neighbors_*.csv for anchor A")
    ap.add_argument("--b", required=True, help="Path to neighbors_*.csv for anchor B")
    ap.add_argument("--align", default="lag", choices=("lag", "shift_date"), help="Alignment mode")
    ap.add_argument("--out", default=None, help="Output PNG path (optional)")
    ap.add_argument("--title", default=None, help="Custom plot title (optional)")
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    if not a_path.exists():
        raise FileNotFoundError(a_path)
    if not b_path.exists():
        raise FileNotFoundError(b_path)

    df_a = _load(a_path)
    df_b = _load(b_path)
    if df_a.empty or df_b.empty:
        raise RuntimeError("One of the inputs has no rows.")

    anchor_a = pd.to_datetime(df_a["anchor_date"].iloc[0])
    anchor_b = pd.to_datetime(df_b["anchor_date"].iloc[0])

    align = str(args.align)
    if align == "lag":
        # More intuitive if "more recent" is to the right:
        # lag_days is positive in CSV, so we plot -lag_days.
        x_a = -df_a["lag_days"].to_numpy(dtype=float)
        x_b = -df_b["lag_days"].to_numpy(dtype=float)
        xlab = "Days relative to anchor (negative = past)"
    else:
        # Shift neighbor dates so anchors coincide on the same calendar x-axis.
        shift_a = (anchor_b - anchor_a)
        x_a = (df_a["neighbor_date"] + shift_a).to_numpy()
        x_b = df_b["neighbor_date"].to_numpy()
        xlab = f"Neighbor date (A shifted by {shift_a.days}d to match B anchor)"

    w_a = df_a["total_weight"].to_numpy(dtype=float)
    w_b = df_b["total_weight"].to_numpy(dtype=float)
    d_a = df_a["dist_embedding"].to_numpy(dtype=float) if "dist_embedding" in df_a.columns else None
    d_b = df_b["dist_embedding"].to_numpy(dtype=float) if "dist_embedding" in df_b.columns else None

    # Marker size: emphasize weight but keep readable
    def msize(w: np.ndarray) -> np.ndarray:
        w = np.clip(w, 0.0, np.nanmax(w) if np.isfinite(np.nanmax(w)) else 1.0)
        return 40.0 + 900.0 * (w / max(float(np.nanmax(w)), 1e-12))

    s_a = msize(w_a)
    s_b = msize(w_b)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.6))

    # Color by embedding distance within each series (optional).
    # Use different colormaps + marker shapes so sets are clearly distinguishable.
    if d_a is not None and np.isfinite(d_a).any():
        sca = ax.scatter(x_a, w_a, s=s_a, c=d_a, cmap="Blues", marker="o", alpha=0.85, edgecolor="black", linewidth=0.6, label=f"A {anchor_a.date()}")
    else:
        sca = ax.scatter(x_a, w_a, s=s_a, color="#1f77b4", marker="o", alpha=0.85, edgecolor="black", linewidth=0.6, label=f"A {anchor_a.date()}")

    if d_b is not None and np.isfinite(d_b).any():
        scb = ax.scatter(x_b, w_b, s=s_b, c=d_b, cmap="Oranges", marker="^", alpha=0.85, edgecolor="black", linewidth=0.6, label=f"B {anchor_b.date()}")
    else:
        scb = ax.scatter(x_b, w_b, s=s_b, color="#ff7f0e", marker="^", alpha=0.85, edgecolor="black", linewidth=0.6, label=f"B {anchor_b.date()}")

    # Add a single colorbar only if both have distances; otherwise skip.
    if d_a is not None and d_b is not None and np.isfinite(d_a).any() and np.isfinite(d_b).any():
        # Two colormaps → two colorbars would be messy; show none by default.
        pass

    title = args.title
    if not title:
        title = f"Neighbor weights overlay ({anchor_a.date()} vs {anchor_b.date()}; align={align})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Total neighbor weight")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    # Make y-limits a bit padded
    ymax = float(np.nanmax(np.r_[w_a, w_b])) if np.isfinite(np.nanmax(np.r_[w_a, w_b])) else 1.0
    ax.set_ylim(-0.01 * ymax, 1.12 * ymax)

    out = Path(args.out) if args.out else _default_out(a_path, b_path, align)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

