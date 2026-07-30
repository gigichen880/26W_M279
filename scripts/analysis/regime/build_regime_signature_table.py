"""
Build the ICAIF regime signature table (Track B / Bree).

Reads the locked honest harness (`results/oos_final`) and writes:
  - results/regime_characterization/paper_regime_numbering.json
  - results/regime_characterization/regime_signature.csv
  - paper/tables/regime_signature.tex

Paper regimes are numbered 1–4 (no narrative labels). Mapping is locked from
`results/oos_final/regime_name_map.json` via the Canonical name→number map in
ICAIF_SUBMISSION_TODO.md:
  Calm Bull → 1 · Moderate Bull → 2 · Normal → 3 · High Stress → 4

Usage (from repo root):
  .venv/bin/python -m scripts.analysis.regime.build_regime_signature_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

# Locked paper numbering (name → Regime 1..4). Do not reshuffle without agreeing
# with Devansh — every Track A/B table/figure must use this map.
PAPER_REGIME_BY_LEGACY_NAME: dict[str, int] = {
    "Calm Bull": 1,
    "Moderate Bull": 2,
    "Normal": 3,
    "High Stress": 4,
}


def _load_backtest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _transition_persistence(dom: np.ndarray, k: int) -> np.ndarray:
    c = np.zeros((k, k), dtype=float)
    for a, b in zip(dom[:-1], dom[1:]):
        c[int(a), int(b)] += 1.0
    row = c.sum(axis=1, keepdims=True)
    return np.divide(c, row, out=np.zeros_like(c), where=row > 0)


def build_signature(df: pd.DataFrame, legacy_name_map: dict[int, str]) -> pd.DataFrame:
    need = ("realized_vol", "avg_corr", "market_ret")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest missing columns {missing}")

    dom_col = "dominant_regime" if "dominant_regime" in df.columns else "regime_assigned"
    if dom_col not in df.columns:
        raise ValueError("Backtest missing dominant_regime / regime_assigned")

    dom = pd.to_numeric(df[dom_col], errors="coerce")
    k = len(legacy_name_map)
    n_total = int(dom.notna().sum())
    if n_total == 0:
        raise ValueError("No regime assignments in backtest")

    pers = _transition_persistence(dom.dropna().astype(int).to_numpy(), k)

    rows: list[dict] = []
    for cluster_id, legacy_name in sorted(legacy_name_map.items()):
        if legacy_name not in PAPER_REGIME_BY_LEGACY_NAME:
            raise ValueError(
                f"Legacy name {legacy_name!r} not in locked paper map "
                f"{sorted(PAPER_REGIME_BY_LEGACY_NAME)}"
            )
        sub = df.loc[dom == cluster_id]
        n_days = int(len(sub))
        rows.append(
            {
                "paper_regime": PAPER_REGIME_BY_LEGACY_NAME[legacy_name],
                "legacy_name": legacy_name,
                "cluster_id": int(cluster_id),
                "realized_vol": float(pd.to_numeric(sub["realized_vol"], errors="coerce").mean())
                if n_days
                else float("nan"),
                "avg_corr": float(pd.to_numeric(sub["avg_corr"], errors="coerce").mean())
                if n_days
                else float("nan"),
                "mean_return": float(pd.to_numeric(sub["market_ret"], errors="coerce").mean())
                if n_days
                else float("nan"),
                "persistence": float(pers[cluster_id, cluster_id]),
                "pct_days": float(n_days / n_total * 100.0),
                "n_days": n_days,
            }
        )
    return pd.DataFrame(rows).sort_values("paper_regime").reset_index(drop=True)


def write_numbering_json(path: Path, legacy_name_map: dict[int, str]) -> None:
    cluster_to_paper = {
        str(cid): PAPER_REGIME_BY_LEGACY_NAME[name] for cid, name in legacy_name_map.items()
    }
    paper_to_cluster = {str(v): int(k) for k, v in cluster_to_paper.items()}
    paper_to_legacy = {
        str(PAPER_REGIME_BY_LEGACY_NAME[name]): name for name in legacy_name_map.values()
    }
    payload = {
        "source": "results/oos_final/regime_name_map.json",
        "note": (
            "Paper uses Regimes 1–4 only. legacy_name is internal bookkeeping "
            "for sanity checks against Canonical Numbers; do not print in the paper."
        ),
        "legacy_name_by_cluster_id": {str(k): v for k, v in sorted(legacy_name_map.items())},
        "paper_regime_by_cluster_id": cluster_to_paper,
        "cluster_id_by_paper_regime": paper_to_cluster,
        "legacy_name_by_paper_regime": paper_to_legacy,
        "paper_regime_by_legacy_name": PAPER_REGIME_BY_LEGACY_NAME,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_latex(sig: pd.DataFrame, path: Path) -> None:
    """Standalone include: table environment ready for \\input{tables/regime_signature}."""
    lines = [
        "% Auto-generated by scripts/analysis/regime/build_regime_signature_table.py",
        "% Do not edit by hand — regenerate from results/oos_final.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Regime signatures (descriptive). Regimes are numbered 1--4 with no "
        "narrative labels. Realized vol, average pairwise correlation, and mean return "
        "are horizon cross-sectional diagnostics from the evaluation window; persistence "
        "is the self-transition probability of the hard regime assignment; \\% days is "
        "the share of anchors assigned to each regime.}",
        "\\label{tab:regime-signature}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Regime & Realized vol & Avg.\\ corr. & Mean return & Persistence & \\% days \\\\",
        "\\midrule",
    ]
    for _, r in sig.iterrows():
        lines.append(
            f"{int(r.paper_regime)} & "
            f"{r.realized_vol:.4f} & "
            f"{r.avg_corr:.3f} & "
            f"{r.mean_return:.4f} & "
            f"{r.persistence:.3f} & "
            f"{r.pct_days:.1f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build regime signature table (Track B)")
    ap.add_argument(
        "--input",
        default=str(REPO_ROOT / "results" / "oos_final" / "backtest.csv"),
        help="Honest-harness backtest CSV/parquet",
    )
    ap.add_argument(
        "--name-map",
        default=str(REPO_ROOT / "results" / "oos_final" / "regime_name_map.json"),
        help="Cluster-id → legacy name map (bookkeeping only)",
    )
    args = ap.parse_args()

    backtest_path = Path(args.input)
    name_map_path = Path(args.name_map)
    if not backtest_path.exists():
        raise SystemExit(f"Missing backtest: {backtest_path}")
    if not name_map_path.exists():
        raise SystemExit(f"Missing regime name map: {name_map_path}")

    legacy_name_map = {int(k): v for k, v in json.loads(name_map_path.read_text()).items()}
    df = _load_backtest(backtest_path)
    sig = build_signature(df, legacy_name_map)

    out_dir = REPO_ROOT / "results" / "regime_characterization"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_numbering_json(out_dir / "paper_regime_numbering.json", legacy_name_map)
    sig.to_csv(out_dir / "regime_signature.csv", index=False)
    write_latex(sig, REPO_ROOT / "paper" / "tables" / "regime_signature.tex")

    print("Locked paper numbering (legacy name → Regime):")
    for name, paper in sorted(PAPER_REGIME_BY_LEGACY_NAME.items(), key=lambda kv: kv[1]):
        cid = next(c for c, n in legacy_name_map.items() if n == name)
        print(f"  {name!r:16s} (cluster {cid}) → Regime {paper}")
    print("\nSignature table:")
    print(sig.to_string(index=False))
    print(f"\nWrote: {out_dir / 'paper_regime_numbering.json'}")
    print(f"Wrote: {out_dir / 'regime_signature.csv'}")
    print(f"Wrote: {REPO_ROOT / 'paper' / 'tables' / 'regime_signature.tex'}")

    # Soft sanity vs Canonical Numbers (Normal ~15–20% of days)
    normal_pct = float(sig.loc[sig["legacy_name"] == "Normal", "pct_days"].iloc[0])
    if not (15.0 <= normal_pct <= 25.0):
        print(
            f"WARNING: Normal pct_days={normal_pct:.1f} outside canonical ~15–20% band — "
            "flag to Devansh (possible stale results)."
        )
    else:
        print(f"Sanity: Normal pct_days={normal_pct:.1f} within ~15–20% canonical band.")


if __name__ == "__main__":
    main()
