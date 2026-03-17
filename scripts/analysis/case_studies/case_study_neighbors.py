#!/usr/bin/env python3
"""
Neighbor-based case studies for the regime-aware similarity forecaster.

For a chosen evaluation date (one of the backtest dates), this script:
  - Rebuilds the model up to that date using the same config as the main backtest.
  - Recomputes the forecast at that anchor.
  - Extracts the nearest neighbors used by the similarity step, including:
        * neighbor dates and raw anchor indices
        * distances in embedding space and kernel weights kappa
        * regime probabilities for each neighbor
        * regime-aware neighbor weights W (per-regime)
  - Saves a tidy CSV with neighbor diagnostics.
  - Optionally produces simple visualizations to support paper-style case studies.

Usage (from repo root):
  python -m scripts.analysis.case_study_neighbors \\
      --config configs/regime_covariance.yaml \\
      --date 2018-02-05 \\
      --k_neighbors 10

You can then:
  - Inspect the CSV table to describe which past periods are being used.
  - Cross-reference neighbor dates with regime timelines and crisis windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.config_utils import load_yaml, deep_update, parse_overrides
from scripts.clean_data import clean_returns_matrix_at_load
from run_backtest import build_model  # reuse core construction logic
from scripts.analysis.utils.paths import RESULTS_DIR, resolve_backtest_path


def _load_backtest_row(backtest_path: Path, date_str: str) -> pd.Series:
    """Load a single backtest row for the requested date."""
    if not backtest_path.exists():
        raise FileNotFoundError(
            f"Backtest not found at {backtest_path}. "
            "Run: python run_backtest.py --config configs/regime_covariance.yaml or configs/regime_volatility.yaml"
        )

    if backtest_path.suffix == ".parquet":
        df = pd.read_parquet(backtest_path)
    else:
        df = pd.read_csv(backtest_path)

    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    target_date = pd.to_datetime(date_str)
    row = df.loc[df["date"] == target_date]
    if row.empty:
        available = df["date"].dt.strftime("%Y-%m-%d").unique()
        raise ValueError(
            f"Date {date_str} not found in backtest results.\n"
            f"Available example dates: {', '.join(list(available[:5]))} ..."
        )
    return row.iloc[0]


def _build_model_from_cfg(cfg: dict) -> tuple:
    """Construct model and supporting objects using the same logic as run_backtest."""
    mcfg = cfg["model"]
    ecfg = cfg["embedder"]
    acfg = cfg["aggregator"]

    model = build_model(
        target_type=str(mcfg.get("target", "covariance")),
        lookback=int(mcfg["lookback"]),
        horizon=int(mcfg["horizon"]),
        ddof=int(mcfg["ddof"]),
        n_regimes=int(mcfg["n_regimes"]),
        tau=float(mcfg["tau"]),
        random_state=int(mcfg["random_state"]),
        transition_estimator=str(mcfg["transition_estimator"]),
        trans_smooth=float(mcfg["trans_smooth"]),
        sample_stride=int(mcfg["sample_stride"]),
        embedder_name=str(ecfg["name"]),
        pca_k=int(ecfg["pca_k"]),
        k_eigs=int(ecfg["k_eigs"]),
        gmm_init_params=str(mcfg["gmm_init_params"]),
        gmm_n_init=int(mcfg["gmm_n_init"]),
        max_window_na_pct=float(ecfg["max_window_na_pct"]),
        min_stocks_with_data_pct=float(ecfg["min_stocks_with_data_pct"]),
        verbose_skip=bool(ecfg["verbose_skip"]),
        aggregator_name=str(acfg["name"]),
        eps_spd=float(acfg["eps_spd"]),
        knn_metric=str(mcfg.get("knn_metric", "l2")),
        regime_aggregation=str(mcfg.get("regime_aggregation", "soft")),
    )
    return model, mcfg


def _case_study_neighbors(cfg_path: str, date_str: str, k_neighbors: Optional[int] = None) -> Path:
    """Main case-study logic; returns path to saved neighbor CSV."""
    cfg = load_yaml(cfg_path)
    # No overrides from CLI for now; keep parity with main backtest by default.

    dcfg = cfg["data"]
    bcfg = cfg["backtest"]

    # Load returns matrix exactly as in run_backtest
    returns_df = clean_returns_matrix_at_load(
        parquet_path=dcfg["parquet_path"],
        policy=dcfg["policy"],
        q99_thresh=float(dcfg["q99_thresh"]),
        max_thresh=float(dcfg["max_thresh"]),
        min_non_nan_frac=float(dcfg["min_non_nan_frac"]),
    ).T
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df = returns_df.sort_index()

    # Backtest path from config tag (works for both covariance and volatility)
    tag = str(cfg.get("outputs", {}).get("tag", "regime_covariance"))
    backtest_path = resolve_backtest_path(tag)
    row = _load_backtest_row(backtest_path, date_str)
    raw_anchor = int(row["raw_anchor"])
    anchor_date = pd.to_datetime(row["date"])

    lookback = int(cfg["model"]["lookback"])
    horizon = int(cfg["model"]["horizon"])

    # Build and fit model on data up to and including the anchor date.
    model, mcfg = _build_model_from_cfg(cfg)
    train_df = returns_df.loc[:anchor_date]
    model.fit(train_df)

    # Align with raw_anchor in full array
    R = returns_df.to_numpy(dtype=float)
    dates = returns_df.index

    if dates[raw_anchor] != anchor_date:
        # Fallback: recompute anchor from date in case of indexing differences
        try:
            raw_anchor = int(np.where(dates == anchor_date)[0][0])
        except IndexError:
            raise RuntimeError("Could not align raw_anchor with returns index for case study.")

    past = R[raw_anchor - lookback + 1 : raw_anchor + 1, :]

    k_case = int(k_neighbors) if k_neighbors is not None else int(bcfg["k_neighbors"])

    use_filter = str(cfg.get("model", {}).get("regime_weighting", "filtered")).lower() == "filtered"
    pred = model.predict_at_raw_anchor(
        past=past,
        raw_anchor=raw_anchor,
        k_neighbors=k_case,
        use_filter=use_filter,
        neighbor_gap=int(bcfg["neighbor_gap"]),
        return_regime=True,
        return_neighbors=True,
    )
    # Unpack: cov returns (Sigma_hat, alpha, pi, neighbors_info); vol returns (vol_hat, alpha, pi, neighbors_info)
    _, alpha_t, pi_t, neighbors_info = pred[0], pred[1], pred[2], pred[3]

    nbr_raw = neighbors_info["raw_anchors"]
    nbr_dates = neighbors_info["dates"]
    if nbr_dates is None:
        nbr_dates = dates[nbr_raw]
    dist = neighbors_info["dist"]
    kappa = neighbors_info["kappa"]
    PI_neighbors = neighbors_info["PI_neighbors"]  # (M, K)
    W = neighbors_info["W"]  # (K, M)

    K = PI_neighbors.shape[1]
    M = PI_neighbors.shape[0]

    # Aggregate per-neighbor total weight under the current filtered alpha
    alpha_vec = np.asarray(alpha_t, dtype=float).reshape(-1)
    total_weight = (alpha_vec @ W).reshape(-1)  # (M,)

    df_neighbors = pd.DataFrame(
        {
            "anchor_date": anchor_date,
            "anchor_raw": raw_anchor,
            "neighbor_idx": np.arange(M),
            "neighbor_raw_anchor": nbr_raw,
            "neighbor_date": pd.to_datetime(nbr_dates),
            "lag_days": (anchor_date - pd.to_datetime(nbr_dates)).days,
            "dist_embedding": dist,
            "kappa": kappa,
            "total_weight": total_weight,
        }
    )

    for k in range(K):
        df_neighbors[f"pi_neighbor_regime_{k}"] = PI_neighbors[:, k]
        df_neighbors[f"W_regime_{k}"] = W[k, :]

    out_dir = RESULTS_DIR / tag / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_date = anchor_date.strftime("%Y%m%d")
    out_csv = out_dir / f"neighbors_{safe_date}.csv"
    df_neighbors.to_csv(out_csv, index=False)

    # Panel A: weights vs date (x-axis = neighbor_date)
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        sc = ax.scatter(
            pd.to_datetime(df_neighbors["neighbor_date"]),
            df_neighbors["total_weight"],
            c=df_neighbors["dist_embedding"],
            cmap="viridis",
            alpha=0.8,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Embedding distance")
        ax.set_xlabel("Neighbor date")
        ax.set_ylabel("Total neighbor weight")
        ax.set_title(f"Neighbor weights over historical dates (anchor {anchor_date.date()})")
        ax.grid(alpha=0.3, axis="y")
        fig.autofmt_xdate()
        fig_path = out_dir / f"neighbors_{safe_date}_weights_vs_date.png"
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # Panel B: overlay neighbors on regime timeline (fresh mini-timeline for this case)
    try:
        if backtest_path.suffix == ".parquet":
            bt_df = pd.read_parquet(backtest_path)
        else:
            bt_df = pd.read_csv(backtest_path)
        if "date" not in bt_df.columns:
            bt_df = bt_df.reset_index()
        bt_df["date"] = pd.to_datetime(bt_df["date"])
        bt_df = bt_df.sort_values("date").reset_index(drop=True)

        if "regime_assigned" in bt_df.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            # base timeline
            regimes = bt_df["regime_assigned"].astype(int)
            ax2.scatter(bt_df["date"], regimes, s=10, alpha=0.4, c="lightgray", label="All anchors")

            # overlay neighbors sized by weight
            nbr_dates_dt = pd.to_datetime(df_neighbors["neighbor_date"])
            weights = df_neighbors["total_weight"].values
            # match neighbor regimes from backtest if possible
            nbr_regimes = []
            bt_date_to_regime = dict(zip(bt_df["date"], regimes))
            for d in nbr_dates_dt:
                nbr_regimes.append(bt_date_to_regime.get(d, np.nan))
            nbr_regimes = np.asarray(nbr_regimes)

            ax2.scatter(
                nbr_dates_dt,
                nbr_regimes,
                s=50 + 300 * (weights / (weights.max() + 1e-12)),
                c="crimson",
                alpha=0.9,
                label="Neighbors (size ∝ weight)",
                edgecolors="black",
                linewidths=0.5,
            )

            ax2.axvline(anchor_date, color="black", linestyle="--", linewidth=1.0, label="Anchor date")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Regime")
            ax2.set_yticks(sorted(regimes.unique()))
            ax2.set_title(f"Neighbor dates on regime timeline (anchor {anchor_date.date()})")
            ax2.grid(alpha=0.3, axis="x")
            ax2.legend(loc="upper left", fontsize=8)
            fig2.autofmt_xdate()

            fig2_path = out_dir / f"neighbors_{safe_date}_on_regime_timeline.png"
            fig2.tight_layout()
            fig2.savefig(fig2_path, dpi=200)
            plt.close(fig2)
    except Exception:
        pass

    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Neighbor case-study tool for regime-aware similarity backtest.")
    ap.add_argument("--config", type=str, default="configs/regime_covariance.yaml", help="Path to main YAML config.")
    ap.add_argument("--date", type=str, required=True, help="Case-study evaluation date (YYYY-MM-DD) matching backtest.")
    ap.add_argument("--k_neighbors", type=int, default=None, help="Optional number of neighbors for the case study.")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Optional config overrides (section.key=value); use with care to stay close to main backtest.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    overrides = parse_overrides(args.set)
    cfg = deep_update(cfg, overrides)

    tag = str(cfg.get("outputs", {}).get("tag", "regime_covariance"))
    case_studies_dir = RESULTS_DIR / tag / "case_studies"
    case_studies_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = case_studies_dir / "tmp_case_cfg.yaml"
    import yaml

    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    out_csv = _case_study_neighbors(str(tmp_cfg_path), args.date, args.k_neighbors)

    print("\n" + "=" * 80)
    print("NEIGHBOR CASE STUDY COMPLETE")
    print("=" * 80 + "\n")
    print(f"Anchor date: {args.date}")
    print(f"Config used: {tmp_cfg_path}")
    print(f"Neighbor diagnostics saved to: {out_csv}")
    print("\nSuggested next steps:")
    print("  - Open the CSV and sort by total_weight to see the most influential neighbors.")
    print("  - Cross-check neighbor_date against regime timelines and crisis periods.")
    print("  - For the paper, pick 1–2 anchors (e.g., COVID crash week) and describe which")
    print("    historical episodes the model considers similar, and how that aligns with intuition.")
    print()


if __name__ == "__main__":
    main()

