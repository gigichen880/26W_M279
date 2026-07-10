"""
Data-driven regime semantic labels for backtests and downstream plots/tables.

Uses per-regime realized cross-sectional volatility, average correlation, and
equal-weight market return over each evaluation horizon, plus concentration
of dominant regimes in fixed crisis windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Crisis windows used for label assignment (anchor dates falling inside these ranges).
DEFAULT_LABEL_CRISIS_WINDOWS: list[tuple[str, str]] = [
    ("2008-09-01", "2009-03-31"),
    ("2015-08-01", "2016-02-28"),
    ("2020-02-01", "2020-06-30"),
]

LABEL_HIGH_STRESS = "High Stress"
LABEL_CALM_BULL = "Calm Bull"
LABEL_MODERATE_BULL = "Moderate Bull"
LABEL_NORMAL = "Normal"


def compute_horizon_cross_sectional_stats(fut: np.ndarray) -> dict[str, float]:
    """
    From future returns (H, N): mean cross-sectional realized vol, mean pairwise
    correlation, and equal-weight market cumulative return over the horizon.
    """
    fut = np.asarray(fut, dtype=float)
    if fut.ndim != 2 or fut.size == 0:
        return {"realized_vol": float("nan"), "avg_corr": float("nan"), "market_ret": float("nan")}

    H, _N = fut.shape
    if H < 2:
        # Still define vol as cross-sectional mean of per-column std with ddof=1
        col_std = np.nanstd(fut, axis=0, ddof=1)
        realized_vol = float(np.nanmean(col_std)) if np.any(np.isfinite(col_std)) else float("nan")
        avg_corr = float("nan")
        mkt = np.nanmean(fut, axis=1)
        mkt = mkt[np.isfinite(mkt)]
        market_ret = float(np.prod(1.0 + mkt) - 1.0) if mkt.size else float("nan")
        return {"realized_vol": realized_vol, "avg_corr": avg_corr, "market_ret": market_ret}

    col_std = np.nanstd(fut, axis=0, ddof=1)
    realized_vol = float(np.nanmean(col_std))

    df_f = pd.DataFrame(fut)
    c = df_f.corr(numeric_only=True).values
    n = c.shape[0]
    if n >= 2:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        vals = c[mask]
        vals = vals[np.isfinite(vals)]
        avg_corr = float(np.mean(vals)) if vals.size else float("nan")
    else:
        avg_corr = float("nan")

    mkt_daily = np.nanmean(fut, axis=1)
    mkt_daily = mkt_daily[np.isfinite(mkt_daily)]
    market_ret = float(np.prod(1.0 + mkt_daily) - 1.0) if mkt_daily.size else float("nan")

    return {"realized_vol": realized_vol, "avg_corr": avg_corr, "market_ret": market_ret}


def resolve_dominant_regime_series(df: pd.DataFrame) -> pd.Series:
    """Prefer `dominant_regime`, else `regime_assigned`."""
    if "dominant_regime" in df.columns:
        return pd.to_numeric(df["dominant_regime"], errors="coerce")
    if "regime_assigned" in df.columns:
        return pd.to_numeric(df["regime_assigned"], errors="coerce")
    raise ValueError("DataFrame must contain dominant_regime or regime_assigned")


def infer_n_regimes(df: pd.DataFrame) -> int:
    prob_cols = [c for c in df.columns if c.startswith("regime_prob_")]
    if prob_cols:
        return len(prob_cols)
    s = resolve_dominant_regime_series(df)
    m = float(np.nanmax(s)) if len(s) else 0.0
    return int(m) + 1 if np.isfinite(m) else 4


def _ensure_diag_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    Ensure realized_vol, avg_corr, market_ret exist; if missing, cannot compute
    labels from data (caller should re-run backtest).
    """
    need = ("realized_vol", "avg_corr", "market_ret")
    if all(c in df.columns for c in need):
        return df, True
    return df, False


def compute_regime_diagnostics(
    df: pd.DataFrame,
    crisis_windows: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Per-regime statistics and crisis-window dominant-regime distributions.

    Returns dict with:
      - regime_stats: DataFrame [regime_id, mean_realized_vol, mean_avg_corr,
        mean_market_ret, n_days, pct_time]
      - crisis_window_distributions: DataFrame, index = window label, columns = regime counts
      - crisis_counts_total: Series, regime_id -> count of anchor dates in any window
    """
    df = df.copy()
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    crisis_windows = crisis_windows or DEFAULT_LABEL_CRISIS_WINDOWS

    dom = resolve_dominant_regime_series(df)
    df["_dominant"] = dom

    df2, has_cols = _ensure_diag_columns(df)
    if not has_cols:
        raise ValueError(
            "Backtest missing realized_vol, avg_corr, and/or market_ret. "
            "Re-run run_backtest.py so these horizon diagnostics are saved."
        )

    K = infer_n_regimes(df2)
    rows = []
    n_total = int(df2["_dominant"].notna().sum())
    for k in range(K):
        mask = df2["_dominant"] == k
        sub = df2.loc[mask]
        n_days = int(mask.sum())
        if n_days == 0:
            rows.append(
                {
                    "regime_id": k,
                    "mean_realized_vol": np.nan,
                    "mean_avg_corr": np.nan,
                    "mean_market_ret": np.nan,
                    "n_days": 0,
                    "pct_time": 0.0,
                }
            )
            continue
        rows.append(
            {
                "regime_id": k,
                "mean_realized_vol": float(pd.to_numeric(sub["realized_vol"], errors="coerce").mean()),
                "mean_avg_corr": float(pd.to_numeric(sub["avg_corr"], errors="coerce").mean()),
                "mean_market_ret": float(pd.to_numeric(sub["market_ret"], errors="coerce").mean()),
                "n_days": n_days,
                "pct_time": float(n_days / max(n_total, 1) * 100.0),
            }
        )

    regime_stats = pd.DataFrame(rows)

    crisis_rows = []
    crisis_masks = []
    for a, b in crisis_windows:
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        label = f"{a}_to_{b}"
        m = (df2["date"] >= start) & (df2["date"] <= end)
        crisis_masks.append(m)
        sub = df2.loc[m & df2["_dominant"].notna()]
        vc = sub["_dominant"].value_counts().reindex(range(K), fill_value=0).astype(int)
        row = {"window": label, "start": str(start.date()), "end": str(end.date())}
        for kk in range(K):
            row[f"regime_{kk}_days"] = int(vc.get(kk, 0))
        crisis_rows.append(row)

    crisis_df = pd.DataFrame(crisis_rows).set_index("window")
    if crisis_masks:
        union = crisis_masks[0].copy()
        for m in crisis_masks[1:]:
            union = union | m
        cc = df2.loc[union & df2["_dominant"].notna(), "_dominant"].value_counts()
        crisis_counts_total = cc.reindex(range(K), fill_value=0).astype(int)
    else:
        crisis_counts_total = pd.Series(0, index=range(K), dtype=int)

    return {
        "regime_stats": regime_stats,
        "crisis_window_distributions": crisis_df,
        "crisis_counts_total": crisis_counts_total,
    }


def assign_regime_labels(
    regime_stats: pd.DataFrame,
    crisis_counts: pd.Series,
) -> dict[int, str]:
    """
    Map each regime_id to a semantic label using:
      - High Stress: highest vol & highest corr & dominates crisis windows (else composite tie-break)
      - Calm Bull: lowest vol among positive-mean-return regimes (else lowest vol)
      - Moderate Bull: among the last two, higher mean market return
      - Normal: remaining id
    """
    rs = regime_stats.set_index("regime_id").sort_index()
    ids = [int(i) for i in rs.index.tolist()]
    K = len(ids)
    if K == 0:
        return {}

    vol = rs["mean_realized_vol"].astype(float)
    corr = rs["mean_avg_corr"].astype(float)
    ret = rs["mean_market_ret"].astype(float)

    cc = crisis_counts.reindex(ids).fillna(0).astype(float)

    vol_max_id = int(vol.idxmax()) if vol.notna().any() else ids[0]
    corr_max_id = int(corr.idxmax()) if corr.notna().any() else ids[0]
    crisis_max_id = int(cc.idxmax()) if float(cc.sum()) > 0 else vol_max_id

    if vol_max_id == corr_max_id == crisis_max_id:
        high_stress = vol_max_id
    else:
        def _z(s: pd.Series) -> pd.Series:
            x = s.astype(float)
            sd = float(x.std(ddof=0))
            if not np.isfinite(sd) or sd < 1e-15:
                return x * 0.0
            return (x - float(x.mean())) / sd

        score = _z(vol.reindex(ids)) + _z(corr.reindex(ids)) + _z(cc.reindex(ids))
        high_stress = int(score.idxmax())

    labels: dict[int, str] = {}
    labels[high_stress] = LABEL_HIGH_STRESS
    remaining = [i for i in ids if i != high_stress]

    # Calm Bull
    candidates = [r for r in remaining if float(ret.get(r, np.nan)) > 0]
    if candidates:
        calm = int(vol.reindex(candidates).idxmin())
    else:
        calm = int(vol.reindex(remaining).idxmin())
    labels[calm] = LABEL_CALM_BULL
    remaining = [i for i in remaining if i != calm]

    if not remaining:
        return labels

    if len(remaining) == 1:
        labels[remaining[0]] = LABEL_NORMAL
        return labels

    if len(remaining) > 2:
        remaining_sorted = sorted(remaining, key=lambda r: float(ret.get(r, np.nan)), reverse=True)
        labels[remaining_sorted[0]] = LABEL_MODERATE_BULL
        for r in remaining_sorted[1:]:
            labels[r] = LABEL_NORMAL
        return labels

    # Moderate Bull vs Normal: prefer higher mean market return for Moderate Bull
    r_a, r_b = remaining[0], remaining[1]
    t_a, t_b = float(ret.get(r_a, np.nan)), float(ret.get(r_b, np.nan))
    if t_a == t_b:
        mod = r_a if vol.get(r_a, np.nan) >= vol.get(r_b, np.nan) else r_b
    else:
        mod = r_a if t_a > t_b else r_b
    norm = r_b if mod == r_a else r_a
    labels[mod] = LABEL_MODERATE_BULL
    labels[norm] = LABEL_NORMAL
    return labels


def build_regime_name_map(df: pd.DataFrame) -> dict[int, str]:
    """Run diagnostics + assignment; returns {regime_id: label}."""
    diag = compute_regime_diagnostics(df)
    return assign_regime_labels(diag["regime_stats"], diag["crisis_counts_total"])


def attach_regime_label_column(df: pd.DataFrame, regime_name_map: dict[int, str]) -> pd.DataFrame:
    """df['regime_label'] = df['dominant_regime'].map(regime_name_map) with fallbacks."""
    out = df.copy()
    dom = resolve_dominant_regime_series(out)
    out["regime_label"] = dom.map(lambda x: regime_name_map.get(int(x), "") if pd.notna(x) else "")
    return out


def attach_regime_labels_from_data(
    results: pd.DataFrame,
    *,
    save_json_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build data-driven regime_name_map from horizon diagnostics, add `regime_label`,
    optionally save mapping JSON. Restores a `date` index when present after reset_index().
    """
    tmp = results.reset_index()
    rmap = build_regime_name_map(tmp)
    if save_json_path is not None:
        save_regime_name_map(save_json_path, rmap)
    tmp = attach_regime_label_column(tmp, rmap)
    if "date" in tmp.columns:
        return tmp.set_index("date").sort_index()
    return tmp


def regime_names_in_id_order(K: int, regime_name_map: dict[int, str]) -> list[str]:
    return [regime_name_map.get(k, f"Regime {k}") for k in range(K)]


def save_regime_name_map(path: str | Path, regime_name_map: dict[int, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON keys as str for portability
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(regime_name_map.items())}, f, indent=2)


def load_regime_name_map(path: str | Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def load_or_build_regime_name_map(
    df: pd.DataFrame,
    *,
    json_path: str | Path | None = None,
) -> dict[int, str]:
    """Prefer saved JSON from a completed backtest; otherwise compute from `df`."""
    if json_path is not None and Path(json_path).exists():
        return load_regime_name_map(json_path)
    d = df.reset_index() if (df.index.name == "date" or "date" not in df.columns) else df.copy()
    return build_regime_name_map(d)


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parents[1]
    default_p = repo / "results" / "regime_covariance" / "backtest.parquet"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_p
    if not path.exists():
        alt = repo / "results" / "regime_volatility" / "backtest.parquet"
        path = alt if alt.exists() else path
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if df.index.name == "date" or "date" not in df.columns:
        df = df.reset_index()
    diag = compute_regime_diagnostics(df)
    mp = assign_regime_labels(diag["regime_stats"], diag["crisis_counts_total"])
    print("regime_stats:")
    print(diag["regime_stats"].to_string(index=False))
    print("\ncrisis regime distributions:")
    print(diag["crisis_window_distributions"].to_string())
    print("\ncrisis_counts_total:")
    print(diag["crisis_counts_total"].to_string())
    print("\nfinal regime_name_map:")
    print(mp)
