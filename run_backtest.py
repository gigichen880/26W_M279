# run_backtest.py
from __future__ import annotations

from datetime import date
import os
import numpy as np
import pandas as pd

from similarity_forecast.backtests import (
    make_eval_portfolios,
    eval_all_metrics,
    baseline_rolling_cov,
    baseline_persistence_realized_cov,
    baseline_shrink_to_diag
)
from similarity_forecast.embeddings import CorrEigenEmbedder, PCAWindowEmbedder
from similarity_forecast.target_objects import CovarianceTarget
from similarity_forecast.core import LogEuclideanSPDMean, ArithmeticSPDMean, validate_window, project_to_spd
from similarity_forecast.regimes import RegimeModel
from similarity_forecast.pipeline import RegimeAwareSimilarityForecaster
from similarity_forecast.regime_weighting import RegimeAwareWeights
from scripts.clean_data import clean_returns_matrix_at_load
import dataclasses
from numpy.typing import NDArray

def debug_R_at(R, t, date=None, tag=""):
    x = R[t, :]
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(date, tag, "EMPTY row")
        return
    print(
        date, tag,
        "row_std", float(np.std(x)),
        "row_q50|r|", float(np.median(np.abs(x))),
        "row_q99|r|", float(np.quantile(np.abs(x), 0.99)),
        "row_max|r|", float(np.max(np.abs(x))),
        "row_mean", float(np.mean(x)),
        "n", int(x.size),
    )
    
def build_model(
    lookback: int = 60,
    horizon: int = 20,
    n_regimes: int = 8,
    tau: float = 2.0,
    k_eigs: int = 32,
    ddof: int = 1,
    random_state: int = 0,
    transition_estimator: str = "hard",
    trans_smooth: float = 1.0,
) -> RegimeAwareSimilarityForecaster:
    # embedder = CorrEigenEmbedder(k=k_eigs)
    embedder = PCAWindowEmbedder(
        lookback=lookback,
        k=10,
        validate_window_fn=validate_window, 
        max_window_na_pct=0.3,
        min_stocks_with_data_pct=0.8,
        verbose_skip=False,
    )
    target = CovarianceTarget(ddof=ddof)
    # aggregator = ArithmeticSPDMean(eps_spd=1e-8)
    aggregator = LogEuclideanSPDMean(eps_spd=1e-8)
    regime_model = RegimeModel(n_regimes=n_regimes, random_state=random_state)
    return RegimeAwareSimilarityForecaster(
        embedder=embedder,
        target_object=target,
        aggregator=aggregator,
        lookback=lookback,
        horizon=horizon,
        regime_model=regime_model,
        tau=tau,
        transition_estimator=transition_estimator, 
        trans_smooth=trans_smooth,           
    )

def run_backtest(
    returns_df: pd.DataFrame,
    lookback: int = 60,
    horizon: int = 20,
    start_date: str | None = None,
    end_date: str | None = None,
    k_neighbors: int = 50,
    refit_every: int = 20,
    long_only: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:

    if not isinstance(returns_df.index, pd.DatetimeIndex):
        raise ValueError("returns_df.index must be a DatetimeIndex for mapping anchor dates.")

    df = returns_df.sort_index()
    if start_date is not None:
        df = df.loc[pd.to_datetime(start_date) :]
    if end_date is not None:
        df = df.loc[: pd.to_datetime(end_date)]

    R = df.to_numpy(dtype=float)  # (T, N)
    dates = df.index
    T, N = R.shape

    burn_in = 252
    raw_anchor_start = lookback + horizon + burn_in
    raw_anchor_end = T - horizon - 1
    min_samples_for_gmm = 50

    # fixed portfolio set for evaluation (kept constant across time)
    W_eval = make_eval_portfolios(N=N, n_rand=20, seed=0, long_only=True)

    rows = []
    model: RegimeAwareSimilarityForecaster | None = None
    last_refit_raw_anchor: int | None = None

    # weight stability tracking (min-var)
    w_prev: np.ndarray | None = None

    for raw_anchor in range(raw_anchor_start, raw_anchor_end + 1):
        if (model is None) or (last_refit_raw_anchor is None) or (raw_anchor - last_refit_raw_anchor >= refit_every):
            train_df = df.iloc[: raw_anchor + 1]
            model = build_model(
                lookback=lookback,
                horizon=horizon,
                transition_estimator="soft",  # or "hard"
                trans_smooth=1.0,
            )

            if verbose:
                print(f"[refit] raw_anchor={raw_anchor} date={dates[raw_anchor].date()} train_T={len(train_df)}")

            model.fit(train_df)

            if model.anchor_dates_ is None or len(model.anchor_dates_) < min_samples_for_gmm:
                if verbose:
                    t0 = 0 if model.anchor_dates_ is None else len(model.anchor_dates_)
                    print(f"[skip-refit] too few valid windows for regimes: T0={t0}")
                model = None
                continue

            last_refit_raw_anchor = raw_anchor
            if verbose:
                print(f"        samples_T0={len(model.anchor_dates_)}")

        assert model is not None
        anchor_date = dates[raw_anchor]

        past = R[raw_anchor - lookback + 1 : raw_anchor + 1, :]  # (L, N)

        try:
            Sigma_hat = model.predict_at_raw_anchor(
                past=past,
                raw_anchor=raw_anchor,
                k_neighbors=k_neighbors,
                use_filter=True,
            )
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} prediction failed: {e}")
            continue
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)

        fut = R[raw_anchor + 1 : raw_anchor + horizon + 1, :]  # (H, N)
        if not validate_window(fut, max_na_pct=0.3, min_stocks_pct=0.8):
            if verbose:
                print(f"[skip] {anchor_date.date()} future window failed validation")
            continue

        try:
            Sigma_true = model.target_object.target(fut)
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} true target failed: {e}")
            continue
        Sigma_true = np.asarray(Sigma_true, dtype=float)

        # --- OUR MODEL ---
        m = eval_all_metrics(Sigma_hat, Sigma_true, fut=fut, long_only=long_only, W_eval=W_eval)
        m = {f"model_{k}": v for k, v in m.items()}

        # --- BASELINE: rolling cov ---
        S_roll = baseline_rolling_cov(past, ddof=1)   # or use your CovarianceTarget
        b1 = eval_all_metrics(S_roll, Sigma_true, fut=fut, long_only=long_only, W_eval=W_eval)
        b1 = {f"roll_{k}": v for k, v in b1.items()}

        # --- BASELINE: persistence (prev horizon realized) ---
        S_pers = baseline_persistence_realized_cov(R, raw_anchor=raw_anchor, horizon=horizon, ddof=1)
        b2 = eval_all_metrics(S_pers, Sigma_true, fut=fut, long_only=long_only, W_eval=W_eval)
        b2 = {f"pers_{k}": v for k, v in b2.items()}

        # --- BASELINE: shrink-to-diag on rolling ---
        S_shrink = baseline_shrink_to_diag(project_to_spd((S_roll + S_roll.T) / 2.0, eps=1e-8), gamma=0.3)
        b3 = eval_all_metrics(S_shrink, Sigma_true, fut=fut, long_only=long_only, W_eval=W_eval)
        b3 = {f"shrink_{k}": v for k, v in b3.items()}

        row = {"date": anchor_date, "raw_anchor": raw_anchor, **m, **b1, **b2, **b3}

        # --- optional: SKILL ratios (model vs baseline) ---
        for key in ["fro", "kl", "nll", "stein", "logeuc", "corr_offdiag_fro", "port_mse_logvar"]:
            mk = f"model_{key}"
            bk = f"pers_{key}"  # choose your primary baseline here
            if mk in row and bk in row and np.isfinite(row[mk]) and np.isfinite(row[bk]) and row[bk] != 0:
                row[f"skill_{key}_vs_pers"] = float(row[mk] / row[bk])

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Backtest produced 0 evaluation points. "
            "Likely causes: burn_in too large, NA validation too strict, or prediction/target exceptions. "
            "Try reducing burn_in, relaxing validate_window thresholds, or printing skips."
        )

    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out


def summarize_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """
    Extended comparison:
      - Mean metric per method
      - Skill vs each baseline
      - Win-rate vs each baseline
      - Skill vs best baseline
      - Win-rate vs best baseline
    """

    compare_keys = [
        "fro", "kl", "nll", "stein", "logeuc",
        "corr_offdiag_fro", "corr_spearman",
        "eig_log_mse", "cond_ratio",
        "port_mse_var", "port_mse_logvar", "port_mae_logvar",
    ]

    baselines = ["pers", "roll", "shrink"]
    rows = []

    for key in compare_keys:
        m_col = f"model_{key}"
        if m_col not in results.columns:
            continue

        row = {"metric": key}

        # ---- Means ----
        for label in ["model"] + baselines:
            col = f"{label}_{key}"
            row[label] = float(results[col].mean()) if col in results.columns else np.nan

        # ---- Skill + Win-rate vs each baseline ----
        for b in baselines:
            b_col = f"{b}_{key}"
            if b_col not in results.columns:
                row[f"skill_vs_{b}"] = np.nan
                row[f"win_rate_vs_{b}"] = np.nan
                continue

            m = results[m_col].to_numpy(dtype=float)
            bb = results[b_col].to_numpy(dtype=float)

            valid = np.isfinite(m) & np.isfinite(bb)
            if not np.any(valid):
                row[f"skill_vs_{b}"] = np.nan
                row[f"win_rate_vs_{b}"] = np.nan
                continue

            m = m[valid]
            bb = bb[valid]

            if key == "corr_spearman":
                diff = m - bb
                row[f"skill_vs_{b}"] = float(np.mean(diff))
                row[f"win_rate_vs_{b}"] = float(np.mean(diff > 0))
            else:
                ratio = m / np.maximum(bb, 1e-12)
                row[f"skill_vs_{b}"] = float(np.mean(ratio))
                row[f"win_rate_vs_{b}"] = float(np.mean(m < bb))

        # ---- vs BEST baseline per date ----
        b_cols = [f"{b}_{key}" for b in baselines if f"{b}_{key}" in results.columns]

        if len(b_cols) >= 2:
            m = results[m_col].to_numpy(dtype=float)
            B = results[b_cols].to_numpy(dtype=float)

            valid = np.isfinite(m) & np.isfinite(B).all(axis=1)
            if np.any(valid):
                m = m[valid]
                B = B[valid]

                if key == "corr_spearman":
                    best = np.max(B, axis=1)
                    diff = m - best
                    row["skill_vs_best"] = float(np.mean(diff))
                    row["win_rate_vs_best"] = float(np.mean(diff > 0))
                else:
                    best = np.min(B, axis=1)
                    ratio = m / np.maximum(best, 1e-12)
                    row["skill_vs_best"] = float(np.mean(ratio))
                    row["win_rate_vs_best"] = float(np.mean(m < best))
            else:
                row["skill_vs_best"] = np.nan
                row["win_rate_vs_best"] = np.nan
        else:
            row["skill_vs_best"] = np.nan
            row["win_rate_vs_best"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)

def main():
    os.makedirs("results", exist_ok=True)

    returns_df = clean_returns_matrix_at_load(
        parquet_path="data/processed/returns_universe_100.parquet",
        policy="drop_date",          # safest first pass
        q99_thresh=0.5,
        max_thresh=1.0,
        min_non_nan_frac=0.2,
    ).T
    print(returns_df.head())
    returns_df.index = pd.to_datetime(returns_df.index)

    results = run_backtest(
        returns_df=returns_df,
        lookback=40,
        horizon=30,
        start_date="2015-01-01",
        end_date="2018-5-31",
        k_neighbors=10,
        refit_every=5,
        long_only=False,
        verbose=True,
    )

    # ----------------------------
    # Headline metrics summary
    # ----------------------------
    BASE = [
        "fro", "kl", "nll", "stein", "logeuc",
        "corr_offdiag_fro", "corr_spearman",
        "eig_log_mse", "cond_ratio",
        "pred_var", "real_var",
        "port_mse_var", "port_mse_logvar", "port_mae_logvar",
        "w_hhi", "w_max_abs", "w_l1",
    ]
    METHODS = ["model", "pers", "roll", "shrink"]

    headline_cols = []
    for m in METHODS:
        headline_cols += [f"{m}_{k}" for k in BASE]
    headline_cols = [c for c in headline_cols if c in results.columns]

    print("\n===== SUMMARY (headline metrics) =====")
    desc = results[headline_cols].describe()
    print(desc)

    # save describe() to CSV (flatten index)
    desc_out = desc.copy()
    desc_out.insert(0, "stat", desc_out.index)
    desc_out.reset_index(drop=True, inplace=True)
    desc_out.to_csv("results/regime_similarity_headline_describe.csv", index=False)

    # ----------------------------
    # Portfolio calibration (GMVP) summary
    # ----------------------------
    calib_row = {}
    pred_col = "model_pred_var" if "model_pred_var" in results.columns else "pred_var"
    real_col = "model_real_var" if "model_real_var" in results.columns else "real_var"

    if pred_col in results.columns and real_col in results.columns:
        tmp = results[[pred_col, real_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) > 5:
            ratio = tmp[real_col] / np.maximum(tmp[pred_col], 1e-12)
            calib_row = {
                "section": "gmvp_calibration",
                "n": int(len(tmp)),
                "real_over_pred_mean": float(ratio.mean()),
                "real_over_pred_median": float(ratio.median()),
                "real_over_pred_p90": float(ratio.quantile(0.90)),
                "real_over_pred_p99": float(ratio.quantile(0.99)),
                "corr_pred_real": float(tmp[pred_col].corr(tmp[real_col])),
            }

            print("\n===== PORTFOLIO (GMVP) CALIBRATION =====")
            for k, v in calib_row.items():
                if k != "section":
                    print(k, "=", v)

    # ----------------------------
    # Multi-portfolio + stability summaries (describe)
    # ----------------------------
    multi_cols = [c for c in ["model_port_mse_var", "model_port_mse_logvar", "model_port_mae_logvar",
                             "port_mse_var", "port_mse_logvar", "port_mae_logvar"] if c in results.columns]
    stab_cols = [c for c in ["turnover_l1", "model_turnover_l1", "model_w_hhi", "model_w_max_abs", "model_w_l1",
                            "w_hhi", "w_max_abs", "w_l1"] if c in results.columns]

    if multi_cols:
        print("\n===== PORTFOLIO (MULTI) RISK ERRORS =====")
        multi_desc = results[multi_cols].describe()
        print(multi_desc)
        multi_out = multi_desc.copy()
        multi_out.insert(0, "stat", multi_out.index)
        multi_out.reset_index(drop=True, inplace=True)
        multi_out.to_csv("results/regime_similarity_multi_portfolio_describe.csv", index=False)

    if stab_cols:
        print("\n===== PORTFOLIO STABILITY =====")
        stab_desc = results[stab_cols].describe()
        print(stab_desc)
        stab_out = stab_desc.copy()
        stab_out.insert(0, "stat", stab_out.index)
        stab_out.reset_index(drop=True, inplace=True)
        stab_out.to_csv("results/regime_similarity_stability_describe.csv", index=False)

    # ----------------------------
    # Extended method comparison (means + skill + win-rates)
    # ----------------------------
    print("\n===== METHOD COMPARISON (extended) =====")
    comparison = summarize_comparison(results)
    print(comparison)
    comparison.to_csv("results/regime_similarity_comparison.csv", index=False)

    # ----------------------------
    # One combined "report all" CSV
    # (easy to read + paste into paper)
    # ----------------------------
    report_rows = []

    # (A) headline means (per method per metric)
    for key in BASE:
        for m in METHODS:
            col = f"{m}_{key}"
            if col in results.columns:
                report_rows.append({
                    "section": "headline_mean",
                    "metric": key,
                    "method": m,
                    "value": float(np.nanmean(results[col].to_numpy(dtype=float))),
                })

    # (B) GMVP calibration
    if calib_row:
        report_rows.append(calib_row)

    # (C) extended comparison table (already aggregated)
    # store it in long format so it's a single CSV (no wide column explosion)
    for _, r in comparison.iterrows():
        metric = r["metric"]
        for c in comparison.columns:
            if c == "metric":
                continue
            report_rows.append({
                "section": "comparison",
                "metric": metric,
                "method": c,          # e.g., 'model', 'skill_vs_pers', 'win_rate_vs_best', ...
                "value": float(r[c]) if pd.notnull(r[c]) else np.nan,
            })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv("results/regime_similarity_report.csv", index=False)

    # ----------------------------
    # Save full backtest outputs
    # ----------------------------
    results.to_parquet("results/regime_similarity_backtest.parquet")
    results.to_csv("results/regime_similarity_backtest.csv")

    print("\nSaved:")
    print("  results/regime_similarity_backtest.parquet")
    print("  results/regime_similarity_backtest.csv")
    print("  results/regime_similarity_headline_describe.csv")
    print("  results/regime_similarity_multi_portfolio_describe.csv" if multi_cols else "  (no multi-portfolio describe)")
    print("  results/regime_similarity_stability_describe.csv" if stab_cols else "  (no stability describe)")
    print("  results/regime_similarity_comparison.csv")
    print("  results/regime_similarity_report.csv")


if __name__ == "__main__":
    main()