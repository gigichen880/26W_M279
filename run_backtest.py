# run_backtest.py
from __future__ import annotations

from datetime import date
import os
import numpy as np
import pandas as pd

from similarity_forecast.backtests import (
    eval_all_metrics,
    baseline_rolling_cov,
    baseline_persistence_realized_cov,
    baseline_shrink_to_diag,
    gmvp_weights,
    hold_period_portfolio_stats,
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

def _spd_floor(S: np.ndarray, eps: float) -> np.ndarray:
    """
    Add diagonal ridge + SPD projection for inversion stability.
    eps is in covariance units (return^2).
    """
    if eps is None or eps <= 0:
        return S
    S2 = (S + S.T) / 2.0
    S2 = S2 + float(eps) * np.eye(S2.shape[0])
    return project_to_spd(S2, eps=1e-8)

def _mix_cov(S_model: np.ndarray, S_shrink: np.ndarray, lam: float) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    S = (1.0 - lam) * S_shrink + lam * S_model
    return project_to_spd((S + S.T) / 2.0, eps=1e-8)

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
    sample_stride: int = 1,
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
        sample_stride=sample_stride,         
    )

def trace_ratio_guardrail(S_hat, S_ref, lo=0.2, hi=5.0) -> tuple[bool, float]:
    tr_hat = float(np.trace(S_hat))
    tr_ref = float(np.trace(S_ref))
    if (not np.isfinite(tr_hat)) or (not np.isfinite(tr_ref)) or tr_ref <= 0:
        return True, np.nan
    r = tr_hat / tr_ref
    return (r > hi), r


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
    stride: int = 5,          # evaluate every stride days
    neighbor_gap: int = 5,    # disallow neighbors too close in time

    # shrinkage mixing + eigenvalue floor
    mix_lambda: float = 0.3,        # λ in (1-λ)S_shrink + λ S_model
    floor_eps: float = 1e-4,        # diagonal ridge (in return^2 units)
    apply_floor_to: str = "all",    # "all" or "gmvp_only"
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

    rows = []
    model: RegimeAwareSimilarityForecaster | None = None
    last_refit_raw_anchor: int | None = None

    # GMVP stability tracking per method
    w_prev = {"model": None, "mix": None, "roll": None, "pers": None, "shrink": None}

    def _turnover(w_prev_i, w_now):
        if w_prev_i is None:
            return np.nan
        return float(np.sum(np.abs(w_now - w_prev_i)))

    def _hhi(w):  return float(np.sum(np.square(w)))
    def _wmax(w): return float(np.max(np.abs(w)))
    def _wl1(w):  return float(np.sum(np.abs(w)))

    for raw_anchor in range(raw_anchor_start, raw_anchor_end + 1):
        if stride > 1:
            # apply stride to reduce number of evaluation points
            if (raw_anchor - raw_anchor_start) % stride != 0:
                continue

        # ----------------------------
        # (0) Refit model occasionally (walk-forward)
        # ----------------------------
        if (model is None) or (last_refit_raw_anchor is None) or (raw_anchor - last_refit_raw_anchor >= refit_every):
            train_df = df.iloc[: raw_anchor + 1]
            model = build_model(
                lookback=lookback,
                horizon=horizon,
                transition_estimator="soft",  # or "hard"
                trans_smooth=1.0,
                sample_stride=5, 
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

        # ----------------------------
        # (1) Build past + future windows
        # ----------------------------
        past = R[raw_anchor - lookback + 1 : raw_anchor + 1, :]          # (L, N)
        fut  = R[raw_anchor + 1 : raw_anchor + horizon + 1, :]           # (H, N)

        if not validate_window(fut, max_na_pct=0.3, min_stocks_pct=0.8):
            if verbose:
                print(f"[skip] {anchor_date.date()} future window failed validation")
            continue

        # ----------------------------
        # (2) Predict covariance (model)
        # ----------------------------
        try:
            Sigma_hat = model.predict_at_raw_anchor(
                past=past,
                raw_anchor=raw_anchor,
                k_neighbors=k_neighbors,
                use_filter=True,
                neighbor_gap=neighbor_gap,
            )
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} prediction failed: {e}")
            continue

        Sigma_hat = np.asarray(Sigma_hat, dtype=float)

        # ----------------------------
        # (3) Realized target covariance
        # ----------------------------
        try:
            Sigma_true = model.target_object.target(fut)
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} true target failed: {e}")
            continue

        Sigma_true = np.asarray(Sigma_true, dtype=float)

        # ----------------------------
        # (4) Baselines (past-only)
        # ----------------------------
        S_roll = baseline_rolling_cov(past, ddof=1)
        S_roll = project_to_spd((S_roll + S_roll.T) / 2.0, eps=1e-8)

        S_pers = baseline_persistence_realized_cov(R, raw_anchor=raw_anchor, horizon=horizon, ddof=1)
        S_pers = project_to_spd((S_pers + S_pers.T) / 2.0, eps=1e-8)

        S_shrink = baseline_shrink_to_diag(S_roll, gamma=0.3)
        S_shrink = project_to_spd((S_shrink + S_shrink.T) / 2.0, eps=1e-8)

        # ----------------------------
        # (5) Guardrail on model Sigma_hat (NO lookahead)
        #     Apply BEFORE GMVP + BEFORE metrics so everything is consistent.
        # ----------------------------
        bad, ratio = trace_ratio_guardrail(Sigma_hat, S_roll, lo=0.2, hi=5.0)
        if bad:
            if verbose:
                print(f"[guardrail] {anchor_date.date()} trace_ratio={ratio} -> fallback shrink")
            Sigma_hat_use = S_shrink
        else:
            Sigma_hat_use = Sigma_hat

        # shrinkage mixing (hybrid)
        # mix uses the post-guardrail model estimate so wild forecasts can't poison the mix
        S_mix = _mix_cov(S_model=Sigma_hat_use, S_shrink=S_shrink, lam=mix_lambda)

        # ----------------------------
        # (6) GMVP strategy backtest (hold-out)
        # ----------------------------
        # eigenvalue floor (ridge) to stabilize inversion / GMVP
        if apply_floor_to not in {"all", "gmvp_only"}:
            raise ValueError("apply_floor_to must be one of {'all','gmvp_only'}")

        S_model_use = _spd_floor(Sigma_hat_use, floor_eps)
        S_mix_use   = _spd_floor(S_mix,         floor_eps)
        S_roll_use  = _spd_floor(S_roll,        floor_eps)
        S_pers_use  = _spd_floor(S_pers,        floor_eps)
        S_shrk_use  = _spd_floor(S_shrink,      floor_eps)

        # compute weights
        w_model  = gmvp_weights(S_model_use, long_only=long_only)
        w_mix    = gmvp_weights(S_mix_use,   long_only=long_only)
        w_roll_i = gmvp_weights(S_roll_use,  long_only=long_only)
        w_pers_i = gmvp_weights(S_pers_use,  long_only=long_only)
        w_shrk_i = gmvp_weights(S_shrk_use,  long_only=long_only)

        # turnover
        turn_model = _turnover(w_prev["model"],  w_model)
        turn_mix   = _turnover(w_prev["mix"],    w_mix)
        turn_roll  = _turnover(w_prev["roll"],   w_roll_i)
        turn_pers  = _turnover(w_prev["pers"],   w_pers_i)
        turn_shrk  = _turnover(w_prev["shrink"], w_shrk_i)

        w_prev["model"], w_prev["mix"], w_prev["roll"], w_prev["pers"], w_prev["shrink"] = (
            w_model, w_mix, w_roll_i, w_pers_i, w_shrk_i
        )

        # realized hold-period stats (uses realized fut returns)
        s_model = hold_period_portfolio_stats(fut=fut, w=w_model)
        s_mix   = hold_period_portfolio_stats(fut=fut, w=w_mix)
        s_roll  = hold_period_portfolio_stats(fut=fut, w=w_roll_i)
        s_pers  = hold_period_portfolio_stats(fut=fut, w=w_pers_i)
        s_shrk  = hold_period_portfolio_stats(fut=fut, w=w_shrk_i)

        # ----------------------------
        # (7) Covariance matrix error metrics (model + baselines + mix)
        # ----------------------------
        # If apply_floor_to == "all", evaluate matrix metrics on floored covariances too.
        # If gmvp_only, evaluate metrics on the original (non-floored) covariances.
        if apply_floor_to == "all":
            S_model_metric = S_model_use
            S_mix_metric   = S_mix_use
            S_roll_metric  = S_roll_use
            S_pers_metric  = S_pers_use
            S_shrk_metric  = S_shrk_use
        else:
            S_model_metric = Sigma_hat_use
            S_mix_metric   = S_mix
            S_roll_metric  = S_roll
            S_pers_metric  = S_pers
            S_shrk_metric  = S_shrink

        m_model = eval_all_metrics(S_model_metric, Sigma_true, fut=fut, long_only=long_only, W_eval=None)
        m_mix   = eval_all_metrics(S_mix_metric,   Sigma_true, fut=fut, long_only=long_only, W_eval=None)
        m_roll  = eval_all_metrics(S_roll_metric,  Sigma_true, fut=fut, long_only=long_only, W_eval=None)
        m_pers  = eval_all_metrics(S_pers_metric,  Sigma_true, fut=fut, long_only=long_only, W_eval=None)
        m_shrk  = eval_all_metrics(S_shrk_metric,  Sigma_true, fut=fut, long_only=long_only, W_eval=None)

        # ----------------------------
        # (8) Assemble row
        # ----------------------------
        row = {
            "date": anchor_date,
            "raw_anchor": raw_anchor,
            "guardrail_trace_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
            "guardrail_triggered": bool(bad),
            "mix_lambda": float(mix_lambda),
            "floor_eps": float(floor_eps),
            "apply_floor_to": apply_floor_to,
        }

        # matrix metrics
        row.update({f"model_{k}": v for k, v in m_model.items()})
        row.update({f"mix_{k}":   v for k, v in m_mix.items()})
        row.update({f"roll_{k}":  v for k, v in m_roll.items()})
        row.update({f"pers_{k}":  v for k, v in m_pers.items()})
        row.update({f"shrink_{k}": v for k, v in m_shrk.items()})

        # GMVP strategy metrics
        for k, v in s_model.items(): row[f"model_{k}"] = v
        for k, v in s_mix.items():   row[f"mix_{k}"]   = v
        for k, v in s_roll.items():  row[f"roll_{k}"]  = v
        for k, v in s_pers.items():  row[f"pers_{k}"]  = v
        for k, v in s_shrk.items():  row[f"shrink_{k}"] = v

        row["model_turnover_l1"]  = turn_model
        row["mix_turnover_l1"]    = turn_mix
        row["roll_turnover_l1"]   = turn_roll
        row["pers_turnover_l1"]   = turn_pers
        row["shrink_turnover_l1"] = turn_shrk

        # weight concentration diagnostics
        row["model_w_hhi"] = _hhi(w_model);  row["model_w_max_abs"] = _wmax(w_model);  row["model_w_l1"] = _wl1(w_model)
        row["mix_w_hhi"]   = _hhi(w_mix);    row["mix_w_max_abs"]   = _wmax(w_mix);    row["mix_w_l1"]   = _wl1(w_mix)
        row["roll_w_hhi"]  = _hhi(w_roll_i); row["roll_w_max_abs"]  = _wmax(w_roll_i); row["roll_w_l1"]  = _wl1(w_roll_i)
        row["pers_w_hhi"]  = _hhi(w_pers_i); row["pers_w_max_abs"]  = _wmax(w_pers_i); row["pers_w_l1"]  = _wl1(w_pers_i)
        row["shrink_w_hhi"]= _hhi(w_shrk_i); row["shrink_w_max_abs"]= _wmax(w_shrk_i); row["shrink_w_l1"]= _wl1(w_shrk_i)

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Backtest produced 0 evaluation points. "
            "Likely causes: burn_in too large, NA validation too strict, or prediction/target exceptions. "
            "Try reducing burn_in or relaxing validate_window thresholds."
        )

    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out

def main():
    os.makedirs("results", exist_ok=True)

    # ----------------------------
    # Load returns panel
    # ----------------------------
    returns_df = clean_returns_matrix_at_load(
        parquet_path="data/processed/returns_universe_100.parquet",
        policy="drop_date",
        q99_thresh=0.5,
        max_thresh=1.0,
        min_non_nan_frac=0.2,
    ).T
    returns_df.index = pd.to_datetime(returns_df.index)

    # ----------------------------
    # Run backtest (matrix errors + GMVP only)
    # ----------------------------
    results = run_backtest(
        returns_df=returns_df,
        lookback=40,
        horizon=30,
        start_date="2019-01-01",
        end_date="2021-12-31",
        k_neighbors=10,
        refit_every=5,
        long_only=False,
        verbose=True,
        stride=5,
        neighbor_gap=5,
    )

    # ----------------------------
    # Report
    # ----------------------------
    METHODS = ["model", "mix", "roll", "pers", "shrink"]

    # (1) Covariance matrix error metrics (edit this list to taste)
    COV_METRICS = [
        "fro", "kl", "stein", "logeuc",
        "corr_offdiag_fro", "corr_spearman",
        "eig_log_mse", "cond_ratio",
    ]

    # (2) GMVP strategy metrics (from hold_period_portfolio_stats + extras)
    GMVP_METRICS = [
        "gmvp_cumret", "gmvp_mean", "gmvp_var", "gmvp_vol", "gmvp_sharpe",
        "turnover_l1",
        "w_hhi", "w_max_abs", "w_l1",
    ]

    # Collect available columns (don’t assume all exist)
    headline_cols = []
    for m in METHODS:
        headline_cols += [f"{m}_{k}" for k in COV_METRICS]
        headline_cols += [f"{m}_{k}" for k in GMVP_METRICS]
    headline_cols = [c for c in headline_cols if c in results.columns]

    # ----------------------------
    # Print + save headline describe()
    # ----------------------------
    print("\n===== SUMMARY (covariance errors + GMVP strategy) =====")
    desc = results[headline_cols].replace([np.inf, -np.inf], np.nan).describe()
    print(desc)

    desc_out = desc.copy()
    desc_out.insert(0, "stat", desc_out.index)
    desc_out.reset_index(drop=True, inplace=True)
    desc_out.to_csv("results/regime_similarity_headline_describe.csv", index=False)

    # ----------------------------
    # Save backtest outputs
    # ----------------------------
    results.to_parquet("results/regime_similarity_backtest.parquet")
    results.to_csv("results/regime_similarity_backtest.csv")

    # ----------------------------
    # Save a compact long-format "report" CSV (easy to paste into paper)
    # ----------------------------
    report_rows = []

    def _mean(col: str) -> float:
        x = results[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    # (A) covariance error means
    for metric in COV_METRICS:
        for m in METHODS:
            col = f"{m}_{metric}"
            if col in results.columns:
                report_rows.append({
                    "section": "cov_error_mean",
                    "metric": metric,
                    "method": m,
                    "value": _mean(col),
                })

    # (B) GMVP strategy means
    for metric in GMVP_METRICS:
        for m in METHODS:
            col = f"{m}_{metric}"
            if col in results.columns:
                report_rows.append({
                    "section": "gmvp_mean",
                    "metric": metric,
                    "method": m,
                    "value": _mean(col),
                })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv("results/regime_similarity_report.csv", index=False)

    print("\nSaved:")
    print("  results/regime_similarity_backtest.parquet")
    print("  results/regime_similarity_backtest.csv")
    print("  results/regime_similarity_headline_describe.csv")
    print("  results/regime_similarity_report.csv")

if __name__ == "__main__":
    main()