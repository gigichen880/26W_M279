# run_backtest.py
'''
Example Usage:
`python run_backtest.py --config configs/regime_covariance.yaml`

Overrides with hyperparams:
python run_backtest.py --config configs/regime_covariance.yaml \
  --set backtest.stride=1 \
  --set backtest.neighbor_gap=10 \
  --set mixing.mix_lambda=0.15 \
  --set model.sample_stride=10

Switch refit policy:
python run_backtest.py --config configs/regime_covariance.yaml \
  --set backtest.refit_mode=days \
  --set backtest.refit_every_days=20

'''

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from scripts.config_utils import load_yaml, deep_update, parse_overrides

from similarity_forecast.backtests import (
    eval_all_metrics,
    eval_vol_metrics,
    baseline_rolling_cov,
    baseline_persistence_realized_cov,
    baseline_shrink_to_diag,
    baseline_rolling_vol,
    baseline_persistence_vol,
    baseline_shrink_vol,
    gmvp_weights,
    hold_period_portfolio_stats,
)
from similarity_forecast.embeddings import CorrEigenEmbedder, PCAWindowEmbedder, VolStatsEmbedder
from similarity_forecast.target_objects import CovarianceTarget, VolTarget
from similarity_forecast.core import (
    LogEuclideanSPDMean,
    ArithmeticSPDMean,
    EuclideanMean,
    validate_window,
    project_to_spd,
)
from similarity_forecast.regimes import RegimeModel
from similarity_forecast.pipeline import RegimeAwareSimilarityForecaster
from scripts.clean_data import clean_returns_matrix_at_load


# ----------------------------
# Helpers
# ----------------------------
def _spd_floor(S: np.ndarray, eps: float, proj_eps: float = 1e-8) -> np.ndarray:
    """
    Add diagonal ridge + SPD projection for inversion stability.
    eps is in covariance units (return^2).
    """
    eps = float(eps)
    if eps <= 0:
        return S
    S2 = (S + S.T) / 2.0
    S2 = S2 + eps * np.eye(S2.shape[0])
    return project_to_spd(S2, eps=proj_eps)


def _mix_cov(S_model: np.ndarray, S_shrink: np.ndarray, lam: float, proj_eps: float = 1e-8) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    S = (1.0 - lam) * S_shrink + lam * S_model
    return project_to_spd((S + S.T) / 2.0, eps=proj_eps)


def trace_ratio_guardrail(S_hat: np.ndarray, S_ref: np.ndarray, lo: float, hi: float) -> tuple[bool, float]:
    """
    Returns (triggered, ratio). Triggered when ratio outside [lo, hi] or invalid.
    """
    lo = float(lo)
    hi = float(hi)
    tr_hat = float(np.trace(S_hat))
    tr_ref = float(np.trace(S_ref))
    if (not np.isfinite(tr_hat)) or (not np.isfinite(tr_ref)) or tr_ref <= 0:
        return True, np.nan
    r = tr_hat / tr_ref
    if (not np.isfinite(r)) or (r < lo) or (r > hi):
        return True, r
    return False, r


def build_model(
    *,
    target_type: str,
    lookback: int,
    horizon: int,
    ddof: int,
    # regimes
    n_regimes: int,
    tau: float,
    random_state: int,
    transition_estimator: str,
    trans_smooth: float,
    sample_stride: int,
    # embedder config
    embedder_name: str,
    pca_k: int,
    k_eigs: int,
    gmm_init_params: str,
    gmm_n_init: int,
    max_window_na_pct: float,
    min_stocks_with_data_pct: float,
    verbose_skip: bool,
    # aggregator config (eps_spd ignored for target_type volatility)
    aggregator_name: str,
    eps_spd: float,
    # ablation / design choices
    knn_metric: str = "l2",
    regime_aggregation: str = "soft",
) -> RegimeAwareSimilarityForecaster:
    embedder_name = str(embedder_name).lower()
    aggregator_name = str(aggregator_name).lower()
    target_type = str(target_type).lower()

    if embedder_name == "corr_eig":
        embedder = CorrEigenEmbedder(k=int(k_eigs))
    elif embedder_name == "pca":
        embedder = PCAWindowEmbedder(
            lookback=int(lookback),
            k=int(pca_k),
            validate_window_fn=validate_window,
            max_window_na_pct=float(max_window_na_pct),
            min_stocks_with_data_pct=float(min_stocks_with_data_pct),
            verbose_skip=bool(verbose_skip),
        )
    elif embedder_name == "vol_stats":
        # Rich vol embedding: distribution (mean, std, quantiles, IQR), trend (1st vs 2nd half), concentration (HHI).
        embedder = VolStatsEmbedder(ddof=int(ddof))
    else:
        raise ValueError("embedder.name must be one of {'pca','corr_eig','vol_stats'}")

    if target_type == "volatility":
        target = VolTarget(ddof=int(ddof))
        if aggregator_name == "euclidean":
            aggregator = EuclideanMean()
        else:
            aggregator = EuclideanMean()  # vol requires vector mean; ignore aggregator_name
    elif target_type == "covariance":
        target = CovarianceTarget(ddof=int(ddof))
        if aggregator_name == "logeuc":
            aggregator = LogEuclideanSPDMean(eps_spd=float(eps_spd))
        elif aggregator_name == "arith":
            aggregator = ArithmeticSPDMean(eps_spd=float(eps_spd))
        else:
            raise ValueError("aggregator.name for covariance must be one of {'logeuc','arith'}")
    else:
        raise ValueError("target_type must be one of {'covariance','volatility'}")

    regime_model = RegimeModel(
        n_regimes=int(n_regimes),
        random_state=int(random_state),
        gmm_init_params=str(gmm_init_params),
        gmm_n_init=int(gmm_n_init),
    )
    return RegimeAwareSimilarityForecaster(
        embedder=embedder,
        target_object=target,
        aggregator=aggregator,
        lookback=int(lookback),
        horizon=int(horizon),
        regime_model=regime_model,
        tau=float(tau),
        transition_estimator=str(transition_estimator),
        trans_smooth=float(trans_smooth),
        sample_stride=int(sample_stride),
        knn_metric=str(knn_metric).lower(),
        regime_aggregation=str(regime_aggregation).lower(),
    )


# ----------------------------
# Backtest
# ----------------------------
def run_backtest(
    *,
    returns_df: pd.DataFrame,
    target_type: str,

    # date range
    start_date: str | None,
    end_date: str | None,

    # model (core)
    lookback: int,
    horizon: int,
    ddof: int,
    n_regimes: int,
    tau: float,
    random_state: int,
    transition_estimator: str,
    trans_smooth: float,
    sample_stride: int,

    # embedder
    embedder_name: str,
    pca_k: int,
    k_eigs: int,
    max_window_na_pct: float,
    min_stocks_with_data_pct: float,
    verbose_skip: bool,

    # aggregator
    aggregator_name: str,
    eps_spd: float,

    # gmm
    gmm_init_params: str,
    gmm_n_init: int,

    # backtest controls
    k_neighbors: int,
    stride: int,
    neighbor_gap: int,
    long_only: bool,
    verbose: bool,

    # refit control
    refit_mode: str,
    refit_every_days: int,
    refit_every_steps: int,

    # validation (future window)
    fut_max_na_pct: float,
    fut_min_stocks_pct: float,

    # internals
    burn_in: int,
    min_samples_for_gmm: int,

    # guardrail
    trace_ratio_lo: float,
    trace_ratio_hi: float,

    # mixing
    mix_lambda: float,
    shrink_gamma: float,
    vol_dampen_toward_roll: float = 0.0,  # vol only: blend model toward rolling (0=off, 0.2=20% roll)

    # stability / floor
    floor_eps: float,
    apply_floor_to: str,

    # design choices (for ablation)
    knn_metric: str = "l2",
    regime_aggregation: str = "soft",
    regime_weighting: str = "filtered",  # "filtered" | "raw_pi"
) -> pd.DataFrame:
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        raise ValueError("returns_df.index must be a DatetimeIndex.")

    df = returns_df.sort_index()
    if start_date is not None:
        df = df.loc[pd.to_datetime(start_date) :]
    if end_date is not None:
        df = df.loc[: pd.to_datetime(end_date)]

    R = df.to_numpy(dtype=float)  # (T, N)
    dates = df.index
    T, N = R.shape

    burn_in = int(burn_in)
    raw_anchor_start = int(lookback) + int(horizon) + burn_in
    raw_anchor_end = T - int(horizon) - 1

    if raw_anchor_end < raw_anchor_start:
        raise RuntimeError(
            f"Invalid anchor range: start={raw_anchor_start}, end={raw_anchor_end}. "
            "Reduce burn_in/lookback/horizon or increase data range."
        )

    refit_mode = str(refit_mode).lower()
    if refit_mode not in {"days", "steps"}:
        raise ValueError("backtest.refit_mode must be 'days' or 'steps'.")

    apply_floor_to = str(apply_floor_to).lower()
    if apply_floor_to not in {"all", "gmvp_only"}:
        raise ValueError("stability.apply_floor_to must be 'all' or 'gmvp_only'.")

    last_refit_raw_anchor: int | None = None
    last_refit_date: pd.Timestamp | None = None
    model: RegimeAwareSimilarityForecaster | None = None

    rows: list[dict] = []
    target_type = str(target_type).lower()
    is_vol = target_type == "volatility"
    use_filter = str(regime_weighting).lower() == "filtered"

    # GMVP stability tracking per method (covariance only)
    w_prev = {"model": None, "mix": None, "roll": None, "pers": None, "shrink": None}

    def _turnover(w_prev_i, w_now):
        if w_prev_i is None:
            return np.nan
        return float(np.sum(np.abs(w_now - w_prev_i)))

    def _hhi(w):  return float(np.sum(np.square(w)))
    def _wmax(w): return float(np.max(np.abs(w)))
    def _wl1(w):  return float(np.sum(np.abs(w)))

    for raw_anchor in range(raw_anchor_start, raw_anchor_end + 1):
        if int(stride) > 1 and (raw_anchor - raw_anchor_start) % int(stride) != 0:
            continue

        anchor_date = dates[raw_anchor]

        # ----------------------------
        # (0) Refit (walk-forward)
        # ----------------------------
        do_refit = False
        if model is None:
            do_refit = True
        elif refit_mode == "days":
            assert last_refit_date is not None
            do_refit = (anchor_date - last_refit_date).days >= int(refit_every_days)
        else:  # steps
            assert last_refit_raw_anchor is not None
            do_refit = (raw_anchor - last_refit_raw_anchor) >= int(refit_every_steps)

        if do_refit:
            train_df = df.iloc[: raw_anchor + 1]
            model = build_model(
                target_type=target_type,
                lookback=int(lookback),
                horizon=int(horizon),
                ddof=int(ddof),
                n_regimes=int(n_regimes),
                tau=float(tau),
                random_state=int(random_state),
                transition_estimator=str(transition_estimator),
                trans_smooth=float(trans_smooth),
                sample_stride=int(sample_stride),
                embedder_name=str(embedder_name),
                pca_k=int(pca_k),
                k_eigs=int(k_eigs),
                gmm_init_params=str(gmm_init_params),
                gmm_n_init=int(gmm_n_init),
                max_window_na_pct=float(max_window_na_pct),
                min_stocks_with_data_pct=float(min_stocks_with_data_pct),
                verbose_skip=bool(verbose_skip),
                aggregator_name=str(aggregator_name),
                eps_spd=float(eps_spd),
                knn_metric=str(knn_metric),
                regime_aggregation=str(regime_aggregation),
            )

            if verbose:
                print(f"[refit] raw_anchor={raw_anchor} date={anchor_date.date()} train_T={len(train_df)}")

            model.fit(train_df)

            t0 = 0 if (model.anchor_dates_ is None) else len(model.anchor_dates_)
            if model.anchor_dates_ is None or t0 < int(min_samples_for_gmm):
                if verbose:
                    print(f"[skip-refit] too few valid windows for regimes: T0={t0}")
                model = None
                continue

            last_refit_raw_anchor = raw_anchor
            last_refit_date = anchor_date

            if verbose:
                print(f"        samples_T0={t0}")

        assert model is not None

        # ----------------------------
        # (1) Build past + future windows
        # ----------------------------
        past = R[raw_anchor - int(lookback) + 1 : raw_anchor + 1, :]
        fut  = R[raw_anchor + 1 : raw_anchor + int(horizon) + 1, :]

        if not validate_window(
            fut,
            max_na_pct=float(fut_max_na_pct),
            min_stocks_pct=float(fut_min_stocks_pct),
        ):
            if verbose:
                print(f"[skip] {anchor_date.date()} future window failed validation")
            continue

        # ----------------------------
        # (2) Predict and get regime assignments
        # ----------------------------
        try:
            pred_out = model.predict_at_raw_anchor(
                past=past,
                raw_anchor=raw_anchor,
                k_neighbors=int(k_neighbors),
                use_filter=use_filter,
                neighbor_gap=int(neighbor_gap),
                return_regime=True,
            )
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} prediction failed: {e}")
            continue

        if is_vol:
            vol_hat, alpha_t, pi_t = pred_out[0], pred_out[1], pred_out[2]
            vol_hat = np.asarray(vol_hat, dtype=float).reshape(-1)
        else:
            Sigma_hat, alpha_t, pi_t = pred_out[0], pred_out[1], pred_out[2]
            Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        K_regimes = alpha_t.size
        regime_assigned = int(np.argmax(alpha_t))

        # ----------------------------
        # (3) Realized target
        # ----------------------------
        try:
            y_true = model.target_object.target(fut)
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} true target failed: {e}")
            continue

        y_true = np.asarray(y_true, dtype=float)

        if is_vol:
            vol_true = y_true.reshape(-1)
            vol_roll = baseline_rolling_vol(past, ddof=int(ddof))
            vol_pers = baseline_persistence_vol(R, raw_anchor=raw_anchor, horizon=int(horizon), ddof=int(ddof))
            vol_shrink = baseline_shrink_vol(past, ddof=int(ddof), gamma=float(shrink_gamma))
            # Dampen model forecast toward rolling to reduce overshooting (helps when kNN is noisy)
            vol_dampen = float(vol_dampen_toward_roll)
            vol_hat_use = (1.0 - vol_dampen) * vol_hat + vol_dampen * vol_roll
            vol_mix = (1.0 - float(mix_lambda)) * vol_shrink + float(mix_lambda) * vol_hat_use
            m_model = eval_vol_metrics(vol_hat_use, vol_true)
            m_mix   = eval_vol_metrics(vol_mix, vol_true)
            m_roll  = eval_vol_metrics(vol_roll, vol_true)
            m_pers  = eval_vol_metrics(vol_pers, vol_true)
            m_shrk  = eval_vol_metrics(vol_shrink, vol_true)
            row = {
                "date": anchor_date,
                "raw_anchor": int(raw_anchor),
                "mix_lambda": float(mix_lambda),
                "shrink_gamma": float(shrink_gamma),
                "regime_assigned": regime_assigned,
            }
            for k in range(K_regimes):
                row[f"regime_prob_{k}"] = float(alpha_t[k])
                row[f"regime_raw_{k}"] = float(pi_t[k])
            row.update({f"model_{k}": v for k, v in m_model.items()})
            row.update({f"mix_{k}":   v for k, v in m_mix.items()})
            row.update({f"roll_{k}":  v for k, v in m_roll.items()})
            row.update({f"pers_{k}":  v for k, v in m_pers.items()})
            row.update({f"shrink_{k}": v for k, v in m_shrk.items()})
            rows.append(row)
            continue

        Sigma_true = y_true
        # ----------------------------
        # (4) Baselines (past-only) [covariance]
        # ----------------------------
        S_roll = baseline_rolling_cov(past, ddof=int(ddof))
        S_roll = project_to_spd((S_roll + S_roll.T) / 2.0, eps=1e-8)

        S_pers = baseline_persistence_realized_cov(R, raw_anchor=raw_anchor, horizon=int(horizon), ddof=int(ddof))
        S_pers = project_to_spd((S_pers + S_pers.T) / 2.0, eps=1e-8)

        S_shrink = baseline_shrink_to_diag(S_roll, gamma=float(shrink_gamma))
        S_shrink = project_to_spd((S_shrink + S_shrink.T) / 2.0, eps=1e-8)

        # ----------------------------
        # (5) Guardrail on model Sigma_hat
        # ----------------------------
        bad, ratio = trace_ratio_guardrail(Sigma_hat, S_roll, lo=float(trace_ratio_lo), hi=float(trace_ratio_hi))
        Sigma_hat_use = S_shrink if bad else Sigma_hat

        # shrinkage mixing (hybrid)
        S_mix = _mix_cov(S_model=Sigma_hat_use, S_shrink=S_shrink, lam=float(mix_lambda))

        # ----------------------------
        # (6) GMVP strategy (optionally floored)
        # ----------------------------
        # Always floor for GMVP to keep inversion stable across methods.
        S_model_use = _spd_floor(Sigma_hat_use, float(floor_eps))
        S_mix_use   = _spd_floor(S_mix,         float(floor_eps))
        S_roll_use  = _spd_floor(S_roll,        float(floor_eps))
        S_pers_use  = _spd_floor(S_pers,        float(floor_eps))
        S_shrk_use  = _spd_floor(S_shrink,      float(floor_eps))

        w_model  = gmvp_weights(S_model_use, long_only=bool(long_only))
        w_mix    = gmvp_weights(S_mix_use,   long_only=bool(long_only))
        w_roll_i = gmvp_weights(S_roll_use,  long_only=bool(long_only))
        w_pers_i = gmvp_weights(S_pers_use,  long_only=bool(long_only))
        w_shrk_i = gmvp_weights(S_shrk_use,  long_only=bool(long_only))

        turn_model = _turnover(w_prev["model"],  w_model)
        turn_mix   = _turnover(w_prev["mix"],    w_mix)
        turn_roll  = _turnover(w_prev["roll"],   w_roll_i)
        turn_pers  = _turnover(w_prev["pers"],   w_pers_i)
        turn_shrk  = _turnover(w_prev["shrink"], w_shrk_i)

        w_prev["model"], w_prev["mix"], w_prev["roll"], w_prev["pers"], w_prev["shrink"] = (
            w_model, w_mix, w_roll_i, w_pers_i, w_shrk_i
        )

        s_model = hold_period_portfolio_stats(fut=fut, w=w_model)
        s_mix   = hold_period_portfolio_stats(fut=fut, w=w_mix)
        s_roll  = hold_period_portfolio_stats(fut=fut, w=w_roll_i)
        s_pers  = hold_period_portfolio_stats(fut=fut, w=w_pers_i)
        s_shrk  = hold_period_portfolio_stats(fut=fut, w=w_shrk_i)

        # ----------------------------
        # (7) Matrix metrics: floored or not
        # ----------------------------
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

        m_model = eval_all_metrics(S_model_metric, Sigma_true, fut=fut, long_only=bool(long_only), W_eval=None)
        m_mix   = eval_all_metrics(S_mix_metric,   Sigma_true, fut=fut, long_only=bool(long_only), W_eval=None)
        m_roll  = eval_all_metrics(S_roll_metric,  Sigma_true, fut=fut, long_only=bool(long_only), W_eval=None)
        m_pers  = eval_all_metrics(S_pers_metric,  Sigma_true, fut=fut, long_only=bool(long_only), W_eval=None)
        m_shrk  = eval_all_metrics(S_shrk_metric,  Sigma_true, fut=fut, long_only=bool(long_only), W_eval=None)

        # ----------------------------
        # (8) Row (including regime data)
        # ----------------------------
        row = {
            "date": anchor_date,
            "raw_anchor": int(raw_anchor),
            "guardrail_trace_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
            "guardrail_triggered": bool(bad),
            "mix_lambda": float(mix_lambda),
            "shrink_gamma": float(shrink_gamma),
            "floor_eps": float(floor_eps),
            "apply_floor_to": str(apply_floor_to),
            "regime_assigned": regime_assigned,
        }
        for k in range(K_regimes):
            row[f"regime_prob_{k}"] = float(alpha_t[k])
            row[f"regime_raw_{k}"] = float(pi_t[k])

        row.update({f"model_{k}": v for k, v in m_model.items()})
        row.update({f"mix_{k}":   v for k, v in m_mix.items()})
        row.update({f"roll_{k}":  v for k, v in m_roll.items()})
        row.update({f"pers_{k}":  v for k, v in m_pers.items()})
        row.update({f"shrink_{k}": v for k, v in m_shrk.items()})

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

        row["model_w_hhi"] = _hhi(w_model);  row["model_w_max_abs"] = _wmax(w_model);  row["model_w_l1"] = _wl1(w_model)
        row["mix_w_hhi"]   = _hhi(w_mix);    row["mix_w_max_abs"]   = _wmax(w_mix);    row["mix_w_l1"]   = _wl1(w_mix)
        row["roll_w_hhi"]  = _hhi(w_roll_i); row["roll_w_max_abs"]  = _wmax(w_roll_i); row["roll_w_l1"]  = _wl1(w_roll_i)
        row["pers_w_hhi"]  = _hhi(w_pers_i); row["pers_w_max_abs"]  = _wmax(w_pers_i); row["pers_w_l1"]  = _wl1(w_pers_i)
        row["shrink_w_hhi"]= _hhi(w_shrk_i); row["shrink_w_max_abs"]= _wmax(w_shrk_i); row["shrink_w_l1"]= _wl1(w_shrk_i)

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Backtest produced 0 evaluation points. "
            "Likely causes: burn_in too large, NA validation too strict, or prediction/target exceptions."
        )

    return pd.DataFrame(rows).set_index("date").sort_index()

def build_report_table(results_df: pd.DataFrame, target_type: str = "covariance") -> pd.DataFrame:
    rows = []
    target_type = str(target_type).lower()

    def add(section: str, metric: str, method: str, value: float):
        rows.append({
            "section": section,
            "metric": metric,
            "method": method,
            "value": float(value),
        })

    methods = ["model", "mix", "roll", "pers", "shrink"]

    if target_type == "volatility":
        for m in methods:
            for key in ("vol_mse", "vol_mae", "vol_rmse"):
                col = f"{m}_{key}"
                if col in results_df.columns:
                    add("vol_error_mean", key, m, results_df[col].mean())
        return pd.DataFrame(rows)

    for m in methods:
        add("cov_error_mean", "fro", m, results_df[f"{m}_fro"].mean())
        add("cov_error_mean", "kl", m, results_df[f"{m}_kl"].mean())
        add("cov_error_mean", "stein", m, results_df[f"{m}_stein"].mean())
        add("cov_error_mean", "logeuc", m, results_df[f"{m}_logeuc"].mean())
        add("gmvp_mean", "gmvp_mean", m, results_df[f"{m}_gmvp_mean"].mean())
        add("gmvp_mean", "gmvp_var", m, results_df[f"{m}_gmvp_var"].mean())
        add("gmvp_mean", "gmvp_vol", m, results_df[f"{m}_gmvp_vol"].mean())
        add("gmvp_mean", "gmvp_sharpe", m, results_df[f"{m}_gmvp_sharpe"].mean())
        add("gmvp_mean", "turnover_l1", m, results_df[f"{m}_turnover_l1"].mean())
    return pd.DataFrame(rows)

# ----------------------------
# Config-driven entry (for ablation / programmatic use)
# ----------------------------
def run_backtest_from_config(cfg: dict, *, verbose: bool | None = None) -> tuple[pd.DataFrame, str]:
    """
    Load data from config, run backtest, return (results_df, target_type).
    Does not write files. Use for ablation or when you need the DataFrame in memory.
    """
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    ecfg = cfg["embedder"]
    acfg = cfg["aggregator"]
    bcfg = cfg["backtest"]
    vcfg = cfg["validation"]
    icfg = cfg["internals"]
    gcfg = cfg["guardrail"]
    mixcfg = cfg["mixing"]
    stcfg = cfg["stability"]
    target_type = str(mcfg.get("target", "covariance")).lower()
    verb = bcfg["verbose"] if verbose is None else verbose

    returns_df = clean_returns_matrix_at_load(
        parquet_path=dcfg["parquet_path"],
        policy=dcfg["policy"],
        q99_thresh=float(dcfg["q99_thresh"]),
        max_thresh=float(dcfg["max_thresh"]),
        min_non_nan_frac=float(dcfg["min_non_nan_frac"]),
    ).T
    returns_df.index = pd.to_datetime(returns_df.index)

    results = run_backtest(
        returns_df=returns_df,
        target_type=target_type,
        start_date=dcfg.get("start_date"),
        end_date=dcfg.get("end_date"),
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
        max_window_na_pct=float(ecfg["max_window_na_pct"]),
        min_stocks_with_data_pct=float(ecfg["min_stocks_with_data_pct"]),
        verbose_skip=bool(ecfg["verbose_skip"]),
        aggregator_name=str(acfg["name"]),
        eps_spd=float(acfg["eps_spd"]),
        gmm_init_params=str(mcfg["gmm_init_params"]),
        gmm_n_init=int(mcfg["gmm_n_init"]),
        k_neighbors=int(bcfg["k_neighbors"]),
        stride=int(bcfg["stride"]),
        neighbor_gap=int(bcfg["neighbor_gap"]),
        long_only=bool(bcfg["long_only"]),
        verbose=bool(verb),
        refit_mode=str(bcfg["refit_mode"]),
        refit_every_days=int(bcfg["refit_every_days"]),
        refit_every_steps=int(bcfg["refit_every_steps"]),
        fut_max_na_pct=float(vcfg["fut_max_na_pct"]),
        fut_min_stocks_pct=float(vcfg["fut_min_stocks_pct"]),
        burn_in=int(icfg["burn_in"]),
        min_samples_for_gmm=int(icfg["min_samples_for_gmm"]),
        trace_ratio_lo=float(gcfg["trace_ratio_lo"]),
        trace_ratio_hi=float(gcfg["trace_ratio_hi"]),
        mix_lambda=float(mixcfg["mix_lambda"]),
        shrink_gamma=float(mixcfg["shrink_gamma"]),
        vol_dampen_toward_roll=float(mixcfg.get("vol_dampen_toward_roll", 0.0)),
        floor_eps=float(stcfg["floor_eps"]),
        apply_floor_to=str(stcfg["apply_floor_to"]),
        knn_metric=str(mcfg.get("knn_metric", "l2")),
        regime_aggregation=str(mcfg.get("regime_aggregation", "soft")),
        regime_weighting=str(mcfg.get("regime_weighting", "filtered")),
    )
    return results, target_type


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/regime_covariance.yaml")
    ap.add_argument("--set", action="append", default=[], help="Override: section.key=value (repeatable)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    overrides = parse_overrides(args.set)
    cfg = deep_update(cfg, overrides)

    outdir = cfg["outputs"]["outdir"]
    tag = cfg["outputs"]["tag"]
    os.makedirs(outdir, exist_ok=True)

    # ----------------------------
    # Load returns panel
    # ----------------------------
    dcfg = cfg["data"]
    returns_df = clean_returns_matrix_at_load(
        parquet_path=dcfg["parquet_path"],
        policy=dcfg["policy"],
        q99_thresh=float(dcfg["q99_thresh"]),
        max_thresh=float(dcfg["max_thresh"]),
        min_non_nan_frac=float(dcfg["min_non_nan_frac"]),
    ).T
    returns_df.index = pd.to_datetime(returns_df.index)

    # ----------------------------
    # Assemble args from YAML
    # ----------------------------
    mcfg = cfg["model"]
    ecfg = cfg["embedder"]
    acfg = cfg["aggregator"]
    bcfg = cfg["backtest"]
    vcfg = cfg["validation"]
    icfg = cfg["internals"]
    gcfg = cfg["guardrail"]
    mixcfg = cfg["mixing"]
    stcfg = cfg["stability"]

    target_type = str(cfg.get("model", {}).get("target", "covariance")).lower()

    results = run_backtest(
        returns_df=returns_df,
        target_type=target_type,
        start_date=dcfg.get("start_date"),
        end_date=dcfg.get("end_date"),

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
        max_window_na_pct=float(ecfg["max_window_na_pct"]),
        min_stocks_with_data_pct=float(ecfg["min_stocks_with_data_pct"]),
        verbose_skip=bool(ecfg["verbose_skip"]),

        aggregator_name=str(acfg["name"]),
        eps_spd=float(acfg["eps_spd"]),

        gmm_init_params=str(mcfg["gmm_init_params"]),
        gmm_n_init=int(mcfg["gmm_n_init"]),

        k_neighbors=int(bcfg["k_neighbors"]),
        stride=int(bcfg["stride"]),
        neighbor_gap=int(bcfg["neighbor_gap"]),
        long_only=bool(bcfg["long_only"]),
        verbose=bool(bcfg["verbose"]),

        refit_mode=str(bcfg["refit_mode"]),
        refit_every_days=int(bcfg["refit_every_days"]),
        refit_every_steps=int(bcfg["refit_every_steps"]),

        fut_max_na_pct=float(vcfg["fut_max_na_pct"]),
        fut_min_stocks_pct=float(vcfg["fut_min_stocks_pct"]),

        burn_in=int(icfg["burn_in"]),
        min_samples_for_gmm=int(icfg["min_samples_for_gmm"]),

        trace_ratio_lo=float(gcfg["trace_ratio_lo"]),
        trace_ratio_hi=float(gcfg["trace_ratio_hi"]),

        mix_lambda=float(mixcfg["mix_lambda"]),
        shrink_gamma=float(mixcfg["shrink_gamma"]),
        vol_dampen_toward_roll=float(mixcfg.get("vol_dampen_toward_roll", 0.0)),
        floor_eps=float(stcfg["floor_eps"]),
        apply_floor_to=str(stcfg["apply_floor_to"]),
        knn_metric=str(mcfg.get("knn_metric", "l2")),
        regime_aggregation=str(mcfg.get("regime_aggregation", "soft")),
        regime_weighting=str(mcfg.get("regime_weighting", "filtered")),
    )

    results.to_parquet(os.path.join(outdir, f"{tag}_backtest.parquet"))
    results.to_csv(os.path.join(outdir, f"{tag}_backtest.csv"))

    # dump resolved config used
    import yaml
    with open(os.path.join(outdir, f"{tag}_config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("Saved:")
    print(f"  {outdir}/{tag}_backtest.parquet")
    print(f"  {outdir}/{tag}_backtest.csv")
    print(f"  {outdir}/{tag}_config_used.yaml")
    print("✓ Regime assignments saved to backtest results (columns: regime_assigned, regime_prob_*, regime_raw_*)")

    report_df = build_report_table(results, target_type=target_type)
    report_name = "regime_volatility_report.csv" if target_type == "volatility" else "regime_covariance_report.csv"
    report_path = os.path.join(outdir, report_name)
    report_df.to_csv(report_path, index=False)
    print(f"Saved report summary: {report_path}")


if __name__ == "__main__":
    main()