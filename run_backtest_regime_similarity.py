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
from similarity_forecast.core import LogEuclideanSPDMean, validate_window, project_to_spd
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



@dataclasses.dataclass(frozen=True)
class ArithmeticSPDMean:
    """
    Simple weighted arithmetic mean of SPD matrices, then project to SPD.
    """
    eps_spd: float = 1e-8

    def aggregate(self, targets: NDArray[np.floating], w: NDArray[np.floating]) -> NDArray[np.floating]:
        S = np.tensordot(w, targets, axes=(0, 0))
        return project_to_spd((S + S.T) / 2.0, eps=self.eps_spd)
    
def build_model(
    lookback: int = 60,
    horizon: int = 20,
    n_regimes: int = 8,
    tau: float = 2.0,
    k_eigs: int = 32,
    ddof: int = 1,
    random_state: int = 0,
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
    aggregator = ArithmeticSPDMean(eps_spd=1e-8)
    regime_model = RegimeModel(n_regimes=n_regimes, random_state=random_state)
    return RegimeAwareSimilarityForecaster(
        embedder=embedder,
        target_object=target,
        aggregator=aggregator,
        lookback=lookback,
        horizon=horizon,
        regime_model=regime_model,
        tau=tau,
    )


def predict_at_raw_anchor(
    model: RegimeAwareSimilarityForecaster,
    past: np.ndarray,            # (L, N)
    raw_anchor: int,             # <-- ADD THIS
    k_neighbors: int = 50,
) -> np.ndarray:
    model._check_fitted()
    assert model.embeds_ is not None and model.targets_ is not None and model.knn_ is not None
    assert model.PI_ is not None and model.ALPHA_ is not None and model.A_ is not None
    assert model.anchor_rows_ is not None

    # Stage 1: embed current window
    e0 = model.embedder.embed(past).astype(float)  # (D,)

    # Stage 2: pi0 for current point (safe: uses trained GMM)
    pi0 = model.regime_model.predict_pi(e0[None, :])[0]  # (K,)

    # Stage 3: for *raw* anchor, safest is alpha = pi0 (no misaligned transition jump)
    alpha = pi0

    # Stage 4: retrieve neighbors from library (overshoot then filter)
    k = min(k_neighbors, model.embeds_.shape[0])
    k_search = min(max(5 * k, k), model.embeds_.shape[0])
    idx_all, dist_all = model.knn_.query(e=e0, k=k_search, exclude_index=None)

    # CRITICAL: only use neighbors whose labels are observable at time raw_anchor
    # neighbor anchor a must satisfy a + H <= raw_anchor  <=>  a <= raw_anchor - H
    H = model.horizon
    is_label_available = model.anchor_rows_[idx_all] <= (raw_anchor - H)
    idx = idx_all[is_label_available]
    dist = dist_all[is_label_available]

    if idx.size == 0:
        raise ValueError("No label-available neighbors (try smaller horizon or larger train history).")
    if idx.size > k:
        idx = idx[:k]
        dist = dist[:k]

    # Stage 4 kernel
    kappa = model._kappa_from_dist(dist)
    kappa = kappa / max(kappa.sum(), model.eps)

    # Stage 5 regime-aware neighbor weights
    PI_nbr = model.PI_[idx]  # (M, K)
    W = RegimeAwareWeights(eps=model.eps).compute(kappa=kappa, PI_neighbors=PI_nbr)  # (K, M)

    yk_list = []
    for kk in range(W.shape[0]):
        yk_list.append(model.aggregator.aggregate(model.targets_[idx], W[kk]))
    YK = np.stack(yk_list, axis=0)

    yhat = np.tensordot(alpha, YK, axes=(0, 0))
    return model.target_object.postprocess(yhat)

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
            model = build_model(lookback=lookback, horizon=horizon)

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
            Sigma_hat = predict_at_raw_anchor(model, past=past, raw_anchor=raw_anchor, k_neighbors=k_neighbors)
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

def summarize_comparison(results: pd.DataFrame):
    compare_keys = [
        "fro", "kl", "nll", "stein", "logeuc",
        "corr_offdiag_fro", "corr_spearman",
        "port_mse_logvar"
    ]

    rows = []

    for key in compare_keys:
        m_col = f"model_{key}"
        p_col = f"pers_{key}"
        r_col = f"roll_{key}"
        s_col = f"shrink_{key}"

        if m_col not in results.columns:
            continue

        row = {"metric": key}

        for label, col in [
            ("model", m_col),
            ("pers", p_col),
            ("roll", r_col),
            ("shrink", s_col),
        ]:
            if col in results.columns:
                row[label] = float(results[col].mean())
            else:
                row[label] = np.nan

        # Skill (ratio for losses, diff for correlation)
        if key == "corr_spearman":
            row["skill_vs_pers"] = row["model"] - row["pers"]
            row["win_rate_vs_pers"] = float(
                (results[m_col] > results[p_col]).mean()
            )
        else:
            row["skill_vs_pers"] = row["model"] / row["pers"]
            row["win_rate_vs_pers"] = float(
                (results[m_col] < results[p_col]).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)

def main():
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
        end_date="2020-5-31",
        k_neighbors=10,
        refit_every=5,
        long_only=False,
        verbose=True,
    )

    # ----------------------------
    # Summaries
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

    # keep only existing
    headline_cols = [c for c in headline_cols if c in results.columns]

    print("\n===== SUMMARY (headline metrics) =====")
    print(results[headline_cols].describe())

    # # Optional: quick “worst offenders” to debug outliers
    # for key in ["kl", "nll", "logeuc", "corr_offdiag_fro", "port_mse_logvar", "turnover_l1"]:
    #     if key in results.columns:
    #         print(f"\n===== WORST 5 by {key} =====")
    #         print(results[[key]].sort_values(key, ascending=False).head(5))

    # ----------------------------
    # Portfolio calibration summary (GMVP)
    # ----------------------------
    if "pred_var" in results.columns and "real_var" in results.columns:
        tmp = results[["pred_var", "real_var"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) > 5:
            ratio = tmp["real_var"] / np.maximum(tmp["pred_var"], 1e-12)
            print("\n===== PORTFOLIO (GMVP) CALIBRATION =====")
            print("n =", len(tmp))
            print("real/pred ratio: mean =", float(ratio.mean()),
                  "median =", float(ratio.median()),
                  "p90 =", float(ratio.quantile(0.90)),
                  "p99 =", float(ratio.quantile(0.99)))
            print("corr(pred_var, real_var) =", float(tmp["pred_var"].corr(tmp["real_var"])))

    # ----------------------------
    # Multi-portfolio risk errors summary
    # ----------------------------
    multi_cols = [c for c in ["port_mse_var", "port_mse_logvar", "port_mae_logvar"] if c in results.columns]
    if multi_cols:
        print("\n===== PORTFOLIO (MULTI) RISK ERRORS =====")
        print(results[multi_cols].describe())

    # ----------------------------
    # Portfolio stability summary
    # ----------------------------
    stab_cols = [c for c in ["turnover_l1", "w_hhi", "w_max_abs", "w_l1"] if c in results.columns]
    if stab_cols:
        print("\n===== PORTFOLIO STABILITY =====")
        print(results[stab_cols].describe())

    print("\n===== METHOD COMPARISON =====")
    comparison = summarize_comparison(results)
    comparison.to_csv("results/regime_similarity_comparison.csv", index=False)
    print(comparison)

    # ----------------------------
    # Save
    # ----------------------------
    os.makedirs("results", exist_ok=True)
    results.to_parquet("results/regime_similarity_backtest.parquet")
    results.to_csv("results/regime_similarity_backtest.csv")
    print("\nSaved:")
    print("  results/regime_similarity_backtest.parquet")
    print("  results/regime_similarity_backtest.csv")

if __name__ == "__main__":
    main()