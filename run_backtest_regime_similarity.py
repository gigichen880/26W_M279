from __future__ import annotations

from datetime import date
import os
import numpy as np
import pandas as pd

from similarity_forecast.backtests import (
    frobenius_error,
    gaussian_kl_divergence,
    stein_loss,
    gaussian_nll_future_window,
    log_euclidean_distance,
    corr_offdiag_fro,
    corr_upper_spearman,
    eigen_log_mse,
    condition_number,
    make_eval_portfolios,
    multi_portfolio_risk_errors,
    min_variance_weights,
    realized_portfolio_variance,
    weight_concentration_stats,
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
    k_neighbors: int = 50,
) -> np.ndarray:
    """
    Predict at a *raw* anchor time using ONLY the lookback window `past`,
    querying neighbors from the fitted sample library (model.embeds_/targets_).

    This avoids requiring the raw anchor date to appear in model.anchor_dates_
    (it usually won't, because targets need future H days inside train_df).
    """
    model._check_fitted()
    assert model.embeds_ is not None and model.targets_ is not None and model.knn_ is not None
    assert model.PI_ is not None and model.ALPHA_ is not None and model.A_ is not None

    # Stage 1: embed current window
    e0 = model.embedder.embed(past).astype(float)  # (D,)

    # Stage 4: retrieve neighbors in embedding library
    idx, dist = model.knn_.query(e=e0, k=min(k_neighbors, model.embeds_.shape[0]), exclude_index=None)

    # Stage 4 kernel
    kappa = model._kappa_from_dist(dist)  # (M,)
    kappa = kappa / max(kappa.sum(), model.eps)

    # Stage 2 for current point: pi0 = p(regime | e0)
    pi0 = model.regime_model.predict_pi(e0[None, :])[0]  # (K,)

    # Stage 3 for current point: filtered alpha
    # Use the last filtered alpha from training as alpha_prev, propagate through A, then combine with pi0
    alpha_prev = model.ALPHA_[-1]            # (K,)
    alpha_prior = alpha_prev @ model.A_      # (K,)
    alpha = alpha_prior * pi0
    alpha = alpha / max(alpha.sum(), model.eps)

    # Stage 5: regime-aware neighbor weights
    PI_nbr = model.PI_[idx]  # (M, K)
    W = RegimeAwareWeights(eps=model.eps).compute(kappa=kappa, PI_neighbors=PI_nbr)  # (K, M)

    # regime-conditional forecasts then mix
    K = W.shape[0]
    yk_list = []
    for kk in range(K):
        w = W[kk]  # (M,)
        yk = model.aggregator.aggregate(model.targets_[idx], w)  # (N, N) for CovarianceTarget
        yk_list.append(yk)
    YK = np.stack(yk_list, axis=0)  # (K, N, N)

    yhat = np.tensordot(alpha, YK, axes=(0, 0))  # (N, N)
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

    # NEW: fixed portfolio set for evaluation (kept constant across time)
    W_eval = make_eval_portfolios(N=N, n_rand=20, seed=0, long_only=True)

    rows = []
    model: RegimeAwareSimilarityForecaster | None = None
    last_refit_raw_anchor: int | None = None

    # NEW: weight stability tracking (min-var)
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
            Sigma_hat = predict_at_raw_anchor(model, past=past, k_neighbors=k_neighbors)
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

        # --- Base metrics (your existing ones) ---
        fro = frobenius_error(Sigma_hat, Sigma_true)

        # Ensure SPD for the SPD-geometry metrics
        Sigma_hat = project_to_spd((Sigma_hat + Sigma_hat.T) / 2.0, eps=1e-8)
        Sigma_true = project_to_spd((Sigma_true + Sigma_true.T) / 2.0, eps=1e-8)

        kl = gaussian_kl_divergence(S_true=Sigma_true, S_hat=Sigma_hat)

        # --- NEW: Likelihood + SPD-geometry metrics ---
        try:
            nll = gaussian_nll_future_window(Sigma_hat=Sigma_hat, fut_returns=fut, eps=1e-10)
        except Exception:
            nll = np.nan

        try:
            stein = stein_loss(S_true=Sigma_true, S_hat=Sigma_hat, eps=1e-10)
        except Exception:
            stein = np.nan

        try:
            logeuc = log_euclidean_distance(S_true=Sigma_true, S_hat=Sigma_hat, eps=1e-10)
        except Exception:
            logeuc = np.nan

        # --- NEW: Correlation skill metrics ---
        try:
            corr_fro = corr_offdiag_fro(S_true=Sigma_true, S_hat=Sigma_hat)
        except Exception:
            corr_fro = np.nan

        try:
            corr_spear = corr_upper_spearman(S_true=Sigma_true, S_hat=Sigma_hat)
        except Exception:
            corr_spear = np.nan

        # --- NEW: Spectrum/conditioning diagnostics ---
        try:
            eig_logmse = eigen_log_mse(S_true=Sigma_true, S_hat=Sigma_hat, eps=1e-10)
        except Exception:
            eig_logmse = np.nan

        try:
            cond_hat = condition_number(Sigma_hat, eps=1e-10)
            cond_true = condition_number(Sigma_true, eps=1e-10)
            cond_ratio = float(cond_hat / max(cond_true, 1e-12))
        except Exception:
            cond_hat, cond_true, cond_ratio = np.nan, np.nan, np.nan

        # --- Portfolio probe (your existing + extra stability) ---
        w = min_variance_weights(Sigma_hat, long_only=long_only)
        pred_var = realized_portfolio_variance(w, Sigma_hat)
        real_var = realized_portfolio_variance(w, Sigma_true)

        # NEW: turnover, concentration
        if w_prev is None:
            turnover_l1 = np.nan
        else:
            turnover_l1 = float(np.sum(np.abs(w - w_prev)))
        w_prev = w.copy()

        w_stats = weight_concentration_stats(w)

        # --- Multi-portfolio risk errors (fixed portfolio set) ---
        port_err = multi_portfolio_risk_errors(Sigma_hat=Sigma_hat, Sigma_true=Sigma_true, W_eval=W_eval)

        rows.append(
            {
                "date": anchor_date,
                "raw_anchor": raw_anchor,

                "fro": fro,
                "kl": kl,
                "pred_var": pred_var,
                "real_var": real_var,

                # probabilistic + SPD geometry
                "nll": nll,
                "stein": stein,
                "logeuc": logeuc,

                # correlation skill
                "corr_offdiag_fro": corr_fro,
                "corr_spearman": corr_spear,

                # spectrum / conditioning
                "eig_log_mse": eig_logmse,
                "cond_hat": cond_hat,
                "cond_true": cond_true,
                "cond_ratio": cond_ratio,

                # portfolio stability diagnostics
                "turnover_l1": turnover_l1,
                **w_stats,

                # multi-portfolio errors
                **port_err,
            }
        )

    if not rows:
        raise RuntimeError(
            "Backtest produced 0 evaluation points. "
            "Likely causes: burn_in too large, NA validation too strict, or prediction/target exceptions. "
            "Try reducing burn_in, relaxing validate_window thresholds, or printing skips."
        )

    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out


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
        start_date="2017-01-01",
        end_date="2022-05-31",
        k_neighbors=10,
        refit_every=5,
        long_only=False,
        verbose=True,
    )

    # ----------------------------
    # Summaries
    # ----------------------------
    headline_cols = [
        "fro", "kl", "nll", "stein", "logeuc",
        "corr_offdiag_fro", "corr_spearman",
        "eig_log_mse", "cond_ratio",
        "pred_var", "real_var",
        "port_mse_var", "port_mse_logvar", "port_mae_logvar",
        "turnover_l1", "w_hhi", "w_max_abs",
    ]
    headline_cols = [c for c in headline_cols if c in results.columns]

    print("\n===== SUMMARY (headline metrics) =====")
    print(results[headline_cols].describe())

    # Optional: quick “worst offenders” to debug outliers
    for key in ["kl", "nll", "logeuc", "corr_offdiag_fro", "port_mse_logvar", "turnover_l1"]:
        if key in results.columns:
            print(f"\n===== WORST 5 by {key} =====")
            print(results[[key]].sort_values(key, ascending=False).head(5))

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