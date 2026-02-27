from __future__ import annotations

import os
import numpy as np
import pandas as pd

from similarity_forecast.backtests import (
    frobenius_error,
    gaussian_kl_divergence,
    min_variance_weights,
    realized_portfolio_variance,
)

from similarity_forecast.embeddings import CorrEigenEmbedder, PCAWindowEmbedder
from similarity_forecast.target_objects import CovarianceTarget
from similarity_forecast.core import LogEuclideanSPDMean, validate_window, project_to_spd
from similarity_forecast.regimes import RegimeModel
from similarity_forecast.pipeline import RegimeAwareSimilarityForecaster
from similarity_forecast.regime_weighting import RegimeAwareWeights
import dataclasses
from numpy.typing import NDArray

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
    n_regimes: int = 4,
    tau: float = 20.0,
    k_eigs: int = 32,
    ddof: int = 1,
    random_state: int = 0,
) -> RegimeAwareSimilarityForecaster:
    # embedder = CorrEigenEmbedder(k=k_eigs)
    embedder = PCAWindowEmbedder(
        lookback=lookback,
        k=5,
        validate_window_fn=validate_window,  # reuse yours
        max_window_na_pct=0.0,
        min_stocks_with_data_pct=1.0,
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
    refit_every: int = 20,  # refit model every N anchors
    long_only: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward evaluation (NO leakage):
      - At raw anchor t: use past window (t-L+1..t) to embed/query neighbors
      - Neighbor library is built by fitting the model on train_df up to t (inclusive),
        which only includes samples whose targets lie fully inside train_df.
      - True target Σtrue is computed from (t+1..t+H) using the SAME TargetObject.
    """
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        raise ValueError("returns_df.index must be a DatetimeIndex for mapping anchor dates.")

    # Optional date slicing
    df = returns_df.sort_index()
    if start_date is not None:
        df = df.loc[pd.to_datetime(start_date) :]
    if end_date is not None:
        df = df.loc[: pd.to_datetime(end_date)]

    R = df.to_numpy(dtype=float)  # (T, N)
    dates = df.index
    T, N = R.shape

    # raw anchors must satisfy: have past L days and future H days
    burn_in = 252
    raw_anchor_start = lookback + horizon + burn_in
    raw_anchor_end = T - horizon - 1  # inclusive last anchor index
    min_samples_for_gmm = 50

    rows = []
    model: RegimeAwareSimilarityForecaster | None = None
    last_refit_raw_anchor: int | None = None

    for raw_anchor in range(raw_anchor_start, raw_anchor_end + 1):
        # refit schedule
        if (model is None) or (last_refit_raw_anchor is None) or (raw_anchor - last_refit_raw_anchor >= refit_every):
            train_df = df.iloc[: raw_anchor + 1]  # up to raw_anchor inclusive
            model = build_model(lookback=lookback, horizon=horizon)

            if verbose:
                print(f"[refit] raw_anchor={raw_anchor} date={dates[raw_anchor].date()} train_T={len(train_df)}")

            model.fit(train_df)

            # Guard: too few samples after NA validation / window construction
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

        # --- Build past window at CURRENT raw_anchor ---
        past = R[raw_anchor - lookback + 1 : raw_anchor + 1, :]  # (L, N)

        # --- Forecast Σhat using past-only query (no need for anchor_pos mapping) ---
        try:
            Sigma_hat = predict_at_raw_anchor(model, past=past, k_neighbors=k_neighbors)
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} prediction failed: {e}")
            continue
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        # print("Sigma_hat", np.trace(Sigma_hat)/Sigma_hat.shape[0])

        # --- True Σtrue from realized future window ---
        fut = R[raw_anchor + 1 : raw_anchor + horizon + 1, :]  # (H, N)
        try:
            Sigma_true = model.target_object.target(fut)        # (N, N)
        except Exception as e:
            if verbose:
                print(f"[skip] raw_anchor={raw_anchor} date={anchor_date.date()} true target failed: {e}")
            continue
        Sigma_true = np.asarray(Sigma_true, dtype=float)
        # print("Sigma_true", np.trace(Sigma_true)/Sigma_true.shape[0])

        # --- Metrics ---
        fro = frobenius_error(Sigma_hat, Sigma_true)
        Sigma_hat = project_to_spd((Sigma_hat + Sigma_hat.T) / 2.0, eps=1e-8)
        Sigma_true = project_to_spd((Sigma_true + Sigma_true.T) / 2.0, eps=1e-8)

        kl = gaussian_kl_divergence(S_true=Sigma_true, S_hat=Sigma_hat)

        # Min-var weights built on forecast Σhat, realized variance under Σtrue
        w = min_variance_weights(Sigma_hat, long_only=long_only)
        pred_var = realized_portfolio_variance(w, Sigma_hat)
        real_var = realized_portfolio_variance(w, Sigma_true)

        rows.append(
            {
                "date": anchor_date,
                "raw_anchor": raw_anchor,
                "fro": fro,
                "kl": kl,
                "pred_var": pred_var,
                "real_var": real_var,
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


if __name__ == "__main__":
    returns_df = pd.read_parquet("data/processed/returns_universe_100.parquet")
    returns_df.index = pd.to_datetime(returns_df.index)
    print(returns_df.shape)

    results = run_backtest(
        returns_df=returns_df,
        lookback=60,
        horizon=20,
        start_date="2018-01-01",
        end_date="2020-12-31", 
        k_neighbors=50,
        refit_every=20,
        long_only=False,
        verbose=True,
    )

    print("\n===== SUMMARY =====")
    print(results[["fro", "kl", "pred_var", "real_var"]].describe())

    os.makedirs("results", exist_ok=True)
    results.to_parquet("results/regime_similarity_backtest.parquet")
    results.to_csv("results/regime_similarity_backtest.csv")
    print("\nSaved:")
    print("  results/regime_similarity_backtest.parquet")
    print("  results/regime_similarity_backtest.csv")