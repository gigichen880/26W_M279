# similarity_forecast/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .embeddings import WindowEmbedder
from .target_objects import TargetObject
from .core import ExactKNN, Weighting, Aggregator, validate_window
from .regimes import RegimeModel
from .regime_weighting import RegimeAwareWeights


@dataclass
class RegimeAwareSimilarityForecaster:
    """
    Updated pipeline implementing 6-stage regime-aware similarity forecasting.
    (For now: implements Stages 1–5, skipping Stage 6 transition-ahead mixing.)

    Stages:
      1) Build window embeddings z_t = f(X_t)
      2) Fit GMM on embeddings -> soft regime membership PI[t,k]
      3) Estimate transition A from hard assignments; filter alpha_t
      4) Retrieve M nearest neighbors via similarity kernel kappa_i = exp(-||z0-zi||/tau)
      5) Regime-aware KNN:
            w_i^(k) ∝ kappa_i * PI[ti,k]
            yhat^(k) = Σ_i w_i^(k) y_i
            yhat      = Σ_k alpha_t(k) * yhat^(k)

    Data alignment:
      - We build samples for each anchor index i corresponding to the END of the past window.
      - embeds_[i] and targets_[i] correspond to that anchor.
      - All similarity search is done over these anchor-indexed samples.
    """
    embedder: WindowEmbedder
    target_object: TargetObject
    aggregator: Aggregator

    lookback: int
    horizon: int

    # regime model
    regime_model: RegimeModel

    # similarity kernel temperature
    tau: float = 1.0
    eps: float = 1e-12

    # NA handling: skip windows that fail validation
    max_window_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8
    verbose_skip: bool = False

    # fitted
    embeds_: Optional[NDArray[np.floating]] = None      # [T0, D]
    targets_: Optional[NDArray[np.floating]] = None    # [T0, ...]
    knn_: Optional[ExactKNN] = None

    PI_: Optional[NDArray[np.floating]] = None         # [T0, K]
    ALPHA_: Optional[NDArray[np.floating]] = None      # [T0, K]
    A_: Optional[NDArray[np.floating]] = None          # [K, K]

    # for debugging / alignment
    anchor_dates_: Optional[pd.DatetimeIndex] = None

    def _build_windows(self, R: NDArray[np.floating]) -> List[Tuple[int, slice, slice]]:
        """
        Returns a list of (anchor_idx, past_slice, fut_slice)
        anchor_idx is the index in R corresponding to the last row of past_slice.
        """
        T = R.shape[0]
        L, H = self.lookback, self.horizon
        out: List[Tuple[int, slice, slice]] = []
        for anchor in range(L - 1, T - H):
            past = slice(anchor - L + 1, anchor + 1)
            fut = slice(anchor + 1, anchor + H + 1)
            out.append((anchor, past, fut))
        return out

    def _check_fitted(self) -> None:
        if (
            self.embeds_ is None
            or self.targets_ is None
            or self.knn_ is None
            or self.PI_ is None
            or self.ALPHA_ is None
            or self.A_ is None
        ):
            raise RuntimeError("Call fit() first.")

    def fit(self, returns_df: pd.DataFrame) -> "RegimeAwareSimilarityForecaster":
        """
        Fit Stages 1–3 on the full sample set (you can later wrap this in walk-forward).
        """
        R = returns_df.to_numpy(dtype=float)  # [T, N]
        windows = self._build_windows(R)
        if not windows:
            raise ValueError("Not enough rows for lookback/horizon.")

        # fit embedder if it supports fit()
        if hasattr(self.embedder, "fit"):
            try:
                self.embedder.fit(returns_df)
            except TypeError:
                pass

        embeds, targets, anchor_pos = [], [], []
        skipped = 0
        
        for anchor, past_sl, fut_sl in windows:
            past = R[past_sl]
            fut = R[fut_sl]

            if not validate_window(
                past,
                max_na_pct=self.max_window_na_pct,
                min_stocks_pct=self.min_stocks_with_data_pct,
            ):
                skipped += 1
                if self.verbose_skip:
                    print(f"Skipping window at t={anchor}: insufficient data quality")
                continue

            e = self.embedder.embed(past)          # [D]
            y = self.target_object.target(fut)     # [...]

            embeds.append(e)
            targets.append(y)
            anchor_pos.append(anchor)

        if not embeds:
            raise ValueError(
                "No valid windows after NA validation. Try relaxing max_window_na_pct "
                "or min_stocks_with_data_pct."
            )
        if skipped and self.verbose_skip:
            print(f"Skipped {skipped} / {len(windows)} windows due to data quality.")

        self.embeds_ = np.stack(embeds, axis=0).astype(float)   # [T0, D]
        self.targets_ = np.stack(targets, axis=0).astype(float) # [T0, ...]
        self.knn_ = ExactKNN(self.embeds_)

        # anchor dates for debugging
        if isinstance(returns_df.index, pd.DatetimeIndex):
            self.anchor_dates_ = returns_df.index[np.array(anchor_pos, dtype=int)]
        else:
            self.anchor_dates_ = None

        # Stage 2: GMM -> PI
        self.regime_model.fit_gmm(self.embeds_)
        self.PI_ = self.regime_model.predict_pi(self.embeds_)  # [T0, K]

        # Stage 3: estimate transition + filter alpha
        self.A_ = self.regime_model.estimate_transition(self.PI_)
        self.ALPHA_ = self.regime_model.filter_alpha(self.PI_, A=self.A_)

        return self

    def _kappa_from_dist(self, dist: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Stage 4 kernel: kappa_i = exp(-||z0-zi|| / tau)
        (note: your previous RBFWeighting used exp(-d^2/tau); this uses exp(-d/tau) per spec)
        """
        tau = max(self.tau, self.eps)
        kappa = np.exp(-dist / tau)
        return kappa.astype(float)

    def predict_at_anchor(
        self,
        anchor_pos: int,
        k: int = 50,
        exclude_self: bool = True,
        return_debug: bool = False,
    ):
        """
        anchor_pos is index in the *sample space* (0..T0-1), not the raw returns row.

        Returns:
          yhat: final mixed forecast  (stage 5)
          optionally debug dict
        """
        self._check_fitted()
        assert self.embeds_ is not None and self.targets_ is not None and self.knn_ is not None
        assert self.PI_ is not None and self.ALPHA_ is not None

        e0 = self.embeds_[anchor_pos]
        exclude = anchor_pos if exclude_self else None

        idx, dist = self.knn_.query(e=e0, k=min(k, self.embeds_.shape[0]), exclude_index=exclude)

        # Stage 4: similarity kernel kappa
        kappa = self._kappa_from_dist(dist)  # [M]
        # normalize kappa globally for stability (optional)
        kappa = kappa / max(kappa.sum(), self.eps)

        # Stage 5: regime-aware neighbor weights
        PI_nbr = self.PI_[idx]  # [M, K]
        W = RegimeAwareWeights(eps=self.eps).compute(kappa=kappa, PI_neighbors=PI_nbr)  # [K, M]

        K = W.shape[0]
        # regime-conditional forecasts yhat^(k)
        yk_list = []
        for kk in range(K):
            w = W[kk]  # [M]
            yk = self.aggregator.aggregate(self.targets_[idx], w)  # [...]
            yk_list.append(yk)
        YK = np.stack(yk_list, axis=0)  # [K, ...]

        alpha = self.ALPHA_[anchor_pos]  # [K]
        # final mix over regimes
        yhat = np.tensordot(alpha, YK, axes=(0, 0))  # [...]

        yhat = self.target_object.postprocess(yhat)

        if not return_debug:
            return yhat

        dbg: Dict[str, Any] = {
            "neighbor_idx": idx,
            "neighbor_dist": dist,
            "kappa": kappa,
            "PI_neighbors": PI_nbr,
            "alpha": alpha,
            "YK": YK,
        }
        if self.anchor_dates_ is not None:
            dbg["anchor_date"] = self.anchor_dates_[anchor_pos]
        return yhat, dbg