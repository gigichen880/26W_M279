# similarity_forecast/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .embeddings import WindowEmbedder
from .target_objects import TargetObject
from .core import ExactKNN, Aggregator, validate_window
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
    anchor_rows_: Optional[np.ndarray] = None  # shape [T0], raw row index for each sample

    # training sample de-duplication:
    # use only every sample_stride-th anchor when building (embed,target) samples
    sample_stride: int = 1

    # hard / soft transition in prediction
    transition_estimator: str = "hard"  # {"hard","soft"}
    trans_smooth: float = 1.0           # Laplace smoothing λ

    def _build_windows(self, R: NDArray[np.floating]) -> List[Tuple[int, slice, slice]]:
        """
        Returns a list of (anchor_idx, past_slice, fut_slice)
        anchor_idx is the index in R corresponding to the last row of past_slice.
        """
        T = R.shape[0]
        L, H = self.lookback, self.horizon
        out: List[Tuple[int, slice, slice]] = []

        stride = max(int(self.sample_stride), 1)

        for anchor in range(L - 1, T - H, stride): # step by stride
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
        self.anchor_rows_ = np.asarray(anchor_pos, dtype=np.int64)

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
        self.A_ = self.regime_model.estimate_transition(
            self.PI_,
            mode=self.transition_estimator,
            trans_smooth=self.trans_smooth,
        )
        self.ALPHA_ = self.regime_model.filter_alpha(self.PI_, A=self.A_)

        return self
    
    def _normalize_prob(self, v: NDArray[np.floating]) -> NDArray[np.floating]:
        s = float(np.sum(v))
        if (not np.isfinite(s)) or s <= self.eps:
            # deterministic fallback: uniform
            out = np.ones_like(v, dtype=float)
            out /= max(out.size, 1)
            return out
        return (v / s).astype(float)

    def _filter_update(
        self,
        alpha_prev: NDArray[np.floating],
        A: NDArray[np.floating],
        pi_t: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        One-step Bayesian filtering update:
            prior = alpha_prev @ A
            alpha_t ∝ prior ⊙ pi_t
        """
        prior = alpha_prev @ A
        post = prior * pi_t
        return self._normalize_prob(post)

    def _latest_alpha_before_raw_anchor(self, raw_anchor: int) -> Optional[NDArray[np.floating]]:
        """
        Return the most recent filtered ALPHA_ available strictly before raw_anchor
        in the raw timeline, using anchor_rows_ alignment.
        """
        assert self.anchor_rows_ is not None and self.ALPHA_ is not None
        eligible = np.where(self.anchor_rows_ <= (raw_anchor - 1))[0]
        if eligible.size == 0:
            return None
        j = int(eligible[-1])
        return self.ALPHA_[j]

    # ----------------------------
    # kernel
    # ----------------------------
    def _kappa_from_dist(self, dist: NDArray[np.floating]) -> NDArray[np.floating]:
        tau = max(self.tau, self.eps)
        return np.exp(-dist / tau).astype(float)

    # ----------------------------
    # prediction (Stages 1–5)
    # ----------------------------
    def predict_at_raw_anchor(
        self,
        past: NDArray[np.floating],     # (L, N)
        raw_anchor: int,
        k_neighbors: int = 50,
        use_filter: bool = True,
        alpha_fallback: str = "pi",     # {"pi","uniform"}
         neighbor_gap: int = 5,         # require neighbor anchor <= raw_anchor - (H + gap)
    ) -> NDArray[np.floating]:
        """
        Stage 1: embed current window -> e0
        Stage 2: regime membership from GMM -> pi0
        Stage 3: alpha choice:
            - if use_filter: alpha := filter_update(alpha_prev, A_, pi0)
            - else:         alpha := pi0

        NOTE:
        - hard vs soft is controlled ONLY by self.transition_estimator during fit()
            when estimating A_. Prediction-time filtering always uses stored A_.
        """
        self._check_fitted()
        assert self.embeds_ is not None and self.targets_ is not None and self.knn_ is not None
        assert self.PI_ is not None and self.ALPHA_ is not None and self.A_ is not None
        assert self.anchor_rows_ is not None

        if alpha_fallback not in {"pi", "uniform"}:
            raise ValueError(f"alpha_fallback must be one of {{'pi','uniform'}}, got {alpha_fallback!r}")

        # ---- Stage 1: embed ----
        e0 = self.embedder.embed(past).astype(float)  # (D,)

        # ---- Stage 2: pi0 ----
        pi0 = self.regime_model.predict_pi(e0[None, :])[0].astype(float)  # (K,)
        pi0 = self._normalize_prob(pi0)

        # ---- Stage 3: alpha ----
        if not use_filter:
            alpha = pi0
        else:
            alpha_prev = self._latest_alpha_before_raw_anchor(raw_anchor)
            if alpha_prev is None:
                alpha = pi0 if alpha_fallback == "pi" else (np.ones_like(pi0) / max(pi0.size, 1))
            else:
                alpha_prev = self._normalize_prob(alpha_prev.astype(float))
                alpha = self._filter_update(alpha_prev=alpha_prev, A=self.A_, pi_t=pi0)

                if not np.all(np.isfinite(alpha)) or float(np.sum(alpha)) <= self.eps:
                    alpha = pi0 if alpha_fallback == "pi" else (np.ones_like(pi0) / max(pi0.size, 1))

        # ---- Stage 4: retrieve neighbors (overshoot then filter) ----
        k = min(k_neighbors, self.embeds_.shape[0])
        k_search = min(max(5 * k, k), self.embeds_.shape[0])

        idx_all, dist_all = self.knn_.query(e=e0, k=k_search, exclude_index=None)

        # label-availability + anti-dup gap:
        # neighbor anchor a must satisfy a <= raw_anchor - H - gap
        H = self.horizon
        gap = max(int(neighbor_gap), 0)
        cutoff = raw_anchor - H - gap

        is_eligible = self.anchor_rows_[idx_all] <= cutoff
        idx = idx_all[is_eligible]
        dist = dist_all[is_eligible]

        if idx.size == 0:
            raise ValueError("No label-available neighbors (try smaller horizon or larger train history).")
        if idx.size > k:
            idx = idx[:k]
            dist = dist[:k]

        kappa = self._kappa_from_dist(dist)
        kappa = kappa / max(float(kappa.sum()), self.eps)

        # ---- Stage 5: regime-aware neighbor weights ----
        PI_nbr = self.PI_[idx]  # (M, K)
        W = RegimeAwareWeights(eps=self.eps).compute(kappa=kappa, PI_neighbors=PI_nbr)  # (K, M)

        yk_list = []
        for kk in range(W.shape[0]):
            yk_list.append(self.aggregator.aggregate(self.targets_[idx], W[kk]))
        YK = np.stack(yk_list, axis=0)  # (K, ...)

        yhat = np.tensordot(alpha, YK, axes=(0, 0))
        return self.target_object.postprocess(yhat)