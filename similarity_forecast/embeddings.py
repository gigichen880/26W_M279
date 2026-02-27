from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np
from numpy.typing import NDArray

from .core import cov_from_returns, corr_from_cov, cov_from_returns_filtered, project_to_spd


class WindowEmbedder(Protocol):
    """
    Maps a raw lookback window of returns to a fixed-D embedding vector.
    past_returns: shape [L, N]
    returns: shape [D]
    """
    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]: ...
    @property
    def dim(self) -> int: ...


@dataclass(frozen=True)
class CorrEigenEmbedder:
    """
    Compute correlation on the lookback window, then embed by top-k log eigenvalues.
    Handles NAs via pairwise-complete covariance; falls back to complete-case or zero if needed.
    """
    k: int
    ddof: int = 1
    eps: float = 1e-12
    min_periods_ratio: float = 0.5

    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        T, N = past_returns.shape
        min_periods = max(2, int(T * self.min_periods_ratio))
        try:
            Sigma, _ = cov_from_returns_filtered(
                past_returns,
                ddof=self.ddof,
                min_periods=min_periods,
                min_frac=0.8,
            )
            n_eff = Sigma.shape[0]
            if n_eff < self.k:
                warnings.warn(
                    f"CorrEigenEmbedder: only {n_eff} assets after filtering (< k={self.k}); using zero embedding",
                    UserWarning,
                    stacklevel=2,
                )
                return np.zeros(self.k, dtype=float)

            C = corr_from_cov(Sigma, eps=self.eps)
            w = np.linalg.eigvalsh(C)
            w = np.maximum(w, self.eps)[::-1][: self.k]
            return np.log(w)

        except Exception as e:
            warnings.warn(f"Embedding failed: {e}, using zeros", UserWarning, stacklevel=2)
            return np.zeros(self.k, dtype=float)
    @property
    def dim(self) -> int:
        return self.k


@dataclass(frozen=True)
class VolStatsEmbedder:
    """
    Simple feature engineering directly from window returns:
      - log vol quantiles
      - mean vol
    Output is fixed-d, independent of N.
    """
    ddof: int = 1
    eps: float = 1e-12
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        # vol per asset over window (NA-safe: nanstd/nanmean)
        v = np.nanstd(past_returns, axis=0, ddof=self.ddof)
        v = np.where(np.isnan(v), 0.0, v)
        v = np.sqrt(np.maximum(v * v, self.eps))
        lv = np.log(np.maximum(v, self.eps))

        feats = [float(np.nanmean(lv))]
        feats.extend(np.nanquantile(lv, self.quantiles).tolist())
        return np.array(feats, dtype=float)

    @property
    def dim(self) -> int:
        return 1 + len(self.quantiles)

class HybridStateEmbedder:
    """
    Deterministic market-state embedding from a lookback window of returns.

    Input
    -----
    X : array, shape (L, N)
        Past returns window (lookback length L, assets N). Assumed already aligned.

    Output
    ------
    z : array, shape (D,)
        State embedding vector.

    Notes
    -----
    Embedding is multi-view:
      (A) Factor strength: top-k log singular values + explained variance ratios (SVD on returns)
      (B) Vol regime: mean/std/quantiles of per-asset vol
      (C) Serial dependence: mean/std of lag-1 autocorr across assets
      (D) Minimal correlation structure: avg pairwise corr + top eig + eigengap (few scalars only)
      (E) Tail proxy: mean/std of absolute returns (or kurtosis if you want)
    """
    def __init__(self, k_factors=5, eps=1e-12):
        self.k = k_factors
        self.eps = eps

    def _safe_zscore(self, X):
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=1, keepdims=True)
        return (X - mu) / (sd + self.eps)

    def _lag1_autocorr(self, X):
        # per-asset lag-1 autocorr, shape (N,)
        x0 = X[:-1]
        x1 = X[1:]
        x0 = x0 - x0.mean(axis=0, keepdims=True)
        x1 = x1 - x1.mean(axis=0, keepdims=True)
        num = (x0 * x1).sum(axis=0)
        den = np.sqrt((x0**2).sum(axis=0) * (x1**2).sum(axis=0)) + self.eps
        return num / den

    def embed(self, X):
        X = np.asarray(X, dtype=float)
        L, N = X.shape

        # --------
        # (A) SVD factor strength on raw returns window (demeaned)
        # --------
        Xc = X - X.mean(axis=0, keepdims=True)
        # economy SVD: Xc = U S Vt
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.k, S.shape[0])

        sv = S[:k]
        log_sv = np.log(sv + self.eps)

        evr = (S**2) / ((S**2).sum() + self.eps)
        top_evr = evr[:k]
        evr_entropy = -np.sum(evr * np.log(evr + self.eps))  # concentration proxy

        # --------
        # (B) Vol regime features
        # --------
        vol = X.std(axis=0, ddof=1)  # per-asset vol
        vol_mean = vol.mean()
        vol_std = vol.std(ddof=1)
        vol_q25, vol_q50, vol_q75 = np.quantile(vol, [0.25, 0.50, 0.75])

        # --------
        # (C) Serial dependence
        # --------
        ac1 = self._lag1_autocorr(X)
        ac1_mean = np.nanmean(ac1)
        ac1_std = np.nanstd(ac1, ddof=1)

        # --------
        # (D) Minimal correlation structure (few scalars, not a spectrum embedding)
        # --------
        Xz = self._safe_zscore(X)
        C = (Xz.T @ Xz) / (L - 1 + self.eps)
        # avg off-diagonal corr
        mean_corr = (C.sum() - np.trace(C)) / (N * (N - 1) + self.eps)
        # top eig and eigengap
        w = np.linalg.eigvalsh(C)
        top_eig = w[-1]
        eig_gap = w[-1] - w[-2] if len(w) >= 2 else 0.0

        # --------
        # (E) Tail / jumpiness proxy 
        # --------
        absret = np.abs(X)
        abs_mean = absret.mean()
        abs_std = absret.std(ddof=1)

        z = np.concatenate([
            log_sv,                 # k
            top_evr,                # k
            np.array([
                evr_entropy,
                vol_mean, vol_std, vol_q25, vol_q50, vol_q75,
                ac1_mean, ac1_std,
                mean_corr, top_eig, eig_gap,
                abs_mean, abs_std
            ], dtype=float)
        ])
        return z
    
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAWindowEmbedder:
    """
    PCA embedder that matches your pipeline: embedder.fit(returns_df) then embedder.embed(past).

    It learns a global PCA on flattened training windows (L*N features), then embeds each
    window with:
      - global PCA coords (k)
      - within-window EVR (k)
      - log singular values (k)
      - time-score mean/std (k each)
    """
    def __init__(
        self,
        lookback: int,
        k: int = 5,
        center_by_asset: bool = True,
        use_scaler: bool = True,
        eps: float = 1e-12,
        # to mirror your validation behavior during fit:
        validate_window_fn=None,
        max_window_na_pct: float = 0.0,
        min_stocks_with_data_pct: float = 1.0,
        verbose_skip: bool = False,
    ):
        self.lookback = int(lookback)
        self.k = int(k)
        self.center_by_asset = bool(center_by_asset)
        self.use_scaler = bool(use_scaler)
        self.eps = float(eps)

        self.validate_window_fn = validate_window_fn
        self.max_window_na_pct = max_window_na_pct
        self.min_stocks_with_data_pct = min_stocks_with_data_pct
        self.verbose_skip = verbose_skip

        self.scaler_ = None
        self.pca_ = None
        self.L_ = None
        self.N_ = None

    def _prep_window(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.center_by_asset:
            X = X - X.mean(axis=0, keepdims=True)
        return X

    def _build_past_windows_only(self, R: np.ndarray):
        """
        Yield past windows only, shape (L,N), for anchors t in [L, T-1].
        (No horizon needed for PCA fit.)
        """
        T, N = R.shape
        L = self.lookback
        for anchor in range(L, T):
            past = R[anchor - L : anchor]
            yield anchor, past

    def fit(self, returns_df: pd.DataFrame) -> "PCAWindowEmbedder":
        R = returns_df.to_numpy(dtype=float)  # [T,N]
        self.L_, self.N_ = self.lookback, R.shape[1]

        X_list = []
        skipped = 0
        total = 0

        for anchor, past in self._build_past_windows_only(R):
            total += 1
            if self.validate_window_fn is not None:
                ok = self.validate_window_fn(
                    past,
                    max_na_pct=self.max_window_na_pct,
                    min_stocks_pct=self.min_stocks_with_data_pct,
                )
                if not ok:
                    skipped += 1
                    if self.verbose_skip:
                        print(f"[PCA fit] Skipping window at t={anchor}: insufficient data quality")
                    continue

            past = self._prep_window(past)
            X_list.append(past.reshape(-1))  # flatten L*N

        if not X_list:
            raise ValueError("PCAWindowEmbedder.fit: no valid windows to fit PCA on.")

        X_flat = np.stack(X_list, axis=0)  # [M, L*N]

        if self.use_scaler:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            X_flat = self.scaler_.fit_transform(X_flat)

        self.pca_ = PCA(n_components=max(self.k, 1), svd_solver="randomized")
        self.pca_.fit(X_flat)

        if skipped and self.verbose_skip:
            print(f"[PCA fit] Skipped {skipped} / {total} windows due to data quality.")

        return self

    def embed(self, past: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCAWindowEmbedder: not fitted. Call fit(returns_df) first.")

        past = np.asarray(past, dtype=float)
        L, N = past.shape
        if self.L_ is not None and (L != self.L_ or N != self.N_):
            raise ValueError(f"PCAWindowEmbedder.embed: expected ({self.L_},{self.N_}), got ({L},{N})")

        Xw = self._prep_window(past)

        # (1) global PCA coords on flattened window
        x_flat = Xw.reshape(1, -1)
        if self.scaler_ is not None:
            x_flat = self.scaler_.transform(x_flat)
        pc_coords = self.pca_.transform(x_flat).ravel()
        k = min(self.k, pc_coords.shape[0])

        # (2) within-window SVD features (factor strength + concentration + dynamics)
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        k2 = min(self.k, S.shape[0])

        ev = S**2
        evr = ev / (ev.sum() + self.eps)
        top_evr = evr[:k2]
        log_sv = np.log(S[:k2] + self.eps)

        scores = U[:, :k2] * S[:k2][None, :]
        score_mean = scores.mean(axis=0)
        score_std = scores.std(axis=0, ddof=1)

        # pad if k2 < k (rare unless L very small)
        def pad_to(x, K):
            if x.shape[0] == K:
                return x
            out = np.zeros(K, dtype=float)
            out[: x.shape[0]] = x
            return out

        pc_coords = pad_to(pc_coords[:k], self.k)
        top_evr = pad_to(top_evr, self.k)
        log_sv = pad_to(log_sv, self.k)
        score_mean = pad_to(score_mean, self.k)
        score_std = pad_to(score_std, self.k)

        return np.concatenate([pc_coords, top_evr, log_sv, score_mean, score_std], axis=0)