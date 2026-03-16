# similarity_forecast/embeddings.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np
from numpy.typing import NDArray

from .core import cov_from_returns, corr_from_cov, impute_returns_window, validate_window

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    k: int
    ddof: int = 1
    eps: float = 1e-12
    max_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8

    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        T, N = past_returns.shape
        if N < self.k:
            return np.zeros(self.k, dtype=float)

        Sigma = cov_from_returns(past_returns, ddof=self.ddof)   # now impute+SPD inside
        C = corr_from_cov(Sigma, eps=self.eps)

        w = np.linalg.eigvalsh(C)
        w = np.maximum(w, self.eps)[::-1][: self.k]
        return np.log(w)

    @property
    def dim(self) -> int:
        return self.k

@dataclass(frozen=True)
class VolStatsEmbedder:
    """
    Rich volatility-state embedding from a lookback window of returns.
    Designed for realized-vol forecasting: similarity in embedding ≈ similar past vol regime.

    Features (all derived from per-asset realized vol in the window):
      - Cross-sectional distribution: mean(log vol), std(log vol), quantiles of log vol
      - IQR of log vol (spread)
      - Vol trend: mean over assets of (log vol_2nd_half - log vol_1st_half) — rising vs falling vol
      - Vol concentration: HHI of vol (normalized); high when one asset dominates
    Output is fixed-d, independent of N.
    """
    ddof: int = 1
    eps: float = 1e-12
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    include_vol_trend: bool = True
    include_vol_concentration: bool = True

    def _per_asset_vol(self, R: NDArray[np.floating]) -> NDArray[np.floating]:
        """(T, N) -> (N,) realized vol per asset. R must be finite (caller imputes). ddof clamped so df >= 0."""
        T = R.shape[0]
        ddof_eff = min(self.ddof, max(0, T - 1))
        v = np.std(R, axis=0, ddof=ddof_eff)
        return np.maximum(np.abs(v), self.eps)

    def embed(self, past_returns: NDArray[np.floating]) -> NDArray[np.floating]:
        past_returns = np.asarray(past_returns, dtype=float)
        # Impute so every column has full count → no "Degrees of freedom <= 0" in nanstd
        past_returns = impute_returns_window(past_returns, fill_all_nan=0.0)
        T, N = past_returns.shape

        # Per-asset vol over full window -> log vol
        v = self._per_asset_vol(past_returns)
        lv = np.log(v)

        feats: list[float] = []
        # Cross-sectional distribution
        feats.append(float(np.nanmean(lv)))
        feats.append(float(np.nanstd(lv, ddof=1)) if N > 1 else 0.0)
        feats.extend(np.nanquantile(lv, self.quantiles).tolist())
        # IQR (spread of log-vol distribution)
        qs = np.nanquantile(lv, [0.25, 0.75])
        feats.append(float(qs[1] - qs[0]))

        # Vol trend: first half vs second half (only if window long enough)
        if self.include_vol_trend and T >= 4:
            mid = T // 2
            v1 = self._per_asset_vol(past_returns[:mid])
            v2 = self._per_asset_vol(past_returns[mid:])
            log_ratio = np.log(np.maximum(v2, self.eps)) - np.log(np.maximum(v1, self.eps))
            feats.append(float(np.nanmean(log_ratio)))
        else:
            feats.append(0.0)

        # Vol concentration (HHI): sum of (vol_i / sum(vol))^2
        if self.include_vol_concentration and N > 0:
            v_sum = float(np.sum(v)) + self.eps
            weights = v / v_sum
            hhi = float(np.sum(weights ** 2))
            feats.append(hhi)
        else:
            feats.append(0.0)

        return np.array(feats, dtype=float)

    @property
    def dim(self) -> int:
        n = 2 + len(self.quantiles) + 1  # mean, std, quantiles, IQR
        n += 1 if self.include_vol_trend else 0
        n += 1 if self.include_vol_concentration else 0
        return n

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
        X = impute_returns_window(X, fill_all_nan=0.0)

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
        validate_window_fn=None,
        max_window_na_pct: float = 0.3,
        min_stocks_with_data_pct: float = 0.8,
        verbose_skip: bool = False,
    ):
        self.lookback = int(lookback)
        self.k = int(k)
        self.center_by_asset = bool(center_by_asset)
        self.use_scaler = bool(use_scaler)
        self.eps = float(eps)

        self.validate_window_fn = validate_window_fn
        self.max_window_na_pct = float(max_window_na_pct)
        self.min_stocks_with_data_pct = float(min_stocks_with_data_pct)
        self.verbose_skip = bool(verbose_skip)

        self.scaler_ = None
        self.pca_ = None
        self.L_ = None
        self.N_ = None

    def _prep_window(self, X: np.ndarray) -> np.ndarray:
        """
        Optional per-asset centering. IMPORTANT: X is assumed finite already.
        """
        X = np.asarray(X, dtype=float)
        if self.center_by_asset:
            # center each asset over time
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

    def _validate(self, X: np.ndarray) -> bool:
        """
        Apply provided validate_window_fn if present.
        """
        if self.validate_window_fn is None:
            return True
        return bool(
            self.validate_window_fn(
                returns_window=X,
                max_na_pct=self.max_window_na_pct,
                min_stocks_pct=self.min_stocks_with_data_pct,
            )
        )

    def fit(self, returns_df: pd.DataFrame) -> "PCAWindowEmbedder":
        R = returns_df.to_numpy(dtype=float)  # [T,N]
        self.L_, self.N_ = self.lookback, R.shape[1]

        X_list = []
        skipped = 0
        total = 0

        for anchor, past in self._build_past_windows_only(R):
            total += 1

            # Skip low-quality windows so PCA isn't trained on junk / sparse windows
            if not self._validate(past):
                skipped += 1
                if self.verbose_skip and skipped <= 5:
                    na_pct = np.isnan(past).sum() / past.size
                    stocks_pct = (~np.isnan(past).all(axis=0)).mean()
                    times_pct = (~np.isnan(past).all(axis=1)).mean()
                    print(
                        f"[PCA fit skip] anchor={anchor} "
                        f"na_pct={na_pct:.3f} stocks_pct={stocks_pct:.3f} times_pct={times_pct:.3f}"
                    )
                continue

            past = impute_returns_window(past, fill_all_nan=0.0)
            past = self._prep_window(past)
            X_list.append(past.reshape(-1))

        if not X_list:
            raise ValueError(
                "PCAWindowEmbedder.fit: no valid windows to fit PCA on. "
                "Try relaxing max_window_na_pct / min_stocks_with_data_pct."
            )

        X_flat = np.stack(X_list, axis=0)  # [M, L*N]

        if self.use_scaler:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            X_flat = self.scaler_.fit_transform(X_flat)

        self.pca_ = PCA(n_components=max(self.k, 1), svd_solver="randomized")
        self.pca_.fit(X_flat)

        if self.verbose_skip:
            print(f"[PCA fit] used={len(X_list)} skipped={skipped} total={total}")

        return self

    def embed(self, past: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCAWindowEmbedder: not fitted. Call fit(returns_df) first.")

        past = impute_returns_window(past, fill_all_nan=0.0)
        L, N = past.shape
        if self.L_ is not None and (L != self.L_ or N != self.N_):
            raise ValueError(f"PCAWindowEmbedder.embed: expected ({self.L_},{self.N_}), got ({L},{N})")

        # Reject low-quality windows at inference time too (prevents empty-slice issues)
        if not self._validate(past):
            na_pct = np.isnan(past).sum() / past.size
            stocks_pct = (~np.isnan(past).all(axis=0)).mean()
            times_pct = (~np.isnan(past).all(axis=1)).mean()
            raise ValueError(
                f"PCAWindowEmbedder.embed: window failed validation "
                f"(na_pct={na_pct:.3f}, stocks_pct={stocks_pct:.3f}, times_pct={times_pct:.3f})"
            )

        Xw = impute_returns_window(past, fill_all_nan=0.0)
        Xw = self._prep_window(Xw)

        # (1) global PCA coords on flattened window
        x_flat = Xw.reshape(1, -1)
        if self.scaler_ is not None:
            x_flat = self.scaler_.transform(x_flat)
        pc_coords = self.pca_.transform(x_flat).ravel()
        k = min(self.k, pc_coords.shape[0])

        # (2) within-window SVD features
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        k2 = min(self.k, S.shape[0])

        ev = S**2
        evr = ev / (ev.sum() + self.eps)
        top_evr = evr[:k2]
        log_sv = np.log(S[:k2] + self.eps)

        scores = U[:, :k2] * S[:k2][None, :]
        score_mean = scores.mean(axis=0)
        score_std = scores.std(axis=0, ddof=1)

        def pad_to(x, K):
            x = np.asarray(x, dtype=float).ravel()
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
