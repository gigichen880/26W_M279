"""Compact state representations for the redesign.

Default PCA path is PCA-coordinates only (no unscaled SVD concat).
Legacy hybrid (D=5k) is retained only for the collapse decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from similarity_forecast.core import (
    corr_from_cov,
    cov_from_returns,
    impute_returns_window,
    validate_window,
)
from similarity_forecast.embeddings import EconomicStateEmbedder


EmbedName = Literal[
    "pca_only",
    "pca_svd_legacy",
    "pca_svd_standardized",
    "market_state",
    "spectral",
    "hybrid",
]


def _prep(past: NDArray[np.floating], center: bool = True) -> NDArray[np.floating]:
    X = impute_returns_window(np.asarray(past, dtype=float), fill_all_nan=0.0)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return X


@dataclass
class PCAOnlyEmbedder:
    """Standardized flattened window → PCA coords only (whitening optional)."""

    lookback: int
    k: int = 8
    whiten: bool = True
    center_by_asset: bool = True
    max_window_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8
    random_state: int = 0

    scaler_: Optional[StandardScaler] = None
    pca_: Optional[PCA] = None

    def fit(self, returns: NDArray[np.floating]) -> "PCAOnlyEmbedder":
        R = np.asarray(returns, dtype=float)
        L, rows = self.lookback, []
        for t in range(L, R.shape[0]):
            past = R[t - L : t]
            if not validate_window(past, self.max_window_na_pct, self.min_stocks_with_data_pct):
                continue
            rows.append(_prep(past, self.center_by_asset).reshape(-1))
        if not rows:
            raise ValueError("PCAOnlyEmbedder.fit: no valid windows")
        X = np.stack(rows, axis=0)
        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        Xs = self.scaler_.fit_transform(X)
        self.pca_ = PCA(
            n_components=min(self.k, Xs.shape[0], Xs.shape[1]),
            whiten=self.whiten,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        self.pca_.fit(Xs)
        return self

    def embed(self, past: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self.scaler_ is not None and self.pca_ is not None
        x = _prep(past, self.center_by_asset).reshape(1, -1)
        z = self.pca_.transform(self.scaler_.transform(x)).ravel()
        out = np.zeros(self.k, dtype=float)
        out[: z.size] = z
        return out

    @property
    def dim(self) -> int:
        return int(self.k)


@dataclass
class LegacyPCASVDEmbedder:
    """Old hybrid: PCA k + 4×k SVD features. Optionally jointly standardize the 5k vector."""

    lookback: int
    k: int = 48
    jointly_standardize: bool = False
    center_by_asset: bool = True
    max_window_na_pct: float = 0.3
    min_stocks_with_data_pct: float = 0.8
    random_state: int = 0
    eps: float = 1e-12

    scaler_flat_: Optional[StandardScaler] = None
    pca_: Optional[PCA] = None
    scaler_joint_: Optional[StandardScaler] = None

    def fit(self, returns: NDArray[np.floating]) -> "LegacyPCASVDEmbedder":
        R = np.asarray(returns, dtype=float)
        L, flats, feats = self.lookback, [], []
        for t in range(L, R.shape[0]):
            past = R[t - L : t]
            if not validate_window(past, self.max_window_na_pct, self.min_stocks_with_data_pct):
                continue
            Xw = _prep(past, self.center_by_asset)
            flats.append(Xw.reshape(-1))
        if not flats:
            raise ValueError("LegacyPCASVDEmbedder.fit: no valid windows")
        X = np.stack(flats, axis=0)
        self.scaler_flat_ = StandardScaler(with_mean=True, with_std=True)
        Xs = self.scaler_flat_.fit_transform(X)
        self.pca_ = PCA(
            n_components=min(self.k, Xs.shape[0], Xs.shape[1]),
            svd_solver="randomized",
            random_state=self.random_state,
        )
        self.pca_.fit(Xs)
        # Fit joint scaler on training embeddings if requested
        if self.jointly_standardize:
            Z = []
            for t in range(L, R.shape[0]):
                past = R[t - L : t]
                if not validate_window(past, self.max_window_na_pct, self.min_stocks_with_data_pct):
                    continue
                Z.append(self._raw_embed(past))
            self.scaler_joint_ = StandardScaler(with_mean=True, with_std=True)
            self.scaler_joint_.fit(np.stack(Z, axis=0))
        return self

    def _raw_embed(self, past: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self.scaler_flat_ is not None and self.pca_ is not None
        Xw = _prep(past, self.center_by_asset)
        pc = self.pca_.transform(self.scaler_flat_.transform(Xw.reshape(1, -1))).ravel()
        k = min(self.k, pc.size)
        U, S, _ = np.linalg.svd(Xw, full_matrices=False)
        k2 = min(self.k, S.size)
        ev = S**2
        evr = ev / (ev.sum() + self.eps)
        scores = U[:, :k2] * S[:k2][None, :]

        def pad(x, K):
            x = np.asarray(x, dtype=float).ravel()
            out = np.zeros(K, dtype=float)
            out[: min(K, x.size)] = x[:K]
            return out

        return np.concatenate(
            [
                pad(pc[:k], self.k),
                pad(evr[:k2], self.k),
                pad(np.log(S[:k2] + self.eps), self.k),
                pad(scores.mean(axis=0), self.k),
                pad(scores.std(axis=0, ddof=1), self.k),
            ]
        )

    def embed(self, past: NDArray[np.floating]) -> NDArray[np.floating]:
        z = self._raw_embed(past)
        if self.scaler_joint_ is not None:
            z = self.scaler_joint_.transform(z.reshape(1, -1)).ravel()
        return z

    @property
    def dim(self) -> int:
        return 5 * int(self.k)


@dataclass
class SpectralStateEmbedder:
    """Compact covariance/correlation spectral state (≈10-D)."""

    ddof: int = 1
    eps: float = 1e-12
    n_eigs: int = 5

    def embed(self, past: NDArray[np.floating]) -> NDArray[np.floating]:
        X = impute_returns_window(np.asarray(past, dtype=float), fill_all_nan=0.0)
        T, N = X.shape
        vols = np.std(X, axis=0, ddof=min(self.ddof, max(0, T - 1)))
        vols = np.maximum(vols, self.eps)
        Sigma = cov_from_returns(X, ddof=self.ddof)
        C = corr_from_cov(Sigma, eps=self.eps)
        off = C[np.triu_indices(N, k=1)]
        mean_corr = float(np.nanmean(off)) if off.size else 0.0
        corr_disp = float(np.nanstd(off)) if off.size > 1 else 0.0
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        top = ev[: self.n_eigs]
        if top.size < self.n_eigs:
            top = np.pad(top, (0, self.n_eigs - top.size))
        pc1_share = float(top[0] / (ev.sum() + self.eps))
        ratios = top[1:] / np.maximum(top[:-1], self.eps) if self.n_eigs > 1 else np.array([])
        feats = [
            float(np.log(np.mean(vols))),
            float(np.log(max(float(np.std(vols)), self.eps))),
            mean_corr,
            corr_disp,
            float(np.log(max(float(np.trace(Sigma)), self.eps))),
            pc1_share,
            *top.tolist(),
            *ratios.tolist(),
        ]
        return np.asarray(feats, dtype=float)

    @property
    def dim(self) -> int:
        # log_mean_vol, log_vov, mean_corr, corr_disp, log_tr, pc1, n_eigs, (n_eigs-1) ratios
        return 6 + self.n_eigs + max(0, self.n_eigs - 1)


@dataclass
class HybridEmbedder:
    """Few PCA coords + market-state scalars, jointly standardized after fit."""

    lookback: int
    pca_k: int = 5
    whiten: bool = True
    random_state: int = 0

    pca: Optional[PCAOnlyEmbedder] = None
    market: EconomicStateEmbedder = None  # type: ignore
    spectral: SpectralStateEmbedder = None  # type: ignore
    scaler_: Optional[StandardScaler] = None

    def __post_init__(self) -> None:
        self.market = EconomicStateEmbedder()
        self.spectral = SpectralStateEmbedder()
        self.pca = PCAOnlyEmbedder(
            lookback=self.lookback, k=self.pca_k, whiten=self.whiten, random_state=self.random_state
        )

    def fit(self, returns: NDArray[np.floating]) -> "HybridEmbedder":
        assert self.pca is not None
        self.pca.fit(returns)
        R = np.asarray(returns, dtype=float)
        L, Z = self.lookback, []
        for t in range(L, R.shape[0]):
            past = R[t - L : t]
            if not validate_window(past):
                continue
            Z.append(self._raw(past))
        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        self.scaler_.fit(np.stack(Z, axis=0))
        return self

    def _raw(self, past: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self.pca is not None
        return np.concatenate([self.pca.embed(past), self.market.embed(past), self.spectral.embed(past)])

    def embed(self, past: NDArray[np.floating]) -> NDArray[np.floating]:
        z = self._raw(past)
        if self.scaler_ is not None:
            z = self.scaler_.transform(z.reshape(1, -1)).ravel()
        return z

    @property
    def dim(self) -> int:
        assert self.pca is not None
        return self.pca.dim + self.market.dim + self.spectral.dim


@dataclass
class MarketStateEmbedder:
    """Thin wrapper + optional StandardScaler fit on daily states."""

    standardize: bool = True
    inner: EconomicStateEmbedder = None  # type: ignore
    scaler_: Optional[StandardScaler] = None

    def __post_init__(self) -> None:
        self.inner = EconomicStateEmbedder()

    def fit(self, returns: NDArray[np.floating], lookback: int = 50) -> "MarketStateEmbedder":
        if not self.standardize:
            return self
        R = np.asarray(returns, dtype=float)
        Z = []
        for t in range(lookback, R.shape[0]):
            past = R[t - lookback : t]
            if validate_window(past):
                Z.append(self.inner.embed(past))
        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        self.scaler_.fit(np.stack(Z, axis=0))
        return self

    def embed(self, past: NDArray[np.floating]) -> NDArray[np.floating]:
        z = self.inner.embed(past)
        if self.scaler_ is not None:
            z = self.scaler_.transform(z.reshape(1, -1)).ravel()
        return z

    @property
    def dim(self) -> int:
        return 6


def build_embedder(name: str, *, lookback: int, pca_k: int = 8, **kwargs):
    name = str(name).lower()
    if name == "pca_only":
        return PCAOnlyEmbedder(lookback=lookback, k=pca_k, **{k: v for k, v in kwargs.items() if k in PCAOnlyEmbedder.__dataclass_fields__})
    if name == "pca_svd_legacy":
        return LegacyPCASVDEmbedder(lookback=lookback, k=pca_k, jointly_standardize=False)
    if name == "pca_svd_standardized":
        return LegacyPCASVDEmbedder(lookback=lookback, k=pca_k, jointly_standardize=True)
    if name == "market_state":
        return MarketStateEmbedder(standardize=kwargs.get("standardize", True))
    if name == "spectral":
        return SpectralStateEmbedder()
    if name == "hybrid":
        return HybridEmbedder(lookback=lookback, pca_k=pca_k)
    raise ValueError(f"Unknown embedder: {name}")
