# similarity_forecast/regimes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .regime_clustering import RegimeClusterer, make_regime_clusterer


def _row_normalize(A: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    s = A.sum(axis=1, keepdims=True)
    return A / np.maximum(s, eps)


@dataclass
class RegimeModel:
    """
    Stage 2 + Stage 3 for the regime-aware similarity pipeline.

    Stage 2:
      - Fit a pluggable RegimeClusterer on embeddings Z -> soft membership pi_t(k)

    Stage 3:
      - Estimate transition matrix A from PI (hard or soft counts)
      - Filtered posterior alpha_t via alpha_t ∝ (alpha_{t-1} A) ⊙ pi_t

    Use ``regime_clustering`` in YAML (see make_regime_clusterer) or pass ``clusterer=``.
    If ``clusterer`` is None, defaults to GMM with the legacy fields below.
    """
    n_regimes: int
    trans_smooth: float = 1.0
    eps: float = 1e-12
    random_state: int = 0

    clusterer: Optional[RegimeClusterer] = None

    # Legacy GMM parameters (used only when clusterer is None)
    covariance_type: str = "diag"
    reg_covar: float = 1e-3
    max_iter: int = 300
    tol: float = 1e-3
    gmm_init_params: str = "kmeans"
    gmm_n_init: int = 1

    A_: Optional[NDArray[np.floating]] = None

    def __post_init__(self) -> None:
        if self.clusterer is None:
            self.clusterer = make_regime_clusterer(
                "gmm",
                int(self.n_regimes),
                int(self.random_state),
                {
                    "covariance_type": self.covariance_type,
                    "reg_covar": self.reg_covar,
                    "max_iter": self.max_iter,
                    "tol": self.tol,
                    "gmm_init_params": self.gmm_init_params,
                    "gmm_n_init": self.gmm_n_init,
                    "eps": self.eps,
                },
            )

    def fit_gmm(self, Z: NDArray[np.floating]) -> "RegimeModel":
        """Fit regime assignment on embeddings Z (name kept for backward compatibility)."""
        assert self.clusterer is not None
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2D array [T0, D], got shape={Z.shape}")
        self.clusterer.fit(Z)
        return self

    def predict_pi(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self.clusterer is not None
        Z = np.asarray(Z, dtype=float)
        return self.clusterer.predict_proba(Z)

    def estimate_transition(
        self,
        PI: NDArray[np.floating],
        mode: str = "hard",
        trans_smooth: Optional[float] = None,
    ) -> NDArray[np.floating]:
        if mode not in {"hard", "soft"}:
            raise ValueError(f"mode must be one of {{'hard','soft'}}, got {mode!r}")

        PI = np.asarray(PI, dtype=float)
        T, K = PI.shape
        if K != self.n_regimes:
            raise ValueError(f"PI has K={K}, but RegimeModel.n_regimes={self.n_regimes}")
        if T < 2:
            A = np.eye(K, dtype=float)
            self.A_ = A
            return A

        lam = float(self.trans_smooth if trans_smooth is None else trans_smooth)
        counts = np.full((K, K), lam, dtype=float)

        if mode == "hard":
            s = np.argmax(PI, axis=1).astype(int)
            for t in range(1, T):
                counts[s[t - 1], s[t]] += 1.0
        else:
            for t in range(1, T):
                counts += np.outer(PI[t - 1], PI[t])

        A = _row_normalize(counts, eps=self.eps)
        self.A_ = A
        return A

    def filter_alpha(
        self,
        PI: NDArray[np.floating],
        A: Optional[NDArray[np.floating]] = None,
        alpha0: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.floating]:
        if A is None:
            if self.A_ is None:
                raise RuntimeError("Need transition matrix A. Call estimate_transition() first.")
            A = self.A_

        PI = np.asarray(PI, dtype=float)
        T, K = PI.shape
        if K != self.n_regimes:
            raise ValueError(f"PI has K={K}, but RegimeModel.n_regimes={self.n_regimes}")

        ALPHA = np.zeros((T, K), dtype=float)

        if alpha0 is None:
            a = PI[0].copy()
        else:
            a = np.maximum(alpha0, self.eps)
            a = a / np.maximum(a.sum(), self.eps)

        ALPHA[0] = a

        for t in range(1, T):
            pred = a @ A
            post = pred * PI[t]
            s = post.sum()
            if s <= self.eps:
                a = PI[t].copy()
            else:
                a = post / s
            ALPHA[t] = a

        return ALPHA
