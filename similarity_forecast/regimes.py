# similarity_forecast/regimes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


def _row_normalize(A: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    s = A.sum(axis=1, keepdims=True)
    return A / np.maximum(s, eps)


@dataclass
class RegimeModel:
    """
    Stage 2 + Stage 3 (V1 smoothing) for the 6-stage regime-aware similarity pipeline.

    Stage 2:
      - Fit a GMM on embeddings Z to get soft membership pi_t(k)

    Stage 3:
      - Estimate transition matrix A from hard assignments (argmax pi)
      - Compute filtered posterior alpha_t via:
            alpha_t ∝ (alpha_{t-1} A) ⊙ pi_t
        normalized.

    Notes:
      - This module is deliberately sklearn-optional.
      - If sklearn is available, we use sklearn.mixture.GaussianMixture.
      - Otherwise, you can pass in precomputed PI externally.
    """
    n_regimes: int
    covariance_type: str = "full"
    reg_covar: float = 1e-6
    random_state: int = 0
    max_iter: int = 300
    tol: float = 1e-3

    # controls GMM initialization (important for macOS threadpoolctl crash)
    # sklearn GaussianMixture defaults to init_params="kmeans" which triggers KMeans
    # and can crash in some macOS/Python builds via threadpoolctl.
    gmm_init_params: str = "kmeans"   # "kmeans" | "random" | "random_from_data"
    gmm_n_init: int = 1              # increase when using random init for stability

    trans_smooth: float = 1.0  # Laplace smoothing for A counts
    eps: float = 1e-12

    gmm_: Optional[object] = None
    A_: Optional[NDArray[np.floating]] = None

    def fit_gmm(self, Z: NDArray[np.floating]) -> "RegimeModel":
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore
        except Exception as e:
            raise ImportError(
                "scikit-learn is required for RegimeModel.fit_gmm(). "
                "Install with: pip install scikit-learn"
            ) from e

        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2D array [T0, D], got shape={Z.shape}")

        init_params = str(self.gmm_init_params).lower()
        if init_params not in {"kmeans", "random", "random_from_data"}:
            raise ValueError("gmm_init_params must be one of {'kmeans','random','random_from_data'}")

        def _make(init: str) -> "GaussianMixture":
            return GaussianMixture(
                n_components=int(self.n_regimes),
                covariance_type=str(self.covariance_type),
                reg_covar=float(self.reg_covar),
                random_state=int(self.random_state),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                init_params=str(init),
                n_init=int(self.gmm_n_init),
            )

        gmm = _make(init_params)

        try:
            gmm.fit(Z)
        except AttributeError as e:
            # Workaround for macOS/Python builds where KMeans->threadpoolctl crashes
            # during GMM initialization when init_params="kmeans".
            if init_params == "kmeans":
                print(f"[warn] GMM init via kmeans failed ({e}); retrying with init_params='random'")
                gmm = _make("random")
                gmm.fit(Z)
            else:
                raise

        self.gmm_ = gmm
        return self

    def predict_pi(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.gmm_ is None:
            raise RuntimeError("Call fit_gmm() first (or supply PI externally).")
        Z = np.asarray(Z, dtype=float)
        PI = self.gmm_.predict_proba(Z)
        # ensure numeric stability
        PI = np.maximum(PI, self.eps)
        PI = PI / np.maximum(PI.sum(axis=1, keepdims=True), self.eps)
        return PI.astype(float)

    def estimate_transition(
        self,
        PI: NDArray[np.floating],
        mode: str = "hard",  # {"hard","soft"}
        trans_smooth: Optional[float] = None,
    ) -> NDArray[np.floating]:
        """
        Estimate Markov transition matrix A.

        Parameters
        ----------
        PI : array, shape [T, K]
            Soft regime membership probabilities (rows sum to 1).
        mode : {"hard","soft"}
            - "hard": use hard labels s_t = argmax_k PI[t,k] and count transitions.
            - "soft": use expected transition counts sum_t PI[t-1,i] * PI[t,j].
        trans_smooth : float, optional
            Laplace smoothing added to all transition counts. Defaults to self.trans_smooth.

        Returns
        -------
        A : array, shape [K, K]
            Row-stochastic transition matrix.
        """
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
        """
        Compute filtered posterior alpha_t over time:
            alpha_t ∝ (alpha_{t-1} A) ⊙ PI[t]
        Returns ALPHA: [T, K]
        """
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