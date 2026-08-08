"""Regime models for Stage C. Probabilistic models vs hard-clustering controls.

Do not label GMM+post-hoc A as an HMM. Use a true HMM library when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


def _normalize(P: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    P = np.asarray(P, dtype=float)
    P = np.clip(P, eps, None)
    return P / P.sum(axis=-1, keepdims=True)


@dataclass
class FCMRegime:
    n_states: int = 4
    fuzziness: float = 2.0
    random_state: int = 0
    _cl: object = None

    def fit(self, Z: NDArray[np.floating]) -> "FCMRegime":
        from similarity_forecast.regime_clustering import FuzzyCMeansRegimeClusterer

        self._cl = FuzzyCMeansRegimeClusterer(
            n_regimes=self.n_states, fuzziness=self.fuzziness, random_state=self.random_state
        )
        self._cl.fit(Z)
        return self

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self._cl is not None
        return self._cl.predict_proba(Z)


@dataclass
class GMMRegime:
    n_states: int = 3
    covariance_type: Literal["diag", "tied", "full"] = "diag"
    random_state: int = 0
    _gmm: object = None

    def fit(self, Z: NDArray[np.floating]) -> "GMMRegime":
        from sklearn.mixture import GaussianMixture

        self._gmm = GaussianMixture(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5,
        )
        self._gmm.fit(Z)
        return self

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self._gmm is not None
        return _normalize(self._gmm.predict_proba(Z))


@dataclass
class HardKMeansRegime:
    """Deterministic control — max p = 1 by construction; not subject to soft-acceptance rules."""

    n_states: int = 4
    random_state: int = 0
    _km: object = None

    def fit(self, Z: NDArray[np.floating]) -> "HardKMeansRegime":
        from sklearn.cluster import KMeans

        self._km = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
        self._km.fit(Z)
        return self

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self._km is not None
        lab = self._km.predict(Z)
        P = np.zeros((Z.shape[0], self.n_states), dtype=float)
        P[np.arange(Z.shape[0]), lab] = 1.0
        return P


@dataclass
class GaussianHMMRegime:
    """True Gaussian HMM via hmmlearn. Raises if unavailable — do not silently fall back."""

    n_states: int = 3
    covariance_type: Literal["diag", "full"] = "diag"
    random_state: int = 0
    n_iter: int = 100
    _hmm: object = None

    def fit(self, Z: NDArray[np.floating]) -> "GaussianHMMRegime":
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError(
                "GaussianHMMRegime requires hmmlearn. Install it, or use GMMRegime / "
                "GMMTransitionForecast (not an HMM) explicitly."
            ) from e
        self._hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_iter=self.n_iter,
        )
        self._hmm.fit(Z)
        return self

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self._hmm is not None
        # hmmlearn posterior per frame
        post = self._hmm.predict_proba(Z)
        return _normalize(post)

    def predictive_occupancy(
        self,
        p0: NDArray[np.floating],
        horizon: int,
    ) -> NDArray[np.floating]:
        """Average expected occupancy over horizon using daily transition matrix A."""
        assert self._hmm is not None
        A = np.asarray(self._hmm.transmat_, dtype=float)
        p = _normalize(np.asarray(p0, dtype=float).ravel())
        acc = np.zeros_like(p)
        for _ in range(int(horizon)):
            p = p @ A
            acc += p
        return acc / max(int(horizon), 1)

    @property
    def transmat(self) -> NDArray[np.floating]:
        assert self._hmm is not None
        return np.asarray(self._hmm.transmat_, dtype=float)


@dataclass
class GMMTransitionForecast:
    """GMM memberships + empirical transition matrix. NOT an HMM — labeled explicitly."""

    n_states: int = 3
    covariance_type: Literal["diag", "tied"] = "diag"
    random_state: int = 0
    _gmm: Optional[GMMRegime] = None
    A_: Optional[NDArray[np.floating]] = None

    def fit(self, Z: NDArray[np.floating]) -> "GMMTransitionForecast":
        self._gmm = GMMRegime(
            n_states=self.n_states, covariance_type=self.covariance_type, random_state=self.random_state
        )
        self._gmm.fit(Z)
        P = self._gmm.predict_proba(Z)
        hard = P.argmax(axis=1)
        K = self.n_states
        A = np.ones((K, K), dtype=float)
        for t in range(len(hard) - 1):
            A[hard[t], hard[t + 1]] += 1.0
        A = A / A.sum(axis=1, keepdims=True)
        self.A_ = A
        return self

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self._gmm is not None
        return self._gmm.predict_proba(Z)

    def predictive_occupancy(self, p0: NDArray[np.floating], horizon: int) -> NDArray[np.floating]:
        assert self.A_ is not None
        p = _normalize(np.asarray(p0, dtype=float).ravel())
        acc = np.zeros_like(p)
        A = self.A_
        for _ in range(int(horizon)):
            p = p @ A
            acc += p
        return acc / max(int(horizon), 1)


def build_regime_model(name: str, **kwargs):
    name = str(name).lower()
    if name in ("fcm", "fuzzy_cmeans"):
        return FCMRegime(**{k: kwargs[k] for k in ("n_states", "fuzziness", "random_state") if k in kwargs})
    if name == "gmm":
        return GMMRegime(**{k: kwargs[k] for k in ("n_states", "covariance_type", "random_state") if k in kwargs})
    if name in ("kmeans", "hard_kmeans"):
        return HardKMeansRegime(**{k: kwargs[k] for k in ("n_states", "random_state") if k in kwargs})
    if name in ("hmm", "gaussian_hmm"):
        return GaussianHMMRegime(**{k: kwargs[k] for k in ("n_states", "covariance_type", "random_state", "n_iter") if k in kwargs})
    if name in ("gmm_transition_forecast", "gmm_transition"):
        return GMMTransitionForecast(**{k: kwargs[k] for k in ("n_states", "covariance_type", "random_state") if k in kwargs})
    raise ValueError(f"Unknown regime model: {name}")
