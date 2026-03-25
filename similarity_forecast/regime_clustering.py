# similarity_forecast/regime_clustering.py
"""
Pluggable regime assignment (Stage 2): map embedding rows Z [T, D] -> soft PI [T, K].

Slide-aligned families (MATH 279 style):
  - Probabilistic: gmm, fuzzy_cmeans
  - Classical Euclidean: kmeans_soft, agglomerative_ward, modified_two_stage
  - Graph / spectral (on embedding rows): spectral_rbf, spectral_knn, signed_knn_spectral

Out-of-sample query points (predict_pi on a new row z) use the same backend where possible:
  - gmm: GaussianMixture.predict_proba
  - k-means / agglomerative / spectral / signed_spectral / two-stage: softmax over
    cluster prototypes in **original embedding space** (means of Z weighted by train assignment)

Not implemented here (need extra inputs beyond Z — use YAML name only to surface a clear error):
  - hermitian_spectral (directed / skew adjacency)
  - disg_co_clustering (bipartite co-clustering)
  - dsbm (directed SBM — generative / benchmarking)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray


def _normalize_pi(PI: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    PI = np.maximum(np.asarray(PI, dtype=float), eps)
    PI = PI / np.maximum(PI.sum(axis=1, keepdims=True), eps)
    return PI.astype(float)


def _softmax_rows(logits: NDArray[np.floating], *, temperature: float = 1.0) -> NDArray[np.floating]:
    t = max(float(temperature), 1e-12)
    x = np.asarray(logits, dtype=float) / t
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    s = ex.sum(axis=1, keepdims=True)
    return (ex / np.maximum(s, 1e-300)).astype(float)


def _prototypes_softassign(Z: NDArray, labels: NDArray[np.integer], n_regimes: int) -> NDArray[np.floating]:
    """Mean prototype per cluster; empty clusters fall back to global mean."""
    Z = np.asarray(Z, dtype=float)
    T, D = Z.shape
    mu = Z.mean(axis=0)
    protos = np.tile(mu[None, :], (n_regimes, 1))
    for k in range(n_regimes):
        m = labels == k
        if np.any(m):
            protos[k] = Z[m].mean(axis=0)
    return protos.astype(float)


def _predict_proba_from_prototypes(
    Z: NDArray[np.floating],
    prototypes: NDArray[np.floating],
    *,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> NDArray[np.floating]:
    """Softmax of negative squared Euclidean distance to prototypes."""
    Z = np.asarray(Z, dtype=float)
    P = np.asarray(prototypes, dtype=float)
    d2 = ((Z[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    return _normalize_pi(_softmax_rows(-beta * d2), eps=eps)


class RegimeClusterer(ABC):
    """Fit on training embeddings; predict_proba for train or query rows."""

    @abstractmethod
    def fit(self, Z: NDArray[np.floating]) -> None:
        ...

    @abstractmethod
    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        ...


@dataclass
class GMMRegimeClusterer(RegimeClusterer):
    n_regimes: int
    random_state: int = 0
    eps: float = 1e-12
    covariance_type: str = "diag"
    reg_covar: float = 1e-3
    max_iter: int = 300
    tol: float = 1e-3
    gmm_init_params: str = "kmeans"
    gmm_n_init: int = 1
    gmm_: Any = field(default=None, repr=False)

    def fit(self, Z: NDArray[np.floating]) -> None:
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore
        except Exception as e:
            raise ImportError("scikit-learn is required for GMMRegimeClusterer.") from e

        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2D [T, D], got {Z.shape}")
        init_params = str(self.gmm_init_params).lower()
        if init_params not in {"kmeans", "random", "random_from_data"}:
            raise ValueError("gmm_init_params must be one of {'kmeans','random','random_from_data'}")

        def _make(init: str) -> Any:
            return GaussianMixture(
                n_components=self.n_regimes,
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
            if init_params == "kmeans":
                print(f"[warn] GMM init via kmeans failed ({e}); retrying with init_params='random'")
                gmm = _make("random")
                gmm.fit(Z)
            else:
                raise
        self.gmm_ = gmm

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.gmm_ is None:
            raise RuntimeError("Call fit() first.")
        Z = np.asarray(Z, dtype=float)
        PI = self.gmm_.predict_proba(Z)
        return _normalize_pi(PI, eps=self.eps)


@dataclass
class KMeansSoftRegimeClusterer(RegimeClusterer):
    """k-means centroids + softmax of negative squared distance (fuzzy-like soft labels)."""

    n_regimes: int
    random_state: int = 0
    eps: float = 1e-12
    n_init: int = 10
    max_iter: int = 300
    temperature: float = 1.0
    km_: Any = field(default=None, repr=False)

    def fit(self, Z: NDArray[np.floating]) -> None:
        from sklearn.cluster import KMeans  # type: ignore

        Z = np.asarray(Z, dtype=float)
        self.km_ = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=int(self.n_init),
            max_iter=int(self.max_iter),
        ).fit(Z)

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.km_ is None:
            raise RuntimeError("Call fit() first.")
        centers = self.km_.cluster_centers_
        d2 = ((np.asarray(Z, dtype=float)[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return _normalize_pi(_softmax_rows(-d2, temperature=self.temperature), eps=self.eps)


@dataclass
class SpectralRegimeClusterer(RegimeClusterer):
    """
    Spectral embedding of rows of Z, k-means in embedding space, soft assignments via
    prototypes in **original** Z space (for valid OOS queries).
    """

    n_regimes: int
    random_state: int = 0
    eps: float = 1e-12
    affinity: str = "rbf"  # "rbf" | "nearest_neighbors"
    gamma: Optional[float] = None
    n_neighbors: int = 10
    temperature: float = 1.0
    labels_: Optional[NDArray[np.integer]] = field(default=None, repr=False)
    prototypes_: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def fit(self, Z: NDArray[np.floating]) -> None:
        from sklearn.cluster import KMeans, SpectralClustering  # type: ignore

        Z = np.asarray(Z, dtype=float)
        T, D = Z.shape
        if T < self.n_regimes:
            raise ValueError(f"Need T >= n_regimes, got T={T}, K={self.n_regimes}")

        aff = str(self.affinity).lower()
        if aff == "rbf":
            gamma = self.gamma
            if gamma is None:
                gamma = 1.0 / max(D, 1)
            spec = SpectralClustering(
                n_clusters=self.n_regimes,
                affinity="rbf",
                gamma=float(gamma),
                random_state=self.random_state,
                assign_labels="kmeans",
            )
        elif aff == "nearest_neighbors":
            spec = SpectralClustering(
                n_clusters=self.n_regimes,
                affinity="nearest_neighbors",
                n_neighbors=int(min(self.n_neighbors, T - 1)),
                random_state=self.random_state,
                assign_labels="kmeans",
            )
        else:
            raise ValueError("affinity must be 'rbf' or 'nearest_neighbors'")

        self.labels_ = spec.fit_predict(Z).astype(int)
        self.prototypes_ = _prototypes_softassign(Z, self.labels_, self.n_regimes)

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.prototypes_ is None:
            raise RuntimeError("Call fit() first.")
        return _predict_proba_from_prototypes(
            Z, self.prototypes_, beta=1.0 / max(float(self.temperature), 1e-12), eps=self.eps
        )


@dataclass
class AgglomerativeRegimeClusterer(RegimeClusterer):
    n_regimes: int
    random_state: int = 0
    eps: float = 1e-12
    linkage: str = "ward"
    temperature: float = 1.0
    labels_: Optional[NDArray[np.integer]] = field(default=None, repr=False)
    prototypes_: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def fit(self, Z: NDArray[np.floating]) -> None:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore

        Z = np.asarray(Z, dtype=float)
        self.labels_ = AgglomerativeClustering(
            n_clusters=self.n_regimes,
            linkage=str(self.linkage),
        ).fit_predict(Z).astype(int)
        self.prototypes_ = _prototypes_softassign(Z, self.labels_, self.n_regimes)

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.prototypes_ is None:
            raise RuntimeError("Call fit() first.")
        return _predict_proba_from_prototypes(
            Z, self.prototypes_, beta=1.0 / max(float(self.temperature), 1e-12), eps=self.eps
        )


@dataclass
class FuzzyCMeansRegimeClusterer(RegimeClusterer):
    """Fuzzy c-means (soft k-means); optional scikit-fuzzy."""

    n_regimes: int
    random_state: int = 0
    eps: float = 1e-12
    fuzziness: float = 2.0
    max_iter: int = 300
    error: float = 1e-5
    centers_: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def fit(self, Z: NDArray[np.floating]) -> None:
        try:
            import skfuzzy as fuzz  # type: ignore
        except Exception as e:
            raise ImportError(
                "FuzzyCMeansRegimeClusterer requires scikit-fuzzy: pip install scikit-fuzzy"
            ) from e

        Z = np.asarray(Z, dtype=float)
        D = Z.shape[1]
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            Z.T,
            c=self.n_regimes,
            m=float(self.fuzziness),
            error=float(self.error),
            maxiter=int(self.max_iter),
            init=None,
            seed=self.random_state,
        )
        self.centers_ = np.asarray(cntr, dtype=float)  # (K, D)

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.centers_ is None:
            raise RuntimeError("Call fit() first.")
        try:
            import skfuzzy as fuzz  # type: ignore
        except Exception as e:
            raise ImportError("scikit-fuzzy required for predict_proba.") from e
        Z = np.asarray(Z, dtype=float)
        # scikit-fuzzy >=0.4: cmeans_predict; older: cpredict
        cluster = fuzz.cluster
        if hasattr(cluster, "cmeans_predict"):
            u, _, _, _, _, _ = cluster.cmeans_predict(
                Z.T,
                self.centers_,
                m=float(self.fuzziness),
                error=float(self.error),
                maxiter=int(self.max_iter),
                seed=self.random_state,
            )
        elif hasattr(cluster, "cpredict"):
            u, _, _, _, _, _ = cluster.cpredict(Z.T, self.centers_)
        else:
            raise ImportError(
                "scikit-fuzzy cluster module has neither cmeans_predict nor cpredict; "
                "upgrade or reinstall scikit-fuzzy."
            )
        PI = np.asarray(u, dtype=float).T  # (T, K)
        return _normalize_pi(PI, eps=self.eps)


@dataclass
class ModifiedTwoStageRegimeClusterer(RegimeClusterer):
    """
    Stage 1: mark high L2 distance to global mean as regime 0 (stress / outlier bucket).
    Stage 2: k-means with cosine geometry on remaining rows for regimes 1..K-1.
    """

    n_regimes: int
    random_state: int = 0
    eps: float = 1e-12
    outlier_quantile: float = 0.9
    n_init: int = 10
    temperature: float = 1.0
    labels_: Optional[NDArray[np.integer]] = field(default=None, repr=False)
    prototypes_: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def fit(self, Z: NDArray[np.floating]) -> None:
        from sklearn.cluster import KMeans  # type: ignore

        Z = np.asarray(Z, dtype=float)
        T, D = Z.shape
        if self.n_regimes < 2:
            raise ValueError("modified_two_stage needs n_regimes >= 2 (outlier + >=1 inner regime).")
        mu = Z.mean(axis=0, keepdims=True)
        d2 = np.sum((Z - mu) ** 2, axis=1)
        thr = float(np.quantile(d2, float(self.outlier_quantile)))
        outlier = d2 >= thr
        labels = np.zeros(T, dtype=int)
        inner = ~outlier
        k_inner = self.n_regimes - 1
        if np.sum(inner) < k_inner:
            inner = np.ones(T, dtype=bool)
            outlier = np.zeros(T, dtype=bool)
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        if k_inner == 1:
            labels[inner] = 1
        else:
            km = KMeans(
                n_clusters=k_inner,
                random_state=self.random_state,
                n_init=int(self.n_init),
            ).fit(Zn[inner])
            labels[inner] = 1 + km.predict(Zn[inner])
        self.labels_ = labels
        self.prototypes_ = _prototypes_softassign(Z, labels, self.n_regimes)

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.prototypes_ is None:
            raise RuntimeError("Call fit() first.")
        return _predict_proba_from_prototypes(
            Z, self.prototypes_, beta=1.0 / max(float(self.temperature), 1e-12), eps=self.eps
        )


@dataclass
class SignedKNNSpectralRegimeClusterer(RegimeClusterer):
    """
    kNN graph on embedding rows; signed weights = cosine similarity (positive/negative edges).
    Symmetric signed Laplacian L = D^{abs} - S with S_ij = cosine(z_i, z_j) on edges;
    smallest eigenvectors -> k-means; prototypes in Z for OOS.
    """

    n_regimes: int
    random_state: int = 0
    eps: float = 1e-12
    n_neighbors: int = 15
    temperature: float = 1.0
    labels_: Optional[NDArray[np.integer]] = field(default=None, repr=False)
    prototypes_: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def fit(self, Z: NDArray[np.floating]) -> None:
        from scipy import sparse as sp  # type: ignore
        from scipy.sparse.linalg import eigsh  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore

        Z = np.asarray(Z, dtype=float)
        T, D = Z.shape
        k_nn = int(min(max(2, self.n_neighbors), T - 1))
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        sim = Zn @ Zn.T
        rows, cols, data = [], [], []
        for i in range(T):
            row_sim = sim[i].copy()
            row_sim[i] = -np.inf
            nn = np.argpartition(-row_sim, k_nn - 1)[:k_nn]
            for j in nn:
                if i == j:
                    continue
                w = float(sim[i, j])
                rows.append(i)
                cols.append(j)
                data.append(w)
        S = sp.coo_matrix((data, (rows, cols)), shape=(T, T)).tocsr()
        S = (S + S.T) * 0.5
        abs_deg = np.array(np.abs(S).sum(axis=1)).ravel()
        L = sp.diags(abs_deg, format="csr") - S
        k_ev = max(2, min(self.n_regimes, T - 2))
        try:
            _, vecs = eigsh(L.astype(float), k=k_ev, which="SA")
        except Exception:
            if T > 512:
                raise
            Ld = L.toarray()
            _, vecs = np.linalg.eigh(Ld)
            vecs = vecs[:, :k_ev]
        km = KMeans(n_clusters=self.n_regimes, random_state=self.random_state, n_init=10).fit(vecs)
        self.labels_ = km.labels_.astype(int)
        self.prototypes_ = _prototypes_softassign(Z, self.labels_, self.n_regimes)

    def predict_proba(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.prototypes_ is None:
            raise RuntimeError("Call fit() first.")
        return _predict_proba_from_prototypes(
            Z, self.prototypes_, beta=1.0 / max(float(self.temperature), 1e-12), eps=self.eps
        )


_UNSUPPORTED_SLIDE_METHODS: Dict[str, str] = {
    "hermitian_spectral": (
        "Hermitian / directed spectral clustering needs a directed adjacency (complex Hermitian). "
        "The walk-forward pipeline only provides embedding matrix Z [T,D]. "
        "Extend the data pipeline with a directed graph, then add a custom RegimeClusterer."
    ),
    "disg_co_clustering": (
        "DISG-style co-clustering needs a bipartite (row/column) matrix. "
        "Not defined from embedding rows Z alone."
    ),
    "dsbm": (
        "The directed stochastic block model is a generative benchmark, not a standard fitter on Z. "
        "Use for simulation studies outside this pipeline."
    ),
}


_CLUSTERER_BUILDERS: Dict[str, Callable[[int, int, Dict[str, Any]], RegimeClusterer]] = {
    "gmm": lambda n_regimes, random_state, p: GMMRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        covariance_type=str(p.get("covariance_type", "diag")),
        reg_covar=float(p.get("reg_covar", 1e-3)),
        max_iter=int(p.get("max_iter", 300)),
        tol=float(p.get("tol", 1e-3)),
        gmm_init_params=str(p.get("gmm_init_params", "kmeans")),
        gmm_n_init=int(p.get("gmm_n_init", 1)),
        eps=float(p.get("eps", 1e-12)),
    ),
    "kmeans_soft": lambda n_regimes, random_state, p: KMeansSoftRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        n_init=int(p.get("n_init", 10)),
        max_iter=int(p.get("max_iter", 300)),
        temperature=float(p.get("temperature", 1.0)),
        eps=float(p.get("eps", 1e-12)),
    ),
    "spectral_rbf": lambda n_regimes, random_state, p: SpectralRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        affinity="rbf",
        gamma=p.get("gamma"),
        temperature=float(p.get("temperature", 1.0)),
        eps=float(p.get("eps", 1e-12)),
    ),
    "spectral_knn": lambda n_regimes, random_state, p: SpectralRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        affinity="nearest_neighbors",
        n_neighbors=int(p.get("n_neighbors", 10)),
        temperature=float(p.get("temperature", 1.0)),
        eps=float(p.get("eps", 1e-12)),
    ),
    "agglomerative_ward": lambda n_regimes, random_state, p: AgglomerativeRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        linkage=str(p.get("linkage", "ward")),
        temperature=float(p.get("temperature", 1.0)),
        eps=float(p.get("eps", 1e-12)),
    ),
    "fuzzy_cmeans": lambda n_regimes, random_state, p: FuzzyCMeansRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        fuzziness=float(p.get("fuzziness", 2.0)),
        max_iter=int(p.get("max_iter", 300)),
        error=float(p.get("error", 1e-5)),
        eps=float(p.get("eps", 1e-12)),
    ),
    "modified_two_stage": lambda n_regimes, random_state, p: ModifiedTwoStageRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        outlier_quantile=float(p.get("outlier_quantile", 0.9)),
        n_init=int(p.get("n_init", 10)),
        temperature=float(p.get("temperature", 1.0)),
        eps=float(p.get("eps", 1e-12)),
    ),
    "signed_knn_spectral": lambda n_regimes, random_state, p: SignedKNNSpectralRegimeClusterer(
        n_regimes=n_regimes,
        random_state=random_state,
        n_neighbors=int(p.get("n_neighbors", 15)),
        temperature=float(p.get("temperature", 1.0)),
        eps=float(p.get("eps", 1e-12)),
    ),
}


def list_regime_clustering_methods() -> list[str]:
    return sorted(_CLUSTERER_BUILDERS.keys()) + sorted(_UNSUPPORTED_SLIDE_METHODS.keys())


def implemented_regime_clustering_names() -> list[str]:
    """Runnable backends for ablation (excludes slide-only stub names)."""
    return sorted(_CLUSTERER_BUILDERS.keys())


def make_regime_clusterer(
    name: str,
    n_regimes: int,
    random_state: int,
    params: Optional[Dict[str, Any]] = None,
) -> RegimeClusterer:
    key = str(name).strip().lower()
    if key in _UNSUPPORTED_SLIDE_METHODS:
        raise ValueError(_UNSUPPORTED_SLIDE_METHODS[key])
    if key not in _CLUSTERER_BUILDERS:
        known = ", ".join(sorted(_CLUSTERER_BUILDERS.keys()))
        raise ValueError(f"Unknown regime_clustering.name={name!r}. Implemented: {known}")
    return _CLUSTERER_BUILDERS[key](int(n_regimes), int(random_state), dict(params or {}))
