"""Clean similarity-only covariance forecaster (Stage A / D0)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from similarity_forecast.core import cov_from_returns, expm_sym, logm_spd, project_to_spd, symmetrize


AggName = Literal["arithmetic", "log_euclidean"]


@dataclass
class SimilarityLibrary:
    """Historical anchors with embeddings and future covariance targets."""

    anchors: NDArray[np.int64]
    embeds: NDArray[np.floating]
    targets: list  # list of (N,N) arrays
    regime_p: Optional[NDArray[np.floating]] = None  # (M, K) predictive memberships at anchor


@dataclass
class SimilarityForecaster:
    metric: Literal["euclidean", "manhattan"] = "euclidean"
    k_neighbors: int = 20
    kernel: Literal["adaptive", "fixed"] = "adaptive"
    tau: float = 1.0
    aggregation: AggName = "log_euclidean"
    eps: float = 1e-12

    def _distances(self, z: NDArray[np.floating], Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.metric == "manhattan":
            return np.sum(np.abs(Z - z[None, :]), axis=1)
        return np.sqrt(np.sum((Z - z[None, :]) ** 2, axis=1) + self.eps)

    def _weights(self, dist: NDArray[np.floating]) -> NDArray[np.floating]:
        d = np.asarray(dist, dtype=float)
        if d.size == 0:
            return d
        if self.kernel == "adaptive":
            # bandwidth = distance to k-th neighbor (last in sorted list)
            bw = max(float(d[-1]), self.eps)
            kappa = np.exp(-d / bw)
        else:
            kappa = np.exp(-d / max(float(self.tau), self.eps))
        kappa = np.maximum(kappa, 0.0)
        s = float(kappa.sum())
        return kappa / s if s > 0 else np.ones_like(kappa) / kappa.size

    def predict(
        self,
        z: NDArray[np.floating],
        library: SimilarityLibrary,
        *,
        raw_anchor: int,
        horizon: int,
        neighbor_gap: int,
        regime_p_query: Optional[NDArray[np.floating]] = None,
        regime_mode: Literal["none", "overlap", "overlap_pred"] = "none",
    ) -> tuple[NDArray[np.floating], dict]:
        cutoff = raw_anchor - horizon - neighbor_gap
        elig = library.anchors <= cutoff
        if not np.any(elig):
            raise ValueError("No eligible neighbors")
        idx_all = np.where(elig)[0]
        Z = library.embeds[idx_all]
        dist = self._distances(z, Z)
        order = np.argsort(dist)
        take = order[: min(self.k_neighbors, order.size)]
        idx = idx_all[take]
        dist_k = dist[take]
        w = self._weights(dist_k)

        # overlap_pred uses the same pᵀp weight; p vectors are predictive occupancies
        if regime_mode in ("overlap", "overlap_pred") and regime_p_query is not None and library.regime_p is not None:
            p_i = library.regime_p[idx]
            overlap = p_i @ np.asarray(regime_p_query, dtype=float).ravel()
            overlap = np.maximum(overlap, self.eps)
            w = w * overlap
            w = w / max(float(w.sum()), self.eps)

        targets = [library.targets[i] for i in idx]
        Sigma = self._aggregate(targets, w)
        info = {"neighbor_idx": idx, "dist": dist_k, "weights": w}
        return Sigma, info

    def _aggregate(self, targets: Sequence[NDArray[np.floating]], w: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.aggregation == "arithmetic":
            S = np.zeros_like(targets[0], dtype=float)
            for wi, Ti in zip(w, targets):
                S += float(wi) * np.asarray(Ti, dtype=float)
            return project_to_spd(symmetrize(S), eps=self.eps)
        # log-Euclidean
        Acc = np.zeros_like(targets[0], dtype=float)
        for wi, Ti in zip(w, targets):
            Acc += float(wi) * logm_spd(project_to_spd(Ti, eps=self.eps), eps=self.eps)
        return project_to_spd(expm_sym(Acc), eps=self.eps)


def build_covariance_target(fut: NDArray[np.floating], ddof: int = 1) -> NDArray[np.floating]:
    return cov_from_returns(fut, ddof=ddof)
