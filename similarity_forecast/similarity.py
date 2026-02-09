from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class ExactKNN:
    """
    Exact KNN in embedding space using L2 distance.
    Stores:
      E: [T, d] embeddings
    """
    E: NDArray[np.floating]  # training embeddings

    def query(self, e: NDArray[np.floating], k: int, exclude_index: Optional[int] = None) -> Tuple[NDArray[np.int64], NDArray[np.floating]]:
        """
        Returns neighbor indices and distances (ascending).
        exclude_index: optionally exclude the same-time point (for backtests)
        """
        diff = self.E - e[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)

        if exclude_index is not None and 0 <= exclude_index < d2.shape[0]:
            d2[exclude_index] = np.inf

        k = min(k, d2.shape[0])
        idx = np.argpartition(d2, kth=k-1)[:k]
        # sort selected
        order = np.argsort(d2[idx])
        idx = idx[order]
        dist = np.sqrt(d2[idx])
        return idx.astype(np.int64), dist


@dataclass
class TwoStageRerank:
    """
    Stage 1: cheap KNN on embeddings
    Stage 2: rerank top-M by a richer distance computed on full state objects (NxN)
            (e.g. Frobenius on correlation matrices).
    """
    knn: ExactKNN
    states: NDArray[np.floating]  # [T, N, N] full states aligned with knn.E
    stage2: str = "fro"           # "fro" or "corr_fro"
    M: int = 200                  # candidates to rerank (>= k)
    eps: float = 1e-12

    def _dist_state(self, A: NDArray[np.floating], B: NDArray[np.floating]) -> float:
        if self.stage2 == "fro":
            D = A - B
            return float(np.sqrt(np.sum(D * D)))
        raise ValueError(f"Unknown stage2 metric: {self.stage2}")

    def query(self, e: NDArray[np.floating], k: int, exclude_index: Optional[int] = None) -> Tuple[NDArray[np.int64], NDArray[np.floating]]:
        cand_k = min(max(k, self.M), self.knn.E.shape[0])
        cand_idx, _ = self.knn.query(e, k=cand_k, exclude_index=exclude_index)

        # rerank by full-state distance
        # NOTE: we need the query full state; pass it in by temporarily storing before call,
        # or provide a separate method. For simplicity, we expose a method that accepts A.
        raise RuntimeError("Use query_with_state(e, A, k, exclude_index)")

    def query_with_state(
        self,
        e: NDArray[np.floating],
        A: NDArray[np.floating],
        k: int,
        exclude_index: Optional[int] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.floating]]:
        cand_k = min(max(k, self.M), self.knn.E.shape[0])
        cand_idx, _ = self.knn.query(e, k=cand_k, exclude_index=exclude_index)

        d = np.empty(cand_idx.shape[0], dtype=float)
        for i, j in enumerate(cand_idx):
            d[i] = self._dist_state(A, self.states[j])

        order = np.argsort(d)[:k]
        nn_idx = cand_idx[order]
        nn_dist = d[order].astype(float)
        return nn_idx.astype(np.int64), nn_dist.astype(np.float64)
