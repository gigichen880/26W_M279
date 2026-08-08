"""Representation geometry diagnostics (distance concentration, train vs query FCM)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def pairwise_distance_stats(Z: NDArray[np.floating], metric: str = "euclidean", max_pairs: int = 50000) -> dict:
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    if n < 3:
        return {"n": n, "cv": np.nan, "nn_med_ratio": np.nan, "rel_contrast": np.nan}
    # subsample pairs for large n
    rng = np.random.default_rng(0)
    if n * (n - 1) // 2 > max_pairs:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        m = i != j
        i, j = i[m], j[m]
    else:
        i, j = np.triu_indices(n, k=1)
    if metric == "manhattan":
        d = np.sum(np.abs(Z[i] - Z[j]), axis=1)
    else:
        d = np.sqrt(np.sum((Z[i] - Z[j]) ** 2, axis=1) + 1e-12)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return {"n": n, "cv": np.nan, "nn_med_ratio": np.nan, "rel_contrast": np.nan}
    med = float(np.median(d))
    # NN distances
    nn = []
    for a in range(min(n, 400)):
        if metric == "manhattan":
            da = np.sum(np.abs(Z - Z[a]), axis=1)
        else:
            da = np.sqrt(np.sum((Z - Z[a]) ** 2, axis=1) + 1e-12)
        da[a] = np.inf
        nn.append(float(np.min(da)))
    nn = np.asarray(nn, dtype=float)
    dmax, dmin = float(np.max(d)), float(np.min(d))
    return {
        "n": n,
        "dim": int(Z.shape[1]),
        "mean_dist": float(np.mean(d)),
        "std_dist": float(np.std(d)),
        "cv": float(np.std(d) / (np.mean(d) + 1e-12)),
        "nn_med_ratio": float(np.mean(nn) / (med + 1e-12)),
        "rel_contrast": float((dmax - dmin) / (dmin + 1e-12)),
        "dmin": dmin,
        "dmax": dmax,
    }


def membership_stats(PI: NDArray[np.floating]) -> dict:
    P = np.asarray(PI, dtype=float)
    P = np.clip(P, 1e-300, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    K = P.shape[1]
    H = -np.sum(P * np.log(P), axis=1)
    Hn = H / max(np.log(K), 1e-12)
    mx = P.max(axis=1)
    # top1 - top2
    sp = np.sort(P, axis=1)
    gap = sp[:, -1] - sp[:, -2] if K >= 2 else mx
    return {
        "n": int(P.shape[0]),
        "K": int(K),
        "mean_entropy": float(np.mean(H)),
        "mean_norm_entropy": float(np.mean(Hn)),
        "median_norm_entropy": float(np.median(Hn)),
        "mean_max_p": float(np.mean(mx)),
        "median_max_p": float(np.median(mx)),
        "mean_top_gap": float(np.mean(gap)),
        "frac_max_gt_0.5": float(np.mean(mx > 0.5)),
        "frac_max_gt_0.7": float(np.mean(mx > 0.7)),
        "frac_max_gt_0.9": float(np.mean(mx > 0.9)),
    }


def centroid_distance_contrast(
    Z: NDArray[np.floating],
    centers: NDArray[np.floating],
) -> dict:
    """Relative distance contrast of each point to K centers: (dmax-dmin)/dmin."""
    Z = np.asarray(Z, dtype=float)
    C = np.asarray(centers, dtype=float)
    contrasts = []
    for i in range(Z.shape[0]):
        d = np.sqrt(np.sum((C - Z[i]) ** 2, axis=1) + 1e-12)
        dmin = float(np.min(d))
        contrasts.append((float(np.max(d)) - dmin) / (dmin + 1e-12))
    c = np.asarray(contrasts, dtype=float)
    return {
        "mean_centroid_rel_contrast": float(np.mean(c)),
        "median_centroid_rel_contrast": float(np.median(c)),
    }


def fit_fcm_predict(
    Z_train: NDArray[np.floating],
    Z_query: NDArray[np.floating],
    *,
    n_regimes: int = 4,
    fuzziness: float = 2.0,
    random_state: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return (PI_train, PI_query, centers)."""
    from similarity_forecast.regime_clustering import FuzzyCMeansRegimeClusterer

    cl = FuzzyCMeansRegimeClusterer(
        n_regimes=n_regimes, fuzziness=fuzziness, random_state=random_state
    )
    cl.fit(Z_train)
    PI_tr = cl.predict_proba(Z_train)
    PI_q = cl.predict_proba(Z_query) if Z_query.size else np.zeros((0, n_regimes))
    return PI_tr, PI_q, np.asarray(cl.centers_, dtype=float)
