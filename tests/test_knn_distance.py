import numpy as np
import pytest

from similarity_forecast.core import (
    AngularEmbeddingDistance,
    CosineEmbeddingDistance,
    ExactKNN,
    LpEmbeddingDistance,
    make_embedding_distance,
)


def test_l2_matches_manual():
    E = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    q = np.array([0.0, 0.0], dtype=float)
    d = LpEmbeddingDistance(p=2.0).pairwise(E, q)
    np.testing.assert_allclose(d, [0.0, 1.0, 1.0])


def test_cosine_symmetry():
    E = np.eye(3, dtype=float)
    q = np.array([1.0, 0.0, 0.0], dtype=float)
    d = CosineEmbeddingDistance().pairwise(E, q)
    np.testing.assert_allclose(d, [0.0, 1.0, 1.0], atol=1e-7)


def test_angular_right_angle():
    E = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    q = np.array([1.0, 0.0], dtype=float)
    d = AngularEmbeddingDistance().pairwise(E, q)
    np.testing.assert_allclose(d[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(d[1], np.pi / 2, atol=1e-7)


def test_exact_knn_query_order():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((20, 5))
    q = rng.standard_normal(5)
    knn = ExactKNN(E, metric="l2")
    idx, dist = knn.query(q, k=5)
    full = LpEmbeddingDistance(p=2.0).pairwise(E, q)
    np.testing.assert_allclose(dist, full[idx])


def test_make_embedding_distance_lp_alias():
    d1 = make_embedding_distance("l1")
    assert isinstance(d1, LpEmbeddingDistance)
    assert d1.p == 1.0
    d3 = make_embedding_distance("lp", lp_p=3.0)
    assert d3.p == 3.0


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        make_embedding_distance("not_a_metric")
