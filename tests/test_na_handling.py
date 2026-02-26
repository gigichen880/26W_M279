"""
Tests for NA handling in similarity_forecast.
"""
import numpy as np
import pandas as pd

from similarity_forecast.core import cov_from_returns, validate_window


def test_cov_with_nas():
    """Test covariance computation with NAs."""
    T, N = 100, 5
    np.random.seed(42)
    returns = np.random.randn(T, N) * 0.01

    # Add some NAs (20%)
    na_mask = np.random.rand(T, N) < 0.2
    returns[na_mask] = np.nan

    cov = cov_from_returns(returns)

    assert cov.shape == (N, N)
    assert np.allclose(cov, cov.T), "Covariance should be symmetric"
    assert np.all(np.linalg.eigvals(cov) > 0), "Covariance should be positive definite"
    assert not np.isnan(cov).any(), "Result should have no NaNs"

    print("✓ Covariance with NAs test passed")


def test_window_validation():
    """Test window validation logic."""
    T, N = 60, 100
    np.random.seed(42)

    # Good window (few NAs)
    returns_good = np.random.randn(T, N) * 0.01
    returns_good[np.random.rand(T, N) < 0.1] = np.nan
    assert validate_window(returns_good) is True

    # Bad window (too many NAs)
    returns_bad = np.random.randn(T, N) * 0.01
    returns_bad[np.random.rand(T, N) < 0.5] = np.nan
    assert validate_window(returns_bad) is False

    # Bad window (many stocks entirely missing)
    returns_bad2 = np.random.randn(T, N) * 0.01
    returns_bad2[:, :30] = np.nan
    assert validate_window(returns_bad2) is False

    print("✓ Window validation test passed")


if __name__ == "__main__":
    test_cov_with_nas()
    test_window_validation()
