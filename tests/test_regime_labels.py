"""Tests for data-driven regime labeling."""

import unittest

import numpy as np
import pandas as pd

from similarity_forecast.regime_labels import (
    assign_regime_labels,
    compute_horizon_cross_sectional_stats,
    compute_regime_diagnostics,
    infer_n_regimes,
)


class TestRegimeLabels(unittest.TestCase):
    def test_horizon_cross_sectional_stats_shape(self):
        rng = np.random.default_rng(0)
        fut = rng.normal(0, 0.01, size=(20, 10))
        out = compute_horizon_cross_sectional_stats(fut)
        self.assertIn("realized_vol", out)
        self.assertTrue(np.isfinite(out["realized_vol"]))

    def test_assign_regime_labels_four_regimes(self):
        regime_stats = pd.DataFrame(
            {
                "regime_id": [0, 1, 2, 3],
                "mean_realized_vol": [0.01, 0.02, 0.08, 0.03],
                "mean_avg_corr": [0.1, 0.15, 0.45, 0.2],
                "mean_market_ret": [0.02, 0.01, -0.05, 0.03],
                "n_days": [100, 100, 100, 100],
                "pct_time": [25.0, 25.0, 25.0, 25.0],
            }
        )
        crisis_counts = pd.Series([5, 10, 80, 5], index=[0, 1, 2, 3])
        mp = assign_regime_labels(regime_stats, crisis_counts)
        self.assertEqual(set(mp.values()), {"High Stress", "Calm Bull", "Moderate Bull", "Normal"})
        self.assertEqual(mp[2], "High Stress")

    def test_compute_regime_diagnostics_synthetic(self):
        n = 400
        rng = np.random.default_rng(1)
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        dom = rng.integers(0, 4, size=n)
        realized_vol = rng.uniform(0.005, 0.1, size=n)
        avg_corr = rng.uniform(-0.1, 0.5, size=n)
        market_ret = rng.normal(0, 0.02, size=n)
        df = pd.DataFrame(
            {
                "date": dates,
                "dominant_regime": dom,
                "realized_vol": realized_vol,
                "avg_corr": avg_corr,
                "market_ret": market_ret,
                "regime_prob_0": np.full(n, 0.25),
                "regime_prob_1": np.full(n, 0.25),
                "regime_prob_2": np.full(n, 0.25),
                "regime_prob_3": np.full(n, 0.25),
            }
        )
        out = compute_regime_diagnostics(df)
        self.assertEqual(len(out["regime_stats"]), 4)
        self.assertEqual(infer_n_regimes(df), 4)


if __name__ == "__main__":
    unittest.main()
