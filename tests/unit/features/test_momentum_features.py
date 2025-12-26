"""
Tests for momentum & technical features (Phase 9.3 — Week 2, TDD)

Covers:
- Basic momentum calculations using price_Nm_ago columns
- RSI edge cases (all up, all down, flat) using synthetic daily history columns
- Moving average crossover signals
- Return stability with varying volatility (sharpe proxy)
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd

# Prefer new path; fallback to legacy if needed
try:
    from finance_ml.features.advanced import engineer_momentum_features
except Exception:  # pragma: no cover
    from finance_ml.advanced_features import engineer_momentum_features  # type: ignore


class TestMomentumBasic(unittest.TestCase):
    def test_basic_momentum_calculations(self):
        df = pd.DataFrame(
            {
                "last_price": [110.0, 90.0],
                "price_1m_ago": [100.0, 100.0],
                "price_3m_ago": [80.0, 120.0],
                "price_6m_ago": [70.0, 100.0],
                "price_1y_ago": [55.0, 180.0],
            }
        )
        res = engineer_momentum_features(df)
        # 1m momentum: (110-100)/100*100 = 10%; (90-100)/100*100 = -10%
        np.testing.assert_allclose(res.loc[0, "price_momentum_1m"], 10.0)
        np.testing.assert_allclose(res.loc[1, "price_momentum_1m"], -10.0)
        # 3m momentum
        np.testing.assert_allclose(res.loc[0, "price_momentum_3m"], (110 - 80) / 80 * 100)
        np.testing.assert_allclose(res.loc[1, "price_momentum_3m"], (90 - 120) / 120 * 100)
        # Acceleration 3m = mom3 - mom1
        np.testing.assert_allclose(
            res.loc[0, "price_acceleration_3m"],
            res.loc[0, "price_momentum_3m"] - res.loc[0, "price_momentum_1m"],
        )
        # 6m and 1y
        self.assertIn("price_momentum_6m", res.columns)
        self.assertIn("price_momentum_1y", res.columns)


class TestRSIEdgeCases(unittest.TestCase):
    def test_rsi_all_up(self):
        # Construct 15 days of steadily increasing prices
        base = 100.0
        history = {f"price_{d}d_ago": [base + (15 - d)] for d in range(14, 0, -1)}
        df = pd.DataFrame({**history, "last_price": [base + 15]})
        res = engineer_momentum_features(df)
        # RSI close to 100 for all-up sequence
        self.assertIn("rsi_14d", res.columns)
        self.assertGreaterEqual(res.loc[0, "rsi_14d"], 99.0)

    def test_rsi_all_down(self):
        base = 100.0
        history = {f"price_{d}d_ago": [base - (15 - d)] for d in range(14, 0, -1)}
        df = pd.DataFrame({**history, "last_price": [base - 15]})
        res = engineer_momentum_features(df)
        self.assertLessEqual(res.loc[0, "rsi_14d"], 1.0)

    def test_rsi_flat(self):
        base = 100.0
        history = {f"price_{d}d_ago": [base] for d in range(14, 0, -1)}
        df = pd.DataFrame({**history, "last_price": [base]})
        res = engineer_momentum_features(df)
        # Flat sequence → RSI ~ 50 by definition used
        self.assertAlmostEqual(float(res.loc[0, "rsi_14d"]), 50.0, delta=1e-6)


class TestMovingAverageSignals(unittest.TestCase):
    def test_ma_crossover_and_distance(self):
        # Build 60-day history where price rises so 20d MA > 50d MA and last_price above MA50
        prices = [i for i in range(1, 61)]  # increasing 1..60
        # Provide 59 days of history plus last_price -> total 60 points
        history = {f"price_{d}d_ago": [prices[-(d + 1)]] for d in range(59, 0, -1)}
        df = pd.DataFrame({**history, "last_price": [prices[-1]]})
        res = engineer_momentum_features(df)
        self.assertIn("ma_crossover_signal", res.columns)
        self.assertEqual(int(res.loc[0, "ma_crossover_signal"]), 1)
        # Distance from MA should be positive
        self.assertGreater(float(res.loc[0, "price_distance_from_ma"]), 0.0)


class TestReturnStability(unittest.TestCase):
    def test_return_stability_and_sharpe_proxy(self):
        df = pd.DataFrame(
            {
                "last_price": [110.0],
                "price_1y_ago": [100.0],
                "volatility_1y_pct": [10.0],  # percent volatility
            }
        )
        # set risk-free via env for deterministic test
        os.environ["RISK_FREE_RATE_PCT"] = "2.0"
        res = engineer_momentum_features(df)
        # total return proxy ~ 10% / vol 10% => 1.0, sharpe (10-2)/10 = 0.8
        self.assertAlmostEqual(float(res.loc[0, "return_stability_score"]), 1.0, places=4)
        self.assertAlmostEqual(float(res.loc[0, "sharpe_proxy"]), 0.8, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
