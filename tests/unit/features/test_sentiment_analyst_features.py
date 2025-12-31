"""
Phase 9.3 (Week 5) — Market Sentiment & Analyst Features (TDD)

Covers:
- Analyst consensus calculations (bullish/bearish %, conviction)
- Price target features (upside, range, revision, coverage quality)
- Market sentiment (short interest, beta stability, systematic risk trend)
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

# Prefer consolidated module path; fall back to legacy for BC
try:  # pragma: no cover - import resolution shim
    from finance_ml.features.advanced import (
        engineer_analyst_quality_features,
        engineer_market_sentiment_features,
    )
except Exception:  # pragma: no cover - legacy import path
    from finance_ml.advanced_features import (  # type: ignore
        engineer_analyst_quality_features,
        engineer_market_sentiment_features,  # type: ignore[attr-defined]
    )


class TestAnalystConsensus(unittest.TestCase):
    def test_analyst_consensus_and_targets(self):
        df = pd.DataFrame(
            {
                "last_price": [100.0],
                # Ratings
                "num_strong_buys_ratings": [3],
                "num_buys_ratings": [4],
                "num_hold_ratings": [2],
                "num_sell_ratings": [1],
                "num_strong_sell_ratings": [0],
                # Targets
                "price_target_median": [120.0],
                "price_target_high": [150.0],
                "price_target_low": [90.0],
                "price_target_ytd_ago": [110.0],
                # Coverage
                "price_target_count": [10],  # number of analysts
                "market_cap": [1.0e10],
            }
        )
        res = engineer_analyst_quality_features(df)
        # Bullish/Bearish %
        self.assertIn("analyst_bullish_pct", res.columns)
        self.assertIn("analyst_bearish_pct", res.columns)
        self.assertAlmostEqual(float(res.loc[0, "analyst_bullish_pct"]), 70.0, places=6)
        self.assertAlmostEqual(float(res.loc[0, "analyst_bearish_pct"]), 10.0, places=6)
        # Conviction absolute spread
        self.assertIn("analyst_conviction", res.columns)
        self.assertAlmostEqual(float(res.loc[0, "analyst_conviction"]), 60.0, places=6)
        # Upside potential (%), target range (% of median), revision (ratio)
        self.assertIn("upside_potential", res.columns)
        self.assertIn("price_target_range", res.columns)
        self.assertIn("price_target_revision", res.columns)
        self.assertAlmostEqual(float(res.loc[0, "upside_potential"]), 20.0, places=6)
        self.assertAlmostEqual(
            float(res.loc[0, "price_target_range"]), (150.0 - 90.0) / 120.0 * 100.0, places=6
        )
        self.assertAlmostEqual(
            float(res.loc[0, "price_target_revision"]), (120.0 - 110.0) / 110.0, places=6
        )
        # Coverage quality: #analysts / log1p(market_cap)
        self.assertIn("analyst_coverage_quality", res.columns)
        expected_cov = 10.0 / math.log1p(1.0e10)
        self.assertAlmostEqual(
            float(res.loc[0, "analyst_coverage_quality"]), expected_cov, places=6
        )


class TestMarketSentiment(unittest.TestCase):
    def test_short_interest_and_beta_signals(self):
        df = pd.DataFrame(
            {
                "short_int_pct": [5.0],  # already in percent units
                "beta_1y": [1.2],
                "beta_2y": [1.1],
                "beta_5y": [0.9],
            }
        )
        res = engineer_market_sentiment_features(df)
        self.assertIn("short_interest_ratio", res.columns)
        self.assertAlmostEqual(float(res.loc[0, "short_interest_ratio"]), 5.0, places=6)
        self.assertIn("beta_stability", res.columns)
        self.assertIn("systematic_risk_trend", res.columns)
        # population variance of [1.2, 1.1, 0.9]
        expected_var = float(np.var([1.2, 1.1, 0.9]))
        self.assertAlmostEqual(float(res.loc[0, "beta_stability"]), expected_var, places=6)
        self.assertAlmostEqual(float(res.loc[0, "systematic_risk_trend"]), 1.2 - 0.9, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
