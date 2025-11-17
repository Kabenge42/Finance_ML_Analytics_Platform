"""Tests for technical analysis integration features (Phase 9.3 Schema 1.3).

This suite focuses on the new engineer_technical_analysis_features function
that consumes normalized EMA and 52-week high/low columns coming from the
preprocessing.normalize_columns mapping (ema_20d, ema_50d, ema_100d,
ema_250d, 52w_high_adj, 52w_low_adj, rel_volume).

Covered behaviors:
- EMA crossover signals and price-vs-EMA deviations
- 52-week range position and near-high/near-low flags
- Volume-momentum composite score and breakout signal

Edge cases:
- Missing inputs (columns absent or containing NaNs/zeros)
- Numerical hygiene (no +/- inf introduced)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.utils.feature_test_helpers import assert_no_inf, assert_within_range


class TestTechnicalAnalysisEmaSignals(unittest.TestCase):
    def _make_base_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "last_price": [110.0, 90.0, 100.0],
                # EMA stack: bullish for A (short > long, price above),
                # bearish for B (short < long, price below), mixed for C.
                "ema_20d": [105.0, 95.0, 100.0],
                "ema_50d": [100.0, 100.0, 100.0],
                "ema_100d": [98.0, 102.0, 100.0],
                "ema_250d": [95.0, 105.0, 100.0],
            }
        )

    def test_ema_crossover_and_price_vs_ema(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_momentum_features,
        )

        # Import inside test so it fails clearly if function is missing
        from finance_ml.ml_workflow.features.advanced import (
            engineer_technical_analysis_features,
        )

        df = self._make_base_df()
        # Provide a simple 1y momentum proxy to be re-used by technical features
        df["price_1y_ago"] = [100.0, 100.0, 100.0]
        df = engineer_momentum_features(df)
        result = engineer_technical_analysis_features(df)

        # EMA crossover 20/50: A bullish (+1), B bearish (-1), C neutral (0)
        self.assertIn("ema_crossover_20_50", result.columns)
        self.assertEqual(int(result.loc[0, "ema_crossover_20_50"]), 1)
        self.assertEqual(int(result.loc[1, "ema_crossover_20_50"]), -1)
        self.assertEqual(int(result.loc[2, "ema_crossover_20_50"]), 0)

        # Long-term 50/250 signal should agree in direction for A and B
        self.assertIn("ema_crossover_50_250", result.columns)
        self.assertEqual(int(result.loc[0, "ema_crossover_50_250"]), 1)
        self.assertEqual(int(result.loc[1, "ema_crossover_50_250"]), -1)

        # Price vs EMA deviations in percent, positive for A, negative for B
        self.assertIn("price_vs_ema_20d", result.columns)
        self.assertIn("price_vs_ema_250d", result.columns)
        self.assertGreater(float(result.loc[0, "price_vs_ema_20d"]), 0.0)
        self.assertLess(float(result.loc[1, "price_vs_ema_20d"]), 0.0)

        # Trend consistency score in [-1, 1]
        self.assertIn("ema_trend_consistency", result.columns)
        assert_within_range(result, column="ema_trend_consistency", min_value=-1.0, max_value=1.0)


class TestTechnicalAnalysis52WeekRange(unittest.TestCase):
    def test_52w_position_and_flags(self):
        from finance_ml.ml_workflow.features.advanced import engineer_technical_analysis_features

        df = pd.DataFrame(
            {
                "last_price": [95.0, 10.0, 100.0],
                "52w_high_adj": [100.0, 20.0, 100.0],
                "52w_low_adj": [50.0, 10.0, 100.0],
            }
        )

        result = engineer_technical_analysis_features(df)

        # pct_off_52w_high = (high - price) / high
        self.assertIn("pct_off_52w_high", result.columns)
        np.testing.assert_allclose(result.loc[0, "pct_off_52w_high"], (100.0 - 95.0) / 100.0)
        # pct_above_52w_low = (price - low) / low
        self.assertIn("pct_above_52w_low", result.columns)
        np.testing.assert_allclose(result.loc[0, "pct_above_52w_low"], (95.0 - 50.0) / 50.0)

        # 52w_range_position is within [0, 1]
        self.assertIn("52w_range_position", result.columns)
        assert_within_range(result, column="52w_range_position", min_value=0.0, max_value=1.0)

        # Near-high flag within 5% of high, near-low flag within 5% of low
        self.assertIn("near_52w_high_flag", result.columns)
        self.assertIn("near_52w_low_flag", result.columns)
        # First row: close to high, not close to low
        self.assertEqual(int(result.loc[0, "near_52w_high_flag"]), 1)
        self.assertEqual(int(result.loc[0, "near_52w_low_flag"]), 0)
        # Second row: at low boundary
        self.assertEqual(int(result.loc[1, "near_52w_low_flag"]), 1)
        # Third row: price equals both high and low -> position undefined but should not crash


class TestTechnicalAnalysisVolumeMomentum(unittest.TestCase):
    def test_volume_momentum_and_breakout_signal(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_momentum_features,
            engineer_technical_analysis_features,
        )

        df = pd.DataFrame(
            {
                "last_price": [110.0, 90.0],
                "price_1m_ago": [100.0, 100.0],
                "price_3m_ago": [100.0, 100.0],
                # Choose a high such that the first row is within 5% of the 52w high
                # so that near_52w_high_flag can activate for the breakout scenario.
                "52w_high_adj": [115.0, 100.0],
                "52w_low_adj": [80.0, 80.0],
                "rel_volume": [2.0, 0.5],  # elevated vs subdued volume
            }
        )

        df = engineer_momentum_features(df)
        result = engineer_technical_analysis_features(df)

        # volume_momentum_score uses rel_volume and a price momentum proxy
        self.assertIn("volume_momentum_score", result.columns)
        # First row should have higher score than second due to higher rel_volume and positive momentum
        self.assertGreater(
            float(result.loc[0, "volume_momentum_score"]),
            float(result.loc[1, "volume_momentum_score"]),
        )

        # Breakout signal should be binary and highlight strong price near high with volume
        self.assertIn("breakout_signal", result.columns)
        assert_within_range(result, column="breakout_signal", min_value=0.0, max_value=1.0)
        # High momentum & near-high & strong volume triggers breakout for first row
        self.assertEqual(int(result.loc[0, "breakout_signal"]), 1)

        # Numerical hygiene
        assert_no_inf(result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
