"""Tests for dividend reliability & income stock features (Schema 1.3).

The tests exercise a subset of features from ``engineer_dividend_reliability_features``:

- dividend_streak_years
- dividend_frequency_encoded
- dividend_consistency_score
- income_stock_flag
- dividend_payout_ratio

We focus on simple yet representative scenarios and NaN/edge-case handling.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.utils.feature_test_helpers import assert_no_inf, assert_within_range


class TestDividendReliabilityCore(unittest.TestCase):
    def test_dividend_streak_and_consistency_basic(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_dividend_reliability_features,
        )

        df = pd.DataFrame(
            {
                "dividend_streak": [2, 10, 7],
                "dividend_record_frequency": ["annual", "quarterly", "monthly"],
                "common_dividends_paid_ltm": [2.0, 20.0, 7.0],
                "net_income": [10.0, 40.0, 14.0],
            }
        )

        res = engineer_dividend_reliability_features(df)

        # Streak is passed through in years
        self.assertEqual(res.loc[0, "dividend_streak_years"], 2)
        self.assertEqual(res.loc[1, "dividend_streak_years"], 10)

        # Frequency encoding: annual=1, quarterly=4, monthly=12
        self.assertEqual(res.loc[0, "dividend_frequency_encoded"], 1)
        self.assertEqual(res.loc[1, "dividend_frequency_encoded"], 4)
        self.assertEqual(res.loc[2, "dividend_frequency_encoded"], 12)

        # Consistency score should be in [0, 100]
        assert_within_range(
            res, column="dividend_consistency_score", min_value=0.0, max_value=100.0
        )

        # Income stock flag: streak>=5 and freq>=4
        self.assertEqual(int(res.loc[0, "income_stock_flag"]), 0)
        self.assertEqual(int(res.loc[1, "income_stock_flag"]), 1)
        self.assertEqual(int(res.loc[2, "income_stock_flag"]), 1)

        # Payout ratio = dividends / net_income
        np.testing.assert_allclose(res.loc[0, "dividend_payout_ratio"], 2.0 / 10.0)
        np.testing.assert_allclose(res.loc[1, "dividend_payout_ratio"], 20.0 / 40.0)
        assert_no_inf(res)

    def test_dividend_features_missing_data(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_dividend_reliability_features,
        )

        df = pd.DataFrame(
            {
                "dividend_streak": [np.nan],
                "dividend_record_frequency": [None],
                "common_dividends_paid_ltm": [np.nan],
                "net_income": [0.0],
            }
        )

        res = engineer_dividend_reliability_features(df)

        # All derived numeric features should be finite or NaN but not inf
        assert_no_inf(res)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
