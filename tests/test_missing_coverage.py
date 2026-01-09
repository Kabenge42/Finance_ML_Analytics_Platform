"""Tests for missing_coverage feature engineering module."""

import unittest

import numpy as np
import pandas as pd

from finance_ml.features.advanced.missing_coverage import (
    engineer_missing_dividend_features,
    engineer_value_score,
    engineer_all_missing_features,
)


class TestMissingCoverageFeatures(unittest.TestCase):
    """Test cases for missing Phase 9.3 coverage features."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "eps_adj_ltm": [6.5, 11.0, 5.8],
                "dps_ltm": [0.96, 2.72, 0.0],
                "dps_3fy": [0.75, 2.0, 0.0],
                "dps_5fy": [0.50, 1.5, 0.0],
                "p_e_ltm": [28.5, 35.0, 25.0],
                "p_b_ltm": [45.0, 12.0, 6.0],
                "ev_ebitda_ltm": [22.0, 25.0, 18.0],
                "div_yield_ltm": [0.5, 0.8, 0.0],
                "fcf_yield_ltm": [3.5, 2.8, 4.2],
            }
        )

    def test_engineer_missing_dividend_features(self):
        """Test dividend coverage and growth features."""
        result = engineer_missing_dividend_features(self.df)

        # Check dividend_coverage_ratio is computed
        self.assertIn("dividend_coverage_ratio", result.columns)
        # AAPL: 6.5 / 0.96 ≈ 6.77
        self.assertAlmostEqual(result.loc[0, "dividend_coverage_ratio"], 6.77, places=1)

        # Check dividend_growth_3y is computed
        self.assertIn("dividend_growth_3y", result.columns)
        # AAPL: (0.96/0.75)^(1/3) - 1 ≈ 8.6%
        self.assertGreater(result.loc[0, "dividend_growth_3y"], 0)

        # Check dividend_growth_5y is computed
        self.assertIn("dividend_growth_5y", result.columns)
        # AAPL: (0.96/0.50)^(1/5) - 1 ≈ 13.9%
        self.assertGreater(result.loc[0, "dividend_growth_5y"], 0)

    def test_engineer_value_score(self):
        """Test value score composite calculation."""
        result = engineer_value_score(self.df)

        # Check value_score is computed
        self.assertIn("value_score", result.columns)

        # Value scores should be between 0 and 100
        self.assertTrue(all(result["value_score"].dropna() >= 0))
        self.assertTrue(all(result["value_score"].dropna() <= 100))

    def test_engineer_all_missing_features(self):
        """Test all missing features are added."""
        result = engineer_all_missing_features(self.df)

        # Check all expected features are present
        expected_features = [
            "dividend_coverage_ratio",
            "dividend_growth_3y",
            "dividend_growth_5y",
            "value_score",
        ]
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing feature: {feature}")

    def test_handles_missing_columns_gracefully(self):
        """Test that functions handle missing input columns gracefully."""
        # DataFrame with no dividend columns
        df_minimal = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "last_price": [100.0],
            }
        )

        # Should not raise errors
        result = engineer_all_missing_features(df_minimal)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
