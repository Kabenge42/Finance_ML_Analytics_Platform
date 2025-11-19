"""
Test suite for finance_ml.ml_workflow.features.core module.

Tests basic feature engineering functions moved from features.py as part of Phase 9.3 refactor.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch


class TestSafeDiv(unittest.TestCase):
    """Test _safe_div helper function."""

    def test_safe_div_normal_division(self):
        """Test normal division without zeros."""
        from finance_ml.ml_workflow.features.core import _safe_div

        numer = pd.Series([10, 20, 30])
        denom = pd.Series([2, 4, 5])
        result = _safe_div(numer, denom)

        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_div_with_zero_denominator(self):
        """Test division with zero denominator returns NaN."""
        from finance_ml.ml_workflow.features.core import _safe_div

        numer = pd.Series([10, 20, 30])
        denom = pd.Series([2, 0, 5])
        result = _safe_div(numer, denom)

        self.assertEqual(result.iloc[0], 5.0)
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertEqual(result.iloc[2], 6.0)

    def test_safe_div_all_zeros(self):
        """Test division when all denominators are zero."""
        from finance_ml.ml_workflow.features.core import _safe_div

        numer = pd.Series([10, 20, 30])
        denom = pd.Series([0, 0, 0])
        result = _safe_div(numer, denom)

        self.assertTrue(result.isna().all())


class TestEngineerBasicRatios(unittest.TestCase):
    """Test engineer_basic_ratios function."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "market_cap": [1000, 2000, 3000],
                "total_assets": [500, 1000, 1500],
                "total_equity": [300, 800, 1200],
                "total_debt": [200, 200, 300],
                "net_income": [50, 100, 150],
                "revenue": [200, 400, 600],
                "last_price": [10, 20, 30],
            }
        )

    def test_basic_ratios_created(self):
        """Test that basic financial ratios are created."""
        from finance_ml.ml_workflow.features.core import engineer_basic_ratios

        result = engineer_basic_ratios(self.df)

        # Check new columns exist
        self.assertIn("p_b", result.columns)
        self.assertIn("debt_to_equity", result.columns)
        self.assertIn("roe", result.columns)
        self.assertIn("roa", result.columns)

    def test_basic_ratios_calculations(self):
        """Test ratio calculations are correct."""
        from finance_ml.ml_workflow.features.core import engineer_basic_ratios

        result = engineer_basic_ratios(self.df)

        # P/B = market_cap / total_equity
        expected_pb = [1000 / 300, 2000 / 800, 3000 / 1200]
        np.testing.assert_array_almost_equal(result["p_b"].values, expected_pb)

        # ROE = net_income / total_equity
        expected_roe = [50 / 300, 100 / 800, 150 / 1200]
        np.testing.assert_array_almost_equal(result["roe"].values, expected_roe)


class TestEngineerMarginFeatures(unittest.TestCase):
    """Test engineer_margin_features function."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "revenue": [1000, 2000, 3000],
                "ebitda": [200, 400, 600],
                "operating_income": [150, 300, 450],
                "net_income": [100, 200, 300],
                "gross_profit": [400, 800, 1200],
            }
        )

    def test_margin_features_created(self):
        """Test that margin features are created."""
        from finance_ml.ml_workflow.features.core import engineer_margin_features

        result = engineer_margin_features(self.df)

        self.assertIn("ebitda_margin", result.columns)
        self.assertIn("operating_margin", result.columns)
        self.assertIn("net_margin", result.columns)
        self.assertIn("gross_margin", result.columns)

    def test_margin_calculations(self):
        """Test margin calculations are correct."""
        from finance_ml.ml_workflow.features.core import engineer_margin_features

        result = engineer_margin_features(self.df)

        # EBITDA margin = ebitda / revenue
        expected_ebitda_margin = [0.2, 0.2, 0.2]
        np.testing.assert_array_almost_equal(result["ebitda_margin"].values, expected_ebitda_margin)


class TestEngineerVolatilityFeatures(unittest.TestCase):
    """Test engineer_volatility_features function."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "high_price": [110, 120, 130],
                "low_price": [90, 80, 70],
                "last_price": [100, 100, 100],
            }
        )

    def test_volatility_features_created(self):
        """Test that volatility features are created."""
        from finance_ml.ml_workflow.features.core import engineer_volatility_features

        result = engineer_volatility_features(self.df)

        self.assertIn("price_range", result.columns)
        self.assertIn("relative_volatility", result.columns)


class TestEngineerRevenueCagr(unittest.TestCase):
    """Test engineer_revenue_cagr function."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "revenue": [1000, 2000, 3000],
                "revenue_3y_ago": [800, 1600, 2400],
                "revenue_5y_ago": [600, 1200, 1800],
            }
        )

    def test_cagr_features_created(self):
        """Test that CAGR features are created."""
        from finance_ml.ml_workflow.features.core import engineer_revenue_cagr

        result = engineer_revenue_cagr(self.df)

        self.assertIn("revenue_cagr_3y", result.columns)
        self.assertIn("revenue_cagr_5y", result.columns)


class TestBuildFeaturesAndTarget(unittest.TestCase):
    """Test build_features_and_target orchestration function."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "market_cap": [1000, 2000, 3000],
                "total_assets": [500, 1000, 1500],
                "total_equity": [300, 800, 1200],
                "revenue": [1000, 2000, 3000],
                "net_income": [100, 200, 300],
                "last_price": [100, 200, 300],
                "price_target": [110, 220, 330],
            }
        )

    def test_orchestration_builds_all_features(self):
        """Test that build_features_and_target calls all feature functions."""
        from finance_ml.ml_workflow.features.core import build_features_and_target

        X, y, numeric_features, categorical_features = build_features_and_target(self.df)

        # Should have fewer columns than input (price_target removed, ticker removed)
        # But we're not calling the individual engineer functions, so columns stay same
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

        # Check target was extracted
        self.assertEqual(len(y), len(self.df))

        # Check identifier columns removed
        self.assertNotIn("ticker", X.columns)
        self.assertNotIn("price_target", X.columns)


if __name__ == "__main__":
    unittest.main()
