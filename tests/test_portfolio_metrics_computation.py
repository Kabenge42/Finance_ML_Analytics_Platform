"""Unit tests for portfolio metrics computation module.

TDD implementation per portfolio_optimization_enhancement_plan.md
Tests are written FIRST (RED phase) before implementation exists.

Test coverage target: ≥80% for new portfolio_metrics.py module.
"""

import unittest
import numpy as np
import pandas as pd

# Import will fail initially - this is expected in TDD RED phase
try:
    from finance_ml.ml_workflow.analytics.portfolio_metrics import (
        compute_return_1y,
        compute_expected_return,
        ensure_portfolio_metrics,
    )

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False


class TestComputeReturn1Y(unittest.TestCase):
    """Test compute_return_1y() function."""

    def setUp(self):
        """Create test DataFrame with price columns."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                "last_price": [150.0, 2800.0, 300.0, 200.0],
                "price_1y_ago": [100.0, 2500.0, 250.0, 250.0],
            }
        )

    def test_compute_return_1y_basic(self):
        """Test basic 1-year return calculation."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        result = compute_return_1y(self.df)
        self.assertIn("return_1y", result.columns)

        # Expected returns: (150/100 - 1) = 0.5, (2800/2500 - 1) = 0.12, etc.
        expected = [0.5, 0.12, 0.2, -0.2]
        np.testing.assert_array_almost_equal(result["return_1y"].values, expected, decimal=4)

    def test_compute_return_1y_missing_last_price(self):
        """Test handling when last_price column is missing."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_no_price = self.df.drop(columns=["last_price"])
        result = compute_return_1y(df_no_price)

        # Should add column with zeros or fallback value
        self.assertIn("return_1y", result.columns)
        self.assertTrue((result["return_1y"] == 0.0).all())

    def test_compute_return_1y_missing_price_1y_ago(self):
        """Test fallback when price_1y_ago is missing."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_no_historical = self.df.drop(columns=["price_1y_ago"])
        result = compute_return_1y(df_no_historical)

        self.assertIn("return_1y", result.columns)
        self.assertTrue((result["return_1y"] == 0.0).all())

    def test_compute_return_1y_with_nan(self):
        """Test handling of NaN values in price columns."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_with_nan = self.df.copy()
        df_with_nan.loc[1, "price_1y_ago"] = np.nan

        result = compute_return_1y(df_with_nan)

        # Row with NaN should have return_1y = 0 or NaN
        self.assertTrue(pd.isna(result.loc[1, "return_1y"]) or result.loc[1, "return_1y"] == 0.0)
        # Other rows should be computed
        self.assertAlmostEqual(result.loc[0, "return_1y"], 0.5, places=4)

    def test_compute_return_1y_already_exists(self):
        """Test behavior when return_1y already exists."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_with_return = self.df.copy()
        df_with_return["return_1y"] = [0.1, 0.2, 0.3, 0.4]

        result = compute_return_1y(df_with_return, overwrite=False)

        # Should preserve existing values
        np.testing.assert_array_equal(result["return_1y"].values, [0.1, 0.2, 0.3, 0.4])


class TestComputeExpectedReturn(unittest.TestCase):
    """Test compute_expected_return() function."""

    def setUp(self):
        """Create test DataFrame with prediction columns."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                "last_price": [150.0, 2800.0, 300.0, 200.0],
                "predicted_price_target": [180.0, 3080.0, 330.0, 180.0],
            }
        )

    def test_compute_expected_return_basic(self):
        """Test basic expected return calculation."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        result = compute_expected_return(self.df)
        self.assertIn("expected_return", result.columns)

        # Expected: (180/150 - 1) = 0.2, (3080/2800 - 1) = 0.1, etc.
        expected = [0.2, 0.1, 0.1, -0.1]
        np.testing.assert_array_almost_equal(result["expected_return"].values, expected, decimal=4)

    def test_compute_expected_return_missing_prediction(self):
        """Test fallback when predicted_price_target is missing."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_no_pred = self.df.drop(columns=["predicted_price_target"])

        # Should try mispricing_score as fallback
        df_no_pred["mispricing_score"] = [0.15, 0.08, 0.12, -0.05]
        result = compute_expected_return(df_no_pred)

        self.assertIn("expected_return", result.columns)
        np.testing.assert_array_almost_equal(
            result["expected_return"].values, [0.15, 0.08, 0.12, -0.05], decimal=4
        )

    def test_compute_expected_return_no_fallback(self):
        """Test behavior when neither prediction nor mispricing exists."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_no_fallback = self.df[["ticker", "last_price"]].copy()
        result = compute_expected_return(df_no_fallback)

        self.assertIn("expected_return", result.columns)
        self.assertTrue((result["expected_return"] == 0.0).all())

    def test_compute_expected_return_with_nan(self):
        """Test handling of NaN in prediction columns."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_with_nan = self.df.copy()
        df_with_nan.loc[2, "predicted_price_target"] = np.nan

        result = compute_expected_return(df_with_nan)

        # Row with NaN should get 0 or stay NaN
        self.assertTrue(
            pd.isna(result.loc[2, "expected_return"]) or result.loc[2, "expected_return"] == 0.0
        )

    def test_compute_expected_return_already_exists(self):
        """Test behavior when expected_return already exists."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_with_er = self.df.copy()
        df_with_er["expected_return"] = [0.25, 0.15, 0.18, -0.08]

        result = compute_expected_return(df_with_er, overwrite=False)

        # Should preserve existing values
        np.testing.assert_array_equal(result["expected_return"].values, [0.25, 0.15, 0.18, -0.08])


class TestEnsurePortfolioMetrics(unittest.TestCase):
    """Test ensure_portfolio_metrics() wrapper function."""

    def setUp(self):
        """Create comprehensive test DataFrame."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
                "sector": ["Technology", "Technology", "Technology", "Automotive", "Technology"],
                "last_price": [150.0, 2800.0, 300.0, 200.0, 500.0],
                "price_1y_ago": [100.0, 2500.0, 250.0, 250.0, 400.0],
                "predicted_price_target": [180.0, 3080.0, 330.0, 180.0, 600.0],
                "mispricing_score": [0.20, 0.10, 0.10, -0.10, 0.20],
            }
        )

    def test_ensure_portfolio_metrics_all_computed(self):
        """Test that all three required metrics are computed."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        result = ensure_portfolio_metrics(self.df)

        required_cols = ["expected_return", "return_1y", "mispricing_score"]
        for col in required_cols:
            self.assertIn(col, result.columns, f"Missing required column: {col}")

    def test_ensure_portfolio_metrics_preserves_data(self):
        """Test that original DataFrame columns are preserved."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        result = ensure_portfolio_metrics(self.df)

        # All original columns should be present
        for col in self.df.columns:
            self.assertIn(col, result.columns)

        # Original data should be unchanged
        pd.testing.assert_series_equal(result["ticker"], self.df["ticker"])
        pd.testing.assert_series_equal(result["last_price"], self.df["last_price"])

    def test_ensure_portfolio_metrics_minimal_dataframe(self):
        """Test with minimal DataFrame (missing most source columns)."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_minimal = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL"],
                "last_price": [150.0, 2800.0],
            }
        )

        result = ensure_portfolio_metrics(df_minimal)

        # Should have all required metrics with fallback values
        self.assertIn("expected_return", result.columns)
        self.assertIn("return_1y", result.columns)
        self.assertIn("mispricing_score", result.columns)

    def test_ensure_portfolio_metrics_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_empty = pd.DataFrame()
        result = ensure_portfolio_metrics(df_empty)

        # Should return empty DataFrame with required columns
        self.assertEqual(len(result), 0)

    def test_ensure_portfolio_metrics_idempotent(self):
        """Test that calling twice doesn't change results."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        result1 = ensure_portfolio_metrics(self.df)
        result2 = ensure_portfolio_metrics(result1)

        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_ensure_portfolio_metrics_returns_copy(self):
        """Test that original DataFrame is not modified (returns copy)."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        df_original = self.df.copy()
        result = ensure_portfolio_metrics(self.df)

        # Original should be unchanged
        pd.testing.assert_frame_equal(self.df, df_original)


class TestIntegrationPortfolioMetrics(unittest.TestCase):
    """Integration tests simulating notebook workflow."""

    def test_metrics_before_select_portfolio_candidates(self):
        """Test that metrics are computed before calling select_portfolio_candidates."""
        if not IMPORT_SUCCESS:
            self.skipTest("Module not yet implemented (TDD RED phase)")

        # Simulate Phase 9.5 output
        # Note: market_cap must be in absolute amounts (not billions) for filter_stocks_by_criteria
        # Test uses cap_unit="B", so we provide absolute values (e.g., 2.5e12 = $2.5T)
        df_phase95 = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
                "sector": ["Technology", "Technology", "Technology", "Automotive", "Technology"],
                "last_price": [150.0, 2800.0, 300.0, 200.0, 500.0],
                "price_1y_ago": [100.0, 2500.0, 250.0, 250.0, 400.0],
                "predicted_price_target": [180.0, 3080.0, 330.0, 180.0, 600.0],
                "mispricing_score": [0.20, 0.10, 0.10, -0.10, 0.20],
                "market_cap": [
                    2.5e12,
                    1.8e12,
                    2.2e12,
                    800e9,
                    1.2e12,
                ],  # Absolute amounts in dollars
            }
        )

        # Step 1: Compute metrics
        df_with_metrics = ensure_portfolio_metrics(df_phase95)

        # Step 2: Verify all required columns exist
        required_cols = ["expected_return", "return_1y", "mispricing_score"]
        for col in required_cols:
            self.assertIn(col, df_with_metrics.columns)

        # Step 3: Simulate select_portfolio_candidates (just check columns)
        from finance_ml.ml_workflow.analytics.stock_selection import select_portfolio_candidates

        # Should not raise KeyError
        try:
            result = select_portfolio_candidates(df_with_metrics, min_market_cap=1.0, top_n=3)
            self.assertGreater(len(result), 0)
        except KeyError as e:
            self.fail(f"select_portfolio_candidates raised KeyError: {e}")


if __name__ == "__main__":
    unittest.main()
