"""
Tests for mispricing functions extracted from analytics/eval.py

TDD Step 1: Testing mispricing functions (lines 40-296 of eval.py)
These tests validate the mispricing calculation and stock ranking functions
that will be extracted to analytics/mispricing.py module.

Coverage Target: 80% for mispricing module
"""

import unittest
import pandas as pd
import numpy as np


class TestCalculateMispricingScore(unittest.TestCase):
    """Tests for calculate_mispricing_score function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "predicted_price_target": [150.0, 2800.0, 350.0, 3500.0],
                "last_price": [140.0, 2900.0, 300.0, 3400.0],
                "sector": ["Technology", "Technology", "Technology", "Consumer"],
            }
        )

    def test_basic_mispricing_calculation(self):
        """Test basic mispricing score calculation."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        result = calculate_mispricing_score(self.df)

        # Check that mispricing columns are added
        self.assertIn("mispricing_pct", result.columns)
        self.assertIn("mispricing_score", result.columns)

        # AAPL: (150 - 140) / 140 = 0.0714... -> 7.14%
        expected_aapl_pct = ((150 - 140) / 140) * 100
        self.assertAlmostEqual(result.loc[0, "mispricing_pct"], expected_aapl_pct, places=2)

        # GOOGL: (2800 - 2900) / 2900 = -0.0345 -> -3.45% (overvalued)
        expected_googl_pct = ((2800 - 2900) / 2900) * 100
        self.assertAlmostEqual(result.loc[1, "mispricing_pct"], expected_googl_pct, places=2)

    def test_mispricing_score_is_decimal_form(self):
        """Test that mispricing_score is in decimal form (not percentage)."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        result = calculate_mispricing_score(self.df)

        # mispricing_score should be decimal, mispricing_pct should be percentage
        self.assertAlmostEqual(
            result.loc[0, "mispricing_score"] * 100, result.loc[0, "mispricing_pct"], places=5
        )

    def test_custom_column_names(self):
        """Test with custom column names."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        df = pd.DataFrame({"ticker": ["AAPL"], "pred": [150.0], "current": [140.0]})

        result = calculate_mispricing_score(df, predicted_col="pred", current_col="current")
        self.assertIn("mispricing_pct", result.columns)

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        df = pd.DataFrame({"ticker": ["AAPL"], "other_col": [100]})

        with self.assertRaises(ValueError) as context:
            calculate_mispricing_score(df)

        self.assertIn("Missing required columns", str(context.exception))

    def test_preserves_original_columns(self):
        """Test that original DataFrame columns are preserved."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        result = calculate_mispricing_score(self.df)

        # Check all original columns are preserved
        for col in self.df.columns:
            self.assertIn(col, result.columns)

    def test_does_not_modify_original_dataframe(self):
        """Test that original DataFrame is not modified."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        original_cols = list(self.df.columns)
        calculate_mispricing_score(self.df)

        self.assertEqual(list(self.df.columns), original_cols)


class TestCalculateMispricingFromPredictionsSchema(unittest.TestCase):
    """Tests for calculate_mispricing_from_predictions_schema function."""

    def setUp(self):
        """Set up test fixtures with standardized predictions schema."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL"],
                "y_pred": [150.0, 2800.0],
                "y_pred_calibrated": [155.0, 2750.0],
                "last_price": [140.0, 2900.0],
                "sector": ["Technology", "Technology"],
            }
        )

    def test_basic_calculation_with_y_pred(self):
        """Test calculation using y_pred column."""
        from finance_ml.ml_workflow.analytics.mispricing import (
            calculate_mispricing_from_predictions_schema,
        )

        result = calculate_mispricing_from_predictions_schema(self.df)

        self.assertIn("mispricing_pct", result.columns)
        self.assertIn("mispricing_score", result.columns)

    def test_uses_calibrated_when_requested(self):
        """Test that calibrated predictions are used when use_calibrated=True."""
        from finance_ml.ml_workflow.analytics.mispricing import (
            calculate_mispricing_from_predictions_schema,
        )

        result_raw = calculate_mispricing_from_predictions_schema(self.df, use_calibrated=False)
        result_cal = calculate_mispricing_from_predictions_schema(self.df, use_calibrated=True)

        # Results should differ since y_pred != y_pred_calibrated
        self.assertNotEqual(
            result_raw.loc[0, "mispricing_pct"], result_cal.loc[0, "mispricing_pct"]
        )

    def test_falls_back_to_y_pred_when_calibrated_missing(self):
        """Test fallback to y_pred when calibrated column doesn't exist."""
        from finance_ml.ml_workflow.analytics.mispricing import (
            calculate_mispricing_from_predictions_schema,
        )

        df_no_cal = self.df.drop(columns=["y_pred_calibrated"])
        result = calculate_mispricing_from_predictions_schema(df_no_cal, use_calibrated=True)

        # Should still work, using y_pred
        self.assertIn("mispricing_pct", result.columns)


class TestCalculateRiskAdjustedMispricing(unittest.TestCase):
    """Tests for calculate_risk_adjusted_mispricing function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "predicted_price_target": [120.0, 90.0, 110.0],
                "last_price": [100.0, 100.0, 100.0],
                "volatility": [0.20, 0.30, 0.15],
            }
        )

    def test_basic_risk_adjusted_calculation(self):
        """Test basic risk-adjusted mispricing calculation."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_risk_adjusted_mispricing

        result = calculate_risk_adjusted_mispricing(self.df)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)

        # First stock: (0.20 - 0) / 0.20 = 1.0
        expected_first = (0.20 - 0) / 0.20
        self.assertAlmostEqual(result.iloc[0], expected_first, places=2)

    def test_with_risk_free_rate(self):
        """Test calculation with non-zero risk-free rate."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_risk_adjusted_mispricing

        result = calculate_risk_adjusted_mispricing(self.df, risk_free_rate=0.05)

        # First stock: (0.20 - 0.05) / 0.20 = 0.75
        expected_first = (0.20 - 0.05) / 0.20
        self.assertAlmostEqual(result.iloc[0], expected_first, places=2)

    def test_uses_default_volatility_when_missing(self):
        """Test that default volatility is used when column is missing."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_risk_adjusted_mispricing

        df_no_vol = self.df.drop(columns=["volatility"])
        result = calculate_risk_adjusted_mispricing(df_no_vol, default_volatility=0.25)

        # Should still produce results using default volatility
        self.assertEqual(len(result), 3)

    def test_handles_zero_volatility(self):
        """Test that zero volatility is handled (clipped to minimum)."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_risk_adjusted_mispricing

        df_zero_vol = pd.DataFrame(
            {"predicted_price_target": [120.0], "last_price": [100.0], "volatility": [0.0]}
        )

        result = calculate_risk_adjusted_mispricing(df_zero_vol)

        # Should not be infinite (volatility clipped to 0.01)
        self.assertFalse(np.isinf(result.iloc[0]))


class TestCalculateRiskAdjustedMispricingFromPredictionsSchema(unittest.TestCase):
    """Tests for calculate_risk_adjusted_mispricing_from_predictions_schema function."""

    def setUp(self):
        """Set up test fixtures with predictions schema."""
        self.df = pd.DataFrame(
            {
                "y_pred": [120.0, 90.0],
                "last_price": [100.0, 100.0],
                "volatility": [0.20, 0.30],
                "pred_p10": [110.0, 80.0],
                "pred_p90": [130.0, 100.0],
            }
        )

    def test_basic_calculation(self):
        """Test basic calculation with predictions schema."""
        from finance_ml.ml_workflow.analytics.mispricing import (
            calculate_risk_adjusted_mispricing_from_predictions_schema,
        )

        result = calculate_risk_adjusted_mispricing_from_predictions_schema(self.df)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 2)

    def test_uses_quantile_interval_as_volatility_proxy(self):
        """Test using quantile interval as volatility proxy."""
        from finance_ml.ml_workflow.analytics.mispricing import (
            calculate_risk_adjusted_mispricing_from_predictions_schema,
        )

        df_no_vol = self.df.drop(columns=["volatility"])
        result = calculate_risk_adjusted_mispricing_from_predictions_schema(
            df_no_vol, use_quantile_interval=True
        )

        self.assertEqual(len(result), 2)


class TestRankUndervaluedStocks(unittest.TestCase):
    """Tests for rank_undervalued_stocks function."""

    def setUp(self):
        """Set up test fixtures with mispricing scores."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
                "mispricing_score": [0.15, -0.05, 0.25, 0.10, 0.30],
            }
        )

    def test_returns_top_n_undervalued(self):
        """Test that top N most undervalued stocks are returned."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_undervalued_stocks

        result = rank_undervalued_stocks(self.df, top_n=3)

        self.assertEqual(len(result), 3)
        # META (0.30), MSFT (0.25), AAPL (0.15) should be top 3
        self.assertEqual(result.iloc[0]["ticker"], "META")
        self.assertEqual(result.iloc[1]["ticker"], "MSFT")
        self.assertEqual(result.iloc[2]["ticker"], "AAPL")

    def test_sorted_descending(self):
        """Test that results are sorted by mispricing_score descending."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_undervalued_stocks

        result = rank_undervalued_stocks(self.df, top_n=5)

        scores = result["mispricing_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_default_top_n(self):
        """Test default top_n value."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_undervalued_stocks

        # Create larger DataFrame
        df = pd.DataFrame(
            {
                "ticker": [f"STOCK{i}" for i in range(15)],
                "mispricing_score": [i * 0.01 for i in range(15)],
            }
        )

        result = rank_undervalued_stocks(df)  # Default top_n=10
        self.assertEqual(len(result), 10)


class TestRankOvervaluedStocks(unittest.TestCase):
    """Tests for rank_overvalued_stocks function."""

    def setUp(self):
        """Set up test fixtures with mispricing scores."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
                "mispricing_score": [0.15, -0.25, 0.05, -0.10, -0.30],
            }
        )

    def test_returns_top_n_overvalued(self):
        """Test that top N most overvalued stocks are returned."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_overvalued_stocks

        result = rank_overvalued_stocks(self.df, top_n=3)

        self.assertEqual(len(result), 3)
        # META (-0.30), GOOGL (-0.25), AMZN (-0.10) should be top 3 overvalued
        self.assertEqual(result.iloc[0]["ticker"], "META")
        self.assertEqual(result.iloc[1]["ticker"], "GOOGL")
        self.assertEqual(result.iloc[2]["ticker"], "AMZN")

    def test_sorted_ascending(self):
        """Test that results are sorted by mispricing_score ascending."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_overvalued_stocks

        result = rank_overvalued_stocks(self.df, top_n=5)

        scores = result["mispricing_score"].tolist()
        self.assertEqual(scores, sorted(scores))


class TestRankStocksBySector(unittest.TestCase):
    """Tests for rank_stocks_by_sector function."""

    def setUp(self):
        """Set up test fixtures with sectors."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "JPM", "BAC", "GS"],
                "sector": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "Financials",
                    "Financials",
                    "Financials",
                ],
                "mispricing_score": [0.15, 0.25, 0.10, 0.20, -0.05, 0.30],
            }
        )

    def test_returns_dict_by_sector(self):
        """Test that result is a dict with sector keys."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=2)

        self.assertIsInstance(result, dict)
        self.assertIn("Technology", result)
        self.assertIn("Financials", result)

    def test_top_n_per_sector(self):
        """Test that each sector has at most top_n stocks."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=2)

        for sector, sector_df in result.items():
            self.assertLessEqual(len(sector_df), 2)

    def test_undervalued_order(self):
        """Test undervalued ordering (descending mispricing score)."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=3, order="undervalued")

        # Technology: GOOGL (0.25), AAPL (0.15), MSFT (0.10)
        tech_tickers = result["Technology"]["ticker"].tolist()
        self.assertEqual(tech_tickers[0], "GOOGL")

    def test_overvalued_order(self):
        """Test overvalued ordering (ascending mispricing score)."""
        from finance_ml.ml_workflow.analytics.mispricing import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=3, order="overvalued")

        # Financials: BAC (-0.05), JPM (0.20), GS (0.30)
        fin_tickers = result["Financials"]["ticker"].tolist()
        self.assertEqual(fin_tickers[0], "BAC")


class TestMispricingEdgeCases(unittest.TestCase):
    """Tests for edge cases in mispricing calculations."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        df = pd.DataFrame(columns=["predicted_price_target", "last_price"])
        result = calculate_mispricing_score(df)

        self.assertEqual(len(result), 0)

    def test_nan_values(self):
        """Test handling of NaN values in calculations."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        df = pd.DataFrame(
            {"predicted_price_target": [150.0, np.nan, 110.0], "last_price": [140.0, 100.0, np.nan]}
        )

        result = calculate_mispricing_score(df)

        # First row should be valid
        self.assertFalse(np.isnan(result.loc[0, "mispricing_pct"]))
        # Rows with NaN should produce NaN mispricing
        self.assertTrue(np.isnan(result.loc[1, "mispricing_pct"]))
        self.assertTrue(np.isnan(result.loc[2, "mispricing_pct"]))

    def test_zero_current_price(self):
        """Test handling of zero current price (division by zero)."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        df = pd.DataFrame({"predicted_price_target": [150.0], "last_price": [0.0]})

        result = calculate_mispricing_score(df)

        # Should produce inf or NaN, not crash
        self.assertTrue(
            np.isinf(result.loc[0, "mispricing_pct"]) or np.isnan(result.loc[0, "mispricing_pct"])
        )

    def test_negative_prices(self):
        """Test handling of negative prices (shouldn't happen but test robustness)."""
        from finance_ml.ml_workflow.analytics.mispricing import calculate_mispricing_score

        df = pd.DataFrame(
            {
                "predicted_price_target": [150.0],
                "last_price": [-100.0],  # Invalid but test robustness
            }
        )

        # Should not crash
        result = calculate_mispricing_score(df)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
