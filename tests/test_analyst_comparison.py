"""
Tests for Phase 9.8: Prediction vs. Analyst Comparison Analytics
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from finance_ml.analyst_comparison import (
    compare_prediction_vs_analyst_targets,
    calculate_agreement_rate,
    calculate_directional_accuracy,
    analyze_systematic_bias,
    identify_disagreement_opportunities,
    segment_comparison_by_attribute,
    generate_prediction_analyst_excel_report,
    PredictionAnalystAnalytics,
)


class TestPredictionAnalystComparison(unittest.TestCase):
    """Test prediction vs analyst comparison functions."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        self.sample_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "JPM", "BAC"],
                "sector": ["Technology", "Technology", "Technology", "Financials", "Financials"],
                "region": ["US", "US", "US", "US", "US"],
                "last_price": [150.0, 300.0, 120.0, 140.0, 30.0],
                "predicted_price_target": [165.0, 330.0, 135.0, 155.0, 35.0],  # Model predictions
                "price_target": [160.0, 320.0, 130.0, 150.0, 33.0],  # Analyst targets
            }
        )

    def test_compare_prediction_vs_analyst_targets(self):
        """Test comparison calculation."""
        result = compare_prediction_vs_analyst_targets(self.sample_df)

        # Check new columns exist
        self.assertIn("model_analyst_diff", result.columns)
        self.assertIn("model_analyst_diff_pct", result.columns)
        self.assertIn("agreement_direction", result.columns)
        self.assertIn("model_direction", result.columns)
        self.assertIn("analyst_direction", result.columns)

        # Verify calculations for first row (AAPL)
        self.assertAlmostEqual(result.loc[0, "model_analyst_diff"], 5.0)  # 165 - 160
        self.assertAlmostEqual(
            result.loc[0, "model_analyst_diff_pct"], 3.125, places=2
        )  # 5/160*100

        # Verify direction calculations
        self.assertEqual(result.loc[0, "model_direction"], "up")  # 165 > 150
        self.assertEqual(result.loc[0, "analyst_direction"], "up")  # 160 > 150
        self.assertTrue(result.loc[0, "agreement_direction"])  # both up

    def test_compare_missing_columns(self):
        """Test error handling for missing columns."""
        incomplete_df = pd.DataFrame({"ticker": ["AAPL"], "last_price": [150.0]})

        with self.assertRaises(ValueError) as context:
            compare_prediction_vs_analyst_targets(incomplete_df)

        self.assertIn("Missing required columns", str(context.exception))

    def test_calculate_agreement_rate(self):
        """Test agreement rate calculation."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)
        metrics = calculate_agreement_rate(comparison_df)

        # Check return structure
        self.assertIn("agreement_rate", metrics)
        self.assertIn("same_direction_count", metrics)
        self.assertIn("total_count", metrics)

        # Verify types
        self.assertIsInstance(metrics["agreement_rate"], float)
        self.assertIsInstance(metrics["same_direction_count"], int)
        self.assertIsInstance(metrics["total_count"], int)

        # Verify range
        self.assertTrue(0 <= metrics["agreement_rate"] <= 1)
        self.assertEqual(metrics["total_count"], 5)

    def test_calculate_agreement_rate_empty(self):
        """Test agreement rate with empty dataframe."""
        empty_df = pd.DataFrame({"agreement_direction": []})

        metrics = calculate_agreement_rate(empty_df)
        self.assertEqual(metrics["agreement_rate"], 0)
        self.assertEqual(metrics["total_count"], 0)

    def test_calculate_directional_accuracy(self):
        """Test directional accuracy calculation."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)
        metrics = calculate_directional_accuracy(comparison_df)

        # Check return structure
        self.assertIn("accuracy", metrics)
        self.assertIn("correct_predictions", metrics)
        self.assertIn("total_predictions", metrics)

        # Verify types
        self.assertIsInstance(metrics["accuracy"], float)
        self.assertIsInstance(metrics["correct_predictions"], int)
        self.assertIsInstance(metrics["total_predictions"], int)

        # Verify range
        self.assertTrue(0 <= metrics["accuracy"] <= 1)

    def test_analyze_systematic_bias(self):
        """Test systematic bias analysis."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)
        bias_metrics = analyze_systematic_bias(comparison_df)

        # Check return structure
        self.assertIn("mean_model_bias", bias_metrics)
        self.assertIn("median_model_bias", bias_metrics)
        self.assertIn("bias_direction", bias_metrics)

        # Verify types
        self.assertIsInstance(bias_metrics["mean_model_bias"], float)
        self.assertIsInstance(bias_metrics["median_model_bias"], float)
        self.assertIsInstance(bias_metrics["bias_direction"], str)

        # Verify bias direction is one of expected values
        self.assertIn(bias_metrics["bias_direction"], ["bullish", "bearish", "neutral"])

    def test_analyze_systematic_bias_bullish(self):
        """Test bias detection for bullish model."""
        df = pd.DataFrame(
            {
                "predicted_price_target": [110.0, 120.0, 130.0],
                "price_target": [100.0, 110.0, 120.0],
                "last_price": [100.0, 110.0, 120.0],
            }
        )
        comparison_df = compare_prediction_vs_analyst_targets(df)
        bias_metrics = analyze_systematic_bias(comparison_df)

        self.assertEqual(bias_metrics["bias_direction"], "bullish")
        self.assertGreater(bias_metrics["mean_model_bias"], 0)

    def test_analyze_systematic_bias_bearish(self):
        """Test bias detection for bearish model."""
        df = pd.DataFrame(
            {
                "predicted_price_target": [90.0, 100.0, 110.0],
                "price_target": [100.0, 110.0, 120.0],
                "last_price": [100.0, 110.0, 120.0],
            }
        )
        comparison_df = compare_prediction_vs_analyst_targets(df)
        bias_metrics = analyze_systematic_bias(comparison_df)

        self.assertEqual(bias_metrics["bias_direction"], "bearish")
        self.assertLess(bias_metrics["mean_model_bias"], 0)

    def test_identify_disagreement_opportunities(self):
        """Test disagreement identification."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)
        opportunities = identify_disagreement_opportunities(comparison_df, threshold_pct=2.0)

        # Should find stocks with >2% difference
        self.assertGreater(len(opportunities), 0)

        # Verify all opportunities exceed threshold
        for _, row in opportunities.iterrows():
            self.assertGreater(abs(row["model_analyst_diff_pct"]), 2.0)

        # Verify sorted by absolute difference (descending)
        abs_diffs = opportunities["model_analyst_diff_pct"].abs().values
        self.assertTrue(all(abs_diffs[i] >= abs_diffs[i + 1] for i in range(len(abs_diffs) - 1)))

    def test_identify_disagreement_opportunities_high_threshold(self):
        """Test with high threshold returns fewer results."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)

        low_threshold = identify_disagreement_opportunities(comparison_df, threshold_pct=1.0)
        high_threshold = identify_disagreement_opportunities(comparison_df, threshold_pct=10.0)

        self.assertGreaterEqual(len(low_threshold), len(high_threshold))

    def test_segment_comparison_by_attribute_sector(self):
        """Test segmentation by sector."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)
        results = segment_comparison_by_attribute(comparison_df, "sector")

        # Should have two sectors
        self.assertIn("Technology", results)
        self.assertIn("Financials", results)

        # Verify structure of each segment
        for sector_name, metrics in results.items():
            self.assertIn("count", metrics)
            self.assertIn("agreement_rate", metrics)
            self.assertIn("avg_model_analyst_diff", metrics)

            self.assertIsInstance(metrics["count"], int)
            self.assertIsInstance(metrics["agreement_rate"], float)
            self.assertIsInstance(metrics["avg_model_analyst_diff"], float)

            self.assertTrue(0 <= metrics["agreement_rate"] <= 1)

    def test_segment_comparison_by_attribute_region(self):
        """Test segmentation by region."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)
        results = segment_comparison_by_attribute(comparison_df, "region")

        # Should have one region
        self.assertIn("US", results)
        self.assertEqual(results["US"]["count"], 5)

    def test_segment_comparison_missing_column(self):
        """Test segmentation with missing column."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)
        results = segment_comparison_by_attribute(comparison_df, "nonexistent_column")

        # Should return empty dict
        self.assertEqual(results, {})

    def test_generate_excel_report(self):
        """Test Excel report generation."""
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.xlsx"
            generate_prediction_analyst_excel_report(
                comparison_df, output_path, top_n_opportunities=10
            )

            # Verify file created
            self.assertTrue(output_path.exists())

            # Verify file is not empty
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_generate_excel_report_xlsxwriter_missing(self):
        """Test error handling when xlsxwriter is not available."""
        # This test assumes xlsxwriter is installed
        # In real scenario, we'd mock the import
        comparison_df = compare_prediction_vs_analyst_targets(self.sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.xlsx"

            # Should not raise error if xlsxwriter is available
            try:
                generate_prediction_analyst_excel_report(
                    comparison_df, output_path, top_n_opportunities=5
                )
            except ImportError as e:
                self.assertIn("xlsxwriter", str(e).lower())


class TestPredictionAnalystAnalyticsClass(unittest.TestCase):
    """Test PredictionAnalystAnalytics class."""

    def setUp(self):
        """Create sample data and config for testing."""
        np.random.seed(42)
        self.sample_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "XOM"],
                "sector": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "Financials",
                    "Financials",
                    "Energy",
                ],
                "region": ["US", "US", "US", "US", "US", "US"],
                "last_price": [150.0, 300.0, 120.0, 140.0, 30.0, 80.0],
                "predicted_price_target": [165.0, 330.0, 135.0, 155.0, 35.0, 95.0],
                "price_target": [160.0, 320.0, 130.0, 150.0, 33.0, 90.0],
            }
        )

        # Create mock config
        from finance_ml import FinanceMLConfig

        self.config = FinanceMLConfig(output_dir=Path(tempfile.gettempdir()))

    def test_class_initialization(self):
        """Test class initialization."""
        analytics = PredictionAnalystAnalytics(self.sample_df, self.config)

        self.assertIsNotNone(analytics.all_stocks_featured)
        self.assertEqual(analytics.config, self.config)
        self.assertIsNone(analytics.comparison_df)
        self.assertIsNone(analytics.agreement_metrics)
        self.assertIsNone(analytics.directional_metrics)
        self.assertIsNone(analytics.bias_metrics)
        self.assertIsNone(analytics.disagreements)

    def test_prepare_analyst_data(self):
        """Test data preparation."""
        analytics = PredictionAnalystAnalytics(self.sample_df, self.config)
        analytics.prepare_analyst_data()

        # Should have all 6 stocks (no missing values)
        self.assertEqual(len(analytics.all_stocks_featured), 6)

    def test_prepare_analyst_data_with_missing(self):
        """Test data preparation with missing values."""
        df_with_missing = self.sample_df.copy()
        df_with_missing.loc[0, "price_target"] = np.nan

        analytics = PredictionAnalystAnalytics(df_with_missing, self.config)
        analytics.prepare_analyst_data()

        # Should drop row with missing value
        self.assertEqual(len(analytics.all_stocks_featured), 5)

    def test_prepare_analyst_data_missing_column(self):
        """Test error when required column is missing."""
        incomplete_df = pd.DataFrame({"ticker": ["AAPL"], "last_price": [150.0]})

        analytics = PredictionAnalystAnalytics(incomplete_df, self.config)

        with self.assertRaises(ValueError) as context:
            analytics.prepare_analyst_data()

        self.assertIn("Missing required columns", str(context.exception))

    def test_perform_comparison(self):
        """Test comparison execution."""
        analytics = PredictionAnalystAnalytics(self.sample_df, self.config)
        analytics.prepare_analyst_data()
        analytics.perform_comparison()

        self.assertIsNotNone(analytics.comparison_df)
        self.assertIn("model_analyst_diff", analytics.comparison_df.columns)
        self.assertIn("model_analyst_diff_pct", analytics.comparison_df.columns)

    def test_analyze_agreement(self):
        """Test agreement analysis."""
        analytics = PredictionAnalystAnalytics(self.sample_df, self.config)
        analytics.prepare_analyst_data()
        analytics.perform_comparison()
        analytics.analyze_agreement()

        self.assertIsNotNone(analytics.agreement_metrics)
        self.assertIsNotNone(analytics.directional_metrics)
        self.assertIsNotNone(analytics.bias_metrics)

        self.assertIn("agreement_rate", analytics.agreement_metrics)
        self.assertIn("accuracy", analytics.directional_metrics)
        self.assertIn("bias_direction", analytics.bias_metrics)

    def test_identify_opportunities(self):
        """Test opportunity identification."""
        analytics = PredictionAnalystAnalytics(self.sample_df, self.config)
        analytics.prepare_analyst_data()
        analytics.perform_comparison()

        result = analytics.identify_opportunities(threshold_pct=2.0)

        self.assertIsNotNone(analytics.disagreements)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_segment_analysis(self):
        """Test segmented analysis (no exceptions raised)."""
        analytics = PredictionAnalystAnalytics(self.sample_df, self.config)
        analytics.prepare_analyst_data()
        analytics.perform_comparison()

        # Should not raise exception
        analytics.segment_analysis()

    def test_run_full_analysis(self):
        """Test full analysis pipeline."""
        analytics = PredictionAnalystAnalytics(self.sample_df, self.config)

        # Suppress output during test
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            results = analytics.run_full_analysis(disagreement_threshold=5.0, top_n=10)
        finally:
            sys.stdout = old_stdout

        # Verify results structure
        self.assertIsNotNone(results)
        self.assertIn("comparison_df", results)
        self.assertIn("agreement_metrics", results)
        self.assertIn("directional_metrics", results)
        self.assertIn("bias_metrics", results)
        self.assertIn("disagreements", results)

        # Verify all analytics were computed
        self.assertIsNotNone(results["comparison_df"])
        self.assertIsNotNone(results["agreement_metrics"])
        self.assertIsNotNone(results["directional_metrics"])
        self.assertIsNotNone(results["bias_metrics"])

    def test_run_full_analysis_with_error(self):
        """Test error handling in full analysis."""
        # Create invalid dataframe
        invalid_df = pd.DataFrame({"ticker": ["AAPL"]})

        analytics = PredictionAnalystAnalytics(invalid_df, self.config)

        # Suppress output during test
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            results = analytics.run_full_analysis()
        finally:
            sys.stdout = old_stdout

        # Should return None on error
        self.assertIsNone(results)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_all_agreements(self):
        """Test with all predictions agreeing with analysts."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "predicted_price_target": [110.0, 120.0, 130.0],
                "price_target": [110.0, 120.0, 130.0],
                "last_price": [100.0, 110.0, 120.0],
            }
        )

        comparison_df = compare_prediction_vs_analyst_targets(df)
        metrics = calculate_agreement_rate(comparison_df)

        self.assertEqual(metrics["agreement_rate"], 1.0)
        self.assertEqual(metrics["same_direction_count"], 3)

    def test_no_disagreements_high_threshold(self):
        """Test disagreement identification with no opportunities."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "predicted_price_target": [101.0, 102.0],
                "price_target": [100.0, 101.0],
                "last_price": [100.0, 100.0],
            }
        )

        comparison_df = compare_prediction_vs_analyst_targets(df)
        opportunities = identify_disagreement_opportunities(comparison_df, threshold_pct=50.0)

        self.assertEqual(len(opportunities), 0)

    def test_mixed_directions(self):
        """Test with mixed up/down directions."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "predicted_price_target": [110.0, 90.0, 110.0, 90.0],
                "price_target": [110.0, 90.0, 90.0, 110.0],
                "last_price": [100.0, 100.0, 100.0, 100.0],
            }
        )

        comparison_df = compare_prediction_vs_analyst_targets(df)
        metrics = calculate_agreement_rate(comparison_df)

        # A and B agree, C and D disagree
        self.assertEqual(metrics["same_direction_count"], 2)
        self.assertEqual(metrics["agreement_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
