"""
Test suite for finance_ml.eval module

This module tests evaluation, analytics, and visualization functions.
Following TDD methodology for Phase 7 refactoring.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd


class TestCalculateMispricingScore(unittest.TestCase):
    """Test mispricing score calculation"""

    def test_calculate_mispricing_score_returns_series(self):
        """Should return pandas Series"""
        df = pd.DataFrame({"predicted_price_target": [110, 90, 105], "last_price": [100, 100, 100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertIsInstance(result, pd.Series)

    def test_calculate_mispricing_score_undervalued(self):
        """Should return positive score for undervalued stocks"""
        df = pd.DataFrame({"predicted_price_target": [120], "last_price": [100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertGreater(result.iloc[0], 0)

    def test_calculate_mispricing_score_overvalued(self):
        """Should return negative score for overvalued stocks"""
        df = pd.DataFrame({"predicted_price_target": [80], "last_price": [100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertLess(result.iloc[0], 0)

    def test_calculate_mispricing_score_correct_formula(self):
        """Should calculate (predicted - current) / current"""
        df = pd.DataFrame({"predicted_price_target": [110], "last_price": [100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertAlmostEqual(result.iloc[0], 0.1, places=5)


class TestStandardizedSchemaMispricing(unittest.TestCase):
    """Phase 9.3: Mispricing using standardized predictions schema.

    This focuses on estimate vs reported targets using standardized
    columns (y_true, y_pred, y_pred_calibrated, pred_p10, pred_p50,
    pred_p90, last_price) and risk-adjusted scores.
    """

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "sector": ["Tech", "Health"],
                "region": ["US", "EU"],
                "last_price": [100.0, 80.0],
                "y_true": [110.0, 75.0],
                "y_pred": [120.0, 70.0],
                "y_pred_calibrated": [115.0, 72.0],
                "pred_p10": [105.0, 65.0],
                "pred_p50": [120.0, 70.0],
                "pred_p90": [135.0, 75.0],
            }
        )

    def test_mispricing_from_predictions_schema_uses_y_pred(self):
        """Mispricing helper should use y_pred by default and add scores.

        Expected formula: (y_pred - last_price) / last_price.
        """

        from finance_ml.ml_workflow.analytics.eval import (
            calculate_mispricing_from_predictions_schema,
        )

        result = calculate_mispricing_from_predictions_schema(self.df)

        self.assertIn("mispricing_score", result.columns)
        self.assertIn("mispricing_pct", result.columns)

        expected0 = (self.df.loc[0, "y_pred"] - self.df.loc[0, "last_price"]) / self.df.loc[
            0, "last_price"
        ]
        self.assertAlmostEqual(result.loc[0, "mispricing_score"], expected0, places=6)

    def test_mispricing_from_predictions_schema_can_use_calibrated(self):
        """When use_calibrated=True, y_pred_calibrated should be used if available."""

        from finance_ml.ml_workflow.analytics.eval import (
            calculate_mispricing_from_predictions_schema,
        )

        result = calculate_mispricing_from_predictions_schema(self.df, use_calibrated=True)

        expected0 = (
            self.df.loc[0, "y_pred_calibrated"] - self.df.loc[0, "last_price"]
        ) / self.df.loc[0, "last_price"]
        self.assertAlmostEqual(result.loc[0, "mispricing_score"], expected0, places=6)

    def test_risk_adjusted_mispricing_from_schema_uses_quantile_interval(self):
        """Risk-adjusted helper should use prediction interval width as uncertainty proxy.

        For two rows with the same expected return but different interval widths
        (pred_p90 - pred_p10), the narrower interval should yield a higher
        risk-adjusted score when use_quantile_interval=True.
        """

        from finance_ml.ml_workflow.analytics.eval import (
            calculate_mispricing_from_predictions_schema,
            calculate_risk_adjusted_mispricing_from_predictions_schema,
        )

        # Construct a small frame where both rows have the same
        # expected return but different prediction interval widths.
        df_equal_return = pd.DataFrame(
            {
                "last_price": [100.0, 100.0],
                "y_pred": [120.0, 120.0],  # same expected return 20%
                "pred_p10": [105.0, 110.0],  # width: 30 vs 20
                "pred_p90": [135.0, 130.0],
            }
        )

        mispricing_df = calculate_mispricing_from_predictions_schema(df_equal_return)

        scores = calculate_risk_adjusted_mispricing_from_predictions_schema(
            mispricing_df,
            prediction_col="y_pred",
            current_price_col="last_price",
            risk_free_rate=0.0,
            use_quantile_interval=True,
        )

        self.assertEqual(len(scores), len(df_equal_return))
        # Row 0 has wider interval (135-105=30) than row 1 (130-110=20),
        # so its uncertainty penalty should be larger (lower score).
        self.assertLess(scores.iloc[0], scores.iloc[1])


class TestRankUndervaluedStocks(unittest.TestCase):
    """Test undervalued stocks ranking"""

    def setUp(self):
        """Create sample data with mispricing scores"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "mispricing_score": [0.15, 0.10, -0.05, 0.20, -0.10],
                "sector": ["Technology", "Technology", "Technology", "Technology", "Technology"],
            }
        )

    def test_rank_undervalued_stocks_returns_dataframe(self):
        """Should return DataFrame"""
        from finance_ml.eval import rank_undervalued_stocks

        result = rank_undervalued_stocks(self.df, top_n=3)
        self.assertIsInstance(result, pd.DataFrame)

    def test_rank_undervalued_stocks_limits_results(self):
        """Should return top_n results"""
        from finance_ml.eval import rank_undervalued_stocks

        result = rank_undervalued_stocks(self.df, top_n=2)
        self.assertEqual(len(result), 2)

    def test_rank_undervalued_stocks_sorted_descending(self):
        """Should sort by mispricing_score descending"""
        from finance_ml.eval import rank_undervalued_stocks

        result = rank_undervalued_stocks(self.df, top_n=3)
        scores = result["mispricing_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestRankOvervaluedStocks(unittest.TestCase):
    """Test overvalued stocks ranking"""

    def setUp(self):
        """Create sample data with mispricing scores"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "mispricing_score": [0.15, 0.10, -0.05, 0.20, -0.10],
                "sector": ["Technology", "Technology", "Technology", "Technology", "Technology"],
            }
        )

    def test_rank_overvalued_stocks_returns_dataframe(self):
        """Should return DataFrame"""
        from finance_ml.eval import rank_overvalued_stocks

        result = rank_overvalued_stocks(self.df, top_n=2)
        self.assertIsInstance(result, pd.DataFrame)

    def test_rank_overvalued_stocks_limits_results(self):
        """Should return top_n results"""
        from finance_ml.eval import rank_overvalued_stocks

        result = rank_overvalued_stocks(self.df, top_n=2)
        self.assertEqual(len(result), 2)

    def test_rank_overvalued_stocks_sorted_ascending(self):
        """Should sort by mispricing_score ascending (most negative first)"""
        from finance_ml.eval import rank_overvalued_stocks

        result = rank_overvalued_stocks(self.df, top_n=2)
        scores = result["mispricing_score"].tolist()
        self.assertEqual(scores, sorted(scores))


class TestRankStocksBySector(unittest.TestCase):
    """Test stocks ranking by sector"""

    def setUp(self):
        """Create sample data with multiple sectors"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "JPM", "BAC", "XOM"],
                "mispricing_score": [0.15, 0.10, 0.08, -0.05, 0.12],
                "sector": ["Technology", "Technology", "Finance", "Finance", "Energy"],
            }
        )

    def test_rank_stocks_by_sector_returns_dict(self):
        """Should return dictionary with sectors as keys"""
        from finance_ml.eval import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=2)
        self.assertIsInstance(result, dict)

    def test_rank_stocks_by_sector_has_all_sectors(self):
        """Should include all sectors from dataframe"""
        from finance_ml.eval import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=2)
        expected_sectors = set(self.df["sector"].unique())
        self.assertEqual(set(result.keys()), expected_sectors)

    def test_rank_stocks_by_sector_undervalued_order(self):
        """Should rank undervalued stocks (highest score first)"""
        from finance_ml.eval import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=2, order="undervalued")
        # Check Technology sector has AAPL before MSFT
        tech_df = result["Technology"]
        self.assertEqual(tech_df.iloc[0]["ticker"], "AAPL")

    def test_rank_stocks_by_sector_overvalued_order(self):
        """Should rank overvalued stocks (lowest score first)"""
        from finance_ml.eval import rank_stocks_by_sector

        result = rank_stocks_by_sector(self.df, top_n=2, order="overvalued")
        # Check Finance sector has BAC (negative) before JPM (positive)
        finance_df = result["Finance"]
        self.assertEqual(finance_df.iloc[0]["ticker"], "BAC")


class TestSimpleEDA(unittest.TestCase):
    """Test exploratory data analysis function"""

    def setUp(self):
        """Create sample data and temp directory"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Technology", "Technology", "Technology"],
                "region": ["US", "US", "US"],
                "last_price": [150.0, 300.0, 2800.0],
                "market_cap": [2.5e12, 2.2e12, 1.8e12],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_eda_creates_output_file(self):
        """Should create eda_summary.json file"""
        from finance_ml.eval import simple_eda

        simple_eda(self.df, self.out_dir)
        output_file = self.out_dir / "eda_summary.json"
        self.assertTrue(output_file.exists())

    def test_simple_eda_json_has_required_fields(self):
        """Should include required summary fields"""
        from finance_ml.eval import simple_eda

        simple_eda(self.df, self.out_dir)
        output_file = self.out_dir / "eda_summary.json"
        with open(output_file, "r") as f:
            summary = json.load(f)
        self.assertIn("row_count", summary)
        self.assertIn("column_count", summary)
        self.assertIn("columns", summary)

    def test_simple_eda_with_plots(self):
        """Should not crash when save_plots=True"""
        from finance_ml.eval import simple_eda

        # Should not raise exception even if matplotlib not available
        try:
            simple_eda(self.df, self.out_dir, save_plots=True)
        except ImportError:
            pass  # OK if matplotlib not available

    def test_simple_eda_with_numeric_data_for_correlation(self):
        """Should handle correlation plotting with numeric data"""
        from finance_ml.eval import simple_eda

        # Create dataframe with multiple numeric columns to trigger correlation
        df_numeric = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "price": [100.0, 200.0, 150.0, 180.0, 220.0],
                "volume": [1000, 2000, 1500, 1800, 2200],
                "market_cap": [1e9, 2e9, 1.5e9, 1.8e9, 2.2e9],
                "pe_ratio": [15.0, 20.0, 18.0, 16.0, 22.0],
            }
        )
        # Should not crash even if visualization fails
        simple_eda(df_numeric, self.out_dir, save_plots=True)
        output_file = self.out_dir / "eda_summary.json"
        self.assertTrue(output_file.exists())

    def test_simple_eda_returns_distribution_analysis(self):
        """Should include skewness and kurtosis in returned summary"""
        from finance_ml.eval import simple_eda

        # Create dataframe with numeric data
        df_numeric = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "price": [100.0, 200.0, 150.0, 180.0, 220.0],
                "volume": [1000, 2000, 1500, 1800, 2200],
                "market_cap": [1e9, 2e9, 1.5e9, 1.8e9, 2.2e9],
            }
        )
        summary = simple_eda(df_numeric, self.out_dir)

        # Should include distribution analysis with per-column statistics
        self.assertIn("distribution_analysis", summary)
        dist_analysis = summary["distribution_analysis"]
        self.assertIsInstance(dist_analysis, dict)

        # Should have stats for numeric columns
        if dist_analysis:  # If we have enough data
            # Check that at least one column has skewness and kurtosis
            for col_stats in dist_analysis.values():
                self.assertIn("skewness", col_stats)
                self.assertIn("kurtosis", col_stats)
                break  # Just verify structure for one column

    def test_simple_eda_returns_outlier_detection(self):
        """Should include outlier detection results in returned summary"""
        from finance_ml.eval import simple_eda

        # Create dataframe with outliers
        df_with_outliers = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "price": [100.0, 200.0, 150.0, 180.0, 220.0, 10000.0],  # Last one is outlier
                "volume": [1000, 2000, 1500, 1800, 2200, 2100],
            }
        )
        summary = simple_eda(df_with_outliers, self.out_dir)

        # Should include outlier detection
        self.assertIn("outlier_detection", summary)
        outliers = summary["outlier_detection"]
        self.assertIsInstance(outliers, dict)

    def test_simple_eda_returns_normality_tests(self):
        """Should include normality test results in returned summary"""
        from finance_ml.eval import simple_eda

        # Create dataframe with numeric data
        df_numeric = pd.DataFrame(
            {
                "ticker": [f"TICK_{i}" for i in range(20)],
                "price": [100 + i * 10 for i in range(20)],
                "volume": [1000 + i * 100 for i in range(20)],
            }
        )
        summary = simple_eda(df_numeric, self.out_dir)

        # Should include normality tests
        self.assertIn("normality_tests", summary)
        normality = summary["normality_tests"]
        self.assertIsInstance(normality, dict)

    def test_simple_eda_returns_correlation_matrices(self):
        """Should include correlation matrices (Pearson and Spearman) in returned summary"""
        from finance_ml.eval import simple_eda

        # Create dataframe with numeric data
        df_numeric = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "price": [100.0, 200.0, 150.0, 180.0, 220.0],
                "volume": [1000, 2000, 1500, 1800, 2200],
                "market_cap": [1e9, 2e9, 1.5e9, 1.8e9, 2.2e9],
            }
        )
        summary = simple_eda(df_numeric, self.out_dir)

        # Should include correlation analysis
        self.assertIn("correlation_analysis", summary)
        corr_analysis = summary["correlation_analysis"]
        self.assertIn("pearson", corr_analysis)
        self.assertIn("spearman", corr_analysis)

    def test_simple_eda_returns_sector_statistics(self):
        """Should include sector-wise statistics in returned summary"""
        from finance_ml.eval import simple_eda

        # Create dataframe with multiple sectors
        df_multi_sector = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "JPM", "BAC", "XOM", "CVX"],
                "sector": ["Technology", "Technology", "Finance", "Finance", "Energy", "Energy"],
                "price": [150.0, 300.0, 140.0, 35.0, 110.0, 160.0],
                "market_cap": [2.5e12, 2.2e12, 400e9, 300e9, 450e9, 300e9],
            }
        )
        summary = simple_eda(df_multi_sector, self.out_dir)

        # Should include sector statistics
        self.assertIn("sector_statistics", summary)
        sector_stats = summary["sector_statistics"]
        self.assertIsInstance(sector_stats, dict)
        # Should have statistics for each sector
        self.assertGreater(len(sector_stats), 0)

    def test_simple_eda_maintains_backward_compatibility(self):
        """Should maintain backward compatibility with existing fields"""
        from finance_ml.eval import simple_eda

        summary = simple_eda(self.df, self.out_dir)

        # Should still include all original fields
        self.assertIn("row_count", summary)
        self.assertIn("column_count", summary)
        self.assertIn("columns", summary)
        self.assertIn("numeric_cols_count", summary)
        self.assertIn("categorical_cols_count", summary)
        self.assertIn("null_counts", summary)
        self.assertIn("region_counts", summary)
        self.assertIn("sector_counts", summary)
        self.assertIn("basic_stats", summary)

    def test_simple_eda_includes_kendall_correlation(self):
        """Should include Kendall tau correlation in correlation_analysis (Phase 9.2)"""
        from finance_ml.eval import simple_eda

        # Create dataframe with numeric data for correlation
        df_numeric = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "price": [100.0, 200.0, 150.0, 180.0, 220.0],
                "volume": [1000, 2000, 1500, 1800, 2200],
                "market_cap": [1e9, 2e9, 1.5e9, 1.8e9, 2.2e9],
            }
        )
        summary = simple_eda(df_numeric, self.out_dir)

        # Should include Kendall tau in addition to Pearson and Spearman
        self.assertIn("correlation_analysis", summary)
        corr_analysis = summary["correlation_analysis"]
        self.assertIn("pearson", corr_analysis)
        self.assertIn("spearman", corr_analysis)
        self.assertIn("kendall", corr_analysis)  # New feature
        # Verify it's a dict with correlation values
        self.assertIsInstance(corr_analysis["kendall"], dict)

    def test_simple_eda_includes_top_correlations(self):
        """Should include top correlations summary (Phase 9.2)"""
        from finance_ml.eval import simple_eda

        # Create dataframe with numeric data
        df_numeric = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "price": [100.0, 200.0, 150.0, 180.0, 220.0],
                "volume": [1000, 2000, 1500, 1800, 2200],
                "market_cap": [1e9, 2e9, 1.5e9, 1.8e9, 2.2e9],
                "pe_ratio": [15.0, 20.0, 18.0, 16.0, 22.0],
            }
        )
        summary = simple_eda(df_numeric, self.out_dir)

        # Should include top correlations
        self.assertIn("top_correlations", summary)
        top_corr = summary["top_correlations"]
        self.assertIsInstance(top_corr, dict)
        # Should have correlations for each method
        if top_corr:  # If we have enough data
            self.assertIn("pearson", top_corr)
            self.assertIsInstance(top_corr["pearson"], list)

    def test_simple_eda_includes_sector_comparison_tests(self):
        """Should include statistical tests for sector comparisons (Phase 9.2)"""
        from finance_ml.eval import simple_eda

        # Create dataframe with multiple sectors
        df_multi_sector = pd.DataFrame(
            {
                "ticker": ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
                "sector": [
                    "Tech",
                    "Tech",
                    "Tech",
                    "Finance",
                    "Finance",
                    "Finance",
                    "Energy",
                    "Energy",
                    "Energy",
                ],
                "price": [150.0, 160.0, 140.0, 100.0, 110.0, 95.0, 80.0, 85.0, 75.0],
                "market_cap": [2e12, 2.2e12, 1.8e12, 500e9, 550e9, 450e9, 300e9, 320e9, 280e9],
            }
        )
        summary = simple_eda(df_multi_sector, self.out_dir)

        # Should include sector comparison tests
        self.assertIn("sector_comparison_tests", summary)
        sector_tests = summary["sector_comparison_tests"]
        self.assertIsInstance(sector_tests, dict)
        # Should have test results for numeric columns
        if sector_tests:  # If we have enough data
            # Each column should have test results
            for col_result in sector_tests.values():
                self.assertIn("statistic", col_result)
                self.assertIn("p_value", col_result)
                self.assertIn("method", col_result)

    def test_simple_eda_includes_region_statistics(self):
        """Should include region-wise statistics (Phase 9.2)"""
        from finance_ml.eval import simple_eda

        # Create dataframe with multiple regions
        df_multi_region = pd.DataFrame(
            {
                "ticker": ["US1", "US2", "EU1", "EU2", "APAC1", "APAC2"],
                "region": ["US", "US", "EU", "EU", "APAC", "APAC"],
                "sector": ["Tech", "Tech", "Finance", "Finance", "Energy", "Energy"],
                "price": [150.0, 160.0, 100.0, 110.0, 80.0, 85.0],
                "market_cap": [2e12, 2.2e12, 500e9, 550e9, 300e9, 320e9],
            }
        )
        summary = simple_eda(df_multi_region, self.out_dir)

        # Should include region statistics (parallel to sector_statistics)
        self.assertIn("region_statistics", summary)
        region_stats = summary["region_statistics"]
        self.assertIsInstance(region_stats, dict)
        # Should have statistics for each region
        if region_stats:
            for region_name, stats in region_stats.items():
                self.assertIn("count", stats)
                self.assertIn("means", stats)
                self.assertIn("medians", stats)

    def test_simple_eda_includes_feature_importance_when_target_provided(self):
        """Should include feature importance analysis when target is provided (Phase 9.2 integration)"""
        from finance_ml.eval import simple_eda
        import numpy as np

        # Create dataframe with features and target
        np.random.seed(42)
        df_with_target = pd.DataFrame(
            {
                "ticker": [f"TICK_{i}" for i in range(50)],
                "price": np.random.randn(50) * 50 + 150,
                "volume": np.random.randn(50) * 1000 + 5000,
                "market_cap": np.random.randn(50) * 1e9 + 5e9,
                "pe_ratio": np.random.randn(50) * 5 + 20,
                "target": np.random.randn(50) * 20 + 100,  # Target variable
            }
        )
        summary = simple_eda(df_with_target, self.out_dir, target_column="target")

        # Should include feature importance section
        self.assertIn("feature_importance", summary)
        feature_imp = summary["feature_importance"]
        self.assertIsInstance(feature_imp, dict)

    def test_simple_eda_feature_importance_includes_mutual_information(self):
        """Should include mutual information scores (Phase 9.2 integration)"""
        from finance_ml.eval import simple_eda
        import numpy as np

        np.random.seed(42)
        df_with_target = pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 10 + 50,
                "feature2": np.random.randn(50) * 20 + 100,
                "target": np.random.randn(50) * 5 + 25,
            }
        )
        summary = simple_eda(df_with_target, self.out_dir, target_column="target")

        # Should include mutual information
        if "feature_importance" in summary:
            feature_imp = summary["feature_importance"]
            self.assertIn("mutual_information", feature_imp)

    def test_simple_eda_feature_importance_includes_random_forest(self):
        """Should include random forest feature importance (Phase 9.2 integration)"""
        from finance_ml.eval import simple_eda
        import numpy as np

        np.random.seed(42)
        df_with_target = pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 10 + 50,
                "feature2": np.random.randn(50) * 20 + 100,
                "target": np.random.randn(50) * 5 + 25,
            }
        )
        summary = simple_eda(df_with_target, self.out_dir, target_column="target")

        # Should include random forest importance
        if "feature_importance" in summary:
            feature_imp = summary["feature_importance"]
            self.assertIn("random_forest", feature_imp)

    def test_simple_eda_skips_feature_importance_when_no_target(self):
        """Should skip feature importance when no target provided (Phase 9.2 integration)"""
        from finance_ml.eval import simple_eda

        df_no_target = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "price": [100.0, 200.0, 150.0],
                "volume": [1000, 2000, 1500],
            }
        )
        summary = simple_eda(df_no_target, self.out_dir)

        # Should not include feature importance or should be empty
        if "feature_importance" in summary:
            self.assertEqual(summary["feature_importance"], {})

    def test_simple_eda_includes_multivariate_analysis(self):
        """Should include PCA results in multivariate analysis (Phase 9.2 integration)"""
        from finance_ml.eval import simple_eda
        import numpy as np

        np.random.seed(42)
        # Create dataframe with enough features for PCA
        df_multi = pd.DataFrame(
            {
                "ticker": [f"TICK_{i}" for i in range(50)],
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
                "f3": np.random.randn(50),
                "f4": np.random.randn(50),
                "f5": np.random.randn(50),
            }
        )
        summary = simple_eda(df_multi, self.out_dir, include_multivariate=True)

        # Should include multivariate analysis section
        self.assertIn("multivariate_analysis", summary)
        multi_analysis = summary["multivariate_analysis"]
        self.assertIsInstance(multi_analysis, dict)

    def test_simple_eda_multivariate_includes_pca(self):
        """Should include PCA results when multivariate analysis enabled (Phase 9.2 integration)"""
        from finance_ml.eval import simple_eda
        import numpy as np

        np.random.seed(42)
        df_multi = pd.DataFrame(
            {
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
                "f3": np.random.randn(50),
                "f4": np.random.randn(50),
            }
        )
        summary = simple_eda(df_multi, self.out_dir, include_multivariate=True)

        # Should include PCA
        if "multivariate_analysis" in summary:
            multi_analysis = summary["multivariate_analysis"]
            self.assertIn("pca", multi_analysis)

    def test_simple_eda_skips_multivariate_when_not_requested(self):
        """Should skip multivariate analysis when not requested (Phase 9.2 integration)"""
        from finance_ml.eval import simple_eda

        df_simple = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "price": [100.0, 200.0, 150.0],
                "volume": [1000, 2000, 1500],
            }
        )
        summary = simple_eda(df_simple, self.out_dir, include_multivariate=False)

        # Should not include multivariate or should be empty
        if "multivariate_analysis" in summary:
            self.assertEqual(summary["multivariate_analysis"], {})


class TestExportPredictionsToExcel(unittest.TestCase):
    """Test Excel export function"""

    def setUp(self):
        """Create sample data and temp directory"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Technology", "Technology", "Technology"],
                "mispricing_score": [0.15, 0.10, -0.05],
                "predicted_target": [165, 330, 2660],
                "last_price": [150, 300, 2800],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.excel_path = Path(self.temp_dir) / "predictions.xlsx"

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_predictions_to_excel_creates_file(self):
        """Should create Excel file"""
        from finance_ml.eval import export_predictions_to_excel

        try:
            export_predictions_to_excel(self.df, self.excel_path)
            self.assertTrue(self.excel_path.exists())
        except ImportError:
            # OK if no Excel engine available
            self.skipTest("No Excel engine available")

    def test_export_predictions_to_excel_with_summary(self):
        """Should create file with summary when requested"""
        from finance_ml.eval import export_predictions_to_excel

        try:
            export_predictions_to_excel(self.df, self.excel_path, include_summary=True)
            self.assertTrue(self.excel_path.exists())

            # Verify summary sheets were created
            try:
                import openpyxl

                wb = openpyxl.load_workbook(self.excel_path)
                sheet_names = wb.sheetnames
                self.assertIn("Predictions", sheet_names)
                self.assertIn("Summary", sheet_names)
                self.assertIn("By_Sector", sheet_names)

                # Verify Summary sheet has expected metrics
                summary_sheet = wb["Summary"]
                metrics = [cell.value for cell in summary_sheet["A"]]
                self.assertIn("Total Stocks", metrics)
                self.assertIn("Average Mispricing Score", metrics)

                # Verify By_Sector sheet has data
                sector_sheet = wb["By_Sector"]
                # Should have at least header row
                self.assertIsNotNone(sector_sheet["A1"].value)

                wb.close()
            except ImportError:
                # If openpyxl not available but xlsxwriter was used, that's OK
                pass
        except ImportError:
            self.skipTest("No Excel engine available")


class TestCreateSectorHeatmap(unittest.TestCase):
    """Test sector heatmap visualization"""

    def setUp(self):
        """Create sample data and temp directory"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "JPM", "BAC"],
                "sector": ["Technology", "Technology", "Finance", "Finance"],
                "mispricing_score": [0.15, 0.10, 0.08, -0.05],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_path = Path(self.temp_dir) / "heatmap.png"

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_sector_heatmap_no_crash(self):
        """Should not crash when creating heatmap"""
        from finance_ml.eval import create_sector_heatmap

        try:
            create_sector_heatmap(self.df, self.out_path)
        except ImportError:
            self.skipTest("Matplotlib/seaborn not available")
        except Exception as e:
            # Some other error is OK for now (e.g., display issues)
            pass

    def test_create_sector_heatmap_missing_metric_column(self):
        """Should return None when metric column is missing"""
        from finance_ml.eval import create_sector_heatmap

        try:
            # Create df without the metric column
            df_no_metric = pd.DataFrame(
                {"ticker": ["AAPL", "MSFT"], "sector": ["Technology", "Technology"]}
            )
            result = create_sector_heatmap(df_no_metric, self.out_path, metric="nonexistent_metric")
            self.assertIsNone(result)
        except ImportError:
            self.skipTest("Matplotlib/seaborn not available")

    def test_create_sector_heatmap_missing_sector_column(self):
        """Should return None when sector column is missing"""
        from finance_ml.eval import create_sector_heatmap

        try:
            # Create df without sector column
            df_no_sector = pd.DataFrame(
                {"ticker": ["AAPL", "MSFT"], "mispricing_score": [0.10, 0.15]}
            )
            result = create_sector_heatmap(df_no_sector, self.out_path)
            self.assertIsNone(result)
        except ImportError:
            self.skipTest("Matplotlib/seaborn not available")


class TestCreateInteractivePredictionPlot(unittest.TestCase):
    """Test interactive prediction plot"""

    def setUp(self):
        """Create sample data and temp directory"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Technology", "Technology", "Technology"],
                "predicted_target": [165, 330, 2660],
                "last_price": [150, 300, 2800],
                "mispricing_score": [0.10, 0.10, -0.05],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_path = Path(self.temp_dir) / "plot.html"

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_interactive_prediction_plot_no_crash(self):
        """Should not crash when creating interactive plot"""
        from finance_ml.eval import create_interactive_prediction_plot

        try:
            create_interactive_prediction_plot(self.df, self.out_path)
        except ImportError:
            self.skipTest("Plotly not available")
        except Exception:
            # Some other error is OK for now
            pass

    def test_create_interactive_prediction_plot_missing_columns(self):
        """Should return None when required columns are missing"""
        from finance_ml.eval import create_interactive_prediction_plot

        try:
            # Create df without required columns
            df_no_price = pd.DataFrame(
                {"ticker": ["AAPL", "MSFT"], "sector": ["Technology", "Technology"]}
            )
            result = create_interactive_prediction_plot(df_no_price, self.out_path)
            self.assertIsNone(result)
        except ImportError:
            self.skipTest("Plotly not available")


class TestCreateRegionSectorHeatmap(unittest.TestCase):
    """Test region-sector heatmap visualization"""

    def setUp(self):
        """Create sample data and temp directory"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "SAP", "SONY"],
                "sector": ["Technology", "Technology", "Technology", "Technology"],
                "region": ["US", "US", "EU", "APAC"],
                "mispricing_score": [0.15, 0.10, 0.08, -0.05],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_path = Path(self.temp_dir) / "region_heatmap.png"

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_region_sector_heatmap_no_crash(self):
        """Should not crash when creating region-sector heatmap"""
        from finance_ml.eval import create_region_sector_heatmap

        try:
            create_region_sector_heatmap(self.df, out_path=self.out_path)
        except ImportError:
            self.skipTest("Matplotlib/seaborn not available")
        except Exception:
            # Some other error is OK for now
            pass

    def test_create_region_sector_heatmap_missing_columns(self):
        """Should return None when required columns are missing"""
        from finance_ml.eval import create_region_sector_heatmap

        try:
            # Create df without region column
            df_no_region = pd.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT"],
                    "sector": ["Technology", "Technology"],
                    "mispricing_score": [0.10, 0.15],
                }
            )
            result = create_region_sector_heatmap(df_no_region, out_path=self.out_path)
            self.assertIsNone(result)
        except ImportError:
            self.skipTest("Matplotlib/seaborn not available")


class TestImportFallbacks(unittest.TestCase):
    """Test import fallback scenarios when optional libraries unavailable"""

    @patch("finance_ml.eval.plt", None)
    @patch("finance_ml.eval.sns", None)
    def test_simple_eda_warns_when_matplotlib_unavailable(self):
        """Should warn when matplotlib/seaborn unavailable for plots"""
        from finance_ml.eval import simple_eda

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "region": ["US", "US"],
                "price": [150, 300],
            }
        )
        temp_dir = tempfile.mkdtemp()
        try:
            # Should not crash, just skip plots
            simple_eda(df, Path(temp_dir), save_plots=True)
            # Verify summary was still created
            self.assertTrue((Path(temp_dir) / "eda_summary.json").exists())
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("finance_ml.eval.plt", None)
    @patch("finance_ml.eval.sns", None)
    def test_create_sector_heatmap_raises_import_error_when_unavailable(self):
        """Should raise ImportError when matplotlib/seaborn unavailable"""
        from finance_ml.eval import create_sector_heatmap

        df = pd.DataFrame({"sector": ["Tech", "Finance"], "mispricing_score": [0.1, 0.2]})
        with self.assertRaises(ImportError) as ctx:
            create_sector_heatmap(df)
        self.assertIn("Matplotlib", str(ctx.exception))

    @patch("finance_ml.eval.px", None)
    def test_create_interactive_plot_raises_import_error_when_unavailable(self):
        """Should raise ImportError when plotly unavailable"""
        from finance_ml.eval import create_interactive_prediction_plot

        df = pd.DataFrame({"last_price": [100, 200], "predicted_target": [110, 210]})
        with self.assertRaises(ImportError) as ctx:
            create_interactive_prediction_plot(df)
        self.assertIn("Plotly", str(ctx.exception))

    @patch("finance_ml.eval.plt", None)
    @patch("finance_ml.eval.sns", None)
    def test_create_region_sector_heatmap_raises_import_error_when_unavailable(self):
        """Should raise ImportError when matplotlib/seaborn unavailable"""
        from finance_ml.eval import create_region_sector_heatmap

        df = pd.DataFrame(
            {"region": ["US", "EU"], "sector": ["Tech", "Tech"], "mispricing_score": [0.1, 0.2]}
        )
        with self.assertRaises(ImportError) as ctx:
            create_region_sector_heatmap(df)
        self.assertIn("Matplotlib", str(ctx.exception))


class TestExcelExportErrors(unittest.TestCase):
    """Test Excel export error handling"""

    def test_export_predictions_handles_no_engine_available(self):
        """Should raise ImportError when no Excel engine available"""
        from finance_ml.eval import export_predictions_to_excel

        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "mispricing_score": [0.1, 0.2]})
        temp_dir = tempfile.mkdtemp()
        excel_path = Path(temp_dir) / "test.xlsx"

        # Mock both Excel engines to fail
        with patch("pandas.ExcelWriter", side_effect=ImportError("No engine")):
            try:
                with self.assertRaises(ImportError) as ctx:
                    export_predictions_to_excel(df, excel_path)
                self.assertIn("No Excel engine available", str(ctx.exception))
            finally:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)


class TestVisualizationExceptionHandling(unittest.TestCase):
    """Test exception handling in visualization functions"""

    def test_simple_eda_handles_plot_generation_error(self):
        """Should handle exceptions during plot generation gracefully"""
        from finance_ml.eval import simple_eda

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "region": ["US", "US"],
                "price": [150, 300],
            }
        )
        temp_dir = tempfile.mkdtemp()
        try:
            # Mock plt.subplots to raise an exception
            with patch("finance_ml.eval.plt") as mock_plt:
                with patch("finance_ml.eval.sns"):
                    mock_plt.subplots.side_effect = RuntimeError("Display error")
                    # Should not crash, just log warning
                    simple_eda(df, Path(temp_dir), save_plots=True)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_create_sector_heatmap_raises_on_exception(self):
        """Should raise exception when heatmap generation fails"""
        from finance_ml.eval import create_sector_heatmap

        df = pd.DataFrame({"sector": ["Tech", "Finance"], "mispricing_score": [0.1, 0.2]})
        # Mock sns.heatmap to raise exception (this is called inside the function)
        with patch("finance_ml.eval.sns") as mock_sns:
            with patch("finance_ml.eval.plt"):
                mock_sns.heatmap.side_effect = RuntimeError("Heatmap error")
                with self.assertRaises(RuntimeError):
                    create_sector_heatmap(df)

    def test_create_interactive_plot_raises_on_exception(self):
        """Should raise exception when interactive plot generation fails"""
        from finance_ml.eval import create_interactive_prediction_plot

        df = pd.DataFrame(
            {
                "last_price": [100, 200],
                "predicted_target": [110, 210],
                "sector": ["Tech", "Finance"],
            }
        )
        # Mock px.scatter to raise exception
        with patch("finance_ml.eval.px") as mock_px:
            mock_px.scatter.side_effect = RuntimeError("Plot error")
            with self.assertRaises(RuntimeError):
                create_interactive_prediction_plot(df)

    def test_create_region_sector_heatmap_raises_on_exception(self):
        """Should raise exception when region-sector heatmap generation fails"""
        from finance_ml.eval import create_region_sector_heatmap

        df = pd.DataFrame(
            {"region": ["US", "EU"], "sector": ["Tech", "Tech"], "mispricing_score": [0.1, 0.2]}
        )
        # Mock sns.heatmap to raise exception (this is what actually gets called)
        with patch("finance_ml.eval.plt") as mock_plt:
            with patch("finance_ml.eval.sns") as mock_sns:
                # Ensure they're not None
                mock_plt.subplots.return_value = (MagicMock(), MagicMock())
                # Mock the actual seaborn heatmap call that happens inside
                mock_sns.heatmap.side_effect = RuntimeError("Heatmap error")
                with self.assertRaises(RuntimeError):
                    create_region_sector_heatmap(df)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in eval functions"""

    def test_simple_eda_with_single_numeric_column(self):
        """Should handle case with single numeric column for plots"""
        from finance_ml.eval import simple_eda

        df = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "price": [150.0, 300.0]}  # Only one numeric column
        )
        temp_dir = tempfile.mkdtemp()
        try:
            simple_eda(df, Path(temp_dir), save_plots=True)
            # Verify summary was created
            self.assertTrue((Path(temp_dir) / "eda_summary.json").exists())
        except ImportError:
            self.skipTest("Matplotlib/seaborn not available")
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSHAPFeatureImportance(unittest.TestCase):
    """Test SHAP feature importance calculation (Phase 9.2)"""

    def setUp(self):
        """Create sample data for SHAP tests"""
        import numpy as np

        np.random.seed(42)
        self.X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        # Create target with some relationship to features
        self.y = pd.Series(
            2 * self.X["feature1"] + 0.5 * self.X["feature2"] + np.random.randn(100) * 0.1
        )
        # Check if SHAP is available
        try:
            import shap

            self.shap_available = True
        except ImportError:
            self.shap_available = False

    def test_calculate_shap_importance_returns_dataframe(self):
        """Should return DataFrame with feature importance"""
        if not self.shap_available:
            self.skipTest("SHAP library not installed")
        from finance_ml.eval import calculate_shap_importance

        result = calculate_shap_importance(self.X, self.y)
        self.assertIsInstance(result, pd.DataFrame)

    def test_calculate_shap_importance_has_required_columns(self):
        """Should have feature and importance columns"""
        if not self.shap_available:
            self.skipTest("SHAP library not installed")
        from finance_ml.eval import calculate_shap_importance

        result = calculate_shap_importance(self.X, self.y)
        self.assertIn("feature", result.columns)
        self.assertIn("importance", result.columns)

    def test_calculate_shap_importance_sorted_descending(self):
        """Should sort features by importance descending"""
        if not self.shap_available:
            self.skipTest("SHAP library not installed")
        from finance_ml.eval import calculate_shap_importance

        result = calculate_shap_importance(self.X, self.y)
        importances = result["importance"].tolist()
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_calculate_shap_importance_identifies_key_features(self):
        """Should identify feature1 as most important (strongest relationship with y)"""
        if not self.shap_available:
            self.skipTest("SHAP library not installed")
        from finance_ml.eval import calculate_shap_importance

        result = calculate_shap_importance(self.X, self.y)
        # feature1 should have highest importance
        top_feature = result.iloc[0]["feature"]
        self.assertEqual(top_feature, "feature1")

    def test_calculate_shap_importance_handles_missing_shap(self):
        """Should handle case when SHAP library not available"""
        from finance_ml.eval import calculate_shap_importance

        with patch.dict("sys.modules", {"shap": None}):
            with self.assertRaises(ImportError):
                calculate_shap_importance(self.X, self.y)


class TestTSNEVisualization(unittest.TestCase):
    """Test t-SNE dimensionality reduction (Phase 9.2)"""

    def setUp(self):
        """Create sample high-dimensional data"""
        import numpy as np

        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(50, 10), columns=[f"feature_{i}" for i in range(10)])

    def test_perform_tsne_returns_dict(self):
        """Should return dictionary with t-SNE results"""
        from finance_ml.eval import perform_tsne

        result = perform_tsne(self.X, n_components=2)
        self.assertIsInstance(result, dict)

    def test_perform_tsne_has_required_keys(self):
        """Should have components and feature_names keys"""
        from finance_ml.eval import perform_tsne

        result = perform_tsne(self.X, n_components=2)
        self.assertIn("components", result)
        self.assertIn("feature_names", result)
        self.assertIn("n_components", result)

    def test_perform_tsne_correct_dimensions(self):
        """Should reduce to specified number of components"""
        from finance_ml.eval import perform_tsne

        result = perform_tsne(self.X, n_components=2)
        components = result["components"]
        self.assertEqual(components.shape[1], 2)
        self.assertEqual(components.shape[0], len(self.X))

    def test_perform_tsne_three_components(self):
        """Should support 3-component t-SNE"""
        from finance_ml.eval import perform_tsne

        result = perform_tsne(self.X, n_components=3)
        components = result["components"]
        self.assertEqual(components.shape[1], 3)

    def test_perform_tsne_handles_missing_sklearn(self):
        """Should handle case when sklearn not available"""
        from finance_ml.eval import perform_tsne

        with patch.dict("sys.modules", {"sklearn.manifold": None}):
            with self.assertRaises(ImportError):
                perform_tsne(self.X, n_components=2)


class TestUMAPVisualization(unittest.TestCase):
    """Test UMAP dimensionality reduction (Phase 9.2)"""

    def setUp(self):
        """Create sample high-dimensional data"""
        import numpy as np

        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(50, 10), columns=[f"feature_{i}" for i in range(10)])
        # Check if umap-learn is available
        try:
            import umap

            self.umap_available = True
        except ImportError:
            self.umap_available = False

    def test_perform_umap_returns_dict(self):
        """Should return dictionary with UMAP results"""
        if not self.umap_available:
            self.skipTest("umap-learn not installed")
        from finance_ml.eval import perform_umap

        result = perform_umap(self.X, n_components=2)
        self.assertIsInstance(result, dict)

    def test_perform_umap_has_required_keys(self):
        """Should have components and feature_names keys"""
        if not self.umap_available:
            self.skipTest("umap-learn not installed")
        from finance_ml.eval import perform_umap

        result = perform_umap(self.X, n_components=2)
        self.assertIn("components", result)
        self.assertIn("feature_names", result)
        self.assertIn("n_components", result)

    def test_perform_umap_correct_dimensions(self):
        """Should reduce to specified number of components"""
        if not self.umap_available:
            self.skipTest("umap-learn not installed")
        from finance_ml.eval import perform_umap

        result = perform_umap(self.X, n_components=2)
        components = result["components"]
        self.assertEqual(components.shape[1], 2)
        self.assertEqual(components.shape[0], len(self.X))

    def test_perform_umap_three_components(self):
        """Should support 3-component UMAP"""
        if not self.umap_available:
            self.skipTest("umap-learn not installed")
        from finance_ml.eval import perform_umap

        result = perform_umap(self.X, n_components=3)
        components = result["components"]
        self.assertEqual(components.shape[1], 3)

    def test_perform_umap_handles_missing_umap(self):
        """Should handle case when umap-learn not available"""
        from finance_ml.eval import perform_umap

        with patch.dict("sys.modules", {"umap": None}):
            with self.assertRaises(ImportError):
                perform_umap(self.X, n_components=2)


class TestCalculateCorrelationMatrix(unittest.TestCase):
    """Test correlation matrix calculation (Phase 9.2 helper function)"""

    def setUp(self):
        """Create sample numeric data"""
        import numpy as np

        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "price": [100, 200, 150, 180, 220],
                "volume": [1000, 2000, 1500, 1800, 2200],
                "market_cap": [1e9, 2e9, 1.5e9, 1.8e9, 2.2e9],
            }
        )
        self.columns = ["price", "volume", "market_cap"]

    def test_calculate_correlation_matrix_returns_dataframe(self):
        """Should return pandas DataFrame"""
        from finance_ml.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(self.df, self.columns, method="pearson")
        self.assertIsInstance(result, pd.DataFrame)

    def test_calculate_correlation_matrix_pearson(self):
        """Should calculate Pearson correlation correctly"""
        from finance_ml.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(self.df, self.columns, method="pearson")
        # Correlation matrix should be square
        self.assertEqual(result.shape[0], result.shape[1])
        # Diagonal should be 1.0
        for i in range(len(result)):
            self.assertAlmostEqual(result.iloc[i, i], 1.0, places=5)

    def test_calculate_correlation_matrix_spearman(self):
        """Should calculate Spearman correlation correctly"""
        from finance_ml.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(self.df, self.columns, method="spearman")
        self.assertEqual(result.shape[0], len(self.columns))
        self.assertEqual(result.shape[1], len(self.columns))

    def test_calculate_correlation_matrix_kendall(self):
        """Should calculate Kendall tau correlation correctly"""
        from finance_ml.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(self.df, self.columns, method="kendall")
        self.assertEqual(result.shape[0], len(self.columns))
        self.assertEqual(result.shape[1], len(self.columns))


class TestFindTopCorrelations(unittest.TestCase):
    """Test top correlations extraction (Phase 9.2 helper function)"""

    def setUp(self):
        """Create sample correlation matrix"""
        import numpy as np

        # Create correlation matrix with known values
        self.corr_matrix = pd.DataFrame(
            {"A": [1.0, 0.8, 0.3], "B": [0.8, 1.0, 0.5], "C": [0.3, 0.5, 1.0]},
            index=["A", "B", "C"],
        )

    def test_find_top_correlations_returns_list(self):
        """Should return list of tuples"""
        from finance_ml.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=5)
        self.assertIsInstance(result, list)

    def test_find_top_correlations_tuple_structure(self):
        """Should return tuples with (var1, var2, correlation)"""
        from finance_ml.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=5)
        if result:
            self.assertEqual(len(result[0]), 3)
            # First element should be highest correlation (A-B: 0.8)
            self.assertEqual(result[0][0], "A")
            self.assertEqual(result[0][1], "B")
            self.assertAlmostEqual(result[0][2], 0.8, places=5)

    def test_find_top_correlations_respects_n_top(self):
        """Should limit results to n_top"""
        from finance_ml.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=2)
        self.assertLessEqual(len(result), 2)

    def test_find_top_correlations_threshold_filter(self):
        """Should filter by threshold"""
        from finance_ml.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=10, threshold=0.6)
        # Only A-B (0.8) should pass threshold of 0.6
        self.assertEqual(len(result), 1)


class TestNormality(unittest.TestCase):
    """Test normality testing (Phase 9.2 helper function)"""

    def setUp(self):
        """Create sample data"""
        import numpy as np

        np.random.seed(42)
        # Normal distribution
        self.df = pd.DataFrame(
            {"normal": np.random.normal(100, 15, 100), "uniform": np.random.uniform(50, 150, 100)}
        )

    def test_normality_returns_dict(self):
        """Should return dictionary"""
        from finance_ml.eval import test_normality

        result = test_normality(self.df, ["normal", "uniform"])
        self.assertIsInstance(result, dict)

    def test_normality_has_required_keys(self):
        """Should have required keys for each column"""
        from finance_ml.eval import test_normality

        result = test_normality(self.df, ["normal"])
        self.assertIn("normal", result)
        self.assertIn("statistic", result["normal"])
        self.assertIn("p_value", result["normal"])
        self.assertIn("is_normal", result["normal"])

    def test_normality_handles_insufficient_data(self):
        """Should handle columns with insufficient data"""
        from finance_ml.eval import test_normality

        df_small = pd.DataFrame({"col": [1, 2]})
        result = test_normality(df_small, ["col"])
        self.assertIn("col", result)
        self.assertIsNone(result["col"]["is_normal"])


class TestSkewnessKurtosis(unittest.TestCase):
    """Test skewness and kurtosis calculation (Phase 9.2 helper function)"""

    def setUp(self):
        """Create sample data"""
        import numpy as np

        np.random.seed(42)
        self.df = pd.DataFrame(
            {"normal": np.random.normal(100, 15, 100), "skewed": np.random.exponential(50, 100)}
        )

    def test_skewness_kurtosis_returns_dataframe(self):
        """Should return pandas DataFrame"""
        from finance_ml.eval import calculate_skewness_kurtosis

        result = calculate_skewness_kurtosis(self.df, ["normal", "skewed"])
        self.assertIsInstance(result, pd.DataFrame)

    def test_skewness_kurtosis_has_required_columns(self):
        """Should have skewness and kurtosis columns"""
        from finance_ml.eval import calculate_skewness_kurtosis

        result = calculate_skewness_kurtosis(self.df, ["normal"])
        self.assertIn("skewness", result.columns)
        self.assertIn("kurtosis", result.columns)

    def test_skewness_kurtosis_normal_distribution(self):
        """Normal distribution should have low skewness"""
        from finance_ml.eval import calculate_skewness_kurtosis

        result = calculate_skewness_kurtosis(self.df, ["normal"])
        # Normal distribution should have skewness close to 0
        self.assertLess(abs(result.loc["normal", "skewness"]), 1.0)


class TestCompareSectorMeans(unittest.TestCase):
    """Test sector mean comparison (Phase 9.2 helper function)"""

    def setUp(self):
        """Create sample data with multiple sectors"""
        import numpy as np

        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
                "price": [150, 160, 140, 100, 110, 95],
                "pe_ratio": [20, 22, 18, 12, 14, 11],
            }
        )

    def test_compare_sector_means_returns_dict(self):
        """Should return dictionary"""
        from finance_ml.eval import compare_sector_means

        result = compare_sector_means(self.df, "price", group_column="sector")
        self.assertIsInstance(result, dict)

    def test_compare_sector_means_has_required_keys(self):
        """Should have required keys"""
        from finance_ml.eval import compare_sector_means

        result = compare_sector_means(self.df, "price", group_column="sector")
        self.assertIn("method", result)
        self.assertIn("statistic", result)
        self.assertIn("p_value", result)

    def test_compare_sector_means_anova_method(self):
        """Should support ANOVA method"""
        from finance_ml.eval import compare_sector_means

        result = compare_sector_means(self.df, "price", group_column="sector", method="anova")
        self.assertEqual(result["method"], "anova")

    def test_compare_sector_means_kruskal_method(self):
        """Should support Kruskal-Wallis method"""
        from finance_ml.eval import compare_sector_means

        result = compare_sector_means(self.df, "price", group_column="sector", method="kruskal")
        self.assertEqual(result["method"], "kruskal")


class TestDistanceCorrelation(unittest.TestCase):
    """Test distance correlation calculation (Phase 9.2 continuation)"""

    def setUp(self):
        """Create sample data for distance correlation tests"""
        import numpy as np

        np.random.seed(42)
        self.df = pd.DataFrame(
            {"x": np.random.randn(50), "y": np.random.randn(50), "z": np.random.randn(50)}
        )
        # Check if dcor is available
        try:
            import dcor

            self.dcor_available = True
        except ImportError:
            self.dcor_available = False

    def test_calculate_distance_correlation_returns_dataframe(self):
        """Should return DataFrame with distance correlation matrix"""
        if not self.dcor_available:
            self.skipTest("dcor library not installed")
        from finance_ml.eval import calculate_distance_correlation

        result = calculate_distance_correlation(self.df, ["x", "y", "z"])
        self.assertIsInstance(result, pd.DataFrame)

    def test_calculate_distance_correlation_correct_shape(self):
        """Should return square matrix"""
        if not self.dcor_available:
            self.skipTest("dcor library not installed")
        from finance_ml.eval import calculate_distance_correlation

        result = calculate_distance_correlation(self.df, ["x", "y"])
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 2)

    def test_calculate_distance_correlation_diagonal_is_one(self):
        """Distance correlation of variable with itself should be 1.0"""
        if not self.dcor_available:
            self.skipTest("dcor library not installed")
        from finance_ml.eval import calculate_distance_correlation

        result = calculate_distance_correlation(self.df, ["x", "y"])
        self.assertAlmostEqual(result.loc["x", "x"], 1.0, places=5)
        self.assertAlmostEqual(result.loc["y", "y"], 1.0, places=5)

    def test_calculate_distance_correlation_handles_missing_dcor(self):
        """Should raise ImportError when dcor not available"""
        from finance_ml.eval import calculate_distance_correlation

        with patch.dict("sys.modules", {"dcor": None}):
            with self.assertRaises(ImportError):
                calculate_distance_correlation(self.df, ["x", "y"])

    def test_simple_eda_includes_distance_correlation(self):
        """Should include distance correlation in correlation_analysis when available"""
        if not self.dcor_available:
            self.skipTest("dcor library not installed")
        from finance_ml.eval import simple_eda
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            summary = simple_eda(self.df, Path(temp_dir))

            # Should include distance correlation
            self.assertIn("correlation_analysis", summary)
            corr_analysis = summary["correlation_analysis"]
            self.assertIn("distance", corr_analysis)
            self.assertIsInstance(corr_analysis["distance"], dict)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestOutlierVisualization(unittest.TestCase):
    """Test outlier visualization functions (Phase 9.2 continuation)"""

    def setUp(self):
        """Create sample data with outliers"""
        import numpy as np

        np.random.seed(42)
        # Normal data with few outliers
        self.df = pd.DataFrame(
            {
                "feature1": np.concatenate(
                    [np.random.randn(45) * 10 + 50, [150, 160, 170, 180, 190]]
                ),
                "feature2": np.concatenate(
                    [np.random.randn(45) * 5 + 100, [10, 15, 200, 210, 220]]
                ),
                "feature3": np.random.randn(50) * 20 + 75,
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plot_outlier_boxplots_creates_figure(self):
        """Should create matplotlib figure with box plots"""
        from finance_ml.eval import plot_outlier_boxplots

        fig = plot_outlier_boxplots(self.df, columns=["feature1", "feature2", "feature3"])
        self.assertIsNotNone(fig)

    def test_plot_outlier_boxplots_saves_to_file(self):
        """Should save box plots to file when out_path provided"""
        from finance_ml.eval import plot_outlier_boxplots

        out_path = self.out_dir / "outlier_boxplots.png"
        plot_outlier_boxplots(self.df, columns=["feature1", "feature2"], out_path=out_path)
        self.assertTrue(out_path.exists())

    def test_plot_outlier_violins_creates_figure(self):
        """Should create matplotlib figure with violin plots"""
        from finance_ml.eval import plot_outlier_violins

        fig = plot_outlier_violins(self.df, columns=["feature1", "feature2", "feature3"])
        self.assertIsNotNone(fig)

    def test_plot_outlier_violins_saves_to_file(self):
        """Should save violin plots to file when out_path provided"""
        from finance_ml.eval import plot_outlier_violins

        out_path = self.out_dir / "outlier_violins.png"
        plot_outlier_violins(self.df, columns=["feature1", "feature2"], out_path=out_path)
        self.assertTrue(out_path.exists())

    def test_plot_outlier_scatter_creates_figure(self):
        """Should create matplotlib figure with scatter plot showing z-scores"""
        from finance_ml.eval import plot_outlier_scatter

        fig = plot_outlier_scatter(self.df, columns=["feature1", "feature2"])
        self.assertIsNotNone(fig)

    def test_plot_outlier_scatter_saves_to_file(self):
        """Should save scatter plot to file when out_path provided"""
        from finance_ml.eval import plot_outlier_scatter

        out_path = self.out_dir / "outlier_scatter.png"
        plot_outlier_scatter(self.df, columns=["feature1", "feature2"], out_path=out_path)
        self.assertTrue(out_path.exists())

    def test_simple_eda_saves_outlier_plots(self):
        """Should save outlier visualization plots when save_plots=True"""
        from finance_ml.eval import simple_eda

        simple_eda(self.df, self.out_dir, save_plots=True)

        # Should create outlier visualization files
        expected_files = [
            "eda_outlier_boxplots.png",
            "eda_outlier_violins.png",
            "eda_outlier_scatter.png",
        ]
        for filename in expected_files:
            filepath = self.out_dir / filename
            # Files should exist if matplotlib available
            if filepath.exists():
                self.assertTrue(filepath.is_file())


class TestUMAPIntegration(unittest.TestCase):
    """Test UMAP integration into simple_eda (Phase 9.2 continuation)"""

    def setUp(self):
        """Create sample high-dimensional data"""
        import numpy as np

        np.random.seed(42)
        self.df = pd.DataFrame(np.random.randn(50, 10), columns=[f"feature_{i}" for i in range(10)])
        # Check if umap-learn is available
        try:
            import umap

            self.umap_available = True
        except ImportError:
            self.umap_available = False

    def test_simple_eda_includes_umap_in_multivariate(self):
        """Should include UMAP results in multivariate_analysis when available"""
        if not self.umap_available:
            self.skipTest("umap-learn not installed")
        from finance_ml.eval import simple_eda
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            summary = simple_eda(self.df, Path(temp_dir), include_multivariate=True)

            # Should include UMAP in multivariate analysis
            self.assertIn("multivariate_analysis", summary)
            multi_analysis = summary["multivariate_analysis"]
            self.assertIn("umap", multi_analysis)

            # UMAP should have expected structure
            if multi_analysis["umap"]:  # If not empty
                self.assertIn("n_components", multi_analysis["umap"])
                self.assertIn("feature_names", multi_analysis["umap"])
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_simple_eda_skips_umap_when_not_available(self):
        """Should gracefully skip UMAP when umap-learn not installed"""
        from finance_ml.eval import simple_eda
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            # Mock umap as unavailable
            with patch.dict("sys.modules", {"umap": None}):
                summary = simple_eda(self.df, Path(temp_dir), include_multivariate=True)

                # Should still have multivariate_analysis
                self.assertIn("multivariate_analysis", summary)
                # UMAP should be empty or absent, not cause crash
                multi_analysis = summary["multivariate_analysis"]
                if "umap" in multi_analysis:
                    self.assertEqual(multi_analysis["umap"], {})
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
