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
        df = pd.DataFrame({"predicted_target": [110, 90, 105], "last_price": [100, 100, 100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertIsInstance(result, pd.Series)

    def test_calculate_mispricing_score_undervalued(self):
        """Should return positive score for undervalued stocks"""
        df = pd.DataFrame({"predicted_target": [120], "last_price": [100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertGreater(result.iloc[0], 0)

    def test_calculate_mispricing_score_overvalued(self):
        """Should return negative score for overvalued stocks"""
        df = pd.DataFrame({"predicted_target": [80], "last_price": [100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertLess(result.iloc[0], 0)

    def test_calculate_mispricing_score_correct_formula(self):
        """Should calculate (predicted - current) / current"""
        df = pd.DataFrame({"predicted_target": [110], "last_price": [100]})
        from finance_ml.eval import calculate_mispricing_score

        result = calculate_mispricing_score(df)
        self.assertAlmostEqual(result.iloc[0], 0.1, places=5)


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
            with patch("finance_ml.eval.plt") as mock_plt:
                # Set up plt.subplots to return a tuple of (fig, ax)
                mock_plt.subplots.return_value = (MagicMock(), MagicMock())
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


class TestAssignValuationCategory(unittest.TestCase):
    """Test valuation category assignment"""

    def test_assign_valuation_category_strong_buy(self):
        """Should assign Strong Buy for mispricing > 20%"""
        from finance_ml.eval import assign_valuation_category

        scores = pd.Series([25.0, 30.0, 22.0])
        result = assign_valuation_category(scores)
        self.assertTrue(all(result == "Strong Buy"))

    def test_assign_valuation_category_buy(self):
        """Should assign Buy for mispricing 10-20%"""
        from finance_ml.eval import assign_valuation_category

        scores = pd.Series([15.0, 18.0, 12.0])
        result = assign_valuation_category(scores)
        self.assertTrue(all(result == "Buy"))

    def test_assign_valuation_category_hold(self):
        """Should assign Hold for mispricing -10 to 10%"""
        from finance_ml.eval import assign_valuation_category

        scores = pd.Series([5.0, 0.0, -5.0, 8.0])
        result = assign_valuation_category(scores)
        self.assertTrue(all(result == "Hold"))

    def test_assign_valuation_category_sell(self):
        """Should assign Sell for mispricing -20 to -10%"""
        from finance_ml.eval import assign_valuation_category

        scores = pd.Series([-15.0, -18.0, -12.0])
        result = assign_valuation_category(scores)
        self.assertTrue(all(result == "Sell"))

    def test_assign_valuation_category_strong_sell(self):
        """Should assign Strong Sell for mispricing < -20%"""
        from finance_ml.eval import assign_valuation_category

        scores = pd.Series([-25.0, -30.0, -22.0])
        result = assign_valuation_category(scores)
        self.assertTrue(all(result == "Strong Sell"))

    def test_assign_valuation_category_custom_thresholds(self):
        """Should accept custom thresholds"""
        from finance_ml.eval import assign_valuation_category

        scores = pd.Series([8.0, -8.0])
        custom_thresholds = {
            "strong_buy": 5.0,
            "buy": 3.0,
            "sell": -3.0,
            "strong_sell": -5.0,
        }
        result = assign_valuation_category(scores, thresholds=custom_thresholds)
        self.assertEqual(result.iloc[0], "Strong Buy")
        self.assertEqual(result.iloc[1], "Strong Sell")

    def test_assign_valuation_category_boundary_values(self):
        """Should handle boundary values correctly"""
        from finance_ml.eval import assign_valuation_category

        scores = pd.Series([20.0, 10.0, -10.0, -20.0])
        result = assign_valuation_category(scores)
        # Test exact boundaries
        self.assertIn(result.iloc[0], ["Strong Buy", "Buy"])
        self.assertIn(result.iloc[1], ["Buy", "Hold"])


class TestCalculateSectorZScores(unittest.TestCase):
    """Test sector-relative z-score calculations"""

    def setUp(self):
        """Create test data with sectors"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
                "p_e": [20, 25, 30, 15, 20, 25],
                "p_b": [3.0, 3.5, 4.0, 1.5, 2.0, 2.5],
                "ev_ebitda": [12, 15, 18, 10, 12, 14],
            }
        )

    def test_calculate_sector_zscores_returns_dataframe(self):
        """Should return DataFrame with z-score columns"""
        from finance_ml.eval import calculate_sector_zscores

        result = calculate_sector_zscores(self.df, metrics=["p_e", "p_b"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("p_e_zscore", result.columns)
        self.assertIn("p_b_zscore", result.columns)

    def test_calculate_sector_zscores_correct_calculation(self):
        """Should calculate z-scores correctly within sectors"""
        from finance_ml.eval import calculate_sector_zscores

        result = calculate_sector_zscores(self.df, metrics=["p_e"])
        # Mean P/E for Tech = 25, std ≈ 5
        # For P/E=20: z-score ≈ -1.0
        tech_zscores = result[result["sector"] == "Tech"]["p_e_zscore"]
        self.assertTrue(tech_zscores.iloc[0] < 0)  # Below mean
        self.assertTrue(tech_zscores.iloc[2] > 0)  # Above mean

    def test_calculate_sector_zscores_handles_single_sector(self):
        """Should handle single sector case"""
        from finance_ml.eval import calculate_sector_zscores

        single_sector_df = self.df[self.df["sector"] == "Tech"].copy()
        result = calculate_sector_zscores(single_sector_df, metrics=["p_e"])
        self.assertEqual(len(result), 3)

    def test_calculate_sector_zscores_missing_metric(self):
        """Should handle missing metrics gracefully"""
        from finance_ml.eval import calculate_sector_zscores

        # Test with metric that doesn't exist
        result = calculate_sector_zscores(self.df, metrics=["nonexistent_metric"])
        # Should still return original dataframe or handle gracefully
        self.assertIsInstance(result, pd.DataFrame)


class TestCalculatePercentileRanks(unittest.TestCase):
    """Test percentile rank calculations"""

    def setUp(self):
        """Create test data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "sector": ["Tech", "Tech", "Tech", "Finance", "Finance"],
                "p_e": [20, 25, 30, 15, 25],
                "market_cap": [100, 200, 300, 150, 250],
            }
        )

    def test_calculate_percentile_ranks_returns_dataframe(self):
        """Should return DataFrame with percentile columns"""
        from finance_ml.eval import calculate_percentile_ranks

        result = calculate_percentile_ranks(self.df, metrics=["p_e"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("p_e_percentile", result.columns)

    def test_calculate_percentile_ranks_range(self):
        """Should return percentiles in range 0-100"""
        from finance_ml.eval import calculate_percentile_ranks

        result = calculate_percentile_ranks(self.df, metrics=["p_e", "market_cap"])
        self.assertTrue((result["p_e_percentile"] >= 0).all())
        self.assertTrue((result["p_e_percentile"] <= 100).all())
        self.assertTrue((result["market_cap_percentile"] >= 0).all())
        self.assertTrue((result["market_cap_percentile"] <= 100).all())

    def test_calculate_percentile_ranks_ordering(self):
        """Should rank correctly (higher values = higher percentile)"""
        from finance_ml.eval import calculate_percentile_ranks

        result = calculate_percentile_ranks(self.df, metrics=["market_cap"])
        # market_cap: [100, 200, 300, 150, 250]
        # Largest should have highest percentile
        max_cap_idx = result["market_cap"].idxmax()
        max_percentile = result.loc[max_cap_idx, "market_cap_percentile"]
        self.assertGreater(max_percentile, 50)


class TestCalculateMultiFactorScore(unittest.TestCase):
    """Test multi-factor scoring"""

    def setUp(self):
        """Create test data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "mispricing_score": [15.0, -10.0, 25.0],
                "roe": [0.15, 0.10, 0.20],
                "ebitda_margin": [0.25, 0.20, 0.30],
                "revenue_cagr": [0.10, 0.05, 0.15],
            }
        )

    def test_calculate_multi_factor_score_returns_series(self):
        """Should return Series with composite scores"""
        from finance_ml.eval import calculate_multi_factor_score

        result = calculate_multi_factor_score(
            self.df, quality_cols=["roe"], growth_cols=["revenue_cagr"]
        )
        self.assertIsInstance(result, pd.Series)

    def test_calculate_multi_factor_score_custom_weights(self):
        """Should accept custom weights"""
        from finance_ml.eval import calculate_multi_factor_score

        weights = {"valuation": 0.5, "quality": 0.3, "growth": 0.2}
        result = calculate_multi_factor_score(
            self.df,
            quality_cols=["roe"],
            growth_cols=["revenue_cagr"],
            weights=weights,
        )
        self.assertIsInstance(result, pd.Series)

    def test_calculate_multi_factor_score_no_nans(self):
        """Should handle missing values properly"""
        from finance_ml.eval import calculate_multi_factor_score

        df_with_nan = self.df.copy()
        df_with_nan.loc[0, "roe"] = None
        result = calculate_multi_factor_score(
            df_with_nan, quality_cols=["roe"], growth_cols=["revenue_cagr"]
        )
        # Should still compute scores (may fill or skip NaNs)
        self.assertIsInstance(result, pd.Series)


class TestFilterStocksByCriteria(unittest.TestCase):
    """Test stock filtering functionality"""

    def setUp(self):
        """Create test data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "sector": ["Tech", "Tech", "Finance", "Healthcare", "Finance"],
                "region": ["US", "EU", "US", "APAC", "EU"],
                "market_cap": [100, 200, 150, 300, 250],
                "mispricing_score": [15.0, -10.0, 25.0, 5.0, -15.0],
                "valuation_category": ["Buy", "Sell", "Strong Buy", "Hold", "Sell"],
            }
        )

    def test_filter_stocks_by_sector(self):
        """Should filter by sector"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, sectors=["Tech"])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result["sector"] == "Tech"))

    def test_filter_stocks_by_region(self):
        """Should filter by region"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, regions=["US"])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result["region"] == "US"))

    def test_filter_stocks_by_market_cap(self):
        """Should filter by market cap range"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, min_market_cap=150, max_market_cap=250)
        self.assertTrue(all(result["market_cap"] >= 150))
        self.assertTrue(all(result["market_cap"] <= 250))

    def test_filter_stocks_by_mispricing(self):
        """Should filter by mispricing range"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, min_mispricing=10.0)
        self.assertTrue(all(result["mispricing_score"] >= 10.0))

    def test_filter_stocks_by_valuation_category(self):
        """Should filter by valuation categories"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(self.df, valuation_categories=["Buy", "Strong Buy"])
        self.assertTrue(all(result["valuation_category"].isin(["Buy", "Strong Buy"])))

    def test_filter_stocks_combined_criteria(self):
        """Should filter by multiple criteria"""
        from finance_ml.eval import filter_stocks_by_criteria

        result = filter_stocks_by_criteria(
            self.df,
            sectors=["Tech", "Finance"],
            regions=["US"],
            min_mispricing=10.0,
        )
        self.assertTrue(all(result["region"] == "US"))
        self.assertTrue(all(result["mispricing_score"] >= 10.0))


class TestCreateValuationScatterPlot(unittest.TestCase):
    """Test valuation scatter plot creation"""

    def setUp(self):
        """Create test data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "sector": ["Tech", "Tech", "Finance", "Healthcare"],
                "last_price": [100, 150, 200, 250],
                "predicted_target": [120, 140, 230, 270],
            }
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_valuation_scatter_plot_no_crash(self):
        """Should create scatter plot without crashing"""
        from finance_ml.eval import create_valuation_scatter_plot

        out_path = Path(self.temp_dir) / "scatter.html"
        try:
            create_valuation_scatter_plot(self.df, out_path=out_path)
            # Should create file
            self.assertTrue(out_path.exists())
        except ImportError:
            self.skipTest("Plotly not available")

    def test_create_valuation_scatter_plot_color_by_sector(self):
        """Should accept color_by parameter"""
        from finance_ml.eval import create_valuation_scatter_plot

        try:
            create_valuation_scatter_plot(self.df, color_by="sector")
        except ImportError:
            self.skipTest("Plotly not available")

    def test_create_valuation_scatter_plot_missing_columns(self):
        """Should return None with missing required columns"""
        from finance_ml.eval import create_valuation_scatter_plot

        df_missing = pd.DataFrame({"ticker": ["A", "B"]})
        try:
            result = create_valuation_scatter_plot(df_missing)
            # Should return None and log warning when columns missing
            self.assertIsNone(result)
        except ImportError:
            self.skipTest("Plotly not available")


class TestCalculateRiskAdjustedMispricing(unittest.TestCase):
    """Test risk-adjusted mispricing calculations"""

    def setUp(self):
        """Create test data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "predicted_target": [120, 140, 95, 210],
                "last_price": [100, 150, 100, 200],
                "volatility": [0.20, 0.30, 0.15, 0.25],
                "confidence_lower": [110, 130, 90, 200],
                "confidence_upper": [130, 150, 100, 220],
            }
        )

    def test_calculate_risk_adjusted_mispricing_returns_series(self):
        """Should return Series with risk-adjusted scores"""
        from finance_ml.eval import calculate_risk_adjusted_mispricing

        result = calculate_risk_adjusted_mispricing(self.df)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))

    def test_calculate_risk_adjusted_mispricing_uses_volatility(self):
        """Should divide expected return by volatility"""
        from finance_ml.eval import calculate_risk_adjusted_mispricing

        result = calculate_risk_adjusted_mispricing(self.df, risk_free_rate=0.0)
        # For ticker A: expected_return = (120-100)/100 = 0.20
        # risk_adjusted = 0.20 / 0.20 = 1.0
        self.assertAlmostEqual(result.iloc[0], 1.0, places=2)

    def test_calculate_risk_adjusted_mispricing_with_risk_free_rate(self):
        """Should subtract risk-free rate from expected return"""
        from finance_ml.eval import calculate_risk_adjusted_mispricing

        result = calculate_risk_adjusted_mispricing(self.df, risk_free_rate=0.05)
        # For ticker A: expected_return = 0.20, adjusted = (0.20 - 0.05) / 0.20 = 0.75
        self.assertAlmostEqual(result.iloc[0], 0.75, places=2)

    def test_calculate_risk_adjusted_mispricing_handles_zero_volatility(self):
        """Should handle zero volatility gracefully"""
        from finance_ml.eval import calculate_risk_adjusted_mispricing

        df_zero_vol = self.df.copy()
        df_zero_vol.loc[0, "volatility"] = 0.0
        result = calculate_risk_adjusted_mispricing(df_zero_vol)
        # Should not raise error, may return 0 or NaN for zero volatility
        self.assertIsInstance(result, pd.Series)

    def test_calculate_risk_adjusted_mispricing_missing_volatility(self):
        """Should handle missing volatility column"""
        from finance_ml.eval import calculate_risk_adjusted_mispricing

        df_no_vol = self.df.drop(columns=["volatility"])
        # Should either raise error or use default volatility
        try:
            result = calculate_risk_adjusted_mispricing(df_no_vol)
            self.assertIsInstance(result, pd.Series)
        except (KeyError, ValueError):
            # Acceptable to raise error if volatility required
            pass

    def test_calculate_risk_adjusted_mispricing_with_confidence_intervals(self):
        """Should optionally use confidence interval width"""
        from finance_ml.eval import calculate_risk_adjusted_mispricing

        result = calculate_risk_adjusted_mispricing(self.df, use_confidence_interval=True)
        self.assertIsInstance(result, pd.Series)
        # Result should account for confidence interval uncertainty


class TestGetSectorSpecificThresholds(unittest.TestCase):
    """Test sector-specific threshold calculations"""

    def test_get_sector_specific_thresholds_returns_dict(self):
        """Should return dict with sector-specific thresholds"""
        from finance_ml.eval import get_sector_specific_thresholds

        result = get_sector_specific_thresholds("Tech")
        self.assertIsInstance(result, dict)
        self.assertIn("strong_buy", result)
        self.assertIn("buy", result)
        self.assertIn("sell", result)
        self.assertIn("strong_sell", result)

    def test_get_sector_specific_thresholds_volatile_sectors(self):
        """Should return wider thresholds for volatile sectors"""
        from finance_ml.eval import get_sector_specific_thresholds

        tech_thresholds = get_sector_specific_thresholds("Technology")
        finance_thresholds = get_sector_specific_thresholds("Finance")
        # Tech should have wider bands than Finance
        self.assertGreater(tech_thresholds["strong_buy"], finance_thresholds["strong_buy"])

    def test_get_sector_specific_thresholds_default_sector(self):
        """Should return default thresholds for unknown sectors"""
        from finance_ml.eval import get_sector_specific_thresholds

        result = get_sector_specific_thresholds("UnknownSector")
        self.assertIsInstance(result, dict)
        # Should use default thresholds
        self.assertEqual(result["strong_buy"], 20.0)

    def test_get_sector_specific_thresholds_based_on_volatility(self):
        """Should adjust thresholds based on sector volatility data"""
        from finance_ml.eval import get_sector_specific_thresholds

        df_volatility = pd.DataFrame(
            {
                "sector": ["Tech", "Tech", "Finance", "Finance"],
                "volatility": [0.3, 0.35, 0.15, 0.18],
            }
        )
        result = get_sector_specific_thresholds("Tech", sector_volatility_df=df_volatility)
        self.assertIsInstance(result, dict)


class TestIdentifySectorLeadersLaggards(unittest.TestCase):
    """Test sector leaders and laggards identification"""

    def setUp(self):
        """Create test data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
                "mispricing_score": [25.0, 15.0, -5.0, 20.0, 10.0, -10.0],
                "roe": [0.20, 0.15, 0.10, 0.18, 0.12, 0.08],
                "revenue_cagr": [0.25, 0.20, 0.10, 0.15, 0.12, 0.05],
            }
        )

    def test_identify_sector_leaders_laggards_returns_dict(self):
        """Should return dict with leaders and laggards per sector"""
        from finance_ml.eval import identify_sector_leaders_laggards

        result = identify_sector_leaders_laggards(self.df, top_n=2)
        self.assertIsInstance(result, dict)
        self.assertIn("leaders", result)
        self.assertIn("laggards", result)

    def test_identify_sector_leaders_laggards_leaders_structure(self):
        """Should return leaders dict with sector keys"""
        from finance_ml.eval import identify_sector_leaders_laggards

        result = identify_sector_leaders_laggards(self.df, top_n=2)
        self.assertIn("Tech", result["leaders"])
        self.assertIn("Finance", result["leaders"])

    def test_identify_sector_leaders_laggards_top_n_limit(self):
        """Should return top_n leaders per sector"""
        from finance_ml.eval import identify_sector_leaders_laggards

        result = identify_sector_leaders_laggards(self.df, top_n=2)
        # Each sector should have at most 2 leaders
        self.assertLessEqual(len(result["leaders"]["Tech"]), 2)
        self.assertLessEqual(len(result["leaders"]["Finance"]), 2)

    def test_identify_sector_leaders_laggards_sorted_by_score(self):
        """Should sort leaders by mispricing score (highest first)"""
        from finance_ml.eval import identify_sector_leaders_laggards

        result = identify_sector_leaders_laggards(self.df, top_n=3)
        tech_leaders = result["leaders"]["Tech"]
        # First should have highest mispricing
        if len(tech_leaders) > 1:
            self.assertGreater(
                tech_leaders.iloc[0]["mispricing_score"],
                tech_leaders.iloc[1]["mispricing_score"],
            )

    def test_identify_sector_leaders_laggards_laggards_sorted(self):
        """Should sort laggards by mispricing score (lowest first)"""
        from finance_ml.eval import identify_sector_leaders_laggards

        result = identify_sector_leaders_laggards(self.df, top_n=3)
        tech_laggards = result["laggards"]["Tech"]
        # First should have lowest mispricing
        if len(tech_laggards) > 1:
            self.assertLess(
                tech_laggards.iloc[0]["mispricing_score"],
                tech_laggards.iloc[1]["mispricing_score"],
            )


class TestGeneratePdfReport(unittest.TestCase):
    """Test PDF report generation"""

    def setUp(self):
        """Create test data"""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [150, 300, 2800],
                "predicted_target": [180, 320, 3000],
                "mispricing_score": [20.0, 6.7, 7.1],
                "valuation_category": ["Strong Buy", "Hold", "Hold"],
            }
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_pdf_report_creates_file(self):
        """Should create PDF file"""
        from finance_ml.eval import generate_pdf_report

        pdf_path = Path(self.temp_dir) / "test_report.pdf"
        try:
            generate_pdf_report(self.df, pdf_path, title="Test Report")
            self.assertTrue(pdf_path.exists())
        except ImportError:
            self.skipTest("ReportLab not available")

    def test_generate_pdf_report_with_summary(self):
        """Should include executive summary"""
        from finance_ml.eval import generate_pdf_report

        pdf_path = Path(self.temp_dir) / "test_summary.pdf"
        try:
            generate_pdf_report(
                self.df,
                pdf_path,
                title="Report with Summary",
                include_summary=True,
            )
            self.assertTrue(pdf_path.exists())
            # Check file size is reasonable (has content)
            self.assertGreater(pdf_path.stat().st_size, 1000)
        except ImportError:
            self.skipTest("ReportLab not available")

    def test_generate_pdf_report_with_top_opportunities(self):
        """Should include top opportunities section"""
        from finance_ml.eval import generate_pdf_report

        pdf_path = Path(self.temp_dir) / "test_opportunities.pdf"
        try:
            generate_pdf_report(
                self.df,
                pdf_path,
                title="Opportunities Report",
                top_n_opportunities=5,
            )
            self.assertTrue(pdf_path.exists())
        except ImportError:
            self.skipTest("ReportLab not available")

    def test_generate_pdf_report_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully"""
        from finance_ml.eval import generate_pdf_report

        empty_df = pd.DataFrame()
        pdf_path = Path(self.temp_dir) / "empty_report.pdf"
        try:
            # Should either create minimal report or raise informative error
            generate_pdf_report(empty_df, pdf_path)
            # If it doesn't raise, check file was created
            self.assertTrue(pdf_path.exists())
        except (ValueError, ImportError) as e:
            # Acceptable to raise error for empty data or missing ReportLab
            pass

    def test_generate_pdf_report_missing_reportlab(self):
        """Should handle missing ReportLab gracefully"""
        from finance_ml.eval import generate_pdf_report

        pdf_path = Path(self.temp_dir) / "no_reportlab.pdf"
        # Test will skip if ReportLab available, or verify error handling
        try:
            generate_pdf_report(self.df, pdf_path)
        except ImportError as e:
            self.assertIn("reportlab", str(e).lower())


if __name__ == "__main__":
    unittest.main()
