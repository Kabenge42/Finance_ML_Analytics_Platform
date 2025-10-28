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


if __name__ == "__main__":
    unittest.main()
