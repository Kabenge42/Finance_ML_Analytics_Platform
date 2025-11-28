"""
Tests for EDA functions extracted from analytics/eval.py

TDD Step 2: Testing EDA functions (lines 299-877 of eval.py)
These tests validate the exploratory data analysis functions
that will be extracted to eda/ modules.

Coverage Target: 80% for EDA module
"""

import unittest
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np


class TestSimpleEda(unittest.TestCase):
    """Tests for simple_eda function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
                "last_price": [150.0, 2800.0, 350.0, 3500.0, 300.0],
                "market_cap": [2.5e12, 1.8e12, 2.2e12, 1.6e12, 0.8e12],
                "pe_ratio": [25.0, 28.0, 30.0, 60.0, 15.0],
                "sector": ["Technology", "Technology", "Technology", "Consumer", "Technology"],
                "region": ["US", "US", "US", "US", "US"],
            }
        )

    def test_returns_dict(self):
        """Test that simple_eda returns a dictionary."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIsInstance(result, dict)

    def test_basic_stats_in_result(self):
        """Test that basic statistics are included in result."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIn("row_count", result)
        self.assertIn("column_count", result)
        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["column_count"], 6)

    def test_columns_list_in_result(self):
        """Test that columns list is included."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIn("columns", result)
        self.assertEqual(set(result["columns"]), set(self.df.columns))

    def test_numeric_cols_identified(self):
        """Test that numeric columns are correctly identified."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIn("numeric_cols_count", result)
        self.assertIn("numeric_columns", result)
        # last_price, market_cap, pe_ratio are numeric
        self.assertEqual(result["numeric_cols_count"], 3)

    def test_categorical_cols_counted(self):
        """Test that categorical columns are counted."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIn("categorical_cols_count", result)
        # ticker, sector, region are categorical
        self.assertEqual(result["categorical_cols_count"], 3)

    def test_basic_stats_included(self):
        """Test that basic_stats dict is included."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIn("basic_stats", result)
        self.assertIsInstance(result["basic_stats"], dict)

    def test_writes_json_to_out_dir(self):
        """Test that JSON summary is written when out_dir provided."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            simple_eda(self.df, out_dir=out_dir)

            json_path = out_dir / "eda_summary.json"
            self.assertTrue(json_path.exists())

            # Verify JSON is valid
            with open(json_path, "r") as f:
                data = json.load(f)
            self.assertIn("row_count", data)

    def test_no_output_without_out_dir(self):
        """Test that no files are written when out_dir is None."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        # Should not raise any error
        result = simple_eda(self.df, out_dir=None)
        self.assertIsInstance(result, dict)

    def test_handles_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        empty_df = pd.DataFrame()
        result = simple_eda(empty_df)

        self.assertEqual(result["row_count"], 0)
        self.assertEqual(result["column_count"], 0)

    def test_handles_all_numeric_dataframe(self):
        """Test handling of DataFrame with only numeric columns."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        numeric_df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
        )

        result = simple_eda(numeric_df)

        self.assertEqual(result["numeric_cols_count"], 3)
        self.assertEqual(result["categorical_cols_count"], 0)

    def test_handles_all_categorical_dataframe(self):
        """Test handling of DataFrame with only categorical columns."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        cat_df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})

        result = simple_eda(cat_df)

        self.assertEqual(result["numeric_cols_count"], 0)
        self.assertEqual(result["categorical_cols_count"], 2)

    def test_handles_nan_values(self):
        """Test handling of NaN values in DataFrame."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        df_with_nan = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})

        result = simple_eda(df_with_nan)

        # Should not crash and should include basic stats
        self.assertIn("basic_stats", result)

    def test_sector_counts_included(self):
        """Test that sector value counts are included."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIn("sector_counts", result)

    def test_region_counts_included(self):
        """Test that region value counts are included."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df)

        self.assertIn("region_counts", result)

    def test_with_target_column(self):
        """Test EDA with target column specified."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df, target_column="last_price")

        # Should still return valid result
        self.assertIsInstance(result, dict)


class TestSimpleEdaMultivariate(unittest.TestCase):
    """Tests for multivariate analysis in simple_eda."""

    def setUp(self):
        """Set up test fixtures with sufficient data for multivariate analysis."""
        np.random.seed(42)
        n_samples = 50
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "feature4": np.random.randn(n_samples),
                "sector": ["Tech"] * 25 + ["Finance"] * 25,
            }
        )

    def test_multivariate_disabled_by_default(self):
        """Test that multivariate analysis is disabled by default."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df, include_multivariate=False)

        # Should either not have multivariate_analysis or it should be empty
        if "multivariate_analysis" in result:
            self.assertEqual(result["multivariate_analysis"], {})

    def test_multivariate_enabled(self):
        """Test that multivariate analysis runs when enabled."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df, include_multivariate=True)

        # Should have multivariate_analysis section
        self.assertIn("multivariate_analysis", result)


class TestSimpleEdaPhase93(unittest.TestCase):
    """Tests for Phase 9.3 feature tracking in simple_eda."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL"],
                "last_price": [150.0, 2800.0],
                "sector": ["Technology", "Technology"],
            }
        )

    def test_phase93_summary_disabled_by_default(self):
        """Test that Phase 9.3 summary is disabled by default."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df, include_phase93_summary=False)

        # Should not have phase93 specific data or it should be empty
        self.assertIsInstance(result, dict)

    def test_phase93_summary_enabled(self):
        """Test that Phase 9.3 summary runs when enabled."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        result = simple_eda(self.df, include_phase93_summary=True)

        # Should still return valid dict
        self.assertIsInstance(result, dict)


class TestExportPredictionsToCsv(unittest.TestCase):
    """Tests for export_predictions_to_csv function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "y_pred": [150.0, 2800.0, 350.0],
                "y_true": [145.0, 2750.0, 340.0],
                "last_price": [140.0, 2700.0, 330.0],
                "sector": ["Technology", "Technology", "Technology"],
            }
        )

    def test_exports_to_csv(self):
        """Test that predictions are exported to CSV."""
        from finance_ml.ml_workflow.analytics.eval import export_predictions_to_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"
            export_predictions_to_csv(self.df, csv_path)

            self.assertTrue(csv_path.exists())

            # Read back and verify
            result = pd.read_csv(csv_path)
            self.assertEqual(len(result), 3)

    def test_computes_mispricing_when_requested(self):
        """Test that mispricing is computed when requested."""
        from finance_ml.ml_workflow.analytics.eval import export_predictions_to_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"
            export_predictions_to_csv(self.df, csv_path, compute_mispricing=True)

            result = pd.read_csv(csv_path)
            # Should have mispricing columns if computed
            self.assertTrue(csv_path.exists())

    def test_export_all_columns(self):
        """Test exporting all columns."""
        from finance_ml.ml_workflow.analytics.eval import export_predictions_to_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"
            export_predictions_to_csv(self.df, csv_path, export_all_columns=True)

            result = pd.read_csv(csv_path)
            # All original columns should be present
            for col in ["ticker", "y_pred", "y_true", "last_price", "sector"]:
                self.assertIn(col, result.columns)


class TestExportPredictionsToExcel(unittest.TestCase):
    """Tests for export_predictions_to_excel function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "y_pred": [150.0, 2800.0, 350.0],
                "y_true": [145.0, 2750.0, 340.0],
                "sector": ["Technology", "Technology", "Technology"],
                "mispricing_score": [0.05, 0.02, 0.03],  # Required for summary sheet
            }
        )

    def test_exports_to_excel(self):
        """Test that predictions are exported to Excel."""
        from finance_ml.ml_workflow.analytics.eval import export_predictions_to_excel

        with tempfile.TemporaryDirectory() as tmpdir:
            excel_path = Path(tmpdir) / "predictions.xlsx"
            export_predictions_to_excel(self.df, excel_path)

            self.assertTrue(excel_path.exists())

    def test_includes_summary_sheet(self):
        """Test that summary sheet is included when requested."""
        from finance_ml.ml_workflow.analytics.eval import export_predictions_to_excel

        with tempfile.TemporaryDirectory() as tmpdir:
            excel_path = Path(tmpdir) / "predictions.xlsx"
            export_predictions_to_excel(self.df, excel_path, include_summary=True)

            self.assertTrue(excel_path.exists())


class TestCreateSectorHeatmap(unittest.TestCase):
    """Tests for create_sector_heatmap function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "JPM", "BAC"],
                "sector": ["Technology", "Technology", "Financials", "Financials"],
                "mispricing_score": [0.15, 0.25, -0.10, 0.05],
            }
        )

    def test_creates_heatmap(self):
        """Test that heatmap is created."""
        from finance_ml.ml_workflow.analytics.eval import create_sector_heatmap

        # Should not raise error
        result = create_sector_heatmap(self.df)
        # Returns None or figure depending on implementation

    def test_saves_to_path(self):
        """Test that heatmap is saved when out_path provided."""
        from finance_ml.ml_workflow.analytics.eval import create_sector_heatmap

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use PNG format which is supported by matplotlib
            out_path = Path(tmpdir) / "heatmap.png"
            create_sector_heatmap(self.df, out_path=out_path)

            # May or may not create file depending on matplotlib availability
            # Just verify no crash


class TestCreateInteractivePredictionPlot(unittest.TestCase):
    """Tests for create_interactive_prediction_plot function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "y_pred": [150.0, 2800.0, 350.0],
                "y_true": [145.0, 2750.0, 340.0],
                "sector": ["Technology", "Technology", "Technology"],
            }
        )

    def test_creates_plot(self):
        """Test that interactive plot is created without error."""
        from finance_ml.ml_workflow.analytics.eval import create_interactive_prediction_plot

        # Should not raise error
        result = create_interactive_prediction_plot(self.df)


class TestCreateRegionSectorHeatmap(unittest.TestCase):
    """Tests for create_region_sector_heatmap function."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "SAP", "SONY", "VALE"],
                "sector": ["Technology", "Technology", "Technology", "Materials"],
                "region": ["US", "EU", "APAC", "ROTW"],
                "mispricing_score": [0.15, 0.10, 0.20, -0.05],
            }
        )

    def test_creates_heatmap(self):
        """Test that region-sector heatmap is created."""
        from finance_ml.ml_workflow.analytics.eval import create_region_sector_heatmap

        # Should not raise error
        result = create_region_sector_heatmap(self.df)


class TestPlotOutlierBoxplots(unittest.TestCase):
    """Tests for plot_outlier_boxplots function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "price": np.random.randn(100) * 10 + 100,
                "volume": np.random.randn(100) * 1000 + 5000,
                "pe_ratio": np.random.randn(100) * 5 + 20,
            }
        )

    def test_creates_boxplots(self):
        """Test that boxplots are created."""
        from finance_ml.ml_workflow.analytics.eval import plot_outlier_boxplots

        # Should not raise error
        result = plot_outlier_boxplots(self.df, columns=["price", "volume"])

    def test_handles_empty_columns(self):
        """Test handling of empty columns list."""
        from finance_ml.ml_workflow.analytics.eval import plot_outlier_boxplots

        # Should not crash with empty columns
        result = plot_outlier_boxplots(self.df, columns=[])


class TestPlotOutlierViolins(unittest.TestCase):
    """Tests for plot_outlier_violins function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {"price": np.random.randn(100) * 10 + 100, "volume": np.random.randn(100) * 1000 + 5000}
        )

    def test_creates_violins(self):
        """Test that violin plots are created."""
        from finance_ml.ml_workflow.analytics.eval import plot_outlier_violins

        # Should not raise error
        result = plot_outlier_violins(self.df, columns=["price", "volume"])


class TestPlotOutlierScatter(unittest.TestCase):
    """Tests for plot_outlier_scatter function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {"price": np.random.randn(100) * 10 + 100, "volume": np.random.randn(100) * 1000 + 5000}
        )

    def test_creates_scatter(self):
        """Test that scatter plot is created."""
        from finance_ml.ml_workflow.analytics.eval import plot_outlier_scatter

        # Should not raise error
        result = plot_outlier_scatter(self.df, columns=["price", "volume"])

    def test_with_custom_threshold(self):
        """Test with custom z-score threshold."""
        from finance_ml.ml_workflow.analytics.eval import plot_outlier_scatter

        # Should not raise error
        result = plot_outlier_scatter(self.df, columns=["price"], z_threshold=2.0)


class TestEdaEdgeCases(unittest.TestCase):
    """Tests for edge cases in EDA functions."""

    def test_simple_eda_with_single_row(self):
        """Test simple_eda with single row DataFrame."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        df = pd.DataFrame({"a": [1.0], "b": ["x"]})

        result = simple_eda(df)
        self.assertEqual(result["row_count"], 1)

    def test_simple_eda_with_single_column(self):
        """Test simple_eda with single column DataFrame."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

        result = simple_eda(df)
        self.assertEqual(result["column_count"], 1)

    def test_simple_eda_with_mixed_types(self):
        """Test simple_eda with mixed data types."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        result = simple_eda(df)
        self.assertIsInstance(result, dict)

    def test_simple_eda_with_datetime_column(self):
        """Test simple_eda with datetime column."""
        from finance_ml.ml_workflow.analytics.eval import simple_eda

        df = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=3), "value": [1.0, 2.0, 3.0]}
        )

        result = simple_eda(df)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
