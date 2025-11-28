"""
Test suite for finance_ml.ml_workflow.eda.eda module (Phase 9.2)

This module tests EDA summary functions for quick data exploration.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestEdaSummary(unittest.TestCase):
    """Test eda_summary function"""

    def setUp(self):
        """Create sample data for testing"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] * 20,
                "sector": ["Technology"] * 50 + ["Consumer"] * 50,
                "region": ["US"] * 100,
                "last_price": np.random.uniform(50, 500, 100),
                "market_cap": np.random.uniform(1e9, 1e12, 100),
                "pe_ratio": np.random.uniform(10, 50, 100),
                "revenue": np.random.uniform(1e8, 1e11, 100),
                "ebitda": np.random.uniform(1e7, 1e10, 100),
            }
        )

    def test_eda_summary_import(self):
        """Test that eda_summary can be imported from new location"""
        try:
            from finance_ml.ml_workflow.eda.eda import eda_summary

            self.assertTrue(callable(eda_summary))
        except ImportError as e:
            self.fail(f"Cannot import eda_summary: {e}")

    def test_eda_summary_returns_dict(self):
        """Test that eda_summary returns a dictionary"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        result = eda_summary(self.df)
        self.assertIsInstance(result, dict)

    def test_eda_summary_includes_basic_stats(self):
        """Test that summary includes basic statistics"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        result = eda_summary(self.df)
        self.assertIn("shape", result)
        self.assertIn("columns", result)
        self.assertIn("dtypes", result)

    def test_eda_summary_includes_missing_values(self):
        """Test that summary includes missing value analysis"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        # Add some missing values
        df_with_na = self.df.copy()
        df_with_na.loc[0:5, "pe_ratio"] = np.nan
        result = eda_summary(df_with_na)
        self.assertIn("missing_values", result)

    def test_eda_summary_includes_numeric_summary(self):
        """Test that summary includes numeric column statistics"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        result = eda_summary(self.df)
        self.assertIn("numeric_summary", result)

    def test_eda_summary_includes_categorical_summary(self):
        """Test that summary includes categorical column statistics"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        result = eda_summary(self.df)
        self.assertIn("categorical_summary", result)

    def test_eda_summary_with_sector_column(self):
        """Test that summary respects sector column parameter"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        result = eda_summary(self.df, sector_column="sector")
        # Should include sector-specific information
        self.assertIsInstance(result, dict)

    def test_eda_summary_with_correlations(self):
        """Test that summary can include correlations"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        result = eda_summary(self.df, include_correlations=True)
        self.assertIn("correlations", result)

    def test_eda_summary_handles_empty_dataframe(self):
        """Test that eda_summary handles empty dataframe gracefully"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        empty_df = pd.DataFrame()
        result = eda_summary(empty_df)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["shape"][0], 0)

    def test_eda_summary_handles_single_column(self):
        """Test that eda_summary handles single column dataframe"""
        from finance_ml.ml_workflow.eda.eda import eda_summary

        single_col_df = self.df[["last_price"]].copy()
        result = eda_summary(single_col_df)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["shape"][1], 1)


class TestEdaReports(unittest.TestCase):
    """Test reports.py module"""

    def setUp(self):
        """Create sample data and temp directory"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"] * 10,
                "sector": ["Technology"] * 30,
                "last_price": np.random.uniform(50, 500, 30),
                "market_cap": np.random.uniform(1e9, 1e12, 30),
            }
        )
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temp directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_generate_eda_report_import(self):
        """Test that generate_eda_report can be imported"""
        try:
            from finance_ml.ml_workflow.eda.reports import generate_eda_report

            self.assertTrue(callable(generate_eda_report))
        except ImportError as e:
            self.fail(f"Cannot import generate_eda_report: {e}")

    def test_generate_eda_report_returns_path(self):
        """Test that generate_eda_report returns a file path"""
        from finance_ml.ml_workflow.eda.reports import generate_eda_report

        result = generate_eda_report(self.df, out_dir=self.temp_dir)
        self.assertIsInstance(result, (str, Path))

    def test_generate_eda_report_creates_file(self):
        """Test that generate_eda_report creates an output file"""
        from finance_ml.ml_workflow.eda.reports import generate_eda_report

        result = generate_eda_report(self.df, out_dir=self.temp_dir)
        self.assertTrue(Path(result).exists())

    def test_generate_eda_report_html_format(self):
        """Test that generate_eda_report creates HTML by default"""
        from finance_ml.ml_workflow.eda.reports import generate_eda_report

        result = generate_eda_report(self.df, out_dir=self.temp_dir)
        self.assertTrue(str(result).endswith(".html"))

    def test_generate_eda_report_with_sector_column(self):
        """Test that generate_eda_report respects sector column parameter"""
        from finance_ml.ml_workflow.eda.reports import generate_eda_report

        result = generate_eda_report(self.df, sector_column="sector", out_dir=self.temp_dir)
        self.assertTrue(Path(result).exists())


if __name__ == "__main__":
    unittest.main()
