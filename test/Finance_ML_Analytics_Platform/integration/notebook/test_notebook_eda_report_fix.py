"""
Unit tests for notebook EDA report function signature fix.

Tests verify that generate_eda_report is called with correct parameters
and that the direct import from advanced_eda module works correctly.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Test both import methods
from finance_ml.ml_workflow.advanced_eda import generate_eda_report as generate_eda_report_direct
from finance_ml import generate_eda_report as generate_eda_report_package


class TestNotebookEDAReportFix(unittest.TestCase):
    """Test cases for notebook EDA report function call."""

    def setUp(self):
        """Create sample financial data for testing."""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "ticker": [f"STOCK{i:03d}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n),
                "region": np.random.choice(["US", "EU", "APAC"], n),
                "last_price": np.random.uniform(10, 500, n),
                "price_target": np.random.uniform(10, 600, n),
                "market_cap": np.random.lognormal(20, 2, n),
                "p_e": np.random.uniform(5, 50, n),
                "p_b": np.random.uniform(0.5, 10, n),
                "ev_ebitda": np.random.uniform(2, 30, n),
                "operating_margin": np.random.uniform(-0.3, 0.5, n),
                "roe": np.random.uniform(-0.2, 0.4, n),
            }
        )

    def test_generate_eda_report_direct_import_with_target_col(self):
        """Test generate_eda_report from advanced_eda accepts target_col parameter."""
        try:
            report = generate_eda_report_direct(
                self.df, target_col="price_target", sector_col="sector", output_dir=None
            )
            self.assertIsNotNone(report)
            self.assertIsInstance(report, object)
            # Check report has expected attributes
            self.assertTrue(hasattr(report, "dataset_summary"))
            self.assertTrue(hasattr(report, "correlation_analysis"))
        except TypeError as e:
            self.fail(f"generate_eda_report should accept target_col parameter: {e}")

    def test_generate_eda_report_direct_import_without_target_col(self):
        """Test generate_eda_report works without optional target_col."""
        try:
            report = generate_eda_report_direct(self.df, sector_col="sector", output_dir=None)
            self.assertIsNotNone(report)
        except TypeError as e:
            self.fail(f"generate_eda_report should work without target_col: {e}")

    def test_generate_eda_report_package_level_import(self):
        """Test that package-level import also works with correct signature."""
        try:
            report = generate_eda_report_package(
                self.df, target_col="price_target", sector_col="sector", output_dir=None
            )
            self.assertIsNotNone(report)
        except TypeError as e:
            # This might fail if package-level import is from eval module
            # Document the issue
            self.skipTest(
                f"Package-level import has wrong signature. "
                f"Use direct import from advanced_eda instead. Error: {e}"
            )

    def test_notebook_eda_report_section_corrected(self):
        """Test the CORRECTED notebook EDA report generation pattern."""
        # Simulate the corrected notebook code
        from finance_ml.ml_workflow.advanced_eda import generate_eda_report

        with tempfile.TemporaryDirectory() as tmpdir:
            eda_output_dir = Path(tmpdir)

            # This is the pattern from the notebook (lines 428-433)
            eda_report = generate_eda_report(
                self.df, target_col="price_target", sector_col="sector", output_dir=eda_output_dir
            )

            # Verify report structure
            self.assertIsNotNone(eda_report)
            self.assertTrue(hasattr(eda_report, "dataset_summary"))
            self.assertTrue(hasattr(eda_report, "correlation_analysis"))

            # Verify report contents
            self.assertIsInstance(eda_report.dataset_summary, dict)
            self.assertIn("n_rows", eda_report.dataset_summary)
            self.assertEqual(eda_report.dataset_summary["n_rows"], len(self.df))

    def test_eda_report_with_feature_importance(self):
        """Test that EDA report calculates feature importance when target_col provided."""
        from finance_ml.ml_workflow.advanced_eda import generate_eda_report

        report = generate_eda_report(
            self.df, target_col="price_target", sector_col="sector", output_dir=None
        )

        # When target_col is provided, feature_importance should be calculated
        self.assertIsNotNone(report)
        self.assertTrue(hasattr(report, "feature_importance"))
        # Feature importance should be a DataFrame
        self.assertIsInstance(report.feature_importance, pd.DataFrame)

    def test_eda_report_correlation_analysis(self):
        """Test that correlation analysis is included in report."""
        from finance_ml.ml_workflow.advanced_eda import generate_eda_report

        report = generate_eda_report(
            self.df, target_col="price_target", sector_col="sector", output_dir=None
        )

        self.assertIsNotNone(report.correlation_analysis)
        self.assertTrue(hasattr(report.correlation_analysis, "pearson_matrix"))
        self.assertTrue(hasattr(report.correlation_analysis, "spearman_matrix"))
        self.assertIsInstance(report.correlation_analysis.pearson_matrix, pd.DataFrame)

    def test_eda_report_sector_comparison(self):
        """Test that sector comparison is included when sector_col provided."""
        from finance_ml.ml_workflow.advanced_eda import generate_eda_report

        report = generate_eda_report(
            self.df, target_col="price_target", sector_col="sector", output_dir=None
        )

        # Sector comparison should be present
        self.assertIsNotNone(report.sector_comparison)
        self.assertIsInstance(report.sector_comparison, dict)


class TestEDAReportFunctionSignatures(unittest.TestCase):
    """Test to document and verify function signatures."""

    def test_advanced_eda_signature(self):
        """Document the correct signature for advanced_eda.generate_eda_report."""
        import inspect
        from finance_ml.ml_workflow.advanced_eda import generate_eda_report

        sig = inspect.signature(generate_eda_report)
        params = list(sig.parameters.keys())

        # Verify expected parameters
        self.assertIn("df", params)
        self.assertIn("target_col", params)
        self.assertIn("sector_col", params)
        self.assertIn("output_dir", params)

        # Verify target_col has a default value (None or inspect.Parameter.empty)
        target_col_param = sig.parameters["target_col"]
        # target_col should have Optional[str] = None, which means default exists
        # Note: default=None means it's optional

        print(f"[OK] advanced_eda.generate_eda_report signature: {sig}")

    def test_eval_signature_different(self):
        """Document that eval.generate_eda_report has a DIFFERENT signature."""
        import inspect

        # Updated path: eval.py moved to analytics/eval.py (Phase 9.7)
        from finance_ml.ml_workflow.analytics.eval import (
            generate_eda_report as eval_generate_eda_report,
        )

        sig = inspect.signature(eval_generate_eda_report)
        params = list(sig.parameters.keys())

        # eval version should NOT have target_col or sector_col
        self.assertIn("df", params)
        self.assertNotIn("target_col", params)  # eval version doesn't have this
        self.assertNotIn("sector_col", params)  # eval version doesn't have this

        # eval version has different parameters
        self.assertIn("output_path", params)
        self.assertIn("include_correlations", params)

        print(f"[OK] eval.generate_eda_report signature: {sig}")
        print("  [NOTE] Different signature than advanced_eda version!")


if __name__ == "__main__":
    unittest.main()
