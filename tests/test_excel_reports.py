"""
Tests for Excel report generation (Phase 9.7).

TDD Implementation: These tests are written first, before the implementation.
Target module: finance_ml/ml_workflow/analytics/excel_reports.py

Test coverage target: ≥80% (per code_guidelines.md Section 6.2)
"""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np


class TestExcelReports(unittest.TestCase):
    """Test Excel report generation functions."""

    @classmethod
    def setUpClass(cls):
        """Create sample dataframe for testing."""
        np.random.seed(42)
        n_stocks = 100

        cls.sample_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_stocks)],
                "sector": np.random.choice(
                    ["Technology", "Healthcare", "Financials", "Energy", "Consumer"], n_stocks
                ),
                "region": np.random.choice(["US", "EU", "APAC"], n_stocks),
                "last_price": np.random.uniform(10, 500, n_stocks),
                "price_target": np.random.uniform(10, 600, n_stocks),
                "mispricing_score": np.random.uniform(-0.5, 0.5, n_stocks),
                "quality_score": np.random.uniform(0, 1, n_stocks),
                "profitability_score": np.random.uniform(0, 1, n_stocks),
                "volatility": np.random.uniform(0.1, 0.8, n_stocks),
                "z_score": np.random.uniform(-3, 3, n_stocks),
                "distress_score": np.random.uniform(0, 1, n_stocks),
                "market_cap": np.random.uniform(1e9, 1e12, n_stocks),
            }
        )

        cls.sample_metrics = {
            "r2": 0.85,
            "mae": 12.5,
            "rmse": 18.3,
            "mape": 0.08,
        }

        cls.sample_category_stats = {
            "momentum": {"feature_count": 15, "coverage_pct": 0.92, "avg_value": 0.45},
            "valuation": {"feature_count": 12, "coverage_pct": 0.88, "avg_value": 0.52},
            "profitability": {"feature_count": 10, "coverage_pct": 0.95, "avg_value": 0.61},
            "quality_risk": {"feature_count": 8, "coverage_pct": 0.78, "avg_value": 0.38},
        }

    def test_create_executive_summary_sheet(self):
        """Test executive summary sheet creation."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            create_executive_summary_sheet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_exec_summary.xlsx"

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                create_executive_summary_sheet(writer, self.sample_df, self.sample_metrics)

            # Verify file was created and has the sheet
            self.assertTrue(output_path.exists())

            # Read back and verify (use context manager to ensure file is closed)
            with pd.ExcelFile(output_path) as xl:
                self.assertIn("Executive_Summary", xl.sheet_names)
                df_summary = pd.read_excel(xl, sheet_name="Executive_Summary")
                self.assertGreater(len(df_summary), 0)

    def test_create_sector_leaders_laggards_sheet(self):
        """Test sector leaders sheet creation."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            create_sector_leaders_sheet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_leaders.xlsx"

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                create_sector_leaders_sheet(writer, self.sample_df, top_n=5)

            # Verify file was created
            self.assertTrue(output_path.exists())

            with pd.ExcelFile(output_path) as xl:
                self.assertIn("Sector_Leaders", xl.sheet_names)
                df_leaders = pd.read_excel(xl, sheet_name="Sector_Leaders")
                # Should have top 5 per sector
                sectors = self.sample_df["sector"].nunique()
                self.assertLessEqual(len(df_leaders), sectors * 5)

    def test_create_quality_filtered_opportunities_sheet(self):
        """Test quality-filtered opportunities sheet creation."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            create_quality_opportunities_sheet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_quality.xlsx"

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                create_quality_opportunities_sheet(writer, self.sample_df, quality_threshold=0.5)

            # Verify file was created
            self.assertTrue(output_path.exists())

            with pd.ExcelFile(output_path) as xl:
                self.assertIn("Quality_Opportunities", xl.sheet_names)

    def test_create_risk_assessment_sheet(self):
        """Test risk assessment sheet creation."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            create_risk_assessment_sheet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_risk.xlsx"

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                create_risk_assessment_sheet(writer, self.sample_df)

            # Verify file was created
            self.assertTrue(output_path.exists())

            with pd.ExcelFile(output_path) as xl:
                self.assertIn("Risk_Assessment", xl.sheet_names)

    def test_create_phase93_analysis_sheet(self):
        """Test Phase 9.3 analysis sheet creation."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            create_phase93_analysis_sheet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_phase93.xlsx"

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                create_phase93_analysis_sheet(writer, self.sample_df, self.sample_category_stats)

            # Verify file was created
            self.assertTrue(output_path.exists())

            with pd.ExcelFile(output_path) as xl:
                self.assertIn("Phase93_Analysis", xl.sheet_names)
                df_phase93 = pd.read_excel(xl, sheet_name="Phase93_Analysis")
                # Should have one row per category
                self.assertEqual(len(df_phase93), len(self.sample_category_stats))

    def test_full_excel_report_has_all_sheets(self):
        """Test full enhanced Excel report has all required sheets."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            generate_enhanced_excel_report,
        )
        from finance_ml.ml_workflow.analytics.report_config import ExcelReportConfig

        config = ExcelReportConfig(
            include_executive_summary=True,
            include_quality_opportunities=True,
            include_sector_leaders=True,
            include_risk_assessment=True,
            include_phase93_analysis=True,
            top_n_per_sector=5,
            quality_threshold=0.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full_report.xlsx"

            result_path = generate_enhanced_excel_report(
                df=self.sample_df,
                output_path=output_path,
                config=config,
                model_metrics=self.sample_metrics,
                category_stats=self.sample_category_stats,
            )

            # Verify file was created
            self.assertTrue(result_path.exists())

            # Read and verify all sheets
            with pd.ExcelFile(result_path) as xl:
                expected_sheets = [
                    "Executive_Summary",
                    "Quality_Opportunities",
                    "Sector_Leaders",
                    "Sector_Laggards",
                    "Risk_Assessment",
                    "Phase93_Analysis",
                ]

                for sheet in expected_sheets:
                    self.assertIn(sheet, xl.sheet_names, f"Missing sheet: {sheet}")

    def test_excel_report_with_minimal_config(self):
        """Test Excel report with minimal configuration (all sheets disabled)."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            generate_enhanced_excel_report,
        )
        from finance_ml.ml_workflow.analytics.report_config import ExcelReportConfig

        config = ExcelReportConfig(
            include_executive_summary=False,
            include_quality_opportunities=False,
            include_sector_leaders=False,
            include_risk_assessment=False,
            include_phase93_analysis=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "minimal_report.xlsx"

            result_path = generate_enhanced_excel_report(
                df=self.sample_df,
                output_path=output_path,
                config=config,
            )

            # Verify file was created (should still have at least one sheet)
            self.assertTrue(result_path.exists())

    def test_excel_report_handles_empty_dataframe(self):
        """Test Excel report handles empty dataframe gracefully."""
        from finance_ml.ml_workflow.analytics.excel_reports import (
            generate_enhanced_excel_report,
        )
        from finance_ml.ml_workflow.analytics.report_config import ExcelReportConfig

        empty_df = pd.DataFrame(columns=self.sample_df.columns)
        config = ExcelReportConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty_report.xlsx"

            result_path = generate_enhanced_excel_report(
                df=empty_df,
                output_path=output_path,
                config=config,
            )

            # Should still create a valid file
            self.assertTrue(result_path.exists())


if __name__ == "__main__":
    unittest.main()
