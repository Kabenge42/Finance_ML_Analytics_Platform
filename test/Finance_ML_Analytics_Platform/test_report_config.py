"""
Tests for report configuration dataclasses (Phase 9.7).

TDD Implementation: These tests are written first, before the implementation.
Target module: finance_ml/ml_workflow/analytics/report_config.py

Test coverage target: ≥80% (per code_guidelines.md Section 6.2)
"""

import unittest
from dataclasses import is_dataclass


class TestHTMLReportConfig(unittest.TestCase):
    """Test HTMLReportConfig dataclass configuration."""

    def test_html_config_is_dataclass(self):
        """Test that HTMLReportConfig is a proper dataclass."""
        from finance_ml.ml_workflow.analytics.report_config import HTMLReportConfig

        self.assertTrue(is_dataclass(HTMLReportConfig))

    def test_html_config_default_values(self):
        """Test HTMLReportConfig has correct default values."""
        from finance_ml.ml_workflow.analytics.report_config import HTMLReportConfig

        config = HTMLReportConfig()

        self.assertTrue(config.include_executive_summary)
        self.assertTrue(config.include_sector_breakdown)
        self.assertTrue(config.include_quality_filtered)
        self.assertTrue(config.include_risk_warnings)
        self.assertTrue(config.include_phase93_summary)
        self.assertEqual(config.top_n_stocks, 20)
        self.assertEqual(config.quality_threshold, 0.5)
        self.assertEqual(config.template, "modern")

    def test_html_config_custom_values(self):
        """Test HTMLReportConfig accepts custom values."""
        from finance_ml.ml_workflow.analytics.report_config import HTMLReportConfig

        config = HTMLReportConfig(
            include_executive_summary=False,
            include_sector_breakdown=False,
            include_quality_filtered=False,
            include_risk_warnings=False,
            include_phase93_summary=False,
            top_n_stocks=50,
            quality_threshold=0.7,
            template="minimal",
        )

        self.assertFalse(config.include_executive_summary)
        self.assertFalse(config.include_sector_breakdown)
        self.assertFalse(config.include_quality_filtered)
        self.assertFalse(config.include_risk_warnings)
        self.assertFalse(config.include_phase93_summary)
        self.assertEqual(config.top_n_stocks, 50)
        self.assertEqual(config.quality_threshold, 0.7)
        self.assertEqual(config.template, "minimal")


class TestExcelReportConfig(unittest.TestCase):
    """Test ExcelReportConfig dataclass configuration."""

    def test_excel_config_is_dataclass(self):
        """Test that ExcelReportConfig is a proper dataclass."""
        from finance_ml.ml_workflow.analytics.report_config import ExcelReportConfig

        self.assertTrue(is_dataclass(ExcelReportConfig))

    def test_excel_config_default_values(self):
        """Test ExcelReportConfig has correct default values."""
        from finance_ml.ml_workflow.analytics.report_config import ExcelReportConfig

        config = ExcelReportConfig()

        self.assertTrue(config.include_executive_summary)
        self.assertTrue(config.include_quality_opportunities)
        self.assertTrue(config.include_sector_leaders)
        self.assertTrue(config.include_risk_assessment)
        self.assertTrue(config.include_phase93_analysis)
        self.assertEqual(config.top_n_per_sector, 5)
        self.assertEqual(config.quality_threshold, 0.5)
        self.assertTrue(config.embed_visualizations)

    def test_excel_config_custom_values(self):
        """Test ExcelReportConfig accepts custom values."""
        from finance_ml.ml_workflow.analytics.report_config import ExcelReportConfig

        config = ExcelReportConfig(
            include_executive_summary=False,
            include_quality_opportunities=False,
            include_sector_leaders=False,
            include_risk_assessment=False,
            include_phase93_analysis=False,
            top_n_per_sector=10,
            quality_threshold=0.8,
            embed_visualizations=False,
        )

        self.assertFalse(config.include_executive_summary)
        self.assertFalse(config.include_quality_opportunities)
        self.assertFalse(config.include_sector_leaders)
        self.assertFalse(config.include_risk_assessment)
        self.assertFalse(config.include_phase93_analysis)
        self.assertEqual(config.top_n_per_sector, 10)
        self.assertEqual(config.quality_threshold, 0.8)
        self.assertFalse(config.embed_visualizations)


class TestReportConstants(unittest.TestCase):
    """Test report configuration constants."""

    def test_report_top_n_default_constant(self):
        """Test REPORT_TOP_N_DEFAULT constant exists and has correct value."""
        from finance_ml.ml_workflow.analytics.report_config import REPORT_TOP_N_DEFAULT

        self.assertEqual(REPORT_TOP_N_DEFAULT, 50)

    def test_quality_threshold_default_constant(self):
        """Test QUALITY_THRESHOLD_DEFAULT constant exists and has correct value."""
        from finance_ml.ml_workflow.analytics.report_config import QUALITY_THRESHOLD_DEFAULT

        self.assertEqual(QUALITY_THRESHOLD_DEFAULT, 0.5)

    def test_risk_zscore_threshold_constant(self):
        """Test RISK_ZSCORE_THRESHOLD constant exists and has correct value."""
        from finance_ml.ml_workflow.analytics.report_config import RISK_ZSCORE_THRESHOLD

        self.assertEqual(RISK_ZSCORE_THRESHOLD, 2.0)

    def test_distress_score_threshold_constant(self):
        """Test DISTRESS_SCORE_THRESHOLD constant exists and has correct value."""
        from finance_ml.ml_workflow.analytics.report_config import DISTRESS_SCORE_THRESHOLD

        self.assertEqual(DISTRESS_SCORE_THRESHOLD, 0.7)


if __name__ == "__main__":
    unittest.main()
