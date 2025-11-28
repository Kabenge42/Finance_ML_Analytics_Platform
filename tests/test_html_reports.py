"""
Tests for HTML report generation (Phase 9.7).

TDD Implementation: These tests are written first, before the implementation.
Target module: finance_ml/ml_workflow/analytics/html_reports.py

Test coverage target: ≥80% (per code_guidelines.md Section 6.2)
"""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np


class TestHTMLReports(unittest.TestCase):
    """Test HTML report generation functions."""

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

    def test_generate_executive_summary_section(self):
        """Test executive summary HTML generation."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_executive_summary_html,
        )

        html = generate_executive_summary_html(self.sample_df, self.sample_metrics)

        # Verify HTML structure
        self.assertIsInstance(html, str)
        self.assertIn("Executive Summary", html)
        self.assertIn("Key Findings", html)
        # Verify metrics are included
        self.assertIn("0.85", html)  # R²
        self.assertIn("12.5", html)  # MAE

    def test_generate_sector_breakdown_section(self):
        """Test sector breakdown HTML generation."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_sector_breakdown_html,
        )

        html = generate_sector_breakdown_html(self.sample_df, sector_col="sector")

        # Verify HTML structure
        self.assertIsInstance(html, str)
        self.assertIn("Sector", html)
        # Verify sectors are included
        self.assertIn("Technology", html)
        self.assertIn("Healthcare", html)
        self.assertIn("Financials", html)

    def test_generate_quality_filtered_rankings(self):
        """Test quality-filtered rankings HTML generation."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_quality_filtered_html,
        )

        html = generate_quality_filtered_html(self.sample_df, quality_threshold=0.5)

        # Verify HTML structure
        self.assertIsInstance(html, str)
        self.assertIn("Quality", html)
        # Should contain table structure
        self.assertIn("<table", html)
        self.assertIn("</table>", html)

    def test_generate_risk_warnings_section(self):
        """Test risk warnings HTML generation."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_risk_warnings_html,
        )

        html = generate_risk_warnings_html(self.sample_df)

        # Verify HTML structure
        self.assertIsInstance(html, str)
        self.assertIn("Risk", html)
        # Should include risk categories
        self.assertIn("Volatility", html)

    def test_generate_phase93_category_summary(self):
        """Test Phase 9.3 category summary HTML generation."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_phase93_summary_html,
        )

        html = generate_phase93_summary_html(self.sample_df, self.sample_category_stats)

        # Verify HTML structure
        self.assertIsInstance(html, str)
        self.assertIn("Feature", html)
        # Verify categories are included
        self.assertIn("momentum", html.lower())
        self.assertIn("valuation", html.lower())

    def test_full_html_report_structure(self):
        """Test full enhanced HTML report generation."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_enhanced_analysis_html,
        )
        from finance_ml.ml_workflow.analytics.report_config import HTMLReportConfig

        config = HTMLReportConfig(
            include_executive_summary=True,
            include_sector_breakdown=True,
            include_quality_filtered=True,
            include_risk_warnings=True,
            include_phase93_summary=True,
            top_n_stocks=20,
            quality_threshold=0.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"

            result_path = generate_enhanced_analysis_html(
                df=self.sample_df,
                output_path=output_path,
                config=config,
                model_metrics=self.sample_metrics,
                category_stats=self.sample_category_stats,
            )

            # Verify file was created
            self.assertTrue(result_path.exists())

            # Read and verify content
            content = result_path.read_text(encoding="utf-8")

            # Verify all sections are present
            self.assertIn("Executive Summary", content)
            self.assertIn("Sector", content)
            self.assertIn("Risk", content)

    def test_html_report_with_minimal_config(self):
        """Test HTML report with minimal configuration (all sections disabled)."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_enhanced_analysis_html,
        )
        from finance_ml.ml_workflow.analytics.report_config import HTMLReportConfig

        config = HTMLReportConfig(
            include_executive_summary=False,
            include_sector_breakdown=False,
            include_quality_filtered=False,
            include_risk_warnings=False,
            include_phase93_summary=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "minimal_report.html"

            result_path = generate_enhanced_analysis_html(
                df=self.sample_df,
                output_path=output_path,
                config=config,
            )

            # Verify file was created
            self.assertTrue(result_path.exists())

            # File should still be valid HTML
            content = result_path.read_text(encoding="utf-8")
            self.assertIn("<html", content)
            self.assertIn("</html>", content)

    def test_html_report_handles_empty_dataframe(self):
        """Test HTML report handles empty dataframe gracefully."""
        from finance_ml.ml_workflow.analytics.html_reports import (
            generate_enhanced_analysis_html,
        )
        from finance_ml.ml_workflow.analytics.report_config import HTMLReportConfig

        empty_df = pd.DataFrame(columns=self.sample_df.columns)
        config = HTMLReportConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty_report.html"

            result_path = generate_enhanced_analysis_html(
                df=empty_df,
                output_path=output_path,
                config=config,
            )

            # Should still create a valid file
            self.assertTrue(result_path.exists())


if __name__ == "__main__":
    unittest.main()
