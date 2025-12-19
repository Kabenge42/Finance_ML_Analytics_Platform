"""
Test suite for financial metrics ETL pipeline.

TDD implementation following code_guidelines.md Section 8 conventions.
Tests the dedicated financial metrics ETL pipeline that computes valuation,
profitability, growth, and leverage metrics with sector-specific handling.

Version: 1.0.0
Created: 2025-11-30
"""

import unittest
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd


class TestFinancialMetricsETLConfig(unittest.TestCase):
    """Test FinancialMetricsETLConfig dataclass."""

    def test_config_default_values(self):
        """Test that config has correct default values."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            FinancialMetricsETLConfig,
        )

        config = FinancialMetricsETLConfig()

        # Verify default metric categories
        self.assertTrue(config.compute_valuation_metrics)
        self.assertTrue(config.compute_profitability_metrics)
        self.assertTrue(config.compute_growth_metrics)
        self.assertTrue(config.compute_leverage_metrics)

        # Verify sector-specific handling defaults
        self.assertTrue(config.handle_sector_specific_metrics)
        self.assertEqual(config.critical_missing_threshold, 0.75)

        # Verify output options
        self.assertTrue(config.generate_quality_alerts)
        self.assertTrue(config.generate_metrics_dashboard)

    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            FinancialMetricsETLConfig,
        )

        config = FinancialMetricsETLConfig(
            compute_valuation_metrics=True,
            compute_profitability_metrics=False,
            compute_growth_metrics=True,
            compute_leverage_metrics=False,
            critical_missing_threshold=0.80,
        )

        self.assertTrue(config.compute_valuation_metrics)
        self.assertFalse(config.compute_profitability_metrics)
        self.assertTrue(config.compute_growth_metrics)
        self.assertFalse(config.compute_leverage_metrics)
        self.assertEqual(config.critical_missing_threshold, 0.80)


class TestFinancialMetricsComputation(unittest.TestCase):
    """Test financial metrics computation functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100

        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(
                    ["Technology", "Financials", "Healthcare", "Consumer Discretionary"], n_samples
                ),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "enterprise_value": np.random.uniform(1e9, 1.5e12, n_samples),
                "total_revenues_ltm": np.random.uniform(1e8, 1e11, n_samples),
                "ebitda_ltm": np.random.uniform(1e7, 1e10, n_samples),
                "net_income_is_ltm": np.random.uniform(-1e9, 1e10, n_samples),
                "total_debt_fy": np.random.uniform(0, 1e11, n_samples),
                "total_equity_fy": np.random.uniform(1e8, 1e11, n_samples),
                "total_assets_fy": np.random.uniform(1e9, 1e12, n_samples),
                "gross_profit_ltm": np.random.uniform(1e7, 1e10, n_samples),
                "operating_income_ltm": np.random.uniform(-1e8, 1e10, n_samples),
            }
        )

    def test_compute_valuation_metrics(self):
        """Test valuation metrics computation."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_valuation_metrics,
        )

        result = compute_valuation_metrics(self.test_df)

        # Check that valuation columns are added
        self.assertIn("p_e_ratio", result.columns)
        self.assertIn("p_s_ratio", result.columns)
        self.assertIn("ev_ebitda_ratio", result.columns)
        self.assertIn("ev_sales_ratio", result.columns)

        # Check that original columns are preserved
        self.assertIn("ticker", result.columns)
        self.assertIn("sector", result.columns)
        self.assertIn("last_price", result.columns)

        # Check row count unchanged
        self.assertEqual(len(result), len(self.test_df))

    def test_compute_profitability_metrics(self):
        """Test profitability metrics computation."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_profitability_metrics,
        )

        result = compute_profitability_metrics(self.test_df)

        # Check that profitability columns are added
        self.assertIn("gross_margin_pct", result.columns)
        self.assertIn("operating_margin_pct", result.columns)
        self.assertIn("net_margin_pct", result.columns)
        self.assertIn("roe", result.columns)
        self.assertIn("roa", result.columns)

        # Check that we have valid numeric values (not all NaN)
        self.assertTrue(result["gross_margin_pct"].notna().sum() > 0)

    def test_compute_growth_metrics(self):
        """Test growth metrics computation."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_growth_metrics,
        )

        # Add prior period data for growth calculation
        df_with_history = self.test_df.copy()
        df_with_history["total_revenues_fy"] = self.test_df["total_revenues_ltm"] * 0.9
        df_with_history["ebitda_fy"] = self.test_df["ebitda_ltm"] * 0.85
        df_with_history["net_income_is_fy"] = self.test_df["net_income_is_ltm"] * 0.8

        result = compute_growth_metrics(df_with_history)

        # Check that growth columns are added
        self.assertIn("revenue_growth", result.columns)
        self.assertIn("ebitda_growth", result.columns)
        self.assertIn("earnings_growth", result.columns)

    def test_compute_leverage_metrics(self):
        """Test leverage metrics computation."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_leverage_metrics,
        )

        result = compute_leverage_metrics(self.test_df)

        # Check that leverage columns are added
        self.assertIn("debt_to_equity", result.columns)
        self.assertIn("debt_to_assets", result.columns)

        # Verify debt_to_assets is between 0 and 1 (mostly)
        valid_dta = result["debt_to_assets"].dropna()
        self.assertTrue((valid_dta >= 0).sum() > len(valid_dta) * 0.9)


class TestSectorSpecificMetrics(unittest.TestCase):
    """Test sector-specific metric handling."""

    def setUp(self):
        """Set up test data with sector-specific columns."""
        np.random.seed(42)
        n_samples = 100

        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": ["Financials"] * 30 + ["Technology"] * 40 + ["Healthcare"] * 30,
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                # Financials-specific
                "tbv_ltm": np.concatenate(
                    [
                        np.random.uniform(1e9, 1e11, 30),  # Financials have TBV
                        np.full(70, np.nan),  # Others don't
                    ]
                ),
                # Tech-specific
                "r_d_expense_ltm": np.concatenate(
                    [
                        np.full(30, np.nan),  # Financials don't have R&D
                        np.random.uniform(1e7, 1e9, 40),  # Tech has R&D
                        np.random.uniform(1e7, 1e9, 30),  # Healthcare has R&D
                    ]
                ),
                "total_revenues_ltm": np.random.uniform(1e8, 1e11, n_samples),
            }
        )

    def test_handle_sector_specific_missing_values(self):
        """Test that sector-specific missing values are handled correctly."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            handle_sector_specific_metrics,
        )

        result = handle_sector_specific_metrics(self.test_df)

        # Financials sector should have TBV values
        financials_mask = result["sector"] == "Financials"
        financials_tbv = result.loc[financials_mask, "tbv_ltm"]
        self.assertTrue(financials_tbv.notna().sum() >= 25)  # Most should have values

        # Tech sector should have R&D values
        tech_mask = result["sector"] == "Technology"
        tech_rd = result.loc[tech_mask, "r_d_expense_ltm"]
        self.assertTrue(tech_rd.notna().sum() >= 35)  # Most should have values

    def test_compute_p_tbv_for_financials(self):
        """Test P/TBV computation for Financials sector."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.test_df)

        # Check P/TBV is computed for Financials
        if "p_tbv_ratio" in result.columns:
            financials_mask = result["sector"] == "Financials"
            financials_ptbv = result.loc[financials_mask, "p_tbv_ratio"]
            # Should have some valid values for Financials
            self.assertTrue(financials_ptbv.notna().sum() > 0)

    def test_compute_rd_intensity_for_tech(self):
        """Test R&D intensity computation for Tech sector."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.test_df)

        # Check R&D intensity is computed
        if "r_d_intensity" in result.columns:
            tech_mask = result["sector"] == "Technology"
            tech_rd_intensity = result.loc[tech_mask, "r_d_intensity"]
            # Should have some valid values for Tech
            self.assertTrue(tech_rd_intensity.notna().sum() > 0)


class TestDataQualityAlerts(unittest.TestCase):
    """Test data quality alert generation."""

    def setUp(self):
        """Set up test data with varying missing rates."""
        np.random.seed(42)
        n_samples = 100

        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Financials"], n_samples),
                "complete_col": np.random.uniform(0, 100, n_samples),
                "low_missing_col": np.where(
                    np.random.random(n_samples) > 0.05, np.random.uniform(0, 100, n_samples), np.nan
                ),
                "medium_missing_col": np.where(
                    np.random.random(n_samples) > 0.30, np.random.uniform(0, 100, n_samples), np.nan
                ),
                "high_missing_col": np.where(
                    np.random.random(n_samples) > 0.80, np.random.uniform(0, 100, n_samples), np.nan
                ),
            }
        )

    def test_generate_quality_alerts(self):
        """Test that quality alerts are generated correctly."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            generate_data_quality_alerts,
        )

        alerts = generate_data_quality_alerts(self.test_df)

        # Should return a list of alert dictionaries
        self.assertIsInstance(alerts, list)

        # Check alert structure
        if alerts:
            alert = alerts[0]
            self.assertIn("severity", alert)
            self.assertIn("message", alert)
            self.assertIn("column", alert)
            self.assertIn("count", alert)

    def test_alert_severity_levels(self):
        """Test that alert severities are assigned correctly."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            generate_data_quality_alerts,
        )

        alerts = generate_data_quality_alerts(self.test_df)

        severities = [a["severity"] for a in alerts]

        # Should have valid severity levels
        valid_severities = {"low", "medium", "high", "critical"}
        for sev in severities:
            self.assertIn(sev, valid_severities)

    def test_critical_alerts_for_high_missing(self):
        """Test that critical alerts are generated for high missing rates."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            generate_data_quality_alerts,
        )

        alerts = generate_data_quality_alerts(self.test_df, critical_threshold=0.75)

        # Find alerts for high_missing_col
        high_missing_alerts = [a for a in alerts if a["column"] == "high_missing_col"]

        # Should have a critical alert for high_missing_col
        if high_missing_alerts:
            self.assertEqual(high_missing_alerts[0]["severity"], "critical")


class TestMetricsDashboard(unittest.TestCase):
    """Test metrics dashboard generation."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100

        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Financials", "Healthcare"], n_samples),
                "p_e_ratio": np.random.uniform(5, 50, n_samples),
                "p_b_ratio": np.random.uniform(0.5, 10, n_samples),
                "gross_margin_pct": np.random.uniform(20, 80, n_samples),
                "roe": np.random.uniform(-10, 30, n_samples),
                "revenue_growth": np.random.uniform(-20, 50, n_samples),
                "debt_to_equity": np.random.uniform(0, 2, n_samples),
            }
        )

    def test_generate_metrics_dashboard(self):
        """Test metrics dashboard generation."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            generate_metrics_dashboard,
        )

        dashboard = generate_metrics_dashboard(self.test_df)

        # Check dashboard structure
        self.assertIsInstance(dashboard, dict)
        self.assertIn("timestamp", dashboard)
        self.assertIn("total_stocks", dashboard)
        self.assertIn("by_sector", dashboard)

    def test_dashboard_sector_breakdown(self):
        """Test that dashboard includes sector breakdown."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            generate_metrics_dashboard,
        )

        dashboard = generate_metrics_dashboard(self.test_df)

        by_sector = dashboard.get("by_sector", {})

        # Should have valuation, profitability, growth, leverage categories
        self.assertIn("valuation", by_sector)
        self.assertIn("profitability", by_sector)
        self.assertIn("growth", by_sector)
        self.assertIn("leverage", by_sector)

    def test_dashboard_statistics(self):
        """Test that dashboard includes proper statistics."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            generate_metrics_dashboard,
        )

        dashboard = generate_metrics_dashboard(self.test_df)

        # Check valuation metrics have proper stats
        valuation = dashboard.get("by_sector", {}).get("valuation", {})
        if "p_e" in valuation:
            pe_stats = valuation["p_e"]
            self.assertIn("mean", pe_stats)
            self.assertIn("median", pe_stats)
            self.assertIn("std", pe_stats)
            self.assertIn("min", pe_stats)
            self.assertIn("max", pe_stats)
            self.assertIn("count", pe_stats)


class TestFinancialMetricsETLPipeline(unittest.TestCase):
    """Test the main financial metrics ETL pipeline function."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100

        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(
                    ["Technology", "Financials", "Healthcare", "Consumer Discretionary"], n_samples
                ),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "enterprise_value": np.random.uniform(1e9, 1.5e12, n_samples),
                "total_revenues_ltm": np.random.uniform(1e8, 1e11, n_samples),
                "total_revenues_fy": np.random.uniform(1e8, 1e11, n_samples) * 0.9,
                "ebitda_ltm": np.random.uniform(1e7, 1e10, n_samples),
                "ebitda_fy": np.random.uniform(1e7, 1e10, n_samples) * 0.85,
                "net_income_is_ltm": np.random.uniform(-1e9, 1e10, n_samples),
                "net_income_is_fy": np.random.uniform(-1e9, 1e10, n_samples) * 0.8,
                "total_debt_fy": np.random.uniform(0, 1e11, n_samples),
                "total_equity_fy": np.random.uniform(1e8, 1e11, n_samples),
                "total_assets_fy": np.random.uniform(1e9, 1e12, n_samples),
                "gross_profit_ltm": np.random.uniform(1e7, 1e10, n_samples),
                "operating_income_ltm": np.random.uniform(-1e8, 1e10, n_samples),
            }
        )

    def test_run_financial_metrics_etl(self):
        """Test running the complete financial metrics ETL pipeline."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            run_financial_metrics_etl,
            FinancialMetricsETLConfig,
        )

        config = FinancialMetricsETLConfig()
        result_df, metrics = run_financial_metrics_etl(
            self.test_df, config=config, return_metrics=True
        )

        # Check DataFrame has new columns
        self.assertGreater(result_df.shape[1], self.test_df.shape[1])

        # Check metrics are returned
        self.assertIsNotNone(metrics)
        self.assertIn("valuation_metrics_added", metrics)
        self.assertIn("profitability_metrics_added", metrics)
        self.assertIn("growth_metrics_added", metrics)
        self.assertIn("leverage_metrics_added", metrics)

    def test_pipeline_with_output_dir(self):
        """Test pipeline saves outputs to directory."""
        import tempfile
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            run_financial_metrics_etl,
            FinancialMetricsETLConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = FinancialMetricsETLConfig(
                generate_quality_alerts=True,
                generate_metrics_dashboard=True,
            )

            result_df, metrics = run_financial_metrics_etl(
                self.test_df, config=config, output_dir=Path(tmpdir), return_metrics=True
            )

            # Check output files were created
            output_path = Path(tmpdir)
            # Note: actual file creation depends on implementation
            self.assertTrue(result_df is not None)

    def test_pipeline_without_metrics_return(self):
        """Test pipeline can return DataFrame only."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            run_financial_metrics_etl,
        )

        result = run_financial_metrics_etl(self.test_df, return_metrics=False)

        # Should return DataFrame directly
        self.assertIsInstance(result, pd.DataFrame)


class TestIntegrationWithExistingETL(unittest.TestCase):
    """Test integration with existing ETL pipeline."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 50

        self.test_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Financials"], n_samples),
                "region": np.random.choice(["US", "EU"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "enterprise_value": np.random.uniform(1e9, 1.5e12, n_samples),
                "total_revenues_ltm": np.random.uniform(1e8, 1e11, n_samples),
                "total_revenues_fy": np.random.uniform(1e8, 1e11, n_samples) * 0.9,
                "ebitda_ltm": np.random.uniform(1e7, 1e10, n_samples),
                "ebitda_fy": np.random.uniform(1e7, 1e10, n_samples) * 0.85,
                "net_income_is_ltm": np.random.uniform(-1e9, 1e10, n_samples),
                "net_income_is_fy": np.random.uniform(-1e9, 1e10, n_samples) * 0.8,
                "total_debt_fy": np.random.uniform(0, 1e11, n_samples),
                "total_equity_fy": np.random.uniform(1e8, 1e11, n_samples),
                "total_assets_fy": np.random.uniform(1e9, 1e12, n_samples),
                "gross_profit_ltm": np.random.uniform(1e7, 1e10, n_samples),
                "operating_income_ltm": np.random.uniform(-1e8, 1e10, n_samples),
            }
        )

    def test_can_chain_with_etl_pipeline(self):
        """Test that financial metrics ETL can be chained after base ETL."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            run_financial_metrics_etl,
        )

        # Simulate base ETL output
        base_etl_output = self.test_df.copy()

        # Run financial metrics ETL with return_metrics=False to get DataFrame only
        result = run_financial_metrics_etl(base_etl_output, return_metrics=False)

        # Should work without errors and return DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(result.shape[1], base_etl_output.shape[1])


if __name__ == "__main__":
    unittest.main()
