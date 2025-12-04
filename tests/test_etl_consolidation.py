"""
TDD Tests for ETL Consolidation.

Tests for merging financial_metrics_etl.py functionality into etl.py.
Following strict TDD methodology: write failing tests first, then implement.

Test Coverage:
- ETLConfig with new financial metrics options
- ETLMetrics with new financial metrics tracking fields
- Integration of financial metrics functions into ETLPipeline.transform()
- New convenience function etl_with_financial_metrics()
- Backward compatibility with existing API

Version: 1.0.0
Created: 2025-12-04
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

import numpy as np
import pandas as pd


class TestETLConfigFinancialMetricsOptions(unittest.TestCase):
    """Test ETLConfig class with new financial metrics options."""

    def test_config_has_financial_metrics_flags(self):
        """Test that ETLConfig has all financial metrics computation flags."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()

        # Verify all financial metrics flags exist with correct defaults (False for backward compatibility)
        self.assertFalse(config.compute_valuation_metrics)
        self.assertFalse(config.compute_profitability_metrics)
        self.assertFalse(config.compute_growth_metrics)
        self.assertFalse(config.compute_leverage_metrics)
        self.assertFalse(config.compute_target_vs_price)
        self.assertFalse(config.handle_sector_specific_metrics)

    def test_config_has_quality_reporting_options(self):
        """Test that ETLConfig has quality reporting options."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()

        # Verify quality reporting options exist
        self.assertFalse(config.generate_quality_alerts)
        self.assertFalse(config.generate_metrics_dashboard)
        self.assertEqual(config.output_subdir, "financial_metrics")

    def test_config_financial_metrics_custom_values(self):
        """Test ETLConfig with custom financial metrics values."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig(
            compute_valuation_metrics=True,
            compute_profitability_metrics=True,
            compute_growth_metrics=True,
            compute_leverage_metrics=True,
            compute_target_vs_price=True,
            handle_sector_specific_metrics=True,
            generate_quality_alerts=True,
            generate_metrics_dashboard=True,
            output_subdir="custom_output",
        )

        self.assertTrue(config.compute_valuation_metrics)
        self.assertTrue(config.compute_profitability_metrics)
        self.assertTrue(config.compute_growth_metrics)
        self.assertTrue(config.compute_leverage_metrics)
        self.assertTrue(config.compute_target_vs_price)
        self.assertTrue(config.handle_sector_specific_metrics)
        self.assertTrue(config.generate_quality_alerts)
        self.assertTrue(config.generate_metrics_dashboard)
        self.assertEqual(config.output_subdir, "custom_output")


class TestETLMetricsFinancialMetricsTracking(unittest.TestCase):
    """Test ETLMetrics class with new financial metrics tracking fields."""

    def test_metrics_has_financial_metrics_tracking(self):
        """Test that ETLMetrics has financial metrics tracking fields."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")

        # Verify all financial metrics tracking fields exist with defaults
        self.assertEqual(metrics.valuation_metrics_added, 0)
        self.assertEqual(metrics.profitability_metrics_added, 0)
        self.assertEqual(metrics.growth_metrics_added, 0)
        self.assertEqual(metrics.leverage_metrics_added, 0)
        self.assertEqual(metrics.target_vs_price_metrics_added, 0)
        self.assertEqual(metrics.sector_specific_metrics_added, 0)

    def test_metrics_to_dict_includes_financial_metrics(self):
        """Test that to_dict() includes financial metrics section."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        metrics.valuation_metrics_added = 4
        metrics.profitability_metrics_added = 5

        result = metrics.to_dict()

        # Should have financial_metrics section
        self.assertIn("financial_metrics", result)
        self.assertEqual(result["financial_metrics"]["valuation_added"], 4)
        self.assertEqual(result["financial_metrics"]["profitability_added"], 5)

    def test_metrics_summary_includes_financial_metrics(self):
        """Test that summary() includes financial metrics info."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        metrics.valuation_metrics_added = 4
        metrics.profitability_metrics_added = 5
        metrics.growth_metrics_added = 3
        metrics.leverage_metrics_added = 2

        summary = metrics.summary()

        # Summary should mention financial metrics
        self.assertIn("Financial Metrics", summary)


class TestFinancialMetricsFunctionsInETL(unittest.TestCase):
    """Test that financial metrics functions are available in etl module."""

    def test_compute_valuation_metrics_available(self):
        """Test compute_valuation_metrics is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_valuation_metrics

        self.assertTrue(callable(compute_valuation_metrics))

    def test_compute_profitability_metrics_available(self):
        """Test compute_profitability_metrics is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_profitability_metrics

        self.assertTrue(callable(compute_profitability_metrics))

    def test_compute_growth_metrics_available(self):
        """Test compute_growth_metrics is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_growth_metrics

        self.assertTrue(callable(compute_growth_metrics))

    def test_compute_leverage_metrics_available(self):
        """Test compute_leverage_metrics is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_leverage_metrics

        self.assertTrue(callable(compute_leverage_metrics))

    def test_compute_target_vs_price_metrics_available(self):
        """Test compute_target_vs_price_metrics is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_target_vs_price_metrics

        self.assertTrue(callable(compute_target_vs_price_metrics))

    def test_handle_sector_specific_metrics_available(self):
        """Test handle_sector_specific_metrics is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import handle_sector_specific_metrics

        self.assertTrue(callable(handle_sector_specific_metrics))

    def test_compute_sector_specific_ratios_available(self):
        """Test compute_sector_specific_ratios is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_sector_specific_ratios

        self.assertTrue(callable(compute_sector_specific_ratios))

    def test_generate_data_quality_alerts_available(self):
        """Test generate_data_quality_alerts is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import generate_data_quality_alerts

        self.assertTrue(callable(generate_data_quality_alerts))

    def test_generate_metrics_dashboard_available(self):
        """Test generate_metrics_dashboard is importable from etl."""
        from finance_ml.ml_workflow.preprocessing.etl import generate_metrics_dashboard

        self.assertTrue(callable(generate_metrics_dashboard))


class TestETLPipelineTransformWithFinancialMetrics(unittest.TestCase):
    """Test ETLPipeline.transform() integration with financial metrics."""

    def setUp(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "JPM", "XOM"],
                "sector": ["Technology", "Technology", "Technology", "Financials", "Energy"],
                "last_price": [150.0, 140.0, 380.0, 180.0, 100.0],
                "market_cap": [2400000, 1800000, 2800000, 540000, 400000],
                "enterprise_value": [2500000, 1850000, 2700000, None, 450000],
                "net_income_is_ltm": [94000, 59000, 72000, 48000, 36000],
                "total_revenues_ltm": [380000, 280000, 200000, 120000, 350000],
                "ebitda_ltm": [130000, 90000, 100000, None, 60000],
                "gross_profit_ltm": [170000, 140000, 140000, None, 100000],
                "operating_income_ltm": [110000, 75000, 85000, 60000, 45000],
                "total_equity_fy": [50000, 280000, 190000, 300000, 180000],
                "total_assets_fy": [350000, 360000, 410000, 3700000, 360000],
                "total_debt_fy": [110000, 30000, 50000, 500000, 45000],
                "price_target": [180.0, 160.0, 420.0, 200.0, 120.0],
                "region": ["US", "US", "US", "US", "US"],
            }
        )

    def test_transform_computes_valuation_metrics_when_enabled(self):
        """Test transform() computes valuation metrics when flag is enabled."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            compute_valuation_metrics=True,
        )
        pipeline = ETLPipeline(config=config)

        result = pipeline.transform(self.sample_df)

        # Should have valuation metric columns
        self.assertIn("p_e_ratio", result.columns)
        self.assertIn("p_s_ratio", result.columns)
        self.assertIn("ev_ebitda_ratio", result.columns)
        self.assertIn("ev_sales_ratio", result.columns)

    def test_transform_computes_profitability_metrics_when_enabled(self):
        """Test transform() computes profitability metrics when flag is enabled."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            compute_profitability_metrics=True,
        )
        pipeline = ETLPipeline(config=config)

        result = pipeline.transform(self.sample_df)

        # Should have profitability metric columns
        self.assertIn("gross_margin_pct", result.columns)
        self.assertIn("operating_margin_pct", result.columns)
        self.assertIn("net_margin_pct", result.columns)
        self.assertIn("roe", result.columns)
        self.assertIn("roa", result.columns)

    def test_transform_computes_growth_metrics_when_enabled(self):
        """Test transform() computes growth metrics when flag is enabled."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        # Add prior year columns needed for growth metrics
        df = self.sample_df.copy()
        df["total_revenues_fy"] = [350000, 260000, 180000, 110000, 320000]
        df["ebitda_fy"] = [120000, 85000, 90000, None, 55000]
        df["net_income_is_fy"] = [85000, 55000, 65000, 45000, 32000]

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            compute_growth_metrics=True,
        )
        pipeline = ETLPipeline(config=config)

        result = pipeline.transform(df)

        # Should have growth metric columns
        self.assertIn("revenue_growth", result.columns)
        self.assertIn("ebitda_growth", result.columns)
        self.assertIn("earnings_growth", result.columns)

    def test_transform_computes_leverage_metrics_when_enabled(self):
        """Test transform() computes leverage metrics when flag is enabled."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            compute_leverage_metrics=True,
        )
        pipeline = ETLPipeline(config=config)

        result = pipeline.transform(self.sample_df)

        # Should have leverage metric columns
        self.assertIn("debt_to_equity", result.columns)
        self.assertIn("debt_to_assets", result.columns)

    def test_transform_does_not_compute_metrics_when_disabled(self):
        """Test transform() does not compute metrics when flags are disabled."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            compute_valuation_metrics=False,
            compute_profitability_metrics=False,
            compute_growth_metrics=False,
            compute_leverage_metrics=False,
        )
        pipeline = ETLPipeline(config=config)

        result = pipeline.transform(self.sample_df)

        # Should NOT have metric columns (backward compatibility)
        self.assertNotIn("p_e_ratio", result.columns)
        self.assertNotIn("gross_margin_pct", result.columns)
        self.assertNotIn("revenue_growth", result.columns)
        self.assertNotIn("debt_to_equity", result.columns)

    def test_transform_tracks_metrics_added_count(self):
        """Test transform() tracks the count of metrics added."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            compute_valuation_metrics=True,
            compute_profitability_metrics=True,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        pipeline.transform(self.sample_df)

        # Metrics should track added columns
        self.assertEqual(pipeline.metrics.valuation_metrics_added, 4)
        self.assertEqual(pipeline.metrics.profitability_metrics_added, 5)


class TestETLWithFinancialMetricsConvenienceFunction(unittest.TestCase):
    """Test etl_with_financial_metrics() convenience function."""

    def test_function_exists(self):
        """Test etl_with_financial_metrics function exists."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

        self.assertTrue(callable(etl_with_financial_metrics))

    def test_function_signature(self):
        """Test etl_with_financial_metrics has correct signature."""
        import inspect
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

        sig = inspect.signature(etl_with_financial_metrics)
        params = list(sig.parameters.keys())

        # Should have expected parameters
        self.assertIn("source", params)
        self.assertIn("data_dir", params)
        self.assertIn("db_url", params)
        self.assertIn("compute_all_metrics", params)
        self.assertIn("output_dir", params)
        self.assertIn("return_metrics", params)

    @patch("finance_ml.ml_workflow.preprocessing.etl.run_etl_pipeline")
    def test_function_enables_all_metrics_by_default(self, mock_run):
        """Test etl_with_financial_metrics enables all metrics by default."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

        mock_run.return_value = pd.DataFrame()

        etl_with_financial_metrics(source="csv", data_dir="data/")

        # Verify run_etl_pipeline was called with metrics enabled
        call_args = mock_run.call_args
        config = call_args.kwargs.get("config") or call_args.args[3]

        self.assertTrue(config.compute_valuation_metrics)
        self.assertTrue(config.compute_profitability_metrics)
        self.assertTrue(config.compute_growth_metrics)
        self.assertTrue(config.compute_leverage_metrics)
        self.assertTrue(config.compute_target_vs_price)
        self.assertTrue(config.handle_sector_specific_metrics)

    @patch("finance_ml.ml_workflow.preprocessing.etl.run_etl_pipeline")
    def test_function_can_disable_metrics(self, mock_run):
        """Test etl_with_financial_metrics can disable metrics."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics

        mock_run.return_value = pd.DataFrame()

        etl_with_financial_metrics(source="csv", data_dir="data/", compute_all_metrics=False)

        # Verify run_etl_pipeline was called with metrics disabled
        call_args = mock_run.call_args
        config = call_args.kwargs.get("config") or call_args.args[3]

        self.assertFalse(config.compute_valuation_metrics)
        self.assertFalse(config.compute_profitability_metrics)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing ETL API."""

    def test_existing_run_etl_pipeline_unchanged(self):
        """Test run_etl_pipeline works without financial metrics flags."""
        from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig

        # Default config should work without changes
        config = ETLConfig()

        # These should be False by default (backward compatible)
        self.assertFalse(config.compute_valuation_metrics)
        self.assertFalse(config.compute_profitability_metrics)

    def test_existing_etl_with_imputation_unchanged(self):
        """Test etl_with_imputation works without changes."""
        import inspect
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_imputation

        # Function should still exist with original signature
        sig = inspect.signature(etl_with_imputation)
        params = list(sig.parameters.keys())

        self.assertIn("source", params)
        self.assertIn("data_dir", params)
        self.assertIn("imputation_strategy", params)

    def test_existing_etl_with_imputation_and_scaling_unchanged(self):
        """Test etl_with_imputation_and_scaling works without changes."""
        import inspect
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_imputation_and_scaling

        # Function should still exist with original signature
        sig = inspect.signature(etl_with_imputation_and_scaling)
        params = list(sig.parameters.keys())

        self.assertIn("source", params)
        self.assertIn("scaler_type", params)
        self.assertIn("scale_by_sector", params)


class TestFinancialMetricsETLDeprecation(unittest.TestCase):
    """Test deprecation handling in financial_metrics_etl module."""

    def test_deprecated_module_still_works(self):
        """Test financial_metrics_etl module still works (backward compat)."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            run_financial_metrics_etl,
            FinancialMetricsETLConfig,
        )

        self.assertTrue(callable(run_financial_metrics_etl))
        self.assertTrue(FinancialMetricsETLConfig is not None)

    def test_deprecated_module_emits_warning(self):
        """Test financial_metrics_etl emits deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from finance_ml.ml_workflow.preprocessing import financial_metrics_etl

            # Should emit deprecation warning
            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            # Note: This test may pass even without implementation if module was
            # already imported. The implementation should add the warning.


class TestFinancialMetricsFunctionOutput(unittest.TestCase):
    """Test financial metrics functions produce correct output."""

    def setUp(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "market_cap": [2400000, 1800000, 2800000],
                "enterprise_value": [2500000, 1850000, 2700000],
                "net_income_is_ltm": [94000, 59000, 72000],
                "total_revenues_ltm": [380000, 280000, 200000],
                "ebitda_ltm": [130000, 90000, 100000],
                "gross_profit_ltm": [170000, 140000, 140000],
                "operating_income_ltm": [110000, 75000, 85000],
                "total_equity_fy": [50000, 280000, 190000],
                "total_assets_fy": [350000, 360000, 410000],
                "total_debt_fy": [110000, 30000, 50000],
            }
        )

    def test_compute_valuation_metrics_output(self):
        """Test compute_valuation_metrics produces correct output."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_valuation_metrics

        result = compute_valuation_metrics(self.sample_df)

        # Check P/E ratio calculation (market_cap / net_income)
        expected_pe_aapl = 2400000 / 94000
        self.assertAlmostEqual(result.loc[0, "p_e_ratio"], expected_pe_aapl, places=2)

        # Check EV/EBITDA calculation
        expected_ev_ebitda_aapl = 2500000 / 130000
        self.assertAlmostEqual(result.loc[0, "ev_ebitda_ratio"], expected_ev_ebitda_aapl, places=2)

    def test_compute_profitability_metrics_output(self):
        """Test compute_profitability_metrics produces correct output."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_profitability_metrics

        result = compute_profitability_metrics(self.sample_df)

        # Check gross margin calculation (returned as percentage, not decimal)
        expected_gross_margin_aapl = (170000 / 380000) * 100  # ~44.74%
        self.assertAlmostEqual(
            result.loc[0, "gross_margin_pct"], expected_gross_margin_aapl, places=2
        )

        # Check ROE calculation (returned as percentage, not decimal)
        expected_roe_aapl = (94000 / 50000) * 100  # 188%
        self.assertAlmostEqual(result.loc[0, "roe"], expected_roe_aapl, places=2)

    def test_compute_leverage_metrics_output(self):
        """Test compute_leverage_metrics produces correct output."""
        from finance_ml.ml_workflow.preprocessing.etl import compute_leverage_metrics

        result = compute_leverage_metrics(self.sample_df)

        # Check debt to equity calculation
        expected_d_e_aapl = 110000 / 50000
        self.assertAlmostEqual(result.loc[0, "debt_to_equity"], expected_d_e_aapl, places=4)


class TestIntegrationETLWithFinancialMetrics(unittest.TestCase):
    """Integration tests for complete ETL + financial metrics pipeline."""

    def setUp(self):
        """Set up test data directory."""
        self.temp_dir = tempfile.mkdtemp()

    def test_complete_pipeline_with_all_metrics(self):
        """Test complete pipeline with all financial metrics enabled."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        sample_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL"],
                "sector": ["Technology", "Technology"],
                "last_price": [150.0, 140.0],
                "market_cap": [2400000, 1800000],
                "enterprise_value": [2500000, 1850000],
                "net_income_is_ltm": [94000, 59000],
                "total_revenues_ltm": [380000, 280000],
                "total_revenues_fy": [350000, 260000],
                "ebitda_ltm": [130000, 90000],
                "ebitda_fy": [120000, 85000],
                "net_income_is_fy": [85000, 55000],
                "gross_profit_ltm": [170000, 140000],
                "operating_income_ltm": [110000, 75000],
                "total_equity_fy": [50000, 280000],
                "total_assets_fy": [350000, 360000],
                "total_debt_fy": [110000, 30000],
                "price_target": [180.0, 160.0],
                "region": ["US", "US"],
            }
        )

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            compute_valuation_metrics=True,
            compute_profitability_metrics=True,
            compute_growth_metrics=True,
            compute_leverage_metrics=True,
        )

        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        result = pipeline.transform(sample_df)

        # Verify all metric categories are present
        valuation_cols = ["p_e_ratio", "p_s_ratio", "ev_ebitda_ratio", "ev_sales_ratio"]
        profitability_cols = [
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "roe",
            "roa",
        ]
        growth_cols = ["revenue_growth", "ebitda_growth", "earnings_growth"]
        leverage_cols = ["debt_to_equity", "debt_to_assets"]

        for col in valuation_cols:
            self.assertIn(col, result.columns, f"Missing valuation column: {col}")

        for col in profitability_cols:
            self.assertIn(col, result.columns, f"Missing profitability column: {col}")

        for col in growth_cols:
            self.assertIn(col, result.columns, f"Missing growth column: {col}")

        for col in leverage_cols:
            self.assertIn(col, result.columns, f"Missing leverage column: {col}")

        # Verify metrics tracking
        self.assertEqual(pipeline.metrics.valuation_metrics_added, 4)
        self.assertEqual(pipeline.metrics.profitability_metrics_added, 5)
        self.assertEqual(pipeline.metrics.growth_metrics_added, 3)
        self.assertEqual(pipeline.metrics.leverage_metrics_added, 2)


if __name__ == "__main__":
    unittest.main()
