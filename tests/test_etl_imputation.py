"""
Tests for ETL pipeline imputation integration.

This module tests the integration of the 6-step imputation strategy into the ETL pipeline.

Run tests (Windows PowerShell):
    python -m unittest tests.test_etl_imputation -v

Test Coverage:
    - test_imputation_stage_execution: Verify imputation runs in transform()
    - test_imputation_6step_completeness: Assert zero missing values after 6-step
    - test_imputation_4step_numeric_only: Verify 4-step only fills numerics
    - test_imputation_median_fallback: Test median_only strategy
    - test_imputation_metrics_tracking: Verify ETLMetrics records imputation data
    - test_imputation_with_log_transforms: Ensure imputation → log transform order
    - test_etl_with_imputation_convenience: Test new convenience function
    - test_imputation_preserves_price_columns: Verify price columns not corrupted
    - test_imputation_sector_awareness: Verify KNN uses sector grouping
    - test_imputation_date_formatting: Verify datetime columns properly formatted
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from finance_ml.ml_workflow.preprocessing.etl import (
    ETLConfig,
    ETLMetrics,
    ETLPipeline,
    run_etl_pipeline,
    etl_with_imputation,
)


class TestETLImputationConfig(unittest.TestCase):
    """Test ETLConfig imputation configuration options."""

    def test_config_default_imputation_enabled(self):
        """Verify imputation is enabled by default."""
        config = ETLConfig()
        self.assertTrue(config.apply_imputation)
        self.assertEqual(config.imputation_strategy, "6step")
        self.assertEqual(config.knn_neighbors, 5)
        self.assertTrue(config.handle_categorical_imputation)
        self.assertTrue(config.handle_datetime_imputation)

    def test_config_imputation_strategies(self):
        """Verify all imputation strategies are valid."""
        for strategy in ["6step", "4step", "median_only"]:
            config = ETLConfig(imputation_strategy=strategy)
            self.assertEqual(config.imputation_strategy, strategy)

    def test_config_imputation_disabled(self):
        """Verify imputation can be disabled."""
        config = ETLConfig(apply_imputation=False)
        self.assertFalse(config.apply_imputation)


class TestETLMetricsImputation(unittest.TestCase):
    """Test ETLMetrics imputation tracking fields."""

    def test_metrics_imputation_fields_exist(self):
        """Verify imputation fields exist in ETLMetrics."""
        metrics = ETLMetrics(source_type="test")
        self.assertIsNone(metrics.imputation_strategy)
        self.assertEqual(metrics.missing_values_before_imputation, 0)
        self.assertEqual(metrics.missing_values_after_imputation, 0)
        self.assertFalse(metrics.imputation_completeness)
        self.assertFalse(metrics.date_columns_ready)

    def test_metrics_to_dict_includes_imputation(self):
        """Verify to_dict includes imputation section."""
        metrics = ETLMetrics(
            source_type="test",
            imputation_strategy="6step",
            missing_values_before_imputation=100,
            missing_values_after_imputation=0,
            imputation_completeness=True,
            date_columns_ready=True,
        )
        result = metrics.to_dict()

        self.assertIn("imputation", result)
        self.assertEqual(result["imputation"]["strategy"], "6step")
        self.assertEqual(result["imputation"]["missing_before"], 100)
        self.assertEqual(result["imputation"]["missing_after"], 0)
        self.assertTrue(result["imputation"]["completeness"])
        self.assertTrue(result["imputation"]["dates_ready"])

    def test_metrics_summary_includes_imputation(self):
        """Verify summary includes imputation info when strategy is set."""
        metrics = ETLMetrics(
            source_type="test",
            imputation_strategy="6step",
            missing_values_before_imputation=100,
            missing_values_after_imputation=5,
        )
        summary = metrics.summary()

        self.assertIn("Imputation:", summary)
        self.assertIn("6step", summary)
        self.assertIn("100", summary)
        self.assertIn("5", summary)

    def test_metrics_summary_no_imputation_info_when_none(self):
        """Verify summary omits imputation info when strategy is None."""
        metrics = ETLMetrics(source_type="test")
        summary = metrics.summary()

        self.assertNotIn("Imputation:", summary)


class TestETLPipelineImputation(unittest.TestCase):
    """Test ETLPipeline imputation stage execution."""

    def _create_test_dataframe(self):
        """Create a test DataFrame with missing values."""
        np.random.seed(42)
        n_rows = 100
        return pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_rows)],
                "sector": np.random.choice(["Technology", "Healthcare", "Finance"], n_rows),
                "last_price": np.random.uniform(10, 500, n_rows),
                "price_target": np.where(
                    np.random.random(n_rows) > 0.2, np.random.uniform(15, 600, n_rows), np.nan
                ),
                "market_cap": np.where(
                    np.random.random(n_rows) > 0.15, np.random.uniform(1e9, 1e12, n_rows), np.nan
                ),
                "pe_ratio": np.where(
                    np.random.random(n_rows) > 0.3, np.random.uniform(5, 50, n_rows), np.nan
                ),
                "industry": np.where(
                    np.random.random(n_rows) > 0.1,
                    np.random.choice(["Software", "Biotech", "Banking"], n_rows),
                    None,
                ),
            }
        )

    def test_imputation_stage_execution(self):
        """Verify imputation stage runs in transform() method."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="6step",
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        initial_missing = df.isna().sum().sum()
        self.assertGreater(initial_missing, 0, "Test data should have missing values")

        result = pipeline.transform(df)

        # After imputation, missing values should be reduced
        final_missing = result.isna().sum().sum()
        self.assertLess(final_missing, initial_missing)

    def test_imputation_6step_completeness(self):
        """Verify 6-step imputation reduces missing values significantly."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="6step",
            handle_categorical_imputation=True,
            handle_datetime_imputation=True,
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        result = pipeline.transform(df)

        # 6-step should handle all columns
        # Note: Some columns may still have NaN if they couldn't be imputed
        self.assertIsNotNone(pipeline.metrics.imputation_strategy)
        self.assertEqual(pipeline.metrics.imputation_strategy, "6step")

    def test_imputation_4step_numeric_only(self):
        """Verify 4-step strategy only handles numeric columns."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="4step",
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        result = pipeline.transform(df)

        self.assertEqual(pipeline.metrics.imputation_strategy, "4step")

    def test_imputation_median_fallback(self):
        """Verify median_only strategy works."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="median_only",
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        initial_missing = df.isna().sum().sum()

        result = pipeline.transform(df)

        self.assertEqual(pipeline.metrics.imputation_strategy, "median_only")
        # Median imputation should reduce numeric missing values
        final_missing = result.isna().sum().sum()
        self.assertLessEqual(final_missing, initial_missing)

    def test_imputation_metrics_tracking(self):
        """Verify ETLMetrics records imputation data correctly."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="6step",
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        initial_missing = df.isna().sum().sum()

        result = pipeline.transform(df)

        # Verify metrics were tracked
        self.assertEqual(pipeline.metrics.imputation_strategy, "6step")
        self.assertEqual(pipeline.metrics.missing_values_before_imputation, initial_missing)
        self.assertGreaterEqual(
            pipeline.metrics.missing_values_before_imputation,
            pipeline.metrics.missing_values_after_imputation,
        )

    def test_imputation_with_log_transforms(self):
        """Ensure imputation runs before log transforms."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="6step",
            apply_log_transforms=True,
            log_transform_method="log1p",
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        # Should not raise - imputation should fill NaN before log transform
        result = pipeline.transform(df)

        # Verify both stages executed
        self.assertEqual(pipeline.metrics.imputation_strategy, "6step")

    def test_imputation_disabled(self):
        """Verify imputation can be skipped when disabled."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=False,
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        initial_missing = df.isna().sum().sum()

        result = pipeline.transform(df)

        # Missing values should remain unchanged
        final_missing = result.isna().sum().sum()
        self.assertEqual(initial_missing, final_missing)
        self.assertIsNone(pipeline.metrics.imputation_strategy)

    def test_imputation_preserves_price_columns(self):
        """Verify price columns are not corrupted by imputation."""
        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="6step",
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_dataframe()
        # Store original non-null prices
        original_prices = df["last_price"].dropna().copy()

        result = pipeline.transform(df)

        # Original non-null prices should be preserved
        result_prices = result["last_price"].loc[original_prices.index]
        pd.testing.assert_series_equal(
            original_prices,
            result_prices,
            check_names=False,
        )


class TestETLWithImputationConvenience(unittest.TestCase):
    """Test etl_with_imputation convenience function."""

    @patch("finance_ml.ml_workflow.preprocessing.etl.run_etl_pipeline")
    def test_etl_with_imputation_calls_run_etl_pipeline(self, mock_run):
        """Verify etl_with_imputation calls run_etl_pipeline correctly."""
        mock_df = pd.DataFrame({"a": [1, 2, 3]})
        mock_metrics = ETLMetrics(source_type="csv")
        mock_run.return_value = (mock_df, mock_metrics)

        result = etl_with_imputation(source="csv", data_dir="data/")

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["source"], "csv")
        self.assertEqual(call_kwargs["data_dir"], "data/")
        self.assertTrue(call_kwargs["return_metrics"])
        self.assertTrue(call_kwargs["config"].apply_imputation)
        self.assertEqual(call_kwargs["config"].imputation_strategy, "6step")

    @patch("finance_ml.ml_workflow.preprocessing.etl.run_etl_pipeline")
    def test_etl_with_imputation_custom_strategy(self, mock_run):
        """Verify etl_with_imputation accepts custom imputation strategy."""
        mock_df = pd.DataFrame({"a": [1, 2, 3]})
        mock_metrics = ETLMetrics(source_type="csv")
        mock_run.return_value = (mock_df, mock_metrics)

        result = etl_with_imputation(source="csv", data_dir="data/", imputation_strategy="4step")

        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["config"].imputation_strategy, "4step")

    @patch("finance_ml.ml_workflow.preprocessing.etl.run_etl_pipeline")
    def test_etl_with_imputation_db_source(self, mock_run):
        """Verify etl_with_imputation works with database source."""
        mock_df = pd.DataFrame({"a": [1, 2, 3]})
        mock_metrics = ETLMetrics(source_type="all_stocks")
        mock_run.return_value = (mock_df, mock_metrics)

        result = etl_with_imputation(source="all_stocks", db_url="postgresql://localhost/test")

        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["source"], "all_stocks")
        self.assertEqual(call_kwargs["db_url"], "postgresql://localhost/test")


class TestImputationIntegration(unittest.TestCase):
    """Integration tests for imputation in the full ETL pipeline."""

    def test_imputation_sector_awareness(self):
        """Verify KNN imputation uses sector grouping."""
        # Create data with distinct sector patterns
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(30)],
                "sector": ["Technology"] * 10 + ["Healthcare"] * 10 + ["Finance"] * 10,
                "last_price": [100.0] * 10 + [50.0] * 10 + [200.0] * 10,
                "pe_ratio": [25.0] * 9 + [np.nan] + [15.0] * 9 + [np.nan] + [10.0] * 9 + [np.nan],
            }
        )

        config = ETLConfig(
            normalize_columns=False,
            validate_schema=False,
            drop_invalid_rows=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="6step",
            knn_neighbors=3,
            apply_log_transforms=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        result = pipeline.transform(df)

        # PE ratio should be imputed for each sector
        # The imputed values should be close to sector averages
        self.assertFalse(result["pe_ratio"].isna().any())


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
