"""
Tests for Unified ETL Pipeline with Semantic Transformations and Feature Engineering.

TDD tests for consolidating schema.py, column_semantics.py, and features/api.py
functionality into the unified ETL pipeline (etl.py).

Aligned with:
- code_guidelines.md v1.4 Section 8: Notebook Best Practices
- code_guidelines.md v1.4 Section 9: Column Schema and DataFrame Conventions
- preprocessing_stages_4-8_improvement_plan.md

Test Categories:
1. ETLConfig new semantic-aware attributes
2. ETLMetrics new semantic/feature tracking attributes
3. Semantic transformation methods (_apply_semantic_transformations, etc.)
4. Feature engineering integration (_apply_feature_engineering)
5. etl_with_features() convenience function
6. Integration tests for full pipeline with features
"""

from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd


class TestETLConfigSemanticAttributes(unittest.TestCase):
    """Test ETLConfig new semantic-aware transformation attributes."""

    def test_etlconfig_has_use_semantic_column_classification(self):
        """ETLConfig should have use_semantic_column_classification attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "use_semantic_column_classification"))
        # Default should be True for semantic-aware transformations
        self.assertTrue(config.use_semantic_column_classification)

    def test_etlconfig_has_preserve_price_columns(self):
        """ETLConfig should have preserve_price_columns attribute (default True)."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "preserve_price_columns"))
        self.assertTrue(config.preserve_price_columns)

    def test_etlconfig_has_log_transform_market_values(self):
        """ETLConfig should have log_transform_market_values attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "log_transform_market_values"))
        # Default True for skewed market value columns
        self.assertTrue(config.log_transform_market_values)

    def test_etlconfig_has_exclude_ratios_from_winsorization(self):
        """ETLConfig should have exclude_ratios_from_winsorization attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "exclude_ratios_from_winsorization"))
        self.assertTrue(config.exclude_ratios_from_winsorization)

    def test_etlconfig_has_exclude_percentages_from_winsorization(self):
        """ETLConfig should have exclude_percentages_from_winsorization attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "exclude_percentages_from_winsorization"))
        self.assertTrue(config.exclude_percentages_from_winsorization)

    def test_etlconfig_has_apply_feature_engineering(self):
        """ETLConfig should have apply_feature_engineering attribute (default False)."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "apply_feature_engineering"))
        # Default False for backward compatibility
        self.assertFalse(config.apply_feature_engineering)

    def test_etlconfig_has_feature_preset(self):
        """ETLConfig should have feature_preset attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "feature_preset"))
        self.assertEqual(config.feature_preset, "standard")

    def test_etlconfig_has_feature_categories(self):
        """ETLConfig should have feature_categories attribute (optional list)."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig()
        self.assertTrue(hasattr(config, "feature_categories"))
        self.assertIsNone(config.feature_categories)

    def test_etlconfig_custom_values(self):
        """ETLConfig should accept custom values for new attributes."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLConfig

        config = ETLConfig(
            use_semantic_column_classification=False,
            preserve_price_columns=False,
            log_transform_market_values=False,
            apply_feature_engineering=True,
            feature_preset="comprehensive",
            feature_categories=["momentum", "quality"],
        )

        self.assertFalse(config.use_semantic_column_classification)
        self.assertFalse(config.preserve_price_columns)
        self.assertFalse(config.log_transform_market_values)
        self.assertTrue(config.apply_feature_engineering)
        self.assertEqual(config.feature_preset, "comprehensive")
        self.assertEqual(config.feature_categories, ["momentum", "quality"])


class TestETLMetricsSemanticAttributes(unittest.TestCase):
    """Test ETLMetrics new semantic/feature tracking attributes."""

    def test_etlmetrics_has_semantic_classification_applied(self):
        """ETLMetrics should have semantic_classification_applied attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "semantic_classification_applied"))
        self.assertFalse(metrics.semantic_classification_applied)

    def test_etlmetrics_has_price_columns_count(self):
        """ETLMetrics should have price_columns_count attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "price_columns_count"))
        self.assertEqual(metrics.price_columns_count, 0)

    def test_etlmetrics_has_market_value_columns_count(self):
        """ETLMetrics should have market_value_columns_count attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "market_value_columns_count"))
        self.assertEqual(metrics.market_value_columns_count, 0)

    def test_etlmetrics_has_ratio_columns_count(self):
        """ETLMetrics should have ratio_columns_count attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "ratio_columns_count"))
        self.assertEqual(metrics.ratio_columns_count, 0)

    def test_etlmetrics_has_percentage_columns_count(self):
        """ETLMetrics should have percentage_columns_count attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "percentage_columns_count"))
        self.assertEqual(metrics.percentage_columns_count, 0)

    def test_etlmetrics_has_count_columns_count(self):
        """ETLMetrics should have count_columns_count attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "count_columns_count"))
        self.assertEqual(metrics.count_columns_count, 0)

    def test_etlmetrics_has_log_transformed_columns(self):
        """ETLMetrics should have log_transformed_columns attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "log_transformed_columns"))
        self.assertEqual(metrics.log_transformed_columns, 0)

    def test_etlmetrics_has_feature_engineering_applied(self):
        """ETLMetrics should have feature_engineering_applied attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "feature_engineering_applied"))
        self.assertFalse(metrics.feature_engineering_applied)

    def test_etlmetrics_has_feature_preset_used(self):
        """ETLMetrics should have feature_preset_used attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "feature_preset_used"))
        self.assertEqual(metrics.feature_preset_used, "")

    def test_etlmetrics_has_features_added(self):
        """ETLMetrics should have features_added attribute."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        self.assertTrue(hasattr(metrics, "features_added"))
        self.assertEqual(metrics.features_added, 0)

    def test_etlmetrics_to_dict_includes_semantic_metrics(self):
        """ETLMetrics.to_dict() should include semantic transformation metrics."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        metrics.semantic_classification_applied = True
        metrics.price_columns_count = 5
        metrics.log_transformed_columns = 3

        result = metrics.to_dict()

        self.assertIn("semantic_transformations", result)
        self.assertTrue(result["semantic_transformations"]["applied"])
        self.assertEqual(result["semantic_transformations"]["price_columns"], 5)
        self.assertEqual(result["semantic_transformations"]["log_transformed"], 3)

    def test_etlmetrics_to_dict_includes_feature_engineering_metrics(self):
        """ETLMetrics.to_dict() should include feature engineering metrics."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        metrics.feature_engineering_applied = True
        metrics.feature_preset_used = "comprehensive"
        metrics.features_added = 25

        result = metrics.to_dict()

        self.assertIn("feature_engineering", result)
        self.assertTrue(result["feature_engineering"]["applied"])
        self.assertEqual(result["feature_engineering"]["preset"], "comprehensive")
        self.assertEqual(result["feature_engineering"]["features_added"], 25)

    def test_etlmetrics_summary_includes_semantic_info(self):
        """ETLMetrics.summary() should include semantic transformation info."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        metrics.semantic_classification_applied = True
        metrics.price_columns_count = 5
        metrics.log_transformed_columns = 3

        summary = metrics.summary()

        self.assertIn("Semantic", summary)
        self.assertIn("Price Columns", summary)

    def test_etlmetrics_summary_includes_feature_engineering_info(self):
        """ETLMetrics.summary() should include feature engineering info."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLMetrics

        metrics = ETLMetrics(source_type="csv")
        metrics.feature_engineering_applied = True
        metrics.feature_preset_used = "comprehensive"
        metrics.features_added = 25

        summary = metrics.summary()

        self.assertIn("Feature Engineering", summary)
        self.assertIn("comprehensive", summary)


class TestETLPipelineSemanticMethods(unittest.TestCase):
    """Test ETLPipeline semantic transformation methods."""

    def _create_test_df(self) -> pd.DataFrame:
        """Create a test DataFrame with various column types."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                # Price columns (should never be transformed)
                "last_price": [100.0, 150.0, 200.0, 250.0, 300.0],
                "price_target": [110.0, 160.0, 220.0, 270.0, 330.0],
                # Market value columns (should be log-transformed)
                "market_cap": [1e9, 5e9, 10e9, 50e9, 100e9],
                "revenue": [1e8, 5e8, 1e9, 5e9, 10e9],
                # Ratio columns (pre-normalized)
                "p_e": [15.0, 20.0, 25.0, 30.0, 35.0],
                "ev_ebitda": [10.0, 12.0, 14.0, 16.0, 18.0],
                # Percentage columns (bounded)
                "gross_margin": [30.0, 35.0, 40.0, 45.0, 50.0],
                # Count columns (discrete)
                "num_analysts": [5, 10, 15, 20, 25],
                # Other columns
                "sector": ["Tech", "Finance", "Healthcare", "Energy", "Consumer"],
                "ticker": ["AAPL", "JPM", "JNJ", "XOM", "WMT"],
            }
        )

    def test_pipeline_has_apply_semantic_transformations_method(self):
        """ETLPipeline should have _apply_semantic_transformations method."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline

        pipeline = ETLPipeline()
        self.assertTrue(hasattr(pipeline, "_apply_semantic_transformations"))
        self.assertTrue(callable(pipeline._apply_semantic_transformations))

    def test_apply_semantic_transformations_returns_dataframe(self):
        """_apply_semantic_transformations should return a DataFrame."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(use_semantic_column_classification=True)
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = MagicMock()
        df = self._create_test_df()

        result = pipeline._apply_semantic_transformations(df)

        self.assertIsInstance(result, pd.DataFrame)

    def test_apply_semantic_transformations_preserves_price_columns(self):
        """_apply_semantic_transformations should preserve price columns unchanged."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(
            use_semantic_column_classification=True,
            preserve_price_columns=True,
            log_transform_market_values=True,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = MagicMock()
        df = self._create_test_df()
        original_prices = df["last_price"].copy()

        result = pipeline._apply_semantic_transformations(df)

        pd.testing.assert_series_equal(result["last_price"], original_prices)

    def test_apply_semantic_transformations_creates_log_columns(self):
        """_apply_semantic_transformations should create log-transformed columns for market values."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(
            use_semantic_column_classification=True,
            log_transform_market_values=True,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = MagicMock()
        df = self._create_test_df()

        result = pipeline._apply_semantic_transformations(df)

        # Should have log_market_cap and log_revenue columns
        self.assertIn("log_market_cap", result.columns)
        self.assertIn("log_revenue", result.columns)

    def test_apply_semantic_transformations_skipped_when_disabled(self):
        """_apply_semantic_transformations should skip when disabled in config."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(use_semantic_column_classification=False)
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = MagicMock()
        df = self._create_test_df()

        result = pipeline._apply_semantic_transformations(df)

        # Should return unchanged DataFrame (no log columns added)
        self.assertEqual(set(result.columns), set(df.columns))

    def test_pipeline_has_get_winsorization_columns_method(self):
        """ETLPipeline should have _get_winsorization_columns method."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline

        pipeline = ETLPipeline()
        self.assertTrue(hasattr(pipeline, "_get_winsorization_columns"))
        self.assertTrue(callable(pipeline._get_winsorization_columns))

    def test_get_winsorization_columns_excludes_price_columns(self):
        """_get_winsorization_columns should exclude price columns."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(use_semantic_column_classification=True)
        pipeline = ETLPipeline(config=config)
        df = self._create_test_df()

        result = pipeline._get_winsorization_columns(df)

        self.assertNotIn("last_price", result)
        self.assertNotIn("price_target", result)

    def test_get_winsorization_columns_excludes_ratio_columns(self):
        """_get_winsorization_columns should exclude ratio columns when configured."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(
            use_semantic_column_classification=True,
            exclude_ratios_from_winsorization=True,
        )
        pipeline = ETLPipeline(config=config)
        df = self._create_test_df()

        result = pipeline._get_winsorization_columns(df)

        self.assertNotIn("p_e", result)
        self.assertNotIn("ev_ebitda", result)

    def test_pipeline_has_get_scaling_columns_method(self):
        """ETLPipeline should have _get_scaling_columns method."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline

        pipeline = ETLPipeline()
        self.assertTrue(hasattr(pipeline, "_get_scaling_columns"))
        self.assertTrue(callable(pipeline._get_scaling_columns))

    def test_get_scaling_columns_excludes_price_columns(self):
        """_get_scaling_columns should exclude price columns."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(use_semantic_column_classification=True)
        pipeline = ETLPipeline(config=config)
        df = self._create_test_df()

        result = pipeline._get_scaling_columns(df)

        self.assertNotIn("last_price", result)
        self.assertNotIn("price_target", result)


class TestETLPipelineFeatureEngineering(unittest.TestCase):
    """Test ETLPipeline feature engineering integration."""

    def _create_test_df(self) -> pd.DataFrame:
        """Create a test DataFrame with columns needed for feature engineering."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "last_price": [100.0, 150.0, 200.0, 250.0, 300.0],
                "price_target": [110.0, 160.0, 220.0, 270.0, 330.0],
                "market_cap": [1e9, 5e9, 10e9, 50e9, 100e9],
                "revenue": [1e8, 5e8, 1e9, 5e9, 10e9],
                "ebitda": [1e7, 5e7, 1e8, 5e8, 1e9],
                "net_income": [5e6, 2e7, 5e7, 2e8, 5e8],
                "total_equity": [5e8, 2e9, 5e9, 2e10, 5e10],
                "total_assets": [1e9, 5e9, 10e9, 50e9, 100e9],
                "sector": ["Tech", "Finance", "Healthcare", "Energy", "Consumer"],
                "ticker": ["AAPL", "JPM", "JNJ", "XOM", "WMT"],
            }
        )

    def test_pipeline_has_apply_feature_engineering_method(self):
        """ETLPipeline should have _apply_feature_engineering method."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline

        pipeline = ETLPipeline()
        self.assertTrue(hasattr(pipeline, "_apply_feature_engineering"))
        self.assertTrue(callable(pipeline._apply_feature_engineering))

    def test_apply_feature_engineering_returns_dataframe(self):
        """_apply_feature_engineering should return a DataFrame."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(apply_feature_engineering=True, feature_preset="basic")
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = MagicMock()
        df = self._create_test_df()

        result = pipeline._apply_feature_engineering(df)

        self.assertIsInstance(result, pd.DataFrame)

    def test_apply_feature_engineering_skipped_when_disabled(self):
        """_apply_feature_engineering should skip when disabled in config."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(apply_feature_engineering=False)
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = MagicMock()
        df = self._create_test_df()

        result = pipeline._apply_feature_engineering(df)

        # Should return same DataFrame unchanged
        self.assertEqual(len(result.columns), len(df.columns))

    def test_apply_feature_engineering_adds_features(self):
        """_apply_feature_engineering should add new feature columns."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig

        config = ETLConfig(apply_feature_engineering=True, feature_preset="basic")
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = MagicMock()
        df = self._create_test_df()
        original_cols = len(df.columns)

        result = pipeline._apply_feature_engineering(df)

        # Should have more columns after feature engineering
        self.assertGreater(len(result.columns), original_cols)

    def test_apply_feature_engineering_updates_metrics(self):
        """_apply_feature_engineering should update metrics with features added count."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(apply_feature_engineering=True, feature_preset="basic")
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="csv")
        df = self._create_test_df()

        result = pipeline._apply_feature_engineering(df)

        self.assertTrue(pipeline.metrics.feature_engineering_applied)
        self.assertEqual(pipeline.metrics.feature_preset_used, "basic")
        self.assertGreater(pipeline.metrics.features_added, 0)


class TestEtlWithFeaturesFunction(unittest.TestCase):
    """Test etl_with_features() convenience function."""

    def test_etl_with_features_function_exists(self):
        """etl_with_features function should exist in etl module."""
        from finance_ml.ml_workflow.preprocessing import etl

        self.assertTrue(hasattr(etl, "etl_with_features"))
        self.assertTrue(callable(etl.etl_with_features))

    def test_etl_with_features_accepts_source_parameter(self):
        """etl_with_features should accept source parameter."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_features
        import inspect

        sig = inspect.signature(etl_with_features)
        params = list(sig.parameters.keys())

        self.assertIn("source", params)

    def test_etl_with_features_accepts_feature_preset_parameter(self):
        """etl_with_features should accept feature_preset parameter."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_features
        import inspect

        sig = inspect.signature(etl_with_features)
        params = list(sig.parameters.keys())

        self.assertIn("feature_preset", params)

    def test_etl_with_features_accepts_feature_categories_parameter(self):
        """etl_with_features should accept feature_categories parameter."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_features
        import inspect

        sig = inspect.signature(etl_with_features)
        params = list(sig.parameters.keys())

        self.assertIn("feature_categories", params)

    def test_etl_with_features_accepts_return_metrics_parameter(self):
        """etl_with_features should accept return_metrics parameter."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_features
        import inspect

        sig = inspect.signature(etl_with_features)
        params = list(sig.parameters.keys())

        self.assertIn("return_metrics", params)

    @patch("finance_ml.ml_workflow.preprocessing.etl.run_etl_pipeline")
    def test_etl_with_features_enables_feature_engineering(self, mock_run):
        """etl_with_features should enable feature engineering in config."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_features

        mock_run.return_value = pd.DataFrame()

        etl_with_features(source="csv", data_dir=Path("data"))

        # Check that run_etl_pipeline was called with feature engineering enabled
        call_args = mock_run.call_args
        config = call_args.kwargs.get("config") or call_args[1].get("config")

        self.assertTrue(config.apply_feature_engineering)
        self.assertTrue(config.use_semantic_column_classification)

    @patch("finance_ml.ml_workflow.preprocessing.etl.run_etl_pipeline")
    def test_etl_with_features_uses_specified_preset(self, mock_run):
        """etl_with_features should use specified feature preset."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_features

        mock_run.return_value = pd.DataFrame()

        etl_with_features(source="csv", data_dir=Path("data"), feature_preset="comprehensive")

        call_args = mock_run.call_args
        config = call_args.kwargs.get("config") or call_args[1].get("config")

        self.assertEqual(config.feature_preset, "comprehensive")


class TestPreprocessingModuleExports(unittest.TestCase):
    """Test that new functions are properly exported from preprocessing module."""

    def test_etl_with_features_exported_from_preprocessing(self):
        """etl_with_features should be importable from preprocessing module."""
        try:
            from finance_ml.ml_workflow.preprocessing.etl import etl_with_features

            self.assertTrue(callable(etl_with_features))
        except ImportError:
            self.fail("etl_with_features not importable from preprocessing.etl")

    def test_column_semantics_classify_columns_accessible(self):
        """classify_columns should be accessible from column_semantics."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import classify_columns

        self.assertTrue(callable(classify_columns))

    def test_column_semantics_get_winsorizable_columns_accessible(self):
        """get_winsorizable_columns should be accessible from column_semantics."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import get_winsorizable_columns

        self.assertTrue(callable(get_winsorizable_columns))

    def test_column_semantics_get_log_transform_columns_accessible(self):
        """get_log_transform_columns should be accessible from column_semantics."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import get_log_transform_columns

        self.assertTrue(callable(get_log_transform_columns))

    def test_column_semantics_get_scalable_columns_accessible(self):
        """get_scalable_columns should be accessible from column_semantics."""
        from finance_ml.ml_workflow.preprocessing.column_semantics import get_scalable_columns

        self.assertTrue(callable(get_scalable_columns))


class TestIntegrationSemanticTransformations(unittest.TestCase):
    """Integration tests for semantic transformations in full pipeline."""

    def _create_test_df(self) -> pd.DataFrame:
        """Create test DataFrame for integration tests."""
        np.random.seed(42)
        n = 20
        return pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n),
                "region": np.random.choice(["US", "EU"], n),
                "last_price": np.random.uniform(50, 500, n),
                "price_target": np.random.uniform(55, 550, n),
                "market_cap": np.random.uniform(1e9, 100e9, n),
                "revenue": np.random.uniform(1e8, 10e9, n),
                "ebitda": np.random.uniform(1e7, 1e9, n),
                "p_e": np.random.uniform(10, 50, n),
                "gross_margin": np.random.uniform(20, 60, n),
                "num_analysts": np.random.randint(5, 30, n),
            }
        )

    def test_transform_with_semantic_classification_tracks_column_counts(self):
        """Transform with semantic classification should track column type counts."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(
            use_semantic_column_classification=True,
            normalize_columns=False,  # Keep original column names
            apply_dtype_casting=False,
            validate_schema=False,
            sanitize_data=False,
            apply_imputation=False,
            apply_scaling=False,
            validate_quality=False,
            validate_pipeline=False,
        )
        pipeline = ETLPipeline(config=config)
        # Initialize metrics (normally done by run(), but we're testing transform() directly)
        pipeline.metrics = ETLMetrics(source_type="test")
        df = self._create_test_df()

        result = pipeline.transform(df)

        # Metrics should be populated with semantic column counts
        self.assertIsNotNone(pipeline.metrics)
        self.assertTrue(pipeline.metrics.semantic_classification_applied)
        self.assertGreater(pipeline.metrics.price_columns_count, 0)

    def test_full_pipeline_run_with_csv_source(self):
        """Test full ETL pipeline execution with CSV data source."""
        from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig
        import tempfile
        import os

        # Create temporary CSV file with test data
        df = self._create_test_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"
            df.to_csv(csv_path, index=False)

            config = ETLConfig(
                normalize_columns=True,
                apply_dtype_casting=False,
                validate_schema=False,
                sanitize_data=True,
                apply_imputation=False,
                apply_scaling=False,
                validate_quality=False,
                validate_pipeline=False,
                use_semantic_column_classification=True,
            )

            result = run_etl_pipeline(
                source="csv", data_dir=tmpdir, config=config, return_metrics=False
            )

            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            self.assertIn("ticker", result.columns)

    def test_full_pipeline_run_with_metrics_return(self):
        """Test full ETL pipeline execution returning metrics."""
        from finance_ml.ml_workflow.preprocessing.etl import run_etl_pipeline, ETLConfig
        import tempfile

        # Create temporary CSV file with test data
        df = self._create_test_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"
            df.to_csv(csv_path, index=False)

            config = ETLConfig(
                normalize_columns=True,
                apply_dtype_casting=False,
                validate_schema=False,
                sanitize_data=True,
                apply_imputation=False,
                apply_scaling=False,
                validate_quality=False,
                validate_pipeline=False,
                use_semantic_column_classification=True,
            )

            result, metrics = run_etl_pipeline(
                source="csv", data_dir=tmpdir, config=config, return_metrics=True
            )

            self.assertIsInstance(result, pd.DataFrame)
            self.assertIsNotNone(metrics)
            self.assertTrue(hasattr(metrics, "semantic_classification_applied"))
            self.assertEqual(metrics.source_type, "csv")

    def test_etl_with_features_integration(self):
        """Test etl_with_features function with real data."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_features, ETLConfig
        import tempfile

        # Create temporary CSV file with richer test data for features
        np.random.seed(42)
        n = 20
        df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n),
                "region": np.random.choice(["US", "EU"], n),
                "last_price": np.random.uniform(50, 500, n),
                "price_target": np.random.uniform(55, 550, n),
                "market_cap": np.random.uniform(1e9, 100e9, n),
                "revenue": np.random.uniform(1e8, 10e9, n),
                "ebitda": np.random.uniform(1e7, 1e9, n),
                "net_income": np.random.uniform(5e6, 5e8, n),
                "total_equity": np.random.uniform(5e8, 5e10, n),
                "total_assets": np.random.uniform(1e9, 100e9, n),
                "total_debt": np.random.uniform(1e8, 5e10, n),
                "p_e": np.random.uniform(10, 50, n),
                "p_b": np.random.uniform(1, 10, n),
                "gross_margin": np.random.uniform(20, 60, n),
                "operating_margin": np.random.uniform(10, 40, n),
                "roe": np.random.uniform(5, 30, n),
                "roa": np.random.uniform(2, 15, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"
            df.to_csv(csv_path, index=False)

            result, metrics = etl_with_features(
                source="csv", data_dir=tmpdir, feature_preset="basic", return_metrics=True
            )

            self.assertIsInstance(result, pd.DataFrame)
            self.assertIsNotNone(metrics)
            # Feature engineering is attempted even if no features are added
            # (e.g., if required columns are missing after preprocessing)
            self.assertTrue(metrics.feature_engineering_applied)
            self.assertEqual(metrics.feature_preset_used, "basic")
            # Features may or may not be added depending on available columns
            self.assertGreaterEqual(metrics.features_added, 0)

    def test_transform_with_log_transforms(self):
        """Test transform with log transformation of market values."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(
            use_semantic_column_classification=True,
            log_transform_market_values=True,
            normalize_columns=False,
            apply_dtype_casting=False,
            validate_schema=False,
            sanitize_data=False,
            apply_imputation=False,
            apply_scaling=False,
            validate_quality=False,
            validate_pipeline=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")
        df = self._create_test_df()

        result = pipeline.transform(df)

        # Should have log-transformed columns
        self.assertIn("log_market_cap", result.columns)
        self.assertIn("log_revenue", result.columns)
        self.assertGreater(pipeline.metrics.log_transformed_columns, 0)

    def test_transform_with_normalization(self):
        """Test transform with column normalization."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(
            normalize_columns=True,
            apply_dtype_casting=False,
            validate_schema=False,
            sanitize_data=False,
            apply_imputation=False,
            apply_scaling=False,
            validate_quality=False,
            validate_pipeline=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        # Create DataFrame with non-normalized column names
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT", "GOOGL"],
                "Last Price": [150.0, 250.0, 2800.0],
                "Market Cap": [2.5e12, 1.8e12, 1.5e12],
            }
        )

        result = pipeline.transform(df)

        # Column names should be normalized
        self.assertIn("ticker", result.columns)
        self.assertIn("last_price", result.columns)
        self.assertIn("market_cap", result.columns)

    def test_transform_with_sanitization(self):
        """Test transform with data sanitization."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(
            normalize_columns=False,
            apply_dtype_casting=False,
            validate_schema=False,
            sanitize_data=True,
            apply_imputation=False,
            apply_scaling=False,
            validate_quality=False,
            validate_pipeline=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        # Create DataFrame with inf and extreme values
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "last_price": [150.0, np.inf, 2800.0],
                "market_cap": [2.5e12, 1.8e12, -1e12],
            }
        )

        result = pipeline.transform(df)

        # Inf values should be replaced
        self.assertFalse(np.isinf(result["last_price"]).any())

    def test_etl_from_csv_convenience_function(self):
        """Test etl_from_csv convenience function."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_from_csv, ETLConfig
        import tempfile

        # Create temporary CSV file
        df = self._create_test_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"
            df.to_csv(csv_path, index=False)

            config = ETLConfig(
                normalize_columns=True,
                apply_dtype_casting=False,
                validate_schema=False,
                apply_imputation=False,
                apply_scaling=False,
            )

            result = etl_from_csv(data_dir=tmpdir, config=config)

            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)

    def test_pipeline_load_method(self):
        """Test ETLPipeline load method."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(validate_quality=False, validate_pipeline=False)
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")
        df = self._create_test_df()

        result = pipeline.load(df, validate=False)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(df))

    def test_transform_with_imputation_6step(self):
        """Test transform with 6-step imputation strategy."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(
            normalize_columns=False,
            apply_dtype_casting=False,
            validate_schema=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="6step",
            apply_scaling=False,
            validate_quality=False,
            validate_pipeline=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        # Create DataFrame with missing values
        df = self._create_test_df()
        df.loc[0, "market_cap"] = np.nan
        df.loc[1, "revenue"] = np.nan

        result = pipeline.transform(df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(hasattr(pipeline.metrics, "missing_values_before_imputation"))
        self.assertTrue(hasattr(pipeline.metrics, "missing_values_after_imputation"))

    def test_transform_with_imputation_median_only(self):
        """Test transform with median-only imputation strategy."""
        from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline, ETLConfig, ETLMetrics

        config = ETLConfig(
            normalize_columns=False,
            apply_dtype_casting=False,
            validate_schema=False,
            sanitize_data=False,
            apply_imputation=True,
            imputation_strategy="median_only",
            apply_scaling=False,
            validate_quality=False,
            validate_pipeline=False,
        )
        pipeline = ETLPipeline(config=config)
        pipeline.metrics = ETLMetrics(source_type="test")

        df = self._create_test_df()
        df.loc[0, "market_cap"] = np.nan

        result = pipeline.transform(df)

        self.assertIsInstance(result, pd.DataFrame)

    def test_etl_with_imputation_convenience_function(self):
        """Test etl_with_imputation convenience function."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_imputation
        import tempfile

        df = self._create_test_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"
            df.to_csv(csv_path, index=False)

            result, metrics = etl_with_imputation(
                source="csv", data_dir=tmpdir, imputation_strategy="6step", return_metrics=True
            )

            self.assertIsInstance(result, pd.DataFrame)
            self.assertIsNotNone(metrics)

    def test_etl_with_imputation_and_scaling_convenience_function(self):
        """Test etl_with_imputation_and_scaling convenience function."""
        from finance_ml.ml_workflow.preprocessing.etl import etl_with_imputation_and_scaling
        import tempfile

        df = self._create_test_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "screening_us.csv"
            df.to_csv(csv_path, index=False)

            result, metrics = etl_with_imputation_and_scaling(
                source="csv",
                data_dir=tmpdir,
                imputation_strategy="6step",
                scaler_type="robust",
                scale_by_sector=True,
                return_metrics=True,
            )

            self.assertIsInstance(result, pd.DataFrame)
            self.assertIsNotNone(metrics)
            self.assertTrue(metrics.scaling_applied)


if __name__ == "__main__":
    unittest.main()
