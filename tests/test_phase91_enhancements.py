"""
Unit tests for Phase 9.1 Future Enhancements.

Tests for 5 key enhancements:
1. KNN Imputation with sector-aware logic
2. Advanced Target Encoding with regularization
3. Data Quality Dashboard (pandas-profiling/sweetviz)
4. TensorFlow Dataset API for large-scale processing
5. Custom sklearn-compatible Financial Transformers

Follows TDD approach: write failing tests first, then implement minimal code to pass.
"""

import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Import functions to be tested
from finance_ml.advanced_preprocessing import (
    impute_missing_values_knn_sector,
)

from finance_ml.transformers import (
    TargetEncoder,
    RegularizedTargetEncoder,
    FinancialRatioTransformer,
    SafeDivisionTransformer,
    ValuationRatioTransformer,
)

from finance_ml.eval import (
    generate_data_quality_dashboard,
    export_profiling_report,
)

# TensorFlow Dataset functions (not yet implemented)
try:
    from finance_ml.data import (
        create_tf_dataset,
        create_tf_dataset_pipeline,
    )
except ImportError as e:
    warnings.warn(f"Import failed for TensorFlow Dataset (expected for TDD): {e}")


def create_sample_financial_data(n_samples=200, with_missing=True, random_state=42):
    """
    Create sample financial data for testing.

    Args:
        n_samples: Number of samples to generate
        with_missing: Whether to introduce missing values
        random_state: Random seed for reproducibility

    Returns:
        pd.DataFrame: Sample financial dataset
    """
    np.random.seed(random_state)

    sectors = ["Technology", "Finance", "Healthcare", "Energy", "Consumer"]
    regions = ["US", "EU", "APAC", "ROTW"]

    df = pd.DataFrame(
        {
            "ticker": [f"TICK{i:04d}" for i in range(n_samples)],
            "sector": np.random.choice(sectors, n_samples),
            "region": np.random.choice(regions, n_samples),
            "market_cap": np.random.lognormal(10, 2, n_samples),
            "revenue": np.random.lognormal(8, 1.5, n_samples),
            "ebitda": np.random.lognormal(7, 1.5, n_samples),
            "net_income": np.random.lognormal(6, 2, n_samples),
            "total_assets": np.random.lognormal(9, 2, n_samples),
            "total_equity": np.random.lognormal(8, 2, n_samples),
            "total_debt": np.random.lognormal(7, 2, n_samples),
            "enterprise_value": np.random.lognormal(10, 2, n_samples),
            "last_price": np.random.uniform(10, 500, n_samples),
            "book_value_per_share": np.random.uniform(5, 200, n_samples),
            "eps": np.random.uniform(-5, 50, n_samples),
            "price_target": np.random.uniform(10, 500, n_samples),
        }
    )

    # Add some zeros and negative values for edge cases
    df.loc[np.random.choice(n_samples, 5, replace=False), "ebitda"] = 0
    df.loc[np.random.choice(n_samples, 5, replace=False), "net_income"] = np.random.uniform(
        -100, 0, 5
    )
    df.loc[np.random.choice(n_samples, 3, replace=False), "total_equity"] = 0

    if with_missing:
        # Introduce missing values (10-20% per column)
        for col in ["revenue", "ebitda", "net_income", "total_assets", "eps"]:
            n_missing = int(n_samples * np.random.uniform(0.1, 0.2))
            missing_idx = np.random.choice(n_samples, n_missing, replace=False)
            df.loc[missing_idx, col] = np.nan

    return df


# ============================================================================
# Enhancement 1: KNN Imputation with Sector-Aware Logic
# ============================================================================


class TestKNNImputationSectorAware(unittest.TestCase):
    """Test sector-aware KNN imputation."""

    def setUp(self):
        """Set up test data."""
        self.df = create_sample_financial_data(n_samples=150, with_missing=True)

    def test_knn_imputation_fills_all_missing_values(self):
        """Test that KNN imputation fills all missing values."""
        # Count missing values before
        missing_before = self.df["revenue"].isna().sum()
        self.assertGreater(missing_before, 0, "Test data should have missing values")

        # Apply KNN imputation
        df_imputed = impute_missing_values_knn_sector(
            self.df, columns=["revenue"], sector_column="sector", n_neighbors=5
        )

        # Count missing values after
        missing_after = df_imputed["revenue"].isna().sum()
        self.assertEqual(missing_after, 0, "All missing values should be filled")

    def test_knn_imputation_preserves_non_missing_values(self):
        """Test that KNN imputation doesn't change non-missing values."""
        # Get indices of non-missing values
        non_missing_mask = ~self.df["revenue"].isna()
        original_values = self.df.loc[non_missing_mask, "revenue"].copy()

        # Apply KNN imputation
        df_imputed = impute_missing_values_knn_sector(
            self.df, columns=["revenue"], sector_column="sector", n_neighbors=5
        )

        # Check that non-missing values are unchanged
        imputed_values = df_imputed.loc[non_missing_mask, "revenue"]
        pd.testing.assert_series_equal(original_values, imputed_values)

    def test_knn_imputation_sector_aware(self):
        """Test that imputation uses sector-specific neighbors."""
        # Create data where sectors have distinct value ranges
        df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "revenue": np.concatenate(
                    [np.random.uniform(1000, 2000, 50), np.random.uniform(100, 200, 50)]
                ),
            }
        )

        # Introduce missing values
        df.loc[25, "revenue"] = np.nan  # Tech sector
        df.loc[75, "revenue"] = np.nan  # Finance sector

        # Apply sector-aware KNN imputation
        df_imputed = impute_missing_values_knn_sector(
            df, columns=["revenue"], sector_column="sector", n_neighbors=5
        )

        # Imputed values should be in the range of their respective sectors
        tech_imputed = df_imputed.loc[25, "revenue"]
        finance_imputed = df_imputed.loc[75, "revenue"]

        self.assertTrue(
            1000 <= tech_imputed <= 2000,
            f"Tech imputed value {tech_imputed} outside expected range",
        )
        self.assertTrue(
            100 <= finance_imputed <= 200,
            f"Finance imputed value {finance_imputed} outside expected range",
        )

    def test_knn_imputation_configurable_neighbors(self):
        """Test that n_neighbors parameter works correctly."""
        df_k3 = impute_missing_values_knn_sector(
            self.df, columns=["revenue"], sector_column="sector", n_neighbors=3
        )

        df_k7 = impute_missing_values_knn_sector(
            self.df, columns=["revenue"], sector_column="sector", n_neighbors=7
        )

        # Both should fill all missing values
        self.assertEqual(df_k3["revenue"].isna().sum(), 0)
        self.assertEqual(df_k7["revenue"].isna().sum(), 0)

        # Results might be slightly different due to different k
        self.assertIsInstance(df_k3, pd.DataFrame)
        self.assertIsInstance(df_k7, pd.DataFrame)

    def test_knn_imputation_multiple_columns(self):
        """Test KNN imputation on multiple columns simultaneously."""
        columns = ["revenue", "ebitda", "net_income"]

        # Count missing values before
        missing_before = {col: self.df[col].isna().sum() for col in columns}
        self.assertTrue(
            all(m > 0 for m in missing_before.values()),
            "Test data should have missing values in all columns",
        )

        # Apply KNN imputation
        df_imputed = impute_missing_values_knn_sector(
            self.df, columns=columns, sector_column="sector", n_neighbors=5
        )

        # Check all columns are filled
        for col in columns:
            self.assertEqual(
                df_imputed[col].isna().sum(), 0, f"Column {col} should have no missing values"
            )

    def test_knn_imputation_fallback_without_sector(self):
        """Test that imputation works when sector column is missing."""
        df_no_sector = self.df.drop(columns=["sector"])

        # Should fall back to global KNN imputation
        df_imputed = impute_missing_values_knn_sector(
            df_no_sector,
            columns=["revenue"],
            sector_column="sector",  # Column doesn't exist
            n_neighbors=5,
        )

        # Should still fill missing values
        self.assertEqual(df_imputed["revenue"].isna().sum(), 0)


# ============================================================================
# Enhancement 2: Advanced Target Encoding with Regularization
# ============================================================================


class TestRegularizedTargetEncoding(unittest.TestCase):
    """Test regularized target encoding with cross-validation."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200

        sectors = ["Tech", "Finance", "Healthcare", "Energy", "Consumer"]

        # Create sector with varying frequencies
        self.df = pd.DataFrame(
            {
                "sector": np.random.choice(sectors, n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                "industry": np.random.choice(["A", "B", "C", "D", "E", "F", "G", "H"], n),
                "region": np.random.choice(["US", "EU", "APAC"], n),
                "target": np.random.uniform(0, 100, n),
            }
        )

        # Make target correlated with sector
        sector_effects = {"Tech": 20, "Finance": -10, "Healthcare": 5, "Energy": -5, "Consumer": 0}
        for sector, effect in sector_effects.items():
            mask = self.df["sector"] == sector
            self.df.loc[mask, "target"] += effect + np.random.normal(0, 5, mask.sum())

    def test_target_encoder_fit_transform(self):
        """Test basic target encoder fit and transform."""
        encoder = RegularizedTargetEncoder(columns=["sector"])

        # Fit on training data
        encoder.fit(self.df[["sector"]], self.df["target"])

        # Transform should return encoded values
        encoded = encoder.transform(self.df[["sector"]])

        self.assertIsInstance(encoded, pd.DataFrame)
        self.assertEqual(encoded.shape[0], len(self.df))
        self.assertIn("sector", encoded.columns)

    def test_target_encoder_prevents_leakage(self):
        """Test that target encoding uses cross-validation to prevent leakage."""
        encoder = RegularizedTargetEncoder(columns=["sector"], cv_folds=5)

        # Fit and transform with CV
        encoded = encoder.fit_transform(self.df[["sector"]], self.df["target"])

        # Encoded values should not be identical to direct mean (due to CV)
        direct_mean = self.df.groupby("sector")["target"].transform("mean")

        # They should be different due to out-of-fold encoding
        self.assertFalse(
            np.allclose(encoded["sector"].values, direct_mean.values),
            "CV encoding should differ from direct mean to prevent leakage",
        )

    def test_target_encoder_smoothing(self):
        """Test that smoothing parameter affects encoding."""
        encoder_no_smooth = RegularizedTargetEncoder(columns=["industry"], smoothing=0)
        encoder_smooth = RegularizedTargetEncoder(columns=["industry"], smoothing=10)

        # Fit both
        encoded_no_smooth = encoder_no_smooth.fit_transform(
            self.df[["industry"]], self.df["target"]
        )
        encoded_smooth = encoder_smooth.fit_transform(self.df[["industry"]], self.df["target"])

        # Smoothing should pull rare categories toward global mean
        variance_no_smooth = encoded_no_smooth["industry"].var()
        variance_smooth = encoded_smooth["industry"].var()

        self.assertLess(
            variance_smooth,
            variance_no_smooth,
            "Smoothing should reduce variance of encoded values",
        )

    def test_target_encoder_handles_unseen_categories(self):
        """Test handling of categories not seen during training."""
        encoder = RegularizedTargetEncoder(columns=["sector"])

        # Fit on subset
        train_df = self.df[self.df["sector"] != "Consumer"]
        encoder.fit(train_df[["sector"]], train_df["target"])

        # Transform including unseen category
        test_df = self.df[self.df["sector"] == "Consumer"]
        encoded = encoder.transform(test_df[["sector"]])

        # Should handle gracefully (use global mean or default value)
        self.assertFalse(
            encoded["sector"].isna().any(), "Unseen categories should be encoded, not NaN"
        )

    def test_target_encoder_multiple_columns(self):
        """Test encoding multiple categorical columns."""
        encoder = RegularizedTargetEncoder(columns=["sector", "region"])

        encoded = encoder.fit_transform(self.df[["sector", "region"]], self.df["target"])

        self.assertIn("sector", encoded.columns)
        self.assertIn("region", encoded.columns)
        self.assertEqual(encoded.shape[0], len(self.df))

        # Both columns should be numeric after encoding
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded["sector"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded["region"]))


# ============================================================================
# Enhancement 3: Data Quality Dashboard
# ============================================================================


class TestDataQualityDashboard(unittest.TestCase):
    """Test data quality dashboard generation."""

    def setUp(self):
        """Set up test data and temp directory."""
        self.df = create_sample_financial_data(n_samples=200, with_missing=True)
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dashboard_generation_creates_html_report(self):
        """Test that dashboard generation creates HTML report."""
        report_path = generate_data_quality_dashboard(
            self.df, output_dir=self.output_dir, title="Test Financial Data Quality Report"
        )

        self.assertTrue(report_path.exists(), "HTML report should be created")
        self.assertEqual(report_path.suffix, ".html")

        # Check file is not empty
        self.assertGreater(
            report_path.stat().st_size, 1000, "Report should contain substantial content"
        )

    def test_dashboard_includes_data_quality_metrics(self):
        """Test that dashboard includes key data quality metrics."""
        report_path = generate_data_quality_dashboard(self.df, output_dir=self.output_dir)

        # Read HTML content
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for key sections
        self.assertIn("missing", content.lower(), "Should report missing values")
        self.assertTrue(
            "dtype" in content.lower() or "type" in content.lower(), "Should report data types"
        )

    def test_dashboard_with_profiling_library(self):
        """Test dashboard generation using pandas-profiling or sweetviz."""
        # This test will use the underlying profiling library
        report_path = generate_data_quality_dashboard(
            self.df, output_dir=self.output_dir, method="pandas-profiling"  # or 'sweetviz'
        )

        self.assertTrue(report_path.exists())
        self.assertGreater(
            report_path.stat().st_size, 5000, "Profiling report should be comprehensive"
        )

    def test_export_profiling_report_to_file(self):
        """Test exporting profiling report to file."""
        output_path = self.output_dir / "quality_report.html"

        success = export_profiling_report(self.df, output_path=output_path, minimal=False)

        self.assertTrue(success, "Export should succeed")
        self.assertTrue(output_path.exists())


# ============================================================================
# Enhancement 4: TensorFlow Dataset API
# ============================================================================


class TestTensorFlowDatasetAPI(unittest.TestCase):
    """Test TensorFlow Dataset API for large-scale data processing."""

    def setUp(self):
        """Set up test data."""
        self.df = create_sample_financial_data(n_samples=300, with_missing=False)

        # Prepare features and target
        self.feature_cols = ["market_cap", "revenue", "ebitda", "total_assets"]
        self.target_col = "price_target"

    def test_create_tf_dataset_from_dataframe(self):
        """Test creating TensorFlow Dataset from DataFrame."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed")

        dataset = create_tf_dataset(
            self.df, feature_columns=self.feature_cols, target_column=self.target_col
        )

        self.assertIsInstance(dataset, tf.data.Dataset)

    def test_tf_dataset_batching(self):
        """Test that TensorFlow Dataset supports batching."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed")

        batch_size = 32
        dataset = create_tf_dataset(
            self.df,
            feature_columns=self.feature_cols,
            target_column=self.target_col,
            batch_size=batch_size,
        )

        # Check batch shape
        for features, targets in dataset.take(1):
            self.assertEqual(features.shape[0], batch_size)
            self.assertEqual(targets.shape[0], batch_size)

    def test_tf_dataset_pipeline_with_prefetch(self):
        """Test TensorFlow Dataset pipeline with prefetching and caching."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed")

        dataset = create_tf_dataset_pipeline(
            self.df,
            feature_columns=self.feature_cols,
            target_column=self.target_col,
            batch_size=32,
            shuffle=True,
            prefetch=True,
            cache=True,
        )

        self.assertIsInstance(dataset, tf.data.Dataset)

        # Should be iterable
        count = 0
        for _ in dataset.take(5):
            count += 1
        self.assertEqual(count, 5)

    def test_tf_dataset_with_sector_stratification(self):
        """Test creating stratified dataset by sector."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed")

        dataset = create_tf_dataset(
            self.df,
            feature_columns=self.feature_cols,
            target_column=self.target_col,
            stratify_column="sector",
        )

        self.assertIsInstance(dataset, tf.data.Dataset)


# ============================================================================
# Enhancement 5: Custom sklearn-compatible Financial Transformers
# ============================================================================


class TestFinancialRatioTransformer(unittest.TestCase):
    """Test custom sklearn-compatible transformers for financial ratios."""

    def setUp(self):
        """Set up test data."""
        self.df = create_sample_financial_data(n_samples=150, with_missing=False)

    def test_financial_ratio_transformer_sklearn_compatible(self):
        """Test that transformer follows sklearn API."""
        from sklearn.base import BaseEstimator, TransformerMixin

        transformer = FinancialRatioTransformer()

        self.assertTrue(isinstance(transformer, BaseEstimator))
        self.assertTrue(isinstance(transformer, TransformerMixin))
        self.assertTrue(hasattr(transformer, "fit"))
        self.assertTrue(hasattr(transformer, "transform"))
        self.assertTrue(hasattr(transformer, "fit_transform"))

    def test_financial_ratio_transformer_calculates_pe_ratio(self):
        """Test calculation of P/E ratio."""
        transformer = FinancialRatioTransformer(ratios=["p_e"])

        X = self.df[["last_price", "eps"]].copy()
        X_transformed = transformer.fit_transform(X)

        self.assertIn("p_e", X_transformed.columns)

        # Check calculation is correct for non-zero EPS
        mask = X["eps"] > 0
        expected_pe = (X.loc[mask, "last_price"] / X.loc[mask, "eps"]).values
        actual_pe = X_transformed.loc[mask, "p_e"].values

        np.testing.assert_array_almost_equal(expected_pe, actual_pe, decimal=2)

    def test_financial_ratio_transformer_handles_division_by_zero(self):
        """Test safe handling of division by zero."""
        transformer = FinancialRatioTransformer(ratios=["p_b"])

        X = self.df[["last_price", "book_value_per_share"]].copy()

        # Introduce some zero values
        X.loc[0:5, "book_value_per_share"] = 0

        X_transformed = transformer.fit_transform(X)

        # Should not contain inf values
        self.assertFalse(np.isinf(X_transformed["p_b"]).any(), "Should not have infinite values")

        # Zero denominator cases should be handled (NaN or capped value)
        self.assertTrue(X_transformed["p_b"].notna().any(), "Should have some valid values")

    def test_valuation_ratio_transformer_multiple_ratios(self):
        """Test calculation of multiple valuation ratios."""
        transformer = ValuationRatioTransformer(ratios=["ev_ebitda", "p_e", "p_b"])

        X = self.df[
            ["enterprise_value", "ebitda", "last_price", "eps", "book_value_per_share"]
        ].copy()

        X_transformed = transformer.fit_transform(X)

        # Check all ratios are computed
        self.assertIn("ev_ebitda", X_transformed.columns)
        self.assertIn("p_e", X_transformed.columns)
        self.assertIn("p_b", X_transformed.columns)

    def test_safe_division_transformer(self):
        """Test safe division transformer handles edge cases."""
        transformer = SafeDivisionTransformer(
            numerator_col="market_cap",
            denominator_col="total_equity",
            output_col="market_to_book",
            fill_value=np.nan,
        )

        X = self.df[["market_cap", "total_equity"]].copy()

        X_transformed = transformer.fit_transform(X)

        self.assertIn("market_to_book", X_transformed.columns)

        # Should handle zero denominators
        self.assertFalse(np.isinf(X_transformed["market_to_book"]).any())

    def test_transformer_in_sklearn_pipeline(self):
        """Test that custom transformer works in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(
            [
                ("ratios", FinancialRatioTransformer(ratios=["p_e", "p_b"])),
                ("scaler", StandardScaler()),
            ]
        )

        X = self.df[["last_price", "eps", "book_value_per_share"]].copy()

        # Should be able to fit and transform
        X_transformed = pipeline.fit_transform(X)

        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertEqual(X_transformed.shape[0], len(X))


if __name__ == "__main__":
    unittest.main()
