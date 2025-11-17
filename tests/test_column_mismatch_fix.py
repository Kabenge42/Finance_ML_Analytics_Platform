"""
Tests for column mismatch fix - TDD approach.

This module tests the extract_numeric_feature_columns() utility and
enhancements to train_sector_specific_models() to handle column mismatches
gracefully with auto-extraction and diagnostics.
"""

import unittest
import pandas as pd
import numpy as np
from finance_ml.advanced_models import (
    extract_numeric_feature_columns,
    train_sector_specific_models,
)


class TestExtractNumericFeatureColumns(unittest.TestCase):
    """Test the extract_numeric_feature_columns utility function."""

    def setUp(self):
        """Create test DataFrames with various column structures."""
        # Basic DataFrame with mixed types
        self.df_basic = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Technology", "Technology", "Technology"],
                "region": ["US", "US", "US"],
                "last_price": [150.0, 300.0, 2800.0],
                "market_cap": [2.5e12, 2.3e12, 1.8e12],
                "p_e_ltm": [28.5, 32.1, 25.3],
                "ebitda_ltm": [1e11, 9e10, 8e10],
                "price_target": [180.0, 350.0, 3000.0],
            }
        )

        # DataFrame with additional metadata columns
        self.df_with_metadata = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "name": ["Apple Inc.", "Microsoft Corporation"],
                "isin": ["US0378331005", "US5949181045"],
                "sector": ["Technology", "Technology"],
                "last_price": [150.0, 300.0],
                "revenue": [3.94e11, 1.98e11],
                "net_income": [9.7e10, 7.3e10],
                "price_target": [180.0, 350.0],
                "analyst_target_price": [175.0, 340.0],
            }
        )

        # DataFrame with event classification features
        self.df_with_events = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [150.0, 300.0, 2800.0],
                "market_cap": [2.5e12, 2.3e12, 1.8e12],
                "event_label": [0, 1, 2],
                "event_proba_neutral": [0.6, 0.2, 0.1],
                "event_proba_positive": [0.3, 0.7, 0.2],
                "event_proba_negative": [0.1, 0.1, 0.7],
                "price_target": [180.0, 350.0, 3000.0],
            }
        )

    def test_extract_excludes_target_columns(self):
        """Test that target columns are excluded."""
        result = extract_numeric_feature_columns(self.df_basic, exclude_cols=["price_target"])
        self.assertNotIn("price_target", result)
        self.assertIn("last_price", result)
        self.assertIn("market_cap", result)

    def test_extract_excludes_metadata(self):
        """Test that metadata columns are excluded by default."""
        result = extract_numeric_feature_columns(self.df_with_metadata)
        # Should exclude ticker, name, isin by default
        self.assertNotIn("ticker", result)
        self.assertNotIn("name", result)
        self.assertNotIn("isin", result)
        # Should include numeric features
        self.assertIn("last_price", result)
        self.assertIn("revenue", result)

    def test_extract_excludes_event_labels(self):
        """Test that event labels and probabilities are excluded."""
        result = extract_numeric_feature_columns(
            self.df_with_events, exclude_cols=["price_target", "event_label"]
        )
        self.assertNotIn("event_label", result)
        self.assertNotIn("event_proba_neutral", result)
        self.assertNotIn("event_proba_positive", result)
        self.assertNotIn("event_proba_negative", result)

    def test_extract_only_numeric_columns(self):
        """Test that only numeric columns are returned."""
        result = extract_numeric_feature_columns(self.df_basic)
        for col in result:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(self.df_basic[col]), f"Column {col} is not numeric"
            )

    def test_extract_custom_exclude_list(self):
        """Test custom exclusion list."""
        result = extract_numeric_feature_columns(
            self.df_basic, exclude_cols=["last_price", "price_target"]
        )
        self.assertNotIn("last_price", result)
        self.assertNotIn("price_target", result)
        self.assertIn("market_cap", result)

    def test_extract_empty_dataframe(self):
        """Test with empty DataFrame."""
        df_empty = pd.DataFrame()
        result = extract_numeric_feature_columns(df_empty)
        self.assertEqual(len(result), 0)

    def test_extract_no_numeric_columns(self):
        """Test DataFrame with no numeric columns."""
        df_text = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "name": ["Apple", "Microsoft"], "sector": ["Tech", "Tech"]}
        )
        result = extract_numeric_feature_columns(df_text)
        self.assertEqual(len(result), 0)

    def test_extract_returns_list(self):
        """Test that result is a list."""
        result = extract_numeric_feature_columns(self.df_basic)
        self.assertIsInstance(result, list)

    def test_extract_preserves_column_order(self):
        """Test that column order is preserved."""
        result = extract_numeric_feature_columns(self.df_basic)
        # Get original order of numeric columns
        original_numeric = [
            col
            for col in self.df_basic.columns
            if pd.api.types.is_numeric_dtype(self.df_basic[col])
        ]
        # Filter to just those in result
        expected_order = [col for col in original_numeric if col in result]
        self.assertEqual(result, expected_order)


class TestTrainSectorSpecificModelsWithAutoExtract(unittest.TestCase):
    """Test train_sector_specific_models with auto-extraction fallback."""

    def setUp(self):
        """Create test DataFrame with multiple sectors."""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(50, 500, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "p_e_ltm": np.random.uniform(10, 40, n_samples),
                "revenue": np.random.uniform(1e8, 1e11, n_samples),
                "ebitda": np.random.uniform(1e7, 1e10, n_samples),
                "price_target": np.random.uniform(60, 600, n_samples),
            }
        )

    def test_dict_with_invalid_features_raises_error(self):
        """Test that dict with invalid features raises ValueError currently."""
        invalid_dict = {
            "all_features": ["nonexistent_col1", "nonexistent_col2"],
            "numeric_features": ["fake_feature1", "fake_feature2"],
        }
        with self.assertRaises(ValueError) as ctx:
            train_sector_specific_models(
                df=self.df,
                feature_cols=invalid_dict,
                target_col="price_target",
                min_samples=10,
            )
        self.assertIn("No valid feature columns remain", str(ctx.exception))

    def test_list_with_valid_features_succeeds(self):
        """Test that list with valid features works."""
        valid_features = ["last_price", "market_cap", "p_e_ltm", "revenue"]
        sector_models, results = train_sector_specific_models(
            df=self.df,
            feature_cols=valid_features,
            target_col="price_target",
            min_samples=10,
        )
        self.assertGreater(len(sector_models), 0)
        self.assertIn("n_sectors", results)

    def test_dict_with_some_valid_features_succeeds(self):
        """Test that dict with some valid features filters to valid ones."""
        mixed_dict = {
            "all_features": [
                "last_price",
                "market_cap",  # Valid
                "nonexistent1",
                "nonexistent2",  # Invalid
            ]
        }
        sector_models, results = train_sector_specific_models(
            df=self.df,
            feature_cols=mixed_dict,
            target_col="price_target",
            min_samples=10,
        )
        self.assertGreater(len(sector_models), 0)

    def test_auto_extract_fallback_works(self):
        """Test that auto-extract fallback successfully extracts features."""
        invalid_dict = {"all_features": ["nonexistent_col1", "nonexistent_col2"]}

        # With auto_extract_fallback=True, should succeed by extracting actual features
        sector_models, results = train_sector_specific_models(
            df=self.df,
            feature_cols=invalid_dict,
            target_col="price_target",
            min_samples=10,
            auto_extract_fallback=True,
        )

        # Should have successfully trained regression
        self.assertGreater(len(sector_models), 0)
        self.assertIn("n_sectors", results)

    def test_auto_extract_fallback_disabled_raises_error(self):
        """Test that without auto_extract_fallback, invalid features raise ValueError."""
        invalid_dict = {"all_features": ["nonexistent_col1", "nonexistent_col2"]}

        # Without auto_extract_fallback (default False), should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            train_sector_specific_models(
                df=self.df,
                feature_cols=invalid_dict,
                target_col="price_target",
                min_samples=10,
                auto_extract_fallback=False,  # Explicitly disabled
            )

        # Error message should be helpful
        error_msg = str(ctx.exception)
        self.assertIn("No valid feature columns remain", error_msg)
        self.assertIn("auto_extract_fallback=True", error_msg)


class TestDiagnosticMessages(unittest.TestCase):
    """Test diagnostic messages and logging."""

    def setUp(self):
        """Create minimal test DataFrame."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"] * 20,
                "sector": ["Tech"] * 60,
                "feature1": np.random.rand(60),
                "feature2": np.random.rand(60),
                "price_target": np.random.rand(60) * 100,
            }
        )

    def test_diagnostic_output_structure(self):
        """Test that diagnostics include DataFrame structure info."""
        # This is a placeholder - actual diagnostic output will be logged
        result = extract_numeric_feature_columns(self.df)
        self.assertIsInstance(result, list)
        # Future: capture and verify log messages


if __name__ == "__main__":
    unittest.main()
