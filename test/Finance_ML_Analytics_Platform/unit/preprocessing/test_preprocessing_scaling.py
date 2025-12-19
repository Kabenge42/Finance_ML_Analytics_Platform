"""
Tests for finance_ml.ml_workflow.preprocessing.scaling module.

Phase 9.1 refactor: Testing scaling functions moved from advanced_preprocessing.py.
"""

import unittest
import numpy as np
import pandas as pd
from finance_ml.ml_workflow.preprocessing.scaling import (
    create_scaler_pipeline,
    scale_features,
)


class TestScalerPipeline(unittest.TestCase):
    """Test scaler pipeline creation."""

    def test_create_robust_scaler(self):
        """Test creating robust scaler."""
        scaler, by_sector = create_scaler_pipeline(scaler_type="robust", by_sector=False)

        # Should return a scaler object and flag
        self.assertIsNotNone(scaler)
        self.assertTrue(hasattr(scaler, "fit"))
        self.assertTrue(hasattr(scaler, "transform"))
        self.assertFalse(by_sector)

    def test_create_standard_scaler(self):
        """Test creating standard scaler."""
        scaler, by_sector = create_scaler_pipeline(scaler_type="standard", by_sector=False)

        self.assertIsNotNone(scaler)
        self.assertTrue(hasattr(scaler, "fit"))
        self.assertFalse(by_sector)

    def test_create_minmax_scaler(self):
        """Test creating min-max scaler."""
        scaler, by_sector = create_scaler_pipeline(scaler_type="minmax", by_sector=False)

        self.assertIsNotNone(scaler)
        self.assertTrue(hasattr(scaler, "fit"))
        self.assertFalse(by_sector)

    def test_invalid_scaler_type(self):
        """Test that invalid scaler type raises error or returns default."""
        # Should either raise ValueError or return a default scaler
        try:
            scaler = create_scaler_pipeline(scaler_type="invalid", by_sector=False)
            # If it doesn't raise, should return some valid scaler
            self.assertIsNotNone(scaler)
        except ValueError:
            # Expected behavior
            pass


class TestScaleFeatures(unittest.TestCase):
    """Test feature scaling functionality."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "metric1": np.random.normal(100, 20, 100),
                "metric2": np.random.normal(50, 10, 100),
                "metric3": np.random.uniform(0, 1000, 100),
            }
        )

    def test_scale_features_robust(self):
        """Test robust scaling."""
        result = scale_features(
            self.df.copy(), columns=["metric1", "metric2"], scaler_type="robust", by_sector=False
        )

        # Should return DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.df.shape)

        # Scaled features should have different statistics
        self.assertNotAlmostEqual(result["metric1"].mean(), self.df["metric1"].mean(), places=1)

    def test_scale_features_by_sector(self):
        """Test scaling with sector grouping."""
        result = scale_features(
            self.df.copy(), columns=["metric1"], scaler_type="robust", by_sector=True
        )

        self.assertIsInstance(result, pd.DataFrame)

        # Check that sector column is preserved
        pd.testing.assert_series_equal(result["sector"], self.df["sector"])

    def test_scale_features_standard(self):
        """Test standard scaling."""
        result = scale_features(
            self.df.copy(), columns=["metric1", "metric2"], scaler_type="standard", by_sector=False
        )

        # Standard scaling should result in approximately zero mean
        self.assertAlmostEqual(result["metric1"].mean(), 0, places=10)

    def test_scale_features_minmax(self):
        """Test min-max scaling."""
        result = scale_features(
            self.df.copy(), columns=["metric1"], scaler_type="minmax", by_sector=False
        )

        # Min-max scaling should result in [0, 1] range
        self.assertGreaterEqual(result["metric1"].min(), -0.01)
        self.assertLessEqual(result["metric1"].max(), 1.01)

    def test_scale_features_preserves_non_scaled_columns(self):
        """Test that non-scaled columns are preserved."""
        result = scale_features(self.df.copy(), columns=["metric1"], scaler_type="robust")

        # metric2 and metric3 should be unchanged
        pd.testing.assert_series_equal(result["metric2"], self.df["metric2"])

    def test_scale_features_default_columns(self):
        """Test scaling with default columns (all numeric)."""
        result = scale_features(
            self.df.copy(),
            columns=None,  # Should default to all numeric
            scaler_type="robust",
            by_sector=False,
        )

        self.assertIsInstance(result, pd.DataFrame)
        # Should have scaled all numeric columns


class TestScalingEdgeCases(unittest.TestCase):
    """Test edge cases for scaling."""

    def test_scale_empty_dataframe(self):
        """Test scaling with empty DataFrame."""
        df = pd.DataFrame()

        # Empty dataframe with no columns should return empty
        # If columns parameter is None, it will select numeric columns (none exist)
        # So this should either return empty df or handle gracefully
        try:
            result = scale_features(df, scaler_type="robust")
            self.assertEqual(len(result), 0)
        except (ValueError, IndexError):
            # Acceptable to raise error on empty dataframe
            pass

    def test_scale_single_column(self):
        """Test scaling with single column."""
        df = pd.DataFrame({"metric1": [1, 2, 3, 4, 5]})

        result = scale_features(df.copy(), columns=["metric1"], scaler_type="robust")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)

    def test_scale_with_missing_values(self):
        """Test scaling with missing values."""
        df = pd.DataFrame({"metric1": [1, 2, np.nan, 4, 5]})

        # Should handle NaNs gracefully
        result = scale_features(df.copy(), columns=["metric1"], scaler_type="robust")

        self.assertIsInstance(result, pd.DataFrame)

    def test_scale_constant_column(self):
        """Test scaling column with constant values."""
        df = pd.DataFrame({"metric1": [5, 5, 5, 5, 5]})

        result = scale_features(df.copy(), columns=["metric1"], scaler_type="robust")

        # Should handle constant values without error
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
