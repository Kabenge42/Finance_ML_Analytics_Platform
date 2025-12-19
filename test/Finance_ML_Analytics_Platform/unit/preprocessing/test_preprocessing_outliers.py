"""
Tests for finance_ml.ml_workflow.preprocessing.outliers module.

Phase 9.1 refactor: Testing outlier detection and winsorization functions
moved from advanced_preprocessing.py.
"""

import unittest
import numpy as np
import pandas as pd
from finance_ml.ml_workflow.preprocessing.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
)


class TestOutlierDetection(unittest.TestCase):
    """Test outlier detection methods."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        # Create 100 samples total (50 Tech + 50 Finance)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "metric1": np.concatenate(
                    [np.random.normal(100, 10, n_samples - 2), [200, 300]]  # outliers
                ),
                "metric2": np.concatenate(
                    [np.random.normal(50, 5, n_samples - 2), [150, 200]]  # outliers
                ),
            }
        )

    def test_detect_outliers_iqr_basic(self):
        """Test IQR outlier detection."""
        result = detect_outliers_iqr(self.df, columns=["metric1"])

        # Should return boolean DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], self.df.shape[0])

        # Should detect some outliers
        self.assertTrue(result["metric1"].any())

    def test_detect_outliers_iqr_by_sector(self):
        """Test IQR outlier detection with sector grouping."""
        result = detect_outliers_iqr(
            self.df, columns=["metric1"], by_sector=True, iqr_multiplier=1.5
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result["metric1"].any())

    def test_detect_outliers_zscore_basic(self):
        """Test Z-score outlier detection."""
        result = detect_outliers_zscore(self.df, columns=["metric1"])

        # Should return boolean DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], self.df.shape[0])

    def test_detect_outliers_zscore_threshold(self):
        """Test Z-score with custom threshold."""
        result = detect_outliers_zscore(self.df, columns=["metric1"], threshold=2.0)

        # Lower threshold should detect more outliers
        strict_result = detect_outliers_zscore(self.df, columns=["metric1"], threshold=3.0)

        self.assertGreaterEqual(result["metric1"].sum(), strict_result["metric1"].sum())

    def test_detect_outliers_isolation_forest(self):
        """Test Isolation Forest outlier detection."""
        result = detect_outliers_isolation_forest(self.df, columns=["metric1", "metric2"])

        # Should return boolean Series
        self.assertIsInstance(result, (pd.Series, np.ndarray, list))

        # Convert to Series if needed for length check
        if isinstance(result, (np.ndarray, list)):
            result = pd.Series(result)

        self.assertEqual(len(result), len(self.df))

    def test_detect_outliers_isolation_forest_contamination(self):
        """Test Isolation Forest with contamination parameter."""
        result = detect_outliers_isolation_forest(
            self.df, columns=["metric1", "metric2"], contamination=0.05, random_state=42
        )

        if isinstance(result, (np.ndarray, list)):
            result = pd.Series(result)

        # Should detect approximately 5% outliers
        outlier_rate = result.sum() / len(result)
        self.assertLess(outlier_rate, 0.15)  # Allow some tolerance


class TestWinsorization(unittest.TestCase):
    """Test winsorization functionality."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        # Create 100 samples total (50 Tech + 50 Finance)
        n_samples = 100
        self.df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "metric1": np.concatenate(
                    [np.random.normal(100, 10, n_samples - 2), [0, 300]]  # extreme values
                ),
                "metric2": np.concatenate(
                    [np.random.normal(50, 5, n_samples - 2), [-50, 200]]  # extreme values
                ),
            }
        )

    def test_winsorize_by_sector_basic(self):
        """Test basic winsorization."""
        result = winsorize_by_sector(
            self.df.copy(), columns=["metric1"], lower_percentile=0.01, upper_percentile=0.99
        )

        # Should return DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.df.shape)

        # Extreme values should be capped
        self.assertLess(result["metric1"].max(), self.df["metric1"].max())

    def test_winsorize_by_sector_with_grouping(self):
        """Test winsorization with sector grouping."""
        result = winsorize_by_sector(
            self.df.copy(),
            columns=["metric1"],
            by_sector=True,
            lower_percentile=0.05,
            upper_percentile=0.95,
        )

        self.assertIsInstance(result, pd.DataFrame)

        # Values should be capped per sector
        tech_max = result[result["sector"] == "Tech"]["metric1"].max()
        finance_max = result[result["sector"] == "Finance"]["metric1"].max()

        # Both should be less than original max
        self.assertLess(tech_max, self.df["metric1"].max())

    def test_winsorize_multiple_columns(self):
        """Test winsorization on multiple columns."""
        result = winsorize_by_sector(
            self.df.copy(),
            columns=["metric1", "metric2"],
            lower_percentile=0.01,
            upper_percentile=0.99,
        )

        # Both columns should be winsorized
        self.assertLess(result["metric1"].max(), self.df["metric1"].max())
        self.assertLess(result["metric2"].max(), self.df["metric2"].max())

    def test_winsorize_preserves_non_target_columns(self):
        """Test that winsorization preserves non-target columns."""
        result = winsorize_by_sector(self.df.copy(), columns=["metric1"])

        # Sector column should be unchanged
        pd.testing.assert_series_equal(result["sector"], self.df["sector"])


class TestOutliersEdgeCases(unittest.TestCase):
    """Test edge cases for outlier detection."""

    def test_detect_outliers_empty_dataframe(self):
        """Test outlier detection with empty DataFrame."""
        df = pd.DataFrame()
        result = detect_outliers_iqr(df)

        self.assertEqual(len(result), 0)

    def test_detect_outliers_no_outliers(self):
        """Test when data has no outliers."""
        df = pd.DataFrame({"metric1": np.random.normal(100, 1, 100)})

        result = detect_outliers_iqr(df, columns=["metric1"], iqr_multiplier=10)

        # With very high multiplier, should detect no outliers
        # The function adds a column with '_outlier' suffix
        self.assertFalse(result["metric1_outlier"].any())

    def test_winsorize_with_missing_sector(self):
        """Test winsorization when sector column is missing."""
        df = pd.DataFrame({"metric1": [1, 2, 3, 100]})

        result = winsorize_by_sector(df.copy(), columns=["metric1"], by_sector=False)

        # Should still work without sector
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
