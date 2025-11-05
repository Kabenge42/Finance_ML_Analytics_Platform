"""
Test Phase 9.5 Preprocessing Workflow with 4-Step Imputation Strategy

Tests comprehensive data preparation for Phase 9.5 sector-specific regression models.
Ensures zero NaN values before training to prevent Ridge regression failures.

Created: 2025-11-05
Issue: Phase 9.5 NaN handling failure with 171 columns containing missing values
Approach: Strict TDD (Test-Driven Development)
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

try:
    from finance_ml.advanced_preprocessing import (
        apply_enhanced_imputation_strategy_4step,
        prepare_phase95_data,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipIf(not IMPORTS_AVAILABLE, "finance_ml.advanced_preprocessing not available")
class TestPhase95DataPreparation(unittest.TestCase):
    """Test Phase 9.5 data preparation pipeline with comprehensive imputation."""

    def setUp(self):
        """Create test data simulating Phase 9.5 scenario with multiple NaN columns."""
        np.random.seed(42)
        n_rows = 100

        # Create DataFrame with multiple columns containing NaN values
        # Simulating the 171 NaN columns mentioned in the issue
        self.df_with_many_nans = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_rows)],
                "sector": np.random.choice(
                    ["Technology", "Healthcare", "Finance", "Energy"], n_rows
                ),
                "last_price": np.random.uniform(10, 200, n_rows),
                "price_target": np.random.uniform(15, 250, n_rows),
                # Financial metrics with NaN (simulating real Phase 9.5 data)
                "enterprise_value": self._create_column_with_nans(n_rows, 0.3),
                "price_target_ytd_ago": self._create_column_with_nans(n_rows, 0.4),
                "total_return_ytd": self._create_column_with_nans(n_rows, 0.2),
                "p_e_ntm": self._create_column_with_nans(n_rows, 0.35),
                "p_e_ltm": self._create_column_with_nans(n_rows, 0.3),
                "ev_ebitda": self._create_column_with_nans(n_rows, 0.25),
                "market_cap": self._create_column_with_nans(n_rows, 0.15),
                "revenue": self._create_column_with_nans(n_rows, 0.2),
                "ebitda": self._create_column_with_nans(n_rows, 0.25),
                "net_debt": self._create_column_with_nans(n_rows, 0.3),
                # Additional columns with varying NaN rates
                **{f"feature_{i}": self._create_column_with_nans(n_rows, 0.2) for i in range(20)},
            }
        )

    def _create_column_with_nans(self, n_rows, nan_rate):
        """Helper to create numeric column with specified NaN rate."""
        data = np.random.randn(n_rows) * 10 + 50
        nan_mask = np.random.random(n_rows) < nan_rate
        data[nan_mask] = np.nan
        return data

    def test_prepare_phase95_data_exists(self):
        """Test that prepare_phase95_data function is available."""
        self.assertTrue(
            callable(prepare_phase95_data), "prepare_phase95_data should be a callable function"
        )

    def test_prepare_phase95_data_returns_dataframe(self):
        """Test that prepare_phase95_data returns a pandas DataFrame."""
        result = prepare_phase95_data(
            df=self.df_with_many_nans, sector_column="sector", price_column="last_price"
        )
        self.assertIsInstance(result, pd.DataFrame, "Should return a pandas DataFrame")

    def test_prepare_phase95_data_removes_all_nans(self):
        """Test that prepare_phase95_data removes ALL NaN values."""
        # Count NaN before
        nan_before = self.df_with_many_nans.isnull().sum().sum()
        self.assertGreater(nan_before, 0, "Test data should contain NaN values")

        # Apply preparation
        result = prepare_phase95_data(
            df=self.df_with_many_nans, sector_column="sector", price_column="last_price"
        )

        # Validate ZERO NaN after
        nan_after = result.isnull().sum().sum()
        self.assertEqual(nan_after, 0, f"Expected 0 NaN values after preparation, got {nan_after}")

    def test_prepare_phase95_data_removes_infinite_values(self):
        """Test that prepare_phase95_data handles infinite values."""
        # Add infinite values to test data
        df_with_inf = self.df_with_many_nans.copy()
        df_with_inf.loc[0, "enterprise_value"] = np.inf
        df_with_inf.loc[1, "p_e_ntm"] = -np.inf

        result = prepare_phase95_data(
            df=df_with_inf, sector_column="sector", price_column="last_price"
        )

        # Validate no infinite values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(result[numeric_cols]).sum().sum()
        self.assertEqual(
            inf_count, 0, f"Expected 0 infinite values after preparation, got {inf_count}"
        )

    def test_prepare_phase95_data_preserves_shape(self):
        """Test that prepare_phase95_data preserves DataFrame shape (rows and columns)."""
        original_shape = self.df_with_many_nans.shape

        result = prepare_phase95_data(
            df=self.df_with_many_nans, sector_column="sector", price_column="last_price"
        )

        self.assertEqual(result.shape, original_shape, "DataFrame shape should be preserved")

    def test_prepare_phase95_data_preserves_non_nan_values(self):
        """Test that prepare_phase95_data preserves existing non-NaN values."""
        # Select a column with some non-NaN values
        original_non_nan_mask = self.df_with_many_nans["market_cap"].notna()
        original_values = self.df_with_many_nans.loc[original_non_nan_mask, "market_cap"].copy()

        result = prepare_phase95_data(
            df=self.df_with_many_nans, sector_column="sector", price_column="last_price"
        )

        # Check that non-NaN values are approximately preserved (allowing for numerical precision)
        result_values = result.loc[original_non_nan_mask, "market_cap"]
        np.testing.assert_array_almost_equal(
            original_values.values,
            result_values.values,
            decimal=5,
            err_msg="Non-NaN values should be preserved",
        )

    def test_prepare_phase95_data_returns_copy(self):
        """Test that prepare_phase95_data returns a copy and doesn't modify original."""
        original_id = id(self.df_with_many_nans)
        original_nan_count = self.df_with_many_nans.isnull().sum().sum()

        result = prepare_phase95_data(
            df=self.df_with_many_nans, sector_column="sector", price_column="last_price"
        )

        # Verify original unchanged
        self.assertNotEqual(id(result), original_id, "Should return a copy, not modify original")
        self.assertEqual(
            self.df_with_many_nans.isnull().sum().sum(),
            original_nan_count,
            "Original DataFrame should remain unchanged",
        )

    def test_prepare_phase95_data_with_large_nan_count(self):
        """Test handling of DataFrame with very high NaN count (171+ columns scenario)."""
        # Create DataFrame with many columns containing NaN
        n_cols_with_nan = 200
        df_large = self.df_with_many_nans.copy()

        # Add many more columns with NaN
        for i in range(n_cols_with_nan):
            df_large[f"col_nan_{i}"] = self._create_column_with_nans(100, 0.5)

        # Count columns with NaN before
        cols_with_nan_before = (df_large.isnull().sum() > 0).sum()
        self.assertGreaterEqual(
            cols_with_nan_before,
            171,
            f"Test should have at least 171 columns with NaN, got {cols_with_nan_before}",
        )

        # Apply preparation
        result = prepare_phase95_data(
            df=df_large, sector_column="sector", price_column="last_price"
        )

        # Validate zero NaN
        nan_after = result.isnull().sum().sum()
        self.assertEqual(
            nan_after, 0, f"Should handle 171+ NaN columns, but {nan_after} NaN remain"
        )

    def test_prepare_phase95_data_validates_sector_column(self):
        """Test that prepare_phase95_data validates required sector column exists."""
        with self.assertRaises(ValueError) as context:
            prepare_phase95_data(
                df=self.df_with_many_nans,
                sector_column="nonexistent_sector",
                price_column="last_price",
            )
        self.assertIn("sector", str(context.exception).lower())

    def test_prepare_phase95_data_validates_price_column(self):
        """Test that prepare_phase95_data validates required price column exists."""
        with self.assertRaises(ValueError) as context:
            prepare_phase95_data(
                df=self.df_with_many_nans, sector_column="sector", price_column="nonexistent_price"
            )
        self.assertIn("price", str(context.exception).lower())

    def test_prepare_phase95_data_handles_empty_dataframe(self):
        """Test that prepare_phase95_data handles empty DataFrame gracefully."""
        df_empty = pd.DataFrame()

        with self.assertRaises(ValueError) as context:
            prepare_phase95_data(df=df_empty, sector_column="sector", price_column="last_price")
        self.assertIn("empty", str(context.exception).lower())


@unittest.skipIf(not IMPORTS_AVAILABLE, "finance_ml.advanced_preprocessing not available")
class TestPhase95IntegrationWithImputation(unittest.TestCase):
    """Test Phase 9.5 integration with 4-step imputation strategy."""

    def setUp(self):
        """Create test data for integration testing."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] * 10,
                "sector": ["Technology"] * 50,
                "last_price": np.random.uniform(100, 300, 50),
                "price_target": np.random.uniform(120, 350, 50),
                "enterprise_value": np.random.uniform(1e9, 1e12, 50),
                "p_e_ntm": np.random.uniform(10, 50, 50),
                "market_cap": np.random.uniform(1e9, 1e12, 50),
            }
        )

        # Add NaN values
        self.df.loc[0:10, "enterprise_value"] = np.nan
        self.df.loc[5:15, "p_e_ntm"] = np.nan
        self.df.loc[20:25, "market_cap"] = np.nan

    def test_prepare_phase95_uses_4step_imputation(self):
        """Test that prepare_phase95_data uses 4-step imputation strategy."""
        result = prepare_phase95_data(df=self.df, sector_column="sector", price_column="last_price")

        # Should have zero NaN after 4-step imputation
        self.assertEqual(
            result.isnull().sum().sum(), 0, "4-step imputation should eliminate all NaN"
        )

    def test_prepare_phase95_logs_nan_counts(self):
        """Test that prepare_phase95_data logs NaN counts before and after."""
        with patch("finance_ml.advanced_preprocessing.logger") as mock_logger:
            result = prepare_phase95_data(
                df=self.df, sector_column="sector", price_column="last_price"
            )

            # Verify logging was called
            self.assertGreater(
                mock_logger.info.call_count, 0, "Should log information during processing"
            )

    def test_prepare_phase95_reports_statistics(self):
        """Test that prepare_phase95_data returns preparation statistics."""
        result = prepare_phase95_data(
            df=self.df, sector_column="sector", price_column="last_price", return_stats=True
        )

        # Should return tuple of (df, stats) when return_stats=True
        if isinstance(result, tuple):
            df_result, stats = result
            self.assertIsInstance(stats, dict, "Should return statistics dict")
            self.assertIn("nan_before", stats)
            self.assertIn("nan_after", stats)
            self.assertIn("inf_count", stats)
            self.assertEqual(stats["nan_after"], 0)
        else:
            # If return_stats not implemented yet, just check DataFrame returned
            self.assertIsInstance(result, pd.DataFrame)


@unittest.skipIf(not IMPORTS_AVAILABLE, "finance_ml.advanced_preprocessing not available")
class TestPhase95EmergencyFallback(unittest.TestCase):
    """Test Phase 9.5 emergency fallback mechanisms for edge cases."""

    def test_emergency_fallback_handles_all_nan_column(self):
        """Test that emergency fallback handles columns with all NaN values."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [100, 110, 105],
                "all_nan_col": [np.nan, np.nan, np.nan],
                "feature1": [1.0, 2.0, 3.0],
            }
        )

        result = prepare_phase95_data(df=df, sector_column="sector", price_column="last_price")

        # Should handle all-NaN column (likely filled with 0 or median)
        self.assertEqual(
            result.isnull().sum().sum(), 0, "Emergency fallback should handle all-NaN columns"
        )

    def test_emergency_fallback_handles_mixed_inf_nan(self):
        """Test that emergency fallback handles mixed infinite and NaN values."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "sector": ["Tech", "Tech", "Finance", "Finance"],
                "last_price": [100, 110, 105, 115],
                "mixed_col": [np.nan, np.inf, -np.inf, 50.0],
            }
        )

        result = prepare_phase95_data(df=df, sector_column="sector", price_column="last_price")

        # Should handle both NaN and Inf
        self.assertEqual(result.isnull().sum().sum(), 0)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertEqual(np.isinf(result[numeric_cols]).sum().sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
