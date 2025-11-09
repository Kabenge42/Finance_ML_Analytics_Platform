"""
Test suite for finance_ml.ml_workflow.preprocessing.imputation module.

This test module covers the 4-step imputation strategy:
1. Zero imputation for metrics that can be zero
2. Price-based imputation for price-derived metrics
3. KNN imputation (sector-aware) for complex relationships
4. Median imputation (sector-aware) as final fallback

Tests are written following TDD principles - these tests will fail until
imputation.py is implemented by extracting from advanced_preprocessing.py.
"""

import unittest
import numpy as np
import pandas as pd
from typing import List


class TestImputationFunctions(unittest.TestCase):
    """Test individual imputation functions."""

    def setUp(self):
        """Create sample data with missing values for testing."""
        self.sample_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
                "sector": ["Technology", "Technology", "Technology", "Automotive", "Technology"],
                "last_price": [150.0, 300.0, 2800.0, 700.0, 3300.0],
                "market_cap": [2.5e12, 2.3e12, 1.8e12, np.nan, 1.7e12],
                "p_e": [28.5, 35.2, np.nan, 95.3, 70.1],
                "dividend_yield": [0.0, np.nan, 0.0, 0.0, np.nan],
                "debt_to_equity": [1.5, np.nan, 0.8, 2.1, np.nan],
                "roe": [0.15, 0.20, np.nan, 0.05, 0.18],
            }
        )

    def test_import_imputation_module(self):
        """Test that imputation module can be imported."""
        try:
            from finance_ml.ml_workflow.preprocessing import imputation

            self.assertIsNotNone(imputation)
        except ImportError as e:
            self.fail(f"Failed to import imputation module: {e}")

    def test_apply_zero_imputation(self):
        """Test zero imputation for columns that can naturally be zero."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_zero_imputation

        df = self.sample_data.copy()
        result = apply_zero_imputation(df, columns=["dividend_yield"])

        # Check that NaN values in dividend_yield are filled with 0
        self.assertEqual(result["dividend_yield"].isna().sum(), 0)
        self.assertEqual(result.loc[1, "dividend_yield"], 0.0)
        self.assertEqual(result.loc[4, "dividend_yield"], 0.0)

    def test_apply_price_imputation(self):
        """Test price-based imputation for price-derived metrics."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_price_imputation

        df = self.sample_data.copy()
        # Add a price-derived column with missing values
        df["price_to_book"] = [3.5, np.nan, 4.2, np.nan, 5.1]

        result = apply_price_imputation(df, price_column="last_price", columns=["price_to_book"])

        # Check that missing values are imputed (should not be NaN)
        self.assertFalse(result["price_to_book"].isna().any())

    def test_apply_knn_imputation_enhanced(self):
        """Test KNN imputation with sector awareness."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_knn_imputation_enhanced

        df = self.sample_data.copy()
        result = apply_knn_imputation_enhanced(
            df, columns=["p_e", "roe"], sector_column="sector", n_neighbors=2
        )

        # Check that missing values are reduced
        self.assertLessEqual(result["p_e"].isna().sum(), df["p_e"].isna().sum())
        self.assertLessEqual(result["roe"].isna().sum(), df["roe"].isna().sum())

    def test_apply_median_imputation(self):
        """Test median imputation as final fallback."""
        from finance_ml.ml_workflow.preprocessing.imputation import apply_median_imputation

        df = self.sample_data.copy()
        result = apply_median_imputation(df)

        # Check that all missing values are filled
        self.assertEqual(result.select_dtypes(include=[np.number]).isna().sum().sum(), 0)


class TestEnhancedImputation4Step(unittest.TestCase):
    """Test the complete 4-step imputation strategy."""

    def setUp(self):
        """Create realistic financial data with various missing patterns."""
        np.random.seed(42)
        n = 100

        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]

        self.data = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(sectors, n),
                "last_price": np.random.uniform(10, 500, n),
                "market_cap": np.random.uniform(1e9, 1e12, n),
                "p_e": np.random.uniform(5, 100, n),
                "p_b": np.random.uniform(0.5, 10, n),
                "dividend_yield": np.random.uniform(0, 0.05, n),
                "debt_to_equity": np.random.uniform(0, 3, n),
                "roe": np.random.uniform(0.01, 0.30, n),
                "operating_margin": np.random.uniform(0.05, 0.40, n),
            }
        )

        # Introduce missing values in various columns
        for col in ["p_e", "p_b", "dividend_yield", "debt_to_equity", "roe"]:
            missing_mask = np.random.random(n) < 0.15
            self.data.loc[missing_mask, col] = np.nan

    def test_apply_enhanced_imputation_strategy_4step(self):
        """Test the complete 4-step imputation strategy."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_4step,
        )

        df = self.data.copy()
        missing_before = df.isna().sum().sum()

        result = apply_enhanced_imputation_strategy_4step(
            df, sector_column="sector", n_neighbors=5, price_column="last_price"
        )

        missing_after = result.isna().sum().sum()

        # Check that missing values are reduced or eliminated
        self.assertLessEqual(missing_after, missing_before)

        # Ideally, all missing values should be filled
        self.assertEqual(
            result.select_dtypes(include=[np.number]).isna().sum().sum(),
            0,
            "4-step imputation should fill all missing values",
        )

    def test_4step_preserves_existing_values(self):
        """Test that 4-step imputation doesn't modify non-missing values."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_4step,
        )

        df = self.data.copy()
        # Get mask of non-missing values
        non_missing_mask = ~df.isna()
        original_values = df.copy()

        result = apply_enhanced_imputation_strategy_4step(
            df, sector_column="sector", n_neighbors=5, price_column="last_price"
        )

        # Check that non-missing values are preserved
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in original_values.columns:
                original_non_missing = original_values.loc[non_missing_mask[col], col]
                result_non_missing = result.loc[non_missing_mask[col], col]

                if len(original_non_missing) > 0:
                    # Allow small floating point differences
                    np.testing.assert_array_almost_equal(
                        original_non_missing.values,
                        result_non_missing.values,
                        decimal=5,
                        err_msg=f"Non-missing values changed in column {col}",
                    )


class TestImputationHelpers(unittest.TestCase):
    """Test helper functions for imputation."""

    def test_get_zero_imputation_columns(self):
        """Test retrieval of columns suitable for zero imputation."""
        from finance_ml.ml_workflow.preprocessing.imputation import get_zero_imputation_columns

        columns = get_zero_imputation_columns()

        # Should return a list
        self.assertIsInstance(columns, list)

        # Should contain dividend-related columns
        self.assertTrue(any("dividend" in col.lower() for col in columns))

    def test_get_knn_imputation_columns(self):
        """Test retrieval of columns suitable for KNN imputation."""
        from finance_ml.ml_workflow.preprocessing.imputation import get_knn_imputation_columns

        columns = get_knn_imputation_columns()

        # Should return a list
        self.assertIsInstance(columns, list)

        # Should contain financial ratio columns
        self.assertTrue(len(columns) > 0)


class TestImputationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test imputation on empty dataframe."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_4step,
        )

        df = pd.DataFrame()
        result = apply_enhanced_imputation_strategy_4step(df)

        # Should return empty dataframe without error
        self.assertEqual(len(result), 0)

    def test_no_missing_values(self):
        """Test imputation when there are no missing values."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_4step,
        )

        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Health"],
                "last_price": [100, 200, 300],
                "p_e": [20, 25, 30],
            }
        )

        result = apply_enhanced_imputation_strategy_4step(df, sector_column="sector")

        # Should return same data without modifications
        pd.testing.assert_frame_equal(result, df)

    def test_all_missing_in_column(self):
        """Test imputation when entire column is missing."""
        from finance_ml.ml_workflow.preprocessing.imputation import (
            apply_enhanced_imputation_strategy_4step,
        )

        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Health"],
                "last_price": [100, 200, 300],
                "p_e": [np.nan, np.nan, np.nan],
            }
        )

        result = apply_enhanced_imputation_strategy_4step(df, sector_column="sector")

        # Should handle gracefully (may fill with 0 or skip)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
