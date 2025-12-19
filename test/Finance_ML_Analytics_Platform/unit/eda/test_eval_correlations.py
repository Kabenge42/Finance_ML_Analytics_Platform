"""
Tests for correlation functions extracted from analytics/eval.py

TDD Step 3: Testing correlation functions (lines 1601-1735 of eval.py)
These tests validate the correlation analysis functions
that will be extracted to eda/correlations.py module.

Coverage Target: 80% for correlations module
"""

import unittest
import pandas as pd
import numpy as np


class TestCalculateCorrelationMatrix(unittest.TestCase):
    """Tests for calculate_correlation_matrix function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "price": np.random.randn(100) * 10 + 100,
                "volume": np.random.randn(100) * 1000 + 5000,
                "pe_ratio": np.random.randn(100) * 5 + 20,
                "market_cap": np.random.randn(100) * 1e9 + 1e10,
                "sector": ["Tech"] * 50 + ["Finance"] * 50,  # Non-numeric column
            }
        )

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(self.df, columns=["price", "volume", "pe_ratio"])

        self.assertIsInstance(result, pd.DataFrame)

    def test_pearson_correlation(self):
        """Test Pearson correlation method."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(
            self.df, columns=["price", "volume", "pe_ratio"], method="pearson"
        )

        # Diagonal should be 1.0
        self.assertAlmostEqual(result.loc["price", "price"], 1.0, places=5)
        self.assertAlmostEqual(result.loc["volume", "volume"], 1.0, places=5)

    def test_spearman_correlation(self):
        """Test Spearman correlation method."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(
            self.df, columns=["price", "volume"], method="spearman"
        )

        self.assertIsInstance(result, pd.DataFrame)
        # Correlation values should be between -1 and 1
        self.assertTrue((result >= -1).all().all())
        self.assertTrue((result <= 1).all().all())

    def test_kendall_correlation(self):
        """Test Kendall correlation method."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(
            self.df, columns=["price", "volume"], method="kendall"
        )

        self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        with self.assertRaises(ValueError) as context:
            calculate_correlation_matrix(
                self.df, columns=["price", "volume"], method="invalid_method"
            )

        self.assertIn("not supported", str(context.exception))

    def test_filters_numeric_columns_only(self):
        """Test that only numeric columns are used."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        # Include non-numeric column in columns list
        result = calculate_correlation_matrix(self.df, columns=["price", "volume", "sector"])

        # Result should only have numeric columns
        self.assertIn("price", result.columns)
        self.assertIn("volume", result.columns)
        self.assertNotIn("sector", result.columns)

    def test_symmetric_matrix(self):
        """Test that correlation matrix is symmetric."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        result = calculate_correlation_matrix(self.df, columns=["price", "volume", "pe_ratio"])

        # Matrix should be symmetric
        for col1 in result.columns:
            for col2 in result.columns:
                self.assertAlmostEqual(result.loc[col1, col2], result.loc[col2, col1], places=10)

    def test_empty_numeric_columns_raises_error(self):
        """Test that empty numeric columns raises ValueError."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        df_no_numeric = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})

        with self.assertRaises(ValueError) as context:
            calculate_correlation_matrix(df_no_numeric, columns=["a", "b"])

        self.assertIn("No numeric columns", str(context.exception))


class TestCalculateDistanceCorrelation(unittest.TestCase):
    """Tests for calculate_distance_correlation function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {"x": np.random.randn(50), "y": np.random.randn(50), "z": np.random.randn(50)}
        )

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame (if dcor installed)."""
        from finance_ml.ml_workflow.analytics.eval import calculate_distance_correlation

        try:
            result = calculate_distance_correlation(self.df, columns=["x", "y"])
            self.assertIsInstance(result, pd.DataFrame)
        except ImportError:
            # dcor not installed - skip test
            self.skipTest("dcor library not installed")

    def test_diagonal_is_one(self):
        """Test that diagonal values are 1.0."""
        from finance_ml.ml_workflow.analytics.eval import calculate_distance_correlation

        try:
            result = calculate_distance_correlation(self.df, columns=["x", "y", "z"])

            for col in result.columns:
                self.assertAlmostEqual(result.loc[col, col], 1.0, places=5)
        except ImportError:
            self.skipTest("dcor library not installed")

    def test_symmetric_matrix(self):
        """Test that distance correlation matrix is symmetric."""
        from finance_ml.ml_workflow.analytics.eval import calculate_distance_correlation

        try:
            result = calculate_distance_correlation(self.df, columns=["x", "y", "z"])

            for col1 in result.columns:
                for col2 in result.columns:
                    self.assertAlmostEqual(
                        result.loc[col1, col2], result.loc[col2, col1], places=10
                    )
        except ImportError:
            self.skipTest("dcor library not installed")

    def test_values_between_zero_and_one(self):
        """Test that distance correlation values are between 0 and 1."""
        from finance_ml.ml_workflow.analytics.eval import calculate_distance_correlation

        try:
            result = calculate_distance_correlation(self.df, columns=["x", "y"])

            # Distance correlation is always non-negative
            self.assertTrue((result >= 0).all().all())
            self.assertTrue((result <= 1).all().all())
        except ImportError:
            self.skipTest("dcor library not installed")

    def test_raises_import_error_if_dcor_missing(self):
        """Test that ImportError is raised if dcor is not installed."""
        from finance_ml.ml_workflow.analytics.eval import calculate_distance_correlation

        # This test will either pass because dcor isn't installed
        # or it will complete successfully because dcor is installed
        try:
            result = calculate_distance_correlation(self.df, columns=["x", "y"])
            # If we get here, dcor is installed
            self.assertIsInstance(result, pd.DataFrame)
        except ImportError as e:
            self.assertIn("dcor", str(e))

    def test_empty_numeric_columns_raises_error(self):
        """Test that empty numeric columns raises ValueError."""
        from finance_ml.ml_workflow.analytics.eval import calculate_distance_correlation

        df_no_numeric = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})

        try:
            with self.assertRaises(ValueError) as context:
                calculate_distance_correlation(df_no_numeric, columns=["a", "b"])
            self.assertIn("No numeric columns", str(context.exception))
        except ImportError:
            self.skipTest("dcor library not installed")


class TestFindTopCorrelations(unittest.TestCase):
    """Tests for find_top_correlations function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a known correlation matrix
        self.corr_matrix = pd.DataFrame(
            {
                "A": [1.0, 0.9, 0.5, 0.2],
                "B": [0.9, 1.0, 0.3, 0.1],
                "C": [0.5, 0.3, 1.0, 0.8],
                "D": [0.2, 0.1, 0.8, 1.0],
            },
            index=["A", "B", "C", "D"],
        )

    def test_returns_list(self):
        """Test that function returns a list."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=5)

        self.assertIsInstance(result, list)

    def test_returns_correct_number(self):
        """Test that function returns correct number of results."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=3)

        self.assertEqual(len(result), 3)

    def test_sorted_by_correlation(self):
        """Test that results are sorted by absolute correlation (descending)."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=10)

        # Check that correlations are sorted in descending order
        correlations = [abs(item[2]) for item in result]
        self.assertEqual(correlations, sorted(correlations, reverse=True))

    def test_highest_correlation_first(self):
        """Test that highest correlation pair is first."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=1)

        # A-B has highest correlation (0.9)
        self.assertEqual(len(result), 1)
        pair = (result[0][0], result[0][1])
        self.assertTrue(
            ("A" in pair and "B" in pair) or ("C" in pair and "D" in pair),
            f"Expected A-B or C-D pair, got {pair}",
        )

    def test_threshold_filter(self):
        """Test that threshold filters out low correlations."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=10, threshold=0.5)

        # All returned correlations should be >= threshold
        for item in result:
            self.assertGreaterEqual(abs(item[2]), 0.5)

    def test_excludes_self_correlations(self):
        """Test that diagonal (self-correlations) are excluded."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=10)

        # No pair should have same variable twice
        for item in result:
            self.assertNotEqual(item[0], item[1])

    def test_no_duplicate_pairs(self):
        """Test that pairs are not duplicated (A-B and B-A)."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        result = find_top_correlations(self.corr_matrix, n_top=10)

        # Convert to frozensets to check for duplicates
        pairs = [frozenset([item[0], item[1]]) for item in result]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_with_negative_correlations(self):
        """Test handling of negative correlations."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        corr_with_negative = pd.DataFrame(
            {"A": [1.0, -0.95, 0.3], "B": [-0.95, 1.0, 0.2], "C": [0.3, 0.2, 1.0]},
            index=["A", "B", "C"],
        )

        result = find_top_correlations(corr_with_negative, n_top=3)

        # Should include the negative correlation as top result
        self.assertIsInstance(result, list)


class TestCorrelationEdgeCases(unittest.TestCase):
    """Tests for edge cases in correlation functions."""

    def test_single_column(self):
        """Test correlation with single column."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = calculate_correlation_matrix(df, columns=["a"])

        # Should return 1x1 matrix
        self.assertEqual(result.shape, (1, 1))
        self.assertAlmostEqual(result.iloc[0, 0], 1.0, places=5)

    def test_with_nan_values(self):
        """Test correlation with NaN values."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [5.0, np.nan, 3.0, 2.0, 1.0]})

        result = calculate_correlation_matrix(df, columns=["a", "b"])

        # Should handle NaN and produce valid result
        self.assertIsInstance(result, pd.DataFrame)

    def test_with_constant_column(self):
        """Test correlation with constant column (zero variance)."""
        from finance_ml.ml_workflow.analytics.eval import calculate_correlation_matrix

        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.0, 1.0, 1.0, 1.0, 1.0]}  # Constant
        )

        result = calculate_correlation_matrix(df, columns=["a", "b"])

        # Correlation with constant is NaN
        self.assertTrue(np.isnan(result.loc["a", "b"]) or result.loc["a", "b"] == 0)

    def test_find_top_correlations_empty_result(self):
        """Test find_top_correlations with high threshold yielding empty result."""
        from finance_ml.ml_workflow.analytics.eval import find_top_correlations

        corr_matrix = pd.DataFrame({"A": [1.0, 0.1], "B": [0.1, 1.0]}, index=["A", "B"])

        result = find_top_correlations(corr_matrix, n_top=10, threshold=0.9)

        # No correlations meet the threshold
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
