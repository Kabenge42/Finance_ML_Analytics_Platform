"""
Test suite for simple_eda StringDtype handling.

This module tests that simple_eda properly handles pandas StringDtype
('string[python]') without raising dtype interpretation errors.

Context: The original implementation used np.issubdtype() which fails
on StringDtype with "Cannot interpret 'string[python]' as a data type".
The fix uses pd.api.types.is_numeric_dtype() which properly handles
all pandas dtype variants.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import warnings


class TestSimpleEdaStringDtype(unittest.TestCase):
    """Test simple_eda StringDtype compatibility."""

    def setUp(self):
        """Create sample data with StringDtype columns."""
        np.random.seed(42)
        # Create dataframe with explicit StringDtype
        self.df_with_stringdtype = pd.DataFrame(
            {
                "ticker": pd.array(["AAPL", "MSFT", "GOOGL", "AMZN", "META"] * 20, dtype="string"),
                "sector": pd.array(["Technology"] * 50 + ["Consumer"] * 50, dtype="string"),
                "region": pd.array(["US"] * 100, dtype="string"),
                "last_price": np.random.uniform(50, 500, 100),
                "market_cap": np.random.uniform(1e9, 1e12, 100),
                "pe_ratio": np.random.uniform(10, 50, 100),
            }
        )

        # Verify StringDtype is actually applied
        self.assertEqual(str(self.df_with_stringdtype["ticker"].dtype), "string")
        self.assertEqual(str(self.df_with_stringdtype["sector"].dtype), "string")

    def test_simple_eda_handles_stringdtype_without_error(self):
        """Test that simple_eda handles StringDtype columns without raising errors.

        This test validates the fix for:
        WARNING: simple_eda: dtype inspection failed: Cannot interpret 'string[python]' as a data type

        Arrange: DataFrame with explicit StringDtype columns.
        Act: Call simple_eda() on the DataFrame.
        Assert:
            - No exceptions raised
            - Function returns a dict
            - Dict contains expected keys
            - Numeric columns are correctly identified (excludes StringDtype columns)
        """
        try:
            from finance_ml.ml_workflow.analytics.eval import simple_eda
        except ImportError:
            self.skipTest("simple_eda not available")

        # Act: Call simple_eda with StringDtype columns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simple_eda(self.df_with_stringdtype)

            # Check for the specific warning we're trying to fix
            dtype_warnings = [
                warning
                for warning in w
                if "Cannot interpret 'string[python]'" in str(warning.message)
            ]
            self.assertEqual(
                len(dtype_warnings),
                0,
                f"Should not have StringDtype interpretation warnings, got: {dtype_warnings}",
            )

        # Assert: Result is valid
        self.assertIsInstance(result, dict, "simple_eda should return a dict")

        # Assert: Expected keys are present
        expected_keys = ["row_count", "column_count", "numeric_columns", "categorical_columns"]
        for key in expected_keys:
            self.assertIn(key, result, f"Result should contain '{key}' key")

        # Assert: Numeric columns correctly identified (StringDtype should be excluded)
        numeric_cols = result.get("numeric_columns", [])
        self.assertIsInstance(numeric_cols, list)
        self.assertIn("last_price", numeric_cols, "last_price should be identified as numeric")
        self.assertIn("market_cap", numeric_cols, "market_cap should be identified as numeric")
        self.assertNotIn(
            "ticker", numeric_cols, "ticker (StringDtype) should NOT be identified as numeric"
        )
        self.assertNotIn(
            "sector", numeric_cols, "sector (StringDtype) should NOT be identified as numeric"
        )

    def test_simple_eda_categorical_count_includes_stringdtype(self):
        """Test that simple_eda counts StringDtype columns as categorical.

        Arrange: DataFrame with 3 StringDtype columns and 3 numeric columns.
        Act: Call simple_eda().
        Assert:
            - categorical_cols_count == 3
            - numeric_cols_count == 3
        """
        try:
            from finance_ml.ml_workflow.analytics.eval import simple_eda
        except ImportError:
            self.skipTest("simple_eda not available")

        # Act
        result = simple_eda(self.df_with_stringdtype)

        # Assert: Counts are correct
        numeric_count = result.get("numeric_cols_count", 0)
        categorical_count = result.get("categorical_cols_count", 0)

        self.assertEqual(numeric_count, 3, f"Should have 3 numeric columns, got {numeric_count}")
        self.assertEqual(
            categorical_count,
            3,
            f"Should have 3 categorical columns (StringDtype), got {categorical_count}",
        )

    def test_simple_eda_with_mixed_dtypes(self):
        """Test simple_eda with mix of StringDtype, object, category, and numeric dtypes.

        This tests robustness across all pandas dtype variants.

        Arrange: DataFrame with StringDtype, object, category, and numeric columns.
        Act: Call simple_eda().
        Assert:
            - No errors raised
            - Numeric columns correctly identified
            - Non-numeric columns correctly classified
        """
        try:
            from finance_ml.ml_workflow.analytics.eval import simple_eda
        except ImportError:
            self.skipTest("simple_eda not available")

        # Arrange: Mix of dtypes
        df_mixed = pd.DataFrame(
            {
                "string_col": pd.array(["A", "B", "C"] * 10, dtype="string"),
                "object_col": ["X", "Y", "Z"] * 10,
                "category_col": pd.Categorical(["cat1", "cat2", "cat3"] * 10),
                "int_col": range(30),
                "float_col": np.random.uniform(0, 100, 30),
            }
        )

        # Act
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simple_eda(df_mixed)

            # No dtype interpretation warnings
            dtype_warnings = [
                warning
                for warning in w
                if "Cannot interpret" in str(warning.message)
                or "dtype inspection failed" in str(warning.message)
            ]
            self.assertEqual(
                len(dtype_warnings), 0, f"Should not have dtype warnings, got: {dtype_warnings}"
            )

        # Assert: Numeric columns correctly identified
        numeric_cols = result.get("numeric_columns", [])
        self.assertIn("int_col", numeric_cols)
        self.assertIn("float_col", numeric_cols)
        self.assertNotIn("string_col", numeric_cols)
        self.assertNotIn("object_col", numeric_cols)
        self.assertNotIn("category_col", numeric_cols)


if __name__ == "__main__":
    unittest.main()
