"""
Test suite for apply_median_imputation Int64 dtype fix.

This module implements strict TDD for fixing the TypeError that occurs when
apply_median_imputation tries to fill NaN values in nullable integer (Int64)
columns with float median values.

Issue: TypeError: Invalid value '2135.5' for dtype 'Int64'
Root Cause: Median of integer data with NaNs becomes float, incompatible with Int64 dtype

Test Strategy:
1. RED: Write failing tests that reproduce the bug and define expected behavior
2. GREEN: Implement minimal fix to make tests pass
3. REFACTOR: Ensure code quality and no regression

Coverage Target: ≥80% for apply_median_imputation function
"""

import unittest
import pandas as pd
import numpy as np
from finance_ml.ml_workflow.preprocessing.imputation import apply_median_imputation


class TestMedianImputationInt64Fix(unittest.TestCase):
    """Test cases for Int64 dtype handling in median imputation."""

    def test_int64_nullable_column_with_nan_causes_typeerror(self):
        """
        RED PHASE: Reproduce the bug - Int64 column with NaN should cause TypeError.

        This test documents the current broken behavior where trying to fill
        NaN values in an Int64 column with a float median raises TypeError.

        Expected to FAIL initially, then be fixed by casting to float64.
        """
        # Create DataFrame with Int64 nullable column
        df = pd.DataFrame(
            {
                "value": pd.array([1000, 2000, np.nan, 3000, 4000], dtype="Int64"),
                "name": ["A", "B", "C", "D", "E"],
            }
        )

        # Verify column is Int64 dtype
        self.assertEqual(str(df["value"].dtype), "Int64")

        # This should NOT raise TypeError after fix
        # Before fix: TypeError: Invalid value '2500.0' for dtype 'Int64'
        try:
            result = apply_median_imputation(df)
            # After fix, should succeed and fill NaN with median
            self.assertFalse(result["value"].isna().any(), "NaN values should be imputed")
            # Median of [1000, 2000, 3000, 4000] = 2500.0
            self.assertEqual(result.loc[2, "value"], 2500.0)
        except TypeError as e:
            if "Invalid value" in str(e) and "Int64" in str(e):
                self.fail(f"Int64 TypeError not fixed: {e}")
            else:
                raise

    def test_int64_column_cast_to_float64_before_imputation(self):
        """
        Test that Int64 columns are cast to float64 before median imputation.

        This is the core fix - integer dtypes should be converted to float64
        to avoid TypeError when filling with float median values.
        """
        df = pd.DataFrame(
            {
                "int64_col": pd.array([10, 20, np.nan, 40], dtype="Int64"),
                "regular_int": [100, 200, 300, 400],  # int64
            }
        )

        result = apply_median_imputation(df)

        # Both integer columns should be float64 after imputation
        self.assertEqual(result["int64_col"].dtype, np.float64, "Int64 should be cast to float64")
        self.assertEqual(
            result["regular_int"].dtype, np.float64, "Regular int should be cast to float64"
        )

        # Median of [10, 20, 40] = 20.0
        self.assertEqual(result.loc[2, "int64_col"], 20.0)

        # Regular int column had no NaN, so values unchanged (but dtype changed)
        self.assertEqual(result["regular_int"].iloc[0], 100.0)

    def test_column_with_all_nan_median_is_skipped(self):
        """
        Test that columns where median is NaN (all values missing) are skipped.

        When all values are NaN, median() returns NaN, and we should skip
        imputation rather than filling NaN with NaN.
        """
        df = pd.DataFrame(
            {"all_nan": [np.nan, np.nan, np.nan, np.nan], "has_values": [1.0, 2.0, np.nan, 4.0]}
        )

        result = apply_median_imputation(df)

        # all_nan column should still have NaN values (imputation skipped)
        self.assertTrue(result["all_nan"].isna().all(), "Column with all NaN should be skipped")

        # has_values column should be imputed
        self.assertFalse(result["has_values"].isna().any(), "Column with values should be imputed")
        self.assertEqual(result.loc[2, "has_values"], 2.0)  # median of [1,2,4] = 2

    def test_float_columns_still_work_correctly(self):
        """
        REGRESSION TEST: Verify existing float column behavior is unchanged.

        Float columns should continue to work as before - this ensures
        the fix doesn't break existing functionality.
        """
        df = pd.DataFrame(
            {
                "float_col": [1.5, 2.5, np.nan, 4.5, 5.5],
                "another_float": [10.0, np.nan, 30.0, 40.0, 50.0],
            }
        )

        result = apply_median_imputation(df)

        # Both should remain float64
        self.assertEqual(result["float_col"].dtype, np.float64)
        self.assertEqual(result["another_float"].dtype, np.float64)

        # Check imputation correctness
        # Median of [1.5, 2.5, 4.5, 5.5] = 3.5
        self.assertEqual(result.loc[2, "float_col"], 3.5)
        # Median of [10, 30, 40, 50] = 35.0
        self.assertEqual(result.loc[1, "another_float"], 35.0)

    def test_boolean_columns_not_affected(self):
        """
        Test that boolean columns are not treated as integers for imputation.

        Boolean columns should be excluded from integer casting logic even though
        they are technically integer-based (True=1, False=0).
        """
        df = pd.DataFrame(
            {
                "bool_col": [True, False, True, False],
                "int_col": pd.array([1, 2, np.nan, 4], dtype="Int64"),
            }
        )

        result = apply_median_imputation(df)

        # Boolean column should remain boolean (not cast to float)
        self.assertEqual(result["bool_col"].dtype, bool)

        # Int column should be cast to float and imputed
        self.assertEqual(result["int_col"].dtype, np.float64)
        self.assertFalse(result["int_col"].isna().any())

    def test_mixed_integer_types_all_handled(self):
        """
        Test that various integer dtypes are all handled correctly.

        Covers: int8, int16, int32, int64, Int8, Int16, Int32, Int64
        """
        df = pd.DataFrame(
            {
                "int8": pd.array([1, 2, np.nan, 4], dtype="Int8"),
                "int16": pd.array([10, 20, np.nan, 40], dtype="Int16"),
                "int32": pd.array([100, 200, np.nan, 400], dtype="Int32"),
                "int64": pd.array([1000, 2000, np.nan, 4000], dtype="Int64"),
            }
        )

        result = apply_median_imputation(df)

        # All should be cast to float64 after imputation
        for col in df.columns:
            self.assertEqual(result[col].dtype, np.float64, f"{col} should be cast to float64")
            self.assertFalse(
                result[col].isna().any(), f"{col} should have no NaN values after imputation"
            )

    def test_no_numeric_columns_returns_unchanged(self):
        """
        Test edge case: DataFrame with no numeric columns.

        Should return DataFrame unchanged with a warning (logged).
        """
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "category": ["A", "B", "C"]})

        result = apply_median_imputation(df)

        # Should return copy unchanged
        self.assertEqual(result.shape, df.shape)
        self.assertEqual(list(result.columns), list(df.columns))
        pd.testing.assert_frame_equal(result, df)

    def test_dataframe_not_modified_inplace(self):
        """
        Test that original DataFrame is not modified (immutability).

        apply_median_imputation should return a copy, not modify in place.
        """
        df = pd.DataFrame({"value": pd.array([1, 2, np.nan, 4], dtype="Int64")})

        # Store original dtype and NaN count
        original_dtype = df["value"].dtype
        original_nan_count = df["value"].isna().sum()

        result = apply_median_imputation(df)

        # Original should be unchanged
        self.assertEqual(str(df["value"].dtype), str(original_dtype))
        self.assertEqual(df["value"].isna().sum(), original_nan_count)

        # Result should be different
        self.assertEqual(result["value"].dtype, np.float64)
        self.assertEqual(result["value"].isna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
