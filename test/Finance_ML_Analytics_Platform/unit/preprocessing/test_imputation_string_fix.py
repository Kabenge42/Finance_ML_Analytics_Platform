"""
Test to validate the KNN imputation string coercion fix.

This test verifies that:
1. String values in numeric columns are coerced to NaN before KNN imputation
2. KNN imputation completes without "could not convert string to float" errors
3. All numeric columns remain numeric after imputation
"""

import numpy as np
import pandas as pd
import pytest
from finance_ml.ml_workflow.preprocessing.imputation import impute_missing_values_knn_sector


def test_knn_imputation_with_string_contamination():
    """Test that KNN imputation handles string-contaminated numeric columns."""

    # Create test data with string contamination (whitespace strings)
    df = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
            "revenue": [100.0, 200.0, "                 ", 150.0, 180.0, "                  "],
            "profit": [10.0, np.nan, 30.0, 15.0, np.nan, 25.0],
            "market_cap": [1000.0, 2000.0, 3000.0, 1500.0, 1800.0, 2200.0],
        }
    )

    # Columns to impute (including the string-contaminated 'revenue')
    columns = ["revenue", "profit", "market_cap"]

    # Run imputation with sector awareness
    result = impute_missing_values_knn_sector(
        df, columns=columns, sector_column="sector", n_neighbors=2
    )

    # Validation 1: No string dtypes remain in result
    for col in columns:
        assert result[col].dtype in [
            np.float64,
            np.int64,
            np.float32,
            np.int32,
        ], f"Column '{col}' has non-numeric dtype: {result[col].dtype}"

    # Validation 2: String values were coerced to numeric (NaN or imputed)
    # Original string values should now be numeric
    assert pd.api.types.is_numeric_dtype(
        result["revenue"]
    ), "Revenue column should be numeric after coercion"

    # Validation 3: No NaN values remain after imputation
    # (6-step strategy should fill all missing values)
    assert result[columns].isna().sum().sum() == 0, "KNN imputation should fill all missing values"

    # Validation 4: Imputed values are reasonable (within data range)
    for col in columns:
        assert result[col].min() >= 0, f"Column '{col}' has negative values after imputation"
        # Compare against numeric version of original data (coerce strings to NaN)
        original_numeric = pd.to_numeric(df[col], errors="coerce")
        max_original = original_numeric.max()
        if pd.notna(max_original):
            assert (
                result[col].max() <= max_original * 2
            ), f"Column '{col}' has unreasonably large values after imputation"

    print("[PASS] All validations passed!")
    print(f"   Result dtypes: {result[columns].dtypes.to_dict()}")
    print(f"   Missing values: {result[columns].isna().sum().sum()}")


def test_knn_imputation_without_string_contamination():
    """Test that normal numeric data still works correctly (regression test)."""

    df = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
            "revenue": [100.0, 200.0, np.nan, 150.0, 180.0, np.nan],
            "profit": [10.0, np.nan, 30.0, 15.0, np.nan, 25.0],
        }
    )

    columns = ["revenue", "profit"]

    result = impute_missing_values_knn_sector(
        df, columns=columns, sector_column="sector", n_neighbors=2
    )

    # Should impute all missing values
    assert result[columns].isna().sum().sum() == 0, "KNN imputation should fill all missing values"

    # Should preserve numeric dtypes
    for col in columns:
        assert pd.api.types.is_numeric_dtype(result[col]), f"Column '{col}' should remain numeric"

    print("[PASS] Regression test passed (normal numeric data)")


if __name__ == "__main__":
    # Run tests
    print("Testing KNN imputation with string contamination fix...")
    print("=" * 60)

    test_knn_imputation_with_string_contamination()
    print()
    test_knn_imputation_without_string_contamination()

    print("\n" + "=" * 60)
    print("All tests passed! Fix is working correctly.")
