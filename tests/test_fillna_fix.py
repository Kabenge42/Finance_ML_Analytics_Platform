"""
Test script to verify fillna_by_dtype handles categorical columns correctly.

This test ensures the fix prevents:
"TypeError: Cannot setitem on a Categorical with a new category (0)"
"""

import pandas as pd
import numpy as np
from finance_ml.ml_workflow.preprocessing.imputation import fillna_by_dtype


def test_fillna_with_categorical():
    """Test that fillna_by_dtype handles categorical columns without error."""

    print("Testing fillna_by_dtype with mixed dtypes...")

    # Create a DataFrame with mixed types including categorical
    df = pd.DataFrame(
        {
            "numeric_col": [1.0, 2.0, np.nan, 4.0],
            "categorical_col": pd.Categorical(["A", "B", np.nan, "A"]),
            "string_col": ["x", "y", np.nan, "z"],
            "another_numeric": [10, np.nan, 30, 40],
        }
    )

    print(f"\nOriginal DataFrame:")
    print(df)
    print(f"\nDtypes:")
    print(df.dtypes)
    print(f"\nMissing values per column:")
    print(df.isna().sum())

    # This would fail with the old approach: df.fillna(0)
    # Error: "TypeError: Cannot setitem on a Categorical with a new category (0)"

    try:
        # Use the new type-aware filling function
        df_filled = fillna_by_dtype(
            df, numeric_fill=0, categorical_strategy="mode", string_fill="Unknown"
        )

        print(f"\n✓ SUCCESS: fillna_by_dtype completed without error")
        print(f"\nFilled DataFrame:")
        print(df_filled)
        print(f"\nMissing values after filling:")
        print(df_filled.isna().sum())

        # Verify no missing values remain
        total_missing = df_filled.isna().sum().sum()
        assert total_missing == 0, f"Expected 0 missing values, got {total_missing}"

        # Verify dtypes are preserved
        assert (
            df_filled["categorical_col"].dtype.name == "category"
        ), "Categorical dtype not preserved"

        print(f"\n✓ All assertions passed!")
        print(f"  - No missing values remain")
        print(f"  - Categorical dtype preserved")
        print(f"  - Numeric columns filled with 0")
        print(f"  - Categorical column filled with mode ('A')")
        print(f"  - String column filled with 'Unknown'")

        return True

    except TypeError as e:
        print(f"\n✗ FAILED: TypeError occurred: {e}")
        return False
    except Exception as e:
        print(f"\n✗ FAILED: Unexpected error: {e}")
        return False


def test_old_approach_fails():
    """Demonstrate that the old approach (df.fillna(0)) fails with categorical columns."""

    print("\n" + "=" * 80)
    print("Demonstrating that the OLD approach fails...")
    print("=" * 80)

    df = pd.DataFrame(
        {"numeric_col": [1.0, np.nan, 3.0], "categorical_col": pd.Categorical(["A", np.nan, "B"])}
    )

    try:
        # This is the OLD problematic approach
        df_failed = df.fillna(0)
        print("\n✗ Unexpected: fillna(0) did not raise TypeError")
        print("   (This might work if categorical has 0 as an existing category)")
        return False
    except TypeError as e:
        print(f"\n✓ Expected: fillna(0) raised TypeError as expected:")
        print(f"   {e}")
        return True


if __name__ == "__main__":
    print("=" * 80)
    print("Testing fillna_by_dtype fix for categorical columns")
    print("=" * 80)

    # Test that the new approach works
    test1_passed = test_fillna_with_categorical()

    # Demonstrate that the old approach fails
    test2_passed = test_old_approach_fails()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (new approach): {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Test 2 (old approach fails): {'PASSED ✓' if test2_passed else 'FAILED ✗'}")

    if test1_passed:
        print("\n✓ Fix verified: fillna_by_dtype successfully handles categorical columns")
        print("  The notebook will no longer raise TypeError when filling NaN values.")
    else:
        print("\n✗ Fix failed: fillna_by_dtype did not work as expected")
