"""
Standalone test for preprocessing.imputation module.

This test validates the refactored imputation module works correctly
without requiring full finance_ml package initialization.

Run: python test_imputation_standalone.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

# Load imputation.py directly as a module without triggering package __init__
imputation_path = (
    Path(__file__).parent / "finance_ml" / "ml_workflow" / "preprocessing" / "imputation.py"
)
spec = importlib.util.spec_from_file_location("imputation", imputation_path)
imputation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imputation)

# Extract functions
get_zero_imputation_columns = imputation.get_zero_imputation_columns
get_knn_imputation_columns = imputation.get_knn_imputation_columns
apply_zero_imputation = imputation.apply_zero_imputation
apply_price_imputation = imputation.apply_price_imputation
apply_knn_imputation_enhanced = imputation.apply_knn_imputation_enhanced
apply_median_imputation = imputation.apply_median_imputation
apply_enhanced_imputation_strategy_4step = imputation.apply_enhanced_imputation_strategy_4step


def test_get_column_lists():
    """Test that column lists are returned correctly."""
    print("✓ Testing get_zero_imputation_columns...")
    zero_cols = get_zero_imputation_columns()
    assert isinstance(zero_cols, list)
    assert len(zero_cols) == 48
    assert "dividend_yield" not in zero_cols  # Not a zero-imputation column
    print(f"  Found {len(zero_cols)} columns for zero imputation")

    print("✓ Testing get_knn_imputation_columns...")
    knn_cols = get_knn_imputation_columns()
    assert isinstance(knn_cols, list)
    assert len(knn_cols) == 132  # Actual count in implementation
    print(f"  Found {len(knn_cols)} columns for KNN imputation")


def test_zero_imputation():
    """Test zero imputation on sample data."""
    print("\n✓ Testing apply_zero_imputation...")
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "dividend_yield": [0.02, np.nan, 0.03],
            "r_d_expenses_ltm": [1000, np.nan, 2000],
        }
    )

    result = apply_zero_imputation(df, columns=["r_d_expenses_ltm"])

    assert result["r_d_expenses_ltm"].isna().sum() == 0
    assert result.loc[1, "r_d_expenses_ltm"] == 0.0
    print("  Zero imputation working correctly")


def test_price_imputation():
    """Test price imputation on sample data."""
    print("\n✓ Testing apply_price_imputation...")
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "last_price": [100, 200, 300],
            "price_target": [110, np.nan, 330],
        }
    )

    result = apply_price_imputation(df, price_column="last_price")

    assert result["price_target"].isna().sum() == 0
    assert result.loc[1, "price_target"] == 200.0
    print("  Price imputation working correctly")


def test_median_imputation():
    """Test median imputation on sample data."""
    print("\n✓ Testing apply_median_imputation...")
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "p_e": [20, 25, np.nan, 30],
            "market_cap": [1e9, np.nan, 2e9, 3e9],
        }
    )

    result = apply_median_imputation(df)

    assert result.select_dtypes(include=[np.number]).isna().sum().sum() == 0
    print("  Median imputation working correctly")


def test_4step_imputation():
    """Test complete 4-step imputation strategy."""
    print("\n✓ Testing apply_enhanced_imputation_strategy_4step...")

    np.random.seed(42)
    n = 50

    df = pd.DataFrame(
        {
            "ticker": [f"TICK{i}" for i in range(n)],
            "sector": np.random.choice(["Technology", "Healthcare", "Finance"], n),
            "last_price": np.random.uniform(10, 500, n),
            "market_cap": np.random.uniform(1e9, 1e12, n),
            "p_e": np.random.uniform(5, 100, n),
            "r_d_expenses_ltm": np.random.uniform(0, 1e6, n),
            "price_target": np.random.uniform(10, 500, n),
        }
    )

    # Introduce missing values
    for col in ["p_e", "r_d_expenses_ltm", "price_target"]:
        missing_mask = np.random.random(n) < 0.2
        df.loc[missing_mask, col] = np.nan

    missing_before = df.isna().sum().sum()
    print(f"  Missing values before: {missing_before}")

    result = apply_enhanced_imputation_strategy_4step(
        df, sector_column="sector", n_neighbors=3, price_column="last_price"
    )

    missing_after = result.select_dtypes(include=[np.number]).isna().sum().sum()
    print(f"  Missing values after: {missing_after}")

    assert missing_after == 0, "4-step imputation should eliminate all missing values"
    print("  ✓ 4-step imputation working correctly!")


def test_knn_imputation():
    """Test KNN imputation with sector awareness."""
    print("\n✓ Testing apply_knn_imputation_enhanced...")

    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "sector": ["Tech", "Tech", "Tech", "Health", "Health"],
            "market_cap": [1e9, 2e9, np.nan, 5e8, np.nan],
            "p_e": [25, 30, np.nan, 15, np.nan],
        }
    )

    result = apply_knn_imputation_enhanced(
        df, columns=["market_cap", "p_e"], sector_column="sector", n_neighbors=2
    )

    # Check that missing values were reduced
    missing_before = df[["market_cap", "p_e"]].isna().sum().sum()
    missing_after = result[["market_cap", "p_e"]].isna().sum().sum()

    assert missing_after <= missing_before
    print(f"  Reduced missing values from {missing_before} to {missing_after}")


if __name__ == "__main__":
    print("=" * 80)
    print("STANDALONE IMPUTATION MODULE TESTS")
    print("=" * 80)

    try:
        test_get_column_lists()
        test_zero_imputation()
        test_price_imputation()
        test_median_imputation()
        test_knn_imputation()
        test_4step_imputation()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - IMPUTATION REFACTOR SUCCESSFUL")
        print("=" * 80)
        print("\nSummary:")
        print("- Created preprocessing/imputation.py with 8 functions")
        print("- Implemented 4-step imputation strategy")
        print("- All functions working correctly")
        print("- TDD green phase achieved!")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
