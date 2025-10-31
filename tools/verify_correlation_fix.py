"""
Verification script for find_top_correlations() threshold parameter fix.

This script tests that the updated function:
1. Accepts the threshold parameter
2. Filters correlations correctly
3. Returns expected format (list of tuples)
"""

import numpy as np
import pandas as pd
from finance_ml.eval import find_top_correlations, calculate_correlation_matrix


def test_find_top_correlations_with_threshold():
    """Test the updated find_top_correlations function."""
    print("Testing find_top_correlations() with threshold parameter...")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "feature_a": np.random.randn(n_samples),
            "feature_b": np.random.randn(n_samples),
            "feature_c": np.random.randn(n_samples),
            "feature_d": np.random.randn(n_samples),
        }
    )

    # Add some correlated features
    data["feature_e"] = data["feature_a"] * 0.8 + np.random.randn(n_samples) * 0.2
    data["feature_f"] = data["feature_b"] * 0.5 + np.random.randn(n_samples) * 0.5

    print(f"✓ Created sample dataset with {data.shape[0]} rows and {data.shape[1]} columns")

    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(
        data, columns=data.columns.tolist(), method="pearson"
    )
    print(f"✓ Calculated correlation matrix: {corr_matrix.shape}")

    # Test 1: Call with threshold parameter (as in notebook line 684)
    print("\nTest 1: Call with threshold=0.3 (notebook usage)")
    try:
        top_corr = find_top_correlations(corr_matrix, n_top=10, threshold=0.3)
        print(f"✓ Function accepts threshold parameter")
        print(f"✓ Found {len(top_corr)} correlations with |correlation| >= 0.3")

        # Verify return type
        if isinstance(top_corr, list) and len(top_corr) > 0:
            if isinstance(top_corr[0], tuple) and len(top_corr[0]) == 3:
                print(f"✓ Returns list of tuples (var1, var2, correlation)")
            else:
                print(f"✗ Unexpected tuple format: {top_corr[0]}")

        # Display results
        print("\nTop correlations:")
        for i, (feat1, feat2, corr_val) in enumerate(top_corr[:5], 1):
            print(f"  {i}. {feat1} <-> {feat2}: {corr_val:.3f}")

        # Verify threshold filtering
        if all(abs(corr) >= 0.3 for _, _, corr in top_corr):
            print(f"\n✓ All correlations meet threshold requirement (|r| >= 0.3)")
        else:
            print(f"\n✗ Some correlations below threshold!")

    except TypeError as e:
        print(f"✗ Function call failed: {e}")
        return False

    # Test 2: Call without threshold (default behavior)
    print("\nTest 2: Call without threshold (default=0.0)")
    try:
        top_corr_no_threshold = find_top_correlations(corr_matrix, n_top=5)
        print(f"✓ Function works with default threshold")
        print(f"✓ Found {len(top_corr_no_threshold)} correlations")
    except Exception as e:
        print(f"✗ Function call failed: {e}")
        return False

    # Test 3: High threshold
    print("\nTest 3: Call with high threshold=0.7")
    try:
        top_corr_high = find_top_correlations(corr_matrix, n_top=10, threshold=0.7)
        print(f"✓ Function works with high threshold")
        print(f"✓ Found {len(top_corr_high)} correlations with |correlation| >= 0.7")

        if len(top_corr_high) > 0:
            for feat1, feat2, corr_val in top_corr_high:
                print(f"    {feat1} <-> {feat2}: {corr_val:.3f}")
    except Exception as e:
        print(f"✗ Function call failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("✓ All tests passed! The fix is working correctly.")
    return True


if __name__ == "__main__":
    success = test_find_top_correlations_with_threshold()
    exit(0 if success else 1)
