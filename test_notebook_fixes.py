"""
Quick validation script to verify notebook fixes are working correctly.
Tests the three critical imports that were added to the notebook.
"""

print("Testing notebook fixes...\n")

# Test 1: Verify prepare_phase95_data is importable
print("Test 1: Import prepare_phase95_data from finance_ml.advanced_preprocessing")
try:
    from finance_ml.advanced_preprocessing import prepare_phase95_data

    print("  ✓ prepare_phase95_data imported successfully")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 2: Verify extract_numeric_feature_columns is importable
print("\nTest 2: Import extract_numeric_feature_columns from finance_ml.advanced_models")
try:
    from finance_ml.advanced_models import extract_numeric_feature_columns

    print("  ✓ extract_numeric_feature_columns imported successfully")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 3: Verify standardize_comparison_results is importable
print("\nTest 3: Import standardize_comparison_results from finance_ml.advanced_models")
try:
    from finance_ml.advanced_models import standardize_comparison_results

    print("  ✓ standardize_comparison_results imported successfully")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 4: Verify validate_training_data is importable
print("\nTest 4: Import validate_training_data from finance_ml.advanced_models")
try:
    from finance_ml.advanced_models import validate_training_data

    print("  ✓ validate_training_data imported successfully")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 5: Test extract_numeric_feature_columns functionality
print("\nTest 5: Test extract_numeric_feature_columns functionality")
try:
    import pandas as pd
    import numpy as np

    # Create test dataframe
    test_df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "sector": ["Technology", "Technology", "Technology"],
            "price": [150.0, 250.0, 2800.0],
            "volume": [1000000, 2000000, 500000],
            "market_cap": [2.5e12, 2.0e12, 1.5e12],
            "text_col": ["a", "b", "c"],
        }
    )

    numeric_cols = extract_numeric_feature_columns(test_df, exclude_cols=["ticker", "sector"])

    expected = ["price", "volume", "market_cap"]
    if set(numeric_cols) == set(expected):
        print(f"  ✓ Correctly identified numeric columns: {numeric_cols}")
    else:
        print(f"  ✗ FAILED: Expected {expected}, got {numeric_cols}")
        exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 6: Test standardize_comparison_results functionality
print("\nTest 6: Test standardize_comparison_results functionality")
try:
    # Create test comparison results
    test_results = {
        "Ridge": {"mae": 10.5, "rmse": 15.2, "r2": 0.85},
        "Lasso": {"mae": 11.2, "rmse": 16.1, "r2": 0.82},
    }

    results_df = standardize_comparison_results(test_results)

    if "Model" in results_df.columns and len(results_df) == 2:
        print(f"  ✓ Standardized results DataFrame shape: {results_df.shape}")
        print(f"    Columns: {list(results_df.columns)}")
    else:
        print(f"  ✗ FAILED: Unexpected DataFrame structure")
        exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 7: Test validate_training_data functionality
print("\nTest 7: Test validate_training_data functionality")
try:
    import pandas as pd
    import numpy as np

    # Create test data
    X_test = pd.DataFrame(
        {"feature1": [1.0, 2.0, 3.0, 4.0, 5.0], "feature2": [10.0, 20.0, 30.0, 40.0, 50.0]}
    )
    y_test = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])

    result = validate_training_data(X_test, y_test, strict=True)

    if result is not None and "valid" in result:
        print(f"  ✓ Validation returned result: {result}")
    else:
        print(f"  ✗ FAILED: Unexpected validation result")
        exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - Notebook fixes are working correctly!")
print("=" * 80)
