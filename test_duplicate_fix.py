#!/usr/bin/env python
"""
Test script to verify the duplicate function fix in advanced_models.py

This script validates:
1. validate_training_data can be imported
2. The function is defined only once
3. Basic functionality works as expected
"""

import sys
import re
from pathlib import Path


def test_no_duplicate_definition():
    """Test that validate_training_data is defined only once."""
    print("=" * 70)
    print("TEST 1: Checking for duplicate function definitions")
    print("=" * 70)

    file_path = Path(__file__).parent / "finance_ml" / "advanced_models.py"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all validate_training_data definitions
    pattern = r"^def validate_training_data\s*\("
    matches = list(re.finditer(pattern, content, re.MULTILINE))

    print(f"Found {len(matches)} definition(s) of validate_training_data")

    for i, match in enumerate(matches, 1):
        line_num = content[: match.start()].count("\n") + 1
        print(f"  Definition {i} at line {line_num}")

    if len(matches) == 1:
        print("PASS: Only one definition found (as expected)")
        return True
    else:
        print(f"FAIL: Expected 1 definition, found {len(matches)}")
        return False


def test_import_works():
    """Test that the function can be imported."""
    print("\n" + "=" * 70)
    print("TEST 2: Testing import")
    print("=" * 70)

    try:
        from finance_ml.advanced_models import validate_training_data

        print("PASS: Successfully imported validate_training_data")
        print(f"  Function name: {validate_training_data.__name__}")
        print(f"  Defined at line: {validate_training_data.__code__.co_firstlineno}")
        return True
    except Exception as e:
        print(f"FAIL: Import failed with error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of validate_training_data."""
    print("\n" + "=" * 70)
    print("TEST 3: Testing basic functionality")
    print("=" * 70)

    try:
        from finance_ml.advanced_models import validate_training_data
        import pandas as pd
        import numpy as np

        # Create valid test data
        X_valid = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
        y_valid = pd.Series([10, 20, 30, 40, 50])

        # Test 1: Valid data (non-strict mode)
        result = validate_training_data(X_valid, y_valid, strict=False)
        if result["valid"]:
            print("PASS: Valid data recognized correctly")
        else:
            print(f"FAIL: Valid data marked as invalid: {result['issues']}")
            return False

        # Test 2: Data with NaN (non-strict mode)
        X_nan = X_valid.copy()
        X_nan.loc[0, "feature1"] = np.nan
        result_nan = validate_training_data(X_nan, y_valid, strict=False)

        if not result_nan["valid"] and result_nan["nan_features"] > 0:
            print("PASS: NaN detection working correctly")
        else:
            print("FAIL: NaN not detected properly")
            return False

        print("PASS: All basic functionality tests passed")
        return True

    except Exception as e:
        print(f"FAIL: Functionality test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DUPLICATE FUNCTION FIX VALIDATION")
    print("Testing advanced_models.py after removing duplicate validate_training_data")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("No duplicate definitions", test_no_duplicate_definition()))
    results.append(("Import works", test_import_works()))
    results.append(("Basic functionality", test_basic_functionality()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! The duplicate function issue has been resolved.")
        return 0
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
