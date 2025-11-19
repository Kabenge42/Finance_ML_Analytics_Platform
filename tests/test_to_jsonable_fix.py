"""
Quick test to verify to_jsonable() fixes the JSON serialization issue.

Tests the fix for: TypeError: Object of type int64 is not JSON serializable
"""

import json
import numpy as np
from finance_ml.ml_workflow.preprocessing.dtypes import to_jsonable


def test_numpy_scalar_conversion():
    """Test that NumPy scalar types are converted correctly."""
    print("Testing NumPy scalar conversion...")

    # Create test data with NumPy scalars (similar to dtype_diagnostics structure)
    test_data = {
        "inferred_dtypes": {"col1": "object", "col2": "object"},
        "cast_applied": {"col1": "float", "col2": "int"},
        "coercion_warnings": {
            "col1": np.int64(5),  # NumPy int64
            "col2": np.int64(10),  # NumPy int64
        },
        "unknown_columns": ["unknown1", "unknown2"],
        "missing_expected_columns": ["expected1"],
        "nested_stats": {
            "count": np.int64(100),
            "mean": np.float64(42.5),
            "flag": np.bool_(True),
            "values": [np.int64(1), np.int64(2), np.int64(3)],
        },
    }

    # Convert to JSON-serializable format
    jsonable_data = to_jsonable(test_data)

    # Try to serialize - this should NOT raise TypeError
    try:
        json_str = json.dumps(jsonable_data, indent=2)
        print("✓ JSON serialization successful!")
        print(f"  Serialized {len(json_str)} characters")

        # Verify the structure is preserved
        parsed = json.loads(json_str)
        assert parsed["coercion_warnings"]["col1"] == 5
        assert parsed["coercion_warnings"]["col2"] == 10
        assert parsed["nested_stats"]["count"] == 100
        assert parsed["nested_stats"]["mean"] == 42.5
        assert parsed["nested_stats"]["flag"] is True
        assert parsed["nested_stats"]["values"] == [1, 2, 3]
        print("✓ Data structure preserved correctly!")

        # Verify types are Python native types
        assert isinstance(parsed["coercion_warnings"]["col1"], int)
        assert isinstance(parsed["nested_stats"]["mean"], float)
        assert isinstance(parsed["nested_stats"]["flag"], bool)
        print("✓ All NumPy types converted to Python native types!")

        return True

    except TypeError as e:
        print(f"✗ JSON serialization failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and mixed types."""
    print("\nTesting edge cases...")

    test_cases = [
        # Deep nesting
        {"level1": {"level2": {"level3": {"value": np.int64(42)}}}},
        # Mixed lists
        [np.int64(1), "string", np.float64(3.14), None, True],
        # Empty containers
        {"empty_dict": {}, "empty_list": []},
        # Already JSON-safe
        {"str": "hello", "int": 42, "float": 3.14, "bool": True, "none": None},
    ]

    for i, test_case in enumerate(test_cases, 1):
        try:
            jsonable = to_jsonable(test_case)
            json.dumps(jsonable)
            print(f"✓ Edge case {i} passed")
        except Exception as e:
            print(f"✗ Edge case {i} failed: {e}")
            return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing to_jsonable() fix for NumPy scalar serialization")
    print("=" * 60)
    print()

    success = True
    success &= test_numpy_scalar_conversion()
    success &= test_edge_cases()

    print()
    print("=" * 60)
    if success:
        print("✓ All tests passed! The fix works correctly.")
    else:
        print("✗ Some tests failed. Check the implementation.")
    print("=" * 60)
