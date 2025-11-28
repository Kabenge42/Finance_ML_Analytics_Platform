"""Test to verify the validate_expected_returns KeyError fix."""

import numpy as np
import pytest
from finance_ml.ml_workflow.analytics.ml_returns import validate_expected_returns


def test_validate_empty_array_has_warnings_key():
    """Test that validate_expected_returns returns consistent schema for empty arrays."""
    # Empty array scenario (the bug trigger)
    empty_returns = np.array([])

    diagnostics = validate_expected_returns(empty_returns)

    # Critical: 'warnings' key must exist
    assert "warnings" in diagnostics, "Missing 'warnings' key in diagnostics"
    assert isinstance(diagnostics["warnings"], list), "'warnings' must be a list"
    assert len(diagnostics["warnings"]) == 0, "Empty array should have empty warnings list"

    # Verify other required keys
    assert diagnostics["is_realistic"] is False
    assert diagnostics["reason"] == "Empty returns array"
    assert np.isnan(diagnostics["mean_return"])
    assert np.isnan(diagnostics["std_return"])
    assert diagnostics["n_samples"] == 0


def test_validate_normal_returns_has_warnings_key():
    """Test that validate_expected_returns returns consistent schema for normal data."""
    # Normal data scenario
    normal_returns = np.array([0.10, 0.15, 0.20, 0.08, 0.12])

    diagnostics = validate_expected_returns(normal_returns)

    # Critical: 'warnings' key must exist
    assert "warnings" in diagnostics, "Missing 'warnings' key in diagnostics"
    assert isinstance(diagnostics["warnings"], list), "'warnings' must be a list"

    # Should be realistic with no warnings
    assert diagnostics["is_realistic"] is True
    assert len(diagnostics["warnings"]) == 0


def test_validate_unrealistic_returns_has_warnings():
    """Test that unrealistic returns populate warnings list."""
    # Unrealistic returns (mean > 30%)
    unrealistic_returns = np.array([0.50, 0.60, 0.70, 0.80])

    diagnostics = validate_expected_returns(unrealistic_returns)

    # Should have warnings
    assert "warnings" in diagnostics
    assert isinstance(diagnostics["warnings"], list)
    assert len(diagnostics["warnings"]) > 0, "Unrealistic returns should have warnings"
    assert diagnostics["is_realistic"] is False


def test_validate_all_nan_returns():
    """Test that all-NaN returns are handled correctly."""
    # All NaN returns
    nan_returns = np.array([np.nan, np.nan, np.nan])

    diagnostics = validate_expected_returns(nan_returns)

    # After filtering NaN, becomes empty array
    assert "warnings" in diagnostics
    assert diagnostics["is_realistic"] is False
    assert diagnostics["n_samples"] == 0


def test_notebook_usage_pattern():
    """Test the exact usage pattern from the notebook."""
    # Simulate notebook scenario
    selected_stocks_data = {"expected_return": [0.15, 0.20, np.nan, 0.10, 0.25]}

    # Extract returns as in notebook
    raw_returns = np.array(selected_stocks_data["expected_return"])
    non_null_returns = raw_returns[~np.isnan(raw_returns)]

    # Validate (this is where the bug occurred)
    diagnostics = validate_expected_returns(non_null_returns)

    # The notebook code that was failing
    assert diagnostics["is_realistic"] in [True, False]

    # Safe access pattern (what notebook now uses)
    if diagnostics.get("warnings"):
        for warn in diagnostics["warnings"]:
            assert isinstance(warn, str)


def test_schema_consistency():
    """Verify all return paths have the same schema keys."""
    test_cases = [
        np.array([]),  # Empty
        np.array([0.10]),  # Single value
        np.array([0.10, 0.20, 0.30]),  # Normal
        np.array([0.50, 0.60, 0.70]),  # Unrealistic
        np.array([np.nan]),  # All NaN
    ]

    expected_keys = {
        "is_realistic",
        "mean",
        "mean_return",
        "std_return",
        "max",
        "min",
        "n_samples",
        "warnings",
    }

    for test_array in test_cases:
        diagnostics = validate_expected_returns(test_array)
        actual_keys = set(diagnostics.keys())

        # All expected keys must be present
        assert expected_keys.issubset(
            actual_keys
        ), f"Missing keys: {expected_keys - actual_keys} for array: {test_array}"


if __name__ == "__main__":
    # Run tests
    print("Testing validate_expected_returns KeyError fix...")

    test_validate_empty_array_has_warnings_key()
    print("[PASS] Empty array test passed")

    test_validate_normal_returns_has_warnings_key()
    print("[PASS] Normal returns test passed")

    test_validate_unrealistic_returns_has_warnings()
    print("[PASS] Unrealistic returns test passed")

    test_validate_all_nan_returns()
    print("[PASS] All-NaN returns test passed")

    test_notebook_usage_pattern()
    print("[PASS] Notebook usage pattern test passed")

    test_schema_consistency()
    print("[PASS] Schema consistency test passed")

    print("\n[SUCCESS] All tests passed! The KeyError fix is working correctly.")
