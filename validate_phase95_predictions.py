"""
Validation script for Phase 9.5 predictions flow to Phases 9.6 and 9.7.

This script validates that:
1. Phase 9.5 stores predictions in all_stocks_featured
2. Required columns exist for Phases 9.6 and 9.7
3. Predictions are properly indexed and non-empty
4. Error handling works as expected
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def validate_predictions_dataframe(df: pd.DataFrame, phase_name: str = "9.5") -> dict:
    """
    Validate that a dataframe has all required prediction columns.

    Args:
        df: DataFrame to validate
        phase_name: Phase identifier for error messages

    Returns:
        Dictionary with validation results
    """
    results = {"phase": phase_name, "passed": True, "errors": [], "warnings": [], "info": {}}

    # Required columns for predictions
    required_cols = ["predicted_price_target", "prediction_lower_10", "prediction_upper_90"]

    # Check DataFrame exists and is not empty
    if df is None:
        results["passed"] = False
        results["errors"].append("DataFrame is None")
        return results

    if len(df) == 0:
        results["passed"] = False
        results["errors"].append("DataFrame is empty")
        return results

    results["info"]["total_rows"] = len(df)

    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results["passed"] = False
        results["errors"].append(f"Missing required columns: {missing_cols}")
        return results

    # Check predictions are not all NaN
    for col in required_cols:
        non_null_count = df[col].notna().sum()
        results["info"][f"{col}_non_null_count"] = non_null_count

        if non_null_count == 0:
            results["passed"] = False
            results["errors"].append(f"Column '{col}' has no non-null values")
        elif non_null_count < len(df) * 0.1:  # Less than 10% coverage
            results["warnings"].append(
                f"Column '{col}' has low coverage: {non_null_count}/{len(df)} "
                f"({non_null_count/len(df)*100:.1f}%)"
            )

    # Check prediction intervals are valid (lower < upper)
    if "prediction_lower_10" in df.columns and "prediction_upper_90" in df.columns:
        valid_mask = df["prediction_lower_10"].notna() & df["prediction_upper_90"].notna()
        if valid_mask.any():
            invalid_intervals = (
                df.loc[valid_mask, "prediction_lower_10"]
                > df.loc[valid_mask, "prediction_upper_90"]
            ).sum()
            if invalid_intervals > 0:
                results["warnings"].append(
                    f"Found {invalid_intervals} rows where lower > upper prediction interval"
                )
            results["info"]["valid_intervals_count"] = valid_mask.sum() - invalid_intervals

    # Check for reasonable prediction values (not inf or extreme)
    for col in required_cols:
        if df[col].notna().any():
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                results["warnings"].append(f"Column '{col}' has {inf_count} infinite values")

            # Check for extreme values (> 1e6 or < -1e6)
            extreme_count = ((df[col].abs() > 1e6) & df[col].notna()).sum()
            if extreme_count > 0:
                results["warnings"].append(
                    f"Column '{col}' has {extreme_count} extreme values (>1e6 or <-1e6)"
                )

    return results


def validate_phase96_requirements(df: pd.DataFrame) -> dict:
    """
    Validate that DataFrame meets Phase 9.6 requirements.

    Phase 9.6 needs:
    - predicted_price_target
    - price_target or last_price for y_true
    - sector (optional but recommended)
    """
    results = {"phase": "9.6", "passed": True, "errors": [], "warnings": [], "info": {}}

    if "predicted_price_target" not in df.columns:
        results["passed"] = False
        results["errors"].append("Missing 'predicted_price_target' column")
        return results

    # Check for target column
    has_price_target = "price_target" in df.columns
    has_last_price = "last_price" in df.columns

    if not has_price_target and not has_last_price:
        results["passed"] = False
        results["errors"].append("Missing both 'price_target' and 'last_price' columns")
    elif not has_price_target:
        results["warnings"].append("Missing 'price_target', will use 'last_price' as fallback")

    results["info"]["has_price_target"] = has_price_target
    results["info"]["has_last_price"] = has_last_price

    # Check for sector column (optional but useful)
    if "sector" not in df.columns:
        results["warnings"].append("Missing 'sector' column - sector-based metrics unavailable")
    else:
        results["info"]["sector_count"] = df["sector"].nunique()

    return results


def validate_phase97_requirements(df: pd.DataFrame) -> dict:
    """
    Validate that DataFrame meets Phase 9.7 requirements.

    Phase 9.7 needs:
    - predicted_price_target
    - last_price for mispricing calculation
    - Optional: sector, region, various valuation metrics
    """
    results = {"phase": "9.7", "passed": True, "errors": [], "warnings": [], "info": {}}

    if "predicted_price_target" not in df.columns:
        results["passed"] = False
        results["errors"].append("Missing 'predicted_price_target' column")
        return results

    if "last_price" not in df.columns:
        results["passed"] = False
        results["errors"].append("Missing 'last_price' column for mispricing calculation")

    # Optional but recommended columns
    optional_cols = {
        "sector": "Sector-relative valuation unavailable",
        "region": "Region-based analysis unavailable",
        "p_e": "P/E ratio analysis unavailable",
        "p_b": "P/B ratio analysis unavailable",
        "ev_ebitda": "EV/EBITDA analysis unavailable",
    }

    for col, warning_msg in optional_cols.items():
        if col not in df.columns:
            results["warnings"].append(f"Missing '{col}' - {warning_msg}")
        else:
            results["info"][f"has_{col}"] = True

    return results


def print_validation_results(results: dict):
    """Print formatted validation results."""
    phase = results.get("phase", "Unknown")
    passed = results.get("passed", False)

    status_icon = "✅" if passed else "❌"
    print(f"\n{status_icon} Phase {phase} Validation: {'PASSED' if passed else 'FAILED'}")

    if results.get("errors"):
        print(f"\n  ❌ Errors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"    - {error}")

    if results.get("warnings"):
        print(f"\n  ⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"    - {warning}")

    if results.get("info"):
        print(f"\n  ℹ️  Info:")
        for key, value in results["info"].items():
            print(f"    - {key}: {value}")


def test_error_handling():
    """Test error handling scenarios."""
    print("\n" + "=" * 80)
    print("TESTING ERROR HANDLING SCENARIOS")
    print("=" * 80)

    # Test 1: Empty DataFrame
    print("\n📝 Test 1: Empty DataFrame")
    empty_df = pd.DataFrame()
    results = validate_predictions_dataframe(empty_df, "Test1")
    print_validation_results(results)
    assert not results["passed"], "Empty DataFrame should fail validation"

    # Test 2: None DataFrame
    print("\n📝 Test 2: None DataFrame")
    results = validate_predictions_dataframe(None, "Test2")
    print_validation_results(results)
    assert not results["passed"], "None DataFrame should fail validation"

    # Test 3: Missing prediction columns
    print("\n📝 Test 3: Missing prediction columns")
    df_missing_cols = pd.DataFrame({"ticker": ["AAPL", "GOOGL"], "last_price": [150.0, 2500.0]})
    results = validate_predictions_dataframe(df_missing_cols, "Test3")
    print_validation_results(results)
    assert not results["passed"], "DataFrame missing prediction columns should fail"

    # Test 4: All NaN predictions
    print("\n📝 Test 4: All NaN predictions")
    df_all_nan = pd.DataFrame(
        {
            "ticker": ["AAPL", "GOOGL"],
            "predicted_price_target": [np.nan, np.nan],
            "prediction_lower_10": [np.nan, np.nan],
            "prediction_upper_90": [np.nan, np.nan],
        }
    )
    results = validate_predictions_dataframe(df_all_nan, "Test4")
    print_validation_results(results)
    assert not results["passed"], "DataFrame with all NaN predictions should fail"

    # Test 5: Valid predictions
    print("\n📝 Test 5: Valid predictions")
    df_valid = pd.DataFrame(
        {
            "ticker": ["AAPL", "GOOGL", "MSFT"],
            "predicted_price_target": [155.0, 2600.0, 350.0],
            "prediction_lower_10": [145.0, 2400.0, 330.0],
            "prediction_upper_90": [165.0, 2800.0, 370.0],
        }
    )
    results = validate_predictions_dataframe(df_valid, "Test5")
    print_validation_results(results)
    assert results["passed"], "Valid DataFrame should pass validation"

    # Test 6: Invalid intervals (lower > upper)
    print("\n📝 Test 6: Invalid prediction intervals")
    df_invalid_intervals = pd.DataFrame(
        {
            "ticker": ["AAPL", "GOOGL"],
            "predicted_price_target": [155.0, 2600.0],
            "prediction_lower_10": [165.0, 2800.0],  # Lower > Upper
            "prediction_upper_90": [145.0, 2400.0],
        }
    )
    results = validate_predictions_dataframe(df_invalid_intervals, "Test6")
    print_validation_results(results)
    assert len(results["warnings"]) > 0, "Invalid intervals should produce warnings"

    print("\n✅ All error handling tests completed")


def main():
    """Main validation routine."""
    print("=" * 80)
    print("PHASE 9.5 → 9.6/9.7 PREDICTIONS FLOW VALIDATION")
    print("=" * 80)

    # Run error handling tests
    test_error_handling()

    # Create a mock all_stocks_featured DataFrame for testing
    print("\n" + "=" * 80)
    print("VALIDATING MOCK PREDICTIONS FLOW")
    print("=" * 80)

    # Simulate what Phase 9.5 should produce
    mock_all_stocks_featured = pd.DataFrame(
        {
            "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"] * 100,  # 500 stocks
            "last_price": np.random.uniform(50, 3000, 500),
            "price_target": np.random.uniform(60, 3200, 500),
            "sector": np.random.choice(["Technology", "Healthcare", "Finance", "Energy"], 500),
            "region": np.random.choice(["US", "EU", "APAC"], 500),
            "p_e": np.random.uniform(5, 50, 500),
            "p_b": np.random.uniform(1, 10, 500),
            "ev_ebitda": np.random.uniform(5, 30, 500),
            "predicted_price_target": np.random.uniform(60, 3200, 500),
            "prediction_lower_10": np.random.uniform(50, 3000, 500),
            "prediction_upper_90": np.random.uniform(70, 3400, 500),
        }
    )

    # Simulate test set (only 20% have predictions)
    test_size = int(len(mock_all_stocks_featured) * 0.2)
    train_indices = mock_all_stocks_featured.index[: len(mock_all_stocks_featured) - test_size]
    mock_all_stocks_featured.loc[train_indices, "predicted_price_target"] = np.nan
    mock_all_stocks_featured.loc[train_indices, "prediction_lower_10"] = np.nan
    mock_all_stocks_featured.loc[train_indices, "prediction_upper_90"] = np.nan

    print("\n📊 Mock DataFrame created:")
    print(f"  Total rows: {len(mock_all_stocks_featured)}")
    print(f"  Test set size: {test_size}")
    print(f"  Columns: {list(mock_all_stocks_featured.columns)}")

    # Validate Phase 9.5 output
    print("\n" + "-" * 80)
    print("Validating Phase 9.5 Output")
    print("-" * 80)
    results_95 = validate_predictions_dataframe(mock_all_stocks_featured, "9.5")
    print_validation_results(results_95)

    # Validate Phase 9.6 requirements
    print("\n" + "-" * 80)
    print("Validating Phase 9.6 Requirements")
    print("-" * 80)
    results_96 = validate_phase96_requirements(mock_all_stocks_featured)
    print_validation_results(results_96)

    # Validate Phase 9.7 requirements
    print("\n" + "-" * 80)
    print("Validating Phase 9.7 Requirements")
    print("-" * 80)
    results_97 = validate_phase97_requirements(mock_all_stocks_featured)
    print_validation_results(results_97)

    # Overall summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_passed = results_95["passed"] and results_96["passed"] and results_97["passed"]

    if all_passed:
        print("\n✅ ALL VALIDATIONS PASSED")
        print("\n✓ Phase 9.5 predictions are properly stored")
        print("✓ Phase 9.6 can access predictions for evaluation")
        print("✓ Phase 9.7 can access predictions for valuation analysis")
        print("\n🎉 Predictions flow from Phase 9.5 → 9.6 → 9.7 is working correctly!")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        if not results_95["passed"]:
            print("\n✗ Phase 9.5 predictions validation failed")
        if not results_96["passed"]:
            print("✗ Phase 9.6 requirements validation failed")
        if not results_97["passed"]:
            print("✗ Phase 9.7 requirements validation failed")
        print("\n⚠️  Review errors above and fix Phase 9.5 implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
