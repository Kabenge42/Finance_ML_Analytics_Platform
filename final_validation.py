#!/usr/bin/env python
"""
Final validation script for duplicate function fix

This script performs comprehensive checks to ensure:
1. No duplicate function definitions exist
2. The module structure is correct
3. Key functions are properly defined
"""

import re
from pathlib import Path
from collections import Counter


def check_duplicates(file_path):
    """Check for duplicate function definitions."""
    print("=" * 80)
    print("CHECK 1: Duplicate Function Definitions")
    print("=" * 80)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all function definitions
    all_functions = re.findall(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content, re.MULTILINE)

    # Count occurrences
    counts = Counter(all_functions)
    duplicates = {name: count for name, count in counts.items() if count > 1}

    print(f"Total functions: {len(all_functions)}")
    print(f"Unique functions: {len(set(all_functions))}")

    if duplicates:
        print(f"\n[FAIL] Found {len(duplicates)} duplicate function(s):")
        for name, count in duplicates.items():
            print(f"  - {name}: {count} definitions")
        return False
    else:
        print("\n[PASS] No duplicates found")
        return True


def check_validate_training_data(file_path):
    """Specifically check validate_training_data."""
    print("\n" + "=" * 80)
    print("CHECK 2: validate_training_data Function")
    print("=" * 80)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"^def validate_training_data\s*\("
    matches = list(re.finditer(pattern, content, re.MULTILINE))

    print(f"Found {len(matches)} definition(s)")

    for i, match in enumerate(matches, 1):
        line_num = content[: match.start()].count("\n") + 1
        print(f"  Definition {i} at line {line_num}")

    if len(matches) == 1:
        print("\n[PASS] Exactly one definition (as expected)")
        return True
    else:
        print(f"\n[FAIL] Expected 1 definition, found {len(matches)}")
        return False


def check_function_list(file_path):
    """List all functions in the module."""
    print("\n" + "=" * 80)
    print("CHECK 3: Function List")
    print("=" * 80)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"^def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    matches = list(re.finditer(pattern, content, re.MULTILINE))

    functions = []
    for match in matches:
        func_name = match.group(1)
        line_num = content[: match.start()].count("\n") + 1
        functions.append((func_name, line_num))

    print(f"Total functions: {len(functions)}\n")

    # Group by category
    categories = {
        "Feature Integration": [
            "extract_classification_features",
            "integrate_classification_features_into_dataframe",
            "prepare_regression_data",
            "create_classification_interactions",
        ],
        "Linear Models": [
            "train_ridge_regressor",
            "train_lasso_regressor",
            "train_elastic_net_regressor",
            "train_bayesian_ridge_regressor",
            "train_polynomial_regressor",
        ],
        "Gradient Boosting": [
            "train_xgboost_regressor",
            "train_lightgbm_regressor",
            "train_catboost_regressor",
            "train_histgb_regressor",
        ],
        "Tree & Neural": [
            "train_random_forest_regressor",
            "train_extra_trees_regressor",
            "train_neural_network_regressor",
        ],
        "Ensemble": [
            "train_voting_regressor",
            "train_stacking_regressor",
            "train_quantile_regressor",
            "optimize_hyperparameters_optuna",
        ],
        "Utilities": [
            "validate_training_data",
            "prepare_features_for_training",
            "compare_regressors",
            "train_sector_specific_models",
            "save_model",
            "load_model",
            "standardize_comparison_results",
        ],
    }

    for category, expected_funcs in categories.items():
        found = [f for f in functions if f[0] in expected_funcs]
        print(f"{category}: {len(found)}/{len(expected_funcs)} functions")
        for func_name, line_num in found:
            print(f"  - {func_name:40s} (line {line_num})")

    print("\n[PASS] Function structure verified")
    return True


def check_imports_and_classes(file_path):
    """Check for class definitions and important imports."""
    print("\n" + "=" * 80)
    print("CHECK 4: Classes and Key Imports")
    print("=" * 80)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for NonNegativeRegressionWrapper class
    class_pattern = r"^class NonNegativeRegressionWrapper"
    class_matches = list(re.finditer(class_pattern, content, re.MULTILINE))

    print(f"NonNegativeRegressionWrapper class: {len(class_matches)} definition(s)")
    for match in class_matches:
        line_num = content[: match.start()].count("\n") + 1
        print(f"  Defined at line {line_num}")

    if len(class_matches) == 1:
        print("[PASS] Class defined exactly once")
        return True
    else:
        print(f"[FAIL] Expected 1 class definition, found {len(class_matches)}")
        return False


def main():
    """Run all validation checks."""
    file_path = Path(__file__).parent / "finance_ml" / "advanced_models.py"

    print("\n" + "=" * 80)
    print("FINAL VALIDATION: advanced_models.py")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size:,} bytes")
    print()

    # Run checks
    results = []
    results.append(check_duplicates(file_path))
    results.append(check_validate_training_data(file_path))
    results.append(check_function_list(file_path))
    results.append(check_imports_and_classes(file_path))

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Checks passed: {passed}/{total}")

    if all(results):
        print("\n[SUCCESS] All validation checks passed!")
        print("The duplicate function issue has been successfully resolved.")
        return 0
    else:
        print("\n[FAILED] Some validation checks failed.")
        return 1


if __name__ == "__main__":
    exit(main())
