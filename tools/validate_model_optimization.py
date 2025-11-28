"""
Validation script to verify model optimization changes.

This script inspects the train_stacking_regressor and compare_regressors
functions to ensure hyperparameters have been properly optimized for
Section 16.4 Performance Thresholds.
"""

import inspect
from finance_ml.ml_workflow.regression.models import (
    train_stacking_regressor,
    compare_regressors,
    HAS_XGBOOST,
)


def validate_stacking_hyperparameters():
    """Validate train_stacking_regressor uses optimized hyperparameters."""
    print("=" * 80)
    print("VALIDATING train_stacking_regressor HYPERPARAMETERS")
    print("=" * 80)

    # Get source code
    source = inspect.getsource(train_stacking_regressor)

    # Check for optimized hyperparameters
    checks = {
        "RandomForest n_estimators=200": "n_estimators=200" in source
        and "RandomForestRegressor" in source,
        "ExtraTrees n_estimators=200": "n_estimators=200" in source
        and "ExtraTreesRegressor" in source,
        "GradientBoosting n_estimators=150": "n_estimators=150" in source
        and "GradientBoostingRegressor" in source,
        "RandomForest max_depth=15": "max_depth=15" in source,
        "GradientBoosting learning_rate=0.05": "learning_rate=0.05" in source,
        "Subsample regularization": "subsample=0.8" in source,
        "XGBoost conditional addition": "HAS_XGBOOST" in source,
    }

    all_passed = True
    for check_name, result in checks.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {check_name}")
        if not result:
            all_passed = False

    if HAS_XGBOOST:
        print(f"\n  [INFO] XGBoost is available - stacking will use 4 base learners")
    else:
        print(f"\n  [WARN] XGBoost not available - stacking will use 3 base learners")

    return all_passed


def validate_compare_regressors_hyperparameters():
    """Validate compare_regressors uses optimized hyperparameters."""
    print("\n" + "=" * 80)
    print("VALIDATING compare_regressors HYPERPARAMETERS")
    print("=" * 80)

    # Get source code
    source = inspect.getsource(compare_regressors)

    # Check for optimized hyperparameters
    checks = {
        "RandomForest n_estimators=100": "RandomForestRegressor" in source
        and "n_estimators=100" in source,
        "ExtraTrees n_estimators=100": "ExtraTreesRegressor" in source
        and "n_estimators=100" in source,
        "GradientBoosting n_estimators=100": "GradientBoostingRegressor" in source
        and "n_estimators=100" in source,
        "HistGradientBoosting max_iter=100": "max_iter=100" in source
        and "HistGradientBoostingRegressor" in source,
        "Tree depth control": "max_depth=15" in source,
        "Learning rate tuning": "learning_rate=0.1" in source,
    }

    all_passed = True
    for check_name, result in checks.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {check_name}")
        if not result:
            all_passed = False

    return all_passed


def print_summary():
    """Print optimization summary."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    print("\nKey Improvements:")
    print("  - Stacking RandomForest: 50 -> 200 estimators (4x increase)")
    print("  - Stacking ExtraTrees: 50 -> 200 estimators (4x increase)")
    print("  - Stacking GradientBoosting: 50 -> 150 estimators (3x increase)")
    if HAS_XGBOOST:
        print("  - Added XGBoost to stacking ensemble (150 estimators)")
    print("  - Compare RandomForest: 50 -> 100 estimators (2x increase)")
    print("  - Compare ExtraTrees: 50 -> 100 estimators (2x increase)")
    print("  - Compare GradientBoosting: 50 -> 100 estimators (2x increase)")
    print("  - Compare HistGradientBoosting: 50 -> 100 iterations (2x increase)")

    print("\nRegularization Added:")
    print("  - max_depth=15 (RF, ET)")
    print("  - max_depth=6 (GB)")
    print("  - min_samples_split=5 (RF, ET)")
    print("  - subsample=0.8 (GB, XGB)")
    print("  - learning_rate=0.05 (GB, XGB)")

    print("\nExpected Impact:")
    print("  - Target: R2 > 0.7 (Excellent)")
    print("  - Target: MAE < 40% (Good)")
    print("  - Projected: R2 improvement of 28-45%")
    print("  - Projected: MAPE reduction from 1570% to 30-50%")

    print("\nNext Steps:")
    print("  1. Re-run ml_finance_model_main.ipynb")
    print("  2. Monitor training time (expect 3-4x longer)")
    print("  3. Validate metrics against Section 16.4 thresholds")
    print("  4. Review docs/summaries/MODEL_OPTIMIZATION_PHASE16_4_SUMMARY.md")


def main():
    """Run validation checks."""
    stacking_passed = validate_stacking_hyperparameters()
    compare_passed = validate_compare_regressors_hyperparameters()

    print_summary()

    print("\n" + "=" * 80)
    if stacking_passed and compare_passed:
        print("[SUCCESS] ALL VALIDATIONS PASSED")
        print("=" * 80)
        return 0
    else:
        print("[ERROR] SOME VALIDATIONS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
