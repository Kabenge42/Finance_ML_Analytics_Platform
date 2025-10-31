"""
Validation script for Phase 9 module integrations.
Tests that all modules can be imported and key functions are available.
"""

import sys

print("=" * 80)
print("PHASE 9 MODULE INTEGRATION VALIDATION")
print("=" * 80)

# Test 1: Import core finance_ml package
print("\n[1/5] Testing core package import...")
try:
    import finance_ml

    print(f"✓ finance_ml version: {finance_ml.__version__}")
except Exception as e:
    print(f"✗ Failed to import finance_ml: {e}")
    sys.exit(1)

# Test 2: Import Phase 9.1 Advanced Preprocessing
print("\n[2/5] Testing Phase 9.1 - Advanced Preprocessing...")
try:
    from finance_ml import (
        detect_outliers_iqr_method,
        detect_outliers_zscore_method,
        winsorize_by_sector_method,
        calculate_data_quality_score,
        impute_missing_values,
    )

    print("✓ Advanced preprocessing functions imported")
    print(f"  - detect_outliers_iqr_method: {callable(detect_outliers_iqr_method)}")
    print(f"  - winsorize_by_sector_method: {callable(winsorize_by_sector_method)}")
except Exception as e:
    print(f"✗ Failed to import Phase 9.1 functions: {e}")
    sys.exit(1)

# Test 3: Import Phase 9.2 Advanced EDA
print("\n[3/5] Testing Phase 9.2 - Advanced EDA...")
try:
    from finance_ml import (
        generate_eda_report,
        calculate_correlation_matrix,
        find_top_correlations,
        test_normality,
        perform_pca,
        compare_sector_means,
    )

    print("✓ Advanced EDA functions imported")
    print(f"  - generate_eda_report: {callable(generate_eda_report)}")
    print(f"  - calculate_correlation_matrix: {callable(calculate_correlation_matrix)}")
except Exception as e:
    print(f"✗ Failed to import Phase 9.2 functions: {e}")
    sys.exit(1)

# Test 4: Import Phase 9.4 Classification
print("\n[4/5] Testing Phase 9.4 - Classification...")
try:
    from finance_ml.classification import (
        create_enhanced_event_labels,
        prepare_classification_data,
        train_xgboost_classifier,
        train_voting_classifier,
        train_stacking_classifier,
        export_classification_features,
        compare_classifiers,
    )

    print("✓ Classification functions imported")
    print(f"  - create_enhanced_event_labels: {callable(create_enhanced_event_labels)}")
    print(f"  - export_classification_features: {callable(export_classification_features)}")
except Exception as e:
    print(f"✗ Failed to import Phase 9.4 functions: {e}")
    sys.exit(1)

# Test 5: Import Phase 9.5 Advanced Regression
print("\n[5/5] Testing Phase 9.5 - Advanced Regression...")
try:
    from finance_ml.advanced_models import (
        prepare_regression_data,
        create_classification_interactions,
        train_ridge_regressor,
        train_stacking_regressor,
        train_quantile_regressor,
        compare_regressors,
        train_sector_specific_models,
        save_model,
        load_model,
    )

    print("✓ Advanced regression functions imported")
    print(f"  - prepare_regression_data: {callable(prepare_regression_data)}")
    print(f"  - create_classification_interactions: {callable(create_classification_interactions)}")
    print(f"  - train_stacking_regressor: {callable(train_stacking_regressor)}")
    print(f"  - train_quantile_regressor: {callable(train_quantile_regressor)}")
except Exception as e:
    print(f"✗ Failed to import Phase 9.5 functions: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✓ ALL PHASE 9 MODULES VALIDATED SUCCESSFULLY")
print("=" * 80)
print("\nModule Summary:")
print("  ✓ Phase 9.1: Advanced Preprocessing (5 functions)")
print("  ✓ Phase 9.2: Advanced EDA (6 functions)")
print("  ✓ Phase 9.4: Classification (7 functions)")
print("  ✓ Phase 9.5: Advanced Regression (9 functions)")
print("\nTotal: 27 key functions validated")
print("\nNotebook ml_finance_model_main.ipynb is ready to use!")
