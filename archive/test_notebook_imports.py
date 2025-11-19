"""
Test that all imports in ml_finance_model_main_v10.ipynb are valid.
"""

import sys
import traceback


def test_imports():
    """Test all imports from the notebook."""
    print("=" * 80)
    print("TESTING NOTEBOOK IMPORTS")
    print("=" * 80)

    errors = []
    successes = []

    # Standard library imports
    print("\n1. Testing standard library imports...")
    try:
        import logging
        import traceback as tb
        import warnings
        from dataclasses import dataclass
        from pathlib import Path
        from typing import List, Optional
        from urllib.parse import urljoin
        from urllib.request import pathname2url

        successes.append("Standard library")
        print("  ✓ Standard library imports OK")
    except Exception as e:
        errors.append(("Standard library", str(e)))
        print(f"  ✗ Standard library imports FAILED: {e}")

    # Data science libraries
    print("\n2. Testing data science libraries...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go

        successes.append("Data science libraries")
        print("  ✓ Data science libraries OK")
    except Exception as e:
        errors.append(("Data science libraries", str(e)))
        print(f"  ✗ Data science libraries FAILED: {e}")

    # Sklearn
    print("\n3. Testing sklearn imports...")
    try:
        from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler

        successes.append("Sklearn")
        print("  ✓ Sklearn imports OK")
    except Exception as e:
        errors.append(("Sklearn", str(e)))
        print(f"  ✗ Sklearn imports FAILED: {e}")

    # Finance ML package - Core
    print("\n4. Testing finance_ml core imports...")
    try:
        from finance_ml import (
            __version__,
            load_config,
            setup_logging,
            display_config_summary,
            load_stock_data,
            display_data_summary,
            NotebookConfig,
            PredictionAnalystAnalytics,
            find_peer_group,
            compare_to_peers,
            analyze_metric_trend,
            generate_benchmarking_report,
        )

        successes.append("Finance ML core")
        print("  ✓ Finance ML core imports OK")
    except Exception as e:
        errors.append(("Finance ML core", str(e)))
        print(f"  ✗ Finance ML core imports FAILED: {e}")

    # Finance ML - Data
    print("\n5. Testing finance_ml.data imports...")
    try:
        from finance_ml.data import validate_schema

        successes.append("Finance ML data")
        print("  ✓ Finance ML data imports OK")
    except Exception as e:
        errors.append(("Finance ML data", str(e)))
        print(f"  ✗ Finance ML data imports FAILED: {e}")

    # Finance ML - Preprocessing
    print("\n6. Testing finance_ml.advanced_preprocessing imports...")
    try:
        from finance_ml.advanced_preprocessing import (
            apply_enhanced_imputation_strategy_4step,
            get_zero_imputation_columns,
            get_knn_imputation_columns,
        )

        successes.append("Finance ML preprocessing")
        print("  ✓ Finance ML preprocessing imports OK")
    except Exception as e:
        errors.append(("Finance ML preprocessing", str(e)))
        print(f"  ✗ Finance ML preprocessing imports FAILED: {e}")

    # Finance ML - Features
    print("\n7. Testing finance_ml.features imports...")
    try:
        from finance_ml.features import (
            build_comprehensive_features,
            engineer_basic_ratios,
            engineer_margin_features,
            engineer_volatility_features,
            engineer_revenue_cagr,
        )
        from finance_ml.advanced_features import (
            calculate_feature_importance_rf as calc_rf_importance_adv,
            engineer_valuation_ratios,
            engineer_profitability_ratios,
            engineer_leverage_ratios,
            engineer_liquidity_ratios,
            engineer_efficiency_ratios,
            engineer_growth_metrics,
            engineer_sector_specific_features,
        )

        successes.append("Finance ML features")
        print("  ✓ Finance ML features imports OK")
    except Exception as e:
        errors.append(("Finance ML features", str(e)))
        print(f"  ✗ Finance ML features imports FAILED: {e}")

    # Finance ML - Models
    print("\n8. Testing finance_ml.regression imports...")
    try:
        from finance_ml.models import (
            create_event_labels,
            train_event_classifier,
            train_and_evaluate_regression,
            train_and_evaluate_regression_by_sector,
            train_quantile_regression,
            train_stacking_ensemble,
        )
        from finance_ml.advanced_models import (
            prepare_regression_data,
            create_classification_interactions,
            train_stacking_regressor,
            train_quantile_regressor,
            compare_regressors,
            train_sector_specific_models,
            save_model,
            load_model,
            validate_training_data,
            prepare_features_for_training,
        )

        successes.append("Finance ML regression")
        print("  ✓ Finance ML regression imports OK")
    except Exception as e:
        errors.append(("Finance ML regression", str(e)))
        print(f"  ✗ Finance ML regression imports FAILED: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(successes)}")
    print(f"✗ Failed: {len(errors)}")

    if errors:
        print("\nErrors:")
        for category, error in errors:
            print(f"  - {category}: {error}")
        return False
    else:
        print("\n🎉 All imports are valid!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
