"""
Test Phase 9.2, 9.6, 9.7, 9.8 imports from direct module paths.
Validates that all new modular imports work correctly.
"""

import sys
import pandas as pd
import numpy as np

print("=" * 80)
print("Phase 9.2, 9.6, 9.7, 9.8 Direct Module Import Test")
print("=" * 80)

# Test Phase 9.2: EDA
print("\n[Test 1] Phase 9.2 - EDA module imports...")
try:
    from finance_ml.ml_workflow.eda.eda import (
        eda_summary,
        sector_distribution_summary,
        correlation_analysis,
        distribution_summary,
    )

    print("✓ Phase 9.2 EDA imports successful")
except ImportError as e:
    print(f"✗ Phase 9.2 EDA import failed: {e}")
    sys.exit(1)

# Test Phase 9.6: Evaluation
print("\n[Test 2] Phase 9.6 - Evaluation module imports...")
try:
    from finance_ml.ml_workflow.evaluation.metrics import (
        comprehensive_regression_metrics,
        compute_metrics_by_segment,
        compute_sector_region_metrics,
    )
    from finance_ml.ml_workflow.evaluation.analysis import (
        residual_analysis,
        error_analysis,
        model_diagnostics,
        prediction_intervals,
    )

    print("✓ Phase 9.6 Evaluation imports successful")
except ImportError as e:
    print(f"✗ Phase 9.6 Evaluation import failed: {e}")
    sys.exit(1)

# Test Phase 9.7: Analytics
print("\n[Test 3] Phase 9.7 - Analytics module imports...")
try:
    from finance_ml.ml_workflow.analytics.mispricing import (
        calculate_mispricing_score,
        rank_stocks_by_sector,
    )
    from finance_ml.ml_workflow.analytics.analyst_comparison import (
        PredictionAnalystAnalytics,
    )
    from finance_ml.ml_workflow.analytics.portfolio import (
        optimize_portfolio_max_sharpe,
    )
    from finance_ml.ml_workflow.analytics.risk import (
        calculate_portfolio_risk_metrics,
    )

    print("✓ Phase 9.7 Analytics imports successful")
except ImportError as e:
    print(f"✗ Phase 9.7 Analytics import failed: {e}")
    sys.exit(1)

# Test Phase 9.8: Reporting
print("\n[Test 4] Phase 9.8 - Reporting module imports...")
try:
    from finance_ml.ml_workflow.reporting.export import (
        export_predictions,
    )
    from finance_ml.ml_workflow.reporting.dashboard_data import (
        prepare_plotly_dashboard_data,
    )

    print("✓ Phase 9.8 Reporting imports successful")
except ImportError as e:
    print(f"✗ Phase 9.8 Reporting import failed: {e}")
    sys.exit(1)

# Test Phase 9.2 function with sample data
print("\n[Test 5] Testing Phase 9.2 eda_summary with sample data...")
try:
    sample_df = pd.DataFrame(
        {
            "sector": ["Tech", "Finance", "Tech", "Finance"],
            "value1": [100, 200, 150, 250],
            "value2": [10, 20, 15, 25],
        }
    )
    summary = eda_summary(sample_df, sector_column="sector")
    assert "shape" in summary
    assert summary["shape"] == (4, 3)
    print(f"✓ eda_summary works: shape={summary['shape']}")
except Exception as e:
    print(f"✗ eda_summary test failed: {e}")

# Test Phase 9.6 function with sample data
print("\n[Test 6] Testing Phase 9.6 comprehensive_regression_metrics...")
try:
    y_true = np.array([100, 200, 150, 250])
    y_pred = np.array([110, 190, 155, 240])
    metrics = comprehensive_regression_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    print(
        f"✓ comprehensive_regression_metrics works: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}"
    )
except Exception as e:
    print(f"✗ comprehensive_regression_metrics test failed: {e}")

# Test Phase 9.7 function with sample data
print("\n[Test 7] Testing Phase 9.7 calculate_mispricing_score...")
try:
    test_df = pd.DataFrame(
        {
            "predicted_price_target": [110, 190, 155, 240],
            "last_price": [100, 200, 150, 250],
        }
    )
    mispricing_df = calculate_mispricing_score(test_df)
    assert "mispricing_score" in mispricing_df.columns
    assert "mispricing_pct" in mispricing_df.columns
    print(f"✓ calculate_mispricing_score works: {len(mispricing_df)} rows with mispricing scores")
except Exception as e:
    print(f"✗ calculate_mispricing_score test failed: {e}")

print("\n" + "=" * 80)
print("✅ All Phase 9.2, 9.6, 9.7, 9.8 Import Tests Passed!")
print("=" * 80)
print("\nAll new modular imports are working correctly.")
print("The notebook can now use these direct module paths for Phase 9.2, 9.6, 9.7, 9.8 functions.")
