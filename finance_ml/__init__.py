"""
Finance ML Analytics Platform - Package

Modular package for equity screening, feature engineering, and ML models.

Modules:
- finance_ml.data: Data loading, normalization, and validation
- finance_ml.features: Feature engineering functions
- finance_ml.models: Classification, regression, and ensemble models
- finance_ml.eval: Analytics, visualizations, and reporting
- finance_ml.config: Configuration management
- finance_ml.cli: Command-line interface
"""

__version__ = "0.3.0"

# Import from new modular structure (Phase 7 TDD implementation)
from finance_ml.data import (
    setup_logging,
    get_env,
    normalize_columns,
    infer_region_from_filename,
    load_from_csv,
    load_from_db,
    preprocess,
    validate_schema,
    check_missing_values,
    detect_outliers_iqr,
    validate_numeric_ranges,
    _safe_div as _data_safe_div,
)

from finance_ml.features import (
    _safe_div,
    engineer_basic_ratios,
    engineer_margin_features,
    engineer_volatility_features,
    engineer_revenue_cagr,
    build_features_and_target,
)

# Import from models module (Phase 7 TDD implementation complete)
from finance_ml.models import (
    create_event_labels,
    train_event_classifier,
    build_regression_pipeline,
    train_and_evaluate_regression,
    train_and_evaluate_regression_by_sector,
    train_quantile_regression,
    predict_quantile_regression,
    train_quantile_regression_by_sector,
    train_stacking_ensemble,
    train_stacking_ensemble_by_sector,
)

# Import from eval module (Phase 7 TDD implementation complete)
from finance_ml.eval import (
    calculate_mispricing_score,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    rank_stocks_by_sector,
    simple_eda,
    export_predictions_to_excel,
    create_sector_heatmap,
    create_interactive_prediction_plot,
    create_region_sector_heatmap,
)

# Import config module
from finance_ml.config import (
    FinanceMLConfig,
    load_config,
    get_config,
    set_config,
    reset_config,
)

__all__ = [
    # Version
    '__version__',
    # Utilities
    'setup_logging',
    'get_env',
    # Config module
    'FinanceMLConfig',
    'load_config',
    'get_config',
    'set_config',
    'reset_config',
    # Data module
    'normalize_columns',
    'infer_region_from_filename',
    'load_from_csv',
    'load_from_db',
    'preprocess',
    'validate_schema',
    'check_missing_values',
    'detect_outliers_iqr',
    'validate_numeric_ranges',
    # Features module
    'engineer_basic_ratios',
    'engineer_margin_features',
    'engineer_volatility_features',
    'engineer_revenue_cagr',
    'build_features_and_target',
    # Models module
    'create_event_labels',
    'train_event_classifier',
    'build_regression_pipeline',
    'train_and_evaluate_regression',
    'train_and_evaluate_regression_by_sector',
    'train_quantile_regression',
    'predict_quantile_regression',
    'train_quantile_regression_by_sector',
    'train_stacking_ensemble',
    'train_stacking_ensemble_by_sector',
    # Evaluation module
    'calculate_mispricing_score',
    'rank_undervalued_stocks',
    'rank_overvalued_stocks',
    'rank_stocks_by_sector',
    'simple_eda',
    'export_predictions_to_excel',
    'create_sector_heatmap',
    'create_interactive_prediction_plot',
    'create_region_sector_heatmap',
]
