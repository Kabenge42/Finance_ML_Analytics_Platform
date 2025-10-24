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
- finance_ml.risk_metrics: Risk metrics and portfolio risk analysis
- finance_ml.logging_config: Logging configuration and file handlers
- finance_ml.portfolio_optimization: Modern Portfolio Theory and optimization
"""

__version__ = "0.3.0"

# Import config module
from finance_ml.config import (
    FinanceMLConfig,
    load_config,
    get_config,
    set_config,
    reset_config,
    )
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
    create_sample_financial_dataset,
    validate_financial_data_quality,
    sanitize_dataframe_with_logging,
    perform_early_pipeline_validation,
    _safe_div as _data_safe_div,
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
from finance_ml.features import (
    _safe_div,
    engineer_basic_ratios,
    engineer_margin_features,
    engineer_volatility_features,
    engineer_revenue_cagr,
    build_features_and_target,
    )
# Import logging configuration module (TDD implementation)
from finance_ml.logging_config import (
    setup_file_logging,
    configure_logging,
    get_logger,
    add_file_handler,
    remove_file_handlers,
    get_log_level,
    set_log_level,
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
    monitor_ensemble_training,
)

# Notebook utility helpers (display + loading strategy)
from finance_ml.notebook_utils import (
    display_config_summary,
    load_stock_data,
    display_data_summary,
    display_validation_results,
    display_missing_values_summary,
    validate_and_display_data,
    perform_and_display_eda,
    )
# Import portfolio optimization module (TDD implementation)
from finance_ml.portfolio_optimization import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe_ratio,
    validate_weights,
    generate_efficient_frontier,
    optimize_portfolio_max_sharpe,
    optimize_portfolio_min_volatility,
    optimize_portfolio_target_return,
    rebalance_portfolio,
    )
# Import risk metrics module (TDD implementation)
from finance_ml.risk_metrics import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_portfolio_risk_metrics,
    )

__all__ = [
    # Version
    "__version__",
    # Utilities
    "setup_logging",
    "get_env",
    # Config module
    "FinanceMLConfig",
    "load_config",
    "get_config",
    "set_config",
    "reset_config",
    # Data module
    "normalize_columns",
    "infer_region_from_filename",
    "load_from_csv",
    "load_from_db",
    "preprocess",
    "validate_schema",
    "check_missing_values",
    "detect_outliers_iqr",
    "validate_numeric_ranges",
    "create_sample_financial_dataset",
    "validate_financial_data_quality",
    "sanitize_dataframe_with_logging",
    "perform_early_pipeline_validation",
    # Features module
    "engineer_basic_ratios",
    "engineer_margin_features",
    "engineer_volatility_features",
    "engineer_revenue_cagr",
    "build_features_and_target",
    # Models module
    "create_event_labels",
    "train_event_classifier",
    "build_regression_pipeline",
    "train_and_evaluate_regression",
    "train_and_evaluate_regression_by_sector",
    "train_quantile_regression",
    "predict_quantile_regression",
    "train_quantile_regression_by_sector",
    "train_stacking_ensemble",
    "train_stacking_ensemble_by_sector",
    "monitor_ensemble_training",
    # Evaluation module
    "calculate_mispricing_score",
    "rank_undervalued_stocks",
    "rank_overvalued_stocks",
    "rank_stocks_by_sector",
    "simple_eda",
    "export_predictions_to_excel",
    "create_sector_heatmap",
    "create_interactive_prediction_plot",
    "create_region_sector_heatmap",
    # Notebook utilities (display + loading strategy)
    "display_config_summary",
    "load_stock_data",
    "display_data_summary",
    "display_validation_results",
    "display_missing_values_summary",
    "validate_and_display_data",
    "perform_and_display_eda",
    # Risk metrics module
    "calculate_var_historical",
    "calculate_var_parametric",
    "calculate_cvar",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_portfolio_risk_metrics",
    # Logging configuration module
    "setup_file_logging",
    "configure_logging",
    "get_logger",
    "add_file_handler",
    "remove_file_handlers",
    "get_log_level",
    "set_log_level",
    # Portfolio optimization module
    "calculate_portfolio_return",
    "calculate_portfolio_volatility",
    "calculate_portfolio_sharpe_ratio",
    "validate_weights",
    "generate_efficient_frontier",
    "optimize_portfolio_max_sharpe",
    "optimize_portfolio_min_volatility",
    "optimize_portfolio_target_return",
    "rebalance_portfolio",
]
