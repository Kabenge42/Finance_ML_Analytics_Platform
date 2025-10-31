"""
Finance ML Analytics Platform - Package

Modular package for equity screening, feature engineering, and ML models.

Modules:
- finance_ml.data: Data loading, normalization, and validation
- finance_ml.advanced_preprocessing: Advanced preprocessing (Phase 9.1)
- finance_ml.features: Feature engineering functions
- finance_ml.models: Classification, regression, and ensemble models
- finance_ml.eval: Analytics, visualizations, and reporting
- finance_ml.config: Configuration management
- finance_ml.cli: Command-line interface
- finance_ml.risk_metrics: Risk metrics and portfolio risk analysis
- finance_ml.logging_config: Logging configuration and file handlers
- finance_ml.portfolio_optimization: Modern Portfolio Theory and optimization
"""

__version__ = "0.4.1"

# Import from advanced_eda module (Phase 9.2)
from finance_ml.advanced_eda import (
    CorrelationReport,
    StatisticalTestResult,
    EDAReport,
    calculate_correlation_matrix as calc_corr_matrix,
    find_top_correlations as find_top_corr,
    test_normality as test_norm,
    calculate_skewness_kurtosis as calc_skew_kurt,
    detect_outliers_statistical as detect_outliers_stat,
    calculate_mutual_information as calc_mutual_info,
    calculate_feature_importance_rf as calc_rf_importance,
    perform_pca,
    calculate_optimal_pca_components as calc_optimal_pca,
    compare_sector_means,
    compare_two_groups,
    generate_eda_report,
    generate_sector_comparison_report,
    )
# Import from advanced_models module (Phase 9.5)
from finance_ml.advanced_models import (
    # Feature Integration
    prepare_regression_data,
    create_classification_interactions,
    # Linear Models
    train_ridge_regressor,
    train_lasso_regressor,
    train_elastic_net_regressor,
    train_bayesian_ridge_regressor,
    train_polynomial_regressor,
    # Gradient Boosting Models
    train_xgboost_regressor,
    train_lightgbm_regressor,
    train_catboost_regressor,
    train_histgb_regressor,
    # Tree and Neural Models
    train_random_forest_regressor,
    train_extra_trees_regressor,
    train_neural_network_regressor,
    # Ensemble Methods
    train_voting_regressor,
    train_stacking_regressor,
    train_quantile_regressor,
    optimize_hyperparameters_optuna,
    # Utilities
    compare_regressors,
    train_sector_specific_models,
    save_model,
    load_model,
    )
# Import from advanced_preprocessing module (Phase 9.1)
from finance_ml.advanced_preprocessing import (
    DataQualityReport,
    detect_outliers_iqr as detect_outliers_iqr_method,
    detect_outliers_zscore as detect_outliers_zscore_method,
    detect_outliers_isolation_forest,
    winsorize_by_sector as winsorize_by_sector_method,
    calculate_data_quality_score,
    impute_missing_values,
    impute_missing_values_knn_sector,
    create_scaler_pipeline,
    scale_features,
)
# Import from benchmarking module (Phase 9.2)
from finance_ml.benchmarking import (
    compare_sector_distributions,
    compare_regional_valuations,
    find_peer_group,
    compare_to_peers,
    analyze_metric_trend,
    generate_benchmarking_report,
)

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
    # Phase 9.1: Advanced preprocessing functions
    detect_outliers_iqr_advanced,
    detect_outliers_by_sector,
    detect_outliers_zscore,
    winsorize_column,
    winsorize_by_sector,
    calculate_completeness_score,
    calculate_consistency_score,
    impute_by_sector,
    safe_divide,
    create_temporal_split,
    create_expanding_windows,
    )
# Import from data_catalog module (Phase 9.1 - Data Catalog)
from finance_ml.data_catalog import (
    SchemaInfo,
    StatisticalProfile,
    DatasetMetadata,
    DataCatalog,
    extract_schema_info,
    create_statistical_profile,
    )
# Import from data_versioning module (Phase 9.1 - Data Versioning)
from finance_ml.data_versioning import (
    DataVersion,
    DataVersionManager,
    calculate_dataframe_hash,
    compare_versions,
    create_version_snapshot,
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
    # Phase 9.2: Advanced EDA functions
    calculate_correlation_matrix,
    find_top_correlations,
    test_normality,
    calculate_skewness_kurtosis,
    detect_outliers_statistical,
    calculate_mutual_information,
    calculate_feature_importance_rf,
    calculate_shap_importance,
    perform_pca,
    perform_tsne,
    perform_umap,
    calculate_optimal_pca_components,
    compare_sector_means,
    compare_two_groups,
    generate_eda_report,
    generate_sector_comparison_report,
    # Phase 9.1 Enhancement #3: Data Quality Dashboard
    generate_data_quality_dashboard,
    export_profiling_report,
    # Phase 9.8: Prediction vs. Analyst Price Target Comparison Analytics
    compare_prediction_vs_analyst_targets,
    calculate_directional_accuracy,
    calculate_agreement_rate,
    identify_disagreement_opportunities,
    calculate_prediction_accuracy_metrics,
    segment_comparison_by_attribute,
    analyze_systematic_bias,
    generate_prediction_analyst_excel_report,
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
# Import from transformers module (Phase 9.1 Enhancements #2 and #5)
from finance_ml.transformers import (
    RegularizedTargetEncoder,
    TargetEncoder,
    FinancialRatioTransformer,
    SafeDivisionTransformer,
    ValuationRatioTransformer,
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
# Import notebook configuration module
from finance_ml.notebook_config import NotebookConfig
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
    # Notebook configuration
    "NotebookConfig",
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
    # Phase 9.1: Advanced preprocessing
    "detect_outliers_iqr_advanced",
    "detect_outliers_by_sector",
    "detect_outliers_zscore",
    "winsorize_column",
    "winsorize_by_sector",
    "calculate_completeness_score",
    "calculate_consistency_score",
    "impute_by_sector",
    "safe_divide",
    "create_temporal_split",
    "create_expanding_windows",
    # Advanced preprocessing module (Phase 9.1)
    "DataQualityReport",
    "detect_outliers_iqr_method",
    "detect_outliers_zscore_method",
    "detect_outliers_isolation_forest",
    "winsorize_by_sector_method",
    "calculate_data_quality_score",
    "impute_missing_values",
    "impute_missing_values_knn_sector",
    "create_scaler_pipeline",
    "scale_features",
    # Data Catalog module (Phase 9.1)
    "SchemaInfo",
    "StatisticalProfile",
    "DatasetMetadata",
    "DataCatalog",
    "extract_schema_info",
    "create_statistical_profile",
    # Data Versioning module (Phase 9.1)
    "DataVersion",
    "DataVersionManager",
    "calculate_dataframe_hash",
    "compare_versions",
    "create_version_snapshot",
    # Advanced EDA module (Phase 9.2)
    "CorrelationReport",
    "StatisticalTestResult",
    "EDAReport",
    "calc_corr_matrix",
    "find_top_corr",
    "test_norm",
    "calc_skew_kurt",
    "detect_outliers_stat",
    "calc_mutual_info",
    "calc_rf_importance",
    "perform_pca",
    "calc_optimal_pca",
    "compare_sector_means",
    "compare_two_groups",
    "generate_eda_report",
    "generate_sector_comparison_report",
    # Features module
    "engineer_basic_ratios",
    "engineer_margin_features",
    "engineer_volatility_features",
    "engineer_revenue_cagr",
    "build_features_and_target",
    # Transformers module (Phase 9.1 Enhancements #2 and #5)
    "RegularizedTargetEncoder",
    "TargetEncoder",
    "FinancialRatioTransformer",
    "SafeDivisionTransformer",
    "ValuationRatioTransformer",
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
    # Advanced Models module (Phase 9.5)
    "prepare_regression_data",
    "create_classification_interactions",
    "train_ridge_regressor",
    "train_lasso_regressor",
    "train_elastic_net_regressor",
    "train_bayesian_ridge_regressor",
    "train_polynomial_regressor",
    "train_xgboost_regressor",
    "train_lightgbm_regressor",
    "train_catboost_regressor",
    "train_histgb_regressor",
    "train_random_forest_regressor",
    "train_extra_trees_regressor",
    "train_neural_network_regressor",
    "train_voting_regressor",
    "train_stacking_regressor",
    "train_quantile_regressor",
    "optimize_hyperparameters_optuna",
    "compare_regressors",
    "train_sector_specific_models",
    "save_model",
    "load_model",
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
    # Phase 9.2: Advanced EDA
    "calculate_correlation_matrix",
    "find_top_correlations",
    "test_normality",
    "calculate_skewness_kurtosis",
    "detect_outliers_statistical",
    "calculate_mutual_information",
    "calculate_feature_importance_rf",
    "calculate_shap_importance",
    "perform_pca",
    "perform_tsne",
    "perform_umap",
    "calculate_optimal_pca_components",
    "compare_sector_means",
    "compare_two_groups",
    "generate_eda_report",
    "generate_sector_comparison_report",
    # Phase 9.2: Benchmarking
    "compare_sector_distributions",
    "compare_regional_valuations",
    "find_peer_group",
    "compare_to_peers",
    "analyze_metric_trend",
    "generate_benchmarking_report",
    # Phase 9.1 Enhancement #3: Data Quality Dashboard
    "generate_data_quality_dashboard",
    "export_profiling_report",
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
