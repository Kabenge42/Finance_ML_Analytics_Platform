"""
Finance ML Analytics Platform - Package

Modular package for equity screening, feature engineering, and ML regression.

Modules:
- finance_ml.data: Data loading, normalization, and validation
- finance_ml.advanced_preprocessing: Advanced preprocessing (Phase 9.1)
- finance_ml.features: Feature engineering functions
- finance_ml.regression: Classification, regression, and ensemble regression
- finance_ml.eval: Analytics, visualizations, and reporting
- finance_ml.config: Configuration management
- finance_ml.cli: Command-line interface
- finance_ml.risk_metrics: Risk metrics and portfolio risk analysis
- finance_ml.logging_config: Logging configuration and file handlers
- finance_ml.portfolio_optimization: Modern Portfolio Theory and optimization
- finance_ml.analyst_comparison: Prediction vs. Analyst analytics (Phase 9.8)
"""

__version__ = "0.4.1"

# Many legacy imports below are optional; guard them to avoid breaking basic imports
try:
    # Import from advanced_eda module (Phase 9.2)
    from finance_ml.ml_workflow.advanced_eda import (
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
except Exception:  # pragma: no cover - optional import guard
    pass

try:
    # Import from advanced_models module (Phase 9.5)
    from finance_ml.ml_workflow.advanced_models import (
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
        extract_numeric_feature_columns,
        compare_regressors,
        train_sector_specific_models,
        save_model,
        load_model,
        # Data Validation (ML Workflow Improvement Plan)
        validate_training_data,
        prepare_features_for_training,
    )
except Exception:  # pragma: no cover - optional import guard
    pass

try:
    # Import from advanced_preprocessing module (Phase 9.1)
    # Note: These are backward compatibility shims; prefer importing from preprocessing subpackage
    from finance_ml.ml_workflow.advanced_preprocessing import (
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
        # Phase 9.1: Enhanced 4-Step Imputation Strategy
        get_zero_imputation_columns,
        get_knn_imputation_columns,
        apply_zero_imputation,
        apply_knn_imputation_enhanced,
        apply_price_imputation,
        apply_median_imputation,
        apply_enhanced_imputation_strategy_4step,
        # Phase 9.5: Data Preparation Pipeline
        prepare_phase95_data,
    )
except Exception:  # pragma: no cover - optional import guard
    pass

try:
    # Import from analyst_comparison module (Phase 9.8)
    from finance_ml.ml_workflow.analyst_comparison import (
        compare_prediction_vs_analyst_targets,
        calculate_agreement_rate,
        calculate_directional_accuracy,
        analyze_systematic_bias,
        identify_disagreement_opportunities,
        segment_comparison_by_attribute,
        generate_prediction_analyst_excel_report,
        PredictionAnalystAnalytics,
    )
except Exception:  # pragma: no cover - optional import guard
    pass

# Phase 9.7 Refactor: Import from analytics subpackage
from finance_ml.ml_workflow.analytics import (
    calculate_mispricing_score as analytics_calculate_mispricing,
    calculate_risk_adjusted_mispricing as analytics_risk_adjusted_mispricing,
    rank_undervalued_stocks as analytics_rank_undervalued,
    rank_overvalued_stocks as analytics_rank_overvalued,
    rank_stocks_by_sector as analytics_rank_by_sector,
    )
# Phase 9.4 Refactor: Import from new classification subpackage
from finance_ml.ml_workflow.classification.labels import (
    create_enhanced_event_labels as classification_create_enhanced_event_labels,
    )
from finance_ml.ml_workflow.classification.tuning import (
    optimize_classifier_hyperparameters as classification_optimize_hyperparameters,
    cross_validate_with_sector_stratification as classification_cross_validate_sector,
    )
# Phase 9.6 Refactor: Import from evaluation subpackage
from finance_ml.ml_workflow.evaluation import (
    comprehensive_regression_metrics as evaluation_comprehensive_metrics,
    compute_metrics_by_segment as evaluation_metrics_by_segment,
    compute_sector_region_metrics as evaluation_sector_region_metrics,
    )
# Phase 9.3 Refactor: Import from features.advanced (new schema-driven features)
from finance_ml.ml_workflow.features.advanced import (
    engineer_valuation_ratios,
    engineer_profitability_ratios,
    engineer_leverage_ratios,
    engineer_liquidity_ratios,
    engineer_efficiency_ratios,
    engineer_growth_metrics,
    engineer_sector_specific_features,
    engineer_analyst_quality_features,
    engineer_accounting_quality_features,
    engineer_employee_productivity_features,
    build_comprehensive_features as features_build_comprehensive,
    )
# Phase 9.3 API: Import from features.api (public API with presets)
from finance_ml.ml_workflow.features.api import (
    build_features,
    PresetName,
    )
# Phase 9.3 Refactor: Import from new features subpackage
from finance_ml.ml_workflow.features.core import (
    _safe_div as features_safe_div,
    engineer_basic_ratios as features_basic_ratios,
    engineer_margin_features as features_margin_features,
    engineer_volatility_features as features_volatility_features,
    engineer_revenue_cagr as features_revenue_cagr,
    build_features_and_target as features_build_features_and_target,
    )
# Phase 9.3 Refactor: Import from features.selection
from finance_ml.ml_workflow.features.selection import (
    calculate_feature_importance_mutual_info as features_importance_mi,
    calculate_feature_importance_rf as features_importance_rf,
    calculate_feature_importance_shap as features_importance_shap,
    calculate_feature_importance_rfe as features_importance_rfe,
    )
# Phase 9.1 Refactor: Import from new preprocessing subpackage
from finance_ml.ml_workflow.preprocessing import (
    # Quality module
    DataQualityReport as PreprocessingDataQualityReport,
    calculate_data_quality_score as preprocessing_calculate_quality,
    # Pipeline module
    prepare_phase91_data,
    )
# Phase 9.5.0 Refactor: Import from new regression subpackage
from finance_ml.ml_workflow.regression.constraints import (
    NonNegativeRegressionWrapper as regression_nonnegative_wrapper,
    )
from finance_ml.ml_workflow.regression.dataset import (
    # Classification feature integration
    extract_classification_features as regression_extract_classification_features,
    integrate_classification_features_into_dataframe as regression_integrate_classification_features,
    create_classification_interactions as regression_create_classification_interactions,
    # Data preparation
    prepare_regression_data as regression_prepare_data,
    # Validation
    validate_training_data as regression_validate_training_data,
    prepare_features_for_training as regression_prepare_features_for_training,
    extract_numeric_feature_columns as regression_extract_numeric_features,
    # Sector-specific training
    train_sector_specific_models as regression_train_sector_models,
    )
from finance_ml.ml_workflow.regression.io import (
    save_model as regression_save_model,
    load_model as regression_load_model,
    )
# Phase 9.5.1 Refactor: Import model training, quantile, tuning, and I/O functions
from finance_ml.ml_workflow.regression.models import (
    train_ridge_regressor as regression_train_ridge,
    train_lasso_regressor as regression_train_lasso,
    train_elastic_net_regressor as regression_train_elastic_net,
    train_xgboost_regressor as regression_train_xgboost,
    train_lightgbm_regressor as regression_train_lightgbm,
    train_catboost_regressor as regression_train_catboost,
    train_random_forest_regressor as regression_train_random_forest,
    train_extra_trees_regressor as regression_train_extra_trees,
    train_neural_network_regressor as regression_train_neural_network,
    train_voting_regressor as regression_train_voting,
    train_stacking_regressor as regression_train_stacking,
    compare_regressors as regression_compare_regressors,
    )
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor as regression_train_quantile,
    )
from finance_ml.ml_workflow.regression.tuning import (
    optimize_hyperparameters_optuna as regression_optimize_hyperparameters,
    )
# Phase 9.8 Refactor: Import from reporting subpackage
from finance_ml.ml_workflow.reporting import (
    calculate_financial_metrics_dashboard as reporting_financial_metrics,
    generate_data_quality_alerts as reporting_quality_alerts,
    prepare_plotly_dashboard_data as reporting_plotly_data,
    )

# Import from classification_enhanced module (Phase 2.1)
# Note: These are now also available from classification.tuning (Phase 9.4)
try:
    from finance_ml.ml_workflow.classification_enhanced import (
        optimize_classifier_hyperparameters,
        cross_validate_with_sector_stratification,
        analyze_calibration,
    )

    HAVE_CLASSIFICATION_ENHANCED = True
except ImportError:
    HAVE_CLASSIFICATION_ENHANCED = False

# Import from benchmarking module (Phase 9.2)
from finance_ml.ml_workflow.benchmarking import (
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
from finance_ml.ml_workflow.data import (
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
from finance_ml.ml_workflow.data_catalog import (
    SchemaInfo,
    StatisticalProfile,
    DatasetMetadata,
    DataCatalog,
    extract_schema_info,
    create_statistical_profile,
)

# Import from data_versioning module (Phase 9.1 - Data Versioning)
from finance_ml.ml_workflow.data_versioning import (
    DataVersion,
    DataVersionManager,
    calculate_dataframe_hash,
    compare_versions,
    create_version_snapshot,
)

# Import from eval module (Phase 7 TDD implementation complete)
# Updated path: eval.py moved to analytics/eval.py (Phase 9.7)
from finance_ml.ml_workflow.analytics.eval import (
    calculate_mispricing_score,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    rank_stocks_by_sector,
    simple_eda,
    export_predictions_to_excel,
    export_predictions_to_csv,
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
    # NOTE: generate_eda_report is imported from advanced_eda (line 38) - do not re-import from eval
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
    calculate_hit_rate_by_confidence_level,
    calculate_calibration_metrics,
    generate_prediction_analyst_excel_report,
    # Phase 1: Interactive Dashboards - Reporting & Visualization Improvements
    create_structured_output_directory,
    generate_imputation_report,
)

try:
    from finance_ml.ml_workflow.features import (
        _safe_div,
        engineer_basic_ratios,
        engineer_margin_features,
        engineer_volatility_features,
        engineer_revenue_cagr,
        build_features_and_target,
    )
except ImportError:
    # Features subpackage not fully implemented yet
    _safe_div = None
    engineer_basic_ratios = None
    engineer_margin_features = None
    engineer_volatility_features = None
    engineer_revenue_cagr = None
    build_features_and_target = None

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

# Import from regression module (Phase 7 TDD implementation complete)
from finance_ml.ml_workflow.models import (
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
from finance_ml.ml_workflow.portfolio_optimization import (
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
from finance_ml.ml_workflow.risk_metrics import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_portfolio_risk_metrics,
)

# Import from transformers module (Phase 9.1 Enhancements #2 and #5)
from finance_ml.ml_workflow.transformers import (
    RegularizedTargetEncoder,
    TargetEncoder,
    FinancialRatioTransformer,
    SafeDivisionTransformer,
    ValuationRatioTransformer,
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
    # Phase 9.1: Enhanced 4-Step Imputation Strategy
    "get_zero_imputation_columns",
    "get_knn_imputation_columns",
    "apply_zero_imputation",
    "apply_knn_imputation_enhanced",
    "apply_price_imputation",
    "apply_median_imputation",
    "apply_enhanced_imputation_strategy_4step",
    # Phase 9.5: Data Preparation Pipeline
    "prepare_phase95_data",
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
    "validate_training_data",
    "prepare_features_for_training",
    # Evaluation module
    "calculate_mispricing_score",
    "rank_undervalued_stocks",
    "rank_overvalued_stocks",
    "rank_stocks_by_sector",
    "simple_eda",
    "export_predictions_to_excel",
    "export_predictions_to_csv",
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
    # Phase 9.8: Prediction vs. Analyst Analytics
    "PredictionAnalystAnalytics",
    "compare_prediction_vs_analyst_targets",
    "calculate_agreement_rate",
    "calculate_directional_accuracy",
    "analyze_systematic_bias",
    "identify_disagreement_opportunities",
    "segment_comparison_by_attribute",
    "calculate_prediction_accuracy_metrics",
    "calculate_hit_rate_by_confidence_level",
    "calculate_calibration_metrics",
    "generate_prediction_analyst_excel_report",
    "create_structured_output_directory",
    "generate_imputation_report",
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
    # Phase 9.3: Feature Engineering
    "features_safe_div",
    "features_basic_ratios",
    "features_margin_features",
    "features_volatility_features",
    "features_revenue_cagr",
    "features_build_features_and_target",
    "engineer_valuation_ratios",
    "engineer_profitability_ratios",
    "engineer_leverage_ratios",
    "engineer_liquidity_ratios",
    "engineer_efficiency_ratios",
    "engineer_growth_metrics",
    "engineer_sector_specific_features",
    "engineer_analyst_quality_features",
    "engineer_accounting_quality_features",
    "engineer_employee_productivity_features",
    "features_build_comprehensive",
    "features_importance_mi",
    "features_importance_rf",
    "features_importance_shap",
    "features_importance_rfe",
    # Phase 9.3 API: Feature Engineering with Presets
    "build_features",
    "PresetName",
    # Phase 9.4: Classification
    "classification_create_enhanced_event_labels",
    "classification_optimize_hyperparameters",
    "classification_cross_validate_sector",
    # Phase 9.5: Regression
    "regression_nonnegative_wrapper",
    "regression_extract_classification_features",
    "regression_integrate_classification_features",
    "regression_create_classification_interactions",
    "regression_prepare_features_for_training",
    "regression_prepare_data",
    "regression_validate_training_data",
    "regression_extract_numeric_features",
    "regression_train_sector_models",
    "regression_train_ridge",
    "regression_train_lasso",
    "regression_train_elastic_net",
    "regression_train_xgboost",
    "regression_train_lightgbm",
    "regression_train_catboost",
    "regression_train_random_forest",
    "regression_train_extra_trees",
    "regression_train_neural_network",
    "regression_train_voting",
    "regression_train_stacking",
    "regression_compare_regressors",
    "regression_train_quantile",
    "regression_optimize_hyperparameters",
    "regression_save_model",
    "regression_load_model",
    # Phase 9.6: Evaluation
    "evaluation_comprehensive_metrics",
    "evaluation_metrics_by_segment",
    "evaluation_sector_region_metrics",
    # Phase 9.7: Analytics
    "analytics_calculate_mispricing",
    "analytics_risk_adjusted_mispricing",
    "analytics_rank_undervalued",
    "analytics_rank_overvalued",
    "analytics_rank_by_sector",
    # Phase 9.8: Reporting
    "reporting_financial_metrics",
    "reporting_quality_alerts",
    "reporting_plotly_data",
]

# Conditionally add enhanced classification functions
if HAVE_CLASSIFICATION_ENHANCED:
    __all__.extend(
        [
            "optimize_classifier_hyperparameters",
            "cross_validate_with_sector_stratification",
            "analyze_calibration",
        ]
    )

# Additional exports from eval module (dashboard helpers and advanced analytics)
# Updated path: eval.py moved to analytics/eval.py (Phase 9.7)
from finance_ml.ml_workflow.analytics.eval import (
    calculate_risk_adjusted_mispricing,
    plot_outlier_boxplots,
    plot_outlier_violins,
    plot_outlier_scatter,
    calculate_distance_correlation,
    comprehensive_regression_metrics,
    compute_metrics_by_segment,
    residual_analysis_suite,
    error_bucketing_analysis,
    create_stratified_sector_cv,
    create_grouped_ticker_cv,
    evaluate_with_cross_validation,
    compute_shap_values,
    create_shap_summary_plot,
    create_shap_waterfall_plot,
    create_shap_dependence_plot,
    analyze_shap_by_sector,
    explain_with_lime,
    compare_lime_shap_consistency,
    create_model_comparison_table,
    statistical_model_comparison,
    automated_model_selection,
    generate_learning_curve,
    plot_learning_curve,
    generate_validation_curve,
    plot_validation_curve,
    diagnose_bias_variance,
    bias_variance_decomposition,
    plot_bias_variance,
    identify_optimal_complexity,
    create_expanding_window_cv,
    create_rolling_window_cv,
    evaluate_with_time_series_cv,
    compute_sector_region_metrics,
    create_sector_region_performance_heatmap,
    plot_residuals_vs_features,
    identify_systematic_bias_patterns,
    analyze_residual_homoscedasticity,
    compute_permutation_importance,
    rank_features_by_importance,
    feature_importance_stability_across_folds,
    assign_valuation_category,
    get_sector_specific_thresholds,
    calculate_sector_zscores,
    calculate_percentile_ranks,
    calculate_multi_factor_score,
    identify_sector_leaders_laggards,
    filter_stocks_by_criteria,
    create_valuation_scatter_plot,
    generate_pdf_report,
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data,
    perform_comprehensive_hypothesis_tests,
    test_market_efficiency_hypothesis,
    prepare_interactive_dashboard_data,
    apply_dashboard_filters,
    calculate_peer_comparisons,
    perform_time_series_hypothesis_tests,
    perform_multi_factor_anova,
    correct_outliers_with_validation,
    generate_enhanced_pdf_report,
)

__all__.extend(
    [
        "calculate_risk_adjusted_mispricing",
        "plot_outlier_boxplots",
        "plot_outlier_violins",
        "plot_outlier_scatter",
        "calculate_distance_correlation",
        "comprehensive_regression_metrics",
        "compute_metrics_by_segment",
        "residual_analysis_suite",
        "error_bucketing_analysis",
        "create_stratified_sector_cv",
        "create_grouped_ticker_cv",
        "evaluate_with_cross_validation",
        "compute_shap_values",
        "create_shap_summary_plot",
        "create_shap_waterfall_plot",
        "create_shap_dependence_plot",
        "analyze_shap_by_sector",
        "explain_with_lime",
        "compare_lime_shap_consistency",
        "create_model_comparison_table",
        "statistical_model_comparison",
        "automated_model_selection",
        "generate_learning_curve",
        "plot_learning_curve",
        "generate_validation_curve",
        "plot_validation_curve",
        "diagnose_bias_variance",
        "bias_variance_decomposition",
        "plot_bias_variance",
        "identify_optimal_complexity",
        "create_expanding_window_cv",
        "create_rolling_window_cv",
        "evaluate_with_time_series_cv",
        "compute_sector_region_metrics",
        "create_sector_region_performance_heatmap",
        "plot_residuals_vs_features",
        "identify_systematic_bias_patterns",
        "analyze_residual_homoscedasticity",
        "compute_permutation_importance",
        "rank_features_by_importance",
        "feature_importance_stability_across_folds",
        "assign_valuation_category",
        "get_sector_specific_thresholds",
        "calculate_sector_zscores",
        "calculate_percentile_ranks",
        "calculate_multi_factor_score",
        "identify_sector_leaders_laggards",
        "filter_stocks_by_criteria",
        "create_valuation_scatter_plot",
        "generate_pdf_report",
        "calculate_financial_metrics_dashboard",
        "generate_data_quality_alerts",
        "prepare_plotly_dashboard_data",
        "perform_comprehensive_hypothesis_tests",
        "test_market_efficiency_hypothesis",
        "prepare_interactive_dashboard_data",
        "apply_dashboard_filters",
        "calculate_peer_comparisons",
        "perform_time_series_hypothesis_tests",
        "perform_multi_factor_anova",
        "correct_outliers_with_validation",
        "generate_enhanced_pdf_report",
        # Phase 9.1 Refactor: New preprocessing subpackage exports
        "PreprocessingDataQualityReport",
        "preprocessing_calculate_quality",
        "prepare_phase91_data",
        # Phase 9.5.0 Refactor: New regression subpackage exports
        "regression_nonnegative_wrapper",
        "regression_extract_classification_features",
        "regression_integrate_classification_features",
        "regression_create_classification_interactions",
        "regression_prepare_data",
        "regression_validate_training_data",
        "regression_prepare_features_for_training",
        "regression_extract_numeric_features",
        "regression_train_sector_models",
        # Phase 9.5.1 Refactor: Model training, quantile, tuning, I/O exports
        "regression_train_ridge",
        "regression_train_lasso",
        "regression_train_elastic_net",
        "regression_train_xgboost",
        "regression_train_lightgbm",
        "regression_train_catboost",
        "regression_train_random_forest",
        "regression_train_extra_trees",
        "regression_train_neural_network",
        "regression_train_voting",
        "regression_train_stacking",
        "regression_train_quantile",
        "regression_compare_regressors",
        "regression_optimize_hyperparameters",
        "regression_save_model",
        "regression_load_model",
        # Phase 9.6 Refactor: Evaluation subpackage exports
        "evaluation_comprehensive_metrics",
        "evaluation_metrics_by_segment",
        "evaluation_sector_region_metrics",
        # Phase 9.7 Refactor: Analytics subpackage exports
        "analytics_calculate_mispricing",
        "analytics_risk_adjusted_mispricing",
        "analytics_rank_undervalued",
        "analytics_rank_overvalued",
        "analytics_rank_by_sector",
        # Phase 9.8 Refactor: Reporting subpackage exports
        "reporting_financial_metrics",
        "reporting_quality_alerts",
        "reporting_plotly_data",
    ]
)
