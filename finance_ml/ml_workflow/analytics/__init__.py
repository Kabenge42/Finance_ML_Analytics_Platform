"""
Analytics and stock analysis tools.

This package provides analytics functions including:
- Mispricing analysis and stock ranking (mispricing.py)
- Risk-adjusted valuation metrics
- Stock screening and filtering
- Comprehensive evaluation and reporting (eval.py)
- EDA, visualizations, and dashboard data preparation
- Enhanced HTML and Excel report generation (Phase 9.7)

Phase 9.7 - Analytics Refactor & Reporting Enhancement

Usage:
    from finance_ml.ml_workflow.analytics import (
        calculate_mispricing_score,
        rank_undervalued_stocks,
        rank_stocks_by_sector,
        simple_eda,
        export_predictions_to_excel,
        generate_pdf_report,
        # Phase 9.7 Enhanced Reports
        generate_enhanced_analysis_html,
        generate_enhanced_excel_report,
        HTMLReportConfig,
        ExcelReportConfig,
    )
"""

# Import from analyst_comparison module (Phase 9.7)
from finance_ml.ml_workflow.analytics.analyst_comparison import (
    PredictionAnalystAnalytics,
)

# HTML Reports (Phase 9.7 Enhancement)
from finance_ml.ml_workflow.analytics.html_reports import (
    generate_enhanced_analysis_html,
    generate_executive_summary_html,
    generate_sector_breakdown_html,
    generate_quality_filtered_html,
    generate_risk_warnings_html,
    generate_phase93_summary_html,
)

# Excel Reports (Phase 9.7 Enhancement)
from finance_ml.ml_workflow.analytics.excel_reports import (
    generate_enhanced_excel_report,
    create_executive_summary_sheet,
    create_quality_opportunities_sheet,
    create_sector_leaders_sheet,
    create_sector_laggards_sheet,
    create_risk_assessment_sheet,
    create_phase93_analysis_sheet,
)

# Report Configuration (Phase 9.7 Enhancement)
from finance_ml.ml_workflow.analytics.report_config import (
    HTMLReportConfig,
    ExcelReportConfig,
    REPORT_TOP_N_DEFAULT,
    QUALITY_THRESHOLD_DEFAULT,
    RISK_ZSCORE_THRESHOLD,
    DISTRESS_SCORE_THRESHOLD,
)

# Prefer focused modules where available (ongoing decomposition)
from finance_ml.ml_workflow.analytics.mispricing import (
    calculate_mispricing_score,
    calculate_mispricing_from_predictions_schema,
    calculate_risk_adjusted_mispricing,
    calculate_risk_adjusted_mispricing_from_predictions_schema,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    rank_stocks_by_sector,
)

# Import comprehensive eval module functions
from finance_ml.ml_workflow.analytics.eval import (
    # EDA & Visualizations
    simple_eda,
    # Export functions
    export_predictions_to_excel,
    export_predictions_to_csv,
    # Visualization functions
    create_sector_heatmap,
    create_interactive_prediction_plot,
    create_region_sector_heatmap,
    # Advanced EDA functions (correlations)
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
    generate_sector_comparison_report,
    # Data quality dashboard
    generate_data_quality_dashboard,
    export_profiling_report,
    # Prediction vs analyst analytics
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
    # Dashboard and reporting helpers
    create_structured_output_directory,
    generate_imputation_report,
    # Additional eval functions
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
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data,
    generate_pdf_report,
    generate_enhanced_pdf_report,
)
from finance_ml.ml_workflow.analytics.mispricing import (
    calculate_mispricing_score,
    calculate_risk_adjusted_mispricing,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    rank_stocks_by_sector,
)

# Import from portfolio module (Phase 9.7)
from finance_ml.ml_workflow.analytics.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe_ratio,
    optimize_portfolio_max_sharpe,
    optimize_portfolio_min_volatility,
)

# Import from risk module (Phase 9.7)
from finance_ml.ml_workflow.analytics.risk import (
    calculate_var_historical,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_portfolio_risk_metrics,
)

# Import ML returns functions (Portfolio Phase 2)
from finance_ml.ml_workflow.analytics.ml_returns import (
    create_ml_return_features,
    train_linear_return_predictor,
    create_ensemble_return_predictions,
    evaluate_return_predictions,
    # Phase 7.1-7.3 Enhancement functions
    clip_expected_returns,
    calculate_historical_returns,
    get_phase93_return_features,
    create_ml_return_features_enhanced,
    validate_expected_returns,
    # Phase 7.4: DNN Implementation
    build_dnn_return_predictor,
    train_dnn_return_predictor,
    train_dnn_quantile_predictor,
    # Phase 7.5: Ensemble Enhancement
    ReturnEnsemble,
    create_return_ensemble,
    create_dynamic_ensemble,
    # Phase 7.6: Black-Litterman ML Integration
    create_bl_views_from_ml,
    detect_market_regime,
    # Phase 7.7: Robust Covariance Estimation
    estimate_covariance_shrinkage,
    estimate_covariance_ewm,
    # Phase 7.8: Model Validation & Diagnostics
    calculate_return_prediction_diagnostics,
    validate_portfolio_metrics,
)

# Import configuration constants for ML returns (Section 8.1 compliance)
from finance_ml.ml_workflow.config import (
    MIN_DATES_FOR_TIMESERIES,
    MIN_DATES_FOR_RELIABLE_ML,
    MIN_PORTFOLIO_CANDIDATES,
    DEFAULT_EXPECTED_RETURN,
    TRAIN_SIZE,
    TARGET_COL,
    TARGET_COL_FALLBACK,
    LAG_PERIODS,
    TECHNICAL_INDICATORS,
)

__all__ = [
    # Mispricing functions
    "calculate_mispricing_score",
    "calculate_risk_adjusted_mispricing",
    "rank_undervalued_stocks",
    "rank_overvalued_stocks",
    "rank_stocks_by_sector",
    # Analyst comparison (Phase 9.7)
    "PredictionAnalystAnalytics",
    # Portfolio optimization (Phase 9.7)
    "calculate_portfolio_return",
    "calculate_portfolio_volatility",
    "calculate_portfolio_sharpe_ratio",
    "optimize_portfolio_max_sharpe",
    "optimize_portfolio_min_volatility",
    # Risk metrics (Phase 9.7)
    "calculate_var_historical",
    "calculate_cvar",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_portfolio_risk_metrics",
    # ML returns functions (Portfolio Phase 2)
    "create_ml_return_features",
    "train_linear_return_predictor",
    "create_ensemble_return_predictions",
    "evaluate_return_predictions",
    # Phase 7.1-7.3 Enhancement functions
    "clip_expected_returns",
    "calculate_historical_returns",
    "get_phase93_return_features",
    "create_ml_return_features_enhanced",
    "validate_expected_returns",
    # Phase 7.4: DNN Implementation
    "build_dnn_return_predictor",
    "train_dnn_return_predictor",
    "train_dnn_quantile_predictor",
    # Phase 7.5: Ensemble Enhancement
    "ReturnEnsemble",
    "create_return_ensemble",
    "create_dynamic_ensemble",
    # Phase 7.6: Black-Litterman ML Integration
    "create_bl_views_from_ml",
    "detect_market_regime",
    # Phase 7.7: Robust Covariance Estimation
    "estimate_covariance_shrinkage",
    "estimate_covariance_ewm",
    # Phase 7.8: Model Validation & Diagnostics
    "calculate_return_prediction_diagnostics",
    "validate_portfolio_metrics",
    # Configuration constants (Section 8.1)
    "MIN_DATES_FOR_TIMESERIES",
    "MIN_DATES_FOR_RELIABLE_ML",
    "MIN_PORTFOLIO_CANDIDATES",
    "DEFAULT_EXPECTED_RETURN",
    "TRAIN_SIZE",
    "TARGET_COL",
    "TARGET_COL_FALLBACK",
    "LAG_PERIODS",
    "TECHNICAL_INDICATORS",
    # EDA and analysis
    "simple_eda",
    # Export functions
    "export_predictions_to_excel",
    "export_predictions_to_csv",
    # Visualization
    "create_sector_heatmap",
    "create_interactive_prediction_plot",
    "create_region_sector_heatmap",
    # Advanced EDA
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
    "generate_sector_comparison_report",
    # Data quality
    "generate_data_quality_dashboard",
    "export_profiling_report",
    # Prediction vs analyst
    "compare_prediction_vs_analyst_targets",
    "calculate_directional_accuracy",
    "calculate_agreement_rate",
    "identify_disagreement_opportunities",
    "calculate_prediction_accuracy_metrics",
    "segment_comparison_by_attribute",
    "analyze_systematic_bias",
    "calculate_hit_rate_by_confidence_level",
    "calculate_calibration_metrics",
    "generate_prediction_analyst_excel_report",
    # Reporting
    "create_structured_output_directory",
    "generate_imputation_report",
    # Additional eval functions
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
    "calculate_financial_metrics_dashboard",
    "generate_data_quality_alerts",
    "prepare_plotly_dashboard_data",
    "generate_pdf_report",
    "generate_enhanced_pdf_report",
    # Phase 9.7 Enhanced HTML Reports
    "generate_enhanced_analysis_html",
    "generate_executive_summary_html",
    "generate_sector_breakdown_html",
    "generate_quality_filtered_html",
    "generate_risk_warnings_html",
    "generate_phase93_summary_html",
    # Phase 9.7 Enhanced Excel Reports
    "generate_enhanced_excel_report",
    "create_executive_summary_sheet",
    "create_quality_opportunities_sheet",
    "create_sector_leaders_sheet",
    "create_sector_laggards_sheet",
    "create_risk_assessment_sheet",
    "create_phase93_analysis_sheet",
    # Phase 9.7 Report Configuration
    "HTMLReportConfig",
    "ExcelReportConfig",
    "REPORT_TOP_N_DEFAULT",
    "QUALITY_THRESHOLD_DEFAULT",
    "RISK_ZSCORE_THRESHOLD",
    "DISTRESS_SCORE_THRESHOLD",
]
