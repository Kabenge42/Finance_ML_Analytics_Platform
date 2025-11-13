"""
Analytics and stock analysis tools.

This package provides analytics functions including:
- Mispricing analysis and stock ranking (mispricing.py)
- Risk-adjusted valuation metrics
- Stock screening and filtering
- Comprehensive evaluation and reporting (eval.py)
- EDA, visualizations, and dashboard data preparation

Phase 9.7 - Analytics Refactor

Usage:
    from finance_ml.ml_workflow.analytics import (
        calculate_mispricing_score,
        rank_undervalued_stocks,
        rank_stocks_by_sector,
        simple_eda,
        export_predictions_to_excel,
        generate_pdf_report
    )
"""

# Import from analyst_comparison module (Phase 9.7)
from finance_ml.ml_workflow.analytics.analyst_comparison import (
    PredictionAnalystAnalytics,
)

# Import comprehensive eval module functions
from finance_ml.ml_workflow.analytics.eval import (
    # Basic mispricing and ranking (note: some overlap with mispricing.py)
    simple_eda,
    # Export functions
    export_predictions_to_excel,
    export_predictions_to_csv,
    # Visualization functions
    create_sector_heatmap,
    create_interactive_prediction_plot,
    create_region_sector_heatmap,
    # Advanced EDA functions
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
]
