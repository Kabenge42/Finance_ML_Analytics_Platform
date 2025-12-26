"""
finance_ml.ml_workflow.features - Feature engineering subpackage

This package provides comprehensive feature engineering for financial data:

Modules:
- core: Basic financial features (ratios, margins, volatility, CAGR)
- advanced: Advanced features (sector-specific, interactions, relative values, analyst quality, accounting quality, employee productivity)
- selection: Feature importance and selection methods (mutual info, RF, SHAP, RFE)

Phase 9.3 refactor: Consolidated and enhanced from features.py and advanced_features.py.
"""

# Advanced features - importing from new modular location
from finance_ml.features.advanced import (
    engineer_valuation_ratios,
    engineer_valuation_timeseries_features,
    engineer_profitability_ratios,
    engineer_technical_analysis_features,
    engineer_market_microstructure_features,
    engineer_accounting_quality_features,
    engineer_employee_productivity_features,
    engineer_employment_dynamics_features,
    engineer_growth_metrics,
    engineer_leverage_ratios,
    engineer_liquidity_ratios,
    engineer_efficiency_ratios,
    engineer_analyst_quality_features,
    engineer_sector_specific_features,
    create_relative_value_features,
    engineer_temporal_features,
    engineer_dividend_reliability_features,
    engineer_revenue_forecast_features,
    build_comprehensive_features,
)

# Utils
from finance_ml.features.advanced.utils import (
    engineer_nonlinear_transforms,
    create_feature_interactions,
)

# Core features
from finance_ml.ml_workflow.features.core import (
    preprocess_for_lightgbm,
    _safe_div,
    engineer_basic_ratios,
    engineer_margin_features,
    engineer_volatility_features,
    engineer_revenue_cagr,
    build_features_and_target,
    # Aliases for code_guidelines.md v1.10 API
    build_valuation_features,
    build_momentum_features,
    build_quality_features,
)

# Feature selection
from finance_ml.ml_workflow.features.selection import (
    calculate_feature_importance_mutual_info,
    calculate_feature_importance_rf,
    calculate_feature_importance_shap,
    calculate_feature_importance_rfe,
    # Alias for code_guidelines.md v1.10 API
    select_features_rf,
)

# Phase 9.3 — Feature validation and pruning utilities
from finance_ml.ml_workflow.features.validation import (
    validate_feature_coverage,
    prune_low_importance_features,
    save_feature_list,
)

# ============================================================================
# Prefixed aliases for notebook compatibility
# ============================================================================
features_importance_rf = calculate_feature_importance_rf

__all__ = [
    # Core features (from core.py)
    "preprocess_for_lightgbm",
    "_safe_div",
    "engineer_basic_ratios",
    "engineer_margin_features",
    "engineer_volatility_features",
    "engineer_revenue_cagr",
    "build_features_and_target",
    # Aliases for code_guidelines.md v1.10 API (from core.py)
    "build_valuation_features",
    "build_momentum_features",
    "build_quality_features",
    # Advanced features (from advanced.py)
    "engineer_valuation_ratios",
    "engineer_profitability_ratios",
    "engineer_leverage_ratios",
    "engineer_liquidity_ratios",
    "engineer_efficiency_ratios",
    "engineer_growth_metrics",
    "engineer_sector_specific_features",
    "engineer_temporal_features",
    "engineer_market_microstructure_features",
    "engineer_nonlinear_transforms",
    "create_feature_interactions",
    "create_relative_value_features",
    "engineer_analyst_quality_features",
    "engineer_accounting_quality_features",
    "engineer_employee_productivity_features",
    "build_comprehensive_features",
    # Phase 9.3 Schema 1.3 feature engineering functions
    "engineer_technical_analysis_features",
    "engineer_valuation_timeseries_features",
    "engineer_revenue_forecast_features",
    "engineer_dividend_reliability_features",
    "engineer_employment_dynamics_features",
    # Feature selection (from selection.py)
    "calculate_feature_importance_mutual_info",
    "calculate_feature_importance_rf",
    "calculate_feature_importance_shap",
    "calculate_feature_importance_rfe",
    # Alias for code_guidelines.md v1.10 API (from selection.py)
    "select_features_rf",
    # Prefixed alias for notebook compatibility
    "features_importance_rf",
    # Feature validation/pruning (from validation.py)
    "validate_feature_coverage",
    "prune_low_importance_features",
    "save_feature_list",
]
