"""
finance_ml.ml_workflow.features - Feature engineering subpackage

This package provides comprehensive feature engineering for financial data:

Modules:
- core: Basic financial features (ratios, margins, volatility, CAGR)
- advanced: Advanced features (sector-specific, interactions, relative values, analyst quality, accounting quality, employee productivity)
- selection: Feature importance and selection methods (mutual info, RF, SHAP, RFE)

Phase 9.3 refactor: Consolidated and enhanced from features.py and advanced_features.py.
"""

# Core features
from finance_ml.ml_workflow.features.core import (
    preprocess_for_lightgbm,
    _safe_div,
    engineer_basic_ratios,
    engineer_margin_features,
    engineer_volatility_features,
    engineer_revenue_cagr,
    build_features_and_target,
)

# Advanced features
from finance_ml.ml_workflow.features.advanced import (
    engineer_valuation_ratios,
    engineer_profitability_ratios,
    engineer_leverage_ratios,
    engineer_liquidity_ratios,
    engineer_efficiency_ratios,
    engineer_growth_metrics,
    engineer_sector_specific_features,
    engineer_temporal_features,
    engineer_market_microstructure_features,
    engineer_nonlinear_transforms,
    create_feature_interactions,
    create_relative_value_features,
    engineer_analyst_quality_features,
    engineer_accounting_quality_features,
    engineer_employee_productivity_features,
    build_comprehensive_features,
)

# Feature selection
from finance_ml.ml_workflow.features.selection import (
    calculate_feature_importance_mutual_info,
    calculate_feature_importance_rf,
    calculate_feature_importance_shap,
    calculate_feature_importance_rfe,
)

__all__ = [
    # Core features (from core.py)
    "preprocess_for_lightgbm",
    "_safe_div",
    "engineer_basic_ratios",
    "engineer_margin_features",
    "engineer_volatility_features",
    "engineer_revenue_cagr",
    "build_features_and_target",
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
    # Feature selection (from selection.py)
    "calculate_feature_importance_mutual_info",
    "calculate_feature_importance_rf",
    "calculate_feature_importance_shap",
    "calculate_feature_importance_rfe",
]
