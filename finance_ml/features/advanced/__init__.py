"""
Advanced feature engineering - modular implementation.
"""
from __future__ import annotations

from .comprehensive import build_comprehensive_features
from .dividends import (
    engineer_dividend_reliability_features,
)
from .earnings import (
    engineer_estimated_vs_actual_analytics,
    engineer_gaap_vs_adjusted_analytics,
)
from .employment import (
    engineer_employee_productivity_features,
    engineer_employment_dynamics_features,
)
from .growth import (
    engineer_growth_metrics,
)
from .leverage import (
    engineer_leverage_ratios,
    engineer_liquidity_ratios,
    engineer_efficiency_ratios,
    engineer_balance_sheet_trends,
)
from .momentum import (
    engineer_momentum_features,
    engineer_technical_analysis_features,
    engineer_market_microstructure_features,
)
from .profitability import (
    engineer_profitability_ratios,
    engineer_margin_trends,
)
from .quality import (
    engineer_accounting_quality_features,
    engineer_financial_distress_features,
    engineer_cash_flow_quality_features,
    engineer_capital_allocation_features,
    engineer_composite_scores,
)
from .revenue import (
    engineer_revenue_forecast_features,
)
from .sector import (
    engineer_sector_specific_features,
    engineer_sector_relative_interactions,
    create_relative_value_features,
)
from .sentiment import (
    engineer_analyst_quality_features,
    engineer_market_sentiment_features,
)
from .temporal import (
    engineer_temporal_features,
)
from .valuation import (
    engineer_valuation_ratios,
    engineer_valuation_timeseries_features,
)

# Feature Registry for Auto-discovery
FEATURE_REGISTRY = {
    "valuation": {
        "function": engineer_valuation_ratios,
        "category": "Valuation Ratios",
    },
    "profitability": {
        "function": engineer_profitability_ratios,
        "category": "Profitability",
    },
    "momentum": {
        "function": engineer_momentum_features,
        "category": "Momentum & Technical",
    },
    "accounting_quality": {
        "function": engineer_accounting_quality_features,
        "category": "Quality & Risk",
    },
    "composite_scores": {
        "function": engineer_composite_scores,
        "category": "Quality & Risk",
    },
}

def get_feature_generators():
    """Returns the registry of feature generation functions."""
    return FEATURE_REGISTRY

__all__ = [
    "engineer_valuation_ratios",
    "engineer_valuation_timeseries_features",
    "engineer_profitability_ratios",
    "engineer_margin_trends",
    "engineer_momentum_features",
    "engineer_technical_analysis_features",
    "engineer_market_microstructure_features",
    "engineer_accounting_quality_features",
    "engineer_financial_distress_features",
    "engineer_cash_flow_quality_features",
    "engineer_capital_allocation_features",
    "engineer_composite_scores",
    "engineer_estimated_vs_actual_analytics",
    "engineer_gaap_vs_adjusted_analytics",
    "engineer_employee_productivity_features",
    "engineer_employment_dynamics_features",
    "engineer_growth_metrics",
    "engineer_leverage_ratios",
    "engineer_liquidity_ratios",
    "engineer_efficiency_ratios",
    "engineer_balance_sheet_trends",
    "engineer_analyst_quality_features",
    "engineer_market_sentiment_features",
    "engineer_sector_specific_features",
    "engineer_sector_relative_interactions",
    "create_relative_value_features",
    "engineer_temporal_features",
    "engineer_dividend_reliability_features",
    "engineer_revenue_forecast_features",
    "build_comprehensive_features",
]
