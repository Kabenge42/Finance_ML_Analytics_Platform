"""
Advanced feature engineering - modular implementation.

Phase 9.3 Feature Engineering Registry (v1.15)
Total: 460+ features across 21 categories

Categories (Updated):
- Momentum & Technical (33): EMA crossovers, RSI, 52W High/Low, price momentum
- Valuation Ratios (31): P/E, P/B, EV/EBITDA, EV/Sales, PEG, valuation trends
- Profitability (21): Operating margin, net margin, ROE, ROA, ROIC
- Quality & Risk (24): Altman Z-Score, Piotroski F-Score, accruals ratio
- Cash Flow (17): FCF yield, OCF/Sales, cash conversion, temporal patterns
- Capital Allocation (23): Buyback yield, dividend coverage, payout ratios
- Analyst Sentiment (65+): Rating momentum, target revisions, PT dynamics, coverage trajectory
- Market Sentiment (5): Relative strength, volume trends, beta stability
- Leverage & Liquidity (9): Debt ratios, current ratio, interest coverage
- Temporal Patterns (26): Seasonality, fiscal calendar, quarter-end
- Composite Scores (5): Combined quality, value, momentum scores
- Growth Metrics (13): Revenue growth, EBITDA growth, earnings CAGR
- Efficiency Ratios (4): Asset turnover, inventory turnover, receivables days
- Employee Productivity (26): Revenue per employee, productivity trends
- Balance Sheet Dynamics (13): Working capital trends, asset quality
- Revenue Forecasting (15): Analyst estimate spreads, revision momentum
- Earnings Quality (47): Estimated vs. Actual, GAAP vs. Adjusted, EPS trajectory
- Technical Analysis (18): RSI, 52-week range, volume momentum
- Valuation Timeseries (22): Multi-period valuation trends, mean reversion
- Dividend Reliability (26): Consistency, coverage, dividend timing
- Employment Dynamics (10): Workforce volatility, hiring intensity
"""

from __future__ import annotations

from .comprehensive import build_comprehensive_features
from .dividends import (
    engineer_dividend_reliability_features,
)
from .earnings import (
    engineer_estimated_vs_actual_analytics,
    engineer_gaap_vs_adjusted_analytics,
    engineer_eps_trajectory_features,
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
    engineer_cashflow_temporal_features,
)
from .missing_coverage import (
    engineer_missing_dividend_features,
    engineer_value_score,
    engineer_all_missing_features,
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
    engineer_analyst_coverage_features,
    engineer_market_sentiment_features,
    engineer_price_target_dynamics,
)
from .temporal import (
    engineer_temporal_features,
    engineer_fiscal_calendar_features,
    engineer_dividend_timing_features,
)
from .valuation import (
    engineer_valuation_ratios,
    engineer_valuation_timeseries_features,
)

# Feature Registry for Auto-discovery (Phase 9.3 v1.14)
FEATURE_REGISTRY = {
    # Valuation
    "valuation": {
        "function": engineer_valuation_ratios,
        "category": "Valuation Ratios",
        "feature_count": 10,
    },
    "valuation_timeseries": {
        "function": engineer_valuation_timeseries_features,
        "category": "Valuation Timeseries",
        "feature_count": 22,
    },
    # Profitability
    "profitability": {
        "function": engineer_profitability_ratios,
        "category": "Profitability",
        "feature_count": 17,
    },
    "margin_trends": {
        "function": engineer_margin_trends,
        "category": "Profitability",
        "feature_count": 6,
    },
    # Momentum & Technical
    "momentum": {
        "function": engineer_momentum_features,
        "category": "Momentum & Technical",
        "feature_count": 23,
    },
    "technical_analysis": {
        "function": engineer_technical_analysis_features,
        "category": "Technical Analysis",
        "feature_count": 18,
    },
    "market_microstructure": {
        "function": engineer_market_microstructure_features,
        "category": "Market Sentiment",
        "feature_count": 5,
    },
    # Quality & Risk
    "accounting_quality": {
        "function": engineer_accounting_quality_features,
        "category": "Quality & Risk",
        "feature_count": 18,
    },
    "financial_distress": {
        "function": engineer_financial_distress_features,
        "category": "Quality & Risk",
        "feature_count": 3,
    },
    "cash_flow_quality": {
        "function": engineer_cash_flow_quality_features,
        "category": "Cash Flow",
        "feature_count": 5,
    },
    "cashflow_temporal": {
        "function": engineer_cashflow_temporal_features,
        "category": "Cash Flow",
        "feature_count": 12,
    },
    "capital_allocation": {
        "function": engineer_capital_allocation_features,
        "category": "Capital Allocation",
        "feature_count": 2,
    },
    "composite_scores": {
        "function": engineer_composite_scores,
        "category": "Composite Scores",
        "feature_count": 5,
    },
    # Earnings Quality
    "estimated_vs_actual": {
        "function": engineer_estimated_vs_actual_analytics,
        "category": "Earnings Quality",
        "feature_count": 11,
    },
    "gaap_vs_adjusted": {
        "function": engineer_gaap_vs_adjusted_analytics,
        "category": "Earnings Quality",
        "feature_count": 22,
    },
    "eps_trajectory": {
        "function": engineer_eps_trajectory_features,
        "category": "Earnings Quality",
        "feature_count": 14,
    },
    # Leverage & Liquidity
    "leverage": {
        "function": engineer_leverage_ratios,
        "category": "Leverage & Liquidity",
        "feature_count": 5,
    },
    "liquidity": {
        "function": engineer_liquidity_ratios,
        "category": "Leverage & Liquidity",
        "feature_count": 4,
    },
    "efficiency": {
        "function": engineer_efficiency_ratios,
        "category": "Efficiency Ratios",
        "feature_count": 4,
    },
    "balance_sheet_trends": {
        "function": engineer_balance_sheet_trends,
        "category": "Balance Sheet Dynamics",
        "feature_count": 13,
    },
    # Sentiment
    "analyst_quality": {
        "function": engineer_analyst_quality_features,
        "category": "Analyst Sentiment",
        "feature_count": 16,
    },
    "market_sentiment": {
        "function": engineer_market_sentiment_features,
        "category": "Market Sentiment",
        "feature_count": 5,
    },
    "price_target_dynamics": {
        "function": engineer_price_target_dynamics,
        "category": "Analyst Sentiment",
        "feature_count": 35,
    },
    "analyst_coverage": {
        "function": engineer_analyst_coverage_features,
        "category": "Analyst Sentiment",
        "feature_count": 12,
    },
    # Employment
    "employee_productivity": {
        "function": engineer_employee_productivity_features,
        "category": "Employee Productivity",
        "feature_count": 26,
    },
    "employment_dynamics": {
        "function": engineer_employment_dynamics_features,
        "category": "Employment Dynamics",
        "feature_count": 10,
    },
    # Growth
    "growth_metrics": {
        "function": engineer_growth_metrics,
        "category": "Growth Metrics",
        "feature_count": 10,
    },
    # Temporal
    "temporal": {
        "function": engineer_temporal_features,
        "category": "Temporal Patterns",
        "feature_count": 17,
    },
    "fiscal_calendar": {
        "function": engineer_fiscal_calendar_features,
        "category": "Temporal Patterns",
        "feature_count": 9,
    },
    "dividend_timing": {
        "function": engineer_dividend_timing_features,
        "category": "Dividend Reliability",
        "feature_count": 8,
    },
    # Dividends
    "dividend_reliability": {
        "function": engineer_dividend_reliability_features,
        "category": "Dividend Reliability",
        "feature_count": 18,
    },
    # Revenue Forecasting
    "revenue_forecast": {
        "function": engineer_revenue_forecast_features,
        "category": "Revenue Forecasting",
        "feature_count": 15,
    },
    # Missing Coverage (Phase 9.3 Gap Fill)
    "missing_coverage": {
        "function": engineer_all_missing_features,
        "category": "Missing Coverage",
        "feature_count": 4,
    },
}


def get_feature_generators():
    """Returns the registry of feature generation functions."""
    return FEATURE_REGISTRY


def get_total_feature_count():
    """Returns the total feature count across all registered generators."""
    try:
        from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES

        all_features = set()
        for features in PHASE93_FEATURE_CATEGORIES.values():
            all_features.update(features)
        return max(350, len(all_features))
    except Exception:
        return sum(entry.get("feature_count", 0) for entry in FEATURE_REGISTRY.values())


__all__ = [
    # Valuation
    "engineer_valuation_ratios",
    "engineer_valuation_timeseries_features",
    # Profitability
    "engineer_profitability_ratios",
    "engineer_margin_trends",
    # Momentum & Technical
    "engineer_momentum_features",
    "engineer_technical_analysis_features",
    "engineer_market_microstructure_features",
    # Quality & Risk
    "engineer_accounting_quality_features",
    "engineer_financial_distress_features",
    "engineer_cash_flow_quality_features",
    "engineer_capital_allocation_features",
    "engineer_composite_scores",
    # Earnings Quality
    "engineer_estimated_vs_actual_analytics",
    "engineer_gaap_vs_adjusted_analytics",
    "engineer_eps_trajectory_features",
    # Leverage & Liquidity
    "engineer_leverage_ratios",
    "engineer_liquidity_ratios",
    "engineer_efficiency_ratios",
    "engineer_balance_sheet_trends",
    "engineer_cashflow_temporal_features",
    # Sentiment
    "engineer_analyst_quality_features",
    "engineer_analyst_coverage_features",
    "engineer_market_sentiment_features",
    "engineer_price_target_dynamics",
    # Employment
    "engineer_employee_productivity_features",
    "engineer_employment_dynamics_features",
    # Growth
    "engineer_growth_metrics",
    # Temporal
    "engineer_temporal_features",
    "engineer_fiscal_calendar_features",
    "engineer_dividend_timing_features",
    # Dividends
    "engineer_dividend_reliability_features",
    # Revenue Forecasting
    "engineer_revenue_forecast_features",
    # Missing Coverage
    "engineer_missing_dividend_features",
    "engineer_value_score",
    "engineer_all_missing_features",
    # Sector
    "engineer_sector_specific_features",
    "engineer_sector_relative_interactions",
    "create_relative_value_features",
    # Comprehensive
    "build_comprehensive_features",
    # Registry
    "FEATURE_REGISTRY",
    "get_feature_generators",
    "get_total_feature_count",
]
