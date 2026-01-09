"""Comprehensive feature engineering orchestrator."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .earnings import engineer_estimated_vs_actual_analytics, engineer_gaap_vs_adjusted_analytics
from .employment import engineer_employee_productivity_features, engineer_employment_dynamics_features
from .growth import engineer_growth_metrics
from .leverage import (
    engineer_leverage_ratios,
    engineer_liquidity_ratios,
    engineer_efficiency_ratios,
    engineer_balance_sheet_trends
)
from .momentum import (
    engineer_momentum_features,
    engineer_technical_analysis_features,
    engineer_market_microstructure_features
)
from .profitability import engineer_profitability_ratios, engineer_margin_trends
from .quality import (
    engineer_accounting_quality_features,
    engineer_financial_distress_features,
    engineer_cash_flow_quality_features,
    engineer_capital_allocation_features,
    engineer_composite_scores
)
from .revenue import engineer_revenue_forecast_features
from .sector import engineer_sector_specific_features, create_relative_value_features
from .sentiment import engineer_analyst_quality_features, engineer_market_sentiment_features
from .temporal import engineer_temporal_features
from .utils import engineer_nonlinear_transforms, create_feature_interactions
from .valuation import engineer_valuation_ratios, engineer_valuation_timeseries_features
from .dividends import engineer_dividend_reliability_features
from .missing_coverage import engineer_all_missing_features

logger = logging.getLogger(__name__)

def build_comprehensive_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_relative_values: bool = True,
    sector_col: str = "sector",
    preset: Optional[str] = None,
) -> pd.DataFrame:
    """Build feature sets by applying advanced feature engineering functions."""
    preset_norm = (preset or "comprehensive").lower()

    if preset_norm == "momentum":
        result = engineer_momentum_features(df.copy())
        return result.replace([np.inf, -np.inf], np.nan)

    if preset_norm == "quality":
        result = df.copy()
        result = engineer_accounting_quality_features(result)
        result = engineer_financial_distress_features(result)
        result = engineer_analyst_quality_features(result)
        return result.replace([np.inf, -np.inf], np.nan)

    # Default comprehensive path
    result = df.copy()

    # Apply all feature engineering functions in sequence
    result = engineer_valuation_ratios(result)
    result = engineer_profitability_ratios(result)
    result = engineer_margin_trends(result)
    result = engineer_leverage_ratios(result)
    result = engineer_liquidity_ratios(result)
    result = engineer_efficiency_ratios(result)
    result = engineer_growth_metrics(result)
    result = engineer_momentum_features(result)
    result = engineer_sector_specific_features(result, sector_col=sector_col)
    result = engineer_analyst_quality_features(result)
    result = engineer_market_sentiment_features(result)
    result = engineer_market_microstructure_features(result)
    result = engineer_accounting_quality_features(result)
    result = engineer_financial_distress_features(result)
    result = engineer_cash_flow_quality_features(result)
    result = engineer_capital_allocation_features(result)
    result = engineer_employee_productivity_features(result)
    result = engineer_balance_sheet_trends(result)

    result = engineer_technical_analysis_features(result)
    result = engineer_valuation_timeseries_features(result)
    result = engineer_revenue_forecast_features(result)
    result = engineer_dividend_reliability_features(result)
    result = engineer_all_missing_features(result)
    result = engineer_employment_dynamics_features(result)
    result = engineer_estimated_vs_actual_analytics(result)
    result = engineer_gaap_vs_adjusted_analytics(result)

    # Temporal features
    date_col_candidates = ["next_earnings", "last_updated", "income_statement_report_date"]
    effective_date_col = next((c for c in date_col_candidates if c in result.columns), None)
    if effective_date_col:
        result = engineer_temporal_features(result, date_col=effective_date_col)

    # Nonlinear transforms
    log_cols = ["market_cap", "revenue", "total_assets"]
    result = engineer_nonlinear_transforms(result, log_features=[c for c in log_cols if c in result.columns])

    # Optional components
    if include_interactions:
        result = create_feature_interactions(result)
    if include_relative_values:
        result = create_relative_value_features(result, sector_col=sector_col)

    # Composite scores
    result = engineer_composite_scores(result)

    # Final numeric hygiene
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Built comprehensive features: {len(result.columns)} total features")
    return result
