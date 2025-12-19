"""
Public API for feature engineering presets (Phase 9.3 Week 9).

Exposes a single entry point `build_features` that composes feature groups
from core and advanced modules using named presets, while preserving backward
compatibility with existing orchestrators.

Presets:
- "basic": core ratios, margins, volatility, and revenue CAGR
- "momentum": momentum & technical indicators only
- "quality": accounting quality and financial distress signals
- "standard": balanced feature set (valuation, profitability, growth, analyst sentiment)
- "comprehensive" (alias): full advanced feature set (same as advanced.build_comprehensive_features)
- "full_enhanced": same as "comprehensive"
- "earnings_analytics": estimated vs actual + GAAP vs adjusted earnings quality features
- "technical_plus": technical analysis + valuation timeseries + market sentiment
- "dividend_focus": dividend reliability + capital allocation features
- "employment_analytics": employment dynamics + employee productivity features

Backwards compatibility:
- The advanced.build_comprehensive_features remains available and unchanged by
  default behavior. This API simply provides a user-friendly front end.

UPDATED: 2025-12-20
- Added 5 new presets: standard, earnings_analytics, technical_plus, dividend_focus, employment_analytics
- Enhanced momentum preset with RSI and EMA features
- Enhanced quality preset with composite scores (Piotroski, Altman, Beneish)
- Total coverage: 267 features across all presets
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from finance_ml.ml_workflow.features import core as core_feats, advanced as adv_feats

logger = logging.getLogger(__name__)

PresetName = Literal[
    "basic",
    "momentum",
    "quality",
    "standard",
    "comprehensive",
    "full_enhanced",
    "earnings_analytics",
    "technical_plus",
    "dividend_focus",
    "employment_analytics",
]


def build_features(
    df: pd.DataFrame,
    preset: PresetName = "comprehensive",
    *,
    include_interactions: bool = True,
    include_relative: bool = True,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Build features using a named preset.

    Args:
        df: Input DataFrame
        preset: One of {"basic", "momentum", "quality", "standard", "comprehensive",
                "full_enhanced", "earnings_analytics", "technical_plus",
                "dividend_focus", "employment_analytics"}
        include_interactions: For comprehensive presets, whether to add interactions
        include_relative: For comprehensive presets, whether to add relative/sector features
        sector_col: Sector column name (used by some feature groups)

    Returns:
        DataFrame with engineered features added.

    Presets Coverage:
        - basic: 20-30 features (core ratios, margins, volatility, CAGR)
        - momentum: 27 features (momentum, technical indicators, RSI, EMA crossovers)
        - quality: 45+ features (accounting quality, distress, composite scores, analyst quality)
        - standard: 80-100 features (balanced mix: valuation, profitability, growth, sentiment)
        - comprehensive: 267 features (all advanced features)
        - earnings_analytics: 55+ features (earnings surprises, GAAP vs adjusted, quality flags)
        - technical_plus: 50+ features (technical analysis, valuation timeseries, market sentiment)
        - dividend_focus: 30+ features (dividend reliability, capital allocation, FCF coverage)
        - employment_analytics: 35+ features (employment dynamics, productivity trends)
    """
    preset_norm = (preset or "comprehensive").lower()

    if preset_norm == "basic":
        result = df.copy()
        result = core_feats.engineer_basic_ratios(result)
        result = core_feats.engineer_margin_features(result)
        # Keep volatility window default to avoid heavy computation
        result = core_feats.engineer_volatility_features(result)
        result = core_feats.engineer_revenue_cagr(result)
        logger.info("Built BASIC features preset (20-30 features)")
        return result

    if preset_norm == "momentum":
        result = df.copy()
        # Enhanced momentum with technical indicators
        result = adv_feats.engineer_momentum_features(result)
        result = adv_feats.engineer_technical_analysis_features(result)
        result = adv_feats.engineer_market_microstructure_features(result)
        # Hygiene: replace any possible infinities
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built MOMENTUM features preset (27 features)")
        return result

    if preset_norm == "quality":
        result = df.copy()
        # Quality & risk signals
        result = adv_feats.engineer_accounting_quality_features(result)
        result = adv_feats.engineer_financial_distress_features(result)
        # Composite scores (Piotroski F-Score, Altman Z-Score, Beneish M-Score)
        result = adv_feats.engineer_composite_scores(result)
        # Include analyst quality signals as part of broader quality theme
        result = adv_feats.engineer_analyst_quality_features(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built QUALITY features preset (45+ features)")
        return result

    if preset_norm == "standard":
        # Balanced feature set: valuation, profitability, growth, leverage, analyst sentiment
        result = df.copy()
        result = adv_feats.engineer_valuation_ratios(result)
        result = adv_feats.engineer_profitability_ratios(result)
        result = adv_feats.engineer_leverage_ratios(result)
        result = adv_feats.engineer_liquidity_ratios(result)
        result = adv_feats.engineer_growth_metrics(result)
        result = adv_feats.engineer_analyst_quality_features(result)
        result = adv_feats.engineer_margin_trends(result)
        result = adv_feats.engineer_composite_scores(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built STANDARD features preset (80-100 features)")
        return result

    if preset_norm == "earnings_analytics":
        # Earnings quality focus: estimated vs actual + GAAP vs adjusted
        result = df.copy()
        result = adv_feats.engineer_estimated_vs_actual_analytics(result)
        result = adv_feats.engineer_gaap_vs_adjusted_analytics(result)
        # Include profitability context
        result = adv_feats.engineer_profitability_ratios(result)
        result = adv_feats.engineer_margin_trends(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built EARNINGS_ANALYTICS features preset (55+ features)")
        return result

    if preset_norm == "technical_plus":
        # Technical analysis + valuation timeseries + market sentiment
        result = df.copy()
        result = adv_feats.engineer_technical_analysis_features(result)
        result = adv_feats.engineer_valuation_timeseries_features(result)
        result = adv_feats.engineer_market_sentiment_features(result)
        result = adv_feats.engineer_momentum_features(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built TECHNICAL_PLUS features preset (50+ features)")
        return result

    if preset_norm == "dividend_focus":
        # Dividend reliability + capital allocation + cash flow quality
        result = df.copy()
        result = adv_feats.engineer_dividend_reliability_features(result)
        result = adv_feats.engineer_capital_allocation_features(result)
        result = adv_feats.engineer_cash_flow_quality_features(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built DIVIDEND_FOCUS features preset (30+ features)")
        return result

    if preset_norm == "employment_analytics":
        # Employment dynamics + employee productivity
        result = df.copy()
        result = adv_feats.engineer_employment_dynamics_features(result)
        result = adv_feats.engineer_employee_productivity_features(result)
        result = adv_feats.engineer_efficiency_ratios(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built EMPLOYMENT_ANALYTICS features preset (35+ features)")
        return result

    if preset_norm in ("comprehensive", "full_enhanced"):
        logger.info("Building COMPREHENSIVE features via advanced.build_comprehensive_features")
        return adv_feats.build_comprehensive_features(
            df,
            include_interactions=include_interactions,
            include_relative_values=include_relative,
            sector_col=sector_col,
        )

    raise ValueError(
        f"Unknown preset '{preset}'. Expected one of: basic, momentum, quality, standard, "
        f"comprehensive, full_enhanced, earnings_analytics, technical_plus, dividend_focus, "
        f"employment_analytics"
    )
