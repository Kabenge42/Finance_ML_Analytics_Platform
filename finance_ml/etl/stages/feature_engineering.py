"""Feature Engineering stage for ETL."""

import logging
from typing import List, Optional, TYPE_CHECKING

import pandas as pd

from finance_ml.ml_workflow.features.api import build_features

if TYPE_CHECKING:
    from finance_ml.etl.config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)

# Mapping from config flag names to FEATURE_REGISTRY keys
CONFIG_FLAG_TO_REGISTRY_KEY = {
    # Valuation
    "engineer_valuation": "valuation",
    "engineer_valuation_timeseries": "valuation_timeseries",
    # Profitability
    "engineer_profitability": "profitability",
    "engineer_margin_trends": "margin_trends",
    # Momentum & Technical
    "engineer_momentum": "momentum",
    "engineer_technical_analysis": "technical_analysis",
    "engineer_market_microstructure": "market_microstructure",
    # Quality & Risk
    "engineer_accounting_quality": "accounting_quality",
    "engineer_financial_distress": "financial_distress",
    "engineer_cash_flow_quality": "cash_flow_quality",
    "engineer_capital_allocation": "capital_allocation",
    "engineer_composite_scores": "composite_scores",
    # Earnings Quality
    "engineer_estimated_vs_actual": "estimated_vs_actual",
    "engineer_gaap_vs_adjusted": "gaap_vs_adjusted",
    "engineer_eps_trajectory": "eps_trajectory",
    # Leverage & Liquidity
    "engineer_leverage": "leverage",
    "engineer_liquidity": "liquidity",
    "engineer_efficiency": "efficiency",
    "engineer_balance_sheet_trends": "balance_sheet_trends",
    "engineer_cashflow_temporal": "cashflow_temporal",
    # Sentiment
    "engineer_analyst_quality": "analyst_quality",
    "engineer_analyst_coverage": "analyst_coverage",
    "engineer_market_sentiment": "market_sentiment",
    "engineer_price_target_dynamics": "price_target_dynamics",
    # Employment
    "engineer_employee_productivity": "employee_productivity",
    "engineer_employment_dynamics": "employment_dynamics",
    # Growth
    "engineer_growth_metrics": "growth_metrics",
    # Temporal
    "engineer_temporal": "temporal",
    "engineer_fiscal_calendar": "fiscal_calendar",
    "engineer_dividend_timing": "dividend_timing",
    # Dividends
    "engineer_dividend_reliability": "dividend_reliability",
    # Revenue Forecasting
    "engineer_revenue_forecast": "revenue_forecast",
    # Missing Coverage
    "engineer_missing_coverage": "missing_coverage",
}


def run_feature_engineering_stage(
    df: pd.DataFrame,
    config: Optional["FeatureEngineeringConfig"] = None,
    preset: str = "comprehensive",
    categories: Optional[List[str]] = None,
    engineer_earnings_analytics: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Stage 9: Feature engineering.

    Args:
        df: Input DataFrame with preprocessed data.
        config: FeatureEngineeringConfig object. If provided, takes precedence over kwargs.
        preset: Feature preset name ('basic', 'momentum', 'quality', 'comprehensive').
        categories: Optional list of specific feature categories to generate.
        engineer_earnings_analytics: Legacy flag for earnings analytics (deprecated).
        **kwargs: Legacy kwargs for backward compatibility.

    Returns:
        DataFrame with engineered features.
    """
    # Use config values if provided, otherwise fall back to function arguments
    if config is not None:
        preset = config.preset
        categories = config.categories
        engineer_earnings_analytics = config.engineer_earnings_analytics

    logger.info(f"Stage 9: Building features using preset: {preset}")

    # Apply base feature preset
    result = build_features(df, preset=preset)

    # For comprehensive/full_enhanced presets, all features are included
    if preset in ("comprehensive", "full_enhanced"):
        return result

    # Apply granular feature generators based on config or kwargs
    from finance_ml.features import advanced as adv

    # Get enabled generators from config or kwargs
    enabled_generators = _get_enabled_generators(
        config, kwargs, engineer_earnings_analytics
    )

    # Apply each enabled generator
    for registry_key in enabled_generators:
        result = _apply_feature_generator(result, registry_key, adv)

    return result


def _get_enabled_generators(
    config: Optional["FeatureEngineeringConfig"],
    kwargs: dict,
    engineer_earnings_analytics: bool,
) -> List[str]:
    """Determine which feature generators to apply based on config or kwargs.

    Args:
        config: Optional FeatureEngineeringConfig instance.
        kwargs: Legacy keyword arguments.
        engineer_earnings_analytics: Legacy earnings analytics flag.

    Returns:
        List of FEATURE_REGISTRY keys to apply.
    """
    enabled = []

    if config is not None:
        # Use config object - iterate through config flags
        for flag_name, registry_key in CONFIG_FLAG_TO_REGISTRY_KEY.items():
            if getattr(config, flag_name, False):
                enabled.append(registry_key)

        # Handle legacy earnings_analytics flag
        if config.engineer_earnings_analytics:
            if "estimated_vs_actual" not in enabled:
                enabled.append("estimated_vs_actual")
            if "gaap_vs_adjusted" not in enabled:
                enabled.append("gaap_vs_adjusted")
    else:
        # Fall back to kwargs for backward compatibility
        if engineer_earnings_analytics:
            enabled.extend(["estimated_vs_actual", "gaap_vs_adjusted"])

        for flag_name, registry_key in CONFIG_FLAG_TO_REGISTRY_KEY.items():
            if kwargs.get(flag_name, False):
                enabled.append(registry_key)

    return list(set(enabled))  # Remove duplicates


def _apply_feature_generator(
    df: pd.DataFrame,
    registry_key: str,
    adv_module,
) -> pd.DataFrame:
    """Apply a single feature generator from the registry.

    Args:
        df: Input DataFrame.
        registry_key: Key in FEATURE_REGISTRY.
        adv_module: The advanced features module.

    Returns:
        DataFrame with new features added.
    """
    from finance_ml.features.advanced import FEATURE_REGISTRY

    if registry_key not in FEATURE_REGISTRY:
        logger.warning(f"Unknown feature generator: {registry_key}")
        return df

    entry = FEATURE_REGISTRY[registry_key]
    func = entry["function"]
    category = entry.get("category", registry_key)

    logger.info(f"Applying {category} features ({registry_key})")

    try:
        return func(df)
    except Exception as e:
        logger.error(f"Error applying {registry_key}: {e}")
        return df
