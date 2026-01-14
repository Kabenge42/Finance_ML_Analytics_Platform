"""Feature Engineering stage for ETL."""

import logging
from typing import List, Optional

import pandas as pd

from finance_ml.ml_workflow.features.api import build_features

logger = logging.getLogger(__name__)


def run_feature_engineering_stage(
    df: pd.DataFrame,
    preset: str = "comprehensive",
    categories: Optional[List[str]] = None,
    engineer_earnings_analytics: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Stage 9: Feature engineering."""
    logger.info(f"Stage 9: Building features using preset: {preset}")
    # Current build_features doesn't take categories directly, but it's in the config for future use
    result = build_features(df, preset=preset)

    # Earnings analytics are now included in "comprehensive" and "full_enhanced" presets.
    # Only apply explicitly if using a different preset and the flag is True.
    if preset not in ("comprehensive", "full_enhanced"):
        from finance_ml.features import advanced as adv

        if engineer_earnings_analytics:
            logger.info("Applying earnings analytics features")
            result = adv.engineer_estimated_vs_actual_analytics(result)
            result = adv.engineer_gaap_vs_adjusted_analytics(result)

        # Apply new v1.14 granular features if requested via kwargs
        if kwargs.get("engineer_price_target_dynamics"):
            logger.info("Applying price target dynamics")
            result = adv.engineer_price_target_dynamics(result)

        if kwargs.get("engineer_fiscal_calendar"):
            logger.info("Applying fiscal calendar features")
            result = adv.engineer_fiscal_calendar_features(result)

        if kwargs.get("engineer_dividend_timing"):
            logger.info("Applying dividend timing features")
            result = adv.engineer_dividend_timing_features(result)

        if kwargs.get("engineer_eps_trajectory"):
            logger.info("Applying EPS trajectory features")
            result = adv.engineer_eps_trajectory_features(result)

        if kwargs.get("engineer_cashflow_temporal"):
            logger.info("Applying cashflow temporal features")
            result = adv.engineer_cashflow_temporal_features(result)

        if kwargs.get("engineer_valuation_timeseries"):
            logger.info("Applying valuation timeseries features")
            result = adv.engineer_valuation_timeseries_features(result)

        if kwargs.get("engineer_analyst_coverage"):
            logger.info("Applying analyst coverage features")
            result = adv.engineer_analyst_coverage_features(result)

        if kwargs.get("engineer_revenue_forecast"):
            logger.info("Applying revenue forecast features")
            result = adv.engineer_revenue_forecast_features(result)

        if kwargs.get("engineer_dividend_reliability"):
            logger.info("Applying dividend reliability features")
            result = adv.engineer_dividend_reliability_features(result)

        if kwargs.get("engineer_employment_dynamics"):
            logger.info("Applying employment dynamics features")
            result = adv.engineer_employment_dynamics_features(result)

        if kwargs.get("engineer_employee_productivity"):
            logger.info("Applying employee productivity features")
            result = adv.engineer_employee_productivity_features(result)

        if kwargs.get("engineer_balance_sheet_trends"):
            logger.info("Applying balance sheet trends")
            result = adv.engineer_balance_sheet_trends(result)

        if kwargs.get("engineer_margin_trends"):
            logger.info("Applying margin trends")
            result = adv.engineer_margin_trends(result)

        if kwargs.get("engineer_accounting_quality"):
            logger.info("Applying accounting quality features")
            result = adv.engineer_accounting_quality_features(result)

    return result
