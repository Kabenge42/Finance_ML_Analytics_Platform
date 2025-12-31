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
) -> pd.DataFrame:
    """Stage 9: Feature engineering."""
    logger.info(f"Stage 9: Building features using preset: {preset}")
    # Current build_features doesn't take categories directly, but it's in the config for future use
    result = build_features(df, preset=preset)

    # Earnings analytics are now included in "comprehensive" and "full_enhanced" presets.
    # Only apply explicitly if using a different preset and the flag is True.
    if engineer_earnings_analytics and preset not in ("comprehensive", "full_enhanced"):
        logger.info(
            "Applying earnings analytics features (Estimated vs. Actual and GAAP vs. Adjusted)"
        )
        from finance_ml.features.advanced import (
            engineer_estimated_vs_actual_analytics,
            engineer_gaap_vs_adjusted_analytics,
        )

        result = engineer_estimated_vs_actual_analytics(result)
        result = engineer_gaap_vs_adjusted_analytics(result)

    return result
