"""Revenue forecast feature engineering."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_revenue_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer revenue forecast features.

    Features computed:
    - revenue_estimate_momentum: (Revenues Est NTM - Revenues LTM) / Revenues LTM
    - revenue_surprise_volatility: Standard deviation of revenue surprises (if available)
    """
    result = df.copy()

    # Revenue Estimate Momentum
    if "revenues_est_avg_ntm" in df.columns and "total_revenues_ltm" in df.columns:
        est_ntm = pd.to_numeric(df["revenues_est_avg_ntm"], errors="coerce")
        rev_ltm = pd.to_numeric(df["total_revenues_ltm"], errors="coerce")

        result["revenue_estimate_momentum"] = _safe_div(est_ntm - rev_ltm, rev_ltm)

    # Revenue Surprise Volatility
    # Requires historical surprise data. Currently we only have current surprise calculated in earnings.py.
    # If we had multiple periods (e.g. surprise_1q, surprise_2q...), we could calc std dev.
    # For now, we will assign NaN or skip.
    if "revenue_surprise_volatility" not in result.columns:
        result["revenue_surprise_volatility"] = float("nan")

    logger.info("Engineered revenue forecast features")
    return result
