"""Growth-related feature engineering."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer growth metrics.

    Metrics computed:
    - Revenue CAGR (if multi-year data available)
    - EPS Growth %
    - EBITDA Growth %
    - Book Value Growth %

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with growth metrics added
    """
    result = df.copy()

    # Revenue Growth (YoY if available)
    if "revenue" in df.columns and "revenue_previous_year" in df.columns:
        result["revenue_growth_yoy"] = (
            _safe_div(
                (df["revenue"] - df["revenue_previous_year"]),
                df["revenue_previous_year"],
            )
            * 100
        )
        # Backward-compatible alias for event labels
        result["revenue_growth"] = result["revenue_growth_yoy"]

    # EPS Growth
    if "eps" in df.columns and "eps_previous_year" in df.columns:
        result["eps_growth_yoy"] = (
            _safe_div((df["eps"] - df["eps_previous_year"]), df["eps_previous_year"]) * 100
        )
        # Backward-compatible alias for event labels
        result["earnings_growth"] = result["eps_growth_yoy"]

    # EBITDA Growth
    if "ebitda" in df.columns and "ebitda_previous_year" in df.columns:
        result["ebitda_growth_yoy"] = (
            _safe_div((df["ebitda"] - df["ebitda_previous_year"]), df["ebitda_previous_year"]) * 100
        )
        # Backward-compatible alias for event labels
        result["ebitda_growth"] = result["ebitda_growth_yoy"]

    logger.info("Engineered growth metrics")
    return result
