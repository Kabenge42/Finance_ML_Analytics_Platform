"""Temporal calculation utilities for dashboard components.

Per code_guidelines.md Section 9.3.0 Temporal Calculation Standards,
all temporal calculations (days_to_earnings, earnings_report_recency)
must use a consistent reference_date rather than last_updated or mixed
date columns.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_reference_date() -> pd.Timestamp:
    """Get the reference date for temporal calculations.

    Per code_guidelines.md Section 9.3.0, all temporal calculations
    (days_to_earnings, earnings_report_recency) must use a consistent
    reference_date rather than last_updated or mixed date columns.

    Returns:
        Normalized current timestamp for production use.
        Override with explicit date for reproducibility in tests.
    """
    return pd.Timestamp.now().normalize()


def compute_days_to_earnings(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    next_earnings_col: str = "next_earnings",
) -> pd.Series:
    """Compute days to earnings using consistent reference_date.

    Per code_guidelines.md Section 9.3.0 Temporal Calculation Standards.

    Args:
        df: DataFrame with next_earnings column
        reference_date: Reference date for calculation. Defaults to now().
        next_earnings_col: Name of the next earnings date column

    Returns:
        Series of days to earnings (positive = future, negative = past)
    """
    if reference_date is None:
        reference_date = get_reference_date()

    if next_earnings_col not in df.columns:
        logger.warning(f"Column '{next_earnings_col}' not found in DataFrame")
        return pd.Series(dtype=float, index=df.index)

    next_earnings = pd.to_datetime(df[next_earnings_col], errors="coerce")
    days_to_earnings = (next_earnings - reference_date).dt.days

    return days_to_earnings
