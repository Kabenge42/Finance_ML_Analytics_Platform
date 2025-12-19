"""Temporal Calculation Utilities

Implements code_guidelines.md Section 9.3.0 - Temporal Calculation Standards:
- All temporal calculations use reference_date (not last_updated)
- Ensures reproducibility, consistency, and testability
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_reference_date(reference_date: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Get the reference date for temporal calculations.

    Per code_guidelines.md Section 9.3.0: All temporal calculations MUST use
    a consistent reference_date rather than last_updated or mixed date columns.

    Args:
        reference_date: Explicit reference date. If None, defaults to current date.

    Returns:
        Normalized timestamp (midnight) for the reference date.

    Example:
        >>> ref = get_reference_date()  # Uses current date
        >>> ref = get_reference_date(pd.Timestamp('2025-12-18'))  # Explicit date
    """
    if reference_date is not None:
        return pd.Timestamp(reference_date).normalize()
    return pd.Timestamp.now().normalize()


def compute_days_to_earnings(
    next_earnings: pd.Series | pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Compute days until next earnings using reference_date.

    Per code_guidelines.md Section 9.3.0: This function uses reference_date
    (not last_updated) for consistent temporal calculations.

    Args:
        next_earnings: Series of next earnings dates (datetime or string)
                      OR DataFrame containing 'next_earnings' column.
        reference_date: Reference date for calculation. Defaults to today.

    Returns:
        Series of integer days until earnings.
        Positive = future, Negative = past, 0 = today.
    """
    ref_date = get_reference_date(reference_date)

    # Handle DataFrame input for backward compatibility
    if isinstance(next_earnings, pd.DataFrame):
        if "next_earnings" in next_earnings.columns:
            next_earnings_series = next_earnings["next_earnings"]
        else:
            logger.warning("Column 'next_earnings' not found in DataFrame")
            return pd.Series(dtype=float, index=next_earnings.index)
    else:
        next_earnings_series = next_earnings

    # Ensure datetime
    earnings_dt = pd.to_datetime(next_earnings_series, errors="coerce")

    # Calculate days difference
    days = (earnings_dt - ref_date).dt.days

    return days


def compute_earnings_report_recency(
    income_statement_report_date: pd.Series,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Compute days since last earnings report using reference_date.

    Per code_guidelines.md Section 9.3.0: This function uses reference_date
    (not last_updated) for consistent temporal calculations.

    Args:
        income_statement_report_date: Series of report dates.
        reference_date: Reference date for calculation. Defaults to today.

    Returns:
        Series of integer days since last report. Always positive.
    """
    ref_date = get_reference_date(reference_date)

    # Ensure datetime
    report_dt = pd.to_datetime(income_statement_report_date, errors="coerce")

    # Calculate days since report
    recency = (ref_date - report_dt).dt.days

    return recency


def add_reference_date_column(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Add _reference_date column to DataFrame for temporal calculations.

    This enables auditing and reproducibility by recording which reference
    date was used for all temporal feature calculations.

    Args:
        df: Input DataFrame.
        reference_date: Reference date. Defaults to current date.

    Returns:
        DataFrame with _reference_date column added.
    """
    df = df.copy()
    df["_reference_date"] = get_reference_date(reference_date)
    return df


def recalculate_temporal_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Recalculate all temporal features using consistent reference_date.

    This function ensures all temporal calculations in the DataFrame use
    the same reference point, fixing inconsistencies from stale data or
    mixed date references.

    Args:
        df: DataFrame with next_earnings and/or income_statement_report_date columns.
        reference_date: Reference date. Defaults to current date.

    Returns:
        DataFrame with recalculated days_to_earnings and earnings_report_recency.
    """
    ref_date = get_reference_date(reference_date)
    df = df.copy()

    # Add reference date for auditing
    df["_reference_date"] = ref_date

    # Recalculate days_to_earnings
    if "next_earnings" in df.columns:
        df["next_earnings"] = pd.to_datetime(df["next_earnings"], errors="coerce")
        df["days_to_earnings"] = compute_days_to_earnings(
            df["next_earnings"], reference_date=ref_date
        )

    # Recalculate earnings_report_recency
    if "income_statement_report_date" in df.columns:
        df["income_statement_report_date"] = pd.to_datetime(
            df["income_statement_report_date"], errors="coerce"
        )
        df["earnings_report_recency"] = compute_earnings_report_recency(
            df["income_statement_report_date"], reference_date=ref_date
        )

    return df
