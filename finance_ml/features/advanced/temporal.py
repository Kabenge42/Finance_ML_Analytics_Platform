"""Temporal and seasonality feature engineering."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

def engineer_temporal_features(
    df: pd.DataFrame,
    date_col: str = "reference_date",
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Engineer temporal and seasonality features using consistent reference_date.

    Adds:
    - fiscal_quarter, month, year from date_col
    - _reference_date column for auditing
    - days_to_earnings: (next_earnings - reference_date).days
    - earnings_report_recency: (reference_date - income_statement_report_date).days
    - reporting_lag: (next_earnings - income_statement_report_date).days
    - ltm_vs_5yavg_revenue: (total_revenues_1fy - 5Y avg)/5Y avg
    - fq_vs_5yavg_ebitda: (ebitda_fq - ebitda_5yavgfq)/ebitda_5yavgfq
    - quarterly_volatility_score: coefficient of variation across quarterly EBITDA

    Args:
        df: Input DataFrame with date columns.
        date_col: Column name for fiscal timing (quarter, month, year).
        reference_date: Reference date for temporal calculations.
                       Defaults to pd.Timestamp.now().normalize() if not provided.

    Returns:
        DataFrame with temporal features added.
    """
    result = df.copy()

    def _format_date(series: pd.Series) -> pd.Series:
        return series.dt.strftime("%d %b %Y").astype("string").where(series.notna(), pd.NA)

    # Use reference_date per code_guidelines.md Section 9.3.0
    if reference_date is None:
        effective_ref_date = pd.Timestamp.now().normalize()
    else:
        effective_ref_date = pd.Timestamp(reference_date).normalize()

    # Store reference date for auditing/reproducibility
    result["_reference_date"] = effective_ref_date

    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found, skipping some temporal features")
    else:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[date_col]):
            try:
                result[date_col] = pd.to_datetime(result[date_col])
            except Exception as e:
                logger.warning(f"Could not convert {date_col} to datetime: {e}")

        if (
            pd.api.types.is_datetime64_any_dtype(result[date_col])
            and not result[date_col].isna().all()
        ):
            # Extract fiscal timing features
            result["fiscal_quarter"] = result[date_col].dt.quarter
            result["month"] = result[date_col].dt.month
            result["year"] = result[date_col].dt.year

            # Days since reference date (only if explicitly requested via parameter)
            if reference_date is not None:
                result["days_since_reference"] = (result[date_col] - effective_ref_date).dt.days

    # Calculate days_to_earnings using reference_date
    if "next_earnings" in result.columns:
        result["next_earnings"] = pd.to_datetime(result["next_earnings"], errors="coerce")
        result["days_to_earnings"] = (result["next_earnings"] - effective_ref_date).dt.days

    # Calculate days_to_dividend using reference_date
    if "dividend_record_ex_date" in result.columns:
        result["dividend_record_ex_date"] = pd.to_datetime(
            result["dividend_record_ex_date"], errors="coerce"
        )
        result["days_to_dividend"] = (
            result["dividend_record_ex_date"] - effective_ref_date
        ).dt.days

    # Calculate earnings_report_recency using reference_date
    if "income_statement_report_date" in result.columns:
        result["income_statement_report_date"] = pd.to_datetime(
            result["income_statement_report_date"], errors="coerce"
        )
        result["earnings_report_recency"] = (
            effective_ref_date - result["income_statement_report_date"]
        ).dt.days

    if "income_statement_report_date" in result.columns and "next_earnings" in result.columns:
        isrd = pd.to_datetime(result["income_statement_report_date"], errors="coerce")
        ne = pd.to_datetime(result["next_earnings"], errors="coerce")
        result["reporting_lag"] = (ne - isrd).dt.days

    # Add formatted companions for key dates
    date_cols_to_format = [
        "next_earnings",
        "_reference_date",
        "income_statement_report_date",
        "dividend_record_ex_date",
    ]
    for col in date_cols_to_format:
        if col in result.columns:
            # Ensure series is datetime for formatting
            series = pd.to_datetime(result[col], errors="coerce")
            result[f"{col}_formatted"] = _format_date(series)

    logger.info(f"Engineered temporal features using reference_date={effective_ref_date}")
    return result
