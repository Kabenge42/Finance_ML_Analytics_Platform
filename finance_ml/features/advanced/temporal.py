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
            # leverage fy_end_date if available for true fiscal calculations
            if "fy_end_date" in result.columns:
                fy_ends = pd.to_datetime(result["fy_end_date"], errors="coerce")
                dates = result[date_col]

                # Calculate months since FY end
                months_since = (dates.dt.year - fy_ends.dt.year) * 12 + (
                    dates.dt.month - fy_ends.dt.month
                )

                # Fiscal month (1-12)
                f_month = ((months_since - 1) % 12) + 1

                # Current fiscal quarter and year
                result["fiscal_quarter"] = (
                    ((f_month - 1) // 3 + 1).fillna(dates.dt.quarter).fillna(0).astype(int)
                )
                result["year"] = (
                    (fy_ends.dt.year + 1 + ((months_since - 1) // 12))
                    .fillna(dates.dt.year)
                    .fillna(0)
                    .astype(int)
                )
                result["month"] = dates.dt.month
            else:
                result["fiscal_quarter"] = result[date_col].dt.quarter
                result["month"] = result[date_col].dt.month
                result["year"] = result[date_col].dt.year

            # Days since reference date (only if explicitly requested via parameter)
            if reference_date is not None:
                result["days_since_reference"] = (result[date_col] - effective_ref_date).dt.days

    # Perform data quality validation if columns exist
    if "fy_end_date" in result.columns:
        fy_future = result["fy_end_date"] > effective_ref_date
        if fy_future.any():
            logger.debug(f"Data Quality: {fy_future.sum()} rows have FY End Date in the future")

    if "income_statement_report_date" in result.columns and "fy_end_date" in result.columns:
        # Report date before FY End - 1 year is suspicious
        report_too_old = result["income_statement_report_date"] < (
            result["fy_end_date"] - pd.DateOffset(years=1)
        )
        if report_too_old.any():
            logger.debug(
                f"Data Quality: {report_too_old.sum()} rows have report dates predating fiscal year"
            )

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

    # Revenue and EBITDA comparisons (Phase 9.3)
    # Check for LTM and 5Y Average revenues (handle multiple possible column names)
    rev_ltm_col = next(
        (
            c
            for c in ["total_revenues_ltm", "total_revenues_1fy", "total_revenues_fy"]
            if c in result.columns
        ),
        None,
    )
    rev_avg_col = next(
        (
            c
            for c in ["total_revenues_5yavgltm", "total_revenues_5yavg", "total_revenues_5yavgfq"]
            if c in result.columns
        ),
        None,
    )

    if rev_ltm_col and rev_avg_col:
        result["ltm_vs_5yavg_revenue"] = (
            (result[rev_ltm_col] - result[rev_avg_col]) / result[rev_avg_col].replace(0, pd.NA)
        ).astype(
            "Float64"
        )  # Use nullable Float64 instead of float

    ebitda_col = next((c for c in ["ebitda_fq", "ebitda_ltm"] if c in result.columns), None)
    ebitda_avg_col = next(
        (c for c in ["ebitda_5yavgfq", "ebitda_5yavgltm"] if c in result.columns), None
    )

    if ebitda_col and ebitda_avg_col:
        result["fq_vs_5yavg_ebitda"] = (
            (result[ebitda_col] - result[ebitda_avg_col]) / result[ebitda_avg_col].replace(0, pd.NA)
        ).astype(
            "Float64"
        )  # Use nullable Float64 instead of float

    # Quarterly Volatility Score (Coefficient of Variation)
    ebitda_cols = ["ebitda_fq", "ebitda_fq_1", "ebitda_fq_2", "ebitda_fq_3"]
    available_cols = [c for c in ebitda_cols if c in result.columns]
    if len(available_cols) >= 2:
        ebitda_series = result[available_cols].astype("Float64")  # Use nullable Float64
        mean_ebitda = ebitda_series.mean(axis=1)
        std_ebitda = ebitda_series.std(axis=1)
        # CV = std / mean
        result["quarterly_volatility_score"] = (std_ebitda / mean_ebitda.replace(0, pd.NA)).astype(
            "Float64"
        )

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
