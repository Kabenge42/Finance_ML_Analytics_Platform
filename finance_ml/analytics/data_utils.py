"""
Data loading and preprocessing utilities for feature analytics.

This module provides functions for loading feature data from databases,
preprocessing, validation, and backfilling missing columns.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy import Engine

try:
    from sqlalchemy import create_engine
except ImportError:  # pragma: no cover
    create_engine = None  # type: ignore


def load_feature_data_from_db(
    db_url: Optional[str] = None,
    earnings_date_filter: str = "2026-01-01",
    limit: Optional[int] = None,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load feature data from PostgreSQL database materialized view.

    Loads data from public.mv_all_stock_features with optional filtering
    by next_earnings date.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. If None, reads from DB_URL environment variable
    earnings_date_filter : str, default "2026-01-01"
        Filter stocks with next_earnings >= this date (ISO format: YYYY-MM-DD)
    limit : int, optional
        Maximum number of rows to return
    schema : str, optional
        Database schema name. If None, reads from DB_SCHEMA environment variable
        or defaults to 'public'

    Returns
    -------
    pd.DataFrame
        DataFrame with feature data from mv_all_stock_features

    Raises
    ------
    ImportError
        If SQLAlchemy or psycopg2 not available
    ValueError
        If db_url is not provided and DB_URL environment variable is not set

    Examples
    --------
    >>> df = load_feature_data_from_db()
    >>> df = load_feature_data_from_db(db_url="postgresql+psycopg2://user:pass@host:5432/db")
    """
    if create_engine is None:
        raise ImportError(
            "SQLAlchemy not available. Install psycopg2-binary and SQLAlchemy to use database loading."
        )

    # Resolve database URL
    if db_url is None:
        db_url = os.environ.get("DB_URL")
        if db_url is None:
            raise ValueError(
                "db_url parameter not provided and DB_URL environment variable not set. "
                "Please provide a database URL or set the DB_URL environment variable."
            )

    # Resolve schema
    if schema is None:
        schema = os.environ.get("DB_SCHEMA", "public")

    view_name = "mv_all_stock_features"
    view_ref = f"{schema}.{view_name}"

    logging.info(
        "Loading feature data from %s (view: %s, earnings_date_filter: %s)",
        db_url.split("@")[-1] if "@" in db_url else db_url,
        view_ref,
        earnings_date_filter,
    )

    # Create SQLAlchemy engine
    engine = create_engine(db_url)

    # Build SQL query
    base_query = f"""
        SELECT *
        FROM {view_ref}
        WHERE next_earnings >= DATE '{earnings_date_filter}'
        ORDER BY next_earnings
    """

    # Apply limit if specified
    if limit is not None:
        query = f"{base_query} LIMIT {int(limit)}"
    else:
        query = base_query

    # Execute query and load into DataFrame
    df = pd.read_sql(query, engine)

    logging.info("Loaded %d rows from %s", len(df), view_ref)

    return df


def backfill_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill expected columns for charts and analysis.

    This function normalizes SQL results and creates missing columns
    by mapping from alternative column names or calculating from
    existing columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature data

    Returns
    -------
    pd.DataFrame
        DataFrame with backfilled columns

    Examples
    --------
    >>> df = backfill_feature_columns(df)
    >>> print(f"Columns: {len(df.columns)}")
    """
    # Ensure DataFrame type
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return df

    # Backfill analyst_neutral_pct if missing
    if "analyst_neutral_pct" not in df.columns:
        bullish = df.get("analyst_bullish_pct")
        bearish = df.get("analyst_bearish_pct")
        if bullish is not None and bearish is not None:
            neutral = 100 - bullish - bearish
            df["analyst_neutral_pct"] = neutral.clip(lower=0, upper=100)

    # Map inventory_turnover to expected column name
    if "inventory_turnover_mv" not in df.columns:
        if "inventory_turnover" in df.columns:
            df["inventory_turnover_mv"] = df["inventory_turnover"]

    # Calculate inventory_days from turnover
    if "inventory_days" not in df.columns:
        turnover_col = df.get("inventory_turnover_mv")
        if turnover_col is not None:
            turnover = turnover_col.replace(0, pd.NA)
            df["inventory_days"] = 365 / turnover

    # Map R&D intensity columns
    if "rnd_intensity_ltm" not in df.columns:
        for src_col in ["rnd_intensity", "rnd_to_revenue"]:
            if src_col in df.columns:
                df["rnd_intensity_ltm"] = df[src_col]
                break

    # Map tangible book value columns
    if "tangible_book_value_ltm" not in df.columns:
        if "tangible_book_value" in df.columns:
            df["tangible_book_value_ltm"] = df["tangible_book_value"]

    # Map goodwill concentration
    if "goodwill_concentration" not in df.columns:
        for src_col in ["goodwill_to_equity", "goodwill_to_assets_pct"]:
            if src_col in df.columns:
                df["goodwill_concentration"] = df[src_col]
                break

    # Ensure industry column exists
    if "industry" not in df.columns and "sector" in df.columns:
        df["industry"] = df["sector"]

    logging.info("Backfill complete. Columns: %d", len(df.columns))

    return df


def compute_metric_statistics(series: pd.Series) -> Optional[dict]:
    """
    Compute standard statistics for a numeric series.

    Parameters
    ----------
    series : pd.Series
        Input series with numeric data

    Returns
    -------
    dict or None
        Dictionary with statistics (count, mean, median, std, min, max, quartiles, etc.)
        Returns None if series is empty or non-numeric

    Examples
    --------
    >>> stats = compute_metric_statistics(df['p_e_ratio'])
    >>> print(f"Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
    """
    data = pd.to_numeric(series, errors="coerce").dropna()
    if len(data) == 0:
        return None

    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "q25": float(data.quantile(0.25)),
        "q75": float(data.quantile(0.75)),
        "positive_pct": float((data > 0).sum() / len(data) * 100),
        "missing_pct": float((series.isna().sum() / len(series)) * 100),
    }


def validate_feature_alignment(df: pd.DataFrame, categories: dict) -> dict:
    """
    Check which features in categories exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature data
    categories : dict
        Dictionary mapping category names to lists of feature names

    Returns
    -------
    dict
        Dictionary with 'available', 'missing', and 'coverage_pct' per category

    Examples
    --------
    >>> validation = validate_feature_alignment(df, FEATURE_CATEGORIES)
    >>> low_coverage = {k: v for k, v in validation.items() if v['coverage_pct'] < 80}
    """
    validation_results = {}

    for category, features in categories.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        coverage = len(available) / len(features) * 100 if features else 0

        validation_results[category] = {
            "available_count": len(available),
            "missing_count": len(missing),
            "coverage_pct": coverage,
            "missing_features": missing[:5],  # Show first 5 missing
        }

    return validation_results


def safe_get_column(df: pd.DataFrame, *column_names: str, default=None):
    """
    Safely get a column from DataFrame, trying multiple names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    *column_names : str
        Column names to try in order
    default : any, optional
        Default value if no column found

    Returns
    -------
    pd.Series or default
        First found column or default value

    Examples
    --------
    >>> col = safe_get_column(df, 'industry', 'sector', default=pd.Series())
    """
    for col_name in column_names:
        if col_name in df.columns:
            return df[col_name]
    return default
