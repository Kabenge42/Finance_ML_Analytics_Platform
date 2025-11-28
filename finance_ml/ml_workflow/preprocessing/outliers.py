"""
Outlier detection and handling module for finance_ml preprocessing.

Phase 9.1 refactor: Extracted from advanced_preprocessing.py.

This module provides:
- IQR-based outlier detection
- Z-score outlier detection
- Isolation Forest outlier detection
- Sector-aware winsorization
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    by_sector: bool = True,
    iqr_multiplier: float = 2.5,
) -> pd.DataFrame:
    """Detect outliers using Interquartile Range (IQR) method.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (default: all numeric)
        by_sector: Apply IQR separately by sector (default: True)
        iqr_multiplier: IQR multiplier for bounds (default: 1.5)

    Returns:
        DataFrame with boolean columns indicating outliers (col_outlier)
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        outlier_col = f"{col}_outlier"

        if by_sector and "sector" in df.columns:
            # Sector-specific IQR detection
            result[outlier_col] = False

            for sector in df["sector"].dropna().unique():
                mask = df["sector"] == sector
                sector_data = df.loc[mask, col].dropna()

                if len(sector_data) < 4:
                    continue

                q1 = sector_data.quantile(0.25)
                q3 = sector_data.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr

                outliers = (df.loc[mask, col] < lower_bound) | (df.loc[mask, col] > upper_bound)
                result.loc[mask, outlier_col] = outliers
        else:
            # Global IQR detection
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            result[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)

    logger.info(f"Detected outliers for {len(columns)} columns using IQR method")
    return result


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 3.0,
    by_sector: bool = True,
) -> pd.DataFrame:
    """Detect outliers using z-score method.

    Args:
        df: Input DataFrame
        columns: Columns to check (default: all numeric)
        threshold: Z-score threshold (default: 3.0)
        by_sector: Apply z-score separately by sector (default: True)

    Returns:
        DataFrame with boolean columns indicating outliers (col_zscore_outlier)
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        outlier_col = f"{col}_zscore_outlier"

        if by_sector and "sector" in df.columns:
            # Sector-specific z-score
            result[outlier_col] = False

            for sector in df["sector"].dropna().unique():
                mask = df["sector"] == sector
                sector_data = df.loc[mask, col].dropna()

                if len(sector_data) < 3:
                    continue

                mean = sector_data.mean()
                std = sector_data.std()

                if std == 0:
                    continue

                z_scores = np.abs((df.loc[mask, col] - mean) / std)
                result.loc[mask, outlier_col] = z_scores > threshold
        else:
            # Global z-score
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                result[outlier_col] = z_scores > threshold
            else:
                result[outlier_col] = False

    logger.info(f"Detected outliers for {len(columns)} columns using z-score method")
    return result


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """Detect outliers using Isolation Forest algorithm.

    Args:
        df: Input DataFrame
        columns: Columns to use for detection (default: all numeric)
        contamination: Expected proportion of outliers (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Boolean Series indicating outliers (True = outlier)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Select and clean data
    data = df[columns].copy()

    # Fill missing values with median for algorithm compatibility
    for col in data.columns:
        if data[col].isna().any():
            data[col].fillna(data[col].median(), inplace=True)

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)

    predictions = iso_forest.fit_predict(data)
    outliers = predictions == -1  # -1 indicates outlier

    logger.info(f"Detected {outliers.sum()} outliers using Isolation Forest")
    return pd.Series(outliers, index=df.index)


def winsorize_by_sector(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower_percentile: float = 0.05,
    upper_percentile: float = 0.95,
    by_sector: bool = True,
    exclude_price_columns: bool = True,
    exclude_ratio_columns: bool = True,
) -> pd.DataFrame:
    """Winsorize extreme values by replacing with percentile bounds.

    Args:
        df: Input DataFrame
        columns: Columns to winsorize (default: all numeric)
        lower_percentile: Lower percentile bound (default: 0.01 = 1%)
        upper_percentile: Upper percentile bound (default: 0.99 = 99%)
        by_sector: Apply winsorization separately by sector (default: True)
        exclude_price_columns: If True, exclude price/valuation columns (default: True)
        exclude_ratio_columns: If True, exclude pre-normalized ratios (default: True)

    Returns:
        DataFrame with winsorized values

    Note:
        Price columns (last_price, price_target, market_cap) are excluded by default
        to preserve original dollar values required for business metrics:
        (Predicted_Target - Last_Price) / Last_Price
    """
    from finance_ml.ml_workflow.preprocessing.column_semantics import (
        PRICE_COLUMNS,
        RATIO_COLUMNS,
        PERCENTAGE_COLUMNS,
    )

    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Apply semantic filtering
    excluded = set()
    if exclude_price_columns:
        excluded.update(PRICE_COLUMNS)
    if exclude_ratio_columns:
        excluded.update(RATIO_COLUMNS)
        excluded.update(PERCENTAGE_COLUMNS)  # Also exclude percentages when excluding ratios

    # Filter columns: keep only those not in excluded set and present in DataFrame
    winsorizable = [c for c in columns if c.lower() not in excluded and c in df.columns]
    excluded_present = [c for c in columns if c.lower() in excluded and c in df.columns]

    logger.info(
        f"Winsorizing {len(winsorizable)} columns, excluding {len(excluded_present)} semantic columns "
        f"(price={exclude_price_columns}, ratio={exclude_ratio_columns})"
    )

    for col in winsorizable:
        if col not in df.columns:
            continue

        # CRITICAL FIX: Cast integer columns to float64 BEFORE clipping operations
        # This prevents TypeError: "Invalid value 'X.Y' for dtype 'Int64'"
        # when assigning float clip results to Int64/Int32/Int16/Int8 columns
        # Follows same pattern as apply_median_imputation fix (test_median_imputation_int64_fix.py)
        if pd.api.types.is_integer_dtype(result[col]):
            result[col] = result[col].astype(float)

        if by_sector and "sector" in df.columns:
            # Sector-specific winsorization
            for sector in df["sector"].dropna().unique():
                mask = df["sector"] == sector
                # Use result[col] (already float) instead of df[col] to avoid dtype issues
                sector_data = result.loc[mask, col]

                if sector_data.isna().all():
                    continue

                lower_bound = sector_data.quantile(lower_percentile)
                upper_bound = sector_data.quantile(upper_percentile)

                result.loc[mask, col] = sector_data.clip(lower=lower_bound, upper=upper_bound)
        else:
            # Global winsorization
            # Use result[col] (already float) instead of df[col] to avoid dtype issues
            lower_bound = result[col].quantile(lower_percentile)
            upper_bound = result[col].quantile(upper_percentile)

            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

    logger.info(
        f"Winsorized {len(winsorizable)} columns (percentiles: {lower_percentile:.2%}-{upper_percentile:.2%})"
    )
    return result
