"""
Feature scaling module for finance_ml preprocessing.

Phase 9.1 refactor: Extracted from advanced_preprocessing.py.

This module provides:
- Scaler pipeline creation (Standard, Robust, MinMax)
- Feature scaling with optional sector-aware grouping
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)


def create_scaler_pipeline(
    scaler_type: str = "robust",
    by_sector: bool = True,
) -> Tuple[Any, bool]:
    """Create a scaler for feature scaling.

    Args:
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        by_sector: Whether to scale separately by sector

    Returns:
        Tuple of (scaler, by_sector flag)
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    logger.info(f"Created {scaler_type} scaler (sector-specific: {by_sector})")
    return scaler, by_sector


def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    scaler_type: str = "robust",
    by_sector: bool = True,
    exclude_price_columns: bool = True,
    exclude_count_columns: bool = False,
) -> pd.DataFrame:
    """Scale features using specified scaler.

    Args:
        df: Input DataFrame
        columns: Columns to scale (default: all numeric)
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        by_sector: Scale separately by sector (default: True)
        exclude_price_columns: If True, exclude price/valuation columns (default: True)
        exclude_count_columns: If True, exclude count/int columns (default: False)

    Returns:
        DataFrame with scaled features

    Note:
        Price columns (last_price, price_target) are excluded by default to preserve
        original dollar values required for business metrics:
        (Predicted_Target - Last_Price) / Last_Price

        Scaling price columns would destroy the interpretability needed for valuation
        comparison.
    """
    from finance_ml.ml_workflow.preprocessing.column_semantics import (
        classify_columns,
        get_scalable_columns,
    )

    result = df.copy()

    # Auto-detect numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to only columns that exist
    columns = [c for c in columns if c in df.columns]

    # Convert nullable integer columns to float64 before scaling
    # This prevents "TypeError: cannot safely cast non-equivalent object to int64"
    # when assigning float scaled values back to nullable integer columns
    for col in columns:
        if col in result.columns:
            # Check if it's a numeric type (numpy or pandas)
            col_dtype = result[col].dtype
            if isinstance(col_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(
                col_dtype
            ):
                # Attempt to convert object/categorical to numeric if it contains numbers
                try:
                    result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
                except (ValueError, TypeError):
                    # Not numeric data, skip it later in the float check
                    pass
            elif not pd.api.types.is_float_dtype(col_dtype):
                # Convert any other numeric type (e.g. int) to float64 for scaling
                result[col] = result[col].astype("float64")

    # Apply semantic filtering using helper function for comprehensive exclusion
    if exclude_price_columns:
        # Use get_scalable_columns() for comprehensive semantic filtering
        # This automatically excludes price columns while including market_value, ratio, percentage, count, and other
        scalable = get_scalable_columns(columns)
        scalable = [c for c in scalable if c in result.columns]
        excluded_count = len(columns) - len(scalable)
        logger.info(
            f"Scaling {len(scalable)} columns, excluding {excluded_count} price columns (using semantic classification)"
        )
        columns = scalable

    # Optionally exclude count columns (discrete integers) from scaling
    if exclude_count_columns and columns:
        classification = classify_columns(columns)
        count_columns = set(classification.get("count", set()))
        if count_columns:
            before = len(columns)
            columns = [c for c in columns if c not in count_columns]
            excluded_counts = before - len(columns)
            if excluded_counts > 0:
                logger.info(
                    f"Excluded {excluded_counts} count columns from scaling (semantic classification)"
                )

    # If no columns to scale after filtering, return original
    if not columns:
        logger.warning("No columns to scale after applying exclusions")
        return result

    # Re-filter columns to only those that are now float
    # This ensures the scaler receives clean float data
    columns = [
        c for c in columns if c in result.columns and pd.api.types.is_float_dtype(result[c].dtype)
    ]

    if not columns:
        logger.warning("No numeric columns available for scaling after type conversion")
        return result

    scaler, _ = create_scaler_pipeline(scaler_type, by_sector)

    if by_sector and "sector" in result.columns:
        # Sector-specific scaling
        for sector in result["sector"].dropna().unique():
            mask = result["sector"] == sector
            sector_data = result.loc[mask, columns]

            if sector_data.shape[0] > 1:
                # Ensure we're working with float arrays
                scaled_values = scaler.fit_transform(sector_data.values.astype(np.float64))
                result.loc[mask, columns] = scaled_values
    else:
        # Global scaling
        scaled_values = scaler.fit_transform(result[columns].values.astype(np.float64))
        result[columns] = scaled_values

    logger.info(f"Scaled {len(columns)} features using {scaler_type} scaler")
    return result
