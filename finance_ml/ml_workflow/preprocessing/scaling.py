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

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Apply semantic filtering using helper function for comprehensive exclusion
    if exclude_price_columns:
        # Use get_scalable_columns() for comprehensive semantic filtering
        # This automatically excludes price columns while including market_value, ratio, percentage, count, and other
        scalable = get_scalable_columns(columns)
        scalable = [c for c in scalable if c in df.columns]
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

    scaler, _ = create_scaler_pipeline(scaler_type, by_sector)

    if by_sector and "sector" in df.columns:
        # Sector-specific scaling
        for sector in df["sector"].dropna().unique():
            mask = df["sector"] == sector
            sector_data = df.loc[mask, columns]

            if sector_data.shape[0] > 1:
                result.loc[mask, columns] = scaler.fit_transform(sector_data)
    else:
        # Global scaling
        result[columns] = scaler.fit_transform(df[columns])

    logger.info(f"Scaled {len(columns)} features using {scaler_type} scaler")
    return result
