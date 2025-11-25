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
) -> pd.DataFrame:
    """Scale features using specified scaler.

    Args:
        df: Input DataFrame
        columns: Columns to scale (default: all numeric)
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        by_sector: Scale separately by sector (default: True)
        exclude_price_columns: If True, exclude price/valuation columns (default: True)

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
        PRICE_COLUMNS,
    )

    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Apply semantic filtering
    if exclude_price_columns:
        excluded = [c for c in columns if c.lower() in PRICE_COLUMNS and c in df.columns]
        scalable = [c for c in columns if c.lower() not in PRICE_COLUMNS and c in df.columns]
        logger.info(
            f"Scaling {len(scalable)} columns, excluding {len(excluded)} price columns"
        )
        columns = scalable
    
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
