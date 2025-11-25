"""
Log-transform pipeline for highly skewed financial columns.

This module provides log-transforms as an alternative to winsorization for
market value columns (market_cap, revenue, etc.) which typically have
skewness > 2.0.

Log-transforms preserve information about valid extreme values (e.g., mega-cap
stocks) while reducing skewness and outlier impact.

Methods:
- log1p: log(1 + x) for non-negative values
- signed_log: sign(x) * log(1 + |x|) for any value including negatives

Aligned with preprocessing_stages_4-8_improvement_plan.md Task 2.1
and code_guidelines.md v1.5 Section 8.5.3: Alternative Transformations
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_log_transforms(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'log1p',
) -> pd.DataFrame:
    """
    Apply log-transforms to skewed financial columns.
    
    Creates new log-transformed columns (log_*) while preserving original columns
    for interpretability.
    
    Args:
        df: Input DataFrame
        columns: Columns to transform (default: auto-detect market value columns)
        method: Transform method
            - 'log1p': log(1 + x) for non-negative values (handles zeros)
            - 'signed_log': sign(x) * log(1 + |x|) for any value (handles negatives)
    
    Returns:
        DataFrame with new log-transformed columns (log_*) added
        
    Raises:
        ValueError: If method is not 'log1p' or 'signed_log'
        
    Example:
        >>> df = pd.DataFrame({'market_cap': [1e9, 1e10, 1e11]})
        >>> result = apply_log_transforms(df, columns=['market_cap'], method='log1p')
        >>> 'log_market_cap' in result.columns
        True
        >>> 'market_cap' in result.columns  # Original preserved
        True
    """
    from finance_ml.ml_workflow.preprocessing.column_semantics import (
        get_log_transform_columns
    )
    
    if method not in ['log1p', 'signed_log']:
        raise ValueError(f"Unknown method: {method}. Use 'log1p' or 'signed_log'")
    
    result = df.copy()
    
    # Auto-detect market value columns if not specified
    if columns is None:
        columns = get_log_transform_columns(df.columns.tolist())
        logger.info(f"Auto-detected {len(columns)} market value columns for log-transform")
    
    transformed_count = 0
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
        
        # Skip if column is already log-transformed
        if col.lower().startswith('log_'):
            logger.debug(f"Column '{col}' already log-transformed, skipping")
            continue
        
        log_col_name = f'log_{col}'
        
        if method == 'log1p':
            # log(1 + x): Handles zeros, requires non-negative
            # Clip to avoid log of negative values
            result[log_col_name] = np.log1p(df[col].clip(lower=0))
        elif method == 'signed_log':
            # sign(x) * log(1 + |x|): Handles negative values
            result[log_col_name] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
        
        transformed_count += 1
        logger.debug(f"Created '{log_col_name}' using {method}")
    
    logger.info(
        f"Applied {method} transforms to {transformed_count} columns, "
        f"created {transformed_count} new log_* columns"
    )
    
    return result


def inverse_log_transform(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'log1p',
) -> pd.DataFrame:
    """
    Reverse log-transforms for interpretability.
    
    Converts log-transformed columns back to original scale. Useful for
    making predictions interpretable.
    
    Args:
        df: Input DataFrame with log-transformed columns
        columns: Log-transformed columns to reverse (e.g., ['log_market_cap'])
        method: Transform method used for forward transform
            - 'log1p': Inverse is expm1(x) = exp(x) - 1
            - 'signed_log': Inverse is sign(x) * (exp(|x|) - 1)
    
    Returns:
        DataFrame with reversed columns (original names without log_ prefix)
        
    Raises:
        ValueError: If method is not 'log1p' or 'signed_log'
        
    Example:
        >>> df = pd.DataFrame({'log_market_cap': [20.0, 22.0, 24.0]})
        >>> result = inverse_log_transform(df, columns=['log_market_cap'], method='log1p')
        >>> 'market_cap' in result.columns
        True
    """
    if method not in ['log1p', 'signed_log']:
        raise ValueError(f"Unknown method: {method}. Use 'log1p' or 'signed_log'")
    
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
        
        # Determine original column name
        if col.lower().startswith('log_'):
            original_col_name = col[4:]  # Remove 'log_' prefix
        else:
            original_col_name = f'{col}_original'
            logger.warning(
                f"Column '{col}' doesn't start with 'log_', "
                f"creating '{original_col_name}'"
            )
        
        if method == 'log1p':
            # Inverse: exp(x) - 1
            result[original_col_name] = np.expm1(df[col])
        elif method == 'signed_log':
            # Inverse: sign(x) * (exp(|x|) - 1)
            result[original_col_name] = np.sign(df[col]) * np.expm1(np.abs(df[col]))
        
        logger.debug(f"Reversed '{col}' to '{original_col_name}' using {method}")
    
    logger.info(f"Reversed {len(columns)} log-transformed columns")
    
    return result


def get_skewness(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.Series:
    """
    Calculate skewness for numeric columns.
    
    Helper function to identify highly skewed columns that may benefit from
    log-transforms. Typically, columns with |skewness| > 2.0 are considered
    highly skewed.
    
    Args:
        df: Input DataFrame
        columns: Columns to analyze (default: all numeric)
    
    Returns:
        Series with skewness values, sorted by absolute skewness (descending)
        
    Example:
        >>> df = pd.DataFrame({'market_cap': np.random.lognormal(10, 2, 100)})
        >>> skewness = get_skewness(df)
        >>> skewness['market_cap'] > 2.0  # Highly skewed
        True
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    skewness = {}
    for col in columns:
        if col in df.columns:
            # Calculate skewness, dropping NaN values
            col_data = df[col].dropna()
            if len(col_data) > 0:
                skewness[col] = col_data.skew()
    
    result = pd.Series(skewness)
    result = result.reindex(result.abs().sort_values(ascending=False).index)
    
    logger.info(
        f"Calculated skewness for {len(result)} columns. "
        f"Highly skewed (|skew| > 2): {(result.abs() > 2).sum()}"
    )
    
    return result
