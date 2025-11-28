"""
Data and model validation utilities.

Phase 8: Create missing documented modules.
This module consolidates validation functions from across the codebase into
a centralized location for data quality, schema, and prediction validation.

Functions:
    - validate_data_quality: Check for NaN, infinity, extreme values
    - validate_schema: Validate DataFrame schema (columns, dtypes)
    - validate_numeric_range: Validate numeric values within bounds
    - validate_predictions: Validate prediction outputs (non-negative, quantile order)
    - validate_features: Validate feature quality (constant columns, correlations)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Quality Validation
# =============================================================================


def validate_data_quality(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    extreme_threshold: float = 1e10,
) -> Dict[str, Any]:
    """
    Validate data quality and report issues.

    Checks for:
    - NaN/missing values
    - Infinite values
    - Extremely large values (above threshold)

    Args:
        df: Input DataFrame to validate
        columns: Optional list of columns to validate (defaults to all numeric columns)
        extreme_threshold: Threshold for extreme value detection (default: 1e10)

    Returns:
        Dict with validation results:
            - 'is_valid': bool (True if no issues found)
            - 'has_nulls': bool
            - 'has_inf': bool
            - 'has_extreme_values': bool
            - 'issues': list of issue descriptions

    Example:
        >>> df = pd.DataFrame({'price': [100.0, np.nan, 200.0]})
        >>> result = validate_data_quality(df)
        >>> result['has_nulls']
        True
    """
    issues: List[str] = []
    has_nulls = False
    has_inf = False
    has_extreme_values = False

    # Determine columns to check
    if columns is not None:
        cols_to_check = [c for c in columns if c in df.columns]
    else:
        # Check numeric columns only
        cols_to_check = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols_to_check:
        col_data = df[col]

        # Check for NaN values
        if col_data.isna().any():
            has_nulls = True
            nan_count = col_data.isna().sum()
            issues.append(f"Column '{col}': {nan_count} NaN values")

        # Check for infinities (only for numeric columns)
        if np.issubdtype(col_data.dtype, np.number):
            inf_mask = np.isinf(col_data.dropna())
            if inf_mask.any():
                has_inf = True
                inf_count = inf_mask.sum()
                issues.append(f"Column '{col}': {inf_count} infinite values")

            # Check for extremely large values
            finite_data = col_data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_data) > 0:
                max_val = np.abs(finite_data).max()
                if max_val > extreme_threshold:
                    has_extreme_values = True
                    issues.append(f"Column '{col}': extreme values (max abs={max_val:.2e})")

    if issues:
        logger.warning("⚠️ Data Quality Issues Detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return {
        "is_valid": len(issues) == 0,
        "has_nulls": has_nulls,
        "has_inf": has_inf,
        "has_extreme_values": has_extreme_values,
        "issues": issues,
    }


# =============================================================================
# Schema Validation
# =============================================================================


def validate_schema(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    expected_dtypes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Validate DataFrame schema against requirements.

    Checks for:
    - Presence of required columns
    - Column data types match expectations

    Args:
        df: Input DataFrame to validate
        required_columns: List of columns that must be present
        expected_dtypes: Dict mapping column names to expected dtype strings

    Returns:
        Dict with validation results:
            - 'is_valid': bool
            - 'missing_columns': list of missing column names
            - 'dtype_mismatches': dict of column -> (expected, actual) dtype pairs
            - 'issues': list of issue descriptions

    Example:
        >>> df = pd.DataFrame({'ticker': ['AAPL'], 'price': [150.0]})
        >>> result = validate_schema(df, required_columns=['ticker', 'sector'])
        >>> result['missing_columns']
        ['sector']
    """
    issues: List[str] = []
    missing_columns: List[str] = []
    dtype_mismatches: Dict[str, Tuple[str, str]] = {}

    # Check required columns
    if required_columns is not None:
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
                issues.append(f"Missing required column: '{col}'")

    # Check data types
    if expected_dtypes is not None:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                # Flexible dtype matching (e.g., 'float64' matches 'float')
                if not _dtype_matches(actual_dtype, expected_dtype):
                    dtype_mismatches[col] = (expected_dtype, actual_dtype)
                    issues.append(
                        f"Column '{col}': expected dtype '{expected_dtype}', "
                        f"got '{actual_dtype}'"
                    )

    if issues:
        logger.warning("⚠️ Schema Validation Issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return {
        "is_valid": len(issues) == 0,
        "missing_columns": missing_columns,
        "dtype_mismatches": dtype_mismatches,
        "issues": issues,
    }


def _dtype_matches(actual: str, expected: str) -> bool:
    """Check if actual dtype matches expected (with flexibility)."""
    actual_lower = actual.lower()
    expected_lower = expected.lower()

    # Exact match
    if actual_lower == expected_lower:
        return True

    # Float matches (float64, float32, etc.)
    if "float" in expected_lower and "float" in actual_lower:
        return True

    # Int matches
    if "int" in expected_lower and "int" in actual_lower:
        return True

    # Object/string matches
    if expected_lower in ("object", "string", "str"):
        return actual_lower in ("object", "string", "str")

    return False


# =============================================================================
# Numeric Range Validation
# =============================================================================


def validate_numeric_range(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate that numeric values fall within specified bounds.

    Args:
        df: Input DataFrame
        column: Column name to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Dict with validation results:
            - 'is_valid': bool
            - 'out_of_range_count': number of values outside bounds
            - 'min_value': actual minimum in data
            - 'max_value': actual maximum in data
            - 'issues': list of issue descriptions

    Example:
        >>> df = pd.DataFrame({'price': [-5.0, 100.0, 200.0]})
        >>> result = validate_numeric_range(df, 'price', min_val=0)
        >>> result['out_of_range_count']
        1
    """
    issues: List[str] = []
    out_of_range_count = 0

    if column not in df.columns:
        return {
            "is_valid": False,
            "out_of_range_count": 0,
            "min_value": None,
            "max_value": None,
            "issues": [f"Column '{column}' not found in DataFrame"],
        }

    col_data = df[column].dropna()

    if len(col_data) == 0:
        return {
            "is_valid": True,
            "out_of_range_count": 0,
            "min_value": None,
            "max_value": None,
            "issues": [],
        }

    actual_min = col_data.min()
    actual_max = col_data.max()

    # Check minimum bound
    if min_val is not None:
        below_min = (col_data < min_val).sum()
        if below_min > 0:
            out_of_range_count += below_min
            issues.append(f"Column '{column}': {below_min} values below minimum {min_val}")

    # Check maximum bound
    if max_val is not None:
        above_max = (col_data > max_val).sum()
        if above_max > 0:
            out_of_range_count += above_max
            issues.append(f"Column '{column}': {above_max} values above maximum {max_val}")

    if issues:
        logger.warning("⚠️ Numeric Range Issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return {
        "is_valid": len(issues) == 0,
        "out_of_range_count": out_of_range_count,
        "min_value": actual_min,
        "max_value": actual_max,
        "issues": issues,
    }


# =============================================================================
# Prediction Validation
# =============================================================================


def validate_predictions(
    predictions: pd.DataFrame,
    price_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate prediction outputs for common issues.

    Checks for:
    - Negative price predictions (invalid for stock prices)
    - Quantile crossing (pred_p10 > pred_p50 > pred_p90 order violation)

    Args:
        predictions: DataFrame with prediction columns
        price_columns: List of columns containing price predictions
            (defaults to ['y_pred', 'pred_p10', 'pred_p50', 'pred_p90'])

    Returns:
        Dict with validation results:
            - 'is_valid': bool
            - 'issues': dict of issue_type -> count

    Example:
        >>> preds = pd.DataFrame({'y_pred': [-10.0, 100.0]})
        >>> result = validate_predictions(preds)
        >>> 'negative_predictions' in result['issues']
        True
    """
    issues: Dict[str, int] = {}

    # Default price columns to check
    if price_columns is None:
        price_columns = ["y_pred", "pred_p10", "pred_p50", "pred_p90"]

    # Check for negative predictions
    for col in price_columns:
        if col in predictions.columns:
            negative_count = (predictions[col] < 0).sum()
            if negative_count > 0:
                issues["negative_predictions"] = (
                    issues.get("negative_predictions", 0) + negative_count
                )
                logger.warning(f"⚠️ {negative_count} negative predictions found in '{col}'")

    # Check quantile crossing (p10 <= p50 <= p90)
    quantile_cols = ["pred_p10", "pred_p50", "pred_p90"]
    has_all_quantiles = all(col in predictions.columns for col in quantile_cols)

    if has_all_quantiles:
        # p10 should be <= p50
        crossing_10_50 = (predictions["pred_p10"] > predictions["pred_p50"]).sum()
        # p50 should be <= p90
        crossing_50_90 = (predictions["pred_p50"] > predictions["pred_p90"]).sum()

        total_crossing = crossing_10_50 + crossing_50_90
        if total_crossing > 0:
            issues["quantile_crossing"] = total_crossing
            logger.warning(
                f"⚠️ {total_crossing} quantile crossings detected "
                f"(p10>p50: {crossing_10_50}, p50>p90: {crossing_50_90})"
            )

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
    }


# =============================================================================
# Feature Validation
# =============================================================================


def validate_features(
    df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate feature quality for ML modeling.

    Checks for:
    - Constant columns (zero variance)
    - Highly correlated feature pairs

    Args:
        df: DataFrame with feature columns
        correlation_threshold: Threshold above which correlation is flagged (default: 0.95)
        exclude_columns: Columns to exclude from validation

    Returns:
        Dict with validation results:
            - 'constant_columns': list of columns with zero variance
            - 'high_correlation_pairs': list of (col1, col2, corr) tuples

    Example:
        >>> df = pd.DataFrame({'f1': [1,1,1], 'f2': [1,2,3]})
        >>> result = validate_features(df)
        >>> result['constant_columns']
        ['f1']
    """
    constant_columns: List[str] = []
    high_correlation_pairs: List[Tuple[str, str, float]] = []

    # Determine numeric columns to check
    numeric_df = df.select_dtypes(include=[np.number])

    if exclude_columns:
        numeric_df = numeric_df.drop(
            columns=[c for c in exclude_columns if c in numeric_df.columns],
            errors="ignore",
        )

    if numeric_df.empty:
        return {
            "constant_columns": [],
            "high_correlation_pairs": [],
        }

    # Check for constant columns
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        if len(col_data) > 0 and col_data.std() == 0:
            constant_columns.append(col)

    if constant_columns:
        logger.warning(f"⚠️ Constant columns detected: {constant_columns}")

    # Check for high correlations
    if len(numeric_df.columns) > 1:
        try:
            corr_matrix = numeric_df.corr()

            # Find pairs above threshold
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Upper triangle only
                        corr_val = abs(corr_matrix.loc[col1, col2])
                        if corr_val >= correlation_threshold:
                            high_correlation_pairs.append((col1, col2, corr_val))

            if high_correlation_pairs:
                logger.warning(
                    f"⚠️ {len(high_correlation_pairs)} highly correlated feature pairs detected"
                )
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}")

    return {
        "constant_columns": constant_columns,
        "high_correlation_pairs": high_correlation_pairs,
    }


__all__ = [
    "validate_data_quality",
    "validate_schema",
    "validate_numeric_range",
    "validate_predictions",
    "validate_features",
]
