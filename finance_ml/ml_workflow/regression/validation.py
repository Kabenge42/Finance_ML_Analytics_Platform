"""
Validation functions for regression predictions and metrics.

This module provides validation utilities for:
1. Zero predictions detection
2. Realistic metrics bounds checking (leakage detection)
3. Quantile monotonicity validation and enforcement

Aligned with:
- ml_workflow_guidelines.md v1.1 (Critical Issues)
- code_guidelines.md v1.11 Section 5.4

Version: 1.0
"""

import logging
from typing import Dict, List, Any, Union, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target-related column patterns that indicate potential leakage
TARGET_LEAKAGE_PATTERNS = [
    "price_target",
    "price_target_median",
    "price_target_high",
    "price_target_low",
    "price_target_mean",
    "y_true",
    "target",
    "analyst_target",
]


def validate_no_zero_predictions(
    y_pred: Union[np.ndarray, pd.Series, List[float]],
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """
    Validate that predictions contain no zero values.

    Zero predictions indicate model prediction failures and should be
    flagged for investigation.

    Args:
        y_pred: Predicted values
        tolerance: Values below this threshold are considered zero

    Returns:
        Dictionary with validation results:
        - valid: bool indicating if no zero predictions found
        - zero_count: number of zero predictions
        - zero_indices: list of indices with zero predictions
        - zero_percentage: percentage of zero predictions

    Example:
        >>> y_pred = np.array([100.0, 0.0, 150.0])
        >>> result = validate_no_zero_predictions(y_pred)
        >>> print(result["valid"])  # False
        >>> print(result["zero_count"])  # 1
    """
    if isinstance(y_pred, (list, pd.Series)):
        y_pred = np.array(y_pred)

    zero_mask = np.abs(y_pred) < tolerance
    zero_indices = np.where(zero_mask)[0].tolist()
    zero_count = len(zero_indices)

    result = {
        "valid": zero_count == 0,
        "zero_count": zero_count,
        "zero_indices": zero_indices,
        "zero_percentage": (zero_count / len(y_pred) * 100) if len(y_pred) > 0 else 0.0,
    }

    if zero_count > 0:
        logger.warning(
            f"Found {zero_count} zero predictions ({result['zero_percentage']:.2f}%) "
            f"at indices: {zero_indices[:10]}{'...' if zero_count > 10 else ''}"
        )

    return result


def validate_realistic_metrics(
    metrics: Dict[str, float],
    r2_threshold: float = 0.95,
    mae_min_threshold: float = 1e-6,
) -> Dict[str, Any]:
    """
    Validate that model metrics are realistic (not indicating data leakage).

    Perfect or near-perfect metrics (R²=1.0, MAE=0.0) are impossible for
    real financial prediction and indicate target leakage.

    Args:
        metrics: Dictionary with 'r2', 'mae', 'rmse' keys
        r2_threshold: R² values >= this threshold trigger leakage warning
        mae_min_threshold: MAE values below this threshold trigger leakage warning

    Returns:
        Dictionary with validation results:
        - valid: bool indicating if metrics are realistic
        - suspected_leakage: bool indicating if leakage is suspected
        - issues: dict mapping metric names to issue descriptions

    Example:
        >>> metrics = {"r2": 1.0, "mae": 0.0, "rmse": 0.0}
        >>> result = validate_realistic_metrics(metrics)
        >>> print(result["suspected_leakage"])  # True
    """
    issues = {}
    suspected_leakage = False

    r2 = metrics.get("r2", 0.0)
    mae = metrics.get("mae", float("inf"))
    rmse = metrics.get("rmse", float("inf"))

    # Check R² for unrealistic values
    if r2 >= r2_threshold:
        issues["r2"] = f"R²={r2:.4f} >= {r2_threshold} suggests data leakage"
        suspected_leakage = True
        logger.warning(f"LEAKAGE WARNING: {issues['r2']}")

    # Check MAE for zero or near-zero values
    if mae < mae_min_threshold:
        issues["mae"] = f"MAE={mae:.6f} is near zero, impossible for real prediction"
        suspected_leakage = True
        logger.warning(f"LEAKAGE WARNING: {issues['mae']}")

    # Check RMSE for zero or near-zero values
    if rmse < mae_min_threshold:
        issues["rmse"] = f"RMSE={rmse:.6f} is near zero, impossible for real prediction"
        suspected_leakage = True
        logger.warning(f"LEAKAGE WARNING: {issues['rmse']}")

    # Perfect R² with non-zero MAE is also suspicious
    if r2 == 1.0 and mae > 0:
        issues["inconsistent"] = f"R²=1.0 but MAE={mae:.2f} is inconsistent"
        suspected_leakage = True

    result = {
        "valid": not suspected_leakage,
        "suspected_leakage": suspected_leakage,
        "issues": issues,
        "metrics_checked": {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        },
    }

    return result


def validate_quantile_monotonicity(
    predictions: pd.DataFrame,
    p10_col: str = "pred_p10",
    p50_col: str = "pred_p50",
    p90_col: str = "pred_p90",
) -> Dict[str, Any]:
    """
    Validate that quantile predictions satisfy monotonicity constraint.

    For valid quantile predictions: pred_p10 <= pred_p50 <= pred_p90

    Args:
        predictions: DataFrame with quantile prediction columns
        p10_col: Column name for 10th percentile predictions
        p50_col: Column name for 50th percentile predictions
        p90_col: Column name for 90th percentile predictions

    Returns:
        Dictionary with validation results:
        - valid: bool indicating if monotonicity is satisfied
        - violation_count: number of rows violating monotonicity
        - violation_indices: list of row indices with violations
        - violation_details: dict with specific violation types

    Example:
        >>> preds = pd.DataFrame({
        ...     "pred_p10": [100, 120],  # Row 1 violates p10 <= p50
        ...     "pred_p50": [110, 100],
        ...     "pred_p90": [120, 130],
        ... })
        >>> result = validate_quantile_monotonicity(preds)
        >>> print(result["valid"])  # False
    """
    if p10_col not in predictions.columns:
        return {
            "valid": True,
            "violation_count": 0,
            "violation_indices": [],
            "error": f"Column {p10_col} not found",
        }
    if p50_col not in predictions.columns:
        return {
            "valid": True,
            "violation_count": 0,
            "violation_indices": [],
            "error": f"Column {p50_col} not found",
        }
    if p90_col not in predictions.columns:
        return {
            "valid": True,
            "violation_count": 0,
            "violation_indices": [],
            "error": f"Column {p90_col} not found",
        }

    p10 = predictions[p10_col]
    p50 = predictions[p50_col]
    p90 = predictions[p90_col]

    # Check violations
    p10_gt_p50 = p10 > p50
    p50_gt_p90 = p50 > p90

    any_violation = p10_gt_p50 | p50_gt_p90
    violation_indices = predictions.index[any_violation].tolist()

    result = {
        "valid": len(violation_indices) == 0,
        "violation_count": len(violation_indices),
        "violation_indices": violation_indices,
        "violation_details": {
            "p10_gt_p50_count": int(p10_gt_p50.sum()),
            "p50_gt_p90_count": int(p50_gt_p90.sum()),
        },
    }

    if len(violation_indices) > 0:
        logger.warning(
            f"Quantile monotonicity violated in {len(violation_indices)} rows: "
            f"p10>p50: {result['violation_details']['p10_gt_p50_count']}, "
            f"p50>p90: {result['violation_details']['p50_gt_p90_count']}"
        )

    return result


def enforce_quantile_monotonicity(
    predictions: pd.DataFrame,
    p10_col: str = "pred_p10",
    p50_col: str = "pred_p50",
    p90_col: str = "pred_p90",
    method: str = "sort",
) -> pd.DataFrame:
    """
    Enforce quantile monotonicity constraint on predictions.

    Ensures pred_p10 <= pred_p50 <= pred_p90 for all rows.

    Args:
        predictions: DataFrame with quantile prediction columns
        p10_col: Column name for 10th percentile predictions
        p50_col: Column name for 50th percentile predictions
        p90_col: Column name for 90th percentile predictions
        method: Enforcement method:
            - "sort": Sort quantiles per row (default)
            - "clip": Clip p10 to max(p10, p50), p90 to min(p90, p50)

    Returns:
        DataFrame with monotonicity enforced

    Example:
        >>> preds = pd.DataFrame({
        ...     "pred_p10": [120, 90],  # Row 0 violates
        ...     "pred_p50": [100, 100],
        ...     "pred_p90": [110, 110],
        ... })
        >>> fixed = enforce_quantile_monotonicity(preds)
        >>> print((fixed["pred_p10"] <= fixed["pred_p50"]).all())  # True
    """
    result = predictions.copy()

    if (
        p10_col not in result.columns
        or p50_col not in result.columns
        or p90_col not in result.columns
    ):
        logger.warning("Quantile columns not found, returning original predictions")
        return result

    if method == "sort":
        # Sort quantiles per row to enforce monotonicity
        for idx in result.index:
            values = [result.loc[idx, p10_col], result.loc[idx, p50_col], result.loc[idx, p90_col]]
            sorted_values = sorted(values)
            result.loc[idx, p10_col] = sorted_values[0]
            result.loc[idx, p50_col] = sorted_values[1]
            result.loc[idx, p90_col] = sorted_values[2]

    elif method == "clip":
        # Clip approach: ensure p10 <= p50 <= p90
        # First ensure p50 is between p10 and p90
        result[p50_col] = result[[p10_col, p50_col]].max(axis=1)
        result[p50_col] = result[[p50_col, p90_col]].min(axis=1)
        # Then adjust p10 and p90
        result[p10_col] = result[[p10_col, p50_col]].min(axis=1)
        result[p90_col] = result[[p50_col, p90_col]].max(axis=1)

    # Verify monotonicity is now satisfied
    validation = validate_quantile_monotonicity(result, p10_col, p50_col, p90_col)
    if not validation["valid"]:
        logger.error(
            f"Monotonicity enforcement failed: {validation['violation_count']} violations remain"
        )
    else:
        logger.info("Quantile monotonicity successfully enforced")

    return result


def detect_target_leakage_in_features(
    feature_columns: List[str],
    target_col: str = "price_target",
    additional_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect potential target leakage in feature column names.

    Checks if any feature columns contain patterns that suggest they
    are derived from or related to the target variable.

    Args:
        feature_columns: List of feature column names
        target_col: Name of the target column
        additional_patterns: Additional patterns to check for leakage

    Returns:
        Dictionary with detection results:
        - has_leakage: bool indicating if leakage detected
        - leaky_columns: list of column names with potential leakage
        - patterns_matched: dict mapping columns to matched patterns

    Example:
        >>> features = ["p_e_ratio", "price_target_median", "gross_margin"]
        >>> result = detect_target_leakage_in_features(features)
        >>> print(result["has_leakage"])  # True
        >>> print(result["leaky_columns"])  # ["price_target_median"]
    """
    patterns = TARGET_LEAKAGE_PATTERNS.copy()
    if additional_patterns:
        patterns.extend(additional_patterns)

    # Add target column base name to patterns
    target_base = target_col.lower().replace("_", "")

    leaky_columns = []
    patterns_matched = {}

    for col in feature_columns:
        col_lower = col.lower()

        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in col_lower:
                leaky_columns.append(col)
                patterns_matched[col] = pattern
                break

    result = {
        "has_leakage": len(leaky_columns) > 0,
        "leaky_columns": leaky_columns,
        "patterns_matched": patterns_matched,
        "patterns_checked": patterns,
    }

    if result["has_leakage"]:
        logger.warning(
            f"TARGET LEAKAGE DETECTED: {len(leaky_columns)} columns may leak target information: "
            f"{leaky_columns}"
        )

    return result
