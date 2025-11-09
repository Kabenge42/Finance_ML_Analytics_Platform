"""
Evaluation metrics for regression models.

This module provides comprehensive metrics calculation for model evaluation,
including basic metrics (MAE, RMSE, R²), segment-specific metrics, and
sector/region breakdowns.

Phase 9.6 - Evaluation Refactor
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


def comprehensive_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Calculate comprehensive regression metrics with NaN handling.

    Computes MAE, RMSE, MAPE, R², Median Absolute Error, and Max Error.
    Handles NaN and infinite values gracefully by removing them before computation.

    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values

    Returns:
        dict: Dictionary containing all metrics

    Metrics:
        - mae: Mean Absolute Error (interpretable dollar error)
        - rmse: Root Mean Squared Error (penalizes large errors)
        - mape: Mean Absolute Percentage Error (relative error)
        - r2: R² coefficient of determination (variance explained)
        - median_ae: Median Absolute Error (robust to outliers)
        - max_error: Maximum absolute error (worst-case performance)
        - n_samples: Number of valid samples used for computation

    Example:
        >>> y_true = np.array([100, 150, 200, 250])
        >>> y_pred = np.array([110, 145, 205, 240])
        >>> metrics = comprehensive_regression_metrics(y_true, y_pred)
        >>> print(f"MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        median_absolute_error,
        max_error as sklearn_max_error,
    )

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for NaN values
    nan_mask_true = np.isnan(y_true)
    nan_mask_pred = np.isnan(y_pred)
    nan_mask = nan_mask_true | nan_mask_pred

    if nan_mask.any():
        n_nans = nan_mask.sum()
        logger.warning(f"Found {n_nans} NaN values ({n_nans/len(y_true)*100:.2f}% of data)")
        logger.warning(f"  - NaN in y_true: {nan_mask_true.sum()}")
        logger.warning(f"  - NaN in y_pred: {nan_mask_pred.sum()}")

        # Remove NaN values
        valid_mask = ~nan_mask
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        logger.warning(f"  - Remaining valid samples: {len(y_true)}")

    # Check if we have enough valid samples
    if len(y_true) < 2:
        logger.error("Not enough valid samples to compute metrics")
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "median_ae": np.nan,
            "max_error": np.nan,
            "n_samples": len(y_true),
        }

    # Check for infinite values
    inf_mask = np.isinf(y_true) | np.isinf(y_pred)
    if inf_mask.any():
        logger.warning(f"Found {inf_mask.sum()} infinite values, removing them")
        valid_mask = ~inf_mask
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    # Recheck sample count after removing infinities
    if len(y_true) < 2:
        logger.error("Not enough valid samples after removing infinities")
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "median_ae": np.nan,
            "max_error": np.nan,
            "n_samples": len(y_true),
        }

    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    max_err = sklearn_max_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error) - handle division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if np.isnan(mape) or np.isinf(mape):
            mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else np.inf,
        "r2": float(r2),
        "median_ae": float(median_ae),
        "max_error": float(max_err),
        "n_samples": len(y_true),
    }


def compute_metrics_by_segment(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, segment_col: str
) -> pd.DataFrame:
    """
    Compute regression metrics for each segment (sector, region, etc.).

    Args:
        df: DataFrame with predictions and actuals
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        segment_col: Column name for segmentation (e.g., 'sector', 'region')

    Returns:
        DataFrame with metrics by segment

    Example:
        >>> df = pd.DataFrame({
        ...     'sector': ['Tech', 'Tech', 'Finance', 'Finance'],
        ...     'actual': [100, 150, 200, 250],
        ...     'predicted': [110, 145, 205, 240]
        ... })
        >>> metrics = compute_metrics_by_segment(df, 'actual', 'predicted', 'sector')
        >>> print(metrics)
    """
    segments = df[segment_col].unique()
    results = []

    for segment in segments:
        segment_df = df[df[segment_col] == segment]
        y_true = segment_df[y_true_col]
        y_pred = segment_df[y_pred_col]

        metrics = comprehensive_regression_metrics(y_true, y_pred)
        metrics[segment_col] = segment
        results.append(metrics)

    return pd.DataFrame(results)


def compute_sector_region_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    sector_col: str = "sector",
    region_col: str = "region",
) -> pd.DataFrame:
    """
    Compute metrics for each sector-region combination.

    Args:
        df: DataFrame with predictions and metadata
        y_true_col: Column name for true values
        y_pred_col: Column name for predictions
        sector_col: Column name for sector grouping
        region_col: Column name for region grouping

    Returns:
        DataFrame with metrics by sector and region

    Example:
        >>> df = pd.DataFrame({
        ...     'sector': ['Tech', 'Tech', 'Finance', 'Finance'],
        ...     'region': ['US', 'EU', 'US', 'EU'],
        ...     'actual': [100, 150, 200, 250],
        ...     'predicted': [110, 145, 205, 240]
        ... })
        >>> metrics = compute_sector_region_metrics(df, 'actual', 'predicted')
    """
    groups = df.groupby([sector_col, region_col])
    results = []

    for (sector, region), group_df in groups:
        if len(group_df) < 2:
            continue

        y_true = group_df[y_true_col]
        y_pred = group_df[y_pred_col]

        metrics = comprehensive_regression_metrics(y_true, y_pred)
        metrics[sector_col] = sector
        metrics[region_col] = region
        results.append(metrics)

    return pd.DataFrame(results)
