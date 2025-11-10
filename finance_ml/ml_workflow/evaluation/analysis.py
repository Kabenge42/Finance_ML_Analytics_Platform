"""
Phase 9.6: Model Evaluation Analysis Module

Residual analysis, error analysis, and model diagnostics for regression models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def residual_analysis(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sector: Optional[Union[np.ndarray, pd.Series, List]] = None,
) -> Dict[str, Any]:
    """
    Perform residual analysis on model predictions.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        sector: Optional sector labels for grouped analysis

    Returns:
        Dictionary containing:
        - residuals: Array of residuals (y_true - y_pred)
        - mean_residual: Mean of residuals
        - std_residual: Standard deviation of residuals
        - mae: Mean absolute error
        - rmse: Root mean squared error
        - mape: Mean absolute percentage error
        - by_sector: Sector-wise statistics (if sector provided)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    result = {
        "residuals": residuals,
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }

    # MAPE (avoid division by zero)
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        result["mape"] = float(mape)
    else:
        result["mape"] = None

    # Sector-wise analysis
    if sector is not None:
        sector = np.asarray(sector) if not isinstance(sector, (pd.Series, np.ndarray)) else sector
        by_sector = {}

        unique_sectors = np.unique(sector)
        for sec in unique_sectors:
            mask = sector == sec
            sector_residuals = residuals[mask]
            by_sector[str(sec)] = {
                "count": int(np.sum(mask)),
                "mean_residual": float(np.mean(sector_residuals)),
                "std_residual": float(np.std(sector_residuals)),
                "mae": float(np.mean(np.abs(sector_residuals))),
                "rmse": float(np.sqrt(np.mean(sector_residuals**2))),
            }

        result["by_sector"] = by_sector

    return result


def error_analysis(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    bins: int = 10,
) -> Dict[str, Any]:
    """
    Analyze error distributions and patterns.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        bins: Number of bins for error histogram

    Returns:
        Dictionary containing:
        - error_distribution: Histogram of errors
        - percentiles: Error percentiles (5, 25, 50, 75, 95)
        - abs_error_stats: Statistics on absolute errors
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Error distribution histogram
    hist_counts, bin_edges = np.histogram(errors, bins=bins)

    # Percentiles
    percentiles = {
        "p5": float(np.percentile(errors, 5)),
        "p25": float(np.percentile(errors, 25)),
        "p50": float(np.percentile(errors, 50)),
        "p75": float(np.percentile(errors, 75)),
        "p95": float(np.percentile(errors, 95)),
    }

    result = {
        "error_distribution": {
            "counts": hist_counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
        "percentiles": percentiles,
        "abs_error_stats": {
            "mean": float(np.mean(abs_errors)),
            "median": float(np.median(abs_errors)),
            "std": float(np.std(abs_errors)),
            "min": float(np.min(abs_errors)),
            "max": float(np.max(abs_errors)),
        },
    }

    return result


def model_diagnostics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    features: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Comprehensive model diagnostics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        features: Optional feature matrix for additional analysis

    Returns:
        Dictionary with diagnostic metrics including R², adjusted R², etc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    result = {
        "r2_score": float(r2),
        "n_samples": len(y_true),
        "y_true_mean": float(np.mean(y_true)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_pred_std": float(np.std(y_pred)),
    }

    # Adjusted R² if features provided
    if features is not None:
        n = len(y_true)
        p = features.shape[1]
        if n > p + 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            result["adjusted_r2"] = float(adj_r2)

    return result


def prediction_intervals(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Calculate prediction intervals based on residual distribution.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        confidence: Confidence level (default 0.95 for 95% interval)

    Returns:
        Dictionary with lower and upper bounds
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    std_resid = np.std(residuals)

    # Z-score for confidence level
    from scipy import stats

    z = stats.norm.ppf((1 + confidence) / 2)

    margin = z * std_resid

    return {
        "lower_bound": y_pred - margin,
        "upper_bound": y_pred + margin,
        "confidence": confidence,
    }


def cross_validation_analysis(
    cv_results: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Analyze cross-validation results.

    Args:
        cv_results: Dictionary of metric names to lists of fold scores

    Returns:
        Summary statistics for each metric
    """
    summary = {}

    for metric_name, scores in cv_results.items():
        scores_array = np.array(scores)
        summary[metric_name] = {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
        }

    return summary
