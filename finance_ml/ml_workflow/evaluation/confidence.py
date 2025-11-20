"""
Phase 10.2: Prediction Confidence Scoring and Outlier Filtering

This module provides post-prediction outlier detection and confidence scoring
to address Priority 0 issue: extreme outlier problem (mean-median error gap 19x).

Key Features:
- Post-prediction outlier detection (IQR, Z-score, Isolation Forest on errors)
- Prediction confidence scores based on feature completeness and interval width
- Extreme error flagging (>500% percentage error threshold)
- Prediction quality categorization (high/medium/low)
- Separate reporting for high-confidence vs all predictions

Target: Reduce mean-median error gap from 19x to <3x

Integration:
- Works with predictions from regression.models
- Uses interval_width from quantile regression (regression.quantile)
- Exports quality reports for analytics and dashboards

Example:
    >>> from finance_ml.ml_workflow.evaluation.confidence import (
    ...     calculate_prediction_confidence,
    ...     assign_prediction_quality,
    ...     detect_prediction_outliers,
    ...     prediction_quality_report
    ... )
    >>>
    >>> # Calculate confidence scores
    >>> confidence_scores = calculate_prediction_confidence(
    ...     predictions_df,
    ...     feature_completeness_col="feature_completeness",
    ...     interval_width_col="interval_width"
    ... )
    >>>
    >>> # Assign quality categories
    >>> quality_labels = assign_prediction_quality(confidence_scores)
    >>>
    >>> # Detect outliers
    >>> outlier_mask = detect_prediction_outliers(
    ...     predictions_df,
    ...     method="iqr",
    ...     error_col="abs_error"
    ... )
    >>>
    >>> # Generate quality report
    >>> predictions_df["confidence_score"] = confidence_scores
    >>> predictions_df["prediction_quality"] = quality_labels
    >>> report = prediction_quality_report(
    ...     predictions_df,
    ...     quality_col="prediction_quality",
    ...     error_col="abs_error",
    ...     pct_error_col="pct_error"
    ... )

Reference:
- Task 10.2 from finance_ml_improvement_plan.md
- Addresses Model Optimization Recommendations Priority 0
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def calculate_prediction_confidence(
    df: pd.DataFrame,
    feature_completeness_col: str = "feature_completeness",
    interval_width_col: str = "interval_width",
) -> np.ndarray:
    """
    Calculate prediction confidence scores based on feature completeness and interval width.

    Confidence score combines:
    1. Feature completeness: proportion of non-missing features (0-1)
    2. Interval width: narrower intervals = higher confidence

    Formula:
        confidence = feature_completeness * (1 - normalized_interval_width)

    Args:
        df: DataFrame with predictions and metadata
        feature_completeness_col: Column name for feature completeness scores (0-1)
        interval_width_col: Column name for prediction interval widths

    Returns:
        Array of confidence scores in [0, 1] range, same length as df

    Example:
        >>> df = pd.DataFrame({
        ...     "feature_completeness": [0.9, 0.7, 0.5],
        ...     "interval_width": [20, 50, 100]
        ... })
        >>> confidence = calculate_prediction_confidence(df)
        >>> print(confidence)  # Higher completeness + narrower interval = higher confidence
    """
    # Extract feature completeness (already 0-1)
    if feature_completeness_col not in df.columns:
        # Default to 1.0 if not provided
        feature_completeness = np.ones(len(df))
    else:
        feature_completeness = df[feature_completeness_col].values
        # Clip to [0, 1] range
        feature_completeness = np.clip(feature_completeness, 0.0, 1.0)

    # Extract and normalize interval width
    if interval_width_col not in df.columns:
        # Default to 0.5 (medium uncertainty) if not provided
        normalized_width = np.full(len(df), 0.5)
    else:
        interval_width = df[interval_width_col].values

        # Normalize interval width to [0, 1] using min-max scaling
        # Handle edge case: all widths are the same
        if interval_width.max() == interval_width.min():
            normalized_width = np.full(len(df), 0.5)
        else:
            normalized_width = (interval_width - interval_width.min()) / (
                interval_width.max() - interval_width.min()
            )

    # Combine: higher completeness and narrower intervals = higher confidence
    confidence = feature_completeness * (1.0 - normalized_width)

    # Ensure [0, 1] range
    confidence = np.clip(confidence, 0.0, 1.0)

    return confidence


def detect_prediction_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    error_col: str = "abs_error",
    threshold: float = 3.0,
    contamination: float = 0.1,
) -> np.ndarray:
    """
    Detect outliers in prediction errors (not input features).

    This is post-prediction outlier detection to identify catastrophic predictions
    that have extreme errors compared to the distribution of errors.

    Args:
        df: DataFrame with prediction errors
        method: Outlier detection method:
            - "iqr": Interquartile range (robust to outliers)
            - "zscore": Z-score threshold (assumes normality)
            - "isolation_forest": Isolation Forest (unsupervised)
        error_col: Column name for error metric (abs_error or pct_error)
        threshold: Threshold for zscore method (default: 3.0 standard deviations)
        contamination: Expected proportion of outliers for isolation_forest (default: 0.1)

    Returns:
        Boolean array indicating outliers (True = outlier), same length as df

    Example:
        >>> df = pd.DataFrame({"abs_error": [5, 10, 8, 100, 200]})
        >>> outliers = detect_prediction_outliers(df, method="iqr")
        >>> print(outliers)  # [False, False, False, True, True]
    """
    if error_col not in df.columns:
        raise ValueError(f"Column '{error_col}' not found in dataframe")

    errors = df[error_col].values

    if method == "iqr":
        # Interquartile range method
        q1 = np.percentile(errors, 25)
        q3 = np.percentile(errors, 75)
        iqr = q3 - q1

        # Standard IQR outlier detection: values beyond Q1-1.5*IQR or Q3+1.5*IQR
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (errors < lower_bound) | (errors > upper_bound)

    elif method == "zscore":
        # Z-score method
        mean = np.mean(errors)
        std = np.std(errors)

        if std == 0:
            # All errors are the same - no outliers
            outliers = np.zeros(len(errors), dtype=bool)
        else:
            z_scores = np.abs((errors - mean) / std)
            outliers = z_scores > threshold

    elif method == "isolation_forest":
        # Isolation Forest method
        # Reshape for sklearn
        X = errors.reshape(-1, 1)

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X)

        # Isolation Forest returns -1 for outliers, 1 for inliers
        outliers = predictions == -1

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'iqr', 'zscore', 'isolation_forest'"
        )

    logger.info(
        f"Detected {outliers.sum()} outliers ({100*outliers.mean():.1f}%) using {method} method"
    )

    return outliers


def flag_extreme_errors(
    df: pd.DataFrame,
    pct_error_col: str = "pct_error",
    threshold: float = 500.0,
) -> np.ndarray:
    """
    Flag predictions with extreme percentage errors above threshold.

    Args:
        df: DataFrame with percentage errors
        pct_error_col: Column name for percentage error
        threshold: Percentage error threshold (default: 500%)

    Returns:
        Boolean array indicating extreme errors (True = extreme), same length as df

    Example:
        >>> df = pd.DataFrame({"pct_error": [5, 10, 600, 1000]})
        >>> extreme = flag_extreme_errors(df, threshold=500)
        >>> print(extreme)  # [False, False, True, True]
    """
    if pct_error_col not in df.columns:
        raise ValueError(f"Column '{pct_error_col}' not found in dataframe")

    pct_errors = df[pct_error_col].values
    extreme_mask = pct_errors > threshold

    logger.info(
        f"Flagged {extreme_mask.sum()} extreme errors "
        f"({100*extreme_mask.mean():.1f}%) with >{threshold}% error"
    )

    return extreme_mask


def assign_prediction_quality(
    confidence_scores: np.ndarray,
    high_threshold: float = 0.67,
    low_threshold: float = 0.33,
) -> np.ndarray:
    """
    Assign prediction quality categories based on confidence scores.

    Categories:
    - "high": confidence >= high_threshold (default: 0.67)
    - "medium": low_threshold <= confidence < high_threshold
    - "low": confidence < low_threshold (default: 0.33)

    Args:
        confidence_scores: Array of confidence scores in [0, 1]
        high_threshold: Threshold for high quality (default: 0.67)
        low_threshold: Threshold for low quality (default: 0.33)

    Returns:
        Array of quality labels ("high", "medium", "low"), same length as confidence_scores

    Example:
        >>> confidence = np.array([0.9, 0.5, 0.2])
        >>> quality = assign_prediction_quality(confidence)
        >>> print(quality)  # ["high", "medium", "low"]
    """
    quality = np.empty(len(confidence_scores), dtype=object)

    # Assign categories
    quality[confidence_scores >= high_threshold] = "high"
    quality[(confidence_scores >= low_threshold) & (confidence_scores < high_threshold)] = "medium"
    quality[confidence_scores < low_threshold] = "low"

    # Count by category
    high_count = np.sum(quality == "high")
    medium_count = np.sum(quality == "medium")
    low_count = np.sum(quality == "low")

    logger.info(
        f"Assigned quality: {high_count} high ({100*high_count/len(quality):.1f}%), "
        f"{medium_count} medium ({100*medium_count/len(quality):.1f}%), "
        f"{low_count} low ({100*low_count/len(quality):.1f}%)"
    )

    return quality


def filter_low_confidence_predictions(
    df: pd.DataFrame,
    confidence_col: str = "confidence_score",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Filter out low-confidence predictions below threshold.

    Args:
        df: DataFrame with confidence scores
        confidence_col: Column name for confidence scores
        threshold: Minimum confidence threshold (default: 0.5)

    Returns:
        Filtered DataFrame containing only predictions with confidence >= threshold

    Example:
        >>> df = pd.DataFrame({
        ...     "prediction": [100, 200, 300],
        ...     "confidence_score": [0.9, 0.4, 0.6]
        ... })
        >>> filtered = filter_low_confidence_predictions(df, threshold=0.5)
        >>> print(len(filtered))  # 2 (removed the 0.4 confidence prediction)
    """
    if confidence_col not in df.columns:
        raise ValueError(f"Column '{confidence_col}' not found in dataframe")

    filtered_df = df[df[confidence_col] >= threshold].copy()

    removed_count = len(df) - len(filtered_df)
    logger.info(
        f"Filtered {removed_count} low-confidence predictions "
        f"({100*removed_count/len(df):.1f}%) with confidence < {threshold}"
    )

    return filtered_df


def prediction_quality_report(
    df: pd.DataFrame,
    quality_col: str = "prediction_quality",
    error_col: str = "abs_error",
    pct_error_col: str = "pct_error",
) -> Dict[str, Any]:
    """
    Generate comprehensive prediction quality report by quality category.

    Reports statistics for each quality level (high/medium/low):
    - Count of predictions
    - Mean absolute error
    - Median absolute error
    - Mean percentage error
    - Error gap ratio (mean/median)

    Args:
        df: DataFrame with quality labels and errors
        quality_col: Column name for quality labels
        error_col: Column name for absolute errors
        pct_error_col: Column name for percentage errors

    Returns:
        Dictionary with overall and per-quality statistics

    Example:
        >>> report = prediction_quality_report(df)
        >>> print(report["by_quality"]["high"]["error_gap_ratio"])  # Should be <3
    """
    if quality_col not in df.columns:
        raise ValueError(f"Column '{quality_col}' not found in dataframe")
    if error_col not in df.columns:
        raise ValueError(f"Column '{error_col}' not found in dataframe")
    if pct_error_col not in df.columns:
        raise ValueError(f"Column '{pct_error_col}' not found in dataframe")

    report = {"overall": {}, "by_quality": {}}

    # Overall statistics
    report["overall"] = {
        "count": len(df),
        "mean_error": float(df[error_col].mean()),
        "median_error": float(df[error_col].median()),
        "mean_pct_error": float(df[pct_error_col].mean()),
        "error_gap_ratio": float(df[error_col].mean() / df[error_col].median()),
    }

    # Per-quality statistics
    for quality in ["high", "medium", "low"]:
        quality_mask = df[quality_col] == quality
        quality_df = df[quality_mask]

        if len(quality_df) == 0:
            continue

        mean_error = quality_df[error_col].mean()
        median_error = quality_df[error_col].median()

        report["by_quality"][quality] = {
            "count": len(quality_df),
            "mean_error": float(mean_error),
            "median_error": float(median_error),
            "mean_pct_error": float(quality_df[pct_error_col].mean()),
            "error_gap_ratio": (
                float(mean_error / median_error) if median_error > 0 else float("inf")
            ),
        }

    logger.info(
        f"Generated quality report: overall gap {report['overall']['error_gap_ratio']:.2f}x, "
        f"high-quality gap {report['by_quality'].get('high', {}).get('error_gap_ratio', float('inf')):.2f}x"
    )

    return report


def export_quality_report(
    report: Dict[str, Any],
    output_dir: Path,
    filename: str = "prediction_quality_report.csv",
) -> None:
    """
    Export prediction quality report to CSV file.

    Args:
        report: Quality report dictionary from prediction_quality_report()
        output_dir: Directory where CSV file will be saved
        filename: Output filename (default: "prediction_quality_report.csv")

    Example:
        >>> report = prediction_quality_report(df)
        >>> export_quality_report(report, Path("outputs"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert report to DataFrame
    rows = []

    # Add overall row
    overall = report["overall"]
    rows.append(
        {
            "quality": "overall",
            "count": overall["count"],
            "mean_error": overall["mean_error"],
            "median_error": overall["median_error"],
            "mean_pct_error": overall["mean_pct_error"],
            "error_gap_ratio": overall["error_gap_ratio"],
        }
    )

    # Add per-quality rows
    for quality in ["high", "medium", "low"]:
        if quality in report["by_quality"]:
            stats = report["by_quality"][quality]
            rows.append(
                {
                    "quality": quality,
                    "count": stats["count"],
                    "mean_error": stats["mean_error"],
                    "median_error": stats["median_error"],
                    "mean_pct_error": stats["mean_pct_error"],
                    "error_gap_ratio": stats["error_gap_ratio"],
                }
            )

    report_df = pd.DataFrame(rows)

    # Save to CSV
    output_path = output_dir / filename
    report_df.to_csv(output_path, index=False)

    logger.info(f"Exported quality report to {output_path}")
