"""
Prediction constraints for regression models.

This module provides wrappers and utilities to enforce constraints on regression
model predictions, particularly non-negativity constraints for financial predictions
like stock prices and price targets.

Phase 9.5 - Regression Refactor

Enhanced in v1.1 (2025-12-13) with:
- Zero prediction detection and handling
- Minimum threshold enforcement
- Fallback mechanism for zero predictions
- Post-prediction validation checkpoint
- Quantile monotonicity validation and enforcement
- EnhancedNonNegativeWrapper with tracking

Aligned with ml_workflow_guidelines.md and code_guidelines.md v1.10.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Zero Prediction Detection
# =============================================================================


def detect_zero_predictions(predictions: np.ndarray) -> Dict[str, Any]:
    """
    Detect zero predictions in model output.

    Zero predictions indicate model prediction failures and should be
    identified for correction. This is critical for financial price
    predictions where zero is not a valid stock price.

    Args:
        predictions: Array of model predictions

    Returns:
        Dict with:
            - count: Number of zero predictions
            - percentage: Percentage of predictions that are zero
            - indices: Array of indices where predictions are zero

    Example:
        >>> preds = np.array([100.0, 0.0, 50.0, 0.0])
        >>> result = detect_zero_predictions(preds)
        >>> result['count']
        2
    """
    predictions = np.asarray(predictions)
    zero_mask = predictions == 0.0
    indices = np.where(zero_mask)[0]
    count = len(indices)
    percentage = 100.0 * count / len(predictions) if len(predictions) > 0 else 0.0

    if count > 0:
        logger.warning(f"Detected {count}/{len(predictions)} ({percentage:.1f}%) zero predictions")

    return {
        "count": count,
        "percentage": percentage,
        "indices": indices,
    }


# =============================================================================
# Minimum Threshold Enforcement
# =============================================================================


def enforce_minimum_threshold(
    predictions: np.ndarray,
    min_value: float = 0.01,
) -> np.ndarray:
    """
    Enforce minimum prediction threshold.

    Stock prices cannot be zero or negative. This function ensures all
    predictions are at least min_value (default: 0.01, representing one cent).

    Args:
        predictions: Array of model predictions
        min_value: Minimum allowed value (default: 0.01)

    Returns:
        Array with all values >= min_value

    Example:
        >>> preds = np.array([100.0, 0.0, -5.0, 0.005])
        >>> enforce_minimum_threshold(preds, min_value=0.01)
        array([100.  ,   0.01,   0.01,   0.01])
    """
    predictions = np.asarray(predictions).copy()
    below_threshold = predictions < min_value
    n_corrected = np.sum(below_threshold)

    if n_corrected > 0:
        logger.debug(
            f"Enforcing minimum threshold: {n_corrected}/{len(predictions)} "
            f"predictions raised to {min_value}"
        )

    predictions[below_threshold] = min_value
    return predictions


# =============================================================================
# Zero Prediction Fallback
# =============================================================================


def apply_zero_prediction_fallback(
    predictions: np.ndarray,
    last_prices: np.ndarray,
    fallback_factor: float = 1.05,
    min_fallback: float = 0.01,
) -> np.ndarray:
    """
    Apply fallback for zero predictions using last_price.

    When a prediction is zero, use last_price * fallback_factor as the
    predicted value. This assumes a small positive return from the last
    known price.

    Args:
        predictions: Array of model predictions
        last_prices: Array of last known prices for each stock
        fallback_factor: Multiplier for last_price (default: 1.05 = 5% increase)
        min_fallback: Minimum fallback value when last_price is NaN (default: 0.01)

    Returns:
        Array with zero predictions replaced by fallback values

    Example:
        >>> preds = np.array([100.0, 0.0, 50.0])
        >>> prices = np.array([95.0, 80.0, 45.0])
        >>> apply_zero_prediction_fallback(preds, prices)
        array([100. ,  84. ,  50. ])
    """
    predictions = np.asarray(predictions).copy()
    last_prices = np.asarray(last_prices)

    zero_mask = predictions == 0.0
    n_zeros = np.sum(zero_mask)

    if n_zeros == 0:
        return predictions

    # Calculate fallback values
    fallback_values = last_prices * fallback_factor

    # Handle NaN last_prices - use minimum fallback
    nan_price_mask = np.isnan(last_prices)
    fallback_values[nan_price_mask] = min_fallback

    # Apply fallback only where predictions are zero
    predictions[zero_mask] = fallback_values[zero_mask]

    logger.info(
        f"Applied zero prediction fallback: {n_zeros} predictions "
        f"replaced using last_price * {fallback_factor}"
    )

    return predictions


# =============================================================================
# Post-Prediction Validation
# =============================================================================


def validate_predictions(
    predictions: np.ndarray,
    warn_extreme: bool = False,
    extreme_threshold: float = 10.0,
) -> Dict[str, Any]:
    """
    Comprehensive post-prediction validation checkpoint.

    Validates predictions for common issues:
    - Zero predictions (invalid for stock prices)
    - Negative predictions (impossible for stock prices)
    - Extreme predictions (optional warning)

    Args:
        predictions: Array of model predictions
        warn_extreme: If True, warn about extreme values (default: False)
        extreme_threshold: Multiplier of median for extreme detection (default: 10.0)

    Returns:
        Dict with:
            - valid: Boolean indicating if all validations passed
            - issues: List of validation failures
            - warnings: List of validation warnings (non-fatal)
            - stats: Summary statistics

    Example:
        >>> preds = np.array([100.0, 0.0, 50.0])
        >>> result = validate_predictions(preds)
        >>> result['valid']
        False
        >>> 'zero_predictions' in result['issues']
        True
    """
    predictions = np.asarray(predictions)
    issues = []
    warnings = []

    # Check for zeros
    zero_count = np.sum(predictions == 0.0)
    if zero_count > 0:
        issues.append("zero_predictions")
        logger.error(f"Validation failed: {zero_count} zero predictions detected")

    # Check for negatives
    negative_count = np.sum(predictions < 0)
    if negative_count > 0:
        issues.append("negative_predictions")
        logger.error(f"Validation failed: {negative_count} negative predictions detected")

    # Check for extreme values (optional warning)
    if warn_extreme and len(predictions) > 0:
        median_pred = np.median(predictions[predictions > 0]) if np.any(predictions > 0) else 1.0
        extreme_mask = predictions > (median_pred * extreme_threshold)
        extreme_count = np.sum(extreme_mask)
        if extreme_count > 0:
            warnings.append("extreme_predictions")
            logger.warning(
                f"Validation warning: {extreme_count} predictions exceed "
                f"{extreme_threshold}x median ({median_pred:.2f})"
            )

    # Calculate stats
    valid_preds = predictions[predictions > 0]
    stats = {
        "total": len(predictions),
        "zeros": zero_count,
        "negatives": negative_count,
        "mean": float(np.mean(valid_preds)) if len(valid_preds) > 0 else 0.0,
        "median": float(np.median(valid_preds)) if len(valid_preds) > 0 else 0.0,
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
    }

    valid = len(issues) == 0

    if valid:
        logger.info("Prediction validation passed")
    else:
        logger.error(f"Prediction validation failed with issues: {issues}")

    return {
        "valid": valid,
        "issues": issues,
        "warnings": warnings,
        "stats": stats,
    }


# =============================================================================
# Target Leakage Detection
# =============================================================================


def detect_target_leakage(
    feature_columns: List[str],
    target_patterns: Optional[List[str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Detect potential target leakage in feature columns.

    Target leakage occurs when features contain information about the target
    variable that would not be available at prediction time. This is a critical
    issue that can lead to unrealistically high model performance (R²=1.0, MAE=0).

    Args:
        feature_columns: List of feature column names to check
        target_patterns: List of patterns to search for (default: price_target related)
        strict: If True, raise ValueError when leakage detected; if False, return warning

    Returns:
        Dict with:
            - has_leakage: Boolean indicating if potential leakage detected
            - leaky_features: List of feature names that may cause leakage
            - severity: 'critical', 'warning', or 'none'
            - message: Human-readable description

    Example:
        >>> features = ['market_cap', 'price_target', 'p_e_ratio']
        >>> result = detect_target_leakage(features)
        >>> result['has_leakage']
        True
        >>> result['leaky_features']
        ['price_target']

    Raises:
        ValueError: If strict=True and leakage is detected
    """
    if target_patterns is None:
        # Default patterns that indicate target-related columns
        target_patterns = [
            "price_target",
            "target_price",
            "y_true",
            "y_pred",
            "predicted",
            "actual_target",
            "future_price",
            "forward_return",
        ]

    feature_columns_lower = [col.lower() for col in feature_columns]
    leaky_features = []

    for col, col_lower in zip(feature_columns, feature_columns_lower):
        for pattern in target_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in col_lower:
                leaky_features.append(col)
                break

    has_leakage = len(leaky_features) > 0

    if has_leakage:
        severity = "critical"
        message = (
            f"Target leakage detected! {len(leaky_features)} feature(s) may contain "
            f"target information: {leaky_features}. This can cause unrealistic model "
            f"performance (R²=1.0, MAE=0). Remove these features before training."
        )
        logger.error(message)

        if strict:
            raise ValueError(message)
    else:
        severity = "none"
        message = "No target leakage detected in feature columns."
        logger.info(message)

    return {
        "has_leakage": has_leakage,
        "leaky_features": leaky_features,
        "severity": severity,
        "message": message,
        "features_checked": len(feature_columns),
        "patterns_used": target_patterns,
    }


def audit_features_for_training(
    X: "pd.DataFrame",
    target_col: str = "price_target",
    additional_exclusions: Optional[List[str]] = None,
    auto_remove: bool = False,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """
    Audit and optionally clean features before model training.

    This function performs comprehensive feature auditing to prevent target leakage
    and ensure data quality for regression models.

    Args:
        X: Feature DataFrame
        target_col: Name of the target column (to ensure it's not in features)
        additional_exclusions: Additional column patterns to exclude
        auto_remove: If True, automatically remove problematic features

    Returns:
        Tuple of (cleaned_X, audit_report)

    Example:
        >>> X_clean, report = audit_features_for_training(X_train, target_col='price_target')
        >>> if report['has_issues']:
        ...     print(f"Removed {len(report['removed_features'])} problematic features")
    """
    import pandas as pd

    audit_report = {
        "original_features": len(X.columns),
        "has_issues": False,
        "removed_features": [],
        "warnings": [],
    }

    # Check for target leakage
    leakage_result = detect_target_leakage(list(X.columns))
    if leakage_result["has_leakage"]:
        audit_report["has_issues"] = True
        audit_report["leakage_detected"] = leakage_result["leaky_features"]
        audit_report["warnings"].append(leakage_result["message"])

    # Check if target column is in features
    if target_col in X.columns:
        audit_report["has_issues"] = True
        audit_report["warnings"].append(
            f"Target column '{target_col}' found in features - this will cause leakage!"
        )
        if target_col not in leakage_result.get("leaky_features", []):
            leakage_result["leaky_features"] = leakage_result.get("leaky_features", []) + [
                target_col
            ]

    # Build exclusion list
    exclusions = set(leakage_result.get("leaky_features", []))
    if additional_exclusions:
        for pattern in additional_exclusions:
            for col in X.columns:
                if pattern.lower() in col.lower():
                    exclusions.add(col)

    # Auto-remove if requested
    X_clean = X
    if auto_remove and exclusions:
        cols_to_remove = [col for col in exclusions if col in X.columns]
        X_clean = X.drop(columns=cols_to_remove, errors="ignore")
        audit_report["removed_features"] = cols_to_remove
        logger.warning(f"Auto-removed {len(cols_to_remove)} features: {cols_to_remove}")

    audit_report["final_features"] = len(X_clean.columns)

    return X_clean, audit_report


# =============================================================================
# Overfitting Detection
# =============================================================================


def detect_overfitting(
    metrics: Dict[str, float],
    r2_threshold: float = 0.95,
    r2_warning_threshold: float = 0.90,
    mae_min: float = 1.0,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Detect potential overfitting or data leakage based on model metrics.

    Per ml_workflow_guidelines.md, these metrics indicate potential issues:
    - R² >= 0.95 suggests data leakage (expected: 0.60-0.85 for financial data)
    - MAE = 0 is impossible for real financial prediction
    - R² = 1.0 with MAE = 0 is the classic leakage pattern

    Args:
        metrics: Dict containing model metrics (r2, mae, rmse, etc.)
        r2_threshold: R² threshold for critical warning (default: 0.95)
        r2_warning_threshold: R² threshold for warning (default: 0.90)
        mae_min: Minimum expected MAE for real data (default: 1.0)
        strict: If True, raise ValueError when overfitting detected

    Returns:
        Dict with:
            - has_overfitting: Boolean indicating if overfitting detected
            - warnings: List of warning types ('r2_too_high', 'mae_zero')
            - severity: 'critical', 'warning', or 'none'
            - message: Human-readable description
            - recommendations: List of recommended actions

    Example:
        >>> metrics = {'r2': 1.0, 'mae': 0.0, 'rmse': 0.0}
        >>> result = detect_overfitting(metrics)
        >>> result['has_overfitting']
        True
        >>> result['severity']
        'critical'

    Raises:
        ValueError: If strict=True and overfitting is detected
    """
    warnings_list = []
    recommendations = []
    severity = "none"

    r2 = metrics.get("r2", metrics.get("r2_score", 0.0))
    mae = metrics.get("mae", metrics.get("mean_absolute_error", float("inf")))

    # Check for suspiciously high R²
    if r2 >= r2_threshold:
        warnings_list.append("r2_too_high")
        severity = "critical"
        recommendations.append(
            f"R² = {r2:.4f} is suspiciously high (>= {r2_threshold}). "
            "Audit features for target leakage."
        )
        logger.error(f"Overfitting detected: R² = {r2:.4f} >= {r2_threshold}")
    elif r2 >= r2_warning_threshold:
        warnings_list.append("r2_too_high")
        severity = "warning"
        recommendations.append(
            f"R² = {r2:.4f} is borderline high ({r2_warning_threshold}-{r2_threshold}). "
            "Validate on held-out test set."
        )
        logger.warning(f"Potential overfitting: R² = {r2:.4f} >= {r2_warning_threshold}")

    # Check for zero or near-zero MAE
    if mae < mae_min:
        warnings_list.append("mae_zero")
        severity = "critical"  # MAE=0 is always critical
        recommendations.append(
            f"MAE = {mae:.4f} is impossibly low (< {mae_min}). "
            "Check if target column is included in features."
        )
        logger.error(f"Overfitting detected: MAE = {mae:.4f} < {mae_min}")

    has_overfitting = len(warnings_list) > 0

    if has_overfitting:
        message = (
            f"Overfitting detected! {len(warnings_list)} warning(s): {warnings_list}. "
            f"This indicates potential data leakage. "
            f"Recommendations: {'; '.join(recommendations)}"
        )
        if strict:
            raise ValueError(message)
    else:
        message = "No overfitting detected. Metrics are within expected ranges."
        logger.info(message)

    return {
        "has_overfitting": has_overfitting,
        "warnings": warnings_list,
        "severity": severity,
        "message": message,
        "recommendations": recommendations,
        "metrics_checked": {
            "r2": r2,
            "mae": mae,
            "r2_threshold": r2_threshold,
            "mae_min": mae_min,
        },
    }


def validate_model_metrics(
    metrics: Dict[str, float],
    check_overfitting: bool = True,
    check_negative_r2: bool = True,
    check_extreme_mape: bool = True,
    mape_threshold: float = 100.0,
) -> Dict[str, Any]:
    """
    Comprehensive validation of model metrics.

    Performs multiple checks to ensure model metrics are realistic and
    don't indicate data quality issues or overfitting.

    Args:
        metrics: Dict containing model metrics (r2, mae, rmse, mape, etc.)
        check_overfitting: Whether to check for overfitting (default: True)
        check_negative_r2: Whether to check for negative R² (default: True)
        check_extreme_mape: Whether to check for extreme MAPE (default: True)
        mape_threshold: MAPE threshold for warning (default: 100.0)

    Returns:
        Dict with:
            - valid: Boolean indicating if all checks pass
            - issues: List of issue types found
            - details: Detailed results from each check
            - summary: Human-readable summary

    Example:
        >>> metrics = {'r2': 0.72, 'mae': 1200.0, 'rmse': 2500.0, 'mape': 15.5}
        >>> result = validate_model_metrics(metrics)
        >>> result['valid']
        True
    """
    issues = []
    details = {}

    # Check for overfitting
    if check_overfitting:
        overfitting_result = detect_overfitting(metrics)
        details["overfitting"] = overfitting_result
        if overfitting_result["has_overfitting"]:
            issues.append("overfitting")

    # Check for negative R²
    if check_negative_r2:
        r2 = metrics.get("r2", metrics.get("r2_score", 0.0))
        if r2 < 0:
            issues.append("negative_r2")
            details["negative_r2"] = {
                "r2": r2,
                "message": f"Negative R² ({r2:.4f}) indicates model performs worse than mean baseline.",
            }
            logger.warning(f"Negative R² detected: {r2:.4f}")

    # Check for extreme MAPE
    if check_extreme_mape:
        mape = metrics.get("mape", metrics.get("mean_absolute_percentage_error", 0.0))
        if mape > mape_threshold:
            issues.append("extreme_mape")
            details["extreme_mape"] = {
                "mape": mape,
                "threshold": mape_threshold,
                "message": f"MAPE ({mape:.2f}%) exceeds threshold ({mape_threshold}%).",
            }
            logger.warning(f"Extreme MAPE detected: {mape:.2f}%")

    valid = len(issues) == 0

    summary = (
        "All model metrics are within expected ranges."
        if valid
        else f"Model metrics validation failed with {len(issues)} issue(s): {issues}"
    )

    return {
        "valid": valid,
        "issues": issues,
        "details": details,
        "summary": summary,
        "metrics_provided": list(metrics.keys()),
    }


# =============================================================================
# Quantile Monotonicity Validation
# =============================================================================


def validate_quantile_monotonicity(
    pred_p10: np.ndarray,
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate quantile monotonicity constraint.

    For valid quantile predictions, we must have:
    pred_p10 <= pred_p50 <= pred_p90

    Args:
        pred_p10: 10th percentile predictions
        pred_p50: 50th percentile predictions (median)
        pred_p90: 90th percentile predictions

    Returns:
        Dict with:
            - valid: Boolean indicating if monotonicity holds
            - violations: Count of rows violating monotonicity
            - violation_indices: Indices of violating rows

    Example:
        >>> p10 = np.array([90.0, 60.0])
        >>> p50 = np.array([100.0, 50.0])  # Row 1: 60 > 50 violation
        >>> p90 = np.array([110.0, 60.0])
        >>> result = validate_quantile_monotonicity(p10, p50, p90)
        >>> result['valid']
        False
    """
    pred_p10 = np.asarray(pred_p10)
    pred_p50 = np.asarray(pred_p50)
    pred_p90 = np.asarray(pred_p90)

    # Check p10 <= p50
    violation_10_50 = pred_p10 > pred_p50
    # Check p50 <= p90
    violation_50_90 = pred_p50 > pred_p90

    # Combined violations
    any_violation = violation_10_50 | violation_50_90
    violation_indices = np.where(any_violation)[0]
    violations = len(violation_indices)

    if violations > 0:
        logger.warning(f"Quantile monotonicity violated in {violations}/{len(pred_p10)} rows")

    return {
        "valid": violations == 0,
        "violations": violations,
        "violation_indices": violation_indices,
    }


def enforce_quantile_monotonicity(
    pred_p10: np.ndarray,
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce quantile monotonicity by adjusting predictions.

    Uses isotonic regression approach: sort the quantiles and redistribute
    to maintain monotonicity while staying close to original predictions.

    Args:
        pred_p10: 10th percentile predictions
        pred_p50: 50th percentile predictions (median)
        pred_p90: 90th percentile predictions

    Returns:
        Tuple of (fixed_p10, fixed_p50, fixed_p90) arrays

    Example:
        >>> p10 = np.array([90.0, 60.0])
        >>> p50 = np.array([100.0, 50.0])  # Row 1: 60 > 50 violation
        >>> p90 = np.array([110.0, 45.0])  # Row 1: 45 < 50 violation
        >>> p10_f, p50_f, p90_f = enforce_quantile_monotonicity(p10, p50, p90)
        >>> (p10_f <= p50_f).all() and (p50_f <= p90_f).all()
        True
    """
    pred_p10 = np.asarray(pred_p10).copy()
    pred_p50 = np.asarray(pred_p50).copy()
    pred_p90 = np.asarray(pred_p90).copy()

    n_fixed = 0

    for i in range(len(pred_p10)):
        # Stack the three quantiles
        quantiles = np.array([pred_p10[i], pred_p50[i], pred_p90[i]])

        # Check if already monotonic
        if quantiles[0] <= quantiles[1] <= quantiles[2]:
            continue

        # Sort to enforce monotonicity
        sorted_quantiles = np.sort(quantiles)

        pred_p10[i] = sorted_quantiles[0]
        pred_p50[i] = sorted_quantiles[1]
        pred_p90[i] = sorted_quantiles[2]
        n_fixed += 1

    if n_fixed > 0:
        logger.info(f"Enforced quantile monotonicity on {n_fixed} rows")

    return pred_p10, pred_p50, pred_p90


class NonNegativeRegressionWrapper:
    """
    Wrapper for regression models that ensures predictions are non-negative.

    This wrapper clips predictions to be >= 0, which is essential for price
    target predictions since stock prices cannot be negative. Linear regression
    (Ridge, Lasso, ElasticNet) can produce negative predictions without
    constraints, especially when features have extreme values or the model
    is poorly regularized.

    The wrapper applies post-prediction clipping using np.maximum(pred, 0.0),
    which is computationally efficient and maintains differentiability at
    the boundary.

    Args:
        base_model: Any sklearn-compatible regression model

    Attributes:
        base_model: The wrapped regression model

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create training data
        >>> X = pd.DataFrame({'feature1': np.random.randn(100)})
        >>> y = pd.Series(np.abs(np.random.randn(100)) * 10 + 5)
        >>>
        >>> # Train with non-negative constraint
        >>> base = Ridge(alpha=1.0)
        >>> model = NonNegativeRegressionWrapper(base)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> assert (predictions >= 0).all()  # All predictions >= 0

    Phase 9.5 TDD Implementation:
        This class was implemented following strict TDD to solve the critical
        issue of negative price target predictions observed in production regression.
    """

    def __init__(self, base_model):
        """
        Initialize wrapper with base regression model.

        Args:
            base_model: sklearn-compatible regression model (must have fit and predict methods)
        """
        self.base_model = base_model

    def fit(self, X, y):
        """
        Fit the base model.

        Args:
            X: Feature matrix (pandas DataFrame or numpy array)
            y: Target vector (pandas Series or numpy array)

        Returns:
            self (for method chaining)
        """
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict and ensure all predictions are non-negative.

        This method:
        1. Gets predictions from base model
        2. Clips predictions to be >= 0 using np.maximum
        3. Returns clipped predictions

        Args:
            X: Feature matrix (pandas DataFrame or numpy array)

        Returns:
            Non-negative predictions (numpy array with all values >= 0)

        Note:
            The clipping operation is applied element-wise and has minimal
            performance overhead. For most financial regression, less than 5% of
            predictions require clipping.
        """
        predictions = self.base_model.predict(X)

        # Count how many predictions would be negative (for monitoring)
        n_negative = np.sum(predictions < 0)
        if n_negative > 0:
            pct_negative = 100.0 * n_negative / len(predictions)
            logger.debug(
                f"NonNegativeRegressionWrapper: Clipped {n_negative}/{len(predictions)} "
                f"({pct_negative:.1f}%) negative predictions to 0"
            )

        # Clip predictions to ensure they're >= 0
        return np.maximum(predictions, 0.0)

    def __getattr__(self, name):
        """
        Delegate attribute access to base model.

        This method is called when an attribute is not found in the wrapper.
        It delegates to the wrapped base_model, allowing transparent access
        to base model attributes and methods.

        Args:
            name: Name of the attribute to access

        Returns:
            Attribute value from base model
        """
        # Prevent infinite recursion during copying/pickling
        # by not delegating special methods that don't exist
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Prevent recursion if base_model is not yet set (during __init__ or unpickling)
        if "base_model" not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return getattr(self.base_model, name)

    def __getstate__(self):
        """Support for pickling/copying."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Support for unpickling/copying."""
        self.__dict__.update(state)


# =============================================================================
# Enhanced Non-Negative Wrapper with Minimum Threshold and Tracking
# =============================================================================


class EnhancedNonNegativeWrapper:
    """
    Enhanced wrapper for regression models with minimum threshold and tracking.

    This wrapper extends NonNegativeRegressionWrapper with:
    1. Configurable minimum threshold (not just 0)
    2. Tracking of corrections made during prediction
    3. Support for zero prediction detection

    This addresses the critical issue where predictions of exactly 0.0 are
    invalid for stock prices but were not caught by the original wrapper.

    Args:
        base_model: Any sklearn-compatible regression model
        min_value: Minimum allowed prediction value (default: 0.01)

    Attributes:
        base_model: The wrapped regression model
        min_value: Minimum threshold for predictions
        last_correction_count: Number of predictions corrected in last predict() call

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> import numpy as np
        >>>
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([-5, -4, -3, -2])  # Would produce negative predictions
        >>>
        >>> base = Ridge()
        >>> model = EnhancedNonNegativeWrapper(base, min_value=0.01)
        >>> model.fit(X, y)
        >>> preds = model.predict(X)
        >>> assert (preds >= 0.01).all()
        >>> print(f"Corrected {model.last_correction_count} predictions")

    Phase 9.5 Enhancement:
        This class was added to address zero prediction issues identified in
        ml_workflow_guidelines.md (PLTR, BAC, UBER, HD having y_pred = 0.0).
    """

    def __init__(self, base_model, min_value: float = 0.01):
        """
        Initialize enhanced wrapper with configurable minimum threshold.

        Args:
            base_model: sklearn-compatible regression model
            min_value: Minimum allowed prediction value (default: 0.01)
        """
        self.base_model = base_model
        self.min_value = min_value
        self.last_correction_count = 0

    def fit(self, X, y):
        """
        Fit the base model.

        Args:
            X: Feature matrix (pandas DataFrame or numpy array)
            y: Target vector (pandas Series or numpy array)

        Returns:
            self (for method chaining)
        """
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict with minimum threshold enforcement.

        This method:
        1. Gets predictions from base model
        2. Counts predictions below threshold
        3. Clips predictions to be >= min_value
        4. Stores correction count for monitoring

        Args:
            X: Feature matrix (pandas DataFrame or numpy array)

        Returns:
            Predictions with all values >= min_value
        """
        predictions = self.base_model.predict(X)

        # Count predictions below threshold
        below_threshold = predictions < self.min_value
        self.last_correction_count = int(np.sum(below_threshold))

        if self.last_correction_count > 0:
            pct_corrected = 100.0 * self.last_correction_count / len(predictions)
            logger.info(
                f"EnhancedNonNegativeWrapper: Corrected {self.last_correction_count}/"
                f"{len(predictions)} ({pct_corrected:.1f}%) predictions to >= {self.min_value}"
            )

        # Apply minimum threshold
        return np.maximum(predictions, self.min_value)

    def __getattr__(self, name):
        """
        Delegate attribute access to base model.

        Args:
            name: Name of the attribute to access

        Returns:
            Attribute value from base model
        """
        # Prevent infinite recursion during copying/pickling
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Prevent recursion if base_model is not yet set
        if "base_model" not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return getattr(self.base_model, name)

    def __getstate__(self):
        """Support for pickling/copying."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Support for unpickling/copying."""
        self.__dict__.update(state)
