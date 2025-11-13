"""
Phase 10.4: Enhanced Bias Correction with Isotonic Regression

Provides advanced bias correction utilities to reduce systematic over/under-prediction
observed in evaluation reports. Supports multiple correction methods:
- Isotonic regression calibration (monotonicity-preserving)
- Market cap bucket-specific corrections
- Temporal bias adjustment for market trends
- Validation plots and metrics export

Target: Reduce systematic over-prediction bias by 50% across all sectors

Key Features:
- isotonic_calibration(): Fit and apply isotonic regression calibration
- market_cap_bias_correction(): Correct bias by market cap bucket (small/mid/large)
- temporal_bias_adjustment(): Adjust for temporal trends in predictions
- plot_bias_correction_validation(): Generate validation plots by sector
- export_bias_correction_metrics(): Export bias reduction metrics to CSV
- Enhanced calibrate_predictions_by_sector(): Supports isotonic method

Integration:
- Works with predictions from regression.models
- Compatible with quantile regression outputs
- Exports metrics for analytics and reporting

Example:
    >>> from finance_ml.ml_workflow.regression.calibration import (
    ...     calibrate_predictions_by_sector,
    ...     market_cap_bias_correction,
    ...     temporal_bias_adjustment
    ... )
    >>>
    >>> # Split data for calibration
    >>> cal_df = train_df.copy()
    >>> test_df = test_df.copy()
    >>>
    >>> # Apply isotonic calibration by sector
    >>> calibrated_df = calibrate_predictions_by_sector(
    ...     preds_df=test_df,
    ...     cal_df=cal_df,
    ...     method="isotonic",
    ...     sector_col="sector",
    ...     pred_col="y_pred",
    ...     true_col="y_true",
    ...     output_col="y_pred_step1"
    ... )
    >>>
    >>> # Apply market cap correction
    >>> calibrated_df = market_cap_bias_correction(
    ...     preds_df=calibrated_df,
    ...     cal_df=cal_df,
    ...     market_cap_col="market_cap",
    ...     pred_col="y_pred_step1",
    ...     true_col="y_true",
    ...     output_col="y_pred_step2"
    ... )
    >>>
    >>> # Apply temporal adjustment
    >>> calibrated_df = temporal_bias_adjustment(
    ...     preds_df=calibrated_df,
    ...     cal_df=cal_df,
    ...     date_col="date",
    ...     pred_col="y_pred_step2",
    ...     true_col="y_true",
    ...     output_col="y_pred_calibrated"
    ... )

Reference:
- Task 10.4 from finance_ml_improvement_plan.md
- Isotonic Regression: sklearn.isotonic.IsotonicRegression
"""

from __future__ import annotations

from typing import Dict, Optional, List, Any, Union
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

DEFAULT_SECTOR_BIAS: Dict[str, float] = {
    # Negative values indicate over-prediction (subtract to correct)
    "Financials": -795.0,
    "Industrials": +544.0,
    "Communication Services": -755.0,
}


def isotonic_calibration(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    fit: bool = True,
    calibrator: Optional[IsotonicRegression] = None,
) -> Union[IsotonicRegression, np.ndarray]:
    """
    Fit or apply isotonic regression calibration.

    Isotonic regression is a non-parametric monotonic calibration method that
    learns the relationship between predicted and true values while preserving
    monotonicity. It reduces bias while maintaining the ordering of predictions.

    Args:
        y_true: True target values (required if fit=True)
        y_pred: Predicted values
        fit: If True, fit and return calibrator. If False, apply calibrator to y_pred
        calibrator: Pre-fitted IsotonicRegression object (required if fit=False)

    Returns:
        If fit=True: Fitted IsotonicRegression object
        If fit=False: Calibrated predictions (numpy array)

    Example:
        >>> # Fit calibrator
        >>> calibrator = isotonic_calibration(
        ...     y_true=y_train, y_pred=pred_train, fit=True
        ... )
        >>>
        >>> # Apply to test set
        >>> calibrated_preds = isotonic_calibration(
        ...     y_true=None, y_pred=pred_test, fit=False, calibrator=calibrator
        ... )
    """
    if fit:
        if y_true is None:
            raise ValueError("y_true is required when fit=True")

        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(y_pred, y_true)

        logger.debug(f"Fitted isotonic regression on {len(y_pred)} samples")
        return iso_reg

    else:
        if calibrator is None:
            raise ValueError("calibrator is required when fit=False")

        # Apply calibration
        calibrated = calibrator.predict(y_pred)
        logger.debug(f"Applied isotonic calibration to {len(y_pred)} samples")
        return calibrated


def calibrate_predictions_by_sector(
    preds_df: pd.DataFrame,
    sector_bias: Optional[Dict[str, float]] = None,
    sector_col: str = "sector",
    pred_col: str = "y_pred",
    output_col: str = "y_pred_calibrated",
    cal_df: Optional[pd.DataFrame] = None,
    method: str = "additive",
    true_col: str = "y_true",
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Apply sector-specific calibration to predictions.

    Supports two methods:
    - "additive": Simple additive bias correction (original behavior)
    - "isotonic": Isotonic regression calibration per sector

    Args:
        preds_df: DataFrame containing predictions to calibrate
        sector_bias: Optional mapping of sector -> bias (only for additive method)
        sector_col: Column name for sector identifier
        pred_col: Column with base predictions to calibrate
        output_col: Name for calibrated prediction column
        cal_df: Calibration DataFrame with true values (required for isotonic method)
        method: Calibration method ("additive" or "isotonic")
        true_col: Column name for true values (required for isotonic method)
        min_samples: Minimum samples per sector for isotonic (default: 5)

    Returns:
        DataFrame with an added column `output_col` containing calibrated predictions.
        For sectors with insufficient samples, falls back to uncalibrated predictions.

    Example:
        >>> # Additive calibration (original)
        >>> calibrated_df = calibrate_predictions_by_sector(
        ...     preds_df=test_df,
        ...     method="additive"
        ... )
        >>>
        >>> # Isotonic calibration
        >>> calibrated_df = calibrate_predictions_by_sector(
        ...     preds_df=test_df,
        ...     cal_df=train_df,
        ...     method="isotonic",
        ...     sector_col="sector",
        ...     pred_col="y_pred",
        ...     true_col="y_true"
        ... )
    """
    if pred_col not in preds_df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}' in preds_df")

    out = preds_df.copy()

    # Handle missing sector column
    if sector_col not in out.columns:
        logger.warning(f"Sector column '{sector_col}' not found, copying predictions unchanged")
        out[output_col] = out[pred_col]
        return out

    if method == "additive":
        # Original additive calibration
        if sector_bias is None:
            sector_bias = DEFAULT_SECTOR_BIAS

        out[output_col] = out[pred_col]

        # Apply per-sector bias
        for sector, bias in sector_bias.items():
            mask = out[sector_col] == sector
            if mask.any():
                out.loc[mask, output_col] = out.loc[mask, pred_col] + float(bias)

    elif method == "isotonic":
        # Isotonic regression calibration per sector
        if cal_df is None:
            raise ValueError("cal_df is required for isotonic calibration")
        if true_col not in cal_df.columns:
            raise ValueError(f"Missing true value column '{true_col}' in cal_df")
        if pred_col not in cal_df.columns:
            raise ValueError(f"Missing prediction column '{pred_col}' in cal_df")
        if sector_col not in cal_df.columns:
            raise ValueError(f"Missing sector column '{sector_col}' in cal_df")

        # Initialize output with uncalibrated predictions
        out[output_col] = out[pred_col]

        # Fit and apply isotonic calibration per sector
        unique_sectors = out[sector_col].unique()

        for sector in unique_sectors:
            # Get calibration data for this sector
            cal_sector_mask = cal_df[sector_col] == sector
            cal_sector_df = cal_df[cal_sector_mask]

            # Check minimum samples
            if len(cal_sector_df) < min_samples:
                logger.warning(
                    f"Sector {sector}: insufficient samples ({len(cal_sector_df)} < {min_samples}), "
                    f"skipping calibration"
                )
                continue

            # Fit isotonic regression
            try:
                calibrator = isotonic_calibration(
                    y_true=cal_sector_df[true_col].values,
                    y_pred=cal_sector_df[pred_col].values,
                    fit=True,
                )

                # Apply to test samples
                test_sector_mask = out[sector_col] == sector
                if test_sector_mask.sum() > 0:
                    calibrated_preds = isotonic_calibration(
                        y_true=None,
                        y_pred=out.loc[test_sector_mask, pred_col].values,
                        fit=False,
                        calibrator=calibrator,
                    )
                    out.loc[test_sector_mask, output_col] = calibrated_preds

                    logger.debug(
                        f"Sector {sector}: calibrated {test_sector_mask.sum()} predictions"
                    )
            except Exception as e:
                logger.warning(f"Sector {sector}: calibration failed: {e}")
                continue

    else:
        raise ValueError(f"Unknown calibration method '{method}'. Choose 'additive' or 'isotonic'")

    return out


def market_cap_bias_correction(
    preds_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    market_cap_col: str = "market_cap",
    pred_col: str = "y_pred",
    true_col: str = "y_true",
    output_col: str = "y_pred_cap_corrected",
) -> pd.DataFrame:
    """
    Apply market cap bucket-specific bias correction.

    Different market cap categories (small/mid/large) often exhibit different
    systematic biases. This function computes and applies separate corrections
    for each market cap bucket.

    Args:
        preds_df: DataFrame with predictions to correct
        cal_df: Calibration DataFrame to compute bucket biases
        market_cap_col: Column name for market cap category
        pred_col: Column with predictions (must exist in both preds_df and cal_df)
        true_col: Column with true values (must exist in cal_df)
        output_col: Name for corrected prediction column

    Returns:
        DataFrame with corrected predictions in `output_col`

    Example:
        >>> corrected_df = market_cap_bias_correction(
        ...     preds_df=test_df,
        ...     cal_df=train_df,
        ...     market_cap_col="market_cap",
        ...     pred_col="y_pred",
        ...     true_col="y_true"
        ... )

    Notes:
        - pred_col should reference the base prediction column that exists in cal_df
        - For chained corrections, use the original prediction column from cal_df
    """
    if market_cap_col not in cal_df.columns:
        logger.warning(
            f"Market cap column '{market_cap_col}' not found in cal_df, copying unchanged"
        )
        out = preds_df.copy()
        out[output_col] = out[pred_col]
        return out

    if market_cap_col not in preds_df.columns:
        logger.warning(
            f"Market cap column '{market_cap_col}' not found in preds_df, copying unchanged"
        )
        out = preds_df.copy()
        out[output_col] = out[pred_col]
        return out

    # Validate required columns in cal_df
    # Note: For bias computation, we need the base prediction column (y_pred) from cal_df
    # not the intermediate corrected columns (y_pred_step1, etc.)
    base_pred_col = "y_pred"  # Always use base prediction for bias computation

    if base_pred_col not in cal_df.columns or true_col not in cal_df.columns:
        logger.warning(
            f"Required columns '{base_pred_col}' or '{true_col}' not found in cal_df, copying unchanged"
        )
        out = preds_df.copy()
        out[output_col] = out[pred_col]
        return out

    # Compute bias by market cap bucket from calibration data using base predictions
    cap_bias = {}
    for cap in cal_df[market_cap_col].unique():
        cap_mask = cal_df[market_cap_col] == cap
        cap_df_subset = cal_df[cap_mask]

        if len(cap_df_subset) >= 5:  # Minimum samples
            bias = (cap_df_subset[base_pred_col] - cap_df_subset[true_col]).mean()
            cap_bias[cap] = bias
            logger.debug(f"Market cap {cap}: computed bias = {bias:.2f}")

    # Apply corrections to preds_df using the specified pred_col
    out = preds_df.copy()
    out[output_col] = out[pred_col]

    for cap, bias in cap_bias.items():
        cap_mask = out[market_cap_col] == cap
        if cap_mask.any():
            out.loc[cap_mask, output_col] = out.loc[cap_mask, pred_col] - bias
            logger.debug(f"Market cap {cap}: corrected {cap_mask.sum()} predictions")

    return out


def temporal_bias_adjustment(
    preds_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    date_col: str = "date",
    pred_col: str = "y_pred",
    true_col: str = "y_true",
    output_col: str = "y_pred_temporal_adjusted",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Apply temporal bias adjustment to account for market trends.

    Market conditions change over time, leading to temporal drift in prediction bias.
    This function divides the calibration period into bins and computes time-varying
    bias corrections.

    Args:
        preds_df: DataFrame with predictions to adjust
        cal_df: Calibration DataFrame to compute temporal biases
        date_col: Column name for date/timestamp
        pred_col: Column with predictions
        true_col: Column with true values
        output_col: Name for adjusted prediction column
        n_bins: Number of time bins (default: 10)

    Returns:
        DataFrame with temporally adjusted predictions in `output_col`

    Example:
        >>> adjusted_df = temporal_bias_adjustment(
        ...     preds_df=test_df,
        ...     cal_df=train_df,
        ...     date_col="date",
        ...     pred_col="y_pred",
        ...     true_col="y_true"
        ... )
    """
    if date_col not in cal_df.columns:
        logger.warning(f"Date column '{date_col}' not found in cal_df, copying unchanged")
        out = preds_df.copy()
        out[output_col] = out[pred_col]
        return out

    if date_col not in preds_df.columns:
        logger.warning(f"Date column '{date_col}' not found in preds_df, copying unchanged")
        out = preds_df.copy()
        out[output_col] = out[pred_col]
        return out

    # Validate required columns in cal_df
    # Note: For bias computation, we need the base prediction column (y_pred) from cal_df
    # not the intermediate corrected columns (y_pred_step1, y_pred_step2, etc.)
    base_pred_col = "y_pred"  # Always use base prediction for bias computation

    if base_pred_col not in cal_df.columns or true_col not in cal_df.columns:
        logger.warning(
            f"Required columns '{base_pred_col}' or '{true_col}' not found in cal_df, copying unchanged"
        )
        out = preds_df.copy()
        out[output_col] = out[pred_col]
        return out

    # Convert dates to numeric for binning
    cal_df_sorted = cal_df.sort_values(date_col).copy()
    cal_df_sorted["_time_idx"] = np.arange(len(cal_df_sorted))

    # Create time bins
    cal_df_sorted["_time_bin"] = pd.qcut(
        cal_df_sorted["_time_idx"], q=n_bins, labels=False, duplicates="drop"
    )

    # Compute bias per time bin using base predictions from cal_df
    bin_bias = {}
    for bin_idx in cal_df_sorted["_time_bin"].unique():
        bin_mask = cal_df_sorted["_time_bin"] == bin_idx
        bin_df = cal_df_sorted[bin_mask]

        if len(bin_df) >= 3:  # Minimum samples per bin
            bias = (bin_df[base_pred_col] - bin_df[true_col]).mean()

            # Get date range for this bin
            bin_start = bin_df[date_col].min()
            bin_end = bin_df[date_col].max()
            bin_bias[bin_idx] = {"bias": bias, "start": bin_start, "end": bin_end}
            logger.debug(f"Time bin {bin_idx}: bias = {bias:.2f}")

    # Apply temporal adjustment to test data
    out = preds_df.copy()
    out[output_col] = out[pred_col]

    # If no bins were created, compute global bias as fallback
    if not bin_bias:
        global_bias = (cal_df[pred_col] - cal_df[true_col]).mean()
        out[output_col] = out[pred_col] - global_bias
        logger.debug(f"Using global bias correction: {global_bias:.2f}")
        return out

    # Assign test samples to nearest bin based on date
    test_dates = out[date_col].values
    adjusted_mask = np.zeros(len(out), dtype=bool)

    for bin_idx, bin_info in bin_bias.items():
        # Find samples within this bin's date range
        date_mask = (out[date_col] >= bin_info["start"]) & (out[date_col] <= bin_info["end"])

        if date_mask.any():
            out.loc[date_mask, output_col] = out.loc[date_mask, pred_col] - bin_info["bias"]
            adjusted_mask |= date_mask.values
            logger.debug(f"Time bin {bin_idx}: adjusted {date_mask.sum()} predictions")

    # For samples outside all bins, use nearest bin or global average
    if not adjusted_mask.all():
        unadjusted_mask = ~adjusted_mask
        global_bias = np.mean([info["bias"] for info in bin_bias.values()])
        out.loc[unadjusted_mask, output_col] = out.loc[unadjusted_mask, pred_col] - global_bias
        logger.debug(f"Applied global bias to {unadjusted_mask.sum()} samples outside bin ranges")

    return out


def plot_bias_correction_validation(
    df: pd.DataFrame,
    sector_col: str = "sector",
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    calibrated_col: str = "y_pred_calibrated",
    output_dir: Path = None,
) -> List[str]:
    """
    Generate validation plots comparing original and calibrated predictions by sector.

    Creates scatter plots showing:
    - Original predictions vs true values
    - Calibrated predictions vs true values
    - Bias reduction visualization

    Args:
        df: DataFrame with predictions
        sector_col: Column name for sector
        true_col: Column with true values
        pred_col: Column with original predictions
        calibrated_col: Column with calibrated predictions
        output_dir: Directory to save plots (default: current directory)

    Returns:
        List of paths to generated plot files

    Example:
        >>> plot_paths = plot_bias_correction_validation(
        ...     df=calibrated_df,
        ...     sector_col="sector",
        ...     output_dir=Path("outputs/plots")
        ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
        return []

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []

    # Plot by sector
    for sector in df[sector_col].unique():
        sector_mask = df[sector_col] == sector
        sector_df = df[sector_mask]

        if len(sector_df) < 5:
            continue

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original predictions
        axes[0].scatter(sector_df[true_col], sector_df[pred_col], alpha=0.5, label="Original")
        axes[0].plot(
            [sector_df[true_col].min(), sector_df[true_col].max()],
            [sector_df[true_col].min(), sector_df[true_col].max()],
            "r--",
            label="Perfect",
        )
        axes[0].set_xlabel("True Value")
        axes[0].set_ylabel("Predicted Value")
        axes[0].set_title(f"{sector} - Original Predictions")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Calibrated predictions
        axes[1].scatter(
            sector_df[true_col],
            sector_df[calibrated_col],
            alpha=0.5,
            color="green",
            label="Calibrated",
        )
        axes[1].plot(
            [sector_df[true_col].min(), sector_df[true_col].max()],
            [sector_df[true_col].min(), sector_df[true_col].max()],
            "r--",
            label="Perfect",
        )
        axes[1].set_xlabel("True Value")
        axes[1].set_ylabel("Calibrated Value")
        axes[1].set_title(f"{sector} - Calibrated Predictions")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Compute and display bias
        original_bias = (sector_df[pred_col] - sector_df[true_col]).mean()
        calibrated_bias = (sector_df[calibrated_col] - sector_df[true_col]).mean()
        reduction_pct = (
            100 * (abs(original_bias) - abs(calibrated_bias)) / abs(original_bias)
            if original_bias != 0
            else 0
        )

        fig.suptitle(
            f"Bias: Original={original_bias:.2f}, Calibrated={calibrated_bias:.2f}, "
            f"Reduction={reduction_pct:.1f}%",
            fontsize=10,
        )

        plt.tight_layout()

        # Save plot
        safe_sector = sector.replace(" ", "_").replace("/", "_")
        plot_path = output_dir / f"bias_correction_{safe_sector}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        plot_paths.append(str(plot_path))
        logger.info(f"Saved plot: {plot_path}")

    return plot_paths


def export_bias_correction_metrics(
    df: pd.DataFrame,
    sector_col: str = "sector",
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    calibrated_col: str = "y_pred_calibrated",
    output_dir: Path = None,
    filename: str = "bias_correction_metrics.csv",
) -> None:
    """
    Export bias correction metrics to CSV file.

    Exports per-sector metrics showing:
    - Original bias (mean prediction - true)
    - Calibrated bias
    - Bias reduction percentage
    - Sample count

    Args:
        df: DataFrame with predictions
        sector_col: Column name for sector
        true_col: Column with true values
        pred_col: Column with original predictions
        calibrated_col: Column with calibrated predictions
        output_dir: Directory to save CSV (default: current directory)
        filename: Output filename (default: "bias_correction_metrics.csv")

    Example:
        >>> export_bias_correction_metrics(
        ...     df=calibrated_df,
        ...     sector_col="sector",
        ...     output_dir=Path("outputs")
        ... )
    """
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics by sector
    metrics_rows = []

    for sector in df[sector_col].unique():
        sector_mask = df[sector_col] == sector
        sector_df = df[sector_mask]

        if len(sector_df) < 1:
            continue

        original_bias = (sector_df[pred_col] - sector_df[true_col]).mean()
        calibrated_bias = (sector_df[calibrated_col] - sector_df[true_col]).mean()

        # Calculate reduction percentage
        if abs(original_bias) > 0:
            reduction_pct = 100 * (abs(original_bias) - abs(calibrated_bias)) / abs(original_bias)
        else:
            reduction_pct = 0.0

        metrics_rows.append(
            {
                "sector": sector,
                "n_samples": len(sector_df),
                "original_bias": original_bias,
                "calibrated_bias": calibrated_bias,
                "bias_reduction_pct": reduction_pct,
            }
        )

    # Create DataFrame and save
    metrics_df = pd.DataFrame(metrics_rows)
    output_path = output_dir / filename
    metrics_df.to_csv(output_path, index=False)

    logger.info(f"Exported bias correction metrics to {output_path}")
    logger.info(f"Overall bias reduction: {metrics_df['bias_reduction_pct'].mean():.1f}%")
