"""
Sector-specific calibration utilities (Priority 3: Bias Correction).

Provides a simple post-prediction bias correction layer by sector to mitigate
systematic over/under-prediction observed in evaluation reports.

The calibration is additive (in price units). Use with care and validate on a
holdout set. The default bias dictionary is sourced from Model Optimization
Recommendations and can be overridden by passing a custom mapping.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

DEFAULT_SECTOR_BIAS: Dict[str, float] = {
    # Negative values indicate over-prediction (subtract to correct)
    "Financials": -795.0,
    "Industrials": +544.0,
    "Communication Services": -755.0,
}


def calibrate_predictions_by_sector(
    preds_df: pd.DataFrame,
    sector_bias: Optional[Dict[str, float]] = None,
    sector_col: str = "sector",
    pred_col: str = "y_pred",
    output_col: str = "y_pred_calibrated",
) -> pd.DataFrame:
    """Apply sector-specific additive calibration to predictions.

    Args:
        preds_df: DataFrame containing at least sector and prediction columns.
        sector_bias: Optional mapping of sector -> bias to add to y_pred.
        sector_col: Column name for sector identifier.
        pred_col: Column with base predictions to calibrate.
        output_col: Name for calibrated prediction column.

    Returns:
        DataFrame with an added column `output_col` containing calibrated preds.
        Rows for sectors not present in `sector_bias` are copied unchanged.
    """
    if sector_bias is None:
        sector_bias = DEFAULT_SECTOR_BIAS

    if pred_col not in preds_df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}' in preds_df")
    if sector_col not in preds_df.columns:
        # No sector information; return copy with output_col equal to pred_col
        out = preds_df.copy()
        out[output_col] = out[pred_col]
        return out

    out = preds_df.copy()
    out[output_col] = out[pred_col]

    # Apply per-sector bias
    for sector, bias in sector_bias.items():
        mask = out[sector_col] == sector
        if mask.any():
            out.loc[mask, output_col] = out.loc[mask, pred_col] + float(bias)

    return out
