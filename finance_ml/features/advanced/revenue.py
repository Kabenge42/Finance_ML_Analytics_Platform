"""Revenue forecast feature engineering."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_revenue_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer revenue forecast features.

    Features computed:
    - revenue_estimate_momentum: (Revenues Est NTM - Revenues LTM) / Revenues LTM
    - revenue_surprise_volatility: Standard deviation of revenue surprises (if available)
    """
    result = df.copy()

    # Revenue Estimate Momentum
    if "revenues_est_avg_ntm" in df.columns and "total_revenues_ltm" in df.columns:
        est_ntm = pd.to_numeric(df["revenues_est_avg_ntm"], errors="coerce")
        rev_ltm = pd.to_numeric(df["total_revenues_ltm"], errors="coerce")

        result["revenue_estimate_momentum"] = _safe_div(est_ntm - rev_ltm, rev_ltm)

    # Revenue Surprise Volatility
    # Requires historical surprise data. Currently we only have current surprise calculated in earnings.py.
    # If we had multiple periods (e.g. surprise_1q, surprise_2q...), we could calc std dev.
    # For now, we will assign NaN or skip.
    if "revenue_surprise_volatility" not in result.columns:
        result["revenue_surprise_volatility"] = float("nan")

    # Estimate spread (avg vs median)
    if "revenues_est_avg_fy1e" in df.columns and "revenues_est_med_fy1e" in df.columns:
        result["revenue_estimate_skew"] = _safe_div(
            df["revenues_est_avg_fy1e"] - df["revenues_est_med_fy1e"],
            df["revenues_est_med_fy1e"],
        )

    # EBITDA estimate vs actual margin improvement
    if all(
        c in df.columns
        for c in [
            "ebitda_est_avg_fy1e",
            "revenues_est_avg_fy1e",
            "ebitda_ltm",
            "total_revenues_ltm",
        ]
    ):
        current_margin = _safe_div(df["ebitda_ltm"], df["total_revenues_ltm"])
        forward_margin = _safe_div(
            df["ebitda_est_avg_fy1e"], df["revenues_est_avg_fy1e"]
        )
        result["ebitda_margin_improvement_expected"] = forward_margin - current_margin

    # Forward EBIT margin
    if "ebit_est_med_fy1e" in df.columns and "revenues_est_avg_fy1e" in df.columns:
        result["forward_ebit_margin"] = (
            _safe_div(df["ebit_est_med_fy1e"], df["revenues_est_avg_fy1e"]) * 100
        )

    # Analyst coverage depth (for estimate reliability)
    if "eps_norm_est_num_fy1e" in df.columns:
        result["analyst_estimate_coverage"] = df["eps_norm_est_num_fy1e"]
        result["high_coverage_flag"] = (df["eps_norm_est_num_fy1e"] >= 10).astype(int)

    # NTM vs FY1E estimate alignment
    if "revenues_est_avg_ntm" in df.columns and "revenues_est_avg_fy1e" in df.columns:
        result["revenue_estimate_alignment"] = _safe_div(
            df["revenues_est_avg_ntm"], df["revenues_est_avg_fy1e"]
        )

    logger.info("Engineered revenue forecast features")
    return result
