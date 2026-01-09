"""
Feature engineering module for missing Phase 9.3 coverage features.

This module provides functions to engineer features that are defined in
PHASE93_FEATURE_CATEGORIES but not yet implemented in the existing modules.

Missing features identified:
- Dividend Reliability: dividend_coverage_ratio, dividend_growth_3y, dividend_growth_5y
- Composite Scores: value_score
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)


def engineer_missing_dividend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer missing dividend reliability features.

    Adds:
    - dividend_coverage_ratio: Earnings coverage of dividend payments (EPS / DPS)
    - dividend_growth_3y: 3-year dividend growth rate
    - dividend_growth_5y: 5-year dividend growth rate

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing dividend features added
    """
    result = df.copy()

    # Dividend Coverage Ratio: EPS / DPS (how many times earnings cover dividends)
    # Use adjusted EPS and DPS if available
    eps_col = None
    dps_col = None

    # Find EPS column
    for col in ["eps_adj_ltm", "eps_ltm", "eps_basic_ltm", "eps_diluted_ltm"]:
        if col in df.columns:
            eps_col = col
            break

    # Find DPS column
    for col in ["dps_ltm", "dividend_per_share_ltm", "dps_common_ltm"]:
        if col in df.columns:
            dps_col = col
            break

    if eps_col and dps_col:
        result["dividend_coverage_ratio"] = _safe_div(df[eps_col], df[dps_col].replace(0, np.nan))

    # Dividend Growth 3Y: CAGR of dividends over 3 years
    # Look for historical DPS columns
    dps_current = None
    dps_3y_ago = None
    dps_5y_ago = None

    # Current DPS
    for col in ["dps_ltm", "dps_fy", "dividend_per_share_ltm"]:
        if col in df.columns:
            dps_current = df[col]
            break

    # 3-year ago DPS
    for col in ["dps_3fy", "dividend_per_share_3fy"]:
        if col in df.columns:
            dps_3y_ago = df[col]
            break

    # 5-year ago DPS
    for col in ["dps_5fy", "dividend_per_share_5fy"]:
        if col in df.columns:
            dps_5y_ago = df[col]
            break

    # Calculate 3-year dividend growth (CAGR)
    if dps_current is not None and dps_3y_ago is not None:
        # CAGR = (End/Start)^(1/n) - 1
        ratio_3y = _safe_div(dps_current, dps_3y_ago.replace(0, np.nan))
        # Only calculate for positive ratios
        valid_ratio = ratio_3y.where(ratio_3y > 0, np.nan)
        result["dividend_growth_3y"] = (np.power(valid_ratio, 1 / 3) - 1) * 100

    # Calculate 5-year dividend growth (CAGR)
    if dps_current is not None and dps_5y_ago is not None:
        ratio_5y = _safe_div(dps_current, dps_5y_ago.replace(0, np.nan))
        valid_ratio = ratio_5y.where(ratio_5y > 0, np.nan)
        result["dividend_growth_5y"] = (np.power(valid_ratio, 1 / 5) - 1) * 100

    logger.info("Engineered missing dividend features")
    return result


def engineer_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer composite value investing score.

    The value_score combines multiple valuation metrics into a single
    composite score (0-100) where higher values indicate better value.

    Components considered:
    - P/E ratio (lower is better)
    - P/B ratio (lower is better)
    - EV/EBITDA (lower is better)
    - Dividend yield (higher is better)
    - FCF yield (higher is better)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with value_score added
    """
    result = df.copy()

    value_components = []

    # P/E Score (inverse - lower P/E = higher score)
    pe_col = None
    for col in ["p_e_ltm", "pe_ratio", "p_e_ratio_ltm"]:
        if col in df.columns:
            pe_col = col
            break

    if pe_col:
        # Normalize P/E: clip to reasonable range (0-50), invert
        pe_normalized = df[pe_col].clip(0, 50)
        pe_score = (50 - pe_normalized) / 50 * 100  # 0-100 scale
        value_components.append(pe_score.fillna(50))

    # P/B Score (inverse - lower P/B = higher score)
    pb_col = None
    for col in ["p_b_ltm", "pb_ratio", "p_b_ratio_ltm", "price_to_book"]:
        if col in df.columns:
            pb_col = col
            break

    if pb_col:
        pb_normalized = df[pb_col].clip(0, 10)
        pb_score = (10 - pb_normalized) / 10 * 100
        value_components.append(pb_score.fillna(50))

    # EV/EBITDA Score (inverse - lower = higher score)
    ev_ebitda_col = None
    for col in ["ev_ebitda_ltm", "ev_to_ebitda", "ev_ebitda"]:
        if col in df.columns:
            ev_ebitda_col = col
            break

    if ev_ebitda_col:
        ev_ebitda_normalized = df[ev_ebitda_col].clip(0, 30)
        ev_ebitda_score = (30 - ev_ebitda_normalized) / 30 * 100
        value_components.append(ev_ebitda_score.fillna(50))

    # Dividend Yield Score (direct - higher yield = higher score)
    div_yield_col = None
    for col in ["div_yield_ltm", "dividend_yield", "dividend_yield_ltm"]:
        if col in df.columns:
            div_yield_col = col
            break

    if div_yield_col:
        # Normalize yield: 0-10% range
        div_yield_normalized = df[div_yield_col].clip(0, 10)
        div_yield_score = div_yield_normalized / 10 * 100
        value_components.append(div_yield_score.fillna(50))

    # FCF Yield Score (direct - higher = higher score)
    fcf_yield_col = None
    for col in ["fcf_yield_ltm", "fcf_yield", "free_cash_flow_yield"]:
        if col in df.columns:
            fcf_yield_col = col
            break

    if fcf_yield_col:
        fcf_yield_normalized = df[fcf_yield_col].clip(-10, 20)
        fcf_yield_score = (fcf_yield_normalized + 10) / 30 * 100
        value_components.append(fcf_yield_score.fillna(50))

    # Calculate composite value score
    if value_components:
        result["value_score"] = pd.concat(value_components, axis=1).mean(axis=1)
    else:
        result["value_score"] = np.nan

    logger.info("Engineered value_score composite")
    return result


def engineer_all_missing_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Apply all missing feature engineering functions.

    This is the main entry point for filling Phase 9.3 coverage gaps.

    Args:
        df: Input DataFrame
        reference_date: Reference date for temporal features (unused currently)

    Returns:
        DataFrame with all missing features added
    """
    result = df.copy()

    result = engineer_missing_dividend_features(result)
    result = engineer_value_score(result)

    logger.info("Engineered all missing Phase 9.3 coverage features")
    return result
