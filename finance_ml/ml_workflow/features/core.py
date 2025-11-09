"""
finance_ml.ml_workflow.features.core - Core feature engineering functions

This module provides basic feature engineering functions for financial data.
Part of Phase 9.3 refactor: Extracted from features.py for better modularity.

Functions:
- _safe_div: Safe division helper
- engineer_basic_ratios: P/E, P/B, debt ratios, ROE, ROA
- engineer_margin_features: Profit margins (gross, operating, net, EBITDA)
- engineer_volatility_features: Price volatility metrics
- engineer_revenue_cagr: Revenue growth rates
- build_features_and_target: Feature matrix preparation
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safely divide two Series, replacing inf with NaN.

    Args:
        numer: Numerator Series
        denom: Denominator Series

    Returns:
        Result Series with inf values replaced by NaN
    """
    result = numer.astype(float) / denom.astype(float)
    # Replace +/- inf with NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def engineer_basic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add a minimal set of engineered ratio features if source columns exist.

    Ratios computed:
    - ev_to_ebitda = enterprise_value / ebitda (or ev / ebitda)
    - net_debt_to_ebitda = net_debt / ebitda
    - p_e = last_price / eps
    - p_b = last_price / book_value_per_share (or market_cap / total_equity)
    - debt_to_equity = total_debt / total_equity
    - roe = net_income / total_equity
    - roa = net_income / total_assets
    - market_cap_to_revenue = market_cap / revenue

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with ratio features added (preserves original columns)
    """
    out = df.copy()
    cols = set(out.columns)

    # Support both naming conventions
    if {"enterprise_value", "ebitda"}.issubset(cols):
        out["ev_to_ebitda"] = _safe_div(out["enterprise_value"], out["ebitda"])
    elif {"ev", "ebitda"}.issubset(cols):  # Alternative naming
        out["ev_to_ebitda"] = _safe_div(out["ev"], out["ebitda"])

    if {"net_debt", "ebitda"}.issubset(cols):
        out["net_debt_to_ebitda"] = _safe_div(out["net_debt"], out["ebitda"])
    elif {"total_debt", "ebitda"}.issubset(cols):  # Alternative: use total_debt as proxy
        out["net_debt_to_ebitda"] = _safe_div(out["total_debt"], out["ebitda"])

    if {"last_price", "eps"}.issubset(cols):
        out["p_e"] = _safe_div(out["last_price"], out["eps"])

    if {"last_price", "book_value_per_share"}.issubset(cols):
        out["p_b"] = _safe_div(out["last_price"], out["book_value_per_share"])
    elif {"market_cap", "total_equity"}.issubset(cols):
        # Alternative P/B calculation
        out["p_b"] = _safe_div(out["market_cap"], out["total_equity"])

    # Debt to equity
    if {"total_debt", "total_equity"}.issubset(cols):
        out["debt_to_equity"] = _safe_div(out["total_debt"], out["total_equity"])

    # Return on Equity (ROE)
    if {"net_income", "total_equity"}.issubset(cols):
        out["roe"] = _safe_div(out["net_income"], out["total_equity"])

    # Return on Assets (ROA)
    if {"net_income", "total_assets"}.issubset(cols):
        out["roa"] = _safe_div(out["net_income"], out["total_assets"])

    # Market cap to revenue ratio
    if {"market_cap", "total_revenue"}.issubset(cols):
        out["market_cap_to_revenue"] = _safe_div(out["market_cap"], out["total_revenue"])
    elif {"market_cap", "revenue"}.issubset(cols):
        out["market_cap_to_revenue"] = _safe_div(out["market_cap"], out["revenue"])

    return out


def engineer_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add margin features if source columns exist.

    Supports both production columns (_ltm suffix) and simple test columns.

    Margins computed:
    - gross_margin = gross_profit / revenue
    - operating_margin = operating_income / revenue (or _ltm versions)
    - net_margin = net_income / revenue (or _ltm versions)
    - ebitda_margin = ebitda / revenue (or _ltm versions)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with margin features added (preserves original columns)
    """
    out = df.copy()
    cols = set(out.columns)

    # Support simple test column names
    if {"gross_profit", "revenue"}.issubset(cols):
        out["gross_margin"] = _safe_div(out["gross_profit"], out["revenue"])
    if {"operating_income", "revenue"}.issubset(cols):
        out["operating_margin"] = _safe_div(out["operating_income"], out["revenue"])
    if {"net_income", "revenue"}.issubset(cols):
        out["net_margin"] = _safe_div(out["net_income"], out["revenue"])
    if {"ebitda", "revenue"}.issubset(cols):
        out["ebitda_margin"] = _safe_div(out["ebitda"], out["revenue"])

    # Support alternative column names (total_revenue instead of revenue)
    if {"net_income", "total_revenue"}.issubset(cols) and "net_margin" not in out.columns:
        out["net_margin"] = _safe_div(out["net_income"], out["total_revenue"])
    if {"ebitda", "total_revenue"}.issubset(cols) and "ebitda_margin" not in out.columns:
        out["ebitda_margin"] = _safe_div(out["ebitda"], out["total_revenue"])

    # Support production column names with _ltm suffix
    if {"ebitda_ltm", "total_revenues_ltm"}.issubset(cols):
        if "ebitda_margin" not in out.columns:
            out["ebitda_margin"] = _safe_div(out["ebitda_ltm"], out["total_revenues_ltm"])
    if {"operating_income_ltm", "total_revenues_ltm"}.issubset(cols):
        if "operating_margin" not in out.columns:  # Don't overwrite if already created
            out["operating_margin"] = _safe_div(
                out["operating_income_ltm"], out["total_revenues_ltm"]
            )
    if {"net_income_ltm", "total_revenues_ltm"}.issubset(cols):
        if "net_margin" not in out.columns:  # Don't overwrite if already created
            out["net_margin"] = _safe_div(out["net_income_ltm"], out["total_revenues_ltm"])

    return out


def engineer_volatility_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Calculate or aggregate volatility features.

    If 'last_price' column exists, calculates rolling standard deviation.
    If 'high_price' and 'low_price' exist, calculates price range and relative volatility.
    Also aggregates existing volatility columns into summary features.

    Args:
        df: Input DataFrame
        window: Rolling window size for volatility calculation (default: 30)

    Returns:
        DataFrame with volatility features added
    """
    out = df.copy()

    # If last_price exists, calculate rolling volatility
    if "last_price" in out.columns:
        col_name = f"price_volatility_{window}d"
        out[col_name] = out["last_price"].rolling(window=window, min_periods=window).std()

    # Calculate price range if high/low prices available
    if {"high_price", "low_price"}.issubset(out.columns):
        out["price_range"] = out["high_price"] - out["low_price"]

        # Relative volatility: range / last_price
        if "last_price" in out.columns:
            out["relative_volatility"] = _safe_div(out["price_range"], out["last_price"])

    # Find available volatility columns and create average
    vol_cols = [
        c
        for c in out.columns
        if "volatility" in c.lower()
        and c not in ["volatility_avg", f"price_volatility_{window}d", "relative_volatility"]
    ]

    if vol_cols:
        # Calculate average across available volatility columns
        out["volatility_avg"] = out[vol_cols].mean(axis=1)

    return out


def engineer_revenue_cagr(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate revenue CAGR (Compound Annual Growth Rate).

    Supports both production columns (_ltm suffix) and simple test columns.

    Calculates:
    - revenue_cagr_1y: 1-year CAGR
    - revenue_cagr_3y: 3-year CAGR (if 3-year historical data available)
    - revenue_cagr_5y: 5-year CAGR (if 5-year historical data available)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with revenue CAGR features added if source columns exist
    """
    out = df.copy()
    cols = set(out.columns)

    # Support simple test column names
    if {"revenue_current", "revenue_1y_ago"}.issubset(cols):
        # CAGR_1y: (current / 1y_ago)^(1/1) - 1
        out["revenue_cagr_1y"] = (out["revenue_current"] / out["revenue_1y_ago"]) - 1.0
        out["revenue_cagr_1y"] = out["revenue_cagr_1y"].replace([np.inf, -np.inf], np.nan)

    if {"revenue_current", "revenue_3y_ago"}.issubset(cols):
        # CAGR_3y: (current / 3y_ago)^(1/3) - 1
        out["revenue_cagr_3y"] = (out["revenue_current"] / out["revenue_3y_ago"]) ** (
            1.0 / 3.0
        ) - 1.0
        out["revenue_cagr_3y"] = out["revenue_cagr_3y"].replace([np.inf, -np.inf], np.nan)

    # Support alternate column names
    if {"revenue", "revenue_3y_ago"}.issubset(cols) and "revenue_cagr_3y" not in out.columns:
        out["revenue_cagr_3y"] = (out["revenue"] / out["revenue_3y_ago"]) ** (1.0 / 3.0) - 1.0
        out["revenue_cagr_3y"] = out["revenue_cagr_3y"].replace([np.inf, -np.inf], np.nan)

    if {"revenue", "revenue_5y_ago"}.issubset(cols):
        # CAGR_5y: (current / 5y_ago)^(1/5) - 1
        out["revenue_cagr_5y"] = (out["revenue"] / out["revenue_5y_ago"]) ** (1.0 / 5.0) - 1.0
        out["revenue_cagr_5y"] = out["revenue_cagr_5y"].replace([np.inf, -np.inf], np.nan)

    # Support production column names with _ltm suffix
    if {"total_revenues_ltm", "total_revenues_1fy"}.issubset(cols):
        if "revenue_cagr_1y" not in out.columns:  # Don't overwrite if already created
            # Simple growth rate: (current - previous) / previous
            out["revenue_cagr_1y"] = _safe_div(
                out["total_revenues_ltm"] - out["total_revenues_1fy"], out["total_revenues_1fy"]
            )

    return out


def build_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str], List[str]]:
    """Build feature matrix and target variable from DataFrame.

    Returns X (features), y (target), numeric_features, categorical_features.
    Tries to use 'price_target' or 'price_target_median' as y if present;
    otherwise returns y=None.

    Removes identifier columns (ticker, isin, name, description) from features.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of:
        - X: Feature DataFrame
        - y: Target Series (or None if no target column found)
        - numeric_features: List of numeric feature column names
        - categorical_features: List of categorical feature column names
    """
    # PREVENTIVE: Check for duplicate columns and remove them
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Found duplicate columns: {duplicates}")
        logger.warning("Keeping only the first occurrence of each column")
        # Keep only the first occurrence of each column
        df = df.loc[:, ~df.columns.duplicated()]

    y = None
    target_candidates = ["price_target", "price_target_median"]
    y_name = next((t for t in target_candidates if t in df.columns), None)
    if y_name:
        # DEFENSIVE: Ensure we get a Series, not a DataFrame
        y_series = df[y_name]
        if isinstance(y_series, pd.DataFrame):
            # If multiple columns with same name (shouldn't happen after dedup above), take first
            logger.warning(
                f"Column '{y_name}' returned DataFrame instead of Series, taking first column"
            )
            y = pd.to_numeric(y_series.iloc[:, 0], errors="coerce")
        else:
            y = pd.to_numeric(y_series, errors="coerce")

    X = df.copy()
    if y_name:
        X = X.drop(columns=[y_name])

    # Very simple heuristic for feature types
    categorical_features = [c for c in X.columns if X[c].dtype == "object"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    # Drop obvious identifiers from X if present
    drop_cols = [c for c in ["ticker", "isin", "name", "description"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        categorical_features = [c for c in categorical_features if c not in drop_cols]
        numeric_features = [c for c in numeric_features if c not in drop_cols]

    return X, y, numeric_features, categorical_features


__all__ = [
    "_safe_div",
    "engineer_basic_ratios",
    "engineer_margin_features",
    "engineer_volatility_features",
    "engineer_revenue_cagr",
    "build_features_and_target",
]
