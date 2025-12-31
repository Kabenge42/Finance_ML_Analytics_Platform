"""Analyst and market sentiment feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_analyst_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer analyst quality, consensus, and price target features.

    Features computed (when inputs exist):
    - Analyst consensus: analyst_bullish_pct, analyst_bearish_pct, analyst_conviction (abs diff in pct points)
    - Price target metrics: price_target_spread_pct, price_target_range (alias), consensus_strength (100-spread),
      upside_potential ((median-last)/last * 100), price_target_revision ((median - ytd_ago)/ytd_ago)
    - Coverage quality: analyst_coverage_quality = (# analysts) / log1p(market_cap)
    - Backward-compatibility: target_price_upside_pct alias retained if last_price + price_target present

    Column naming (normalized expected; legacy tolerated where possible):
    - Ratings: strong_buy_ratings, buy_ratings, hold_ratings, sell_ratings, strong_sell_ratings
    - Targets: price_target_median, price_target_high, price_target_low, price_target_ytd_ago, price_target_count
    - Other: last_price, market_cap
    """
    result = df.copy()

    # --- Price target spread and consensus strength ---
    if all(
        c in df.columns for c in ("price_target_high", "price_target_low", "price_target_median")
    ):
        target_range = df["price_target_high"].astype(float) - df["price_target_low"].astype(float)
        spread_pct = _safe_div(target_range, df["price_target_median"].astype(float)) * 100
        result["price_target_spread_pct"] = spread_pct
        # Alias used by tests/plan
        result["price_target_range"] = spread_pct
        result["consensus_strength"] = 100 - spread_pct.clip(upper=100)

    # --- Analyst ratings distribution & consensus ---
    # Support normalized names primarily; allow legacy names with leading underscores if present
    cols_norm = [
        "num_strong_buys_ratings",
        "num_buys_ratings",
        "num_hold_ratings",
        "num_sell_ratings",
        "num_strong_sell_ratings",
    ]
    cols_legacy = [
        "_strong_buy_ratings",
        "_buy_ratings",
        "_hold_ratings",
        "_sell_ratings",
        "_strong_sell_ratings",
    ]
    use_cols = None
    if all(c in df.columns for c in cols_norm):
        use_cols = cols_norm
    elif all(c in df.columns for c in cols_legacy):
        use_cols = cols_legacy
    if use_cols is not None:
        sb, b, h, s, ss = [df[c].astype(float).fillna(0) for c in use_cols]
        total = sb + b + h + s + ss
        bullish = sb + b
        bearish = s + ss
        result["analyst_bullish_pct"] = _safe_div(bullish, total) * 100
        result["analyst_bearish_pct"] = _safe_div(bearish, total) * 100
        # Conviction: absolute difference in percentage points
        if "analyst_bullish_pct" in result.columns and "analyst_bearish_pct" in result.columns:
            result["analyst_conviction"] = (
                result["analyst_bullish_pct"] - result["analyst_bearish_pct"]
            ).abs()

    # --- Upside potential and revisions ---
    if all(c in df.columns for c in ("price_target_median", "last_price")):
        upside = (
            _safe_div(
                df["price_target_median"].astype(float) - df["last_price"].astype(float),
                df["last_price"].astype(float),
            )
            * 100
        )
        result["upside_potential"] = upside
        # Backward-compatible alias
        result["target_price_upside_pct"] = upside
    if all(c in df.columns for c in ("price_target_median", "price_target_ytd_ago")):
        result["price_target_revision"] = _safe_div(
            df["price_target_median"].astype(float) - df["price_target_ytd_ago"].astype(float),
            df["price_target_ytd_ago"].astype(float),
        )

    # --- Coverage quality (#analysts scaled by firm size) ---
    if "price_target_count" in df.columns and "market_cap" in df.columns:
        # log1p(market_cap) in denominator; safe-div guards zero/negatives (log1p of negative is NaN)
        denom = pd.Series(np.log1p(df["market_cap"].astype(float)), index=df.index)
        result["analyst_coverage_quality"] = _safe_div(
            df["price_target_count"].astype(float), denom
        )

    logger.info("Engineered analyst quality & consensus features")
    return result

def engineer_market_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer market sentiment features from short interest and betas.

    Features computed (when inputs exist):
    - one_day_chg: Pass-through of short_int_pct (already percent units)
    - beta_stability: Population variance (ddof=0) across available betas (beta_1y, beta_2y, beta_5y)
    - systematic_risk_trend: beta_1y - beta_5y (risk profile change)

    Args:
        df: Input DataFrame with normalized column names

    Returns:
        DataFrame with market sentiment features added
    """
    result = df.copy()

    # One-day price change (percent already)
    if "one_day_pct" in df.columns:
        result["one_day_chg"] = df["one_day_pct"].astype(float)

    # Beta metrics
    beta_cols = [c for c in ("beta_1y", "beta_2y", "beta_5y") if c in df.columns]
    if beta_cols:
        beta_mat = df[beta_cols].astype(float)
        # Population variance across the provided beta horizons
        result["beta_stability"] = beta_mat.var(axis=1, ddof=0)

    if "beta_1y" in df.columns and "beta_5y" in df.columns:
        result["systematic_risk_trend"] = df["beta_1y"].astype(float) - df["beta_5y"].astype(float)

    # Short interest ratio
    if "short_interest" in df.columns and "volume_shrs" in df.columns:
        result["short_interest_ratio"] = _safe_div(
            df["short_interest"].astype(float), df["volume_shrs"].astype(float)
        )
    elif "short_interest" in df.columns and "shares_outstanding" in df.columns:
        result["short_interest_ratio"] = _safe_div(
            df["short_interest"].astype(float), df["shares_outstanding"].astype(float)
        )

    logger.info("Engineered market sentiment features (short interest, betas)")
    return result
