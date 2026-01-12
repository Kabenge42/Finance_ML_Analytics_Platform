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


def engineer_price_target_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer price target temporal dynamics features.

    Leverages historical price target data to derive momentum, acceleration,
    and consensus evolution features.

    Features computed (15):
    - pt_momentum_* (1w, 1m, 3m, 6m, 1y, qtd, ytd)
    - pt_acceleration_short/long
    - pt_consensus_convergence and pt_high_low_spread_trend
    - analyst_coverage_change_* (1m, 3m)
    - pt_vs_price_momentum
    - pt_skew_trend

    Args:
        df: Input DataFrame with price target historical columns

    Returns:
        DataFrame with price target dynamics features added
    """

    result = df.copy()

    # === Price Target Momentum ===
    momentum_pairs = [
        ("pt_momentum_1w", "price_target", "price_target_1w_ago"),
        ("pt_momentum_1m", "price_target", "price_target_1m_ago"),
        ("pt_momentum_3m", "price_target", "price_target_3m_ago"),
        ("pt_momentum_6m", "price_target", "price_target_6m_ago"),
        ("pt_momentum_1y", "price_target", "price_target_1y_ago"),
        ("pt_qtd_momentum", "price_target", "price_target_qtd_ago"),
        ("pt_ytd_momentum", "price_target", "price_target_ytd_ago"),
    ]

    for feature_name, current_col, prior_col in momentum_pairs:
        if current_col in df.columns and prior_col in df.columns:
            result[feature_name] = _safe_pct_change(
                df[current_col].astype(float), df[prior_col].astype(float)
            )

    # === Momentum Acceleration ===
    if "pt_momentum_1m" in result.columns and "pt_momentum_3m" in result.columns:
        result["pt_acceleration_short"] = result["pt_momentum_1m"] - result["pt_momentum_3m"]

    if "pt_momentum_3m" in result.columns and "pt_momentum_1y" in result.columns:
        result["pt_acceleration_long"] = result["pt_momentum_3m"] - result["pt_momentum_1y"]

    # === Consensus Range Evolution ===
    spread_cols_current = [
        "price_target_high",
        "price_target_low",
        "price_target_median",
    ]
    spread_cols_3m = [
        "price_target_high_3m_ago",
        "price_target_low_3m_ago",
        "price_target_median_3m_ago",
    ]

    if all(c in df.columns for c in spread_cols_current + spread_cols_3m):
        current_spread = _safe_div(
            df["price_target_high"].astype(float) - df["price_target_low"].astype(float),
            df["price_target_median"].astype(float),
        )
        spread_3m = _safe_div(
            df["price_target_high_3m_ago"].astype(float)
            - df["price_target_low_3m_ago"].astype(float),
            df["price_target_median_3m_ago"].astype(float),
        )
        result["pt_consensus_convergence"] = spread_3m - current_spread
        result["pt_high_low_spread_trend"] = current_spread - spread_3m

    # === Analyst Coverage Trajectory ===
    count_col = "price_target_count" if "price_target_count" in df.columns else "price_target_num"
    if count_col in df.columns:
        if "price_target_count_1m_ago" in df.columns:
            result["analyst_coverage_change_1m"] = df[count_col].astype(float) - df[
                "price_target_count_1m_ago"
            ].astype(float)
        if "price_target_count_3m_ago" in df.columns:
            result["analyst_coverage_change_3m"] = df[count_col].astype(float) - df[
                "price_target_count_3m_ago"
            ].astype(float)

    # === Target vs Price Momentum Divergence ===
    if all(
        c in df.columns
        for c in ["price_target", "last_price", "price_target_3m_ago", "price_3m_ago"]
    ):
        current_ratio = _safe_div(df["price_target"].astype(float), df["last_price"].astype(float))
        prior_ratio = _safe_div(
            df["price_target_3m_ago"].astype(float), df["price_3m_ago"].astype(float)
        )
        result["pt_vs_price_momentum"] = _safe_pct_change(current_ratio, prior_ratio)

    # === Skewness Trend (Mean vs Median) ===
    if all(
        c in df.columns
        for c in [
            "price_target",
            "price_target_median",
            "price_target_3m_ago",
            "price_target_median_3m_ago",
        ]
    ):
        current_skew = df["price_target"].astype(float) - df["price_target_median"].astype(float)
        prior_skew = df["price_target_3m_ago"].astype(float) - df[
            "price_target_median_3m_ago"
        ].astype(float)
        result["pt_skew_trend"] = current_skew - prior_skew

    logger.info("Engineered price target dynamics features (15 features)")
    return result


def _safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Calculate percentage change with safe division."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (current - previous) / previous.abs().replace(0, pd.NA)
    return result.astype("Float64")
