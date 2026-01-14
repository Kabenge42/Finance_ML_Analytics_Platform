"""Analyst and market sentiment feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import _safe_div

# EPS Estimate Revision columns for analyst sentiment momentum
EPS_REVISION_COLS = {
    "eps_est_avg_rev_pct_fy1e_1w": "1w",
    "eps_est_avg_rev_pct_fy1e_1m": "1m",
    "eps_est_avg_rev_pct_fy1e_3m": "3m",
    "eps_est_avg_rev_pct_fy1e_6m": "6m",
    "eps_est_avg_rev_pct_fy1e_1y": "1y",
}

EPS_GAAP_REVISION_COLS = {
    "eps_gaap_est_avg_rev_pct_fy1e_1m": "1m",
    "eps_gaap_est_avg_rev_pct_fy1e_3m": "3m",
    "eps_gaap_est_avg_rev_pct_fy1e_6m": "6m",
    "eps_gaap_est_avg_rev_pct_fy1e_1y": "1y",
}

logger = logging.getLogger(__name__)


def engineer_analyst_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer analyst quality, consensus, and price target features.

    Features computed (when inputs exist):
    - Analyst consensus: analyst_bullish_pct, analyst_bearish_pct, analyst_conviction (abs diff in pct points)
    - Price target metrics: price_target_spread_pct, price_target_range (alias), consensus_strength (100-spread),
      upside_potential ((median-last)/last * 100), price_target_revision ((median - ytd_ago)/ytd_ago)
    - Coverage quality: analyst_coverage_quality = (# analysts) / log1p(market_cap)
    - Analyst Rating: analyst_rating_normalized, analyst_rating_conviction
    - EPS Revision Momentum: eps_revision_momentum, eps_revision_acceleration,
      eps_gaap_revision_momentum, eps_revision_gaap_divergence
    - Backward-compatibility: target_price_upside_pct alias retained if last_price + price_target present

    Column naming (normalized expected; legacy tolerated where possible):
    - Ratings: strong_buy_ratings, buy_ratings, hold_ratings, sell_ratings, strong_sell_ratings, analyst_rating
    - Targets: price_target_median, price_target_high, price_target_low, price_target_ytd_ago, price_target_count
    - Revisions: eps_est_avg_rev_pct_fy1e_*, eps_gaap_est_avg_rev_pct_fy1e_*
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

    # --- Analyst Rating (numeric consensus score 1-5 scale) ---
    if "analyst_rating" in df.columns:
        rating = df["analyst_rating"].astype(float)
        # Normalize to 0-100 scale (1=Strong Sell -> 0, 5=Strong Buy -> 100)
        result["analyst_rating_normalized"] = (rating - 1) * 25
        # Distance from neutral (3.0 = Hold)
        result["analyst_rating_conviction"] = (rating - 3.0).abs() * 25

    # --- EPS Estimate Revision Momentum ---
    # Captures analyst estimate revision trajectory across multiple time horizons
    eps_rev_cols_present = [c for c in EPS_REVISION_COLS.keys() if c in df.columns]
    if eps_rev_cols_present:
        # Composite revision momentum (weighted average: recent revisions weighted higher)
        weights = {"1w": 0.30, "1m": 0.25, "3m": 0.20, "6m": 0.15, "1y": 0.10}
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for col in eps_rev_cols_present:
            period = EPS_REVISION_COLS[col]
            w = weights.get(period, 0.1)
            weighted_sum += df[col].astype(float).fillna(0) * w
            total_weight += w
        if total_weight > 0:
            result["eps_revision_momentum"] = weighted_sum / total_weight

        # Short-term vs long-term revision divergence (acceleration signal)
        if (
            "eps_est_avg_rev_pct_fy1e_1m" in df.columns
            and "eps_est_avg_rev_pct_fy1e_1y" in df.columns
        ):
            result["eps_revision_acceleration"] = df[
                "eps_est_avg_rev_pct_fy1e_1m"
            ].astype(float) - df["eps_est_avg_rev_pct_fy1e_1y"].astype(float)

    # --- GAAP EPS Revision Momentum (quality signal) ---
    gaap_rev_cols_present = [
        c for c in EPS_GAAP_REVISION_COLS.keys() if c in df.columns
    ]
    if gaap_rev_cols_present:
        weights = {"1m": 0.35, "3m": 0.30, "6m": 0.20, "1y": 0.15}
        weighted_sum = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for col in gaap_rev_cols_present:
            period = EPS_GAAP_REVISION_COLS[col]
            w = weights.get(period, 0.15)
            weighted_sum += df[col].astype(float).fillna(0) * w
            total_weight += w
        if total_weight > 0:
            result["eps_gaap_revision_momentum"] = weighted_sum / total_weight

    # --- GAAP vs Non-GAAP Revision Divergence (earnings quality signal) ---
    if (
        "eps_est_avg_rev_pct_fy1e_3m" in df.columns
        and "eps_gaap_est_avg_rev_pct_fy1e_3m" in df.columns
    ):
        result["eps_revision_gaap_divergence"] = df[
            "eps_est_avg_rev_pct_fy1e_3m"
        ].astype(float) - df["eps_gaap_est_avg_rev_pct_fy1e_3m"].astype(float)

    logger.info("Engineered analyst quality & consensus features")
    return result


def engineer_market_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer market sentiment features from short interest and betas.

    Features computed (when inputs exist):
    - one_day_chg: Pass-through of one_day_pct (already percent units)
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

    logger.info("Engineered market sentiment features (betas, one-day chg)")
    return result


def engineer_price_target_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer price target temporal dynamics features.

    Leverages historical price target data to derive momentum, acceleration,
    and consensus evolution features.

    Features computed (30+):
    - pt_momentum_* (1w, 1m, 3m, 6m, 1y, qtd, ytd, mtd)
    - pt_acceleration_short/long
    - pt_consensus_convergence and pt_high_low_spread_trend
    - analyst_coverage_change_* (1w, 1m, 3m, 6m, 1y)
    - pt_vs_price_momentum
    - pt_skew_trend
    - pt_median_momentum_* (1w, 1m, 3m, 6m, 1y)
    - pt_high_momentum_* and pt_low_momentum_* (range evolution)

    Args:
        df: Input DataFrame with price target historical columns

    Returns:
        DataFrame with price target dynamics features added
    """

    result = df.copy()

    # === Price Target Momentum (Mean/Average) ===
    momentum_pairs = [
        ("pt_momentum_1w", "price_target", "price_target_1w_ago"),
        ("pt_momentum_1m", "price_target", "price_target_1m_ago"),
        ("pt_momentum_3m", "price_target", "price_target_3m_ago"),
        ("pt_momentum_6m", "price_target", "price_target_6m_ago"),
        ("pt_momentum_1y", "price_target", "price_target_1y_ago"),
        ("pt_qtd_momentum", "price_target", "price_target_qtd_ago"),
        ("pt_ytd_momentum", "price_target", "price_target_ytd_ago"),
        ("pt_mtd_momentum", "price_target", "price_target_mtd_ago"),
    ]

    for feature_name, current_col, prior_col in momentum_pairs:
        if current_col in df.columns and prior_col in df.columns:
            result[feature_name] = _safe_pct_change(
                df[current_col].astype(float), df[prior_col].astype(float)
            )

    # === Price Target Median Momentum (more robust to outliers) ===
    median_momentum_pairs = [
        ("pt_median_momentum_1w", "price_target_median", "price_target_median_1w_ago"),
        ("pt_median_momentum_1m", "price_target_median", "price_target_median_1m_ago"),
        ("pt_median_momentum_3m", "price_target_median", "price_target_median_3m_ago"),
        ("pt_median_momentum_6m", "price_target_median", "price_target_median_6m_ago"),
        ("pt_median_momentum_1y", "price_target_median", "price_target_median_1y_ago"),
        (
            "pt_median_momentum_mtd",
            "price_target_median",
            "price_target_median_mtd_ago",
        ),
        (
            "pt_median_momentum_qtd",
            "price_target_median",
            "price_target_median_qtd_ago",
        ),
        (
            "pt_median_momentum_ytd",
            "price_target_median",
            "price_target_median_ytd_ago",
        ),
    ]

    for feature_name, current_col, prior_col in median_momentum_pairs:
        if current_col in df.columns and prior_col in df.columns:
            result[feature_name] = _safe_pct_change(
                df[current_col].astype(float), df[prior_col].astype(float)
            )

    # === Price Target High/Low Momentum (analyst range evolution) ===
    high_momentum_pairs = [
        ("pt_high_momentum_1w", "price_target_high", "price_target_high_1w_ago"),
        ("pt_high_momentum_1m", "price_target_high", "price_target_high_1m_ago"),
        ("pt_high_momentum_3m", "price_target_high", "price_target_high_3m_ago"),
        ("pt_high_momentum_6m", "price_target_high", "price_target_high_6m_ago"),
        ("pt_high_momentum_1y", "price_target_high", "price_target_high_1y_ago"),
    ]

    low_momentum_pairs = [
        ("pt_low_momentum_1w", "price_target_low", "price_target_low_1w_ago"),
        ("pt_low_momentum_1m", "price_target_low", "price_target_low_1m_ago"),
        ("pt_low_momentum_3m", "price_target_low", "price_target_low_3m_ago"),
        ("pt_low_momentum_6m", "price_target_low", "price_target_low_6m_ago"),
        ("pt_low_momentum_1y", "price_target_low", "price_target_low_1y_ago"),
    ]

    for feature_name, current_col, prior_col in (
        high_momentum_pairs + low_momentum_pairs
    ):
        if current_col in df.columns and prior_col in df.columns:
            result[feature_name] = _safe_pct_change(
                df[current_col].astype(float), df[prior_col].astype(float)
            )

    # === Analyst Coverage Count Trajectory (extended periods) ===
    coverage_pairs = [
        ("analyst_coverage_change_1w", "price_target_num_1w_ago"),
        ("analyst_coverage_change_1m", "price_target_num_1m_ago"),
        ("analyst_coverage_change_3m", "price_target_num_3m_ago"),
        ("analyst_coverage_change_6m", "price_target_num_6m_ago"),
        ("analyst_coverage_change_1y", "price_target_num_1y_ago"),
        ("analyst_coverage_change_mtd", "price_target_num_mtd_ago"),
        ("analyst_coverage_change_qtd", "price_target_num_qtd_ago"),
        ("analyst_coverage_change_ytd", "price_target_num_ytd_ago"),
    ]

    count_col = next(
        (c for c in ["price_target_num", "price_target_count"] if c in df.columns),
        None,
    )

    if count_col is not None:
        for feature_name, prior_col in coverage_pairs:
            if prior_col in df.columns:
                result[feature_name] = df[count_col].astype(float) - df[
                    prior_col
                ].astype(float)

    # === Momentum Acceleration ===
    if "pt_momentum_1m" in result.columns and "pt_momentum_3m" in result.columns:
        result["pt_acceleration_short"] = result["pt_momentum_1m"] - result["pt_momentum_3m"]

    if "pt_momentum_3m" in result.columns and "pt_momentum_1y" in result.columns:
        result["pt_acceleration_long"] = result["pt_momentum_3m"] - result["pt_momentum_1y"]

    # Median-based acceleration (more robust)
    if (
        "pt_median_momentum_1m" in result.columns
        and "pt_median_momentum_3m" in result.columns
    ):
        result["pt_median_acceleration_short"] = (
            result["pt_median_momentum_1m"] - result["pt_median_momentum_3m"]
        )

    # === Consensus Range Evolution (extended time horizons) ===
    spread_periods = [
        (
            "3m",
            "price_target_high_3m_ago",
            "price_target_low_3m_ago",
            "price_target_median_3m_ago",
        ),
        (
            "6m",
            "price_target_high_6m_ago",
            "price_target_low_6m_ago",
            "price_target_median_6m_ago",
        ),
        (
            "1y",
            "price_target_high_1y_ago",
            "price_target_low_1y_ago",
            "price_target_median_1y_ago",
        ),
    ]

    if all(
        c in df.columns
        for c in ("price_target_high", "price_target_low", "price_target_median")
    ):
        current_spread = _safe_div(
            df["price_target_high"].astype(float) - df["price_target_low"].astype(float),
            df["price_target_median"].astype(float),
        )

        for period, high_col, low_col, med_col in spread_periods:
            if all(c in df.columns for c in (high_col, low_col, med_col)):
                prior_spread = _safe_div(
                    df[high_col].astype(float) - df[low_col].astype(float),
                    df[med_col].astype(float),
                )
                result[f"pt_consensus_convergence_{period}"] = (
                    prior_spread - current_spread
                )
                result[f"pt_spread_trend_{period}"] = current_spread - prior_spread

        # Default consensus convergence (3m) for backward compatibility
        if "pt_consensus_convergence_3m" in result.columns:
            result["pt_consensus_convergence"] = result["pt_consensus_convergence_3m"]
            result["pt_high_low_spread_trend"] = result["pt_spread_trend_3m"]

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

    logger.info("Engineered price target dynamics features (30+ features)")
    return result


def _safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Calculate percentage change with safe division."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (current - previous) / previous.abs().replace(0, pd.NA)
    return result.astype("Float64")


def engineer_analyst_coverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer analyst coverage momentum features.

    Creates:
    - analyst_coverage_change_* (1w, 1m, 3m, 6m, 1y, mtd, qtd, ytd)
    - analyst_coverage_trend: Weighted trend score
    - analyst_coverage_acceleration: Coverage momentum acceleration
    - analyst_interest_score: Normalized coverage interest indicator

    Args:
        df: DataFrame with price_target_num columns

    Returns:
        DataFrame with analyst coverage features added
    """
    result = df.copy()

    # Current count column variations
    current_col = next(
        (c for c in ["price_target_num", "price_target_count"] if c in result.columns),
        None,
    )

    if current_col is None:
        logger.warning("No price_target_num column found for analyst coverage features")
        return result

    # Coverage change for all available periods
    coverage_periods = {
        "1w": "price_target_num_1w_ago",
        "1m": "price_target_num_1m_ago",
        "3m": "price_target_num_3m_ago",
        "6m": "price_target_num_6m_ago",
        "1y": "price_target_num_1y_ago",
        "mtd": "price_target_num_mtd_ago",
        "qtd": "price_target_num_qtd_ago",
        "ytd": "price_target_num_ytd_ago",
    }

    for period, prior_col in coverage_periods.items():
        feature_name = f"analyst_coverage_change_{period}"
        if prior_col in result.columns:
            result[feature_name] = result[current_col].astype(float) - result[
                prior_col
            ].astype(float)
        elif feature_name not in result.columns:
            result[feature_name] = 0

    # Weighted trend (1M × 0.4 + 3M × 0.35 + 6M × 0.25) / current count
    if current_col in result.columns:
        trend_weights = [
            ("analyst_coverage_change_1m", 0.40),
            ("analyst_coverage_change_3m", 0.35),
            ("analyst_coverage_change_6m", 0.25),
        ]
        weighted_sum = pd.Series(0.0, index=result.index)
        for col, weight in trend_weights:
            if col in result.columns:
                weighted_sum += result[col].fillna(0) * weight

        result["analyst_coverage_trend"] = _safe_div(
            weighted_sum, result[current_col].astype(float)
        ).fillna(0)

    # Coverage acceleration (short-term vs medium-term momentum)
    if (
        "analyst_coverage_change_1m" in result.columns
        and "analyst_coverage_change_3m" in result.columns
    ):
        result["analyst_coverage_acceleration"] = (
            result["analyst_coverage_change_1m"]
            - result["analyst_coverage_change_3m"] / 3
        )

    # Analyst interest score: normalized coverage relative to market cap tier
    if current_col in result.columns and "market_cap" in result.columns:
        # log1p(market_cap) normalizes by firm size
        log_mcap = np.log1p(result["market_cap"].astype(float).clip(lower=1))
        result["analyst_interest_score"] = _safe_div(
            result[current_col].astype(float), log_mcap
        )

    logger.info("Engineered analyst coverage features (12+ features)")
    return result
