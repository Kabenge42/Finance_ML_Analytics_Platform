"""Momentum and technical analysis feature engineering."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer price momentum and technical indicators.

    Features (added when sufficient columns are available):
    - price_momentum_1m, 3m, 6m, 1y: Percent change vs price_Nm_ago columns
    - price_acceleration_3m: mom_3m - mom_1m (rate-of-change proxy)
    - rsi_14d: 14-day RSI computed from last_price and price_{1..14}d_ago columns
    - rsi_30d: 30-day RSI if 30-day history is present
    - ma_crossover_signal: 1 if MA20>MA50 and price>MA50, -1 if MA20<MA50 and price<MA50, else 0
    - price_distance_from_ma: % distance of last_price from MA50
    - return_stability_score: total_return_1y_pct / volatility_1y_pct
    - sharpe_proxy: (total_return_1y_pct - risk_free_rate_pct) / volatility_1y_pct

    Notes:
    - All percentage features are expressed in percent (not decimals).
    - Missing inputs result in NaN for the affected features; no exceptions raised.
    """
    result = df.copy()

    def pct_change(cur: pd.Series, prev: pd.Series) -> pd.Series:
        """Calculate percentage change between current and previous values."""
        return _safe_div(cur - prev, prev) * 100

    # Basic momentum windows
    if "last_price" in df.columns and "price_20d_ago" in df.columns:
        result["momentum_20d"] = pct_change(df["last_price"], df["price_20d_ago"])
    if "last_price" in df.columns and "price_1m_ago" in df.columns:
        result["price_momentum_1m"] = pct_change(df["last_price"], df["price_1m_ago"])
    if "last_price" in df.columns and "price_3m_ago" in df.columns:
        result["price_momentum_3m"] = pct_change(df["last_price"], df["price_3m_ago"])
    if "last_price" in df.columns and "price_6m_ago" in df.columns:
        result["price_momentum_6m"] = pct_change(df["last_price"], df["price_6m_ago"])
    if "last_price" in df.columns and "price_1y_ago" in df.columns:
        result["price_momentum_1y"] = pct_change(df["last_price"], df["price_1y_ago"])

    # Acceleration vs 1m
    if "price_momentum_3m" in result.columns and "price_momentum_1m" in result.columns:
        result["price_acceleration_3m"] = result["price_momentum_3m"] - result["price_momentum_1m"]

    # RSI helper (row-wise due to per-row wide history columns)
    def compute_rsi_row(row: pd.Series, period: int) -> float:
        """Compute RSI (Relative Strength Index) for a single row over specified period."""
        # Build sequence oldest->newest using daily columns if present
        prices = []
        # Include historical days period back to 1 day
        for d in range(period, 0, -1):
            col = f"price_{d}d_ago"
            prices.append(row.get(col, np.nan))
        prices.append(row.get("last_price", np.nan))
        arr = np.asarray(prices, dtype=float)
        if np.isnan(arr).any():
            return np.nan
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0 and avg_gain == 0:
            return 50.0  # flat
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    # RSI 14d
    have_14 = (
        all(f"price_{d}d_ago" in df.columns for d in range(14, 0, -1))
        and "last_price" in df.columns
    )
    if have_14:
        result["rsi_14d"] = df.apply(lambda r: compute_rsi_row(r, 14), axis=1)

    # RSI 30d
    have_30 = (
        all(f"price_{d}d_ago" in df.columns for d in range(30, 0, -1))
        and "last_price" in df.columns
    )
    if have_30:
        result["rsi_30d"] = df.apply(lambda r: compute_rsi_row(r, 30), axis=1)

    # Use existing EMA columns from database instead of computing MAs from non-existent daily history
    if "ema_20d" in df.columns:
        result["ma_20d_simple"] = df["ema_20d"]
    if "ema_50d" in df.columns:
        result["ma_50d_simple"] = df["ema_50d"]

    if "last_price" in df.columns:
        # price distance from MA50
        if "ma_50d_simple" in result.columns:
            result["price_distance_from_ma"] = (
                _safe_div(df["last_price"] - result["ma_50d_simple"], result["ma_50d_simple"]) * 100
            )
        # crossover signal
        if "ma_20d_simple" in result.columns and "ma_50d_simple" in result.columns:
            cond_up = (result["ma_20d_simple"] > result["ma_50d_simple"]) & (
                df["last_price"] > result["ma_50d_simple"]
            )
            cond_down = (result["ma_20d_simple"] < result["ma_50d_simple"]) & (
                df["last_price"] < result["ma_50d_simple"]
            )
            signal = pd.Series(0, index=df.index, dtype=float)
            signal[cond_up] = 1.0
            signal[cond_down] = -1.0
            result["ma_crossover_signal"] = signal

    # Return stability and Sharpe proxy
    if "last_price" in df.columns and "price_1y_ago" in df.columns:
        total_return_pct = pct_change(df["last_price"], df["price_1y_ago"]).rename(
            "total_return_1y_pct"
        )
        result["total_return_1y_pct"] = total_return_pct
        if "volatility_1y_pct" in df.columns:
            vol = df["volatility_1y_pct"].astype(float).replace(0, np.nan)
            result["return_stability_score"] = _safe_div(total_return_pct, vol)
            try:
                rf = float(os.getenv("RISK_FREE_RATE_PCT", "0.0"))
            except (ValueError, TypeError):
                rf = 0.0
            excess = total_return_pct - rf
            result["sharpe_proxy"] = _safe_div(excess, vol)

    # Beta Momentum (1Y - 5Y)
    if "beta_1y" in df.columns and "beta_5y" in df.columns:
        result["beta_momentum"] = df["beta_1y"].astype(float) - df["beta_5y"].astype(float)

    # Volatility Term Structure (1M / 1Y)
    if "volatility_1m" in df.columns and "volatility_1y" in df.columns:
        result["volatility_term_structure"] = _safe_div(
            df["volatility_1m"].astype(float), df["volatility_1y"].astype(float)
        )

    logger.info("Engineered momentum & technical features")
    return result

def engineer_technical_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer technical analysis features using EMA and 52-week data.

    Phase 9.3 Schema Version 1.3: Leverages new technical indicator columns
    (EMA 20D/50D/100D/250D, 52W High/Low, Rel. Volume).

    Features created:
    - EMA crossover signals (20D/50D, 50D/250D)
    - Price vs EMA deviations
    - EMA slope and trend consistency
    - 52-week range position indicators
    - Volume momentum composite

    Args:
        df: Input DataFrame with technical indicator columns

    Returns:
        DataFrame with technical analysis features added
    """
    result = df.copy()

    # 1. EMA-Based Signals
    if "ema_20d" in df.columns and "ema_50d" in df.columns:
        # EMA crossover: 1 if 20D > 50D (bullish), -1 if 20D < 50D (bearish), 0 if equal/missing
        result["ema_crossover_20_50"] = np.where(
            df["ema_20d"] > df["ema_50d"],
            1,
            np.where(df["ema_20d"] < df["ema_50d"], -1, 0),
        )

    if "ema_50d" in df.columns and "ema_250d" in df.columns:
        result["ema_crossover_50_250"] = np.where(
            df["ema_50d"] > df["ema_250d"],
            1,
            np.where(df["ema_50d"] < df["ema_250d"], -1, 0),
        )

    if "last_price" in df.columns and "ema_20d" in df.columns:
        result["price_vs_ema_20d"] = _safe_div(df["last_price"] - df["ema_20d"], df["ema_20d"])

    if "last_price" in df.columns and "ema_250d" in df.columns:
        result["price_vs_ema_250d"] = _safe_div(df["last_price"] - df["ema_250d"], df["ema_250d"])

    # EMA slope (approximate using 20D vs 50D as proxy for slope)
    if "ema_20d" in df.columns and "ema_50d" in df.columns:
        result["ema_slope_20d"] = _safe_div(df["ema_20d"] - df["ema_50d"], df["ema_50d"])

    # EMA trend consistency: check if EMAs are aligned (all ascending or descending)
    if all(c in df.columns for c in ["ema_20d", "ema_50d", "ema_100d", "ema_250d"]):
        bullish = (
            (df["ema_20d"] > df["ema_50d"])
            & (df["ema_50d"] > df["ema_100d"])
            & (df["ema_100d"] > df["ema_250d"])
        )
        bearish = (
            (df["ema_20d"] < df["ema_50d"])
            & (df["ema_50d"] < df["ema_100d"])
            & (df["ema_100d"] < df["ema_250d"])
        )
        result["ema_trend_consistency"] = np.where(bullish, 1, np.where(bearish, -1, 0))

    # 2. 52-Week Position Features
    if "52w_high_adj" in df.columns and "last_price" in df.columns:
        result["pct_off_52w_high"] = _safe_div(
            df["52w_high_adj"] - df["last_price"], df["52w_high_adj"]
        )

    if "52w_low_adj" in df.columns and "last_price" in df.columns:
        result["pct_above_52w_low"] = _safe_div(
            df["last_price"] - df["52w_low_adj"], df["52w_low_adj"]
        )

    if all(c in df.columns for c in ["52w_high_adj", "52w_low_adj", "last_price"]):
        # 52W range position: 0 at low, 1 at high
        range_width = df["52w_high_adj"] - df["52w_low_adj"]
        result["52w_range_position"] = _safe_div(
            df["last_price"] - df["52w_low_adj"], range_width
        ).clip(0, 1)

        # Near 52W high/low flags
        result["near_52w_high_flag"] = (
            (result["pct_off_52w_high"] <= 0.05).fillna(False).astype(int)
        )
        result["near_52w_low_flag"] = (
            (result["pct_above_52w_low"] <= 0.05).fillna(False).astype(int)
        )

    # 3. Volume & Momentum Composite
    if "rel_volume" in df.columns and "price_chg_pct_1m" in df.columns:
        result["volume_momentum_score"] = df["rel_volume"] * df["price_chg_pct_1m"]

    # Breakout signal: EMA crossover + near 52W high
    if "ema_crossover_20_50" in result.columns and "near_52w_high_flag" in result.columns:
        result["breakout_signal"] = (
            ((result["ema_crossover_20_50"] == 1) & (result["near_52w_high_flag"] == 1))
            .fillna(False)
            .astype(int)
        )

    logger.info("Engineered technical analysis features (Phase 9.3 Schema 1.3)")
    return result

def engineer_market_microstructure_features(
    df: pd.DataFrame,
    price_col: str = "last_price",
    high_col: str = "52w_high_adj",
    low_col: str = "52w_low_adj",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Engineer market microstructure features (volatility, momentum, moving averages).

    Args:
        df: Input DataFrame
        price_col: Name of price column
        high_col: Name of high price column (for range calculation)
        low_col: Name of low price column (for range calculation)
        group_col: Optional grouping column (e.g., ticker) for time-series features

    Returns:
        DataFrame with market microstructure features added
    """
    result = df.copy()

    if price_col not in df.columns:
        logger.warning(
            f"Price column '{price_col}' not found, skipping market microstructure features"
        )
        return result

    # Price range indicator (requires high and low prices)
    if high_col in df.columns and low_col in df.columns:
        price_range = df[high_col] - df[low_col]
        result["price_range_pct"] = _safe_div(price_range, df[price_col]) * 100

    # Time-series features (volatility, momentum, moving averages)
    if group_col and group_col in df.columns:
        # Historical volatility (30, 60, 90 day rolling windows)
        for window in [30, 60, 90]:
            result[f"volatility_{window}d"] = df.groupby(group_col)[price_col].transform(
                lambda x: x.pct_change()
                .rolling(window=window, min_periods=max(1, window // 2))
                .std()
                * 100
            )

        # Momentum (rate of change over 20 days)
        result["momentum_20d"] = df.groupby(group_col)[price_col].transform(
            lambda x: x.pct_change(periods=20) * 100
        )

        # Moving averages (20, 50 day)
        for window in [20, 50]:
            result[f"ma_{window}d"] = df.groupby(group_col)[price_col].transform(
                lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).mean()
            )
    else:
        # Without grouping, calculate simple rolling features if enough data
        if len(df) >= 30:
            for window in [30, 60, 90]:
                if len(df) >= window:
                    result[f"volatility_{window}d"] = (
                        df[price_col]
                        .pct_change()
                        .rolling(window=window, min_periods=window // 2)
                        .std()
                        * 100
                    )

            if len(df) >= 20:
                result["momentum_20d"] = df[price_col].pct_change(periods=20) * 100

            for window in [20, 50]:
                if len(df) >= window:
                    result[f"ma_{window}d"] = (
                        df[price_col].rolling(window=window, min_periods=window // 2).mean()
                    )

    logger.info("Engineered market microstructure features")
    return result
