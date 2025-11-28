"""
finance_ml.ml_workflow.classification.labels - Event label creation for classification

This module provides sophisticated event classification label creation methods.

ALL 19 METHODS NOW USE COMPLETE PHASE 9.3 FEATURE SETS (196 features total):

Original methods (enhanced with all Phase 9.3 features):
- price_momentum: 27 Momentum & Technical features
- valuation: 23 Valuation Ratios features  
- fundamental: 12 Profitability features
- volatility: Stability and volatility features
- analyst_rating: 10 Analyst Sentiment features
- market_events: 5 Market Sentiment features
- combined_signals: Multi-metric composite

Specialized methods (Phase 9.4, enhanced with all Phase 9.3 features):
- profitability_event: 12 Profitability features
- leverage_event: 9 Leverage & Liquidity features
- liquidity_event: Liquidity subset features
- efficiency_event: 4 Efficiency Ratios features
- growth_event: 6 Growth Metrics features
- quality_event: 18 Quality & Risk features
- composite_event: 5 Composite Scores features

New methods (Phase 9.3 category coverage - 77 additional features):
- cashflow_event: 5 Cash Flow features
- capital_allocation_event: 23 Capital Allocation features
- employee_productivity_event: 16 Employee Productivity features
- balance_sheet_event: 8 Balance Sheet Dynamics features
- revenue_forecast_event: 9 Revenue Forecasting features

Phase 9.6 enhancement: 5-class label granularity:
- 0 = Strong Negative
- 1 = Negative
- 2 = Neutral
- 3 = Positive
- 4 = Strong Positive
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "create_enhanced_event_labels",
]


def _get_column(df: pd.DataFrame, *column_names: str) -> Optional[pd.Series]:
    """Get the first available column from a list of column name variations.

    This helper function tries multiple column name variations to support both:
    - Original columns from data loading (e.g., "p_e", "roe")
    - Phase 9.3 engineered columns (e.g., "p_e_ratio", "price_momentum_1m")

    Args:
        df: DataFrame to search
        *column_names: Column names to try in order of preference

    Returns:
        Series if any column found, None otherwise

    Example:
        >>> # Try Phase 9.3 column first, fall back to original
        >>> pe_col = _get_column(df, "p_e_ratio", "p_e")
    """
    for col_name in column_names:
        if col_name in df.columns:
            return df[col_name]
    return None


def create_enhanced_event_labels(
    df: pd.DataFrame,
    method: str = "price_momentum",
    threshold_positive: float = 10.0,
    threshold_negative: float = -10.0,
    use_sector_adjustment: bool = False,
) -> np.ndarray:
    """Create sophisticated event classification labels using Phase 9.3 features.

    ALL 19 METHODS - 196 Phase 9.3 features allocated across categories:

    Original methods (enhanced with complete Phase 9.3 feature sets):
    1. price_momentum: 27 Momentum & Technical features (price trends, RSI, MA, EMA, 52W, volume)
    2. valuation: 23 Valuation Ratios features (P/E, P/B, P/S, EV/EBITDA, EV/Sales, trends, stability)
    3. fundamental: 12 Profitability features (margins, ROE/ROA/ROIC, quality, trends)
    4. volatility: Stability and volatility features (return stability, Sharpe proxy)
    5. analyst_rating: 10 Analyst Sentiment features (ratings, conviction, consensus, target revisions)
    6. market_events: 5 Market Sentiment features (sector rotation, short interest, beta, momentum)
    7. combined_signals: Multi-metric composite (momentum + valuation + fundamentals)

    Specialized methods (Phase 9.4, enhanced with complete Phase 9.3 feature sets):
    8. profitability_event: 12 Profitability features (same as fundamental)
    9. leverage_event: 9 Leverage & Liquidity features (debt ratios, coverage, liquidity ratios)
    10. liquidity_event: Liquidity subset features (current, quick, cash ratios)
    11. efficiency_event: 4 Efficiency Ratios features (turnover ratios, revenue per employee)
    12. growth_event: 6 Growth Metrics features (revenue/earnings/EBITDA growth, YoY variants)
    13. quality_event: 18 Quality & Risk features (accounting quality, distress, exceptional items)
    14. composite_event: 5 Composite Scores features (Piotroski, Altman Z, Beneish M, quality, momentum)

    New methods (Phase 9.3 category coverage - 77 additional features):
    15. cashflow_event: 5 Cash Flow features (CFO growth, FCF metrics, cash conversion quality)
    16. capital_allocation_event: 23 Capital Allocation features (dividends, CAPEX, reinvestment, M&A)
    17. employee_productivity_event: 16 Employee Productivity features (revenue/profit per employee, growth)
    18. balance_sheet_event: 8 Balance Sheet Dynamics features (asset/equity/debt growth, working capital)
    19. revenue_forecast_event: 9 Revenue Forecasting features (analyst estimates, consensus, uncertainty)

    Args:
        df: DataFrame with Phase 9.3 engineered features
        method: Event detection method (see list above)
        threshold_positive: Threshold for positive catalyst (%) - legacy parameter, some methods use it
        threshold_negative: Threshold for negative catalyst (%) - legacy parameter, some methods use it
        use_sector_adjustment: If True, adjust thresholds by sector volatility (used by price_momentum)

    Returns:
        numpy array of labels (0=Strong Negative, 1=Negative, 2=Neutral, 3=Positive, 4=Strong Positive)

    Examples:
        >>> # Price momentum with all 27 Momentum & Technical features
        >>> labels = create_enhanced_event_labels(df, method="price_momentum")

        >>> # Capital allocation with all 23 Capital Allocation features
        >>> labels = create_enhanced_event_labels(df, method="capital_allocation_event")

        >>> # Employee productivity with all 16 Employee Productivity features
        >>> labels = create_enhanced_event_labels(df, method="employee_productivity_event")
    """
    labels = np.zeros(len(df), dtype=int)

    if method == "price_momentum":
        # Enhanced price momentum using ALL 27 Phase 9.3 Momentum & Technical features
        # Categories: Price momentum, RSI, MA signals, EMA signals, 52W position, volume, stability
        momentum_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Legacy: price_target vs last_price (backward compatible)
        if "price_target" in df.columns and "last_price" in df.columns:
            price_diff_pct = (df["price_target"] - df["last_price"]) / df["last_price"] * 100.0
            momentum_score += price_diff_pct / 10.0
            signal_count += 1

        # Price momentum features (1m, 3m, 6m, 1y)
        for col, weight in [("price_momentum_1m", 1.0), ("price_momentum_3m", 0.8), 
                           ("price_momentum_6m", 0.6), ("price_momentum_1y", 0.4)]:
            mom = _get_column(df, col)
            if mom is not None:
                momentum_score += (mom / 10.0) * weight
                signal_count += 1

        # Price acceleration
        accel = _get_column(df, "price_acceleration_3m")
        if accel is not None:
            momentum_score += (accel / 20.0)
            signal_count += 1

        # RSI indicators (14d, 30d)
        for col in ["rsi_14d", "rsi_30d"]:
            rsi = _get_column(df, col)
            if rsi is not None:
                momentum_score += (rsi - 50) / 10.0
                signal_count += 1
                break  # Use first available

        # Moving average signals
        ma_signal = _get_column(df, "ma_crossover_signal")
        if ma_signal is not None:
            momentum_score += ma_signal * 3.0
            signal_count += 1

        price_dist = _get_column(df, "price_distance_from_ma")
        if price_dist is not None:
            momentum_score += price_dist / 10.0
            signal_count += 1

        # EMA-based signals
        for col in ["ema_crossover_20_50", "ema_crossover_50_250"]:
            ema_cross = _get_column(df, col)
            if ema_cross is not None:
                momentum_score += ema_cross * 2.0
                signal_count += 1

        for col in ["price_vs_ema_20d", "price_vs_ema_250d"]:
            ema_dev = _get_column(df, col)
            if ema_dev is not None:
                momentum_score += ema_dev * 10.0
                signal_count += 1

        ema_slope = _get_column(df, "ema_slope_20d")
        if ema_slope is not None:
            momentum_score += ema_slope * 10.0
            signal_count += 1

        ema_trend = _get_column(df, "ema_trend_consistency")
        if ema_trend is not None:
            momentum_score += ema_trend * 2.0
            signal_count += 1

        # 52-week position features
        w52_pos = _get_column(df, "52w_range_position")
        if w52_pos is not None:
            momentum_score += (w52_pos - 0.5) * 4.0  # Center at 0.5, scale to -2 to +2
            signal_count += 1

        for col in ["pct_off_52w_high", "pct_above_52w_low"]:
            w52_metric = _get_column(df, col)
            if w52_metric is not None:
                if "off" in col:
                    momentum_score += -w52_metric * 5.0  # Invert: low % off high is bullish
                else:
                    momentum_score += w52_metric * 5.0
                signal_count += 1

        for flag in ["near_52w_high_flag", "near_52w_low_flag"]:
            flag_col = _get_column(df, flag)
            if flag_col is not None:
                if "high" in flag:
                    momentum_score += flag_col * 1.0
                else:
                    momentum_score += -flag_col * 1.0
                signal_count += 1

        breakout = _get_column(df, "breakout_signal")
        if breakout is not None:
            momentum_score += breakout * 3.0
            signal_count += 1

        # Volume momentum
        vol_mom = _get_column(df, "volume_momentum_score")
        if vol_mom is not None:
            momentum_score += vol_mom / 10.0
            signal_count += 1

        # Stability and risk-adjusted return
        stability = _get_column(df, "return_stability_score")
        if stability is not None:
            momentum_score += stability.clip(-2, 2)
            signal_count += 1

        sharpe = _get_column(df, "sharpe_proxy")
        if sharpe is not None:
            momentum_score += sharpe.clip(-2, 2)
            signal_count += 1

        total_ret = _get_column(df, "total_return_1y_pct")
        if total_ret is not None:
            momentum_score += total_ret / 20.0
            signal_count += 1

        # Additional MA features
        for col in ["ma_20d_simple", "ma_50d_simple"]:
            ma = _get_column(df, col)
            if ma is not None and "last_price" in df.columns:
                ma_dev = (df["last_price"] - ma) / ma * 100
                momentum_score += ma_dev / 10.0
                signal_count += 1
                break  # Use first available

        if signal_count == 0:
            logger.warning("No momentum indicators available, returning all neutral")
            return labels

        # Average across available signals
        momentum_score /= signal_count

        # Sector-specific adjustment
        if use_sector_adjustment and "sector" in df.columns:
            for sector in df["sector"].unique():
                sector_mask = df["sector"] == sector
                sector_vol = momentum_score[sector_mask].std()
                # Adjust thresholds based on sector volatility (5-class)
                adj_strong_pos = 1.5 * (1 + sector_vol / 2.0)
                adj_pos = 0.75 * (1 + sector_vol / 2.0)
                adj_neg = -0.75 * (1 + sector_vol / 2.0)
                adj_strong_neg = -1.5 * (1 + sector_vol / 2.0)

                labels[sector_mask & (momentum_score >= adj_strong_pos)] = 4
                labels[
                    sector_mask & (momentum_score >= adj_pos) & (momentum_score < adj_strong_pos)
                ] = 3
                labels[
                    sector_mask & (momentum_score <= adj_neg) & (momentum_score > adj_strong_neg)
                ] = 1
                labels[sector_mask & (momentum_score <= adj_strong_neg)] = 0
        else:
            # Use quantile-based thresholds to ensure balanced class distribution
            labels[momentum_score >= momentum_score.quantile(0.85)] = 4  # Strong positive momentum
            labels[
                (momentum_score >= momentum_score.quantile(0.65))
                & (momentum_score < momentum_score.quantile(0.85))
            ] = 3  # Positive momentum
            labels[
                (momentum_score >= momentum_score.quantile(0.35))
                & (momentum_score < momentum_score.quantile(0.65))
            ] = 2  # Neutral momentum
            labels[
                (momentum_score <= momentum_score.quantile(0.35))
                & (momentum_score > momentum_score.quantile(0.15))
            ] = 1  # Negative momentum
            labels[momentum_score <= momentum_score.quantile(0.15)] = 0  # Strong negative momentum

    elif method == "valuation":
        # Enhanced valuation using ALL 23 Phase 9.3 Valuation Ratios features
        # Lower multiples = undervalued (positive), higher multiples = overvalued (negative)
        valuation_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Core valuation ratios (lower is better)
        for col in ["p_e_ratio", "p_b_ratio", "p_s_ratio", "ev_ebitda_ratio", "ev_sales_ratio", 
                   "peg_ratio", "dividend_yield"]:
            metric = _get_column(df, col, col.replace("_ratio", ""))
            if metric is not None:
                # Calculate percentile ranking within sector if available
                if "sector" in df.columns:
                    percentile = df.groupby("sector")[metric.name].rank(pct=True)
                else:
                    percentile = metric.rank(pct=True)
                # Invert for most metrics: low percentile = undervalued (good)
                if col == "dividend_yield":
                    valuation_score += percentile  # High div yield is good
                else:
                    valuation_score += 1.0 - percentile  # Low multiple is good
                signal_count += 1

        # EV/EBITDA momentum and trends (negative momentum = multiple decreasing = improving valuation)
        for col in ["ev_ebitda_momentum", "ev_ebitda_vs_3y_avg", "ev_ebitda_forward_discount"]:
            metric = _get_column(df, col)
            if metric is not None:
                # Negative values = improving (multiple dropping), positive = worsening
                valuation_score += -metric.clip(-2, 2)  # Invert and clip
                signal_count += 1

        # EV/Sales trends
        for col in ["ev_sales_trend_1y", "ev_sales_trend_3y", "ev_sales_vs_3y_avg", 
                   "ev_sales_forward_discount"]:
            metric = _get_column(df, col)
            if metric is not None:
                valuation_score += -metric.clip(-2, 2)
                signal_count += 1

        # EV/Sales volatility (lower is better stability)
        ev_sales_vol = _get_column(df, "ev_sales_quarterly_volatility")
        if ev_sales_vol is not None:
            percentile = ev_sales_vol.rank(pct=True)
            valuation_score += 1.0 - percentile  # Low volatility is good
            signal_count += 1

        # P/E momentum and trends
        for col in ["p_e_momentum_yoy", "p_e_momentum_qoq", "p_e_vs_3y_avg", "p_e_forward_discount"]:
            metric = _get_column(df, col)
            if metric is not None:
                valuation_score += -metric.clip(-2, 2)
                signal_count += 1

        # Valuation stability and trend consistency (higher is better)
        for col in ["valuation_stability_score", "valuation_trend_consistency"]:
            metric = _get_column(df, col)
            if metric is not None:
                valuation_score += metric.clip(-2, 2)
                signal_count += 1

        # Valuation extreme flag (extreme = bad, inverted)
        extreme_flag = _get_column(df, "valuation_extreme_flag")
        if extreme_flag is not None:
            valuation_score += -extreme_flag * 2.0  # Penalty for extreme valuation
            signal_count += 1

        # Growth implied by valuation (positive = market expects growth = good)
        growth_implied = _get_column(df, "growth_implied_by_valuation")
        if growth_implied is not None:
            valuation_score += growth_implied.clip(-2, 2)
            signal_count += 1

        if signal_count == 0:
            logger.warning("No valuation metrics available, returning all neutral")
            return labels

        # Average across available signals
        valuation_score /= signal_count

        # High score (undervalued) = positive, low score (overvalued) = negative (5-class)
        # Use quantile-based thresholds to ensure balanced class distribution
        labels[valuation_score >= valuation_score.quantile(0.85)] = (
            4  # Top 15% = strongly undervalued
        )
        labels[
            (valuation_score >= valuation_score.quantile(0.65))
            & (valuation_score < valuation_score.quantile(0.85))
        ] = 3  # Undervalued
        labels[
            (valuation_score >= valuation_score.quantile(0.35))
            & (valuation_score < valuation_score.quantile(0.65))
        ] = 2  # Neutral
        labels[
            (valuation_score <= valuation_score.quantile(0.35))
            & (valuation_score > valuation_score.quantile(0.15))
        ] = 1  # Overvalued
        labels[valuation_score <= valuation_score.quantile(0.15)] = 0  # Strongly overvalued

    elif method == "fundamental":
        # Enhanced fundamental using ALL 12 Phase 9.3 Profitability features
        # Margins, profitability ratios, quality, trends, and operating leverage
        fundamental_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Margin metrics (higher is better)
        for col in ["gross_margin_pct", "operating_margin_pct", "net_margin_pct"]:
            margin = _get_column(df, col, col.replace("_pct", ""))
            if margin is not None:
                fundamental_score += margin / 10.0  # Normalize
                signal_count += 1

        # Margin trends (positive trend is better)
        for col in ["gross_margin_trend", "ebitda_margin_trend"]:
            trend = _get_column(df, col)
            if trend is not None:
                fundamental_score += trend * 10.0  # Scale up
                signal_count += 1

        # Profitability ratios (higher is better)
        for col in ["roe", "roa", "roic"]:
            ratio = _get_column(df, col)
            if ratio is not None:
                fundamental_score += ratio / 10.0  # Normalize
                signal_count += 1

        # Earnings quality (higher is better)
        earnings_qual = _get_column(df, "earnings_quality_score")
        if earnings_qual is not None:
            fundamental_score += earnings_qual
            signal_count += 1

        # EBIT/EBITDA adjustment ratios (closer to 1.0 is better, no large adjustments)
        for col in ["ebit_adjustment_ratio", "ebitda_adjustment_ratio"]:
            adj_ratio = _get_column(df, col)
            if adj_ratio is not None:
                # Penalize large deviations from 1.0
                deviation = (adj_ratio - 1.0).abs()
                fundamental_score += -deviation.clip(0, 2)
                signal_count += 1

        # Operating leverage (higher is typically better in growth phases)
        op_lev = _get_column(df, "operating_leverage")
        if op_lev is not None:
            fundamental_score += op_lev.clip(-2, 2)
            signal_count += 1

        if signal_count == 0:
            logger.warning("No fundamental metrics available, returning all neutral")
            return labels

        # Average across available signals
        fundamental_score /= signal_count

        # High fundamentals = positive, low fundamentals = negative (5-class)
        labels[fundamental_score >= fundamental_score.quantile(0.85)] = 4
        labels[
            (fundamental_score >= fundamental_score.quantile(0.65))
            & (fundamental_score < fundamental_score.quantile(0.85))
        ] = 3
        labels[
            (fundamental_score >= fundamental_score.quantile(0.35))
            & (fundamental_score < fundamental_score.quantile(0.65))
        ] = 2
        labels[
            (fundamental_score <= fundamental_score.quantile(0.35))
            & (fundamental_score > fundamental_score.quantile(0.15))
        ] = 1
        labels[fundamental_score <= fundamental_score.quantile(0.15)] = 0

    elif method == "volatility":
        # Enhanced volatility-based events using Phase 9.3 stability features
        # High volatility + low stability = negative (risky)
        # Low volatility + high stability = positive (stable)

        volatility_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Traditional volatility columns (original data)
        vol_cols = [c for c in df.columns if "volatility" in c.lower()]
        if vol_cols:
            volatility = df[vol_cols[0]]
            # Normalize: high vol = positive score (bad), low vol = negative score (good)
            vol_normalized = (volatility - volatility.mean()) / (volatility.std() + 1e-10)
            volatility_score += vol_normalized
            signal_count += 1

        # Return stability score (Phase 9.3) - higher is better
        stability = _get_column(df, "return_stability_score")
        if stability is not None:
            # Normalize and invert: low stability = positive score (bad)
            stab_normalized = -(stability - stability.mean()) / (stability.std() + 1e-10)
            volatility_score += stab_normalized
            signal_count += 1

        # Sharpe proxy (Phase 9.3) - higher is better
        sharpe = _get_column(df, "sharpe_proxy")
        if sharpe is not None:
            # Normalize and invert: low sharpe = positive score (bad)
            sharpe_normalized = -(sharpe - sharpe.mean()) / (sharpe.std() + 1e-10)
            volatility_score += sharpe_normalized
            signal_count += 1

        if signal_count == 0:
            logger.warning("No volatility indicators available, returning all neutral")
            return labels

        # Average across available signals
        volatility_score /= signal_count

        # High volatility score (risky) = negative, low score (stable) = positive (5-class)
        # Use quantile-based thresholds to ensure balanced class distribution
        # Note: For volatility, low score is good, so we invert the quantile logic
        labels[volatility_score <= volatility_score.quantile(0.15)] = (
            4  # Very low volatility/high stability = strong positive
        )
        labels[
            (volatility_score <= volatility_score.quantile(0.35))
            & (volatility_score > volatility_score.quantile(0.15))
        ] = 3  # Low volatility = positive
        labels[
            (volatility_score >= volatility_score.quantile(0.35))
            & (volatility_score < volatility_score.quantile(0.65))
        ] = 2  # Neutral volatility
        labels[
            (volatility_score >= volatility_score.quantile(0.65))
            & (volatility_score < volatility_score.quantile(0.85))
        ] = 1  # High volatility = negative
        labels[volatility_score >= volatility_score.quantile(0.85)] = (
            0  # Very high volatility/low stability = strong negative
        )

    elif method == "analyst_rating":
        # Analyst rating events using ALL 10 Phase 9.3 Analyst Sentiment features
        analyst_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Analyst rating change (legacy)
        if "analyst_rating_change" in df.columns:
            analyst_score += df["analyst_rating_change"] * 2.0
            signal_count += 1
        elif "analyst_rating" in df.columns:
            rating_map = {
                "Strong Buy": 2.0, "Buy": 1.0, "Outperform": 1.0,
                "Hold": 0.0, "Neutral": 0.0,
                "Sell": -1.0, "Strong Sell": -2.0, "Underperform": -1.0,
            }
            analyst_score += df["analyst_rating"].map(rating_map).fillna(0)
            signal_count += 1

        # Upside potential and target price upside
        for col in ["upside_potential", "target_price_upside_pct"]:
            upside = _get_column(df, col)
            if upside is not None:
                analyst_score += upside.clip(-50, 50) / 25.0
                signal_count += 1
                break  # Use first available

        # Bullish/bearish percentages
        bullish_pct = _get_column(df, "analyst_bullish_pct")
        if bullish_pct is not None:
            analyst_score += (bullish_pct - 50) / 25.0
            signal_count += 1

        bearish_pct = _get_column(df, "analyst_bearish_pct")
        if bearish_pct is not None:
            analyst_score += -(bearish_pct - 50) / 25.0  # Invert: high bearish is bad
            signal_count += 1

        # Analyst conviction (higher = stronger signal)
        conviction = _get_column(df, "analyst_conviction")
        if conviction is not None:
            analyst_score += conviction.clip(-2, 2)
            signal_count += 1

        # Consensus strength (higher = more agreement)
        consensus = _get_column(df, "consensus_strength")
        if consensus is not None:
            analyst_score += consensus.clip(-2, 2)
            signal_count += 1

        # Price target range and spread (tighter range = more confident)
        pt_range = _get_column(df, "price_target_range")
        if pt_range is not None:
            # Narrow range is good (inverse)
            percentile = pt_range.rank(pct=True)
            analyst_score += (1.0 - percentile) * 2.0  # Reward narrow ranges
            signal_count += 1

        pt_spread = _get_column(df, "price_target_spread_pct")
        if pt_spread is not None:
            # Low spread is good
            percentile = pt_spread.rank(pct=True)
            analyst_score += (1.0 - percentile) * 2.0
            signal_count += 1

        # Price target revision (positive revision = bullish)
        pt_revision = _get_column(df, "price_target_revision")
        if pt_revision is not None:
            analyst_score += pt_revision.clip(-2, 2)
            signal_count += 1

        # Coverage quality (as amplifier)
        coverage_quality = _get_column(df, "analyst_coverage_quality")
        if coverage_quality is not None:
            quality_weight = 0.5 + coverage_quality.clip(0, 2) / 2.0
            analyst_score *= quality_weight

        if signal_count == 0:
            logger.warning("No analyst rating indicators available, returning all neutral")
            return labels

        analyst_score /= signal_count

        # Positive analyst score = positive catalyst (5-class)
        # Use quantile-based thresholds to ensure balanced class distribution
        labels[analyst_score >= analyst_score.quantile(0.85)] = 4
        labels[
            (analyst_score >= analyst_score.quantile(0.65))
            & (analyst_score < analyst_score.quantile(0.85))
        ] = 3
        labels[
            (analyst_score >= analyst_score.quantile(0.35))
            & (analyst_score < analyst_score.quantile(0.65))
        ] = 2
        labels[
            (analyst_score <= analyst_score.quantile(0.35))
            & (analyst_score > analyst_score.quantile(0.15))
        ] = 1
        labels[analyst_score <= analyst_score.quantile(0.15)] = 0

    elif method == "market_events":
        # Market events using ALL 5 Phase 9.3 Market Sentiment features
        market_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Sector-relative performance (calculated from price)
        if "sector" in df.columns and "last_price" in df.columns:
            sector_perf = df.groupby("sector")["last_price"].transform(
                lambda x: (x / x.mean() - 1.0) * 100
            )
            market_score += sector_perf / 10.0
            signal_count += 1

        # Short interest ratio (high = bearish)
        short_interest = _get_column(df, "short_interest_ratio", "short_int_pct")
        if short_interest is not None:
            short_normalized = -(short_interest - short_interest.median()) / (
                short_interest.std() + 1e-10
            )
            market_score += short_normalized
            signal_count += 1

        # Systematic risk trend (increasing beta = negative)
        beta_trend = _get_column(df, "systematic_risk_trend")
        if beta_trend is not None:
            market_score += -beta_trend * 2.0
            signal_count += 1

        # Beta stability (higher stability = positive)
        beta_stab = _get_column(df, "beta_stability")
        if beta_stab is not None:
            market_score += beta_stab.clip(-2, 2)
            signal_count += 1

        # Momentum indicator
        momentum_20d = _get_column(df, "momentum_20d")
        if momentum_20d is not None:
            market_score += momentum_20d / 10.0
            signal_count += 1

        # Price range (tighter range = more stable = positive)
        price_range = _get_column(df, "price_range_pct")
        if price_range is not None:
            percentile = price_range.rank(pct=True)
            market_score += (1.0 - percentile) * 2.0  # Invert: low range is good
            signal_count += 1

        if signal_count == 0:
            logger.warning("No market event indicators available, returning all neutral")
            return labels

        market_score /= signal_count

        # Positive market signals = positive (5-class)
        # Use quantile-based thresholds to ensure balanced class distribution
        labels[market_score >= market_score.quantile(0.85)] = 4
        labels[
            (market_score >= market_score.quantile(0.65))
            & (market_score < market_score.quantile(0.85))
        ] = 3
        labels[
            (market_score >= market_score.quantile(0.35))
            & (market_score < market_score.quantile(0.65))
        ] = 2
        labels[
            (market_score <= market_score.quantile(0.35))
            & (market_score > market_score.quantile(0.15))
        ] = 1
        labels[market_score <= market_score.quantile(0.15)] = 0

    elif method == "combined_signals":
        # Combined signals: Multi-metric composite combining price momentum, valuation, and fundamentals
        # This method provides a balanced view across different signal types
        composite_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Component 1: Price momentum (from price_momentum method logic)
        if "price_target" in df.columns and "last_price" in df.columns:
            price_diff_pct = (df["price_target"] - df["last_price"]) / df["last_price"] * 100.0
            composite_score += price_diff_pct / 10.0  # Normalize
            signal_count += 1

        # Component 2: Valuation (lower P/E, P/B = undervalued = positive)
        pe_col = _get_column(df, "p_e_ratio", "p_e", "p_e_ltm")
        if pe_col is not None:
            pe_percentile = pe_col.rank(pct=True)
            # Invert: lower P/E = higher score
            composite_score += (1.0 - pe_percentile) * 2.0
            signal_count += 1

        pb_col = _get_column(df, "p_b_ratio", "p_b", "p_b_ltm")
        if pb_col is not None:
            pb_percentile = pb_col.rank(pct=True)
            # Invert: lower P/B = higher score
            composite_score += (1.0 - pb_percentile) * 2.0
            signal_count += 1

        # Component 3: Fundamental (higher margins = positive)
        margin_cols = [
            _get_column(df, "net_margin_pct", "net_income_margin_pct_ltm"),
            _get_column(df, "gross_margin_pct", "gross_profit_margin_pct_ltm"),
        ]
        for margin_col in margin_cols:
            if margin_col is not None:
                margin_percentile = margin_col.rank(pct=True)
                composite_score += margin_percentile * 2.0
                signal_count += 1
                break  # Use first available

        if signal_count == 0:
            logger.warning("No combined signal indicators available, returning all neutral")
            return labels

        # Average across available signals
        composite_score /= signal_count

        # Apply 5-class thresholds
        labels[composite_score >= composite_score.quantile(0.85)] = 4  # Top 15% = strong positive
        labels[
            (composite_score >= composite_score.quantile(0.65))
            & (composite_score < composite_score.quantile(0.85))
        ] = 3  # Positive
        labels[
            (composite_score >= composite_score.quantile(0.35))
            & (composite_score < composite_score.quantile(0.65))
        ] = 2  # Neutral (35-65%)
        labels[
            (composite_score <= composite_score.quantile(0.35))
            & (composite_score > composite_score.quantile(0.15))
        ] = 1  # Negative
        labels[composite_score <= composite_score.quantile(0.15)] = (
            0  # Bottom 15% = strong negative
        )

    elif method == "profitability_event":
        # Profitability events using ALL 12 Phase 9.3 Profitability features
        # Same logic as fundamental method but focused on profitability metrics
        profitability_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Margin metrics
        for col in ["gross_margin_pct", "operating_margin_pct", "net_margin_pct"]:
            margin = _get_column(df, col, col.replace("_pct", ""))
            if margin is not None:
                profitability_score += margin / 10.0
                signal_count += 1

        # Margin trends
        for col in ["gross_margin_trend", "ebitda_margin_trend"]:
            trend = _get_column(df, col)
            if trend is not None:
                profitability_score += trend * 10.0
                signal_count += 1

        # Profitability ratios
        for col in ["roe", "roa", "roic"]:
            ratio = _get_column(df, col)
            if ratio is not None:
                profitability_score += ratio / 10.0
                signal_count += 1

        # Earnings quality
        earnings_qual = _get_column(df, "earnings_quality_score")
        if earnings_qual is not None:
            profitability_score += earnings_qual
            signal_count += 1

        # Adjustment ratios
        for col in ["ebit_adjustment_ratio", "ebitda_adjustment_ratio"]:
            adj_ratio = _get_column(df, col)
            if adj_ratio is not None:
                deviation = (adj_ratio - 1.0).abs()
                profitability_score += -deviation.clip(0, 2)
                signal_count += 1

        # Operating leverage
        op_lev = _get_column(df, "operating_leverage")
        if op_lev is not None:
            profitability_score += op_lev.clip(-2, 2)
            signal_count += 1

        if signal_count == 0:
            logger.warning("No profitability metrics available, returning all neutral")
            return labels

        profitability_score /= signal_count

        # High profitability = positive (5-class)
        labels[profitability_score >= profitability_score.quantile(0.85)] = 4
        labels[
            (profitability_score >= profitability_score.quantile(0.65))
            & (profitability_score < profitability_score.quantile(0.85))
        ] = 3
        labels[
            (profitability_score >= profitability_score.quantile(0.35))
            & (profitability_score < profitability_score.quantile(0.65))
        ] = 2
        labels[
            (profitability_score <= profitability_score.quantile(0.35))
            & (profitability_score > profitability_score.quantile(0.15))
        ] = 1
        labels[profitability_score <= profitability_score.quantile(0.15)] = 0

    elif method == "leverage_event":
        # Leverage events using ALL 9 Phase 9.3 Leverage & Liquidity features
        leverage_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Debt ratios (lower is better)
        for col in ["debt_to_equity", "net_debt_to_ebitda", "debt_to_assets"]:
            debt_metric = _get_column(df, col)
            if debt_metric is not None:
                percentile = debt_metric.rank(pct=True)
                leverage_score += 1.0 - percentile  # Invert: low debt is good
                signal_count += 1

        # Interest coverage (higher is better)
        int_cov = _get_column(df, "interest_coverage")
        if int_cov is not None:
            percentile = int_cov.rank(pct=True)
            leverage_score += percentile  # High coverage is good
            signal_count += 1

        # Liquidity ratios (higher is better)
        for col in ["current_ratio", "quick_ratio", "cash_ratio"]:
            liq_metric = _get_column(df, col)
            if liq_metric is not None:
                percentile = liq_metric.rank(pct=True)
                leverage_score += percentile  # High liquidity is good
                signal_count += 1

        # Equity ratio (higher is better - less levered)
        eq_ratio = _get_column(df, "equity_ratio")
        if eq_ratio is not None:
            percentile = eq_ratio.rank(pct=True)
            leverage_score += percentile
            signal_count += 1

        # Working capital efficiency (higher is better)
        wc_sales = _get_column(df, "working_capital_to_sales")
        if wc_sales is not None:
            leverage_score += wc_sales.clip(-2, 2)
            signal_count += 1

        if signal_count == 0:
            logger.warning("No leverage/liquidity metrics available, returning all neutral")
            return labels

        leverage_score /= signal_count

        # High score (low leverage, high liquidity) = positive (5-class)
        labels[leverage_score >= leverage_score.quantile(0.85)] = 4
        labels[
            (leverage_score >= leverage_score.quantile(0.65))
            & (leverage_score < leverage_score.quantile(0.85))
        ] = 3
        labels[
            (leverage_score >= leverage_score.quantile(0.35))
            & (leverage_score < leverage_score.quantile(0.65))
        ] = 2
        labels[
            (leverage_score <= leverage_score.quantile(0.35))
            & (leverage_score > leverage_score.quantile(0.15))
        ] = 1
        labels[leverage_score <= leverage_score.quantile(0.15)] = 0

    elif method == "liquidity_event":
        # Liquidity-based events using current ratio, quick ratio
        liquidity_cols = [
            c for c in ["current_ratio", "quick_ratio", "cash_ratio"] if c in df.columns
        ]
        if not liquidity_cols:
            logger.warning("No liquidity columns available, returning all neutral")
            return labels

        # Calculate average liquidity score
        liquidity_data = df[liquidity_cols].fillna(0)
        avg_liquidity = liquidity_data.mean(axis=1)

        # Check if there's any variance in the data
        if avg_liquidity.std() > 1e-10:
            # High liquidity = positive, low liquidity = negative (5-class)
            labels[avg_liquidity >= avg_liquidity.quantile(0.85)] = 4  # Top 15% = strong positive
            labels[
                (avg_liquidity >= avg_liquidity.quantile(0.65))
                & (avg_liquidity < avg_liquidity.quantile(0.85))
            ] = 3  # Positive
            labels[
                (avg_liquidity >= avg_liquidity.quantile(0.35))
                & (avg_liquidity < avg_liquidity.quantile(0.65))
            ] = 2  # Neutral (35-65%)
            labels[
                (avg_liquidity <= avg_liquidity.quantile(0.35))
                & (avg_liquidity > avg_liquidity.quantile(0.15))
            ] = 1  # Negative
            labels[avg_liquidity <= avg_liquidity.quantile(0.15)] = (
                0  # Bottom 15% = strong negative
            )
        else:
            logger.warning("No variance in liquidity data, returning all neutral")
            return labels

    elif method == "efficiency_event":
        # Efficiency events using ALL 4 Phase 9.3 Efficiency Ratios features
        efficiency_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Turnover ratios (higher is better)
        for col in ["asset_turnover", "inventory_turnover", "receivables_turnover", "revenue_per_employee"]:
            metric = _get_column(df, col)
            if metric is not None:
                percentile = metric.rank(pct=True)
                efficiency_score += percentile
                signal_count += 1

        if signal_count == 0:
            logger.warning("No efficiency metrics available, returning all neutral")
            return labels

        efficiency_score /= signal_count

        # High efficiency = positive (5-class)
        labels[efficiency_score >= efficiency_score.quantile(0.85)] = 4
        labels[
            (efficiency_score >= efficiency_score.quantile(0.65))
            & (efficiency_score < efficiency_score.quantile(0.85))
        ] = 3
        labels[
            (efficiency_score >= efficiency_score.quantile(0.35))
            & (efficiency_score < efficiency_score.quantile(0.65))
        ] = 2
        labels[
            (efficiency_score <= efficiency_score.quantile(0.35))
            & (efficiency_score > efficiency_score.quantile(0.15))
        ] = 1
        labels[efficiency_score <= efficiency_score.quantile(0.15)] = 0

    elif method == "growth_event":
        # Growth events using ALL 6 Phase 9.3 Growth Metrics features
        growth_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Growth metrics (higher is better)
        for col in ["revenue_growth", "earnings_growth", "ebitda_growth", 
                   "revenue_growth_yoy", "ebitda_growth_yoy", "eps_growth_yoy"]:
            metric = _get_column(df, col)
            if metric is not None:
                growth_score += metric / 10.0  # Normalize
                signal_count += 1

        if signal_count == 0:
            logger.warning("No growth metrics available, returning all neutral")
            return labels

        growth_score /= signal_count

        # High growth = positive (5-class)
        # FIXED 2025-11-26: Use pd.qcut for proper quintile-based binning to ensure balanced distribution
        # Previous manual quantile logic had gaps/overlaps causing 64.2% class 0 imbalance
        try:
            # pd.qcut creates balanced bins; labels 0-4 map to quintiles (low to high growth)
            labels_series = pd.qcut(
                growth_score.rank(method="first"),  # Use rank to handle ties
                q=5,
                labels=[0, 1, 2, 3, 4],
                duplicates="drop",
            )
            labels = labels_series.fillna(2).astype(int).values  # NaN -> neutral (2)
        except ValueError:
            # Fallback if qcut fails (e.g., too few unique values)
            logger.warning("pd.qcut failed for growth_event, using percentile fallback")
            labels[growth_score >= growth_score.quantile(0.85)] = 4
            labels[
                (growth_score >= growth_score.quantile(0.65))
                & (growth_score < growth_score.quantile(0.85))
            ] = 3
            labels[
                (growth_score >= growth_score.quantile(0.35))
                & (growth_score < growth_score.quantile(0.65))
            ] = 2
            labels[
                (growth_score <= growth_score.quantile(0.35))
                & (growth_score > growth_score.quantile(0.15))
            ] = 1
            labels[growth_score <= growth_score.quantile(0.15)] = 0

    elif method == "quality_event":
        # Quality events using ALL 18 Phase 9.3 Quality & Risk features
        quality_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Direct quality metrics (higher is better)
        for col in ["accounting_quality_score", "analyst_coverage_quality"]:
            metric = _get_column(df, col)
            if metric is not None:
                quality_score += metric.clip(-2, 2)
                signal_count += 1

        # Distress and risk scores (lower distress = higher quality)
        distress = _get_column(df, "distress_risk_score")
        if distress is not None:
            percentile = distress.rank(pct=True)
            quality_score += 1.0 - percentile  # Invert: low distress is good
            signal_count += 1

        # Altman Z-Score trend (positive trend = improving quality)
        z_trend = _get_column(df, "altman_z_trend")
        if z_trend is not None:
            quality_score += z_trend.clip(-2, 2)
            signal_count += 1

        # Z-Score volatility (lower is better)
        z_vol = _get_column(df, "z_score_volatility")
        if z_vol is not None:
            percentile = z_vol.rank(pct=True)
            quality_score += 1.0 - percentile
            signal_count += 1

        # Exceptional items (lower is better)
        for col in ["exceptional_items_to_ebitda", "exceptional_items_to_ni_pct", 
                   "exceptional_items_trend", "total_exceptional_items_ltm"]:
            metric = _get_column(df, col)
            if metric is not None:
                percentile = metric.abs().rank(pct=True)
                quality_score += 1.0 - percentile  # Invert: low exceptional items is good
                signal_count += 1

        # Goodwill and intangible metrics (lower intensity = higher quality)
        for col in ["goodwill_to_assets", "goodwill_to_assets_pct", "intangible_intensity", 
                   "intangibles_to_assets_pct"]:
            metric = _get_column(df, col)
            if metric is not None:
                percentile = metric.rank(pct=True)
                quality_score += 1.0 - percentile
                signal_count += 1

        # Goodwill change rate (high change = risk)
        gw_change = _get_column(df, "goodwill_change_rate")
        if gw_change is not None:
            percentile = gw_change.abs().rank(pct=True)
            quality_score += 1.0 - percentile
            signal_count += 1

        # Red flags (presence = bad)
        # FIXED 2025-11-25: Further reduced penalty from -0.5 to -0.2 to achieve balanced distribution
        # Previous -0.5 penalty still caused 63.4% class 0 imbalance (expected: 15% per quantile design)
        # Root cause: Most stocks have ≥1 red flag; even -0.5 penalty per flag creates strong negative bias
        # New -0.2 penalty preserves signal while allowing quantile thresholds to work as designed
        for flag in ["goodwill_impairment_flag", "has_goodwill_impairment", 
                    "has_asset_writedown", "has_restructuring"]:
            metric = _get_column(df, flag)
            if metric is not None:
                quality_score += -metric * 0.2  # Penalty for flags (reduced from -0.5, originally -2.0)
                signal_count += 1

        # Restructuring intensity (lower is better)
        restructuring = _get_column(df, "restructuring_intensity")
        if restructuring is not None:
            percentile = restructuring.rank(pct=True)
            quality_score += 1.0 - percentile
            signal_count += 1

        if signal_count == 0:
            logger.warning("No quality metrics available, returning all neutral")
            return labels

        quality_score /= signal_count

        # High quality = positive (5-class)
        # FIXED 2025-11-26: Use pd.qcut for proper quintile-based binning to ensure balanced distribution
        # Previous manual quantile logic had gaps/overlaps causing 63.2% class 0 imbalance
        try:
            # pd.qcut creates balanced bins; labels 0-4 map to quintiles (low to high quality)
            labels_series = pd.qcut(
                quality_score.rank(method="first"),  # Use rank to handle ties
                q=5,
                labels=[0, 1, 2, 3, 4],
                duplicates="drop",
            )
            labels = labels_series.fillna(2).astype(int).values  # NaN -> neutral (2)
        except ValueError:
            # Fallback if qcut fails (e.g., too few unique values)
            logger.warning("pd.qcut failed for quality_event, using percentile fallback")
            labels[quality_score >= quality_score.quantile(0.85)] = 4
            labels[
                (quality_score >= quality_score.quantile(0.65))
                & (quality_score < quality_score.quantile(0.85))
            ] = 3
            labels[
                (quality_score >= quality_score.quantile(0.35))
                & (quality_score < quality_score.quantile(0.65))
            ] = 2
            labels[
                (quality_score <= quality_score.quantile(0.35))
                & (quality_score > quality_score.quantile(0.15))
            ] = 1
            labels[quality_score <= quality_score.quantile(0.15)] = 0

    elif method == "composite_event":
        # Composite events using ALL 5 Phase 9.3 Composite Scores features
        composite_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Piotroski F-Score (0-9, higher is better)
        f_score = _get_column(df, "piotroski_f_score")
        if f_score is not None:
            composite_score += f_score / 9.0
            signal_count += 1

        # Altman Z-Score (>2.99 safe, <1.81 distress)
        z_score = _get_column(df, "altman_z_score")
        if z_score is not None:
            composite_score += z_score.clip(0, 5) / 5.0
            signal_count += 1

        # Beneish M-Score (<-1.78 unlikely manipulator, lower is better)
        m_score = _get_column(df, "beneish_m_score")
        if m_score is not None:
            composite_score += (1.0 - m_score.clip(-3, 1)) / 4.0
            signal_count += 1

        # Composite quality score (higher is better)
        comp_qual = _get_column(df, "composite_quality_score")
        if comp_qual is not None:
            composite_score += comp_qual.clip(0, 1)
            signal_count += 1

        # Momentum score (higher is better)
        mom_score = _get_column(df, "momentum_score")
        if mom_score is not None:
            composite_score += mom_score.clip(-1, 1) / 2.0 + 0.5  # Normalize to 0-1
            signal_count += 1

        if signal_count == 0:
            logger.warning("No composite score columns available, returning all neutral")
            return labels

        composite_score /= signal_count

        # High composite score = positive (5-class)
        labels[composite_score >= composite_score.quantile(0.85)] = 4
        labels[
            (composite_score >= composite_score.quantile(0.65))
            & (composite_score < composite_score.quantile(0.85))
        ] = 3
        labels[
            (composite_score >= composite_score.quantile(0.35))
            & (composite_score < composite_score.quantile(0.65))
        ] = 2
        labels[
            (composite_score <= composite_score.quantile(0.35))
            & (composite_score > composite_score.quantile(0.15))
        ] = 1
        labels[composite_score <= composite_score.quantile(0.15)] = 0

    elif method == "cashflow_event":
        # Cash flow events using ALL 5 Phase 9.3 Cash Flow features
        cashflow_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # CFO growth (higher is better)
        cfo_growth = _get_column(df, "cfo_growth_yoy")
        if cfo_growth is not None:
            cashflow_score += cfo_growth / 10.0
            signal_count += 1

        # CFO quality metrics (higher is better)
        for col in ["cfo_to_net_income", "fcf_to_net_income"]:
            metric = _get_column(df, col)
            if metric is not None:
                cashflow_score += metric.clip(-2, 2)
                signal_count += 1

        # FCF margin and stability (higher is better)
        for col in ["fcf_margin", "fcf_stability"]:
            metric = _get_column(df, col)
            if metric is not None:
                cashflow_score += metric / 10.0
                signal_count += 1

        if signal_count == 0:
            logger.warning("No cash flow metrics available, returning all neutral")
            return labels

        cashflow_score /= signal_count

        labels[cashflow_score >= cashflow_score.quantile(0.85)] = 4
        labels[
            (cashflow_score >= cashflow_score.quantile(0.65))
            & (cashflow_score < cashflow_score.quantile(0.85))
        ] = 3
        labels[
            (cashflow_score >= cashflow_score.quantile(0.35))
            & (cashflow_score < cashflow_score.quantile(0.65))
        ] = 2
        labels[
            (cashflow_score <= cashflow_score.quantile(0.35))
            & (cashflow_score > cashflow_score.quantile(0.15))
        ] = 1
        labels[cashflow_score <= cashflow_score.quantile(0.15)] = 0

    elif method == "capital_allocation_event":
        # Capital allocation events using ALL 23 Phase 9.3 Capital Allocation features
        capital_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Dividend metrics (higher dividend safety/growth = positive)
        for col in ["dividend_safety_score", "dividend_consistency_score", "dividend_growth_trend",
                   "dividend_aristocrat_flag", "income_stock_flag"]:
            metric = _get_column(df, col)
            if metric is not None:
                capital_score += metric.clip(-2, 2)
                signal_count += 1

        # Dividend coverage and yield (higher is better)
        for col in ["fcf_dividend_coverage", "dividend_yield_vs_sector", "div_yield_ltm"]:
            metric = _get_column(df, col)
            if metric is not None:
                percentile = metric.rank(pct=True)
                capital_score += percentile
                signal_count += 1

        # Payout ratio (moderate is better, clip extremes)
        for col in ["dividend_payout_ratio", "payout_ratio"]:
            metric = _get_column(df, col)
            if metric is not None:
                # Optimal payout around 40-60%, penalize extremes
                optimal_dist = (metric - 50).abs()
                percentile = optimal_dist.rank(pct=True)
                capital_score += 1.0 - percentile
                signal_count += 1
                break

        # CAPEX metrics (stable, moderate growth is better)
        for col in ["capex_intensity", "capex_to_depreciation"]:
            metric = _get_column(df, col)
            if metric is not None:
                capital_score += metric.clip(-2, 2) / 2.0
                signal_count += 1

        capex_growth = _get_column(df, "capex_growth_rate")
        if capex_growth is not None:
            capital_score += capex_growth.clip(-20, 20) / 20.0
            signal_count += 1

        capex_vol = _get_column(df, "capex_volatility")
        if capex_vol is not None:
            percentile = capex_vol.rank(pct=True)
            capital_score += 1.0 - percentile  # Low volatility is good
            signal_count += 1

        # Working capital efficiency (higher is better)
        for col in ["working_capital_efficiency", "working_capital_trend"]:
            metric = _get_column(df, col)
            if metric is not None:
                capital_score += metric.clip(-2, 2)
                signal_count += 1

        # Reinvestment rate (moderate is better)
        reinvest = _get_column(df, "reinvestment_rate")
        if reinvest is not None:
            capital_score += reinvest.clip(-1, 1)
            signal_count += 1

        # Total shareholder return yield (higher is better)
        tsr = _get_column(df, "total_shareholder_return_yield")
        if tsr is not None:
            capital_score += tsr / 10.0
            signal_count += 1

        # Acquisition intensity (lower is typically better)
        acq_int = _get_column(df, "acquisition_intensity")
        if acq_int is not None:
            percentile = acq_int.rank(pct=True)
            capital_score += 1.0 - percentile
            signal_count += 1

        # Currency risk flag (absence is better)
        curr_risk = _get_column(df, "currency_risk_flag")
        if curr_risk is not None:
            capital_score += -curr_risk
            signal_count += 1

        # Dividend timing/frequency signals
        for col in ["days_since_ex_date", "dividend_frequency_encoded", "dividend_streak_years"]:
            metric = _get_column(df, col)
            if metric is not None:
                if "days" in col:
                    # Recent ex-date is neutral
                    pass
                elif "frequency" in col:
                    capital_score += metric / 10.0
                else:  # streak
                    capital_score += metric / 20.0
                signal_count += 1

        if signal_count == 0:
            logger.warning("No capital allocation metrics available, returning all neutral")
            return labels

        capital_score /= signal_count

        # FIXED 2025-11-26: Use pd.qcut for proper quintile-based binning to ensure balanced distribution
        # Previous manual quantile logic had gaps/overlaps causing 69.1% class 0 imbalance
        try:
            # pd.qcut creates balanced bins; labels 0-4 map to quintiles (low to high capital allocation quality)
            labels_series = pd.qcut(
                capital_score.rank(method="first"),  # Use rank to handle ties
                q=5,
                labels=[0, 1, 2, 3, 4],
                duplicates="drop",
            )
            labels = labels_series.fillna(2).astype(int).values  # NaN -> neutral (2)
        except ValueError:
            # Fallback if qcut fails (e.g., too few unique values)
            logger.warning("pd.qcut failed for capital_allocation_event, using percentile fallback")
            labels[capital_score >= capital_score.quantile(0.85)] = 4
            labels[
                (capital_score >= capital_score.quantile(0.65))
                & (capital_score < capital_score.quantile(0.85))
            ] = 3
            labels[
                (capital_score >= capital_score.quantile(0.35))
                & (capital_score < capital_score.quantile(0.65))
            ] = 2
            labels[
                (capital_score <= capital_score.quantile(0.35))
                & (capital_score > capital_score.quantile(0.15))
            ] = 1
            labels[capital_score <= capital_score.quantile(0.15)] = 0

    elif method == "employee_productivity_event":
        # Employee productivity events using ALL 16 Phase 9.3 Employee Productivity features
        productivity_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Revenue/profit per employee (higher is better)
        for col in ["revenue_per_employee_fy", "revenue_per_employee_ltm", "profit_per_employee",
                   "ebitda_per_employee", "operating_income_per_employee", "assets_per_employee"]:
            metric = _get_column(df, col)
            if metric is not None:
                percentile = metric.rank(pct=True)
                productivity_score += percentile
                signal_count += 1

        # Revenue per employee trend and comparison (positive trend is better)
        for col in ["revenue_per_employee_trend", "revenue_per_employee_vs_5y_pct"]:
            metric = _get_column(df, col)
            if metric is not None:
                productivity_score += metric.clip(-20, 20) / 20.0
                signal_count += 1

        # Employee growth metrics (moderate positive growth is good)
        for col in ["employee_growth_yoy", "employee_growth_yoy_pct", "employee_growth_qoq",
                   "employee_growth_cagr_5y"]:
            metric = _get_column(df, col)
            if metric is not None:
                # Moderate growth 0-10% is optimal
                if col.endswith("_pct"):
                    optimal_score = metric.clip(-20, 20) / 20.0
                else:
                    optimal_score = metric.clip(-0.2, 0.2) / 0.2
                productivity_score += optimal_score
                signal_count += 1

        # Employee growth acceleration (positive acceleration is good)
        emp_accel = _get_column(df, "employee_growth_acceleration")
        if emp_accel is not None:
            productivity_score += emp_accel.clip(-2, 2)
            signal_count += 1

        # Hiring intensity (moderate is better)
        hiring = _get_column(df, "hiring_intensity_score")
        if hiring is not None:
            productivity_score += hiring.clip(-2, 2)
            signal_count += 1

        # Workforce volatility (lower is better)
        wf_vol = _get_column(df, "workforce_volatility")
        if wf_vol is not None:
            percentile = wf_vol.rank(pct=True)
            productivity_score += 1.0 - percentile
            signal_count += 1

        # Employee base scale flag (large scale can be advantage)
        scale_flag = _get_column(df, "employee_base_scale_flag")
        if scale_flag is not None:
            productivity_score += scale_flag * 0.5
            signal_count += 1

        if signal_count == 0:
            logger.warning("No employee productivity metrics available, returning all neutral")
            return labels

        productivity_score /= signal_count

        labels[productivity_score >= productivity_score.quantile(0.85)] = 4
        labels[
            (productivity_score >= productivity_score.quantile(0.65))
            & (productivity_score < productivity_score.quantile(0.85))
        ] = 3
        labels[
            (productivity_score >= productivity_score.quantile(0.35))
            & (productivity_score < productivity_score.quantile(0.65))
        ] = 2
        labels[
            (productivity_score <= productivity_score.quantile(0.35))
            & (productivity_score > productivity_score.quantile(0.15))
        ] = 1
        labels[productivity_score <= productivity_score.quantile(0.15)] = 0

    elif method == "balance_sheet_event":
        # Balance sheet events using ALL 8 Phase 9.3 Balance Sheet Dynamics features
        balance_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Growth rates (positive growth is generally good)
        for col in ["asset_growth_rate", "equity_growth_rate", "balance_sheet_expansion"]:
            metric = _get_column(df, col)
            if metric is not None:
                balance_score += metric.clip(-20, 20) / 20.0
                signal_count += 1

        # Debt growth (lower is better)
        debt_growth = _get_column(df, "debt_growth_rate")
        if debt_growth is not None:
            balance_score += -debt_growth.clip(-20, 20) / 20.0
            signal_count += 1

        # Retained earnings growth (higher is better)
        re_growth = _get_column(df, "retained_earnings_growth")
        if re_growth is not None:
            balance_score += re_growth.clip(-20, 20) / 20.0
            signal_count += 1

        # Earnings retention rate (higher is better for growth companies)
        retention = _get_column(df, "earnings_retention_rate")
        if retention is not None:
            balance_score += retention.clip(-1, 1)
            signal_count += 1

        # Working capital ratio (positive is better)
        wc_ratio = _get_column(df, "working_capital_ratio")
        if wc_ratio is not None:
            balance_score += wc_ratio.clip(-2, 2)
            signal_count += 1

        # Current ratio trend (improving is better)
        cr_trend = _get_column(df, "current_ratio_trend")
        if cr_trend is not None:
            balance_score += cr_trend.clip(-2, 2)
            signal_count += 1

        if signal_count == 0:
            logger.warning("No balance sheet metrics available, returning all neutral")
            return labels

        balance_score /= signal_count

        labels[balance_score >= balance_score.quantile(0.85)] = 4
        labels[
            (balance_score >= balance_score.quantile(0.65))
            & (balance_score < balance_score.quantile(0.85))
        ] = 3
        labels[
            (balance_score >= balance_score.quantile(0.35))
            & (balance_score < balance_score.quantile(0.65))
        ] = 2
        labels[
            (balance_score <= balance_score.quantile(0.35))
            & (balance_score > balance_score.quantile(0.15))
        ] = 1
        labels[balance_score <= balance_score.quantile(0.15)] = 0

    elif method == "revenue_forecast_event":
        # Revenue forecast events using ALL 9 Phase 9.3 Revenue Forecasting features
        forecast_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Implied growth rates (higher is better)
        for col in ["revenue_growth_implied_fy1e", "revenue_growth_implied_ntm"]:
            metric = _get_column(df, col)
            if metric is not None:
                forecast_score += metric.clip(-20, 20) / 20.0
                signal_count += 1

        # Revenue growth acceleration (positive is better)
        accel = _get_column(df, "revenue_growth_acceleration")
        if accel is not None:
            forecast_score += accel.clip(-2, 2)
            signal_count += 1

        # Estimate spreads and uncertainty (lower spread = more confidence = better)
        for col in ["revenue_estimate_spread_fy1e", "revenue_estimate_spread_ntm",
                   "revenue_consensus_uncertainty_score"]:
            metric = _get_column(df, col)
            if metric is not None:
                percentile = metric.rank(pct=True)
                forecast_score += 1.0 - percentile  # Invert: low spread is good
                signal_count += 1

        # Estimate bias (low bias is better)
        bias = _get_column(df, "avg_vs_median_bias")
        if bias is not None:
            percentile = bias.abs().rank(pct=True)
            forecast_score += 1.0 - percentile
            signal_count += 1

        # Estimate confidence flag (high confidence is good)
        conf_flag = _get_column(df, "estimate_confidence_flag")
        if conf_flag is not None:
            forecast_score += conf_flag
            signal_count += 1

        # Growth surprise potential (positive surprise = good)
        surprise = _get_column(df, "growth_surprise_potential")
        if surprise is not None:
            forecast_score += surprise.clip(-2, 2)
            signal_count += 1

        if signal_count == 0:
            logger.warning("No revenue forecast metrics available, returning all neutral")
            return labels

        forecast_score /= signal_count

        labels[forecast_score >= forecast_score.quantile(0.85)] = 4
        labels[
            (forecast_score >= forecast_score.quantile(0.65))
            & (forecast_score < forecast_score.quantile(0.85))
        ] = 3
        labels[
            (forecast_score >= forecast_score.quantile(0.35))
            & (forecast_score < forecast_score.quantile(0.65))
        ] = 2
        labels[
            (forecast_score <= forecast_score.quantile(0.35))
            & (forecast_score > forecast_score.quantile(0.15))
        ] = 1
        labels[forecast_score <= forecast_score.quantile(0.15)] = 0

    else:
        logger.error(f"Unknown method: {method}")

    logger.info(
        f"Created labels with method={method}: "
        f"Strong Negative={np.sum(labels == 0)}, "
        f"Negative={np.sum(labels == 1)}, "
        f"Neutral={np.sum(labels == 2)}, "
        f"Positive={np.sum(labels == 3)}, "
        f"Strong Positive={np.sum(labels == 4)}"
    )

    return labels
