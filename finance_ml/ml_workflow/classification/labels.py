"""
finance_ml.ml_workflow.classification.labels - Event label creation for classification

This module provides sophisticated event classification label creation methods:

Original methods (Phase 9.4):
- price_momentum: Based on price target vs current price
- valuation: Based on valuation metric percentiles (P/E, P/B)
- fundamental: Based on margin expansion/contraction
- volatility: Based on price volatility spikes
- analyst_rating: Based on analyst upgrades/downgrades
- market_events: Based on sector rotation and regional trends

New methods (Phase 9.4 - enhanced with Phase 9.3 features):
- profitability_event: Based on ROE, ROA, ROIC profitability ratios
- leverage_event: Based on debt ratios and leverage metrics
- liquidity_event: Based on current ratio, quick ratio
- efficiency_event: Based on asset turnover, inventory turnover
- growth_event: Based on revenue growth, earnings growth
- quality_event: Based on accounting quality and analyst quality
- composite_event: Based on Piotroski F-Score, Altman Z-Score

Phase 9.4 refactor: Extracted from classification.py for better modularity.

Phase 9.6 enhancement: Upgraded from 3-class to 5-class label granularity:
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
    """Create sophisticated event classification labels.

    Multiple methods for event detection:

    Original methods:
    1. price_momentum: Based on price target vs current price
    2. valuation: Based on valuation metric percentiles (P/E, P/B)
    3. fundamental: Based on margin expansion/contraction
    4. volatility: Based on price volatility spikes
    5. analyst_rating: Based on analyst rating changes
    6. market_events: Based on sector rotation and regional trends

    New Phase 9.3 feature-enhanced methods:
    7. profitability_event: Based on ROE, ROA, ROIC profitability ratios
    8. leverage_event: Based on debt ratios (debt_to_equity, net_debt_to_ebitda)
    9. liquidity_event: Based on current_ratio, quick_ratio
    10. efficiency_event: Based on asset_turnover, inventory_turnover
    11. growth_event: Based on revenue_growth, earnings_growth
    12. quality_event: Based on accounting quality and analyst quality metrics
    13. composite_event: Based on Piotroski F-Score, Altman Z-Score

    Args:
        df: DataFrame with required columns
        method: Event detection method (see list above)
        threshold_positive: Threshold for positive catalyst (%) - used by some methods
        threshold_negative: Threshold for negative catalyst (%) - used by some methods
        use_sector_adjustment: If True, adjust thresholds by sector volatility

    Returns:
        numpy array of labels (0=Strong Negative, 1=Negative, 2=Neutral, 3=Positive, 4=Strong Positive)

    Examples:
        >>> # Price momentum method
        >>> labels = create_enhanced_event_labels(
        ...     df, method="price_momentum",
        ...     threshold_positive=10.0,
        ...     threshold_negative=-10.0
        ... )

        >>> # Profitability event method
        >>> labels = create_enhanced_event_labels(
        ...     df, method="profitability_event"
        ... )

        >>> # Composite event method
        >>> labels = create_enhanced_event_labels(
        ...     df, method="composite_event"
        ... )
    """
    labels = np.zeros(len(df), dtype=int)

    if method == "price_momentum":
        # Enhanced price momentum using Phase 9.3 features when available
        # Primary signal: price_target vs last_price
        # Secondary signals: price_momentum_1m/3m/6m (Phase 9.3), rsi_14d, ma_crossover_signal

        # Try price target momentum first (backward compatible)
        momentum_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        if "price_target" in df.columns and "last_price" in df.columns:
            price_diff_pct = (df["price_target"] - df["last_price"]) / df["last_price"] * 100.0
            momentum_score += price_diff_pct / 10.0  # Normalize to ~-10 to +10 range
            signal_count += 1

        # Add Phase 9.3 momentum features if available
        momentum_1m = _get_column(df, "price_momentum_1m")
        if momentum_1m is not None:
            momentum_score += momentum_1m / 10.0
            signal_count += 1

        momentum_3m = _get_column(df, "price_momentum_3m")
        if momentum_3m is not None:
            momentum_score += momentum_3m / 20.0  # Slightly less weight for 3m
            signal_count += 1

        # RSI signal (normalized: >70 bullish, <30 bearish)
        rsi = _get_column(df, "rsi_14d", "rsi_30d")
        if rsi is not None:
            rsi_signal = (rsi - 50) / 10.0  # Normalize to roughly -5 to +5
            momentum_score += rsi_signal
            signal_count += 1

        # MA crossover signal (-1, 0, +1)
        ma_signal = _get_column(df, "ma_crossover_signal")
        if ma_signal is not None:
            momentum_score += ma_signal * 3.0  # Weight crossover signals
            signal_count += 1

        # Return stability score (higher = more stable positive returns)
        stability = _get_column(df, "return_stability_score")
        if stability is not None:
            momentum_score += stability.clip(-2, 2)  # Clip outliers
            signal_count += 1

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
            # Use composite momentum score with 5-class thresholds
            labels[momentum_score >= 1.5] = 4  # Strong positive momentum
            labels[(momentum_score >= 0.75) & (momentum_score < 1.5)] = 3  # Positive momentum
            labels[(momentum_score <= -0.75) & (momentum_score > -1.5)] = 1  # Negative momentum
            labels[momentum_score <= -1.5] = 0  # Strong negative momentum

    elif method == "valuation":
        # Enhanced valuation-based events using Phase 9.3 features when available
        # Uses multiple valuation metrics: P/E, P/B, EV/EBITDA, PEG
        # Lower values = undervalued (positive), higher values = overvalued (negative)

        valuation_metrics = []

        # P/E ratio (try Phase 9.3 first, fall back to original)
        pe_col = _get_column(df, "p_e_ratio", "p_e")
        if pe_col is not None:
            valuation_metrics.append(("p_e", pe_col))

        # P/B ratio (Phase 9.3)
        pb_col = _get_column(df, "p_b_ratio", "p_b")
        if pb_col is not None:
            valuation_metrics.append(("p_b", pb_col))

        # EV/EBITDA ratio (Phase 9.3)
        ev_ebitda_col = _get_column(df, "ev_ebitda_ratio", "ev_ebitda")
        if ev_ebitda_col is not None:
            valuation_metrics.append(("ev_ebitda", ev_ebitda_col))

        # PEG ratio (Phase 9.3) - Price/Earnings to Growth
        peg_col = _get_column(df, "peg_ratio")
        if peg_col is not None:
            valuation_metrics.append(("peg", peg_col))

        if not valuation_metrics:
            logger.warning("No valuation metrics available, returning all neutral")
            return labels

        # Calculate percentile for each metric and average
        valuation_score = pd.Series(0.0, index=df.index)

        for metric_name, metric_col in valuation_metrics:
            # Calculate percentiles within sector if available
            if "sector" in df.columns:
                percentile = df.groupby("sector")[metric_col.name].rank(pct=True)
            else:
                percentile = metric_col.rank(pct=True)

            # Invert: low percentile = undervalued (good)
            valuation_score += 1.0 - percentile

        # Average across available metrics
        valuation_score /= len(valuation_metrics)

        # High score (undervalued) = positive, Low score (overvalued) = negative (5-class)
        labels[valuation_score >= 0.85] = 4  # Top 15% = strongly undervalued = strong positive
        labels[(valuation_score >= 0.65) & (valuation_score < 0.85)] = 3  # undervalued = positive
        labels[(valuation_score <= 0.35) & (valuation_score > 0.15)] = 1  # overvalued = negative
        labels[valuation_score <= 0.15] = 0  # Bottom 15% = strongly overvalued = strong negative

    elif method == "fundamental":
        # Enhanced fundamental events using Phase 9.3 margin and profitability features
        # Uses margin metrics, profitability ratios (ROE, ROA, ROIC), and margin trends

        fundamental_metrics = []

        # Margin metrics (try Phase 9.3 _pct variants first, fall back to originals)
        gross_margin = _get_column(df, "gross_margin_pct", "gross_margin")
        if gross_margin is not None:
            fundamental_metrics.append(gross_margin)

        operating_margin = _get_column(df, "operating_margin_pct", "operating_margin")
        if operating_margin is not None:
            fundamental_metrics.append(operating_margin)

        net_margin = _get_column(df, "net_margin_pct", "net_margin")
        if net_margin is not None:
            fundamental_metrics.append(net_margin)

        # Profitability ratios (Phase 9.3)
        roe = _get_column(df, "roe")
        if roe is not None:
            fundamental_metrics.append(roe)

        roa = _get_column(df, "roa")
        if roa is not None:
            fundamental_metrics.append(roa)

        roic = _get_column(df, "roic")
        if roic is not None:
            fundamental_metrics.append(roic)

        # Margin trends (Phase 9.3)
        ebitda_margin_trend = _get_column(df, "ebitda_margin_trend")
        if ebitda_margin_trend is not None:
            fundamental_metrics.append(ebitda_margin_trend * 10)  # Scale trend

        if not fundamental_metrics:
            logger.warning("No fundamental metrics available, returning all neutral")
            return labels

        # Calculate average fundamental score
        fundamental_data = pd.concat(fundamental_metrics, axis=1).fillna(0)
        avg_fundamental = fundamental_data.mean(axis=1)

        # High fundamentals = positive, low fundamentals = negative (5-class)
        labels[avg_fundamental >= avg_fundamental.quantile(0.85)] = 4  # Top 15% = strong positive
        labels[
            (avg_fundamental >= avg_fundamental.quantile(0.65))
            & (avg_fundamental < avg_fundamental.quantile(0.85))
        ] = 3  # Positive
        labels[
            (avg_fundamental <= avg_fundamental.quantile(0.35))
            & (avg_fundamental > avg_fundamental.quantile(0.15))
        ] = 1  # Negative
        labels[avg_fundamental <= avg_fundamental.quantile(0.15)] = (
            0  # Bottom 15% = strong negative
        )

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
        labels[volatility_score <= -1.0] = 4  # Very low volatility/high stability = strong positive
        labels[(volatility_score <= -0.5) & (volatility_score > -1.0)] = (
            3  # Low volatility = positive
        )
        labels[(volatility_score >= 0.5) & (volatility_score < 1.0)] = (
            1  # High volatility = negative
        )
        labels[volatility_score >= 1.0] = 0  # Very high volatility/low stability = strong negative

    elif method == "analyst_rating":
        # Enhanced analyst rating events using Phase 9.3 analyst quality features
        # Uses rating changes, consensus, upside potential, coverage quality

        analyst_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Analyst rating change (original data)
        if "analyst_rating_change" in df.columns:
            rating_change = df["analyst_rating_change"]
            analyst_score += rating_change * 2.0  # Weight changes heavily
            signal_count += 1
        elif "analyst_rating" in df.columns:
            # Map rating to numeric score
            rating_map = {
                "Strong Buy": 2.0,
                "Buy": 1.0,
                "Outperform": 1.0,
                "Hold": 0.0,
                "Neutral": 0.0,
                "Sell": -1.0,
                "Strong Sell": -2.0,
                "Underperform": -1.0,
            }
            rating_score = df["analyst_rating"].map(rating_map).fillna(0)
            analyst_score += rating_score
            signal_count += 1

        # Upside potential (Phase 9.3): (target - price) / price
        upside = _get_column(df, "upside_potential")
        if upside is not None:
            # Normalize: high upside = positive
            upside_normalized = upside.clip(-50, 50) / 25.0  # Scale to roughly -2 to +2
            analyst_score += upside_normalized
            signal_count += 1

        # Analyst bullish percentage (Phase 9.3)
        bullish_pct = _get_column(df, "analyst_bullish_pct")
        if bullish_pct is not None:
            # Convert to -1 to +1 scale (50% = neutral)
            analyst_score += (bullish_pct - 50) / 25.0
            signal_count += 1

        # Analyst coverage quality (Phase 9.3)
        coverage_quality = _get_column(df, "analyst_coverage_quality")
        if coverage_quality is not None:
            # High coverage quality = more reliable signal (amplifier, not direction)
            # Use as confidence weight (0.5 to 1.5 range)
            quality_weight = 0.5 + coverage_quality.clip(0, 2) / 2.0
            analyst_score *= quality_weight

        if signal_count == 0:
            logger.warning("No analyst rating indicators available, returning all neutral")
            return labels

        # Average across available signals
        analyst_score /= signal_count

        # Positive analyst score = positive catalyst, negative = negative catalyst (5-class)
        labels[analyst_score >= 1.0] = 4  # Very strong bullish signals
        labels[(analyst_score >= 0.5) & (analyst_score < 1.0)] = 3  # Bullish signals
        labels[(analyst_score <= -0.5) & (analyst_score > -1.0)] = 1  # Bearish signals
        labels[analyst_score <= -1.0] = 0  # Very strong bearish signals

    elif method == "market_events":
        # Enhanced market events using Phase 9.3 sector and sentiment features
        # Uses sector rotation, regional trends, sentiment indicators

        market_score = pd.Series(0.0, index=df.index)
        signal_count = 0

        # Sector-relative performance (calculated from price)
        if "sector" in df.columns and "last_price" in df.columns:
            sector_perf = df.groupby("sector")["last_price"].transform(
                lambda x: (x / x.mean() - 1.0) * 100
            )
            market_score += sector_perf / 10.0  # Normalize to similar scale
            signal_count += 1

        # Short interest ratio (Phase 9.3 sentiment indicator)
        short_interest = _get_column(df, "short_interest_ratio", "short_int_pct")
        if short_interest is not None:
            # High short interest = bearish sentiment (negative)
            # Normalize and invert
            short_normalized = -(short_interest - short_interest.median()) / (
                short_interest.std() + 1e-10
            )
            market_score += short_normalized
            signal_count += 1

        # Beta trend (Phase 9.3): systematic risk changes
        beta_trend = _get_column(df, "systematic_risk_trend")
        if beta_trend is not None:
            # Increasing beta = increasing risk (negative)
            market_score += -beta_trend * 2.0
            signal_count += 1

        # Sector-relative valuation (Phase 9.3: vs_sector_median features)
        for metric in ["p_e_ratio_vs_sector_median", "ev_ebitda_ratio_vs_sector_median"]:
            sector_rel = _get_column(df, metric)
            if sector_rel is not None:
                # Negative relative = undervalued = positive
                market_score += -sector_rel
                signal_count += 1
                break  # Use first available

        if signal_count == 0:
            logger.warning("No market event indicators available, returning all neutral")
            return labels

        # Average across available signals
        market_score /= signal_count

        # Positive market signals = positive, negative = negative (5-class)
        labels[market_score >= 1.2] = 4  # Very strong positive sector/market signals
        labels[(market_score >= 0.6) & (market_score < 1.2)] = 3  # Positive signals
        labels[(market_score <= -0.6) & (market_score > -1.2)] = 1  # Negative signals
        labels[market_score <= -1.2] = 0  # Very strong negative sector/market signals

    elif method == "profitability_event":
        # Profitability-based events using ROE, ROA, ROIC
        profitability_cols = [c for c in ["roe", "roa", "roic"] if c in df.columns]
        if not profitability_cols:
            logger.warning("No profitability columns available, returning all neutral")
            return labels

        # Calculate average profitability score
        prof_data = df[profitability_cols].fillna(0)
        avg_profitability = prof_data.mean(axis=1)

        # Check if there's any variance in the data
        if avg_profitability.std() > 1e-10:
            # High profitability = positive, low/negative = negative (5-class)
            labels[avg_profitability >= avg_profitability.quantile(0.85)] = (
                4  # Top 15% = strong positive
            )
            labels[
                (avg_profitability >= avg_profitability.quantile(0.65))
                & (avg_profitability < avg_profitability.quantile(0.85))
            ] = 3  # Positive
            labels[
                (avg_profitability <= avg_profitability.quantile(0.35))
                & (avg_profitability > avg_profitability.quantile(0.15))
            ] = 1  # Negative
            labels[avg_profitability <= avg_profitability.quantile(0.15)] = (
                0  # Bottom 15% = strong negative
            )
        else:
            logger.warning("No variance in profitability data, returning all neutral")
            return labels

    elif method == "leverage_event":
        # Leverage-based events using debt ratios
        leverage_cols = [
            c for c in ["debt_to_equity", "net_debt_to_ebitda", "debt_to_assets"] if c in df.columns
        ]
        if not leverage_cols:
            logger.warning("No leverage columns available, returning all neutral")
            return labels

        # Calculate average leverage score (lower is better)
        leverage_data = df[leverage_cols].fillna(df[leverage_cols].median())
        avg_leverage = leverage_data.mean(axis=1)

        # Low leverage = positive, high leverage = negative (5-class, inverted)
        labels[avg_leverage <= avg_leverage.quantile(0.15)] = (
            4  # Bottom 15% = very low leverage = strong positive
        )
        labels[
            (avg_leverage <= avg_leverage.quantile(0.35))
            & (avg_leverage > avg_leverage.quantile(0.15))
        ] = 3  # Low leverage = positive
        labels[
            (avg_leverage >= avg_leverage.quantile(0.65))
            & (avg_leverage < avg_leverage.quantile(0.85))
        ] = 1  # High leverage = negative
        labels[avg_leverage >= avg_leverage.quantile(0.85)] = (
            0  # Top 15% = very high leverage = strong negative
        )

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
        # Efficiency-based events using turnover ratios
        efficiency_cols = [
            c
            for c in ["asset_turnover", "inventory_turnover", "receivables_turnover"]
            if c in df.columns
        ]
        if not efficiency_cols:
            logger.warning("No efficiency columns available, returning all neutral")
            return labels

        # Calculate average efficiency score
        efficiency_data = df[efficiency_cols].fillna(df[efficiency_cols].median())
        avg_efficiency = efficiency_data.mean(axis=1)

        # High efficiency = positive, low efficiency = negative (5-class)
        labels[avg_efficiency >= avg_efficiency.quantile(0.85)] = 4  # Top 15% = strong positive
        labels[
            (avg_efficiency >= avg_efficiency.quantile(0.65))
            & (avg_efficiency < avg_efficiency.quantile(0.85))
        ] = 3  # Positive
        labels[
            (avg_efficiency <= avg_efficiency.quantile(0.35))
            & (avg_efficiency > avg_efficiency.quantile(0.15))
        ] = 1  # Negative
        labels[avg_efficiency <= avg_efficiency.quantile(0.15)] = 0  # Bottom 15% = strong negative

    elif method == "growth_event":
        # Growth-based events using revenue and earnings growth
        growth_cols = [
            c for c in ["revenue_growth", "earnings_growth", "ebitda_growth"] if c in df.columns
        ]
        if not growth_cols:
            logger.warning("No growth columns available, returning all neutral")
            return labels

        # Calculate average growth score
        growth_data = df[growth_cols].fillna(0)
        avg_growth = growth_data.mean(axis=1)

        # Check if there's any variance in the data
        if avg_growth.std() > 1e-10:
            # High growth = positive, negative growth = negative (5-class)
            labels[avg_growth >= avg_growth.quantile(0.85)] = 4  # Top 15% = strong positive
            labels[
                (avg_growth >= avg_growth.quantile(0.65)) & (avg_growth < avg_growth.quantile(0.85))
            ] = 3  # Positive
            labels[
                (avg_growth <= avg_growth.quantile(0.35)) & (avg_growth > avg_growth.quantile(0.15))
            ] = 1  # Negative
            labels[avg_growth <= avg_growth.quantile(0.15)] = 0  # Bottom 15% = strong negative
        else:
            logger.warning("No variance in growth data, returning all neutral")
            return labels

    elif method == "quality_event":
        # Quality-based events using accounting and analyst quality metrics
        # Updated to use Phase 9.3 generated columns: accounting_quality_score, analyst_coverage_quality
        quality_cols = []

        # Accounting quality score (higher is better - generated by engineer_accounting_quality_features)
        direct_quality_cols = [
            c
            for c in [
                "accounting_quality_score",
                "analyst_coverage_quality",
                "analyst_quality_score",
            ]
            if c in df.columns
        ]
        # Exceptional items and red flags (lower is better)
        inverse_quality_cols = [
            c
            for c in [
                "exceptional_items_to_ebitda",
                "total_exceptional_items_ltm",
                "goodwill_impairment_flag",
                "has_goodwill_impairment",
                "has_asset_writedown",
                "has_restructuring",
            ]
            if c in df.columns
        ]

        if not direct_quality_cols and not inverse_quality_cols:
            logger.warning("No quality columns available, returning all neutral")
            return labels

        # Create composite quality score
        quality_score = pd.Series(0.0, index=df.index)

        if direct_quality_cols:
            # Add direct metrics (higher is better)
            for col in direct_quality_cols:
                col_data = df[col].fillna(df[col].median())
                # Normalize to 0-1 range
                col_min, col_max = col_data.min(), col_data.max()
                if col_max - col_min > 1e-10:
                    quality_score += (col_data - col_min) / (col_max - col_min + 1e-10)

        if inverse_quality_cols:
            # Normalize inverse metrics (lower is better -> invert)
            for col in inverse_quality_cols:
                col_data = df[col].fillna(df[col].median())
                col_min, col_max = col_data.min(), col_data.max()
                if col_max - col_min > 1e-10:
                    # Invert: subtract from max and normalize
                    quality_score += (col_max - col_data) / (col_max - col_min + 1e-10)

        # Normalize by number of metrics
        total_metrics = len(direct_quality_cols) + len(inverse_quality_cols)
        if total_metrics > 0:
            quality_score /= total_metrics

        # High quality = positive, low quality = negative (5-class)
        labels[quality_score >= quality_score.quantile(0.85)] = 4  # Top 15% = strong positive
        labels[
            (quality_score >= quality_score.quantile(0.65))
            & (quality_score < quality_score.quantile(0.85))
        ] = 3  # Positive
        labels[
            (quality_score <= quality_score.quantile(0.35))
            & (quality_score > quality_score.quantile(0.15))
        ] = 1  # Negative
        labels[quality_score <= quality_score.quantile(0.15)] = 0  # Bottom 15% = strong negative

    elif method == "composite_event":
        # Composite events using Piotroski F-Score and Altman Z-Score
        composite_cols = [
            c for c in ["piotroski_f_score", "altman_z_score", "beneish_m_score"] if c in df.columns
        ]
        if not composite_cols:
            logger.warning("No composite score columns available, returning all neutral")
            return labels

        # Create composite score with proper normalization
        composite_score = pd.Series(0.0, index=df.index)

        if "piotroski_f_score" in df.columns:
            # Piotroski F-Score: 0-9 scale, higher is better
            f_score = df["piotroski_f_score"].fillna(df["piotroski_f_score"].median())
            composite_score += f_score / 9.0  # Normalize to 0-1

        if "altman_z_score" in df.columns:
            # Altman Z-Score: >2.99 safe, 1.81-2.99 grey, <1.81 distress
            z_score = df["altman_z_score"].fillna(df["altman_z_score"].median())
            # Normalize: clip at 0 and 5, then scale to 0-1
            composite_score += z_score.clip(0, 5) / 5.0

        if "beneish_m_score" in df.columns:
            # Beneish M-Score: <-1.78 unlikely manipulator, >-1.78 possible manipulator
            m_score = df["beneish_m_score"].fillna(df["beneish_m_score"].median())
            # Invert: lower is better, clip at -3 to 1, then normalize
            composite_score += (1.0 - m_score.clip(-3, 1)) / 4.0

        # Normalize by number of scores
        composite_score /= len(composite_cols)

        # High composite score = positive, low = negative (5-class)
        labels[composite_score >= composite_score.quantile(0.85)] = 4  # Top 15% = strong positive
        labels[
            (composite_score >= composite_score.quantile(0.65))
            & (composite_score < composite_score.quantile(0.85))
        ] = 3  # Positive
        labels[
            (composite_score <= composite_score.quantile(0.35))
            & (composite_score > composite_score.quantile(0.15))
        ] = 1  # Negative
        labels[composite_score <= composite_score.quantile(0.15)] = (
            0  # Bottom 15% = strong negative
        )

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
