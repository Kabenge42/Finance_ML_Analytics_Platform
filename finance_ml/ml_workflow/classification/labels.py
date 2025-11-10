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
        numpy array of labels (0=Neutral, 1=Positive, 2=Negative)

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
        # Price target momentum
        if "price_target" not in df.columns or "last_price" not in df.columns:
            logger.warning("price_target or last_price not available, returning all neutral")
            return labels

        price_diff_pct = (df["price_target"] - df["last_price"]) / df["last_price"] * 100.0

        # Sector-specific adjustment
        if use_sector_adjustment and "sector" in df.columns:
            for sector in df["sector"].unique():
                sector_mask = df["sector"] == sector
                sector_vol = price_diff_pct[sector_mask].std()
                # Adjust thresholds based on sector volatility
                adj_positive = threshold_positive * (1 + sector_vol / 50.0)
                adj_negative = threshold_negative * (1 + sector_vol / 50.0)

                labels[sector_mask & (price_diff_pct >= adj_positive)] = 1
                labels[sector_mask & (price_diff_pct <= adj_negative)] = 2
        else:
            labels[price_diff_pct >= threshold_positive] = 1
            labels[price_diff_pct <= threshold_negative] = 2

    elif method == "valuation":
        # Valuation-based events (undervalued = positive, overvalued = negative)
        if "p_e" not in df.columns:
            logger.warning("p_e not available for valuation method, returning all neutral")
            return labels

        # Calculate percentiles within sector
        if "sector" in df.columns:
            df["p_e_percentile"] = df.groupby("sector")["p_e"].rank(pct=True)
        else:
            df["p_e_percentile"] = df["p_e"].rank(pct=True)

        # Low P/E (undervalued) = positive, High P/E (overvalued) = negative
        labels[df["p_e_percentile"] <= 0.25] = 1  # Bottom quartile = positive
        labels[df["p_e_percentile"] >= 0.75] = 2  # Top quartile = negative

    elif method == "fundamental":
        # Fundamental events based on margin trends
        margin_cols = [
            c for c in ["gross_margin", "operating_margin", "net_margin"] if c in df.columns
        ]
        if not margin_cols:
            logger.warning("No margin columns available, returning all neutral")
            return labels

        # Calculate average margin score
        margin_data = df[margin_cols].fillna(0)
        avg_margin = margin_data.mean(axis=1)

        # High margins = positive, low margins = negative
        labels[avg_margin >= avg_margin.quantile(0.7)] = 1
        labels[avg_margin <= avg_margin.quantile(0.3)] = 2

    elif method == "volatility":
        # Volatility-based events
        vol_cols = [c for c in df.columns if "volatility" in c.lower()]
        if not vol_cols:
            logger.warning("No volatility columns available, returning all neutral")
            return labels

        volatility = df[vol_cols[0]]
        # High volatility = negative (risk), low volatility = positive (stable)
        labels[volatility <= volatility.quantile(0.3)] = 1
        labels[volatility >= volatility.quantile(0.7)] = 2

    elif method == "analyst_rating":
        # Analyst rating changes (upgrades/downgrades)
        if "analyst_rating_change" in df.columns:
            rating_change = df["analyst_rating_change"]
            # Positive changes = upgrades (positive catalyst)
            labels[rating_change >= 0.5] = 1
            labels[rating_change <= -0.5] = 2
        elif "analyst_rating" in df.columns:
            # If only rating available, use current rating
            rating_map = {
                "Buy": 1,
                "Strong Buy": 1,
                "Outperform": 1,
                "Sell": 2,
                "Strong Sell": 2,
                "Underperform": 2,
                "Hold": 0,
                "Neutral": 0,
            }
            for idx, rating in enumerate(df["analyst_rating"]):
                if rating in rating_map:
                    labels[idx] = rating_map[rating]
        else:
            logger.warning("No analyst rating columns available, returning all neutral")
            return labels

    elif method == "market_events":
        # Market events (sector rotation, regional trends)
        if "sector_momentum" in df.columns:
            sector_mom = df["sector_momentum"]
            # High sector momentum = positive, low = negative
            labels[sector_mom >= sector_mom.quantile(0.7)] = 1
            labels[sector_mom <= sector_mom.quantile(0.3)] = 2
        elif "sector" in df.columns and "last_price" in df.columns:
            # Calculate sector-relative performance
            sector_perf = df.groupby("sector")["last_price"].transform(lambda x: x / x.mean())
            labels[sector_perf >= 1.1] = 1  # Outperforming sector
            labels[sector_perf <= 0.9] = 2  # Underperforming sector
        else:
            logger.warning("No market event indicators available, returning all neutral")
            return labels

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
            # High profitability = positive (top 30%), low/negative = negative (bottom 30%)
            labels[avg_profitability >= avg_profitability.quantile(0.7)] = 1
            labels[avg_profitability <= avg_profitability.quantile(0.3)] = 2
        else:
            logger.warning("No variance in profitability data, returning all neutral")
            return labels

    elif method == "leverage_event":
        # Leverage-based events using debt ratios
        leverage_cols = [c for c in ["debt_to_equity", "net_debt_to_ebitda", "debt_to_assets"] if c in df.columns]
        if not leverage_cols:
            logger.warning("No leverage columns available, returning all neutral")
            return labels
        
        # Calculate average leverage score (lower is better)
        leverage_data = df[leverage_cols].fillna(df[leverage_cols].median())
        avg_leverage = leverage_data.mean(axis=1)
        
        # Low leverage = positive (bottom 30%), high leverage = negative (top 30%)
        labels[avg_leverage <= avg_leverage.quantile(0.3)] = 1
        labels[avg_leverage >= avg_leverage.quantile(0.7)] = 2

    elif method == "liquidity_event":
        # Liquidity-based events using current ratio, quick ratio
        liquidity_cols = [c for c in ["current_ratio", "quick_ratio", "cash_ratio"] if c in df.columns]
        if not liquidity_cols:
            logger.warning("No liquidity columns available, returning all neutral")
            return labels
        
        # Calculate average liquidity score
        liquidity_data = df[liquidity_cols].fillna(0)
        avg_liquidity = liquidity_data.mean(axis=1)
        
        # Check if there's any variance in the data
        if avg_liquidity.std() > 1e-10:
            # High liquidity = positive (top 30%), low liquidity = negative (bottom 30%)
            labels[avg_liquidity >= avg_liquidity.quantile(0.7)] = 1
            labels[avg_liquidity <= avg_liquidity.quantile(0.3)] = 2
        else:
            logger.warning("No variance in liquidity data, returning all neutral")
            return labels

    elif method == "efficiency_event":
        # Efficiency-based events using turnover ratios
        efficiency_cols = [c for c in ["asset_turnover", "inventory_turnover", "receivables_turnover"] if c in df.columns]
        if not efficiency_cols:
            logger.warning("No efficiency columns available, returning all neutral")
            return labels
        
        # Calculate average efficiency score
        efficiency_data = df[efficiency_cols].fillna(df[efficiency_cols].median())
        avg_efficiency = efficiency_data.mean(axis=1)
        
        # High efficiency = positive (top 30%), low efficiency = negative (bottom 30%)
        labels[avg_efficiency >= avg_efficiency.quantile(0.7)] = 1
        labels[avg_efficiency <= avg_efficiency.quantile(0.3)] = 2

    elif method == "growth_event":
        # Growth-based events using revenue and earnings growth
        growth_cols = [c for c in ["revenue_growth", "earnings_growth", "ebitda_growth"] if c in df.columns]
        if not growth_cols:
            logger.warning("No growth columns available, returning all neutral")
            return labels
        
        # Calculate average growth score
        growth_data = df[growth_cols].fillna(0)
        avg_growth = growth_data.mean(axis=1)
        
        # Check if there's any variance in the data
        if avg_growth.std() > 1e-10:
            # High growth = positive (top 30%), negative growth = negative (bottom 30%)
            labels[avg_growth >= avg_growth.quantile(0.7)] = 1
            labels[avg_growth <= avg_growth.quantile(0.3)] = 2
        else:
            logger.warning("No variance in growth data, returning all neutral")
            return labels

    elif method == "quality_event":
        # Quality-based events using accounting and analyst quality metrics
        quality_cols = []
        
        # Accounting quality metrics (lower is better for accruals, DSO)
        inverse_quality_cols = [c for c in ["accruals_to_assets", "days_sales_outstanding"] if c in df.columns]
        # Analyst quality metrics (higher is better)
        direct_quality_cols = [c for c in ["analyst_consensus_score", "analyst_revision_score"] if c in df.columns]
        
        if not inverse_quality_cols and not direct_quality_cols:
            logger.warning("No quality columns available, returning all neutral")
            return labels
        
        # Create composite quality score
        quality_score = pd.Series(0.0, index=df.index)
        
        if inverse_quality_cols:
            # Normalize inverse metrics (lower is better -> invert)
            for col in inverse_quality_cols:
                col_data = df[col].fillna(df[col].median())
                # Invert: subtract from max and normalize
                quality_score += (col_data.max() - col_data) / (col_data.max() - col_data.min() + 1e-10)
        
        if direct_quality_cols:
            # Add direct metrics (higher is better)
            for col in direct_quality_cols:
                col_data = df[col].fillna(df[col].median())
                quality_score += col_data
        
        # Normalize by number of metrics
        quality_score /= len(inverse_quality_cols) + len(direct_quality_cols)
        
        # High quality = positive (top 30%), low quality = negative (bottom 30%)
        labels[quality_score >= quality_score.quantile(0.7)] = 1
        labels[quality_score <= quality_score.quantile(0.3)] = 2

    elif method == "composite_event":
        # Composite events using Piotroski F-Score and Altman Z-Score
        composite_cols = [c for c in ["piotroski_f_score", "altman_z_score", "beneish_m_score"] if c in df.columns]
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
        composite_score = composite_score / len(composite_cols)
        
        # High composite score = positive (top 30%), low = negative (bottom 30%)
        labels[composite_score >= composite_score.quantile(0.7)] = 1
        labels[composite_score <= composite_score.quantile(0.3)] = 2

    else:
        logger.error(f"Unknown method: {method}")

    logger.info(
        f"Created labels with method={method}: Neutral={np.sum(labels == 0)}, "
        f"Positive={np.sum(labels == 1)}, Negative={np.sum(labels == 2)}"
    )

    return labels
