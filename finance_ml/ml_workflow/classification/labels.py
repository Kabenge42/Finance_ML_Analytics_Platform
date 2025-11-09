"""
finance_ml.ml_workflow.classification.labels - Event label creation for classification

This module provides sophisticated event classification label creation methods:
- price_momentum: Based on price target vs current price
- valuation: Based on valuation metric percentiles (P/E, P/B)
- fundamental: Based on margin expansion/contraction
- volatility: Based on price volatility spikes
- analyst_rating: Based on analyst upgrades/downgrades
- market_events: Based on sector rotation and regional trends

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
    1. price_momentum: Based on price target vs current price
    2. valuation: Based on valuation metric percentiles (P/E, P/B)
    3. fundamental: Based on margin expansion/contraction
    4. volatility: Based on price volatility spikes
    5. analyst_rating: Based on analyst rating changes
    6. market_events: Based on sector rotation and regional trends

    Args:
        df: DataFrame with required columns
        method: Event detection method ('price_momentum', 'valuation', 'fundamental', 'volatility',
                'analyst_rating', 'market_events')
        threshold_positive: Threshold for positive catalyst (%)
        threshold_negative: Threshold for negative catalyst (%)
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

        >>> # Valuation method with sector adjustment
        >>> labels = create_enhanced_event_labels(
        ...     df, method="valuation",
        ...     use_sector_adjustment=True
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

    else:
        logger.error(f"Unknown method: {method}")

    logger.info(
        f"Created labels with method={method}: Neutral={np.sum(labels == 0)}, "
        f"Positive={np.sum(labels == 1)}, Negative={np.sum(labels == 2)}"
    )

    return labels
