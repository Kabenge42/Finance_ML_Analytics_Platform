"""Sector-specific feature engineering."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .utils import _safe_div, _ensure_float_column

logger = logging.getLogger(__name__)


def engineer_sector_specific_features(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """Add sector-specific engineered features."""
    result = df.copy()

    # Basic sector masks
    financials_mask = result[sector_col] == "Financials"
    energy_mask = result[sector_col].str.contains("Energy|Materials", case=False, na=False)
    tech_mask = result[sector_col].str.contains("Technology|Information", case=False, na=False)
    health_mask = result[sector_col].str.contains("Health", case=False, na=False)
    consumer_mask = result[sector_col].str.contains("Consumer", case=False, na=False)
    industrials_mask = result[sector_col].str.contains("Industrial", case=False, na=False)
    utilities_mask = result[sector_col].str.contains("Utilities", case=False, na=False)

    if financials_mask.any():
        # Tangible Book Value - use direct column if available, else calculate
        tbv = None
        if "tbv_ltm" in df.columns:
            tbv = pd.to_numeric(df["tbv_ltm"], errors="coerce")
        elif "tbv_fy" in df.columns:
            tbv = pd.to_numeric(df["tbv_fy"], errors="coerce")

        if tbv is None:
            # Fallback to calculation
            if "total_equity" in df.columns and "intangible_assets" in df.columns:
                total_equity = pd.to_numeric(df["total_equity"], errors="coerce")
                intangible_assets = pd.to_numeric(df["intangible_assets"], errors="coerce").fillna(
                    0
                )
                tbv = total_equity - intangible_assets

        if tbv is not None:
            result = _ensure_float_column(result, "tangible_book_value")
            result.loc[financials_mask, "tangible_book_value"] = tbv.loc[financials_mask]

        # Price to Tangible Book Value
        if (
            "last_price" in df.columns
            and "shares_outstanding" in df.columns
            and "tangible_book_value" in result.columns
        ):
            market_cap = df["last_price"] * df["shares_outstanding"]
            result = _ensure_float_column(result, "p_tbv_ratio")
            result.loc[financials_mask, "p_tbv_ratio"] = _safe_div(
                market_cap, result["tangible_book_value"]
            ).loc[financials_mask]

    if energy_mask.any():
        if "capex" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "capex_intensity")
            result.loc[energy_mask, "capex_intensity"] = (
                _safe_div(df["capex"], df["revenue"]) * 100
            ).loc[energy_mask]

    # Size Factor Percentile (using Country Rank)
    if "market_cap_country_r" in df.columns:
        if "country" in df.columns:
            max_rank = df.groupby("country")["market_cap_country_r"].transform("max")
            result["size_factor_percentile"] = _safe_div(df["market_cap_country_r"], max_rank) * 100
        else:
            # Global normalization
            max_rank = df["market_cap_country_r"].max()
            # scalar division is safe if max_rank != 0. safe_div expects Series.
            # We can just divide directly as max_rank is scalar.
            if max_rank != 0 and not pd.isna(max_rank):
                result["size_factor_percentile"] = (df["market_cap_country_r"] / max_rank) * 100
            else:
                result["size_factor_percentile"] = np.nan

    # ... other sector specific logic would go here

    logger.info("Engineered sector-specific features")
    return result


def create_relative_value_features(
    df: pd.DataFrame, sector_col: str = "sector", metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create features relative to sector medians/means."""
    if metrics is None:
        metrics = ["p_e_ratio", "p_b_ratio", "ev_ebitda_ratio", "roe", "roa"]

    result = df.copy()
    valid_metrics = [m for m in metrics if m in result.columns]

    if not valid_metrics:
        return result

    grouped = result.groupby(sector_col)

    for metric in valid_metrics:
        sector_median = grouped[metric].transform("median")
        sector_mean = grouped[metric].transform("mean")
        sector_std = grouped[metric].transform("std")

        result[f"{metric}_vs_sector_median"] = df[metric] - sector_median
        result[f"{metric}_sector_zscore"] = _safe_div(df[metric] - sector_mean, sector_std)
        result[f"{metric}_sector_percentile"] = grouped[metric].rank(pct=True) * 100

    logger.info(f"Created relative value features for {len(valid_metrics)} metrics")
    return result


def engineer_sector_relative_interactions(
    df: pd.DataFrame, sector_col: str = "sector"
) -> pd.DataFrame:
    """Engineer interactions between features and sector medians."""
    result = df.copy()
    # Simple implementation for now
    metrics = ["last_price", "market_cap"]
    return create_relative_value_features(result, sector_col=sector_col, metrics=metrics)
