# finance_ml/ml_workflow/eda/analyst_analytics.py
"""Analyst rating and recommendations analytics module."""
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

# Constants
MIN_SAMPLE_SIZE = 5
COVERAGE_BINS = [0, 1, 5, 10, 20, float("inf")]
COVERAGE_LABELS = ["0", "1-5", "6-10", "11-20", "20+"]

RATING_COL_PATTERNS: dict[str, list[str]] = {
    "analyst_rating": ["analyst_rating", "rating", "recommendation", "analyst_recommendation"],
    "buy_ratings": ["buy_ratings", "num_buy", "strong_buy", "buy_count"],
    "hold_ratings": ["hold_ratings", "num_hold", "hold_count"],
    "sell_ratings": ["sell_ratings", "num_sell", "strong_sell", "sell_count"],
    "price_target": ["price_target", "price_target", "analyst_target", "price_target_mean"],
    "target_high": ["price_target_high", "target_high", "high_target"],
    "target_low": ["price_target_low", "target_low", "low_target"],
    "num_analysts": ["num_analysts", "analyst_count", "coverage_count"],
}


def find_rating_columns(df: pd.DataFrame, patterns: dict[str, list[str]]) -> dict[str, str]:
    """Identify available analyst rating columns from dataframe."""
    available = {}
    for category, category_patterns in patterns.items():
        for pattern in category_patterns:
            matching = [c for c in df.columns if pattern.lower() in c.lower()]
            if matching:
                available[category] = matching[0]
                break
    return available


def compute_numeric_stats(data: pd.Series) -> Optional[dict]:
    """Compute common statistics for a numeric series."""
    numeric_data = pd.to_numeric(data, errors="coerce").dropna()
    if len(numeric_data) == 0:
        return None
    return {
        "count": int(len(numeric_data)),
        "mean": float(numeric_data.mean()),
        "median": float(numeric_data.median()),
        "std": float(numeric_data.std()),
        "min": float(numeric_data.min()),
        "max": float(numeric_data.max()),
    }


def compute_upside_stats(upside_data: pd.Series) -> dict:
    """Compute upside statistics for a group."""
    positive_pct = (upside_data > 0).sum() / len(upside_data) * 100
    return {
        "count": int(len(upside_data)),
        "mean_upside": float(upside_data.mean()),
        "median_upside": float(upside_data.median()),
        "positive_pct": float(positive_pct),
    }


def analyze_sector_analyst_data(
    df: pd.DataFrame, target_col: str, min_samples: int = MIN_SAMPLE_SIZE
) -> dict[str, dict]:
    """Perform sector-level analyst analysis."""
    sector_stats = {}
    for sector in df["sector"].dropna().unique():
        sector_df = df[df["sector"] == sector]
        target_data = pd.to_numeric(sector_df[target_col], errors="coerce").dropna()

        if len(target_data) < min_samples:
            continue

        sector_stats[str(sector)] = {
            "count": int(len(target_data)),
            "mean_target": float(target_data.mean()),
            "median_target": float(target_data.median()),
        }

        if "target_vs_price" in sector_df.columns:
            upside_data = sector_df["target_vs_price"].dropna()
            if len(upside_data) > 0:
                upside_stats = compute_upside_stats(upside_data)
                sector_stats[str(sector)]["mean_upside"] = upside_stats["mean_upside"]
                sector_stats[str(sector)]["positive_upside_pct"] = upside_stats["positive_pct"]

    return sector_stats


def run_analyst_recommendations_analytics(df: pd.DataFrame, output_dir, serializer_fn=None) -> dict:
    """Run complete analyst rating and recommendations analytics."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_stocks_analyzed": len(df),
        "available_columns": find_rating_columns(df, RATING_COL_PATTERNS),
        "by_sector": {},
        "by_size_class": {},
        "by_style_class": {},
    }

    if "price_target" in results["available_columns"]:
        price_stats = compute_numeric_stats(df[results["available_columns"]["price_target"]])
        if price_stats:
            results["price_target_stats"] = price_stats

    if "sector" in df.columns and "price_target" in results["available_columns"]:
        results["by_sector"] = analyze_sector_analyst_data(
            df, results["available_columns"]["price_target"]
        )

    return results
