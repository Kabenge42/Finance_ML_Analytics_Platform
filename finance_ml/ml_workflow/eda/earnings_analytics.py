# finance_ml/ml_workflow/eda/earnings_analytics.py
"""Earnings estimates and surprise analytics module."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

EARNINGS_COL_PATTERNS = {
    "eps_actual": ["net_eps_basic_fq", "net_eps_basic_ltm", "net_eps_basic_fy"],
    "eps_estimate": ["eps_norm_est_avg_ntm", "eps_norm_est_avg_fy1e", "eps_gaap_est_avg_ntm"],
    "eps_adjusted": ["eps_adj_fy", "eps_adj_1fy", "eps_adj_ltm"],
    "revenue_actual": ["total_revenues_fq", "total_revenues_ltm", "total_revenues_fy"],
    "ebitda": ["ebitda_ltm", "ebitda_est_avg_fy1e", "ebitda_est_avg_ntm"],
}


def find_available_columns(df_columns: list, patterns_dict: dict) -> dict:
    """Find first available column for each category from pattern lists."""
    available = {}
    for category, patterns in patterns_dict.items():
        for pattern in patterns:
            if pattern in df_columns:
                available[category] = pattern
                break
    return available


def compute_metric_statistics(series: pd.Series) -> Optional[dict]:
    """Compute standard statistics for a numeric series."""
    data = pd.to_numeric(series, errors="coerce").dropna()
    if len(data) == 0:
        return None
    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
        "positive_pct": float((data > 0).sum() / len(data) * 100),
        "negative_pct": float((data < 0).sum() / len(data) * 100),
    }


def compute_earnings_surprise(
    df: pd.DataFrame, actual_col: str, estimate_col: str
) -> Tuple[pd.Series, dict]:
    """Calculate earnings surprise percentage and statistics."""
    mask = df[actual_col].notna() & df[estimate_col].notna()
    actual = pd.to_numeric(df.loc[mask, actual_col], errors="coerce")
    estimated = pd.to_numeric(df.loc[mask, estimate_col], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        surprise_pct = ((actual - estimated) / estimated.abs()) * 100
    surprise_pct = surprise_pct.replace([np.inf, -np.inf], np.nan).dropna()

    if len(surprise_pct) == 0:
        return pd.Series(dtype=float), {}

    stats = {
        "count": int(len(surprise_pct)),
        "mean_surprise_pct": float(surprise_pct.mean()),
        "median_surprise_pct": float(surprise_pct.median()),
        "beat_pct": float((surprise_pct > 0).sum() / len(surprise_pct) * 100),
        "miss_pct": float((surprise_pct < 0).sum() / len(surprise_pct) * 100),
        "large_beat_pct": float((surprise_pct > 10).sum() / len(surprise_pct) * 100),
        "large_miss_pct": float((surprise_pct < -10).sum() / len(surprise_pct) * 100),
    }
    return surprise_pct, stats


def analyze_segment(df: pd.DataFrame, segment_col: str, eps_col: str, min_samples: int = 5) -> dict:
    """Analyze earnings metrics by segment."""
    segment_stats = {}
    for segment in df[segment_col].dropna().unique():
        segment_df = df[df[segment_col] == segment]
        eps_data = pd.to_numeric(segment_df[eps_col], errors="coerce").dropna()

        if len(eps_data) < min_samples:
            continue

        stats = {
            "count": int(len(eps_data)),
            "mean_eps": float(eps_data.mean()),
            "median_eps": float(eps_data.median()),
            "positive_pct": float((eps_data > 0).sum() / len(eps_data) * 100),
        }

        if "calculated_eps_surprise" in segment_df.columns:
            surprise = segment_df["calculated_eps_surprise"].dropna()
            if len(surprise) >= 3:
                stats["mean_surprise"] = float(surprise.mean())
                stats["beat_pct"] = float((surprise > 0).sum() / len(surprise) * 100)

        segment_stats[str(segment)] = stats
    return segment_stats
