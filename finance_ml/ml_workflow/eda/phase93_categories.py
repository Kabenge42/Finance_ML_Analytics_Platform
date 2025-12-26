"""
Phase 9.3 Feature Category Registry

Maps Phase 9.3 engineered features to their categorical families for
EDA, evaluation, and reporting purposes.

This module provides a centralized registry of Phase 9.3 feature families
to enable explicit tracking and analysis in EDA summaries, dashboards,
and analytics workflows.

NOTE: This module imports PHASE93_FEATURE_CATEGORIES from the Single Source
of Truth (finance_ml.core.schema). All category definitions should be
maintained ONLY in schema.py.

UPDATED: 2025-12-27
- Aligned with schema.py as Single Source of Truth
- Added schema-aware helper functions
- Added get_expected_feature_count() for validation
"""

from typing import Dict, List, Optional

import pandas as pd

from finance_ml.core.schema import COLUMN_SCHEMA, PHASE93_FEATURE_CATEGORIES


def get_feature_category(feature_name: str) -> Optional[str]:
    """
    Get the Phase 9.3 category for a given feature name.

    Looks up the feature in PHASE93_FEATURE_CATEGORIES from the schema.

    Args:
        feature_name: Name of the feature column

    Returns:
        Category name if found, None otherwise
    """
    for category, features in PHASE93_FEATURE_CATEGORIES.items():
        if feature_name in features:
            return category
    return None


def get_feature_role(feature_name: str) -> Optional[str]:
    """
    Get the schema role for a given feature name from COLUMN_SCHEMA.

    Args:
        feature_name: Name of the feature column

    Returns:
        Role name (e.g., 'feature', 'ratio', 'percentage') if found, None otherwise
    """
    meta = COLUMN_SCHEMA.get(feature_name)
    if meta:
        return meta.get("role")
    return None


def categorize_dataframe_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize all DataFrame columns by Phase 9.3 feature family.

    Args:
        df: DataFrame with potential Phase 9.3 features

    Returns:
        Dictionary mapping category names to lists of present column names
    """
    categorized = {category: [] for category in PHASE93_FEATURE_CATEGORIES.keys()}

    for col in df.columns:
        category = get_feature_category(col)
        if category:
            categorized[category].append(col)

    # Remove empty categories
    categorized = {k: v for k, v in categorized.items() if v}

    return categorized


def get_phase93_coverage_stats(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get count of Phase 9.3 features present in DataFrame by category.

    Args:
        df: DataFrame with potential Phase 9.3 features

    Returns:
        Dictionary mapping category names to feature counts
    """
    categorized = categorize_dataframe_columns(df)
    return {category: len(features) for category, features in categorized.items()}


def get_expected_feature_count() -> int:
    """
    Get the total expected feature count from PHASE93_FEATURE_CATEGORIES.

    This count is derived from the schema's Single Source of Truth.

    Returns:
        Total number of unique features across all Phase 9.3 categories
    """
    all_features = set()
    for features in PHASE93_FEATURE_CATEGORIES.values():
        all_features.update(features)
    return len(all_features)


def get_expected_features_by_category() -> Dict[str, int]:
    """
    Get expected feature counts by category from PHASE93_FEATURE_CATEGORIES.

    Returns:
        Dictionary mapping category names to expected feature counts
    """
    return {category: len(features) for category, features in PHASE93_FEATURE_CATEGORIES.items()}


def list_all_phase93_features() -> List[str]:
    """
    Get flat list of all unique Phase 9.3 feature names.

    Returns:
        List of all unique feature names across all categories
    """
    all_features = set()
    for features in PHASE93_FEATURE_CATEGORIES.values():
        all_features.update(features)
    return sorted(all_features)


def get_category_description(category: str) -> str:
    """
    Get description of a Phase 9.3 category.

    Args:
        category: Category name

    Returns:
        Description string
    """
    descriptions = {
        "Momentum & Technical": "Price momentum, technical indicators, EMA signals, volume patterns",
        "Valuation Ratios": "EV/EBITDA, P/E, P/B, and other valuation multiples with time-series trends",
        "Profitability": "ROE, ROA, profit margins, and profitability trends",
        "Quality & Risk": "Accounting quality, earnings quality, distress indicators, Altman Z-Score",
        "Cash Flow": "Operating cash flow, FCF metrics, cash conversion, and quality indicators",
        "Capital Allocation": "Dividends, buybacks, reinvestment, M&A intensity, dividend reliability",
        "Analyst Sentiment": "Analyst ratings, consensus strength, target revisions",
        "Market Sentiment": "Market microstructure, liquidity, short interest, sentiment indicators",
        "Leverage & Liquidity": "Debt ratios, coverage ratios, liquidity metrics, financial flexibility",
        "Temporal Patterns": "Seasonality, trend consistency, cyclicality indicators, reporting dates",
        "Composite Scores": "Multi-factor composite scores combining quality, growth, value, momentum",
        "Growth Metrics": "Revenue, earnings, and EBITDA growth rates (YoY and multi-period)",
        "Efficiency Ratios": "Asset turnover, inventory turnover, receivables efficiency, revenue per employee",
        "Employee Productivity": "Workforce metrics, employee growth, revenue/profit per employee, hiring intensity",
        "Balance Sheet Dynamics": "Asset/equity/debt growth rates, balance sheet expansion, retained earnings trends",
        "Revenue Forecasting": "Analyst estimate spreads, consensus uncertainty, implied growth, forecast reliability",
        "Earnings Quality": "EPS/revenue/EBITDA surprises, GAAP vs. Adjusted analytics, earnings quality scores",
        "Technical Analysis": "RSI, 52-week range, volume momentum, EMA crossovers",
        "Valuation Timeseries": "Multi-period valuation trends, mean reversion, forward discounts",
        "Dividend Reliability": "Consistency, coverage, growth streaks, sustainable dividend flags",
        "Employment Dynamics": "Workforce volatility, hiring intensity, employee growth acceleration",
    }
    return descriptions.get(category, "")


def validate_phase93_coverage(
    df: pd.DataFrame,
    min_coverage_pct: float = 90.0,
) -> tuple[bool, Dict[str, any]]:
    """
    Validate that DataFrame has sufficient Phase 9.3 feature coverage.

    Args:
        df: DataFrame with potential Phase 9.3 features
        min_coverage_pct: Minimum percentage of expected features required (default 90%)

    Returns:
        Tuple of (is_valid, report_dict) where report_dict contains:
            - total_expected: Total features defined in schema
            - total_found: Total features found in DataFrame
            - coverage_pct: Percentage coverage
            - by_category: Coverage breakdown by category
            - missing_features: List of missing feature names
    """
    expected_total = get_expected_feature_count()
    expected_by_category = get_expected_features_by_category()
    coverage_stats = get_phase93_coverage_stats(df)
    
    total_found = sum(coverage_stats.values())
    coverage_pct = (total_found / expected_total * 100) if expected_total > 0 else 0.0
    
    # Find missing features
    all_expected = set(list_all_phase93_features())
    found_features = set()
    for features in categorize_dataframe_columns(df).values():
        found_features.update(features)
    missing_features = sorted(all_expected - found_features)
    
    report = {
        "total_expected": expected_total,
        "total_found": total_found,
        "coverage_pct": coverage_pct,
        "by_category": {
            cat: {
                "expected": expected_by_category.get(cat, 0),
                "found": coverage_stats.get(cat, 0),
            }
            for cat in PHASE93_FEATURE_CATEGORIES.keys()
        },
        "missing_features": missing_features,
    }
    
    is_valid = coverage_pct >= min_coverage_pct
    return is_valid, report
