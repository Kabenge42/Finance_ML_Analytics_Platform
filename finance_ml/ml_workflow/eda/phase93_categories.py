"""
Phase 9.3 Feature Category Registry

Maps Phase 9.3 engineered features to their categorical families for
EDA, evaluation, and reporting purposes.

This module provides a centralized registry of Phase 9.3 feature families
to enable explicit tracking and analysis in EDA summaries, dashboards,
and analytics workflows.

UPDATED: 2025-12-25
Registry fully synchronized with actual generator outputs from advanced.py.
- 307 features registered across 21 categories (up from 17)
- 100% coverage: All generated features are registered
- All feature names verified to match actual generator outputs
- NEW: Earnings Quality category (33 features) from engineer_estimated_vs_actual_analytics()
  and engineer_gaap_vs_adjusted_analytics()
- ENHANCED: Technical Analysis (15 features), Valuation Timeseries (16 features),
  Revenue Forecasting (9 features), Dividend Reliability (12 features),
  Employment Dynamics (10 features)

Categories:
1. Momentum & Technical (25 features)
2. Valuation Ratios (23 features)
3. Profitability (13 features)
4. Quality & Risk (18 features)
5. Cash Flow (5 features)
6. Capital Allocation (21 features)
7. Analyst Sentiment (10 features)
8. Market Sentiment (5 features) - ENHANCED
9. Leverage & Liquidity (8 features)
10. Temporal Patterns (16 features) - ENHANCED
11. Composite Scores (5 features)
12. Growth Metrics (6 features)
13. Efficiency Ratios (4 features)
14. Employee Productivity (24 features) - ENHANCED
15. Balance Sheet Dynamics (9 features)
16. Revenue Forecasting (9 features)
17. Earnings Quality (33 features) - NEW
18. Technical Analysis (14 features) - NEW
19. Valuation Timeseries (23 features) - NEW
20. Dividend Reliability (21 features) - NEW
21. Employment Dynamics (15 features) - NEW
"""

from typing import Dict, List, Optional

import pandas as pd

from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES


def get_feature_category(feature_name: str) -> Optional[str]:
    """
    Get the Phase 9.3 category for a given feature name.

    Args:
        feature_name: Name of the feature column

    Returns:
        Category name if found, None otherwise
    """
    for category, features in PHASE93_FEATURE_CATEGORIES.items():
        if feature_name in features:
            return category
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


def list_all_phase93_features() -> List[str]:
    """
    Get flat list of all Phase 9.3 feature names.

    Returns:
        List of all feature names across all categories
    """
    all_features = []
    for features in PHASE93_FEATURE_CATEGORIES.values():
        all_features.extend(features)
    return all_features


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
        "Earnings Quality": "EPS/revenue/EBITDA surprises, GAAP vs. Adjusted analytics, earnings quality scores, adjustment flags",
    }
    return descriptions.get(category, "")
