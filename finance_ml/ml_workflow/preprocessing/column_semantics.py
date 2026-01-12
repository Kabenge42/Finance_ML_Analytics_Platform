"""
Column semantic classification for preprocessing pipeline.

This module defines semantic categories for financial columns to enable
intelligent preprocessing decisions:

- Price columns: Must be preserved in original units (never transform)
  - Includes: current prices, targets, historical prices, 52w bounds, indicators (23 total)
- Market value columns: Highly skewed, require log-transforms
- Ratio columns: Pre-normalized financial ratios
- Percentage columns: Bounded [0, 100]
- Count columns: Discrete integer counts

Aligned with preprocessing_stages_4-8_improvement_plan.md Task 1.1
and code_guidelines.md v1.7 Section 8.5: Preprocessing Stage Naming

Business Rationale:
The core business metric (Predicted_Target - Last_Price) / Last_Price requires
price columns to remain in original dollar units. This extends to historical
prices (for momentum: (price - price_1m_ago) / price_1m_ago), 52-week bounds
(for relative positioning), and EMAs (for technical analysis). Transforming
these columns corrupts the valuation and momentum analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Optional, Any

# Lazy initialization caches
_PRICE_COLUMNS_CACHE: Optional[Set[str]] = None
_MARKET_VALUE_COLUMNS_CACHE: Optional[Set[str]] = None
_RATIO_COLUMNS_CACHE: Optional[Set[str]] = None
_PERCENTAGE_COLUMNS_CACHE: Optional[Set[str]] = None
_COUNT_COLUMNS_CACHE: Optional[Set[str]] = None


def get_price_columns() -> Set[str]:
    """Get price-related columns derived from COLUMN_SCHEMA."""
    global _PRICE_COLUMNS_CACHE
    if _PRICE_COLUMNS_CACHE is None:
        from finance_ml.core.schema import list_price_cols

        _PRICE_COLUMNS_CACHE = set(list_price_cols())
    return _PRICE_COLUMNS_CACHE


def get_market_value_columns() -> Set[str]:
    """Get market value columns derived from COLUMN_SCHEMA."""
    global _MARKET_VALUE_COLUMNS_CACHE
    if _MARKET_VALUE_COLUMNS_CACHE is None:
        from finance_ml.core.schema import COLUMN_SCHEMA

        _MARKET_VALUE_COLUMNS_CACHE = {
            col
            for col, meta in COLUMN_SCHEMA.items()
            if meta.get("role") in ("market", "financial_statement", "balance_sheet", "cash_flow")
        }
    return _MARKET_VALUE_COLUMNS_CACHE


def get_ratio_columns() -> Set[str]:
    """Get ratio columns derived from COLUMN_SCHEMA."""
    global _RATIO_COLUMNS_CACHE
    if _RATIO_COLUMNS_CACHE is None:
        from finance_ml.core.schema import COLUMN_SCHEMA

        _RATIO_COLUMNS_CACHE = {
            col for col, meta in COLUMN_SCHEMA.items() if meta.get("role") == "ratio"
        }
    return _RATIO_COLUMNS_CACHE


def get_percentage_columns() -> Set[str]:
    """Get percentage columns derived from COLUMN_SCHEMA."""
    global _PERCENTAGE_COLUMNS_CACHE
    if _PERCENTAGE_COLUMNS_CACHE is None:
        from finance_ml.core.schema import COLUMN_SCHEMA

        _PERCENTAGE_COLUMNS_CACHE = {
            col for col, meta in COLUMN_SCHEMA.items() if meta.get("role") == "percentage"
        }
    return _PERCENTAGE_COLUMNS_CACHE


def get_count_columns() -> Set[str]:
    """Get count columns derived from COLUMN_SCHEMA."""
    global _COUNT_COLUMNS_CACHE
    if _COUNT_COLUMNS_CACHE is None:
        from finance_ml.core.schema import list_count_cols

        _COUNT_COLUMNS_CACHE = set(list_count_cols())
    return _COUNT_COLUMNS_CACHE


def __getattr__(name: str) -> Any:
    """Module-level attribute access for legacy constants."""
    if name == "PRICE_COLUMNS":
        return get_price_columns()
    if name == "MARKET_VALUE_COLUMNS":
        return get_market_value_columns()
    if name == "RATIO_COLUMNS":
        return get_ratio_columns()
    if name == "PERCENTAGE_COLUMNS":
        return get_percentage_columns()
    if name == "COUNT_COLUMNS":
        return get_count_columns()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


import pandas as pd

logger = logging.getLogger(__name__)

# Explicit exports for public API
__all__ = [
    # Column sets
    "PRICE_COLUMNS",
    "MARKET_VALUE_COLUMNS",
    "RATIO_COLUMNS",
    "PERCENTAGE_COLUMNS",
    "COUNT_COLUMNS",
    # Main classification function
    "classify_columns",
    # Helper functions for specific column categories
    "get_winsorizable_columns",
    "get_log_transform_columns",
    "get_scalable_columns",
    # Pattern-based classification (advanced use)
    "classify_columns_with_patterns",
    "classify_columns_with_schema_fallback",
    # Suffix patterns (advanced use)
    "SUFFIX_PATTERNS",
]

# Note: PRICE_COLUMNS, MARKET_VALUE_COLUMNS, RATIO_COLUMNS, PERCENTAGE_COLUMNS,
# and COUNT_COLUMNS are now dynamically derived from COLUMN_SCHEMA via __getattr__.

# Market Value Columns - Log-transform to handle high skewness
# These columns typically have skewness > 2.0 and need log-transforms
# instead of winsorization to preserve information about valid extremes

# Ratio Columns - Pre-normalized, may not need winsorization
# Financial ratios are already relative metrics

# Percentage Columns - Bounded [0, 100], inappropriate for percentile capping
# These are already normalized as percentages

# Count Columns - Discrete integer counts, inappropriate for continuous scaling

def classify_columns(df_columns: List[str]) -> Dict[str, Set[str]]:
    """
    Classify DataFrame columns by semantic type.

    Enhanced in Phase 9.3 with pattern-based and schema-based fallback classification
    to reduce the 'other' category from 487 columns to <59 (≥90% coverage).

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        Dict mapping semantic category to set of matching columns:
        - 'price': Price columns (preserve original units)
        - 'market_value': Market value columns (log-transform)
        - 'ratio': Financial ratio columns (pre-normalized)
        - 'percentage': Percentage columns (bounded)
        - 'count': Count columns (discrete)
        - 'other': Numeric columns not in above categories

    Notes
    -----
    Phase 9.3 enhancements:
    - Step 1: Existing price/market_value/ratio detection (predefined sets)
    - Step 2: Pattern-based classification for unclassified columns
    - Step 3: Schema-based fallback for remaining 'other' columns
    """
    result = {
        "price": set(),
        "market_value": set(),
        "ratio": set(),
        "percentage": set(),
        "count": set(),
        "date": set(),
        "id": set(),
        "categorical": set(),
        "other": set(),
    }

    # Step 1: Existing classification logic (predefined sets)
    unclassified_cols = []

    from finance_ml.core.schema import list_date_cols, list_categorical_cols

    date_cols = set(list_date_cols())
    categorical_cols = set(list_categorical_cols())
    id_cols = {"ticker", "isin", "name", "description"}

    for col in df_columns:
        col_lower = col.lower().strip()

        if col_lower in get_price_columns():
            result["price"].add(col)
        elif col_lower in get_market_value_columns():
            result["market_value"].add(col)
        elif col_lower in get_ratio_columns():
            result["ratio"].add(col)
        elif col_lower in get_percentage_columns():
            result["percentage"].add(col)
        elif col_lower in get_count_columns():
            result["count"].add(col)
        elif col_lower in id_cols:
            result["id"].add(col)
        elif col_lower in date_cols:
            result["date"].add(col)
        elif col_lower in categorical_cols:
            result["categorical"].add(col)
        else:
            # Check for log-transformed columns (e.g., log_market_cap)
            if col_lower.startswith("log_"):
                base_col = col_lower[4:]  # Remove 'log_' prefix
                if base_col in get_market_value_columns():
                    result["market_value"].add(col)
                else:
                    unclassified_cols.append(col)
            else:
                unclassified_cols.append(col)

    # Step 2: Pattern-based classification for unclassified (Phase 9.3)
    if unclassified_cols:
        pattern_classifications = classify_columns_with_patterns(unclassified_cols)

        still_unclassified = []
        for col, category in pattern_classifications.items():
            if category == "RATIO":
                result["ratio"].add(col)
            elif category == "PERCENTAGE":
                result["percentage"].add(col)
            elif category == "MARKET_VALUE":
                result["market_value"].add(col)
            elif category == "COUNT":
                result["count"].add(col)
            elif category == "PRICE":
                result["price"].add(col)
            elif category == "DATE":
                result["date"].add(col)
            elif category == "ID":
                result["id"].add(col)
            elif category == "CATEGORICAL":
                result["categorical"].add(col)
            else:
                still_unclassified.append(col)

        # Step 3: Schema-based fallback for remaining OTHER (Phase 9.3)
        if still_unclassified:
            schema_classifications = classify_columns_with_schema_fallback(still_unclassified)

            for col, category in schema_classifications.items():
                if category == "RATIO":
                    result["ratio"].add(col)
                elif category == "PERCENTAGE":
                    result["percentage"].add(col)
                elif category == "MARKET_VALUE":
                    result["market_value"].add(col)
                elif category == "COUNT":
                    result["count"].add(col)
                elif category == "PRICE":
                    result["price"].add(col)
                elif category == "DATE":
                    result["date"].add(col)
                elif category == "ID":
                    result["id"].add(col)
                elif category == "CATEGORICAL":
                    result["categorical"].add(col)
                else:
                    result["other"].add(col)

    logger.debug(
        f"Classified {len(df_columns)} columns: "
        f"price={len(result['price'])}, "
        f"market_value={len(result['market_value'])}, "
        f"ratio={len(result['ratio'])}, "
        f"percentage={len(result['percentage'])}, "
        f"count={len(result['count'])}, "
        f"date={len(result['date'])}, "
        f"id={len(result['id'])}, "
        f"categorical={len(result['categorical'])}, "
        f"other={len(result['other'])}"
    )

    return result

def get_winsorizable_columns(
    df_columns: List[str] | pd.DataFrame,
    exclude_ratios: bool = True,
    exclude_percentages: bool = True,
) -> List[str]:
    """
    Return columns safe for winsorization.

    Excludes:
    - Price columns (must preserve original units)
    - Ratio columns (already normalized) if exclude_ratios is True
    - Percentage columns (already bounded) if exclude_percentages is True
    - Count columns (discrete)

    Includes:
    - Market value columns (but log-transform is preferred)
    - Other numeric features

    Args:
        df_columns: List of column names from DataFrame or DataFrame itself
        exclude_ratios: Whether to exclude ratio columns
        exclude_percentages: Whether to exclude percentage columns

    Returns:
        List of column names safe for winsorization
    """
    if isinstance(df_columns, pd.DataFrame):
        df_columns = df_columns.columns.tolist()

    classification = classify_columns(df_columns)

    # Winsorize market value and other numeric columns
    winsorizable_set = classification["market_value"] | classification["other"]

    # Add ratios/percentages if not excluded
    if not exclude_ratios:
        winsorizable_set |= classification["ratio"]
    if not exclude_percentages:
        winsorizable_set |= classification["percentage"]

    winsorizable = list(winsorizable_set)

    logger.info(
        f"Identified {len(winsorizable)} winsorizable columns "
        f"(excluded {len(classification['price'])} price, "
        f"{len(classification['ratio']) if exclude_ratios else 0} ratio, "
        f"{len(classification['percentage']) if exclude_percentages else 0} percentage, "
        f"{len(classification['count'])} count columns)"
    )

    return winsorizable

def get_log_transform_columns(df_columns: List[str] | pd.DataFrame) -> List[str]:
    """
    Return columns requiring log-transform to handle skewness.

    Includes:
    - Market value columns (highly skewed)

    Excludes:
    - Price columns (preserve original units)
    - Ratio columns (already normalized)
    - Percentage columns (already bounded)
    - Count columns (discrete)
    - Columns already log-transformed (log_*)

    Args:
        df_columns: List of column names from DataFrame or DataFrame itself

    Returns:
        List of column names requiring log-transform
    """
    if isinstance(df_columns, pd.DataFrame):
        df_columns = df_columns.columns.tolist()

    classification = classify_columns(df_columns)

    # Only transform market value columns that aren't already log-transformed
    log_transform = [
        col for col in classification["market_value"] if not col.lower().startswith("log_")
    ]

    logger.info(f"Identified {len(log_transform)} columns for log-transform")

    return log_transform

def get_scalable_columns(
    df_columns: List[str] | pd.DataFrame, exclude_price: bool = True, exclude_counts: bool = True
) -> List[str]:
    """
    Return columns safe for scaling (StandardScaler, RobustScaler, etc.).

    Excludes:
    - Price columns (must preserve original units for business metric) if exclude_price is True
    - Count columns (discrete) if exclude_counts is True

    Includes:
    - Market value columns (especially log-transformed versions)
    - Ratio columns
    - Percentage columns
    - Other numeric features

    Args:
        df_columns: List of column names from DataFrame or DataFrame itself
        exclude_price: Whether to exclude price columns
        exclude_counts: Whether to exclude count columns

    Returns:
        List of column names safe for scaling
    """
    if isinstance(df_columns, pd.DataFrame):
        df_columns = df_columns.columns.tolist()

    classification = classify_columns(df_columns)

    # Base scalable set
    scalable_set = (
        classification["market_value"]
        | classification["ratio"]
        | classification["percentage"]
        | classification["other"]
    )

    # Add/Remove based on flags
    if not exclude_price:
        scalable_set |= classification["price"]

    if not exclude_counts:
        scalable_set |= classification["count"]

    scalable = list(scalable_set)

    logger.info(
        f"Identified {len(scalable)} scalable columns "
        f"(excluded {len(classification['price']) if exclude_price else 0} price, "
        f"{len(classification['count']) if exclude_counts else 0} count columns)"
    )

    return scalable

# ============================================================================
# Phase 9.3: Semantic Column Classification Enhancement
# ============================================================================

# Suffix-based classification rules
# IMPORTANT: Order matters! More specific patterns should come before generic ones.
# Check PERCENTAGE and specific patterns before generic time suffixes (_ltm, _fy, etc.)
SUFFIX_PATTERNS = {
    "PERCENTAGE": [
        # Returns and capital efficiency (check BEFORE generic time suffixes)
        r"^roe",
        r"^roa",
        r"^roic",
        r"^roce",
        r"return_on_equity",
        r"return_on_assets",
        # Margin patterns
        r"_margin",
        r"margin_",
        # Growth patterns
        r"_pct",
        r"_yoy",
        r"_cagr",
        r"growth_",
        # Yield / payout
        r"_yield",
        r"payout_ratio",
        r"retention_",
    ],
    "RATIO": [
        # Basic ratio patterns
        r"_to_",
        r"_coverage",
        r"_turnover",
        # Valuation multiples
        r"p_e",
        r"p_b",
        r"p_s",
        r"p_fcf",
        r"p_tbv",
        r"ev_ebitda",
        r"ev_sales",
        r"ev_fcf",
        # Time-series variants (checked last to avoid over-matching)
        r"_1fyltm$",
        r"_2fyltm$",
        r"_3fyltm$",
        r"_ltm$",
        r"_ntm$",
        r"_fq$",
        r"_fy$",
    ],
    "MARKET_VALUE": [
        # Market metrics
        r"^market_cap",
        r"^enterprise_value",
        r"^ev$",
        # Balance sheet
        r"^total_assets",
        r"^total_debt",
        r"^net_debt",
        r"^cash",
        r"^total_equity",
        r"^tangible_book",
        # Income statement
        r"^revenue",
        r"^total_revenues",
        r"^ebitda",
        r"^operating_income",
        r"^net_income",
        r"^gross_profit",
        # Cash flow
        r"^operating_cash_flow",
        r"^free_cash_flow",
        r"^fcf",
        r"^capex",
    ],
    "COUNT": [
        r"^num_",
        r"_count$",
        r"^shares_",
        r"_num$",
    ],
}

def classify_columns_with_patterns(columns: List[str]) -> Dict[str, str]:
    """
    Classify columns using suffix and prefix patterns.

    Implements Phase 9.3 Task 3: Pattern-based semantic classification.
    Aligned with phase_9.3_implementation_plan.md and code_guidelines.md v1.10.

    Parameters
    ----------
    columns : list of str
        Column names to classify

    Returns
    -------
    dict
        Mapping of column -> semantic category

    Notes
    -----
    - Patterns are checked in priority order: PERCENTAGE, MARKET_VALUE, COUNT, RATIO
    - First matching pattern determines the category
    - Unmatched columns are classified as 'OTHER'
    """
    import re

    classifications = {}

    # Define pattern check order (most specific first)
    pattern_order = ["PERCENTAGE", "MARKET_VALUE", "COUNT", "RATIO"]

    for col in columns:
        col_lower = col.lower()
        classified = False

        for category in pattern_order:
            if category not in SUFFIX_PATTERNS:
                continue
            patterns = SUFFIX_PATTERNS[category]
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    classifications[col] = category
                    classified = True
                    break
            if classified:
                break

        if not classified:
            classifications[col] = "OTHER"

    logger.debug(
        f"Pattern classification: {len([c for c in classifications.values() if c != 'OTHER'])} matched, {len([c for c in classifications.values() if c == 'OTHER'])} unmatched"
    )

    return classifications

def classify_columns_with_schema_fallback(columns: List[str]) -> Dict[str, str]:
    """
    Use COLUMN_SCHEMA as fallback for unclassified columns.

    Implements Phase 9.3 Task 3: Schema-based semantic classification fallback.
    Aligned with phase_9.3_implementation_plan.md and code_guidelines.md v1.10.

    Parameters
    ----------
    columns : list of str
        Column names to classify

    Returns
    -------
    dict
        Mapping of column -> semantic category

    Notes
    -----
    - Uses COLUMN_SCHEMA dtype and role information
    - Price-related columns detected by name pattern
    - float64/float32 default to RATIO if role unclear
    - int64 defaults to COUNT
    - object defaults to CATEGORICAL
    """
    from finance_ml.core.schema import COLUMN_SCHEMA

    classifications = {}

    semantic_roles = {
        "market",
        "financial_statement",
        "balance_sheet",
        "cash_flow",
        "ratio",
        "percentage",
        "count",
        "target",
        "target_fallback",
    }

    for col in columns:
        col_lower = col.lower()

        if col_lower in COLUMN_SCHEMA:
            schema_info = COLUMN_SCHEMA[col_lower]
            dtype = str(schema_info.get("dtype", "object"))
            role = schema_info.get("role", "feature")

            if role in semantic_roles:
                # Map to standard categories for this module
                if role in ["market", "financial_statement", "balance_sheet", "cash_flow"]:
                    classifications[col] = "MARKET_VALUE"
                elif role in ["target", "target_fallback"]:
                    classifications[col] = "PRICE"
                else:
                    classifications[col] = role.upper()
                continue

            # Name-based heuristics
            if ("price" in col_lower or "target" in col_lower or "ema_" in col_lower) and not any(
                keyword in col_lower
                for keyword in [
                    "momentum",
                    "change",
                    "pct",
                    "acceleration",
                    "divergence",
                    "convergence",
                    "trend",
                    "streak",
                    "position",
                    "ratio",
                    "spread",
                ]
            ):
                classifications[col] = "PRICE"
            elif any(
                keyword in col_lower
                for keyword in [
                    "margin",
                    "yield",
                    "growth",
                    "yoy",
                    "cagr",
                    "pct",
                    "beta",
                    "volatility",
                    "return_on",
                ]
            ):
                classifications[col] = "PERCENTAGE"
            elif any(
                keyword in col_lower
                for keyword in [
                    "_to_",
                    "ratio",
                    "turnover",
                    "coverage",
                    "p_e",
                    "p_b",
                    "ev_",
                    "net_debt",
                    "debt_to",
                    "asset_turnover",
                    "current_ratio",
                ]
            ):
                classifications[col] = "RATIO"
            elif role == "categorical":
                classifications[col] = "CATEGORICAL"
            elif "float" in dtype.lower():
                classifications[col] = "RATIO"
            elif "int" in dtype.lower():
                classifications[col] = "COUNT"
            else:
                classifications[col] = "OTHER"
        else:
            classifications[col] = "OTHER"

    logger.debug(
        f"Schema fallback classification: {len([c for c in classifications.values() if c != 'OTHER'])} matched, {len([c for c in classifications.values() if c == 'OTHER'])} unmatched"
    )

    return classifications

# Alias for enterprise_value
get_market_value_columns().add("ev")
