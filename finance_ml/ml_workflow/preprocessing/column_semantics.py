"""
Column semantic classification for preprocessing pipeline.

This module defines semantic categories for financial columns to enable
intelligent preprocessing decisions:

- Price columns: Must be preserved in original units (never transform)
  - Includes: current prices, targets, historical prices, 52w bounds, EMAs (21 total)
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
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


# Price Columns - NEVER transform (critical for business metric)
# These columns must remain in original dollar units for valuation analysis
PRICE_COLUMNS: Set[str] = {
    # Current prices and targets
    "last_price",  # Current market price (critical)
    "price_target",  # Analyst consensus target (critical)
    "price_target_median",  # Median analyst target
    "price_target_ytd_ago",  # Historical target (YTD)
    "price_target_low",  # Low analyst target
    "price_target_high",  # High analyst target
    # Historical prices (for momentum calculations)
    "price_5d_ago",  # Price 5 days ago
    "price_1w_ago",  # Price 1 week ago
    "price_1m_ago",  # Price 1 month ago
    "price_3m_ago",  # Price 3 months ago
    "price_6m_ago",  # Price 6 months ago
    "price_1y_ago",  # Price 1 year ago
    "price_3y_ago",  # Price 3 years ago
    "price_5y_ago",  # Price 5 years ago
    "price_qtd_ago",  # Price at quarter-to-date start
    # 52-week bounds (for relative positioning)
    "52w_high_adj",  # 52-week adjusted high
    "52w_low_adj",  # 52-week adjusted low
    # Exponential moving averages (technical indicators)
    "ema_20d",  # 20-day EMA
    "ema_50d",  # 50-day EMA
    "ema_100d",  # 100-day EMA
    "ema_250d",  # 250-day EMA (1-year trend proxy)
}


# Market Value Columns - Log-transform to handle high skewness
# These columns typically have skewness > 2.0 and need log-transforms
# instead of winsorization to preserve information about valid extremes
MARKET_VALUE_COLUMNS: Set[str] = {
    # Market capitalization and enterprise value
    "market_cap",  # Market capitalization (highly skewed)
    "ev",  # Enterprise value
    "enterprise_value",  # Enterprise value (alternative name)
    # Balance sheet items
    "total_assets",  # Total assets
    "total_debt",  # Total debt
    "net_debt",  # Net debt
    "cash_and_equivalents",  # Cash and equivalents
    "total_equity",  # Total equity
    "tangible_book_value",  # Tangible book value
    # Income statement items
    "revenue",  # Revenue (highly skewed)
    "ebitda",  # EBITDA
    "operating_income",  # Operating income
    "net_income",  # Net income (can be negative)
    "gross_profit",  # Gross profit
    # Cash flow items
    "operating_cash_flow",  # Operating cash flow
    "free_cash_flow",  # Free cash flow
    "capex",  # Capital expenditures
}


# Ratio Columns - Pre-normalized, may not need winsorization
# Financial ratios are already relative metrics
RATIO_COLUMNS: Set[str] = {
    # Valuation ratios
    "p_e",
    "p_b",
    "p_s",
    "p_fcf",
    "p_tbv",
    "p_e_ntm",
    "p_e_ltm",
    "p_e_1fyltm",
    "p_b_ltm",
    "p_b_1fy",
    "p_b_5yavg",
    "p_tbv_ltm",
    "ev_ebitda",
    "ev_sales",
    "ev_fcf",
    "ev_ebitda_ltm",
    "ev_ebitda_ntm",
    "ev_ebitda_est_fy1",
    "ev_sales_ltm",
    "ev_sales_ntm",
    "ev_sales_est_fy1",
    "ev_ebitda_1fyltm",
    "ev_ebitda_1fqltm",
    "ev_ebitda_3yavgltm",
    "ev_sales_1fyltm",
    "ev_sales_2fyltm",
    "ev_sales_3fyltm",
    "ev_sales_3yavgltm",
    "ev_sales_1fqltm",
    "ev_sales_2fqltm",
    "ev_sales_3fqltm",
    "ev_sales_4fqltm",
    "p_e_2fyltm",
    "p_e_3fyltm",
    "p_e_3yavgltm",
    "p_e_1fqltm",
    "p_e_2fqltm",
    "p_e_3fqltm",
    "p_e_0fqqoqltm",
    "p_e_0fyyoyltm",
    "p_e_1fyyoyltm",
    "p_e_0fqyoyltm",
    "p_e_est_fy1",
    # Profitability ratios
    "roe",
    "roa",
    "roic",
    "roce",
    "roe_ltm",
    "roa_ltm",
    "roic_ltm",
    # Leverage ratios
    "debt_equity",
    "debt_to_equity",
    "net_debt_ebitda",
    "net_debt_to_ebitda",
    "debt_to_assets",
    # Liquidity ratios
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    # Efficiency ratios
    "asset_turnover",
    "inventory_turnover",
}


# Percentage Columns - Bounded [0, 100], inappropriate for percentile capping
# These are already normalized as percentages
PERCENTAGE_COLUMNS: Set[str] = {
    # Margin metrics
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "gross_margin_ltm",
    "operating_margin_ltm",
    "net_margin_ltm",
    "ebitda_margin_ltm",
    # Growth rates
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "ebitda_growth_yoy",
    "revenue_growth_3y_cagr",
    "revenue_growth_5y_cagr",
    "earnings_growth_3y_cagr",
    "earnings_growth_5y_cagr",
    # Volatility metrics
    "volatility_20d",
    "volatility_60d",
    "volatility_1y",
    "beta",
    "beta_5y",
    # Payout ratios
    "dividend_payout_ratio",
    "payout_ratio",
}


# Count Columns - Discrete integer counts, inappropriate for continuous scaling
COUNT_COLUMNS: Set[str] = {
    # Analyst coverage
    "num_analysts",
    "num_strong_buy_ratings",
    "num_buy_ratings",
    "num_hold_ratings",
    "num_sell_ratings",
    "num_strong_sell_ratings",
    "price_target_num",
    # Company metrics
    "num_employees",
    "num_employees_total",
}


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
        "other": set(),
    }

    # Step 1: Existing classification logic (predefined sets)
    unclassified_cols = []

    for col in df_columns:
        col_lower = col.lower().strip()

        if col_lower in PRICE_COLUMNS:
            result["price"].add(col)
        elif col_lower in MARKET_VALUE_COLUMNS:
            result["market_value"].add(col)
        elif col_lower in RATIO_COLUMNS:
            result["ratio"].add(col)
        elif col_lower in PERCENTAGE_COLUMNS:
            result["percentage"].add(col)
        elif col_lower in COUNT_COLUMNS:
            result["count"].add(col)
        else:
            # Check for log-transformed columns (e.g., log_market_cap)
            if col_lower.startswith("log_"):
                base_col = col_lower[4:]  # Remove 'log_' prefix
                if base_col in MARKET_VALUE_COLUMNS:
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
                elif category == "CATEGORICAL":
                    # Categorical columns go to 'other' for now
                    result["other"].add(col)
                else:
                    result["other"].add(col)

    logger.debug(
        f"Classified {len(df_columns)} columns: "
        f"price={len(result['price'])}, "
        f"market_value={len(result['market_value'])}, "
        f"ratio={len(result['ratio'])}, "
        f"percentage={len(result['percentage'])}, "
        f"count={len(result['count'])}, "
        f"other={len(result['other'])}"
    )

    return result


def get_winsorizable_columns(df_columns: List[str]) -> List[str]:
    """
    Return columns safe for winsorization.

    Excludes:
    - Price columns (must preserve original units)
    - Ratio columns (already normalized)
    - Percentage columns (already bounded)
    - Count columns (discrete)

    Includes:
    - Market value columns (but log-transform is preferred)
    - Other numeric features

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        List of column names safe for winsorization
    """
    classification = classify_columns(df_columns)

    # Winsorize market value and other numeric columns
    # Exclude price, ratio, percentage, and count columns
    winsorizable = list(classification["market_value"] | classification["other"])

    logger.info(
        f"Identified {len(winsorizable)} winsorizable columns "
        f"(excluded {len(classification['price'])} price, "
        f"{len(classification['ratio'])} ratio, "
        f"{len(classification['percentage'])} percentage, "
        f"{len(classification['count'])} count columns)"
    )

    return winsorizable


def get_log_transform_columns(df_columns: List[str]) -> List[str]:
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
        df_columns: List of column names from DataFrame

    Returns:
        List of column names requiring log-transform
    """
    classification = classify_columns(df_columns)

    # Only transform market value columns that aren't already log-transformed
    log_transform = [
        col for col in classification["market_value"] if not col.lower().startswith("log_")
    ]

    logger.info(f"Identified {len(log_transform)} columns for log-transform")

    return log_transform


def get_scalable_columns(df_columns: List[str]) -> List[str]:
    """
    Return columns safe for scaling (StandardScaler, RobustScaler, etc.).

    Excludes:
    - Price columns (must preserve original units for business metric)

    Includes:
    - Market value columns (especially log-transformed versions)
    - Ratio columns
    - Percentage columns
    - Other numeric features

    Note: Count columns are included but may benefit from different treatment

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        List of column names safe for scaling
    """
    classification = classify_columns(df_columns)

    # Scale everything except price columns
    scalable = list(
        classification["market_value"]
        | classification["ratio"]
        | classification["percentage"]
        | classification["count"]
        | classification["other"]
    )

    logger.info(
        f"Identified {len(scalable)} scalable columns "
        f"(excluded {len(classification['price'])} price columns)"
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
        # Returns (check BEFORE time suffixes like _ltm)
        r"^roe",  # return on equity
        r"^roa",  # return on assets
        r"^roic",  # return on invested capital
        r"^roce",  # return on capital employed
        # Margin patterns
        r"_margin",  # operating_margin, net_margin, gross_margin
        r"margin_",  # margin_fy, margin_ltm
        # Growth patterns
        r"_pct",  # growth_pct, change_pct
        r"_yoy",  # year-over-year growth
        r"_cagr",  # compound annual growth rate
        r"growth_",  # growth_rate, growth_3y
        # Other percentage metrics
        r"_yield",  # dividend_yield, fcf_yield
        r"payout_ratio",  # dividend payout ratio
        r"retention_",  # retention rate
    ],
    "RATIO": [
        # Basic ratio patterns
        r"_to_",  # debt_to_equity, price_to_book
        r"_coverage",  # interest_coverage, dividend_coverage
        r"_turnover",  # asset_turnover, inventory_turnover
        # Valuation multiples
        r"p_e",  # p/e and variants (p_e_ntm, p_e_ltm, etc.)
        r"p_b",  # p/b and variants
        r"p_s",  # price to sales
        r"p_fcf",  # price to free cash flow
        r"p_tbv",  # price to tangible book value
        r"ev_ebitda",  # ev/ebitda variants
        r"ev_sales",  # ev/sales variants
        r"ev_fcf",  # ev/fcf variants
        # Financial ratios with time suffixes (checked AFTER PERCENTAGE patterns)
        r"_ltm$",  # Last twelve months metrics (often ratios)
        r"_ntm$",  # Next twelve months metrics
        r"_fq$",  # Fiscal quarter metrics
        r"_fy$",  # Fiscal year metrics
        r"_1fyltm$",  # 1 fiscal year last twelve months
        r"_2fyltm$",  # 2 fiscal year last twelve months
        r"_3fyltm$",  # 3 fiscal year last twelve months
    ],
    "MARKET_VALUE": [
        # Market metrics
        r"^market_cap",
        r"^enterprise_value",
        r"^ev$",  # Enterprise value abbreviation
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
        r"^num_",  # num_employees, num_analysts
        r"_count$",  # analyst_count, rating_count
        r"^shares_",  # shares_outstanding
        r"_num$",  # various _num suffixes
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
    from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

    classifications = {}

    for col in columns:
        col_lower = col.lower()

        if col_lower in COLUMN_SCHEMA:
            schema_info = COLUMN_SCHEMA[col_lower]
            dtype = str(schema_info.get("dtype", "object"))
            role = schema_info.get("role", "feature")

            # Infer from dtype and role
            if "price" in col_lower or "target" in col_lower:
                classifications[col] = "PRICE"
            elif role == "categorical":
                classifications[col] = "CATEGORICAL"
            elif "float" in dtype:  # Matches 'float', 'float64', 'float32'
                # Default numeric to RATIO if unknown
                classifications[col] = "RATIO"
            elif "int" in dtype:  # Matches 'int', 'int64', 'int32'
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
MARKET_VALUE_COLUMNS.add("ev")
