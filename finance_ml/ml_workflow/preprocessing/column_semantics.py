"""
Column semantic classification for preprocessing pipeline.

This module defines semantic categories for financial columns to enable
intelligent preprocessing decisions:

- Price columns: Must be preserved in original units (never transform)
- Market value columns: Highly skewed, require log-transforms
- Ratio columns: Pre-normalized financial ratios
- Percentage columns: Bounded [0, 100]
- Count columns: Discrete integer counts

Aligned with preprocessing_stages_4-8_improvement_plan.md Task 1.1
and code_guidelines.md v1.5 Section 8.5: Preprocessing Stage Naming

Business Rationale:
The core business metric (Predicted_Target - Last_Price) / Last_Price requires
price columns to remain in original dollar units. Transforming prices corrupts
the valuation analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


# Price Columns - NEVER transform (critical for business metric)
# These columns must remain in original dollar units for valuation analysis
PRICE_COLUMNS: Set[str] = {
    'last_price',              # Current market price (critical)
    'price_target',            # Analyst consensus target (critical)
    'price_target_median',     # Median analyst target
    'price_target_ytd_ago',    # Historical target (YTD)
    'price_target_12m_ago',    # Historical target (12M)
    'price_target_low',        # Low analyst target
    'price_target_high',       # High analyst target
}


# Market Value Columns - Log-transform to handle high skewness
# These columns typically have skewness > 2.0 and need log-transforms
# instead of winsorization to preserve information about valid extremes
MARKET_VALUE_COLUMNS: Set[str] = {
    # Market capitalization and enterprise value
    'market_cap',              # Market capitalization (highly skewed)
    'ev',                      # Enterprise value
    'enterprise_value',        # Enterprise value (alternative name)
    
    # Balance sheet items
    'total_assets',            # Total assets
    'total_debt',              # Total debt
    'net_debt',                # Net debt
    'cash_and_equivalents',    # Cash and equivalents
    'total_equity',            # Total equity
    'tangible_book_value',     # Tangible book value
    
    # Income statement items
    'revenue',                 # Revenue (highly skewed)
    'ebitda',                  # EBITDA
    'operating_income',        # Operating income
    'net_income',              # Net income (can be negative)
    'gross_profit',            # Gross profit
    
    # Cash flow items
    'operating_cash_flow',     # Operating cash flow
    'free_cash_flow',          # Free cash flow
    'capex',                   # Capital expenditures
}


# Ratio Columns - Pre-normalized, may not need winsorization
# Financial ratios are already relative metrics
RATIO_COLUMNS: Set[str] = {
    # Valuation ratios
    'p_e', 'p_b', 'p_s', 'p_fcf', 'p_tbv',
    'p_e_ntm', 'p_e_ltm', 'p_e_1fyltm',
    'p_b_ltm', 'p_b_1fy', 'p_b_5yavg',
    'p_tbv_ltm',
    'ev_ebitda', 'ev_sales', 'ev_fcf',
    'ev_ebitda_ltm', 'ev_ebitda_ntm', 'ev_ebitda_est_fy1',
    'ev_sales_ltm', 'ev_sales_ntm', 'ev_sales_est_fy1',
    'ev_ebitda_1fyltm', 'ev_ebitda_1fqltm', 'ev_ebitda_3yavgltm',
    'ev_sales_1fyltm', 'ev_sales_2fyltm', 'ev_sales_3fyltm',
    'ev_sales_3yavgltm', 'ev_sales_1fqltm', 'ev_sales_2fqltm',
    'ev_sales_3fqltm', 'ev_sales_4fqltm',
    'p_e_2fyltm', 'p_e_3fyltm', 'p_e_3yavgltm',
    'p_e_1fqltm', 'p_e_2fqltm', 'p_e_3fqltm',
    'p_e_0fqqoqltm', 'p_e_0fyyoyltm', 'p_e_1fyyoyltm', 'p_e_0fqyoyltm',
    'p_e_est_fy1',
    
    # Profitability ratios
    'roe', 'roa', 'roic', 'roce',
    'roe_ltm', 'roa_ltm', 'roic_ltm',
    
    # Leverage ratios
    'debt_equity', 'debt_to_equity',
    'net_debt_ebitda', 'net_debt_to_ebitda',
    'debt_to_assets',
    
    # Liquidity ratios
    'current_ratio', 'quick_ratio', 'cash_ratio',
    
    # Efficiency ratios
    'asset_turnover', 'inventory_turnover',
}


# Percentage Columns - Bounded [0, 100], inappropriate for percentile capping
# These are already normalized as percentages
PERCENTAGE_COLUMNS: Set[str] = {
    # Margin metrics
    'gross_margin', 'operating_margin', 'net_margin', 'ebitda_margin',
    'gross_margin_ltm', 'operating_margin_ltm', 'net_margin_ltm',
    'ebitda_margin_ltm',
    
    # Growth rates
    'revenue_growth_yoy', 'earnings_growth_yoy', 'ebitda_growth_yoy',
    'revenue_growth_3y_cagr', 'revenue_growth_5y_cagr',
    'earnings_growth_3y_cagr', 'earnings_growth_5y_cagr',
    
    # Volatility metrics
    'volatility_20d', 'volatility_60d', 'volatility_1y',
    'beta', 'beta_5y',
    
    # Payout ratios
    'dividend_payout_ratio', 'payout_ratio',
}


# Count Columns - Discrete integer counts, inappropriate for continuous scaling
COUNT_COLUMNS: Set[str] = {
    # Analyst coverage
    'num_analysts',
    'num_strong_buy_ratings',
    'num_buy_ratings',
    'num_hold_ratings',
    'num_sell_ratings',
    'num_strong_sell_ratings',
    'price_target_num',
    
    # Company metrics
    'num_employees',
    'num_employees_total',
}


def classify_columns(df_columns: List[str]) -> Dict[str, Set[str]]:
    """
    Classify DataFrame columns by semantic type.
    
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
    """
    result = {
        'price': set(),
        'market_value': set(),
        'ratio': set(),
        'percentage': set(),
        'count': set(),
        'other': set(),
    }
    
    for col in df_columns:
        col_lower = col.lower().strip()
        
        if col_lower in PRICE_COLUMNS:
            result['price'].add(col)
        elif col_lower in MARKET_VALUE_COLUMNS:
            result['market_value'].add(col)
        elif col_lower in RATIO_COLUMNS:
            result['ratio'].add(col)
        elif col_lower in PERCENTAGE_COLUMNS:
            result['percentage'].add(col)
        elif col_lower in COUNT_COLUMNS:
            result['count'].add(col)
        else:
            # Check for log-transformed columns (e.g., log_market_cap)
            if col_lower.startswith('log_'):
                base_col = col_lower[4:]  # Remove 'log_' prefix
                if base_col in MARKET_VALUE_COLUMNS:
                    result['market_value'].add(col)
                else:
                    result['other'].add(col)
            else:
                result['other'].add(col)
    
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
    winsorizable = list(classification['market_value'] | classification['other'])
    
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
        col for col in classification['market_value']
        if not col.lower().startswith('log_')
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
        classification['market_value'] |
        classification['ratio'] |
        classification['percentage'] |
        classification['count'] |
        classification['other']
    )
    
    logger.info(
        f"Identified {len(scalable)} scalable columns "
        f"(excluded {len(classification['price'])} price columns)"
    )
    
    return scalable


# Alias for enterprise_value
MARKET_VALUE_COLUMNS.add('ev')
