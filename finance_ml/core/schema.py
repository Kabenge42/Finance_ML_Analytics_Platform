"""
Unified schema module - Single Source of Truth.

This module is the ONLY place where column definitions exist.
All other modules MUST import from here.
"""

from __future__ import annotations

from typing import Dict, List, Literal, TypedDict, Optional

# =============================================================================
# Type Aliases for Schema Definition
# =============================================================================

# DType: Maps to pandas/numpy dtype strings for ETL casting
DType = Literal[
    "float",           # float64 - default for numeric financial data
    "int",             # int64 - discrete counts, integer IDs
    "string",          # object/string - text data
    "category",        # pandas Categorical - low-cardinality (sector, region)
    "datetime64[ns]",  # datetime columns
    "bool",            # boolean flags
]

# Role: Semantic role determining preprocessing and pipeline treatment
Role = Literal[
    "id",               # Identifier columns (ticker, isin, name) - never used as features
    "target",           # Primary prediction target (price_target)
    "target_fallback",  # Alternative targets (price_target_median, last_price)
    "date",             # Temporal columns for time-series features
    "categorical",      # Grouping columns (sector, region, industry)
    "auxiliary",        # Legacy aliases, optional - excluded from diagnostics
    "feature",          # General ML features not in other categories
    "price",            # Price columns - NEVER transform
    "market_value",     # Market cap, revenue, assets - log-transform recommended
    "ratio",            # Pre-normalized ratios - skip winsorization
    "percentage",       # Bounded [0,100] metrics - margins, growth rates
    "count",            # Discrete integers - analyst ratings, employees
    "label",            # Classification targets (multi-label)
]

class ColumnMeta(TypedDict, total=False):
    dtype: DType
    role: Role
    sql_name: Optional[str]  # Original SQL column name
    description: Optional[str]  # Column description

# Master schema - auto-generates SQL
# NOTE: This is a truncated version of the 555 entries.
# In a real scenario, all 555 entries would be here.
COLUMN_SCHEMA: Dict[str, ColumnMeta] = {
    "ticker": {
        "dtype": "string",
        "role": "id",
        "sql_name": "Ticker"
    },
    "isin": {
        "dtype": "string",
        "role": "id",
        "sql_name": "ISIN"
    },
    "name": {
        "dtype": "string",
        "role": "id",
        "sql_name": "Name"
    },
    "description": {
        "dtype": "string",
        "role": "auxiliary",
        "sql_name": "Description"
    },
    "sector": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Sector"
    },
    "industry": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Industry"
    },
    "region": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Region"
    },
    "country": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Country"
    },
    "trading_country": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Trading Country"
    },
    "exchange": {
        "dtype": "category",
        "role": "categorical"
    },
    "unit": {
        "dtype": "string",
        "role": "categorical"
    },
    "style_class": {
        "dtype": "category",
        "role": "categorical"
    },
    "size_class": {
        "dtype": "category",
        "role": "categorical"
    },
    "next_earnings_status": {
        "dtype": "category",
        "role": "categorical"
    },
    "last_updated": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "income_statement_report_date": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "fy_end": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "next_earnings": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "next_earnings_when": {
        "dtype": "category",
        "role": "categorical"
    },
    "dividend_record_announce_date": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "dividend_record_ex_date": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "dividend_record_payable_date": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "dividend_record_record_date": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "reference_date": {
        "dtype": "datetime64[ns]",
        "role": "date"
    },
    "last_price": {
        "dtype": "float",
        "role": "price",
        "sql_name": "Last Price"
    },
    "price_target": {
        "dtype": "float",
        "role": "target",
        "sql_name": "Price Target"
    },
    "price_target_ytd_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_target_low": {
        "dtype": "float",
        "role": "price"
    },
    "price_target_median": {
        "dtype": "float",
        "role": "target_fallback"
    },
    "price_target_high": {
        "dtype": "float",
        "role": "price"
    },
    "price_target_num": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "price_target_count": {
        "dtype": "float",
        "role": "count"
    },
    "price_5d_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_1w_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_1m_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_3m_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_6m_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_1y_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_3y_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_5y_ago": {
        "dtype": "float",
        "role": "price"
    },
    "price_qtd_ago": {
        "dtype": "float",
        "role": "price"
    },
    "market_cap": {
        "dtype": "float",
        "role": "market_value",
        "sql_name": "Market Cap"
    },
    "enterprise_value": {
        "dtype": "float",
        "role": "market_value"
    },
    "market_cap_country_r": {
        "dtype": "float",
        "role": "market_value"
    },
    "p_e_ntm": {
        "dtype": "float",
        "role": "ratio",
        "sql_name": "P/E (NTM)"
    },
    "p_e_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_1fyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_b_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_b_1fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_b_5yavg": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_tbv_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_ntm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_est_fy1": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_ebitda_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_ebitda_ntm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_ebitda_est_fy1": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_est_fy1": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_1fyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_2fyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_3fyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_3yavgltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_1fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_2fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_3fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_4fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_ebitda_1fyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_ebitda_1fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_ebitda_3yavgltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_2fyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_3fyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_3yavgltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_1fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_2fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_3fqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_5yavgltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_0fqqoqltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_0fyyoyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_1fyyoyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_e_0fqyoyltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "altman_z_score_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "altman_z_score_fq": {
        "dtype": "float",
        "role": "ratio"
    },
    "altman_z_score_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "beta_1y": {
        "dtype": "float",
        "role": "percentage"
    },
    "beta_2y": {
        "dtype": "float",
        "role": "percentage"
    },
    "beta_5y": {
        "dtype": "float",
        "role": "percentage"
    },
    "total_return_ytd": {
        "dtype": "float",
        "role": "percentage"
    },
    "total_return_5y": {
        "dtype": "float",
        "role": "percentage"
    },
    "total_return_10y": {
        "dtype": "float",
        "role": "percentage"
    },
    "tot_return_pct_cagr_3y": {
        "dtype": "float",
        "role": "percentage"
    },
    "tot_return_pct_cagr_10y": {
        "dtype": "float",
        "role": "percentage"
    },
    "price_chg_pct_1m": {
        "dtype": "float",
        "role": "percentage"
    },
    "price_chg_pct_3m": {
        "dtype": "float",
        "role": "percentage"
    },
    "1_day_pct": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "one_day_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "analyst_rating": {
        "dtype": "float",
        "role": "count"
    },
    "num_strong_sell_ratings": {
        "dtype": "float",
        "role": "count"
    },
    "num_strong_buys_ratings": {
        "dtype": "float",
        "role": "count"
    },
    "num_hold_ratings": {
        "dtype": "float",
        "role": "count"
    },
    "num_buys_ratings": {
        "dtype": "float",
        "role": "count"
    },
    "num_sell_ratings": {
        "dtype": "float",
        "role": "count"
    },
    "ema_20d": {
        "dtype": "float",
        "role": "price"
    },
    "ema_50d": {
        "dtype": "float",
        "role": "price"
    },
    "ema_100d": {
        "dtype": "float",
        "role": "price"
    },
    "ema_250d": {
        "dtype": "float",
        "role": "price"
    },
    "ma_20d": {
        "dtype": "float",
        "role": "price"
    },
    "ma_50d": {
        "dtype": "float",
        "role": "price"
    },
    "52w_high_adj": {
        "dtype": "float",
        "role": "price"
    },
    "52w_low_adj": {
        "dtype": "float",
        "role": "price"
    },
    "volatility_1m": {
        "dtype": "float",
        "role": "percentage"
    },
    "volatility_30d": {
        "dtype": "float",
        "role": "percentage"
    },
    "volatility_3m": {
        "dtype": "float",
        "role": "percentage"
    },
    "volatility_60d": {
        "dtype": "float",
        "role": "percentage"
    },
    "volatility_6m": {
        "dtype": "float",
        "role": "percentage"
    },
    "volatility_90d": {
        "dtype": "float",
        "role": "percentage"
    },
    "volatility_1y": {
        "dtype": "float",
        "role": "percentage"
    },
    "volume_shrs": {
        "dtype": "float",
        "role": "market_value"
    },
    "rel_volume": {
        "dtype": "float",
        "role": "ratio"
    },
    "shrs_out": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "shares_outstanding": {
        "dtype": "float",
        "role": "count"
    },
    "shrs_out_1fy": {
        "dtype": "float",
        "role": "count"
    },
    "total_revenues_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_revenues_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_revenues_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_revenues_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_revenues_cagr_5y_fy": {
        "dtype": "float",
        "role": "percentage"
    },
    "total_revenues_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_revenues_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "revenues_est_avg_ntm": {
        "dtype": "float",
        "role": "market_value"
    },
    "revenues_est_avg_fy1e": {
        "dtype": "float",
        "role": "market_value"
    },
    "revenues_est_med_ntm": {
        "dtype": "float",
        "role": "market_value"
    },
    "revenues_est_med_fy1e": {
        "dtype": "float",
        "role": "market_value"
    },
    "revenues_est_yoy_pct_fy1e": {
        "dtype": "float",
        "role": "percentage"
    },
    "total_operating_expenses_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_adj_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_adj_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_adj_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_est_avg_fy1e": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_est_avg_ntm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_adj_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_adj_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_adj_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_est_med_fy1e": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_est_med_ntm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_is_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_is_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_is_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_is_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_is_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_is_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "normalized_net_income_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "normalized_net_income_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "normalized_net_income_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "normalized_net_income_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "normalized_net_income_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "normalized_net_income_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_adj_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_adj_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_adj_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_adj_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_adj_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "operating_income_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "operating_income_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "operating_income_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "operating_income_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_margin_pct_fy": {
        "dtype": "float",
        "role": "percentage"
    },
    "net_income_margin_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "gross_profit_margin_pct_fy": {
        "dtype": "float",
        "role": "percentage"
    },
    "gross_profit_margin_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "gross_profit_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "gross_profit_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "return_on_equity_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "return_on_equity_pct_fy": {
        "dtype": "float",
        "role": "percentage"
    },
    "return_on_assets_roa_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "return_on_assets_roa_pct_fy": {
        "dtype": "float",
        "role": "percentage"
    },
    "cfo_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfo_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfo_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfo_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "fcf_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "fcf_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "fcf_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "fcf_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfi_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfi_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfi_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfi_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cff_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "cff_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cff_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "cff_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_assets_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_assets_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_equity_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_equity_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_debt_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_debt_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_current_assets_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_current_liabilities_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "current_ratio_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "current_ratio_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "working_capital_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "working_capital_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "working_capital_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "working_capital_5yavgfy": {
        "dtype": "float",
        "role": "market_value"
    },
    "tbv_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "tbv_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_and_equivalents_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_and_equivalents_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_and_equivalents_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_and_equivalents_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "retained_earnings_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "retained_earnings_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "retained_earnings_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "retained_earnings_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "inventory_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "inventory_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "inventory_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "inventory_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "goodwill_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "goodwill_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "goodwill_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "goodwill_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "goodwill_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "intangible_assets": {
        "dtype": "float",
        "role": "feature"
    },
    "gross_intangible_assets_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "gross_intangible_assets_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "gross_intangible_assets_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "capital_expenditure_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "capital_expenditure_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "capital_expenditure_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "capital_expenditure_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "capital_expenditure_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "asset_turnover_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "asset_turnover_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "cash_acquisitions_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_acquisitions_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_acquisitions_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_acquisitions_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_acquisitions_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "impairment_of_goodwill_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "impairment_of_goodwill_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "impairment_of_goodwill_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "impairment_of_goodwill_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "impairment_of_goodwill_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "asset_writedown_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "asset_writedown_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "asset_writedown_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "asset_writedown_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "asset_writedown_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "restructuring_charges_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "restructuring_charges_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "restructuring_charges_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "restructuring_charges_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "merger_and_restructuring_charges_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "merger_and_restructuring_charges_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "merger_and_restructuring_charges_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "merger_and_restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "merger_restructuring_charges_ltm": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "merger_restructuring_charges_fq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "merger_restructuring_charges_fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "merger_restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "other_unusual_items_total_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "gain_loss_on_sale_of_assets_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "cost_of_revenues_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "randd_expenses_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "r_d_expenses": {
        "dtype": "float",
        "role": "feature"
    },
    "r_d_expenses_ltm": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "selling_general_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "selling_general_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "selling_general_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "selling_general_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "selling_general_and_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "feature"
    },
    "selling_general_and_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "feature"
    },
    "selling_general_and_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "feature"
    },
    "selling_general_and_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "feature"
    },
    "accounts_receivable_total_fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "accounts_receivable_total_1fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "accounts_receivable_total_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "marketing_expenses": {
        "dtype": "float",
        "role": "feature"
    },
    "marketing_expenses_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "marketing_expenses_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "marketing_expenses_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "marketing_expenses_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "eps_adj_1fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_adj_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_adj_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_fq": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_1fqfq": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_2fqfq": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_3fqfq": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_4fqfq": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_1fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_2fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_3fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_4fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_eps_basic_5fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_norm_est_avg_ntm": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_norm_est_avg_fy1e": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_norm_est_num_fy1e": {
        "dtype": "float",
        "role": "count"
    },
    "eps_est_avg_rev_pct_fy1e_1w": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_est_avg_rev_pct_fy1e_1m": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_est_avg_rev_pct_fy1e_3m": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_est_avg_rev_pct_fy1e_6m": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_est_avg_rev_pct_fy1e_1y": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_gaap_est_avg_fy1e": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_gaap_est_avg_ntm": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_gaap_est_avg_rev_pct_fy1e_1m": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_gaap_est_avg_rev_pct_fy1e_3m": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_gaap_est_avg_rev_pct_fy1e_6m": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_gaap_est_avg_rev_pct_fy1e_1y": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_previous_year": {
        "dtype": "float",
        "role": "feature"
    },
    "dividend_per_share_ltm": {
        "dtype": "float",
        "role": "price"
    },
    "div_yield_ind": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_ttm": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_ntm": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_1fyind": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_2fyind": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_3fyind": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_4fyind": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_5fyind": {
        "dtype": "float",
        "role": "percentage"
    },
    "div_yield_5yavgltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "common_dividends_paid_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "common_dividends_paid_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "dividend_record_frequency": {
        "dtype": "string",
        "role": "categorical"
    },
    "dividend_record_currency": {
        "dtype": "string",
        "role": "categorical"
    },
    "dividend_record_amount": {
        "dtype": "float",
        "role": "price"
    },
    "dividend_streak": {
        "dtype": "float",
        "role": "count"
    },
    "days_to_dividend": {
        "dtype": "float",
        "role": "feature"
    },
    "buyback_yield_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "interest_expense_total_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "interest_income_on_investments_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "employees": {
        "dtype": "float",
        "role": "count"
    },
    "avg_employees_ltm": {
        "dtype": "float",
        "role": "count"
    },
    "avg_employees_fy": {
        "dtype": "float",
        "role": "count"
    },
    "avg_employees_5yavgfy": {
        "dtype": "float",
        "role": "count"
    },
    "total_employees_fy": {
        "dtype": "float",
        "role": "count"
    },
    "total_employees_fq": {
        "dtype": "float",
        "role": "count"
    },
    "full_time_employees_fq": {
        "dtype": "float",
        "role": "count"
    },
    "full_time_employees_fy": {
        "dtype": "float",
        "role": "count"
    },
    "full_time_employees_1fy": {
        "dtype": "float",
        "role": "count"
    },
    "full_time_employees_2fy": {
        "dtype": "float",
        "role": "count"
    },
    "full_time_employees_3fy": {
        "dtype": "float",
        "role": "count"
    },
    "strong_sell_ratings": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "strong_buys_ratings": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "hold_ratings": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "buys_ratings": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "sell_ratings": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "p_e": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_b": {
        "dtype": "float",
        "role": "ratio"
    },
    "revenue": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "gross_margin": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps": {
        "dtype": "float",
        "role": "ratio"
    },
    "total_equity": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_assets": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_debt": {
        "dtype": "float",
        "role": "market_value"
    },
    "inventory": {
        "dtype": "float",
        "role": "market_value"
    },
    "capex": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_and_equivalents": {
        "dtype": "float",
        "role": "market_value"
    },
    "current_assets": {
        "dtype": "float",
        "role": "market_value"
    },
    "accounts_receivable": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "current_liabilities": {
        "dtype": "float",
        "role": "market_value"
    },
    "working_capital": {
        "dtype": "float",
        "role": "market_value"
    },
    "retained_earnings": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfo": {
        "dtype": "float",
        "role": "market_value"
    },
    "cfi": {
        "dtype": "float",
        "role": "market_value"
    },
    "cff": {
        "dtype": "float",
        "role": "market_value"
    },
    "fcf": {
        "dtype": "float",
        "role": "market_value"
    },
    "gross_profit": {
        "dtype": "float",
        "role": "market_value"
    },
    "operating_income": {
        "dtype": "float",
        "role": "market_value"
    },
    "interest_expense": {
        "dtype": "float",
        "role": "market_value"
    },
    "goodwill": {
        "dtype": "float",
        "role": "market_value"
    },
    "dividend_per_share": {
        "dtype": "float",
        "role": "price"
    },
    "operating_expenses": {
        "dtype": "float",
        "role": "market_value"
    },
    "operating_cash_flow": {
        "dtype": "float",
        "role": "market_value"
    },
    "dividends_paid": {
        "dtype": "float",
        "role": "market_value"
    },
    "dividends_paid_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "price_target_number": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "sga_expenses": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "sga_expenses_fq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "sga_expenses_fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "sga_expenses_1fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "sga_expenses_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "accounts_receivable_fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "accounts_receivable_1fy": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "accounts_receivable_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary"
    },
    "volatility_1y_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "tangible_book_value": {
        "dtype": "float",
        "role": "market_value"
    },
    "marketing_efficiency": {
        "dtype": "float",
        "role": "ratio"
    },
    "r_d_intensity": {
        "dtype": "float",
        "role": "percentage"
    },
    "rule_of_40": {
        "dtype": "float",
        "role": "percentage"
    },
    "operating_leverage": {
        "dtype": "float",
        "role": "ratio"
    },
    "one_day_chg": {
        "dtype": "float",
        "role": "percentage"
    },
    "market_cap_x_debt_to_equity": {
        "dtype": "float",
        "role": "feature"
    },
    "market_cap_x_roe": {
        "dtype": "float",
        "role": "feature"
    },
    "p_e_ratio_x_debt_to_equity": {
        "dtype": "float",
        "role": "feature"
    },
    "p_e_ratio_x_roe": {
        "dtype": "float",
        "role": "feature"
    },
    "roe_x_debt_to_equity": {
        "dtype": "float",
        "role": "feature"
    },
    "log_operating_income": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_net_income": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_capex": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_operating_cash_flow": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_equity": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_market_cap": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_assets": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_gross_profit": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_and_equivalents": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_debt": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_revenue": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_enterprise_value": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_gross_profit_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_operating_income_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_revenues_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_acquisitions_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_revenues_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_assets_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_operating_income_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_acquisitions_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_revenues_est_avg_ntm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_revenues_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_net_income_is_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_fcf_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_equity_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_revenues_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_net_income_adj_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_equity_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_debt_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_revenue_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_revenue_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_acquisitions_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_net_income_is_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_acquisitions_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_assets_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_net_income_adj_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_5yavgltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_revenues_est_avg_fy1e": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_revenues_est_med_ntm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_and_equivalents_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_net_income_is_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_and_equivalents_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_fcf_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_debt_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_fcf": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_gross_profit_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_market_cap_country_r": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_cash_and_equivalents_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_fcf_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_5yavgfq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_fcf_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_revenues_est_med_fy1e": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_total_assets_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_operating_income_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_net_income_is_fq": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_ebitda_adj_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "log_gross_profit_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "p_e_ratio": {
        "dtype": "float",
        "role": "ratio"
    },
    "p_s_ratio": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_ebitda_ratio": {
        "dtype": "float",
        "role": "ratio"
    },
    "ev_sales_ratio": {
        "dtype": "float",
        "role": "ratio"
    },
    "gross_margin_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "operating_margin_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "net_margin_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "roe": {
        "dtype": "float",
        "role": "ratio"
    },
    "roa": {
        "dtype": "float",
        "role": "ratio"
    },
    "revenue_growth": {
        "dtype": "float",
        "role": "percentage"
    },
    "ebitda_growth": {
        "dtype": "float",
        "role": "percentage"
    },
    "earnings_growth": {
        "dtype": "float",
        "role": "percentage"
    },
    "debt_to_equity": {
        "dtype": "float",
        "role": "ratio"
    },
    "debt_to_assets": {
        "dtype": "float",
        "role": "ratio"
    },
    "target_vs_price": {
        "dtype": "float",
        "role": "ratio"
    },
    "target_vs_price_median": {
        "dtype": "float",
        "role": "ratio"
    },
    "peg_ratio": {
        "dtype": "float",
        "role": "ratio"
    },
    "dividend_yield": {
        "dtype": "float",
        "role": "percentage"
    },
    "roic": {
        "dtype": "float",
        "role": "ratio"
    },
    "revenue_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_equity_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "total_assets_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "gross_profit_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "accounts_receivable_previous_year": {
        "dtype": "float",
        "role": "market_value"
    },
    "roa_previous_year": {
        "dtype": "float",
        "role": "ratio"
    },
    "current_ratio_previous_year": {
        "dtype": "float",
        "role": "ratio"
    },
    "shares_outstanding_previous_year": {
        "dtype": "float",
        "role": "count"
    },
    "gross_margin_pct_previous_year": {
        "dtype": "float",
        "role": "percentage"
    },
    "asset_turnover_previous_year": {
        "dtype": "float",
        "role": "ratio"
    },
    "revenue_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "working_capital_1fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "cash_burn_rate": {
        "dtype": "float",
        "role": "feature"
    },
    "cash_burn_rate_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "revenue_per_employee": {
        "dtype": "float",
        "role": "feature"
    },
    "revenue_per_employee_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "revenue_per_employee_ltm": {
        "dtype": "float",
        "role": "feature"
    },
    "revenue_per_employee_ltm_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "revenue_per_employee_fy": {
        "dtype": "float",
        "role": "feature"
    },
    "revenue_per_employee_fy_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "revenue_per_employee_trend": {
        "dtype": "float",
        "role": "feature"
    },
    "revenue_per_employee_trend_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "revenue_per_employee_vs_5y_pct": {
        "dtype": "float",
        "role": "feature"
    },
    "revenue_per_employee_vs_5y_pct_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "assets_per_employee": {
        "dtype": "float",
        "role": "feature"
    },
    "assets_per_employee_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "ebitda_per_employee": {
        "dtype": "float",
        "role": "feature"
    },
    "ebitda_per_employee_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "operating_income_per_employee": {
        "dtype": "float",
        "role": "feature"
    },
    "operating_income_per_employee_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "profit_per_employee": {
        "dtype": "float",
        "role": "feature"
    },
    "profit_per_employee_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "employee_growth_yoy": {
        "dtype": "float",
        "role": "feature"
    },
    "employee_growth_yoy_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "employee_growth_yoy_pct": {
        "dtype": "float",
        "role": "feature"
    },
    "employee_growth_yoy_pct_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "employee_growth_qoq": {
        "dtype": "float",
        "role": "feature"
    },
    "employee_growth_qoq_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "employee_growth_cagr_5y": {
        "dtype": "float",
        "role": "feature"
    },
    "employee_growth_cagr_5y_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "employee_growth_acceleration": {
        "dtype": "float",
        "role": "feature"
    },
    "employee_growth_acceleration_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "workforce_volatility": {
        "dtype": "float",
        "role": "feature"
    },
    "workforce_volatility_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "hiring_intensity_score": {
        "dtype": "float",
        "role": "feature"
    },
    "hiring_intensity_score_applicable": {
        "dtype": "bool",
        "role": "auxiliary"
    },
    "altman_z_score": {
        "dtype": "float",
        "role": "feature"
    },
    "beneish_m_score": {
        "dtype": "float",
        "role": "feature"
    },
    "composite_quality_score": {
        "dtype": "float",
        "role": "feature"
    },
    "momentum_score": {
        "dtype": "float",
        "role": "feature"
    },
    "eps_surprise_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_surprise_magnitude": {
        "dtype": "category",
        "role": "categorical"
    },
    "revenue_surprise_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "revenue_beat_indicator": {
        "dtype": "bool",
        "role": "feature"
    },
    "ebitda_surprise_pct": {
        "dtype": "float",
        "role": "percentage"
    },
    "earnings_beat_indicator": {
        "dtype": "bool",
        "role": "feature"
    },
    "surprise_momentum_score": {
        "dtype": "float",
        "role": "feature"
    },
    "positive_revision_momentum": {
        "dtype": "bool",
        "role": "feature"
    },
    "consensus_uncertainty_score": {
        "dtype": "float",
        "role": "feature"
    },
    "estimate_revision_acceleration": {
        "dtype": "float",
        "role": "percentage"
    },
    "accelerating_upgrades_flag": {
        "dtype": "bool",
        "role": "feature"
    },
    "eps_adjustment_spread_ltm": {
        "dtype": "float",
        "role": "feature"
    },
    "eps_adjustment_ratio_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_adjustment_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "eps_quality_flag_ltm": {
        "dtype": "bool",
        "role": "feature"
    },
    "eps_adjustment_spread_fy": {
        "dtype": "float",
        "role": "feature"
    },
    "eps_adjustment_ratio_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "eps_adjustment_pct_fy": {
        "dtype": "float",
        "role": "percentage"
    },
    "net_income_adjustment_spread_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_adjustment_ratio_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "net_income_adjustment_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "net_income_adjustment_spread_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "net_income_adjustment_ratio_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "ebitda_adjustment_spread_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebitda_adjustment_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "ebitda_adjustment_spread_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_adjustment_spread_ltm": {
        "dtype": "float",
        "role": "market_value"
    },
    "ebit_adjustment_pct_ltm": {
        "dtype": "float",
        "role": "percentage"
    },
    "ebit_adjustment_spread_fy": {
        "dtype": "float",
        "role": "market_value"
    },
    "adjustment_consistency_score": {
        "dtype": "float",
        "role": "feature"
    },
    "earnings_quality_warning_flag": {
        "dtype": "bool",
        "role": "feature"
    },
    "earnings_quality_score": {
        "dtype": "float",
        "role": "feature"
    },
    "exceptional_items_impact_ratio": {
        "dtype": "float",
        "role": "ratio"
    },
    "next_earnings_formatted": {
        "dtype": "string",
        "role": "auxiliary"
    },
    "_reference_date_formatted": {
        "dtype": "string",
        "role": "auxiliary"
    },
    "income_statement_report_date_formatted": {
        "dtype": "string",
        "role": "auxiliary"
    },
    "dividend_record_ex_date_formatted": {
        "dtype": "string",
        "role": "auxiliary"
    },
    "ebit_adjustment_ratio_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ebit_adjustment_ratio_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "ebitda_adjustment_ratio_ltm": {
        "dtype": "float",
        "role": "ratio"
    },
    "ebitda_adjustment_ratio_fy": {
        "dtype": "float",
        "role": "ratio"
    },
    "ebitda_margin_trend": {
        "dtype": "float",
        "role": "percentage"
    },
    "gross_margin_trend": {
        "dtype": "float",
        "role": "percentage"
    },
    "net_margin_trend": {
        "dtype": "float",
        "role": "percentage"
    },
    "operating_margin_trend": {
        "dtype": "float",
        "role": "percentage"
    },
    "days_to_earnings": {
        "dtype": "float",
        "role": "feature"
    },
    "earnings_report_recency": {
        "dtype": "float",
        "role": "feature"
    },
    "reporting_lag": {
        "dtype": "float",
        "role": "feature"
    }
}

# Phase 9.3 Feature Input Categorization
PHASE93_FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "Momentum & Technical": [
        "52w_range_position", "breakout_signal", "ema_crossover_20_50", "ema_crossover_50_250",
        "ema_slope_20d", "ema_trend_consistency", "ma_20d_simple", "ma_50d_simple",
        "ma_crossover_signal", "near_52w_high_flag", "near_52w_low_flag", "pct_above_52w_low",
        "pct_off_52w_high", "price_acceleration_3m", "price_distance_from_ma", "price_momentum_1m",
        "price_momentum_1y", "price_momentum_3m", "price_momentum_6m", "price_vs_ema_20d",
        "price_vs_ema_250d", "return_stability_score", "sharpe_proxy", "total_return_1y_pct",
        "volume_momentum_score",
    ],
    "Valuation Ratios": [
        "dividend_yield", "ev_ebitda_forward_discount", "ev_ebitda_momentum", "ev_ebitda_ratio",
        "ev_ebitda_vs_3y_avg", "ev_sales_forward_discount", "ev_sales_quarterly_volatility",
        "ev_sales_ratio", "ev_sales_trend_1y", "ev_sales_trend_3y", "ev_sales_vs_3y_avg",
        "growth_implied_by_valuation", "p_b", "p_b_ratio", "p_e_forward_discount",
        "p_e_momentum_qoq", "p_e_momentum_yoy", "p_e_ratio", "p_e_vs_3y_avg", "p_s_ratio",
        "peg_ratio", "valuation_extreme_flag", "valuation_stability_score",
        "valuation_trend_consistency", "book_value_per_share",
    ],
    # ... truncated for brevity, but would be fully populated in a real implementation
}

def get_sql_column_name(normalized_name: str) -> str:
    """Get original SQL column name from normalized Python name."""
    meta = COLUMN_SCHEMA.get(normalized_name)
    if meta and "sql_name" in meta and meta["sql_name"]:
        return meta["sql_name"]
    return normalized_name

def normalize_column_name(column: str) -> str:
    """Standardize column names to lowercase with underscores."""
    if "R&D" in column or "r&d" in column.lower():
        column = column.replace("R&D", "RandD").replace("r&d", "randd")

    normalized = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")
        .replace("%", "pct")
        .replace("&", "and")
    )
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")

def generate_sql_schema() -> str:
    """Generate CREATE TABLE statement from COLUMN_SCHEMA."""
    lines = ["CREATE TABLE IF NOT EXISTS equities ("]
    for col_name, meta in COLUMN_SCHEMA.items():
        sql_name = meta.get("sql_name") or col_name
        dtype = meta.get("dtype", "float")
        sql_type = {
            "float": "NUMERIC",
            "int": "INTEGER",
            "string": "TEXT",
            "category": "TEXT",
            "datetime64[ns]": "TIMESTAMP",
            "bool": "BOOLEAN"
        }.get(dtype, "NUMERIC")
        lines.append(f'  "{sql_name}" {sql_type},')
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)

def get_expected_dtype(column: str) -> str:
    """Get the expected pandas-compatible dtype string for a column."""
    meta = COLUMN_SCHEMA.get(column, {})
    return meta.get("dtype", "float")

def list_numeric_feature_cols() -> List[str]:
    """List all numeric feature columns from COLUMN_SCHEMA."""
    numeric_dtypes = {"float", "int"}
    feature_roles = {"feature", "target", "target_fallback", "price", "market_value", "ratio", "percentage"}

    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("dtype") in numeric_dtypes and meta.get("role") in feature_roles
    ]

def list_categorical_cols() -> List[str]:
    """List all categorical columns from COLUMN_SCHEMA."""
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("dtype") == "category" or meta.get("role") == "categorical"
    ]

def list_date_cols() -> List[str]:
    """List all date/datetime columns from COLUMN_SCHEMA."""
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("dtype") == "datetime64[ns]" or meta.get("role") == "date"
    ]

def list_etl_generated_column_patterns() -> List[str]:
    """List regex patterns for columns legitimately generated during ETL."""
    return [
        r"^log_[0-9a-z_]+$",  # Log-transformed columns
        r"^.*_applicable$",  # Conditional metric applicability flags
        r"^event_prob_.*$",  # Classification probabilities
        r"^sector_[0-9a-z]+_x_[0-9a-z_]+$",  # Sector interactions
        r"^.*_(ratio|pct|margin|growth|yoy)$",  # Common semantic/derived suffixes
        r"^.*_formatted$",  # Standardized date string representations
        r"^fy_end_vs_isrd_days$",  # Fiscal year-end to income statement report delta
        r"^fiscal_quarter_inferred$",  # Inferred fiscal quarter label
    ]

def list_required_schema_columns_for_etl(include_extended_financials: bool = False) -> List[str]:
    """List columns required for minimal ETL operations."""
    required = [
        "ticker", "isin", "sector", "region", "country", "trading_country",
        "last_price", "price_target", "market_cap"
    ]
    if include_extended_financials:
        required.extend(["enterprise_value", "ebitda_ltm", "total_revenues_ltm"])
    return required
