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
    "float",  # float64 - default for numeric financial data (use Float64 for nullable)
    "Float64",  # pandas nullable float - use when pd.NA may be present
    "int",  # int64 - discrete counts, integer IDs
    "Int64",  # pandas nullable int - use when pd.NA may be present
    "string",  # object/string - text data
    "category",  # pandas Categorical - low-cardinality (sector, region)
    "datetime64[ns]",  # datetime columns
    "bool",  # boolean flags
    "boolean",  # pandas nullable boolean - use when pd.NA may be present
]

# Role: Semantic role determining preprocessing and pipeline treatment
Role = Literal[
    "id",  # Identifier columns (ticker, isin, name, description) - never used as features
    "target",  # Primary prediction target (price_target)
    "target_fallback",  # Alternative targets (price_target_median, last_price)
    "date",  # Temporal columns for time-series features
    "categorical",  # Grouping columns (sector, region, industry, exchange, unit, country, trading_country)
    "feature",  # General ML features from phase_93
    "market",  # Market/trading data (price, volume, market cap, dividends, shares outstanding)
    "financial_statement",  # P&L line items (revenues, expenses, recurring/non-recurring items)
    "balance_sheet",  # Balance sheet items (assets, liabilities, equity, working capital)
    "cash_flow",  # Cash flow statement items (CFO, CFI, CFF, FCF, capex)
    "ratio",  # Pre-normalized ratios (P/E, P/B, EV/EBITDA, ROE, ROA, EPS, turnover)
    "percentage",  # Bounded [0-100] metrics (margins, growth rates, returns, volatility, beta)
    "count",  # Discrete integers (analyst ratings, employees, shares, dividend streak)
    "auxiliary",  # Legacy aliases, optional - excluded from diagnostics
    "label",  # Classification targets (multi-label)
    "non_recurring",  # Non-recurring exceptional items (impairments, restructuring) - zero imputation
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
    "ticker": {"dtype": "string", "role": "id", "sql_name": "Ticker"},
    "isin": {"dtype": "string", "role": "id", "sql_name": "ISIN"},
    "name": {"dtype": "string", "role": "id", "sql_name": "Name"},
    "description": {"dtype": "string", "role": "auxiliary", "sql_name": "Description"},
    "sector": {"dtype": "category", "role": "categorical", "sql_name": "Sector"},
    "industry": {"dtype": "category", "role": "categorical", "sql_name": "Industry"},
    "region": {"dtype": "category", "role": "categorical", "sql_name": "Region"},
    "country": {"dtype": "category", "role": "categorical", "sql_name": "Country"},
    "trading_country": {"dtype": "category", "role": "categorical", "sql_name": "Trading Country"},
    "exchange": {"dtype": "category", "role": "categorical", "sql_name": "Exchange"},
    "unit": {"dtype": "category", "role": "categorical", "sql_name": "Unit"},
    "style_class": {"dtype": "category", "role": "categorical", "sql_name": "Style Class"},
    "size_class": {"dtype": "category", "role": "categorical", "sql_name": "Size Class"},
    "next_earnings_status": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Earnings (Status)",
    },
    "next_earnings_report": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Earnings (Report)",
        "description": "Next earnings report type (Full Year/Interim)",
    },
    "last_updated": {"dtype": "datetime64[ns]", "role": "date", "sql_name": "Last Updated"},
    "income_statement_report_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Income Statement Report Date",
    },
    "fy_end": {"dtype": "category", "role": "categorical", "sql_name": "FY End"},
    "fy_end_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "FY End Date",
        "description": "Fiscal year end date (parsed from FY End text)",
    },
    "next_fy_end_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Next FY End Date",
        "description": "Next fiscal year end date",
    },
    "fiscal_month": {
        "dtype": "Int64",
        "role": "feature",
        "sql_name": "Fiscal Month",
        "description": "Months between Income Statement Report Date and FY End Date",
    },
    "fiscal_quarter": {
        "dtype": "Int64",
        "role": "feature",
        "sql_name": "Fiscal Quarter",
        "description": "Fiscal quarter (1-4) from report date",
    },
    "fiscal_year": {
        "dtype": "Int64",
        "role": "feature",
        "sql_name": "Fiscal Year",
        "description": "Fiscal year from report date",
    },
    "current_fiscal_quarter": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Current Fiscal Quarter",
        "description": "Current fiscal quarter (formatted as Q4 2025)",
    },
    "next_fiscal_quarter": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Fiscal Quarter",
        "description": "Next fiscal quarter (formatted as Q4 2025)",
    },
    "next_earnings": {"dtype": "datetime64[ns]", "role": "date", "sql_name": "Next Earnings"},
    "next_earnings_when": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Earnings (When)",
    },
    "dividend_record_announce_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Announce Date)",
    },
    "dividend_record_ex_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Ex Date)",
    },
    "dividend_record_payable_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Payable Date)",
    },
    "dividend_record_record_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Record Date)",
    },
    "reference_date": {"dtype": "datetime64[ns]", "role": "date", "sql_name": "Reference Date"},
    "last_price": {"dtype": "Float64", "role": "market", "sql_name": "Last Price"},
    "price_target": {"dtype": "Float64", "role": "target", "sql_name": "Price Target"},
    "price_target_ytd_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price Target (YTD Ago)",
    },
    "price_target_low": {"dtype": "Float64", "role": "market", "sql_name": "Price Target - Low"},
    "price_target_median": {
        "dtype": "Float64",
        "role": "target_fallback",
        "sql_name": "Price Target - Median",
    },
    "price_target_high": {"dtype": "Float64", "role": "market", "sql_name": "Price Target - High"},
    "price_target_num": {"dtype": "float", "role": "count", "sql_name": "Price Target - #"},
    "price_target_count": {"dtype": "float", "role": "count", "sql_name": "Price Target - #"},
    "price_5d_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (5D Ago)"},
    "price_1w_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (1W Ago)"},
    "price_1m_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (1M Ago)"},
    "price_3m_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (3M Ago)"},
    "price_6m_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (6M Ago)"},
    "price_1y_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (1Y Ago)"},
    "price_3y_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (3Y Ago)"},
    "price_5y_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (5Y Ago)"},
    "price_qtd_ago": {"dtype": "Float64", "role": "market", "sql_name": "Price (QTD Ago)"},
    "market_cap": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Market Cap",
        "description": "Market capitalization",
    },
    "enterprise_value": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Enterprise Value",
        "description": "Enterprise value",
    },
    "market_cap_country_r": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Market Cap (Country R)",
    },
    "p_e_ntm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (NTM)"},
    "p_e_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (LTM)"},
    "p_e_1fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-1FYLTM)"},
    "p_b_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/B (LTM)"},
    "p_b_1fy": {"dtype": "Float64", "role": "ratio", "sql_name": "P/B (-1FY)"},
    "p_b_5yavg": {"dtype": "Float64", "role": "ratio", "sql_name": "P/B (5YAVG)"},
    "p_tbv_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/TBV (LTM)"},
    "ev_sales_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (LTM)"},
    "ev_sales_ntm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (NTM)"},
    "ev_sales_est_fy1": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (EST FY1)"},
    "ev_ebitda_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/EBITDA (LTM)"},
    "ev_ebitda_ntm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/EBITDA (NTM)"},
    "ev_ebitda_est_fy1": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/EBITDA (EST FY1)"},
    "p_e_est_fy1": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (EST FY1)"},
    "ev_sales_1fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (-1FYLTM)"},
    "ev_sales_2fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (-2FYLTM)"},
    "ev_sales_3fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (-3FYLTM)"},
    "ev_sales_3yavgltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (3YAVGLTM)"},
    "ev_sales_1fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (-1FQLTM)"},
    "ev_sales_2fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (-2FQLTM)"},
    "ev_sales_3fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (-3FQLTM)"},
    "ev_sales_4fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (-4FQLTM)"},
    "ev_ebitda_1fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/EBITDA (-1FYLTM)"},
    "ev_ebitda_1fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/EBITDA (-1FQLTM)"},
    "ev_ebitda_3yavgltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/EBITDA (3YAVGLTM)"},
    "p_e_2fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-2FYLTM)"},
    "p_e_3fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-3FYLTM)"},
    "p_e_3yavgltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (3YAVGLTM)"},
    "p_e_1fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-1FQLTM)"},
    "p_e_2fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-2FQLTM)"},
    "p_e_3fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-3FQLTM)"},
    "p_e_5yavgltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (5YAVGLTM)"},
    "p_e_0fqqoqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-0FQQoQLTM)"},
    "p_e_0fyyoyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-0FYYoYLTM)"},
    "p_e_1fyyoyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-1FYYoYLTM)"},
    "p_e_0fqyoyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-0FQYoYLTM)"},
    "altman_z_score_fy": {"dtype": "Float64", "role": "ratio", "sql_name": "Altman Z-Score (FY)"},
    "altman_z_score_fq": {"dtype": "Float64", "role": "ratio", "sql_name": "Altman Z-Score (FQ)"},
    "altman_z_score_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "Altman Z-Score (LTM)"},
    "beta_1y": {"dtype": "Float64", "role": "percentage", "sql_name": "Beta (1Y)"},
    "beta_2y": {"dtype": "Float64", "role": "percentage", "sql_name": "Beta (2Y)"},
    "beta_5y": {"dtype": "Float64", "role": "percentage", "sql_name": "Beta (5Y)"},
    "total_return_ytd": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Total Return (YTD)",
    },
    "total_return_5y": {"dtype": "Float64", "role": "percentage", "sql_name": "Total Return (5Y)"},
    "total_return_10y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Total Return (10Y)",
    },
    "tot_return_pct_cagr_3y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Tot. Return %/CAGR (3Y)",
    },
    "tot_return_pct_cagr_10y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Tot. Return %/CAGR (10Y)",
    },
    "price_chg_pct_1m": {"dtype": "Float64", "role": "percentage", "sql_name": "Price Chg. % (1M)"},
    "price_chg_pct_3m": {"dtype": "Float64", "role": "percentage", "sql_name": "Price Chg. % (3M)"},
    "1_day_pct": {"dtype": "Float64", "role": "percentage", "sql_name": "1-Day %"},
    "one_day_pct": {"dtype": "Float64", "role": "percentage", "sql_name": "1-Day %"},
    "analyst_rating": {"dtype": "float", "role": "count", "sql_name": "Analyst Rating"},
    "num_strong_sell_ratings": {
        "dtype": "float",
        "role": "count",
        "sql_name": "# Strong Sell Ratings",
    },
    "num_strong_buys_ratings": {
        "dtype": "float",
        "role": "count",
        "sql_name": "# Strong Buys Ratings",
    },
    "num_hold_ratings": {"dtype": "float", "role": "count", "sql_name": "# Hold Ratings"},
    "num_buys_ratings": {"dtype": "float", "role": "count", "sql_name": "# Buys Ratings"},
    "num_sell_ratings": {"dtype": "float", "role": "count", "sql_name": "# Sell Ratings"},
    "ema_20d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (20D)"},
    "ema_50d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (50D)"},
    "ema_100d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (100D)"},
    "ema_250d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (250D)"},
    "ma_20d_simple": {"dtype": "Float64", "role": "market"},
    "ma_50d_simple": {"dtype": "Float64", "role": "market"},
    "52w_high_adj": {"dtype": "Float64", "role": "market", "sql_name": "52W High/Adj"},
    "52w_low_adj": {"dtype": "Float64", "role": "market", "sql_name": "52W Low/Adj"},
    "volatility_1m": {"dtype": "Float64", "role": "percentage", "sql_name": "Volatility (1M)"},
    "volatility_30d": {"dtype": "Float64", "role": "percentage"},
    "volatility_3m": {"dtype": "Float64", "role": "percentage", "sql_name": "Volatility (3M)"},
    "volatility_60d": {"dtype": "Float64", "role": "percentage"},
    "volatility_6m": {"dtype": "Float64", "role": "percentage", "sql_name": "Volatility (6M)"},
    "volatility_90d": {"dtype": "Float64", "role": "percentage"},
    "volatility_1y": {"dtype": "Float64", "role": "percentage", "sql_name": "Volatility (1Y)"},
    "volume_shrs": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Volume (Shrs)",
        "description": "Trading volume in shares",
    },
    "rel_volume": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Rel. Volume",
        "description": "Relative trading volume ratio",
    },
    "shrs_out": {"dtype": "float", "role": "count", "sql_name": "Shrs Out"},
    "shares_outstanding": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Shrs Out",
        "description": "Shares outstanding",
    },
    "shrs_out_1fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Shrs Out (-1FY)",
        "description": "Shares outstanding (previous FY)",
    },
    "total_revenues_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (FY)",
        "description": "Total revenues (Fiscal Year)",
    },
    "total_revenues_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (LTM)",
        "description": "Total revenues (Last Twelve Months)",
    },
    "total_revenues_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (FQ)",
    },
    "total_revenues_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (-1FY)",
    },
    "total_revenues_cagr_5y_fy": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Total Revenues/CAGR (5Y FY)",
    },
    "total_revenues_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (5YAVGFQ)",
    },
    "total_revenues_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (5YAVGLTM)",
    },
    "revenues_est_avg_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Avg (NTM)",
    },
    "revenues_est_avg_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Avg (FY1E)",
    },
    "revenues_est_med_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Med (NTM)",
    },
    "revenues_est_med_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Med (FY1E)",
    },
    "revenues_est_yoy_pct_fy1e": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Revenues - Est YoY % (FY1E)",
    },
    "total_operating_expenses_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Operating Expenses (LTM)",
    },
    "ebitda_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (FY)",
        "description": "EBITDA (Fiscal Year)",
    },
    "ebitda_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (LTM)",
        "description": "EBITDA (Last Twelve Months)",
    },
    "ebitda_fq": {"dtype": "float", "role": "financial_statement", "sql_name": "EBITDA (FQ)"},
    "ebitda_1fy": {"dtype": "float", "role": "financial_statement", "sql_name": "EBITDA (-1FY)"},
    "ebitda_adj_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (LTM)",
    },
    "ebitda_adj_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (FY)",
    },
    "ebitda_adj_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-1FY)",
    },
    "ebitda_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (5YAVGFQ)",
    },
    "ebitda_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (5YAVGLTM)",
    },
    "ebitda_est_avg_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA - Est Avg (FY1E)",
    },
    "ebitda_est_avg_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA - Est Avg (NTM)",
    },
    "ebit_fy": {"dtype": "float", "role": "financial_statement", "sql_name": "EBIT (FY)"},
    "ebit_ltm": {"dtype": "float", "role": "financial_statement", "sql_name": "EBIT (LTM)"},
    "ebit_fq": {"dtype": "float", "role": "financial_statement", "sql_name": "EBIT (FQ)"},
    "ebit_1fy": {"dtype": "float", "role": "financial_statement", "sql_name": "EBIT (-1FY)"},
    "ebit_adj_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (LTM)",
    },
    "ebit_adj_fy": {"dtype": "float", "role": "financial_statement", "sql_name": "EBIT/Adj. (FY)"},
    "ebit_adj_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-1FY)",
    },
    "ebit_est_med_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT - Est Med (FY1E)",
    },
    "ebit_est_med_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT - Est Med (NTM)",
    },
    "ebit_5yavgfq": {"dtype": "float", "role": "financial_statement", "sql_name": "EBIT (5YAVGFQ)"},
    "ebit_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (5YAVGLTM)",
    },
    "net_income_is_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (FY)",
        "description": "Net income from income statement (Fiscal Year)",
    },
    "net_income_is_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (LTM)",
        "description": "Net income from income statement (Last Twelve Months)",
    },
    "net_income_is_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (FQ)",
    },
    "net_income_is_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-1FY)",
    },
    "net_income_is_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (5YAVGFQ)",
    },
    "net_income_is_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (5YAVGLTM)",
    },
    "normalized_net_income_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (FY)",
    },
    "normalized_net_income_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (LTM)",
    },
    "normalized_net_income_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (FQ)",
    },
    "normalized_net_income_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-1FY)",
    },
    "normalized_net_income_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (5YAVGFQ)",
    },
    "normalized_net_income_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (5YAVGLTM)",
    },
    "net_income_adj_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (FY)",
    },
    "net_income_adj_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (LTM)",
    },
    "net_income_adj_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (FQ)",
    },
    "net_income_adj_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-1FY)",
    },
    "net_income_adj_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (5YAVGFQ)",
    },
    "operating_income_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (LTM)",
    },
    "operating_income_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (FY)",
    },
    "operating_income_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (FQ)",
    },
    "operating_income_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (5YAVGFQ)",
    },
    "net_income_margin_pct_fy": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Net Income Margin % (FY)",
    },
    "net_income_margin_pct_ltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Net Income Margin % (LTM)",
    },
    "gross_profit_margin_pct_fy": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Gross Profit Margin % (FY)",
    },
    "gross_profit_margin_pct_ltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Gross Profit Margin % (LTM)",
    },
    "gross_profit_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Gross Profit (LTM)",
    },
    "gross_profit_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Gross Profit (FY)",
    },
    "return_on_equity_pct_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return On Equity % (LTM)",
        "description": "Return on equity percentage (Last Twelve Months)",
    },
    "return_on_equity_pct_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return On Equity % (FY)",
    },
    "return_on_assets_roa_pct_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return on Assets (ROA) % (LTM)",
        "description": "Return on assets percentage (Last Twelve Months)",
    },
    "return_on_assets_roa_pct_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return on Assets (ROA) % (FY)",
    },
    "cfo_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "CFO (LTM)",
        "description": "Cash from operations (Last Twelve Months)",
    },
    "cfo_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (FY)"},
    "cfo_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (FQ)"},
    "cfo_1fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-1FY)"},
    "fcf_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "FCF (LTM)",
        "description": "Free cash flow (Last Twelve Months)",
    },
    "fcf_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (FY)"},
    "fcf_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (FQ)"},
    "fcf_5yavgfq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (5YAVGFQ)"},
    "cfi_ltm": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (LTM)"},
    "cfi_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (FY)"},
    "cfi_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (FQ)"},
    "cfi_1fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-1FY)"},
    "cff_ltm": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (LTM)"},
    "cff_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (FY)"},
    "cff_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (FQ)"},
    "cff_1fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-1FY)"},
    "total_assets_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Assets (LTM)",
        "description": "Total assets (Last Twelve Months)",
    },
    "total_assets_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Assets (FY)",
        "description": "Total assets (Fiscal Year)",
    },
    "total_equity_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Equity (FY)",
        "description": "Total equity (Fiscal Year)",
    },
    "total_equity_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Equity (LTM)",
    },
    "total_debt_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Debt (FY)",
        "description": "Total debt (Fiscal Year)",
    },
    "total_debt_ltm": {"dtype": "float", "role": "balance_sheet", "sql_name": "Total Debt (LTM)"},
    "total_current_assets_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Current Assets (LTM)",
    },
    "total_current_liabilities_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Current Liabilities (LTM)",
    },
    "current_ratio_fy": {"dtype": "Float64", "role": "ratio", "sql_name": "Current Ratio (FY)"},
    "current_ratio_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "Current Ratio (LTM)"},
    "working_capital_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (LTM)",
    },
    "working_capital_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (FQ)",
    },
    "working_capital_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (FY)",
    },
    "working_capital_5yavgfy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (5YAVGFY)",
    },
    "tbv_fy": {"dtype": "float", "role": "balance_sheet", "sql_name": "TBV (FY)"},
    "tbv_ltm": {"dtype": "float", "role": "balance_sheet", "sql_name": "TBV (LTM)"},
    "cash_and_equivalents_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (LTM)",
    },
    "cash_and_equivalents_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (FQ)",
    },
    "cash_and_equivalents_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (FY)",
    },
    "cash_and_equivalents_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (5YAVGFQ)",
    },
    "retained_earnings_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (LTM)",
    },
    "retained_earnings_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (FQ)",
    },
    "retained_earnings_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (FY)",
    },
    "retained_earnings_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (5YAVGFQ)",
    },
    "inventory_ltm": {"dtype": "float", "role": "balance_sheet", "sql_name": "Inventory (LTM)"},
    "inventory_fq": {"dtype": "float", "role": "balance_sheet", "sql_name": "Inventory (FQ)"},
    "inventory_fy": {"dtype": "float", "role": "balance_sheet", "sql_name": "Inventory (FY)"},
    "inventory_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Inventory (5YAVGFQ)",
    },
    "goodwill_fq": {"dtype": "float", "role": "balance_sheet", "sql_name": "Goodwill (FQ)"},
    "goodwill_ltm": {"dtype": "float", "role": "balance_sheet", "sql_name": "Goodwill (LTM)"},
    "goodwill_fy": {"dtype": "float", "role": "balance_sheet", "sql_name": "Goodwill (FY)"},
    "goodwill_1fy": {"dtype": "float", "role": "balance_sheet", "sql_name": "Goodwill (-1FY)"},
    "goodwill_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Goodwill (5YAVGFQ)",
    },
    "intangible_assets": {"dtype": "Float64", "role": "feature"},
    "gross_intangible_assets_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Gross Intangible Assets (LTM)",
    },
    "gross_intangible_assets_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Gross Intangible Assets (FY)",
    },
    "gross_intangible_assets_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Gross Intangible Assets (5YAVGFQ)",
    },
    "capital_expenditure_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (LTM)",
        "description": "Capital expenditure (Last Twelve Months)",
    },
    "capital_expenditure_fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (FY)",
    },
    "capital_expenditure_fq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (FQ)",
    },
    "capital_expenditure_1fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (-1FY)",
    },
    "capital_expenditure_5yavgfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (5YAVGFQ)",
    },
    "asset_turnover_fy": {"dtype": "Float64", "role": "ratio", "sql_name": "Asset Turnover (FY)"},
    "asset_turnover_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "Asset Turnover (LTM)"},
    "cash_acquisitions_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (LTM)",
    },
    "cash_acquisitions_fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (FY)",
    },
    "cash_acquisitions_fq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (FQ)",
    },
    "cash_acquisitions_1fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (-1FY)",
    },
    "cash_acquisitions_5yavgfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (5YAVGFQ)",
    },
    "impairment_of_goodwill_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (FQ)",
    },
    "impairment_of_goodwill_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (LTM)",
    },
    "impairment_of_goodwill_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-1FY)",
    },
    "impairment_of_goodwill_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (FY)",
    },
    "impairment_of_goodwill_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (5YAVGFQ)",
    },
    "asset_writedown_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (LTM)",
    },
    "asset_writedown_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (FY)",
    },
    "asset_writedown_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (FQ)",
    },
    "asset_writedown_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-1FY)",
    },
    "asset_writedown_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (5YAVGFQ)",
    },
    "restructuring_charges_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (LTM)",
    },
    "restructuring_charges_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (FQ)",
    },
    "restructuring_charges_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-1FY)",
    },
    "restructuring_charges_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (FY)",
    },
    "restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (5YAVGFQ)",
    },
    "merger_and_restructuring_charges_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (LTM)",
    },
    "merger_and_restructuring_charges_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (FQ)",
    },
    "merger_and_restructuring_charges_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (FY)",
    },
    "merger_and_restructuring_charges_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (-1FY)",
    },
    "merger_and_restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (5YAVGFQ)",
    },
    # --- NEW COLUMNS: Address missing schema warnings ---
    "r_d_expenses_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "R&D Expenses (LTM)",
        "description": "Research and development expenses (Last Twelve Months)",
    },
    "merger_restructuring_charges_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger/Restructuring Charges (LTM)",
        "description": "Merger and restructuring charges (Last Twelve Months) - alternate naming",
    },
    "merger_restructuring_charges_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger/Restructuring Charges (FQ)",
        "description": "Merger and restructuring charges (Fiscal Quarter) - alternate naming",
    },
    "merger_restructuring_charges_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger/Restructuring Charges (FY)",
        "description": "Merger and restructuring charges (Fiscal Year) - alternate naming",
    },
    "merger_restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger/Restructuring Charges (5YAVGFQ)",
        "description": "Merger and restructuring charges (5-year average FQ) - alternate naming",
    },
    "sga_expenses": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "SG&A Expenses",
        "description": "Selling, general, and administrative expenses (alias)",
    },
    "price_target_number": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - #",
        "description": "Number of analyst price targets (alias for price_target_num)",
    },
    "other_unusual_items_total_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Other Unusual Items/Total (LTM)",
    },
    "gain_loss_on_sale_of_assets_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (LTM)",
    },
    "cost_of_revenues_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Cost Of Revenues (LTM)",
    },
    "randd_expenses_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "R&D Expenses (LTM)",
    },
    "r_d_expenses": {"dtype": "Float64", "role": "feature"},
    "selling_general_and_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (FQ)",
    },
    "selling_general_and_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (FY)",
    },
    "selling_general_and_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (-1FY)",
    },
    "selling_general_and_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (5YAVGFQ)",
    },
    "accounts_receivable_total_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Accounts Receivable/Total (FY)",
    },
    "accounts_receivable_total_1fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Accounts Receivable/Total (-1FY)",
    },
    "accounts_receivable_total_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Accounts Receivable/Total (5YAVGFQ)",
    },
    "marketing_expenses": {"dtype": "Float64", "role": "feature"},
    "marketing_expenses_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (FQ)",
    },
    "marketing_expenses_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (FY)",
    },
    "marketing_expenses_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (-1FY)",
    },
    "marketing_expenses_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (5YAVGLTM)",
    },
    "eps_adj_1fy": {"dtype": "Float64", "role": "ratio", "sql_name": "EPS/Adj. (-1FY)"},
    "eps_adj_fy": {"dtype": "Float64", "role": "ratio", "sql_name": "EPS/Adj. (FY)"},
    "eps_adj_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EPS/Adj. (LTM)"},
    "net_eps_basic_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "Net EPS - Basic (LTM)"},
    "net_eps_basic_fq": {"dtype": "Float64", "role": "ratio", "sql_name": "Net EPS - Basic (FQ)"},
    "net_eps_basic_fy": {"dtype": "Float64", "role": "ratio", "sql_name": "Net EPS - Basic (FY)"},
    "net_eps_basic_1fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-1FQFQ)",
    },
    "net_eps_basic_2fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-2FQFQ)",
    },
    "net_eps_basic_3fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-3FQFQ)",
    },
    "net_eps_basic_4fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-4FQFQ)",
    },
    "net_eps_basic_1fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-1FY)",
    },
    "net_eps_basic_2fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-2FY)",
    },
    "net_eps_basic_3fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-3FY)",
    },
    "net_eps_basic_4fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-4FY)",
    },
    "net_eps_basic_5fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-5FY)",
    },
    "eps_norm_est_avg_ntm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS Norm - Est Avg (NTM)",
    },
    "eps_norm_est_avg_fy1e": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS Norm - Est Avg (FY1E)",
    },
    "eps_norm_est_num_fy1e": {
        "dtype": "float",
        "role": "count",
        "sql_name": "EPS Norm - Est # (FY1E)",
    },
    "eps_est_avg_rev_pct_fy1e_1w": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 1W)",
    },
    "eps_est_avg_rev_pct_fy1e_1m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 1M)",
    },
    "eps_est_avg_rev_pct_fy1e_3m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 3M)",
    },
    "eps_est_avg_rev_pct_fy1e_6m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 6M)",
    },
    "eps_est_avg_rev_pct_fy1e_1y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 1Y)",
    },
    "eps_gaap_est_avg_fy1e": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS GAAP - Est Avg (FY1E)",
    },
    "eps_gaap_est_avg_ntm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS GAAP - Est Avg (NTM)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_1m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 1M)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_3m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 3M)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_6m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 6M)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_1y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
    },
    "eps_previous_year": {"dtype": "Float64", "role": "feature"},
    "dividend_per_share_ltm": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Dividend Per Share (LTM)",
    },
    "div_yield_ind": {"dtype": "Float64", "role": "percentage", "sql_name": "Div Yield (Ind)"},
    "div_yield_ltm": {"dtype": "Float64", "role": "percentage", "sql_name": "Div Yield (LTM)"},
    "div_yield_ttm": {"dtype": "Float64", "role": "percentage", "sql_name": "Div Yield (TTM)"},
    "div_yield_ntm": {"dtype": "Float64", "role": "percentage", "sql_name": "Div Yield (NTM)"},
    "div_yield_1fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-1FYInd)",
    },
    "div_yield_2fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-2FYInd)",
    },
    "div_yield_3fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-3FYInd)",
    },
    "div_yield_4fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-4FYInd)",
    },
    "div_yield_5fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-5FYInd)",
    },
    "div_yield_5yavgltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (5YAVGLTM)",
    },
    "common_dividends_paid_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Common Dividends Paid (LTM)",
    },
    "common_dividends_paid_fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Common Dividends Paid (FY)",
    },
    "dividend_record_frequency": {
        "dtype": "string",
        "role": "categorical",
        "sql_name": "Dividend Record (Frequency)",
    },
    "dividend_record_currency": {
        "dtype": "string",
        "role": "categorical",
        "sql_name": "Dividend Record (Currency)",
    },
    "dividend_record_amount": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Dividend Record (Amount)",
        "description": "Dividend amount per share",
    },
    "dividend_streak": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Dividend Streak",
        "description": "Consecutive years of dividend payments",
    },
    "days_to_dividend": {"dtype": "Float64", "role": "feature"},
    "buyback_yield_ltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Buyback Yield (LTM)",
    },
    "interest_expense_total_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Interest Expense/Total (LTM)",
    },
    "interest_income_on_investments_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Interest Income On Investments (LTM)",
    },
    "employees": {"dtype": "float", "role": "count"},
    "avg_employees_ltm": {"dtype": "float", "role": "count"},
    "avg_employees_fy": {"dtype": "float", "role": "count"},
    "avg_employees_5yavgfy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Avg Employees (5YAVGFY)",
    },
    "total_employees_fy": {"dtype": "float", "role": "count"},
    "total_employees_fq": {"dtype": "float", "role": "count"},
    "full_time_employees_fq": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (FQ)",
        "description": "Full time employees (Fiscal Quarter)",
    },
    "full_time_employees_fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (FY)",
        "description": "Full time employees (Fiscal Year)",
    },
    "full_time_employees_1fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (-1FY)",
    },
    "full_time_employees_2fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (-2FY)",
    },
    "full_time_employees_3fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (-3FY)",
    },
    "p_e": {"dtype": "Float64", "role": "ratio"},
    "p_b": {"dtype": "Float64", "role": "ratio"},
    "revenue": {"dtype": "float", "role": "financial_statement"},
    "ebitda": {"dtype": "float", "role": "financial_statement"},
    "ebit": {"dtype": "float", "role": "financial_statement"},
    "net_income": {"dtype": "float", "role": "financial_statement"},
    "net_income_ltm": {"dtype": "float", "role": "financial_statement"},
    "gross_margin": {"dtype": "Float64", "role": "percentage"},
    "eps": {"dtype": "Float64", "role": "ratio"},
    "total_equity": {"dtype": "float", "role": "balance_sheet"},
    "total_assets": {"dtype": "float", "role": "balance_sheet"},
    "total_debt": {"dtype": "float", "role": "balance_sheet"},
    "inventory": {"dtype": "float", "role": "balance_sheet"},
    "capex": {"dtype": "float", "role": "cash_flow"},
    "cash_and_equivalents": {"dtype": "float", "role": "balance_sheet"},
    "current_assets": {"dtype": "float", "role": "balance_sheet"},
    "current_liabilities": {"dtype": "Float64", "role": "market"},
    "working_capital": {"dtype": "float", "role": "balance_sheet"},
    "retained_earnings": {"dtype": "float", "role": "balance_sheet"},
    "cfo": {"dtype": "float", "role": "cash_flow"},
    "cfi": {"dtype": "float", "role": "cash_flow"},
    "cff": {"dtype": "float", "role": "cash_flow"},
    "fcf": {"dtype": "float", "role": "cash_flow"},
    "gross_profit": {"dtype": "float", "role": "financial_statement"},
    "operating_income": {"dtype": "float", "role": "financial_statement"},
    "interest_expense": {"dtype": "float", "role": "financial_statement"},
    "goodwill": {"dtype": "float", "role": "balance_sheet"},
    "dividend_per_share": {"dtype": "Float64", "role": "market"},
    "operating_expenses": {"dtype": "float", "role": "financial_statement"},
    "operating_cash_flow": {"dtype": "Float64", "role": "market"},
    "dividends_paid": {"dtype": "float", "role": "cash_flow"},
    "dividends_paid_ltm": {"dtype": "float", "role": "cash_flow"},
    "volatility_1y_pct": {"dtype": "Float64", "role": "percentage"},
    "tangible_book_value": {"dtype": "Float64", "role": "market"},
    "marketing_efficiency": {"dtype": "Float64", "role": "ratio"},
    "r_d_intensity": {"dtype": "Float64", "role": "percentage"},
    "rule_of_40": {"dtype": "Float64", "role": "percentage"},
    "operating_leverage": {"dtype": "Float64", "role": "ratio"},
    "one_day_chg": {"dtype": "Float64", "role": "percentage"},
    "market_cap_x_debt_to_equity": {"dtype": "Float64", "role": "feature"},
    "market_cap_x_roe": {"dtype": "Float64", "role": "feature"},
    "p_e_ratio_x_debt_to_equity": {"dtype": "Float64", "role": "feature"},
    "p_e_ratio_x_roe": {"dtype": "Float64", "role": "feature"},
    "roe_x_debt_to_equity": {"dtype": "Float64", "role": "feature"},
    "log_operating_income": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda": {"dtype": "float", "role": "financial_statement"},
    "log_net_income": {"dtype": "float", "role": "financial_statement"},
    "log_capex": {"dtype": "float", "role": "cash_flow"},
    "log_operating_cash_flow": {"dtype": "Float64", "role": "market"},
    "log_total_equity": {"dtype": "float", "role": "balance_sheet"},
    "log_market_cap": {"dtype": "Float64", "role": "market"},
    "log_total_assets": {"dtype": "float", "role": "balance_sheet"},
    "log_gross_profit": {"dtype": "float", "role": "financial_statement"},
    "log_cash_and_equivalents": {"dtype": "float", "role": "balance_sheet"},
    "log_total_debt": {"dtype": "float", "role": "balance_sheet"},
    "log_revenue": {"dtype": "float", "role": "financial_statement"},
    "log_enterprise_value": {"dtype": "Float64", "role": "market"},
    "log_gross_profit_previous_year": {"dtype": "float", "role": "financial_statement"},
    "log_operating_income_fq": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_total_revenues_5yavgfq": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_fq": {"dtype": "float", "role": "cash_flow"},
    "log_total_revenues_5yavgltm": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_fy": {"dtype": "float", "role": "financial_statement"},
    "log_total_assets_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_ebitda_previous_year": {"dtype": "float", "role": "financial_statement"},
    "log_operating_income_fy": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_ltm": {"dtype": "float", "role": "cash_flow"},
    "log_revenues_est_avg_ntm": {"dtype": "float", "role": "financial_statement"},
    "log_total_revenues_fy": {"dtype": "float", "role": "financial_statement"},
    "log_net_income_is_1fy": {"dtype": "float", "role": "financial_statement"},
    "log_fcf_fq": {"dtype": "float", "role": "cash_flow"},
    "log_total_equity_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_total_revenues_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_net_income_adj_1fy": {"dtype": "float", "role": "financial_statement"},
    "log_total_equity_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_total_debt_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_revenue_previous_year": {"dtype": "float", "role": "financial_statement"},
    "log_revenue_fy": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_5yavgfq": {"dtype": "float", "role": "cash_flow"},
    "log_net_income_is_5yavgltm": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_fy": {"dtype": "float", "role": "cash_flow"},
    "log_total_assets_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_net_income_adj_fy": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_5yavgltm": {"dtype": "float", "role": "financial_statement"},
    "log_revenues_est_avg_fy1e": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_fq": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_1fy": {"dtype": "float", "role": "financial_statement"},
    "log_revenues_est_med_ntm": {"dtype": "float", "role": "financial_statement"},
    "log_cash_and_equivalents_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_net_income_is_5yavgfq": {"dtype": "float", "role": "financial_statement"},
    "log_cash_and_equivalents_5yavgfq": {"dtype": "float", "role": "balance_sheet"},
    "log_fcf_ltm": {"dtype": "float", "role": "cash_flow"},
    "log_total_debt_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_fcf": {"dtype": "float", "role": "cash_flow"},
    "log_gross_profit_fy": {"dtype": "float", "role": "financial_statement"},
    "log_market_cap_country_r": {"dtype": "Float64", "role": "market"},
    "log_cash_and_equivalents_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_fcf_5yavgfq": {"dtype": "float", "role": "cash_flow"},
    "log_ebitda_5yavgfq": {"dtype": "float", "role": "financial_statement"},
    "log_fcf_fy": {"dtype": "float", "role": "cash_flow"},
    "log_revenues_est_med_fy1e": {"dtype": "float", "role": "financial_statement"},
    "log_total_assets_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "log_operating_income_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_net_income_is_fq": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_adj_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_gross_profit_ltm": {"dtype": "float", "role": "financial_statement"},
    "p_e_ratio": {"dtype": "Float64", "role": "ratio"},
    "p_s_ratio": {"dtype": "Float64", "role": "ratio"},
    "ev_ebitda_ratio": {"dtype": "Float64", "role": "ratio"},
    "ev_sales_ratio": {"dtype": "Float64", "role": "ratio"},
    "gross_margin_pct": {"dtype": "Float64", "role": "percentage"},
    "operating_margin_pct": {"dtype": "Float64", "role": "percentage"},
    "net_margin_pct": {"dtype": "Float64", "role": "percentage"},
    "roe": {"dtype": "Float64", "role": "ratio"},
    "roa": {"dtype": "Float64", "role": "ratio"},
    "revenue_growth": {"dtype": "Float64", "role": "percentage"},
    "ebitda_growth": {"dtype": "Float64", "role": "percentage"},
    "earnings_growth": {"dtype": "Float64", "role": "percentage"},
    "debt_to_equity": {"dtype": "Float64", "role": "ratio"},
    "debt_to_assets": {"dtype": "Float64", "role": "ratio"},
    "target_vs_price": {"dtype": "Float64", "role": "ratio"},
    "target_vs_price_median": {"dtype": "Float64", "role": "ratio"},
    "peg_ratio": {"dtype": "Float64", "role": "ratio"},
    "dividend_yield": {"dtype": "Float64", "role": "percentage"},
    "roic": {"dtype": "Float64", "role": "ratio"},
    "revenue_previous_year": {"dtype": "float", "role": "financial_statement"},
    "ebitda_previous_year": {"dtype": "float", "role": "financial_statement"},
    "total_equity_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "total_assets_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "gross_profit_previous_year": {"dtype": "float", "role": "financial_statement"},
    "accounts_receivable_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "roa_previous_year": {"dtype": "Float64", "role": "ratio"},
    "current_ratio_previous_year": {"dtype": "Float64", "role": "ratio"},
    "shares_outstanding_previous_year": {"dtype": "float", "role": "count"},
    "gross_margin_pct_previous_year": {"dtype": "Float64", "role": "percentage"},
    "asset_turnover_previous_year": {"dtype": "Float64", "role": "ratio"},
    "revenue_fy": {"dtype": "float", "role": "financial_statement"},
    "working_capital_1fy": {"dtype": "float", "role": "balance_sheet"},
    "cash_burn_rate": {"dtype": "Float64", "role": "feature"},
    "cash_burn_rate_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_ltm": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_ltm_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_fy": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_fy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_trend": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_trend_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_vs_5y_pct": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_vs_5y_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "assets_per_employee": {"dtype": "Float64", "role": "feature"},
    "assets_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "ebitda_per_employee": {"dtype": "Float64", "role": "feature"},
    "ebitda_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "operating_income_per_employee": {"dtype": "Float64", "role": "feature"},
    "operating_income_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "profit_per_employee": {"dtype": "Float64", "role": "feature"},
    "profit_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy": {"dtype": "Float64", "role": "feature"},
    "employee_growth_yoy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy_pct": {"dtype": "Float64", "role": "feature"},
    "employee_growth_yoy_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_qoq": {"dtype": "Float64", "role": "feature"},
    "employee_growth_qoq_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_cagr_5y": {"dtype": "Float64", "role": "feature"},
    "employee_growth_cagr_5y_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_acceleration": {"dtype": "Float64", "role": "feature"},
    "employee_growth_acceleration_applicable": {"dtype": "bool", "role": "auxiliary"},
    "workforce_volatility": {"dtype": "Float64", "role": "feature"},
    "workforce_volatility_applicable": {"dtype": "bool", "role": "auxiliary"},
    "hiring_intensity_score": {"dtype": "Float64", "role": "feature"},
    "hiring_intensity_score_applicable": {"dtype": "bool", "role": "auxiliary"},
    "altman_z_score": {"dtype": "Float64", "role": "feature"},
    "beneish_m_score": {"dtype": "Float64", "role": "feature"},
    "composite_quality_score": {"dtype": "Float64", "role": "feature"},
    "momentum_score": {"dtype": "Float64", "role": "feature"},
    "eps_surprise_pct": {"dtype": "Float64", "role": "percentage"},
    "eps_surprise_magnitude": {"dtype": "category", "role": "categorical"},
    "revenue_surprise_pct": {"dtype": "Float64", "role": "percentage"},
    "revenue_beat_indicator": {"dtype": "bool", "role": "feature"},
    "ebitda_surprise_pct": {"dtype": "Float64", "role": "percentage"},
    "earnings_beat_indicator": {"dtype": "bool", "role": "feature"},
    "surprise_momentum_score": {"dtype": "Float64", "role": "feature"},
    "positive_revision_momentum": {"dtype": "bool", "role": "feature"},
    "consensus_uncertainty_score": {"dtype": "Float64", "role": "feature"},
    "estimate_revision_acceleration": {"dtype": "Float64", "role": "percentage"},
    "accelerating_upgrades_flag": {"dtype": "bool", "role": "feature"},
    "eps_adjustment_spread_ltm": {"dtype": "Float64", "role": "feature"},
    "eps_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "eps_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "eps_quality_flag_ltm": {"dtype": "bool", "role": "feature"},
    "eps_adjustment_spread_fy": {"dtype": "Float64", "role": "feature"},
    "eps_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "eps_adjustment_pct_fy": {"dtype": "Float64", "role": "percentage"},
    "net_income_adjustment_spread_ltm": {"dtype": "float", "role": "financial_statement"},
    "net_income_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "net_income_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "net_income_adjustment_spread_fy": {"dtype": "float", "role": "financial_statement"},
    "net_income_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "ebitda_adjustment_spread_ltm": {"dtype": "float", "role": "financial_statement"},
    "ebitda_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "ebitda_adjustment_spread_fy": {"dtype": "float", "role": "financial_statement"},
    "ebit_adjustment_spread_ltm": {"dtype": "float", "role": "financial_statement"},
    "ebit_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "ebit_adjustment_spread_fy": {"dtype": "float", "role": "financial_statement"},
    "adjustment_consistency_score": {"dtype": "Float64", "role": "feature"},
    "earnings_quality_warning_flag": {"dtype": "bool", "role": "feature"},
    "earnings_quality_score": {"dtype": "Float64", "role": "feature"},
    "exceptional_items_impact_ratio": {"dtype": "Float64", "role": "ratio"},
    "ebit_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "ebit_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "ebitda_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "ebitda_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "ebitda_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "gross_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "net_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "operating_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "days_to_earnings": {"dtype": "Float64", "role": "feature"},
    "earnings_report_recency": {"dtype": "Float64", "role": "feature"},
    "reporting_lag": {"dtype": "Float64", "role": "feature", "sql_name": "Reporting Lag"},
    # Temporal features that may contain pd.NA (use nullable Float64)
    "ltm_vs_5yavg_revenue": {
        "dtype": "Float64",
        "role": "feature",
        "description": "LTM revenue vs 5-year average ratio",
    },
    "fq_vs_5yavg_ebitda": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Fiscal quarter EBITDA vs 5-year average ratio",
    },
    "quarterly_volatility_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Coefficient of variation across quarterly EBITDA",
    },
    # =========================================================================
    # MISSING PHASE 9.3 FEATURES - Coverage Gap Fill
    # =========================================================================
    # Analyst Sentiment
    "analyst_coverage_quality": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Quality score based on analyst coverage breadth and consistency",
    },
    "price_target_revision": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Recent price target revision percentage",
    },
    # Technical Analysis
    "rsi_14d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "14-day Relative Strength Index",
    },
    "rsi_30d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "30-day Relative Strength Index",
    },
    "momentum_20d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "20-day price momentum indicator",
    },
    # Quality & Risk
    "distress_risk_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite financial distress probability score",
    },
    "altman_z_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Year-over-year change in Altman Z-Score",
    },
    "z_score_volatility": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Standard deviation of Altman Z-Score over time",
    },
    "exceptional_items_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Trend in exceptional/non-recurring items over time",
    },
    # Employee Productivity
    "fte_cagr_3y_pct": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Full-time employee 3-year compound annual growth rate",
    },
    "fte_growth_1y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Full-time employee 1-year growth percentage",
    },
    "fte_growth_2y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Full-time employee 2-year growth percentage",
    },
    "fte_growth_3y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Full-time employee 3-year growth percentage",
    },
    "revenue_per_employee_1fy": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Revenue per employee from previous fiscal year",
    },
    "workforce_volatility_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Workforce size volatility as percentage",
    },
    # Balance Sheet Dynamics
    "asset_growth_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year total asset growth rate",
    },
    "balance_sheet_expansion": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite balance sheet expansion indicator",
    },
    "current_ratio_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Trend in current ratio over time",
    },
    "debt_growth_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year total debt growth rate",
    },
    "equity_growth_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year total equity growth rate",
    },
    "earnings_retention_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Proportion of earnings retained vs distributed",
    },
    "retained_earnings_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year retained earnings growth",
    },
    "working_capital_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Working capital as ratio of total assets",
    },
    # Revenue Forecasting
    "revenue_forecast_accuracy": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Historical accuracy of revenue forecasts",
    },
    # Valuation Timeseries
    "valuation_extreme_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating extreme valuation vs historical norms",
    },
    "valuation_stability_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of valuation multiples over time",
    },
    "valuation_trend_consistency": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Consistency of valuation trend direction",
    },
    # Earnings Quality
    "earnings_quality_score_composite": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite earnings quality score combining multiple factors",
    },
    "eps_adjustment_ratio_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Ratio of adjusted to GAAP EPS for fiscal year",
    },
    # Dividend Reliability
    "dividend_coverage_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Earnings coverage of dividend payments",
    },
    "dividend_growth_3y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "3-year dividend growth rate",
    },
    "dividend_growth_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "5-year dividend growth rate",
    },
    "dividend_yield_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of dividend yield over time",
    },
    "fcf_dividend_coverage": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Free cash flow coverage of dividends",
    },
    "payout_consistency_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Consistency of dividend payout ratio over time",
    },
    "sustainable_dividend_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating dividend sustainability",
    },
    # Composite Scores
    "value_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite value investing score",
    },
    "piotroski_f_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Piotroski F-Score (0-9 financial strength)",
    },
    # Efficiency Ratios
    "inventory_turnover": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Inventory turnover ratio",
    },
    "receivables_turnover": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Accounts receivable turnover ratio",
    },
    "asset_turnover": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Asset turnover ratio",
    },
    # Cash Flow
    "cfo_growth_yoy": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year cash from operations growth",
    },
    "cfo_to_net_income": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Cash from operations to net income ratio",
    },
    "fcf_margin": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Free cash flow margin (FCF/Revenue)",
    },
    "fcf_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of free cash flow over time",
    },
    "fcf_to_net_income": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Free cash flow to net income ratio",
    },
    # Leverage & Liquidity
    "cash_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Cash ratio (cash/current liabilities)",
    },
    "current_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Current ratio (current assets/current liabilities)",
    },
    "equity_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Equity ratio (equity/total assets)",
    },
    "interest_coverage": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Interest coverage ratio (EBIT/interest expense)",
    },
    "net_debt_to_ebitda": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Net debt to EBITDA ratio",
    },
    "quick_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Quick ratio (liquid assets/current liabilities)",
    },
    "working_capital_to_sales": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Working capital as percentage of sales",
    },
    # Capital Allocation
    "payout_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Dividend payout ratio",
    },
    "reinvestment_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Rate of earnings reinvestment",
    },
    "retention_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Earnings retention rate (1 - payout ratio)",
    },
    "cash_conversion_cycle": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days in cash conversion cycle",
    },
    # Growth Metrics
    "book_value_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year book value growth",
    },
    "fcf_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year free cash flow growth",
    },
    "operating_income_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year operating income growth",
    },
    # Market Sentiment
    "beta_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of beta coefficient over time",
    },
    "short_interest_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Short interest as percentage of float",
    },
    "systematic_risk_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Trend in systematic risk exposure",
    },
    "price_range_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Price range as percentage of mid-price",
    },
    # Valuation Ratios
    "book_value_per_share": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Book value per share",
    },
    "p_b_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Price to book ratio",
    },
    # Temporal Patterns
    "days_since_reference": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days since reference date",
    },
    "quarter_end_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating quarter-end proximity",
    },
    "month_end_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating month-end proximity",
    },
    "week_of_year": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Week number within year",
    },
    "day_of_week": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Day of week (0=Monday)",
    },
    "month": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Month number (1-12)",
    },
    "year": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Calendar year",
    },
}

# Phase 9.3 Feature Input Categorization (v1.13)
# Total: 296 features across 21 categories
PHASE93_FEATURE_CATEGORIES: Dict[str, List[str]] = {
    # =========================================================================
    # MOMENTUM & TECHNICAL (25 features)
    # =========================================================================
    "Momentum & Technical": [
        "52w_range_position",
        "breakout_signal",
        "ema_crossover_20_50",
        "ema_crossover_50_250",
        "ema_slope_20d",
        "ema_trend_consistency",
        "ma_20d_simple",
        "ma_50d_simple",
        "ma_crossover_signal",
        "near_52w_high_flag",
        "near_52w_low_flag",
        "pct_above_52w_low",
        "pct_off_52w_high",
        "price_acceleration_3m",
        "price_distance_from_ma",
        "price_momentum_1m",
        "price_momentum_1y",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_vs_ema_20d",
        "price_vs_ema_250d",
        "return_stability_score",
        "sharpe_proxy",
        "total_return_1y_pct",
        "volume_momentum_score",
    ],
    # =========================================================================
    # VALUATION RATIOS (25 features)
    # =========================================================================
    "Valuation Ratios": [
        "book_value_per_share",
        "dividend_yield",
        "ev_ebitda_forward_discount",
        "ev_ebitda_momentum",
        "ev_ebitda_ratio",
        "ev_ebitda_vs_3y_avg",
        "ev_sales_forward_discount",
        "ev_sales_quarterly_volatility",
        "ev_sales_ratio",
        "ev_sales_trend_1y",
        "ev_sales_trend_3y",
        "ev_sales_vs_3y_avg",
        "growth_implied_by_valuation",
        "p_b",
        "p_b_ratio",
        "p_e_forward_discount",
        "p_e_momentum_qoq",
        "p_e_momentum_yoy",
        "p_e_ratio",
        "p_e_vs_3y_avg",
        "p_s_ratio",
        "peg_ratio",
        "valuation_extreme_flag",
        "valuation_stability_score",
        "valuation_trend_consistency",
    ],
    # =========================================================================
    # PROFITABILITY (16 features)
    # =========================================================================
    "Profitability": [
        "ebit_adjustment_ratio_fy",
        "ebit_adjustment_ratio_ltm",
        "ebitda_adjustment_ratio_fy",
        "ebitda_adjustment_ratio_ltm",
        "ebitda_margin_trend",
        "gross_margin_pct",
        "gross_margin_trend",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_ratio_ltm",
        "net_margin_pct",
        "net_margin_trend",
        "operating_leverage",
        "operating_margin_pct",
        "operating_margin_trend",
        "roa",
        "roe",
        "roic",
    ],
    # =========================================================================
    # QUALITY & RISK (18 features)
    # =========================================================================
    "Quality & Risk": [
        "accounting_quality_score",
        "altman_z_score",
        "altman_z_trend",
        "beneish_m_score",
        "distress_risk_score",
        "exceptional_items_to_ebitda",
        "exceptional_items_to_ni_pct",
        "exceptional_items_trend",
        "goodwill_change_rate",
        "goodwill_impairment_flag",
        "goodwill_to_assets",
        "goodwill_to_assets_pct",
        "has_asset_writedown",
        "has_goodwill_impairment",
        "has_restructuring",
        "intangible_intensity",
        "intangibles_to_assets_pct",
        "restructuring_intensity",
        "total_exceptional_items_ltm",
        "z_score_volatility",
    ],
    # =========================================================================
    # CASH FLOW (5 features)
    # =========================================================================
    "Cash Flow": [
        "cfo_growth_yoy",
        "cfo_to_net_income",
        "fcf_margin",
        "fcf_stability",
        "fcf_to_net_income",
    ],
    # =========================================================================
    # CAPITAL ALLOCATION (23 features)
    # =========================================================================
    "Capital Allocation": [
        "buyback_yield_ltm",
        "cash_conversion_cycle",
        "common_dividends_paid_fy",
        "common_dividends_paid_ltm",
        "dividend_per_share",
        "dividend_per_share_ltm",
        "dividend_record_amount",
        "dividend_streak",
        "div_yield_1fyind",
        "div_yield_2fyind",
        "div_yield_3fyind",
        "div_yield_4fyind",
        "div_yield_5fyind",
        "div_yield_5yavgltm",
        "div_yield_ind",
        "div_yield_ltm",
        "div_yield_ntm",
        "div_yield_ttm",
        "dividends_paid",
        "dividends_paid_ltm",
        "payout_ratio",
        "reinvestment_rate",
        "retention_rate",
    ],
    # =========================================================================
    # ANALYST SENTIMENT (10 features)
    # =========================================================================
    "Analyst Sentiment": [
        "analyst_bullish_pct",
        "analyst_bearish_pct",
        "analyst_conviction",
        "analyst_coverage_quality",
        "consensus_strength",
        "price_target_range",
        "price_target_revision",
        "price_target_spread_pct",
        "target_price_upside_pct",
        "upside_potential",
    ],
    # =========================================================================
    # MARKET SENTIMENT (5 features)
    # =========================================================================
    "Market Sentiment": [
        "beta_stability",
        "one_day_chg",
        "short_interest_ratio",
        "systematic_risk_trend",
        "price_range_pct",
    ],
    # =========================================================================
    # LEVERAGE & LIQUIDITY (9 features)
    # =========================================================================
    "Leverage & Liquidity": [
        "cash_ratio",
        "current_ratio",
        "debt_to_assets",
        "debt_to_equity",
        "equity_ratio",
        "interest_coverage",
        "net_debt_to_ebitda",
        "quick_ratio",
        "working_capital_to_sales",
    ],
    # =========================================================================
    # TEMPORAL PATTERNS (17 features)
    # =========================================================================
    "Temporal Patterns": [
        "reference_date",
        "days_since_reference",
        "days_to_dividend",
        "days_to_earnings",
        "earnings_report_recency",
        "fiscal_quarter",
        "fiscal_year",
        "month",
        "reporting_lag",
        "year",
        "quarter_end_flag",
        "month_end_flag",
        "week_of_year",
        "day_of_week",
        "ltm_vs_5yavg_revenue",
        "fq_vs_5yavg_ebitda",
        "quarterly_volatility_score",
    ],
    # =========================================================================
    # COMPOSITE SCORES (5 features)
    # =========================================================================
    "Composite Scores": [
        "composite_quality_score",
        "earnings_quality_score",
        "momentum_score",
        "piotroski_f_score",
        "value_score",
    ],
    # =========================================================================
    # GROWTH METRICS (9 features)
    # =========================================================================
    "Growth Metrics": [
        "earnings_growth",
        "ebitda_growth",
        "ebitda_growth_yoy",
        "eps_growth_yoy",
        "revenue_growth",
        "revenue_growth_yoy",
        "book_value_growth",
        "fcf_growth",
        "operating_income_growth",
    ],
    # =========================================================================
    # EFFICIENCY RATIOS (4 features)
    # =========================================================================
    "Efficiency Ratios": [
        "asset_turnover",
        "inventory_turnover",
        "receivables_turnover",
        "revenue_per_employee",
    ],
    # =========================================================================
    # EMPLOYEE PRODUCTIVITY (21 features)
    # =========================================================================
    "Employee Productivity": [
        "assets_per_employee",
        "ebitda_per_employee",
        "employee_base_scale_flag",
        "employee_growth_acceleration",
        "employee_growth_cagr_5y",
        "employee_growth_yoy",
        "fte_cagr_3y_pct",
        "fte_growth_1y_pct",
        "fte_growth_2y_pct",
        "fte_growth_3y_pct",
        "operating_income_per_employee",
        "profit_per_employee",
        "revenue_per_employee_1fy",
        "revenue_per_employee_fy",
        "revenue_per_employee_trend",
        "workforce_volatility",
        "workforce_volatility_pct",
        "hiring_intensity_score",
        "revenue_per_employee_ltm",
        "revenue_per_employee_vs_5y_pct",
        "employee_growth_yoy_pct",
    ],
    # =========================================================================
    # BALANCE SHEET DYNAMICS (9 features)
    # =========================================================================
    "Balance Sheet Dynamics": [
        "asset_growth_rate",
        "balance_sheet_expansion",
        "cash_ratio",
        "current_ratio_trend",
        "debt_growth_rate",
        "earnings_retention_rate",
        "equity_growth_rate",
        "retained_earnings_growth",
        "working_capital_ratio",
    ],
    # =========================================================================
    # REVENUE FORECASTING (9 features)
    # =========================================================================
    "Revenue Forecasting": [
        "eps_est_avg_rev_pct_fy1e_1m",
        "eps_est_avg_rev_pct_fy1e_1w",
        "eps_est_avg_rev_pct_fy1e_1y",
        "eps_est_avg_rev_pct_fy1e_3m",
        "eps_est_avg_rev_pct_fy1e_6m",
        "revenue_forecast_accuracy",
        "revenues_est_avg_fy1e",
        "revenues_est_avg_ntm",
        "revenues_est_yoy_pct_fy1e",
    ],
    # =========================================================================
    # EARNINGS QUALITY (33 features)
    # =========================================================================
    "Earnings Quality": [
        # Estimated vs. Actual Analytics (11 features)
        "accelerating_upgrades_flag",
        "consensus_uncertainty_score",
        "earnings_beat_indicator",
        "eps_surprise_magnitude",
        "eps_surprise_pct",
        "estimate_revision_acceleration",
        "ebitda_surprise_pct",
        "positive_revision_momentum",
        "revenue_beat_indicator",
        "revenue_surprise_pct",
        "surprise_momentum_score",
        # GAAP vs. Adjusted Analytics (22 features)
        "adjustment_consistency_score",
        "earnings_quality_score_composite",
        "earnings_quality_warning_flag",
        "ebit_adjustment_pct_ltm",
        "ebit_adjustment_ratio_fy",
        "ebit_adjustment_ratio_ltm",
        "ebit_adjustment_spread_fy",
        "ebit_adjustment_spread_ltm",
        "ebitda_adjustment_pct_ltm",
        "ebitda_adjustment_ratio_fy",
        "ebitda_adjustment_ratio_ltm",
        "ebitda_adjustment_spread_fy",
        "ebitda_adjustment_spread_ltm",
        "eps_adjustment_pct_fy",
        "eps_adjustment_pct_ltm",
        "eps_adjustment_ratio_fy",
        "eps_adjustment_ratio_ltm",
        "eps_adjustment_spread_fy",
        "eps_adjustment_spread_ltm",
        "eps_quality_flag_ltm",
        "exceptional_items_impact_ratio",
        "net_income_adjustment_pct_ltm",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_ratio_ltm",
        "net_income_adjustment_spread_fy",
        "net_income_adjustment_spread_ltm",
    ],
    # =========================================================================
    # TECHNICAL ANALYSIS (15 features)
    # =========================================================================
    "Technical Analysis": [
        "52w_range_position",
        "breakout_signal",
        "ema_crossover_20_50",
        "ema_crossover_50_250",
        "ema_slope_20d",
        "ema_trend_consistency",
        "momentum_20d",
        "near_52w_high_flag",
        "near_52w_low_flag",
        "pct_above_52w_low",
        "pct_off_52w_high",
        "price_vs_ema_20d",
        "price_vs_ema_250d",
        "rsi_14d",
        "rsi_30d",
    ],
    # =========================================================================
    # VALUATION TIMESERIES (16 features)
    # =========================================================================
    "Valuation Timeseries": [
        "ev_ebitda_forward_discount",
        "ev_ebitda_momentum",
        "ev_ebitda_vs_3y_avg",
        "ev_sales_forward_discount",
        "ev_sales_quarterly_volatility",
        "ev_sales_trend_1y",
        "ev_sales_trend_3y",
        "ev_sales_vs_3y_avg",
        "growth_implied_by_valuation",
        "p_e_forward_discount",
        "p_e_momentum_qoq",
        "p_e_momentum_yoy",
        "p_e_vs_3y_avg",
        "valuation_extreme_flag",
        "valuation_stability_score",
        "valuation_trend_consistency",
    ],
    # =========================================================================
    # DIVIDEND RELIABILITY (12 features)
    # =========================================================================
    "Dividend Reliability": [
        "days_to_dividend",
        "dividend_coverage_ratio",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "dividend_payout_ratio",
        "dividend_reliability_score",
        "dividend_streak",
        "dividend_yield_stability",
        "div_yield_5yavgltm",
        "fcf_dividend_coverage",
        "payout_consistency_score",
        "sustainable_dividend_flag",
    ],
    # =========================================================================
    # EMPLOYMENT DYNAMICS (10 features)
    # =========================================================================
    "Employment Dynamics": [
        "employee_base_scale_flag",
        "employee_growth_acceleration",
        "employee_growth_cagr_5y",
        "employee_growth_yoy",
        "hiring_intensity_score",
        "profit_per_employee",
        "revenue_per_employee_fy",
        "revenue_per_employee_trend",
        "workforce_volatility",
        "fte_cagr_3y_pct",
    ],
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
    feature_roles = {
        "feature",
        "target",
        "target_fallback",
        "market",
        "financial_statement",
        "balance_sheet",
        "cash_flow",
        "ratio",
        "percentage",
        "count",
    }

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


def list_non_recurring_cols() -> List[str]:
    """List all non-recurring exceptional item columns from COLUMN_SCHEMA.

    These columns represent rare/exceptional events where missing values
    typically mean the event did not occur. Zero is the economically
    correct imputation for these items.

    Returns:
        List of column names with role='non_recurring'
    """
    return [col for col, meta in COLUMN_SCHEMA.items() if meta.get("role") == "non_recurring"]


def list_knn_imputable_cols() -> List[str]:
    """List all columns suitable for KNN imputation from COLUMN_SCHEMA.

    These are core financial metrics where KNN can leverage sector relationships
    and correlations to provide better estimates than simple statistics.

    Includes columns with roles: feature, market, financial_statement,
    balance_sheet, cash_flow, ratio, percentage.

    Excludes: non_recurring (zero imputation), count (median imputation),
    id, categorical, date, target, auxiliary.

    Returns:
        List of column names suitable for KNN imputation
    """
    knn_roles = {
        "feature",
        "market",
        "financial_statement",
        "balance_sheet",
        "cash_flow",
        "ratio",
        "percentage",
    }

    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("role") in knn_roles and meta.get("dtype") in ["float", "int", "bool"]
    ]


def list_count_cols() -> List[str]:
    """List all count columns from COLUMN_SCHEMA.

    These are discrete integer columns (analyst ratings, employees, shares)
    that should use median imputation rather than KNN.

    Returns:
        List of column names with role='count'
    """
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("role") == "count" and meta.get("dtype") in ["float", "int"]
    ]


def list_price_cols() -> List[str]:
    """List all price-related columns from COLUMN_SCHEMA.

    These columns should be used for price imputation (filling missing
    price targets with last_price as fallback).

    Returns:
        List of column names with role in ('market', 'target', 'target_fallback')
        and containing price-related semantics
    """
    price_roles = {"market", "target", "target_fallback"}
    price_keywords = {"price", "target", "ema_", "ma_", "52w_"}

    result = []
    for col, meta in COLUMN_SCHEMA.items():
        role = meta.get("role", "")
        if role in price_roles:
            # Check if column name suggests price-related data
            if any(kw in col.lower() for kw in price_keywords):
                result.append(col)

    return result


def get_pandas_nullable_dtype(dtype: str) -> str:
    """Convert schema dtype to pandas nullable-safe equivalent.

    Use this when the resulting Series may contain pd.NA values
    (e.g., after division with .replace(0, pd.NA)).

    Args:
        dtype: Schema dtype string (e.g., 'float', 'int', 'bool')

    Returns:
        Pandas nullable dtype string (e.g., 'Float64', 'Int64', 'boolean')
    """
    nullable_map = {
        "float": "Float64",
        "int": "Int64",
        "bool": "boolean",
    }
    return nullable_map.get(dtype, dtype)


def get_numpy_dtype(dtype: str) -> str:
    """Convert schema dtype to numpy-compatible equivalent.

    Use this when you need standard numpy dtypes (e.g., for scikit-learn).
    Note: Will fail if Series contains pd.NA - convert NA to np.nan first.

    Args:
        dtype: Schema dtype string

    Returns:
        NumPy-compatible dtype string
    """
    numpy_map = {
        "Float64": "float64",
        "Int64": "int64",
        "boolean": "bool",
        "float": "float64",
        "int": "int64",
    }
    return numpy_map.get(dtype, dtype)
