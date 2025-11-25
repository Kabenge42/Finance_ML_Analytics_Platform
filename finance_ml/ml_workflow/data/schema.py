"""
Schema definition and column metadata registry.

This module defines the authoritative COLUMN_SCHEMA derived from
create_equities_schema.sql, providing centralized datatype and role
information for all preprocessing, feature engineering, and modeling.

Schema: 318 columns total (262 original + 48 Phase 9.3 additions)

Database Tables:
- equities: Original table with per-region data loading
- all_stocks: Unified table combining four regional screening tables
  (screening_us, screening_eu, screening_apac, screening_rotw)
  Created by: all_stocks/all_stocks.sql
  Primary key: (Ticker, Region)
  Indexes: ticker, region, sector, industry, country, last_price, market_cap, sector_region

Data Loading:
- load_from_csv(): Load from CSV files in data/ directory
- load_from_db(): Load from equities table with Region filter
- load_from_all_stocks(): Load from unified all_stocks table (recommended)

Aligned with code_guidelines.md v1.3+ Schema and Datatype Management.
"""

from typing import Dict, List, Optional, Literal

# Type aliases for clarity
DType = Literal["float", "int", "string", "category", "datetime64[ns]", "bool"]
Role = Literal["id", "feature", "target", "target_fallback", "date", "auxiliary", "categorical"]


# Central column schema registry
# Maps normalized column names (lowercase, underscores) to dtype and role
COLUMN_SCHEMA: Dict[str, Dict[str, str]] = {
    # Identifiers
    "ticker": {"dtype": "string", "role": "id"},
    "isin": {"dtype": "string", "role": "id"},
    "name": {"dtype": "string", "role": "auxiliary"},
    "description": {"dtype": "string", "role": "auxiliary"},
    # Categorical/Classification
    "sector": {"dtype": "category", "role": "categorical"},
    "industry": {"dtype": "category", "role": "categorical"},
    "region": {"dtype": "category", "role": "categorical"},
    "country": {"dtype": "category", "role": "categorical"},
    "trading_country": {"dtype": "category", "role": "categorical"},
    "exchange": {"dtype": "category", "role": "categorical"},
    "unit": {"dtype": "string", "role": "auxiliary"},
    "style_class": {"dtype": "category", "role": "categorical"},
    "size_class": {"dtype": "category", "role": "categorical"},
    "flag": {"dtype": "string", "role": "auxiliary"},
    "next_earnings_status": {"dtype": "category", "role": "categorical"},
    # Date columns
    "last_updated": {"dtype": "datetime64[ns]", "role": "date"},
    "income_statement_report_date": {"dtype": "datetime64[ns]", "role": "date"},
    "next_earnings": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_announce_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_ex_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_payable_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_record_date": {"dtype": "datetime64[ns]", "role": "date"},
    # Target columns
    "last_price": {"dtype": "float", "role": "feature"},  # Also used as basis for targets
    "price_target": {"dtype": "float", "role": "target"},
    "price_target_ytd_ago": {"dtype": "float", "role": "target_fallback"},
    "price_target_low": {"dtype": "float", "role": "target_fallback"},
    "price_target_median": {"dtype": "float", "role": "target_fallback"},
    "price_target_high": {"dtype": "float", "role": "target_fallback"},
    "price_target_num": {"dtype": "float", "role": "auxiliary"},
    # Valuation metrics
    "market_cap": {"dtype": "float", "role": "feature"},
    "enterprise_value": {"dtype": "float", "role": "feature"},
    "p_e_ntm": {"dtype": "float", "role": "feature"},
    "p_e_ltm": {"dtype": "float", "role": "feature"},
    "p_e_1fyltm": {"dtype": "float", "role": "feature"},
    "p_b_ltm": {"dtype": "float", "role": "feature"},
    "p_b_1fy": {"dtype": "float", "role": "feature"},
    "p_b_5yavg": {"dtype": "float", "role": "feature"},
    "p_tbv_ltm": {"dtype": "float", "role": "feature"},
    "ev_sales_ltm": {"dtype": "float", "role": "feature"},
    "ev_sales_ntm": {"dtype": "float", "role": "feature"},
    "ev_sales_est_fy1": {"dtype": "float", "role": "feature"},
    "ev_ebitda_ltm": {"dtype": "float", "role": "feature"},
    "ev_ebitda_ntm": {"dtype": "float", "role": "feature"},
    "ev_ebitda_est_fy1": {"dtype": "float", "role": "feature"},
    "p_e_est_fy1": {"dtype": "float", "role": "feature"},
    # Phase 9.3 Schema 1.3: EV/Sales historical time-series
    "ev_sales_1fyltm": {"dtype": "float", "role": "feature"},
    "ev_sales_2fyltm": {"dtype": "float", "role": "feature"},
    "ev_sales_3fyltm": {"dtype": "float", "role": "feature"},
    "ev_sales_3yavgltm": {"dtype": "float", "role": "feature"},
    "ev_sales_1fqltm": {"dtype": "float", "role": "feature"},
    "ev_sales_2fqltm": {"dtype": "float", "role": "feature"},
    "ev_sales_3fqltm": {"dtype": "float", "role": "feature"},
    "ev_sales_4fqltm": {"dtype": "float", "role": "feature"},
    # Phase 9.3 Schema 1.3: EV/EBITDA historical time-series
    "ev_ebitda_1fyltm": {"dtype": "float", "role": "feature"},
    "ev_ebitda_1fqltm": {"dtype": "float", "role": "feature"},
    "ev_ebitda_3yavgltm": {"dtype": "float", "role": "feature"},
    # Phase 9.3 Schema 1.3: P/E extended time-series
    "p_e_2fyltm": {"dtype": "float", "role": "feature"},
    "p_e_3fyltm": {"dtype": "float", "role": "feature"},
    "p_e_3yavgltm": {"dtype": "float", "role": "feature"},
    "p_e_1fqltm": {"dtype": "float", "role": "feature"},
    "p_e_2fqltm": {"dtype": "float", "role": "feature"},
    "p_e_3fqltm": {"dtype": "float", "role": "feature"},
    "p_e_0fqqoqltm": {"dtype": "float", "role": "feature"},
    "p_e_0fyyoyltm": {"dtype": "float", "role": "feature"},
    "p_e_1fyyoyltm": {"dtype": "float", "role": "feature"},
    "p_e_0fqyoyltm": {"dtype": "float", "role": "feature"},
    # Risk & Quality
    "altman_z_score_fy": {"dtype": "float", "role": "feature"},
    "altman_z_score_fq": {"dtype": "float", "role": "feature"},
    "altman_z_score_ltm": {"dtype": "float", "role": "feature"},
    "beta_1y": {"dtype": "float", "role": "feature"},
    "beta_2y": {"dtype": "float", "role": "feature"},
    "beta_5y": {"dtype": "float", "role": "feature"},
    # Analyst metrics
    "analyst_rating": {"dtype": "float", "role": "feature"},
    "num_strong_sell_ratings": {"dtype": "float", "role": "feature"},
    "num_strong_buys_ratings": {"dtype": "float", "role": "feature"},
    "num_hold_ratings": {"dtype": "float", "role": "feature"},
    "num_buys_ratings": {"dtype": "float", "role": "feature"},
    "num_sell_ratings": {"dtype": "float", "role": "feature"},
    # Returns & Momentum
    "total_return_ytd": {"dtype": "float", "role": "feature"},
    "total_return_5y": {"dtype": "float", "role": "feature"},
    "total_return_10y": {"dtype": "float", "role": "feature"},
    "tot_return_pct_cagr_3y": {"dtype": "float", "role": "feature"},
    "tot_return_pct_cagr_10y": {"dtype": "float", "role": "feature"},
    "price_chg_pct_1m": {"dtype": "float", "role": "feature"},
    "price_chg_pct_3m": {"dtype": "float", "role": "feature"},
    "1_day_pct": {"dtype": "float", "role": "feature"},
    "price_5d_ago": {"dtype": "float", "role": "feature"},
    "price_1w_ago": {"dtype": "float", "role": "feature"},
    "price_1m_ago": {"dtype": "float", "role": "feature"},
    "price_3m_ago": {"dtype": "float", "role": "feature"},
    "price_6m_ago": {"dtype": "float", "role": "feature"},
    "price_1y_ago": {"dtype": "float", "role": "feature"},
    "price_3y_ago": {"dtype": "float", "role": "feature"},
    "price_5y_ago": {"dtype": "float", "role": "feature"},
    "price_qtd_ago": {"dtype": "float", "role": "feature"},
    # Technical indicators
    "ema_20d": {"dtype": "float", "role": "feature"},
    "ema_50d": {"dtype": "float", "role": "feature"},
    "ema_100d": {"dtype": "float", "role": "feature"},
    "ema_250d": {"dtype": "float", "role": "feature"},
    "52w_high_adj": {"dtype": "float", "role": "feature"},
    "52w_low_adj": {"dtype": "float", "role": "feature"},
    # Volatility
    "volatility_1m": {"dtype": "float", "role": "feature"},
    "volatility_3m": {"dtype": "float", "role": "feature"},
    "volatility_6m": {"dtype": "float", "role": "feature"},
    "volatility_1y": {"dtype": "float", "role": "feature"},
    # Volume & Trading
    "volume_shrs": {"dtype": "float", "role": "feature"},
    "rel_volume": {"dtype": "float", "role": "feature"},
    "shrs_out": {"dtype": "float", "role": "feature"},
    "shrs_out_1fy": {"dtype": "float", "role": "feature"},
    # Revenues & Growth
    "total_revenues_fy": {"dtype": "float", "role": "feature"},
    "total_revenues_ltm": {"dtype": "float", "role": "feature"},
    "total_revenues_fq": {"dtype": "float", "role": "feature"},
    "total_revenues_1fy": {"dtype": "float", "role": "feature"},
    "total_revenues_cagr_5y_fy": {"dtype": "float", "role": "feature"},
    "total_revenues_5yavgfq": {"dtype": "float", "role": "feature"},
    "total_revenues_5yavgltm": {"dtype": "float", "role": "feature"},
    "revenues_est_avg_ntm": {"dtype": "float", "role": "feature"},
    "revenues_est_avg_fy1e": {"dtype": "float", "role": "feature"},
    "revenues_est_med_ntm": {"dtype": "float", "role": "feature"},
    "revenues_est_med_fy1e": {"dtype": "float", "role": "feature"},
    "revenues_est_yoy_pct_fy1e": {"dtype": "float", "role": "feature"},
    "total_operating_expenses_ltm": {"dtype": "float", "role": "feature"},
    # Profitability - EBITDA
    "ebitda_fy": {"dtype": "float", "role": "feature"},
    "ebitda_ltm": {"dtype": "float", "role": "feature"},
    "ebitda_fq": {"dtype": "float", "role": "feature"},
    "ebitda_1fy": {"dtype": "float", "role": "feature"},
    "ebitda_adj_ltm": {"dtype": "float", "role": "feature"},
    "ebitda_adj_fy": {"dtype": "float", "role": "feature"},
    "ebitda_adj_1fy": {"dtype": "float", "role": "feature"},
    "ebitda_5yavgfq": {"dtype": "float", "role": "feature"},
    "ebitda_5yavgltm": {"dtype": "float", "role": "feature"},
    # Profitability - EBIT
    "ebit_fy": {"dtype": "float", "role": "feature"},
    "ebit_ltm": {"dtype": "float", "role": "feature"},
    "ebit_fq": {"dtype": "float", "role": "feature"},
    "ebit_1fy": {"dtype": "float", "role": "feature"},
    "ebit_adj_ltm": {"dtype": "float", "role": "feature"},
    "ebit_adj_fy": {"dtype": "float", "role": "feature"},
    "ebit_adj_1fy": {"dtype": "float", "role": "feature"},
    "ebit_est_med_fy1e": {"dtype": "float", "role": "feature"},
    "ebit_est_med_ntm": {"dtype": "float", "role": "feature"},
    "ebit_5yavgfq": {"dtype": "float", "role": "feature"},
    "ebit_5yavgltm": {"dtype": "float", "role": "feature"},
    # Profitability - Net Income
    "net_income_is_fy": {"dtype": "float", "role": "feature"},
    "net_income_is_ltm": {"dtype": "float", "role": "feature"},
    "net_income_is_fq": {"dtype": "float", "role": "feature"},
    "net_income_is_1fy": {"dtype": "float", "role": "feature"},
    "net_income_is_5yavgfq": {"dtype": "float", "role": "feature"},
    "net_income_is_5yavgltm": {"dtype": "float", "role": "feature"},
    "normalized_net_income_fy": {"dtype": "float", "role": "feature"},
    "normalized_net_income_ltm": {"dtype": "float", "role": "feature"},
    "normalized_net_income_fq": {"dtype": "float", "role": "feature"},
    "normalized_net_income_1fy": {"dtype": "float", "role": "feature"},
    "normalized_net_income_5yavgfq": {"dtype": "float", "role": "feature"},
    "normalized_net_income_5yavgltm": {"dtype": "float", "role": "feature"},
    "net_income_adj_fy": {"dtype": "float", "role": "feature"},
    "net_income_adj_ltm": {"dtype": "float", "role": "feature"},
    "net_income_adj_fq": {"dtype": "float", "role": "feature"},
    "net_income_adj_1fy": {"dtype": "float", "role": "feature"},
    "net_income_adj_5yavgfq": {"dtype": "float", "role": "feature"},
    "operating_income_ltm": {"dtype": "float", "role": "feature"},
    "operating_income_fy": {"dtype": "float", "role": "feature"},
    "operating_income_fq": {"dtype": "float", "role": "feature"},
    "operating_income_5yavgfq": {"dtype": "float", "role": "feature"},
    # Margins
    "net_income_margin_pct_fy": {"dtype": "float", "role": "feature"},
    "net_income_margin_pct_ltm": {"dtype": "float", "role": "feature"},
    "gross_profit_margin_pct_fy": {"dtype": "float", "role": "feature"},
    "gross_profit_margin_pct_ltm": {"dtype": "float", "role": "feature"},
    "gross_profit_ltm": {"dtype": "float", "role": "feature"},
    "gross_profit_fy": {"dtype": "float", "role": "feature"},
    # Returns on Capital
    "return_on_equity_pct_ltm": {"dtype": "float", "role": "feature"},
    "return_on_equity_pct_fy": {"dtype": "float", "role": "feature"},
    "return_on_assets_roa_pct_ltm": {"dtype": "float", "role": "feature"},
    "return_on_assets_roa_pct_fy": {"dtype": "float", "role": "feature"},
    # Cash Flow
    "cfo_ltm": {"dtype": "float", "role": "feature"},
    "cfo_fy": {"dtype": "float", "role": "feature"},
    "cfo_fq": {"dtype": "float", "role": "feature"},
    "cfo_1fy": {"dtype": "float", "role": "feature"},
    "fcf_ltm": {"dtype": "float", "role": "feature"},
    "fcf_fy": {"dtype": "float", "role": "feature"},
    "fcf_fq": {"dtype": "float", "role": "feature"},
    "fcf_5yavgfq": {"dtype": "float", "role": "feature"},
    "cfi_ltm": {"dtype": "float", "role": "feature"},
    "cfi_fy": {"dtype": "float", "role": "feature"},
    "cfi_fq": {"dtype": "float", "role": "feature"},
    "cfi_1fy": {"dtype": "float", "role": "feature"},
    "cff_ltm": {"dtype": "float", "role": "feature"},
    "cff_fy": {"dtype": "float", "role": "feature"},
    "cff_fq": {"dtype": "float", "role": "feature"},
    "cff_1fy": {"dtype": "float", "role": "feature"},
    # Balance Sheet
    "total_assets_ltm": {"dtype": "float", "role": "feature"},
    "total_assets_fy": {"dtype": "float", "role": "feature"},
    "total_equity_fy": {"dtype": "float", "role": "feature"},
    "total_equity_ltm": {"dtype": "float", "role": "feature"},
    "total_debt_fy": {"dtype": "float", "role": "feature"},
    "total_debt_ltm": {"dtype": "float", "role": "feature"},
    "total_current_assets_ltm": {"dtype": "float", "role": "feature"},
    "total_current_liabilities_ltm": {"dtype": "float", "role": "feature"},
    "current_ratio_fy": {"dtype": "float", "role": "feature"},
    "current_ratio_ltm": {"dtype": "float", "role": "feature"},
    "working_capital_ltm": {"dtype": "float", "role": "feature"},
    "working_capital_fq": {"dtype": "float", "role": "feature"},
    "working_capital_fy": {"dtype": "float", "role": "feature"},
    "working_capital_5yavgfy": {"dtype": "float", "role": "feature"},
    "tbv_fy": {"dtype": "float", "role": "feature"},
    "tbv_ltm": {"dtype": "float", "role": "feature"},
    "cash_and_equivalents_ltm": {"dtype": "float", "role": "feature"},
    "cash_and_equivalents_fq": {"dtype": "float", "role": "feature"},
    "cash_and_equivalents_fy": {"dtype": "float", "role": "feature"},
    "cash_and_equivalents_5yavgfq": {"dtype": "float", "role": "feature"},
    "retained_earnings_ltm": {"dtype": "float", "role": "feature"},
    "retained_earnings_fq": {"dtype": "float", "role": "feature"},
    "retained_earnings_fy": {"dtype": "float", "role": "feature"},
    "retained_earnings_5yavgfq": {"dtype": "float", "role": "feature"},
    # Asset Details
    "inventory_ltm": {"dtype": "float", "role": "feature"},
    "inventory_fq": {"dtype": "float", "role": "feature"},
    "inventory_fy": {"dtype": "float", "role": "feature"},
    "inventory_5yavgfq": {"dtype": "float", "role": "feature"},
    "goodwill_fq": {"dtype": "float", "role": "feature"},
    "goodwill_ltm": {"dtype": "float", "role": "feature"},
    "goodwill_fy": {"dtype": "float", "role": "feature"},
    "goodwill_1fy": {"dtype": "float", "role": "feature"},
    "goodwill_5yavgfq": {"dtype": "float", "role": "feature"},
    "intangible_assets": {"dtype": "float", "role": "feature"},  # Base column (no time suffix)
    "gross_intangible_assets_ltm": {"dtype": "float", "role": "feature"},
    "gross_intangible_assets_fy": {"dtype": "float", "role": "feature"},
    "gross_intangible_assets_5yavgfq": {"dtype": "float", "role": "feature"},
    # Capex & Investments
    "capital_expenditure_ltm": {"dtype": "float", "role": "feature"},
    "capital_expenditure_fy": {"dtype": "float", "role": "feature"},
    "capital_expenditure_fq": {"dtype": "float", "role": "feature"},
    "capital_expenditure_1fy": {"dtype": "float", "role": "feature"},
    "capital_expenditure_5yavgfq": {"dtype": "float", "role": "feature"},
    "asset_turnover_fy": {"dtype": "float", "role": "feature"},
    "asset_turnover_ltm": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_ltm": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_fy": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_fq": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_1fy": {"dtype": "float", "role": "feature"},
    "cash_acquisitions_5yavgfq": {"dtype": "float", "role": "feature"},
    # Exceptional Items
    "impairment_of_goodwill_fq": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_ltm": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_1fy": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_fy": {"dtype": "float", "role": "feature"},
    "impairment_of_goodwill_5yavgfq": {"dtype": "float", "role": "feature"},
    "asset_writedown_ltm": {"dtype": "float", "role": "feature"},
    "asset_writedown_fy": {"dtype": "float", "role": "feature"},
    "asset_writedown_fq": {"dtype": "float", "role": "feature"},
    "asset_writedown_1fy": {"dtype": "float", "role": "feature"},
    "asset_writedown_5yavgfq": {"dtype": "float", "role": "feature"},
    "restructuring_charges_ltm": {"dtype": "float", "role": "feature"},
    "restructuring_charges_fq": {"dtype": "float", "role": "feature"},
    "restructuring_charges_1fy": {"dtype": "float", "role": "feature"},
    "restructuring_charges_fy": {"dtype": "float", "role": "feature"},
    "restructuring_charges_5yavgfq": {"dtype": "float", "role": "feature"},
    "merger_restructuring_charges_ltm": {"dtype": "float", "role": "feature"},
    "merger_restructuring_charges_fq": {"dtype": "float", "role": "feature"},
    "merger_restructuring_charges_fy": {"dtype": "float", "role": "feature"},
    "merger_restructuring_charges_5yavgfq": {"dtype": "float", "role": "feature"},
    "other_unusual_items_total_ltm": {"dtype": "float", "role": "feature"},
    "gain_loss_on_sale_of_assets_ltm": {"dtype": "float", "role": "feature"},
    # Operating Expenses
    "cost_of_revenues_ltm": {"dtype": "float", "role": "feature"},
    "r_d_expenses": {"dtype": "float", "role": "feature"},  # Base column (no time suffix)
    "r_d_expenses_ltm": {"dtype": "float", "role": "feature"},
    "selling_general_admin_expenses_total_fq": {"dtype": "float", "role": "feature"},
    "selling_general_admin_expenses_total_fy": {"dtype": "float", "role": "feature"},
    "selling_general_admin_expenses_total_1fy": {"dtype": "float", "role": "feature"},
    "selling_general_admin_expenses_total_5yavgfq": {"dtype": "float", "role": "feature"},
    # SG&A with "and" (correct normalization from "Selling General & Admin Expenses/Total")
    "selling_general_and_admin_expenses_total_fq": {"dtype": "float", "role": "feature"},
    "selling_general_and_admin_expenses_total_fy": {"dtype": "float", "role": "feature"},
    "selling_general_and_admin_expenses_total_1fy": {"dtype": "float", "role": "feature"},
    "selling_general_and_admin_expenses_total_5yavgfq": {"dtype": "float", "role": "feature"},
    "accounts_receivable_total_fy": {"dtype": "float", "role": "feature"},
    "accounts_receivable_total_1fy": {"dtype": "float", "role": "feature"},
    "accounts_receivable_total_5yavgfq": {"dtype": "float", "role": "feature"},
    "marketing_expenses": {"dtype": "float", "role": "feature"},  # Base column (no time suffix)
    "marketing_expenses_fq": {"dtype": "float", "role": "feature"},
    "marketing_expenses_fy": {"dtype": "float", "role": "feature"},
    "marketing_expenses_1fy": {"dtype": "float", "role": "feature"},
    "marketing_expenses_5yavgltm": {"dtype": "float", "role": "feature"},
    # Earnings Per Share
    "eps_adj_1fy": {"dtype": "float", "role": "feature"},
    "eps_adj_fy": {"dtype": "float", "role": "feature"},
    "eps_adj_ltm": {"dtype": "float", "role": "feature"},
    "eps_norm_est_avg_ntm": {"dtype": "float", "role": "feature"},
    "eps_norm_est_avg_fy1e": {"dtype": "float", "role": "feature"},
    "eps_previous_year": {"dtype": "float", "role": "feature"},  # Base column for YoY calculations
    # Dividends
    "dividend_per_share_ltm": {"dtype": "float", "role": "feature"},
    "div_yield_ind": {"dtype": "float", "role": "feature"},
    "div_yield_ltm": {"dtype": "float", "role": "feature"},
    "div_yield_ttm": {"dtype": "float", "role": "feature"},
    "div_yield_ntm": {"dtype": "float", "role": "feature"},
    "div_yield_1fyind": {"dtype": "float", "role": "feature"},
    "div_yield_5yavgltm": {"dtype": "float", "role": "feature"},
    "common_dividends_paid_ltm": {"dtype": "float", "role": "feature"},
    "common_dividends_paid_fy": {"dtype": "float", "role": "feature"},
    "dividend_record_frequency": {"dtype": "string", "role": "auxiliary"},
    "dividend_record_currency": {"dtype": "string", "role": "auxiliary"},
    "dividend_record_amount": {"dtype": "float", "role": "feature"},
    "dividend_streak": {"dtype": "float", "role": "feature"},
    "buyback_yield_ltm": {"dtype": "float", "role": "feature"},
    # Interest & Financing
    "interest_expense_total_ltm": {"dtype": "float", "role": "feature"},
    "interest_income_on_investments_ltm": {"dtype": "float", "role": "feature"},
    # Employees
    "employees": {"dtype": "int", "role": "feature"},  # Base column (current employee count)
    "avg_employees_ltm": {"dtype": "float", "role": "feature"},
    "avg_employees_fy": {"dtype": "float", "role": "feature"},
    "avg_employees_5yavgfy": {"dtype": "float", "role": "feature"},
    "total_employees_fy": {"dtype": "float", "role": "feature"},
    "total_employees_fq": {"dtype": "float", "role": "feature"},
    # Country-specific
    "market_cap_country_r": {"dtype": "float", "role": "feature"},
    # ==================================================================================
    # NORMALIZATION VARIANTS & SIMPLIFIED ALIASES
    # Added to resolve unknown column warnings from dtype_diagnostics.json
    # These columns exist in the data pipeline but use different naming conventions
    # ==================================================================================
    # Analyst Ratings (normalized names without "num_" prefix)
    "price_target_count": {"dtype": "float", "role": "auxiliary"},  # Alias for price_target_num
    "strong_sell_ratings": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for num_strong_sell_ratings
    "strong_buys_ratings": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for num_strong_buys_ratings
    "hold_ratings": {"dtype": "float", "role": "feature"},  # Alias for num_hold_ratings
    "buys_ratings": {"dtype": "float", "role": "feature"},  # Alias for num_buys_ratings
    "sell_ratings": {"dtype": "float", "role": "feature"},  # Alias for num_sell_ratings
    # Simplified Base Columns (without time suffixes - used as generic references)
    "p_e": {"dtype": "float", "role": "feature"},  # Generic P/E ratio
    "p_b": {"dtype": "float", "role": "feature"},  # Generic P/B ratio
    "revenue": {"dtype": "float", "role": "feature"},  # Generic revenue
    "ebitda": {"dtype": "float", "role": "feature"},  # Generic EBITDA
    "ebit": {"dtype": "float", "role": "feature"},  # Generic EBIT
    "net_income": {"dtype": "float", "role": "feature"},  # Generic net income
    "net_income_ltm": {"dtype": "float", "role": "feature"},  # Duplicate of net_income_is_ltm
    "gross_margin": {"dtype": "float", "role": "feature"},  # Generic gross margin
    "eps": {"dtype": "float", "role": "feature"},  # Generic EPS
    "total_equity": {"dtype": "float", "role": "feature"},  # Generic total equity
    "total_assets": {"dtype": "float", "role": "feature"},  # Generic total assets
    "total_debt": {"dtype": "float", "role": "feature"},  # Generic total debt
    "inventory": {"dtype": "float", "role": "feature"},  # Generic inventory
    "capex": {"dtype": "float", "role": "feature"},  # Generic capital expenditure
    "cash_and_equivalents": {"dtype": "float", "role": "feature"},  # Generic cash
    "current_assets": {"dtype": "float", "role": "feature"},  # Generic current assets
    "current_liabilities": {"dtype": "float", "role": "feature"},  # Generic current liabilities
    "working_capital": {"dtype": "float", "role": "feature"},  # Generic working capital
    "retained_earnings": {"dtype": "float", "role": "feature"},  # Generic retained earnings
    "cfo": {"dtype": "float", "role": "feature"},  # Generic cash flow from operations
    "cfi": {"dtype": "float", "role": "feature"},  # Generic cash flow from investing
    "cff": {"dtype": "float", "role": "feature"},  # Generic cash flow from financing
    "fcf": {"dtype": "float", "role": "feature"},  # Generic free cash flow
    "gross_profit": {"dtype": "float", "role": "feature"},  # Generic gross profit
    "operating_income": {"dtype": "float", "role": "feature"},  # Generic operating income
    "interest_expense": {"dtype": "float", "role": "feature"},  # Generic interest expense
    "goodwill": {"dtype": "float", "role": "feature"},  # Generic goodwill
    "dividend_per_share": {"dtype": "float", "role": "feature"},  # Generic dividend per share
    "operating_expenses": {"dtype": "float", "role": "feature"},  # Generic operating expenses
    "operating_cash_flow": {"dtype": "float", "role": "feature"},  # Alias for cfo
    "dividends_paid": {"dtype": "float", "role": "feature"},  # Generic dividends paid
    "dividends_paid_ltm": {"dtype": "float", "role": "feature"},  # Dividends paid LTM
    # Additional normalized names
    "price_target_number": {"dtype": "float", "role": "auxiliary"},  # Alias for price_target_num
    "one_day_pct": {"dtype": "float", "role": "feature"},  # Alias for 1_day_pct
    "shares_outstanding": {"dtype": "float", "role": "feature"},  # Alias for shrs_out
    "p_e_5yavgltm": {"dtype": "float", "role": "feature"},  # 5-year average P/E LTM
    # SG&A Expenses (normalized naming)
    "sga_expenses": {"dtype": "float", "role": "feature"},  # Generic SG&A
    "sga_expenses_fq": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for selling_general_admin_expenses_total_fq
    "sga_expenses_fy": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for selling_general_admin_expenses_total_fy
    "sga_expenses_1fy": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for selling_general_admin_expenses_total_1fy
    "sga_expenses_5yavgfq": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for selling_general_admin_expenses_total_5yavgfq
    # Accounts Receivable (normalized naming)
    "accounts_receivable_fy": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for accounts_receivable_total_fy
    "accounts_receivable_1fy": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for accounts_receivable_total_1fy
    "accounts_receivable_5yavgfq": {
        "dtype": "float",
        "role": "feature",
    },  # Alias for accounts_receivable_total_5yavgfq
    # ==================================================================================
    # DERIVED & COMPUTED COLUMNS (Created during preprocessing/feature engineering)
    # ==================================================================================
    # Volatility percentage variants
    "volatility_1y_pct": {"dtype": "float", "role": "feature"},  # 1-year volatility as percentage
    # Year-over-Year (YoY) comparison columns (_previous_year suffix)
    "revenue_previous_year": {"dtype": "float", "role": "feature"},  # Revenue from previous year
    "ebitda_previous_year": {"dtype": "float", "role": "feature"},  # EBITDA from previous year
    "total_equity_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Total equity from previous year
    "total_assets_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Total assets from previous year
    "gross_profit_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Gross profit from previous year
    "accounts_receivable_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # AR from previous year
    "roa_previous_year": {"dtype": "float", "role": "feature"},  # ROA from previous year
    "current_ratio_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Current ratio from previous year
    "shares_outstanding_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Shares outstanding from previous year
    "gross_margin_pct_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Gross margin % from previous year
    "asset_turnover_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Asset turnover from previous year
    # Fiscal year variants (alternative naming)
    "revenue_fy": {"dtype": "float", "role": "feature"},  # Alias for total_revenues_fy
    "working_capital_1fy": {"dtype": "float", "role": "feature"},  # Working capital 1 fiscal year
}


# Phase 9.3 Feature Input Categorization
# Maps feature engineering buckets to required input columns
PHASE93_FEATURE_INPUTS: Dict[str, List[str]] = {
    "momentum": [
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "1_day_pct",
        "price_1m_ago",
        "price_3m_ago",
        "price_6m_ago",
        "ema_20d",
        "ema_50d",
        "ema_100d",
        "ema_250d",
        "52w_high_adj",
        "52w_low_adj",
        "total_return_ytd",
        "total_return_5y",
    ],
    "valuation": [
        "p_e_ltm",
        "p_e_ntm",
        "p_b_ltm",
        "p_tbv_ltm",
        "ev_sales_ltm",
        "ev_ebitda_ltm",
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target_median",
        # Phase 9.3 Schema 1.3: EV/Sales historical time-series
        "ev_sales_est_fy1",
        "ev_sales_1fyltm",
        "ev_sales_2fyltm",
        "ev_sales_3fyltm",
        "ev_sales_3yavgltm",
        "ev_sales_ntm",
        "ev_sales_1fqltm",
        "ev_sales_2fqltm",
        "ev_sales_3fqltm",
        "ev_sales_4fqltm",
        # Phase 9.3 Schema 1.3: EV/EBITDA historical time-series
        "ev_ebitda_est_fy1",
        "ev_ebitda_ntm",
        "ev_ebitda_1fyltm",
        "ev_ebitda_1fqltm",
        "ev_ebitda_3yavgltm",
        # Phase 9.3 Schema 1.3: P/E extended time-series
        "p_e_est_fy1",
        "p_e_1fyltm",
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
    ],
    "profitability": [
        "net_income_margin_pct_ltm",
        "gross_profit_margin_pct_ltm",
        "ebitda_ltm",
        "ebit_ltm",
        "net_income_is_ltm",
        "operating_income_ltm",
        "gross_profit_ltm",
    ],
    "quality_risk": [
        "altman_z_score_ltm",
        "return_on_equity_pct_ltm",
        "return_on_assets_roa_pct_ltm",
        "beta_1y",
        "volatility_1m",
        "volatility_3m",
        "current_ratio_ltm",
        "total_debt_ltm",
        "total_equity_ltm",
    ],
    "cash_flow": [
        "cfo_ltm",
        "fcf_ltm",
        "cfi_ltm",
        "cff_ltm",
        "capital_expenditure_ltm",
    ],
    "growth": [
        "total_revenues_cagr_5y_fy",
        "revenues_est_yoy_pct_fy1e",
        "tot_return_pct_cagr_3y",
        "total_revenues_ltm",
        "total_revenues_fy",
    ],
}


# Helper functions for schema access


def get_expected_dtype(column: str) -> Optional[str]:
    """
    Get the expected dtype for a column from COLUMN_SCHEMA.

    Args:
        column: Column name (normalized: lowercase with underscores)

    Returns:
        Expected dtype string or None if column not in schema
    """
    col_lower = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")
    )
    if col_lower in COLUMN_SCHEMA:
        return COLUMN_SCHEMA[col_lower]["dtype"]
    return None


def get_column_role(column: str) -> Optional[str]:
    """
    Get the role of a column from COLUMN_SCHEMA.

    Args:
        column: Column name (normalized: lowercase with underscores)

    Returns:
        Role string or None if column not in schema
    """
    col_lower = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")
    )
    if col_lower in COLUMN_SCHEMA:
        return COLUMN_SCHEMA[col_lower]["role"]
    return None


def list_numeric_feature_cols() -> List[str]:
    """
    List all numeric feature columns from COLUMN_SCHEMA.

    Returns:
        List of column names with numeric dtypes and feature/target roles
    """
    numeric_dtypes = {"float", "int"}
    feature_roles = {"feature", "target", "target_fallback"}

    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta["dtype"] in numeric_dtypes and meta["role"] in feature_roles
    ]


def list_categorical_cols() -> List[str]:
    """
    List all categorical columns from COLUMN_SCHEMA.

    Returns:
        List of column names with category or categorical role
    """
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta["dtype"] == "category" or meta["role"] == "categorical"
    ]


def list_date_cols() -> List[str]:
    """
    List all date/datetime columns from COLUMN_SCHEMA.

    Returns:
        List of column names with datetime64[ns] dtype or date role
    """
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta["dtype"] == "datetime64[ns]" or meta["role"] == "date"
    ]


def normalize_column_name(column: str) -> str:
    """
    Normalize a column name to match COLUMN_SCHEMA keys.

    Converts to lowercase, replaces spaces/special chars with underscores.

    Args:
        column: Original column name (e.g., "Last Price" or "P/E (LTM)")

    Returns:
        Normalized column name (e.g., "last_price" or "p_e_ltm")
    """
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
    # Remove consecutive underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    # Remove leading/trailing underscores
    return normalized.strip("_")
