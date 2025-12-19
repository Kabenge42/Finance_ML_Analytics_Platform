"""
Schema definition and column metadata registry.

This module defines the authoritative COLUMN_SCHEMA derived from
create_equities_schema.sql, providing centralized datatype and role
information for all preprocessing, feature engineering, and modeling.

Schema Structure (v1.13 - Updated 2025-12-19):
- Source columns from CSV/SQL: 334 (matching create_equities_schema.sql + forward estimates)
- Total COLUMN_SCHEMA entries: 555
  - 334 source columns (from CSV/SQL schema + forward estimates)
  - 61 log-transformed columns (ETL-generated, log1p of market values)
  - 43 legacy aliases (role=auxiliary, for backward compatibility)
  - 36 generic base columns (no time suffix)
  - 34 conditional metrics (with _applicable flags)
  - 26 derived ratios and percentage metrics (ETL semantic transforms)
  - 4 Phase 9.3 composite quality scores (altman_z_score, beneish_m_score, etc.)
  - 17 engineered features (volatility variants, MA aliases, sector-specific, interactions)

New in v1.13:
  - 3 volatility variants (30d, 60d, 90d)
  - 2 simple moving average aliases (ma_20d, ma_50d)
  - 1 temporal dividend feature (days_to_dividend)
  - 6 sector-specific engineered features (tangible_book_value, marketing_efficiency,
    r_d_intensity, rule_of_40, operating_leverage, one_day_chg)
  - 5 interaction features (market_cap×roe, p_e_ratio×debt_to_equity, etc.)
  - Total: 17 engineered features to resolve ETL pipeline warnings

New in v1.12:
  - 4 dividend yield forward estimates (2fy, 3fy, 4fy, 5fy)
  - 2 EBITDA forward estimates (avg_fy1e, avg_ntm)
  - 6 EPS normalized estimate revisions (1w, 1m, 3m, 6m, 1y, analyst count)
  - 6 EPS GAAP estimates & revisions (fy1e, ntm, 1m, 3m, 6m, 1y)
  - 12 basic EPS historical metrics (ltm, fq, fy, 1fqfq-4fqfq, 1fy-5fy)
  - Total: 35 new raw source columns

Database Tables:
- equities: Original table with per-region data loading
- all_stocks: Unified table combining four regional screening tables
  (screening_us, screening_eu, screening_apac, screening_rotw)
  Created by: equities/import_equities_data.sql
  Primary key: (Ticker, Region)
  Indexes: ticker, region, sector, industry, country, last_price, market_cap, sector_region

Data Loading:
- load_from_csv(): Load from CSV files in data/ directory
- load_from_db(): Load from equities table with Region filter
- load_from_all_stocks(): Load from unified all_stocks table (recommended)

Aligned with code_guidelines.md v1.12+ Schema and Datatype Management.
"""

from typing import Dict, List, Optional, Literal

# =============================================================================
# Type Aliases for Schema Definition (Aligned with code_guidelines.md v1.11)
# =============================================================================

# DType: Maps to pandas/numpy dtype strings for ETL casting
DType = Literal[
    "float",  # float64 - default for numeric financial data
    "int",  # int64 - discrete counts, integer IDs
    "string",  # object/string - text data
    "category",  # pandas Categorical - low-cardinality (sector, region)
    "datetime64[ns]",  # datetime columns
    "bool",  # boolean flags
]

# Role: Semantic role determining preprocessing and pipeline treatment
Role = Literal[
    # === Pipeline Stage Roles ===
    "id",  # Identifier columns (ticker, isin, name) - never used as features
    "target",  # Primary prediction target (price_target)
    "target_fallback",  # Alternative targets (price_target_median, last_price)
    "date",  # Temporal columns for time-series features
    "categorical",  # Grouping columns (sector, region, industry)
    "auxiliary",  # Legacy aliases, optional - excluded from diagnostics
    "feature",  # General ML features not in other categories
    # === Semantic Classification Roles (code_guidelines.md v1.11, Section 8.5) ===
    "price",  # Price columns - NEVER transform (21 columns)
    "market_value",  # Market cap, revenue, assets - log-transform recommended
    "ratio",  # Pre-normalized ratios - skip winsorization
    "percentage",  # Bounded [0,100] metrics - margins, growth rates
    "count",  # Discrete integers - analyst ratings, employees
    "label",  # Classification targets (multi-label)
]


# Central column schema registry
# Maps normalized column names (lowercase, underscores) to dtype and role
COLUMN_SCHEMA: Dict[str, Dict[str, str]] = {
    # Identifiers
    "ticker": {"dtype": "string", "role": "id"},
    "isin": {"dtype": "string", "role": "id"},
    "name": {"dtype": "string", "role": "id"},
    "description": {"dtype": "string", "role": "auxiliary"},
    # Categorical/Classification
    "sector": {"dtype": "category", "role": "categorical"},
    "industry": {"dtype": "category", "role": "categorical"},
    "region": {"dtype": "category", "role": "categorical"},
    "country": {"dtype": "category", "role": "categorical"},
    "trading_country": {"dtype": "category", "role": "categorical"},
    "exchange": {"dtype": "category", "role": "categorical"},
    "unit": {"dtype": "string", "role": "categorical"},
    "style_class": {"dtype": "category", "role": "categorical"},
    "size_class": {"dtype": "category", "role": "categorical"},
    "next_earnings_status": {"dtype": "category", "role": "categorical"},
    # Date columns
    "last_updated": {"dtype": "datetime64[ns]", "role": "date"},
    "income_statement_report_date": {"dtype": "datetime64[ns]", "role": "date"},
    "fy_end": {"dtype": "datetime64[ns]", "role": "date"},
    "next_earnings": {"dtype": "datetime64[ns]", "role": "date"},
    "next_earnings_when": {"dtype": "category", "role": "categorical"},
    "dividend_record_announce_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_ex_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_payable_date": {"dtype": "datetime64[ns]", "role": "date"},
    "dividend_record_record_date": {"dtype": "datetime64[ns]", "role": "date"},
    # ====================
    # PRICE COLUMNS - NEVER transform (preserve original dollar units)
    # ====================
    "last_price": {"dtype": "float", "role": "price"},
    "price_target": {"dtype": "float", "role": "target"},
    "price_target_ytd_ago": {"dtype": "float", "role": "price"},
    "price_target_low": {"dtype": "float", "role": "price"},
    "price_target_median": {"dtype": "float", "role": "target_fallback"},
    "price_target_high": {"dtype": "float", "role": "price"},
    "price_target_num": {"dtype": "float", "role": "auxiliary"},  # Legacy alias
    "price_target_count": {"dtype": "float", "role": "count"},
    "price_5d_ago": {"dtype": "float", "role": "price"},
    "price_1w_ago": {"dtype": "float", "role": "price"},
    "price_1m_ago": {"dtype": "float", "role": "price"},
    "price_3m_ago": {"dtype": "float", "role": "price"},
    "price_6m_ago": {"dtype": "float", "role": "price"},
    "price_1y_ago": {"dtype": "float", "role": "price"},
    "price_3y_ago": {"dtype": "float", "role": "price"},
    "price_5y_ago": {"dtype": "float", "role": "price"},
    "price_qtd_ago": {"dtype": "float", "role": "price"},
    # ====================
    # MARKET VALUE COLUMNS - Log-transform recommended
    # ====================
    "market_cap": {"dtype": "float", "role": "market_value"},
    "enterprise_value": {"dtype": "float", "role": "market_value"},
    "market_cap_country_r": {"dtype": "float", "role": "market_value"},
    # ====================
    # RATIO COLUMNS - Pre-normalized financial ratios
    # ====================
    "p_e_ntm": {"dtype": "float", "role": "ratio"},
    "p_e_ltm": {"dtype": "float", "role": "ratio"},
    "p_e_1fyltm": {"dtype": "float", "role": "ratio"},
    "p_b_ltm": {"dtype": "float", "role": "ratio"},
    "p_b_1fy": {"dtype": "float", "role": "ratio"},
    "p_b_5yavg": {"dtype": "float", "role": "ratio"},
    "p_tbv_ltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_ltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_ntm": {"dtype": "float", "role": "ratio"},
    "ev_sales_est_fy1": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_ltm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_ntm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_est_fy1": {"dtype": "float", "role": "ratio"},
    "p_e_est_fy1": {"dtype": "float", "role": "ratio"},
    # Phase 9.3 Schema 1.3: EV/Sales historical time-series (ratio)
    "ev_sales_1fyltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_2fyltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_3fyltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_3yavgltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_1fqltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_2fqltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_3fqltm": {"dtype": "float", "role": "ratio"},
    "ev_sales_4fqltm": {"dtype": "float", "role": "ratio"},
    # Phase 9.3 Schema 1.3: EV/EBITDA historical time-series (ratio)
    "ev_ebitda_1fyltm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_1fqltm": {"dtype": "float", "role": "ratio"},
    "ev_ebitda_3yavgltm": {"dtype": "float", "role": "ratio"},
    # Phase 9.3 Schema 1.3: P/E extended time-series (ratio)
    "p_e_2fyltm": {"dtype": "float", "role": "ratio"},
    "p_e_3fyltm": {"dtype": "float", "role": "ratio"},
    "p_e_3yavgltm": {"dtype": "float", "role": "ratio"},
    "p_e_1fqltm": {"dtype": "float", "role": "ratio"},
    "p_e_2fqltm": {"dtype": "float", "role": "ratio"},
    "p_e_3fqltm": {"dtype": "float", "role": "ratio"},
    "p_e_5yavgltm": {"dtype": "float", "role": "ratio"},
    "p_e_0fqqoqltm": {"dtype": "float", "role": "ratio"},
    "p_e_0fyyoyltm": {"dtype": "float", "role": "ratio"},
    "p_e_1fyyoyltm": {"dtype": "float", "role": "ratio"},
    "p_e_0fqyoyltm": {"dtype": "float", "role": "ratio"},
    # Risk & Quality (ratio)
    "altman_z_score_fy": {"dtype": "float", "role": "ratio"},
    "altman_z_score_fq": {"dtype": "float", "role": "ratio"},
    "altman_z_score_ltm": {"dtype": "float", "role": "ratio"},
    # ====================
    # PERCENTAGE COLUMNS - Bounded metrics
    # ====================
    "beta_1y": {"dtype": "float", "role": "percentage"},
    "beta_2y": {"dtype": "float", "role": "percentage"},
    "beta_5y": {"dtype": "float", "role": "percentage"},
    "total_return_ytd": {"dtype": "float", "role": "percentage"},
    "total_return_5y": {"dtype": "float", "role": "percentage"},
    "total_return_10y": {"dtype": "float", "role": "percentage"},
    "tot_return_pct_cagr_3y": {"dtype": "float", "role": "percentage"},
    "tot_return_pct_cagr_10y": {"dtype": "float", "role": "percentage"},
    "price_chg_pct_1m": {"dtype": "float", "role": "percentage"},
    "price_chg_pct_3m": {"dtype": "float", "role": "percentage"},
    "1_day_pct": {"dtype": "float", "role": "auxiliary"},  # Legacy alias
    "one_day_pct": {"dtype": "float", "role": "percentage"},
    # ====================
    # COUNT COLUMNS - Discrete integers
    # ====================
    "analyst_rating": {"dtype": "float", "role": "count"},
    "num_strong_sell_ratings": {"dtype": "float", "role": "count"},
    "num_strong_buys_ratings": {"dtype": "float", "role": "count"},
    "num_hold_ratings": {"dtype": "float", "role": "count"},
    "num_buys_ratings": {"dtype": "float", "role": "count"},
    "num_sell_ratings": {"dtype": "float", "role": "count"},
    # ====================
    # TECHNICAL INDICATORS - Price-based (role: price)
    # ====================
    "ema_20d": {"dtype": "float", "role": "price"},
    "ema_50d": {"dtype": "float", "role": "price"},
    "ema_100d": {"dtype": "float", "role": "price"},
    "ema_250d": {"dtype": "float", "role": "price"},
    "ma_20d": {"dtype": "float", "role": "price"},  # Simple MA (alias for ema_20d)
    "ma_50d": {"dtype": "float", "role": "price"},  # Simple MA (alias for ema_50d)
    "52w_high_adj": {"dtype": "float", "role": "price"},
    "52w_low_adj": {"dtype": "float", "role": "price"},
    # ====================
    # VOLATILITY - Percentage metrics
    # ====================
    "volatility_1m": {"dtype": "float", "role": "percentage"},
    "volatility_30d": {"dtype": "float", "role": "percentage"},
    "volatility_3m": {"dtype": "float", "role": "percentage"},
    "volatility_60d": {"dtype": "float", "role": "percentage"},
    "volatility_6m": {"dtype": "float", "role": "percentage"},
    "volatility_90d": {"dtype": "float", "role": "percentage"},
    "volatility_1y": {"dtype": "float", "role": "percentage"},
    # ====================
    # VOLUME & TRADING - Market value/count
    # ====================
    "volume_shrs": {"dtype": "float", "role": "market_value"},
    "rel_volume": {"dtype": "float", "role": "ratio"},
    "shrs_out": {"dtype": "float", "role": "auxiliary"},  # Legacy alias
    "shares_outstanding": {"dtype": "float", "role": "count"},
    "shrs_out_1fy": {"dtype": "float", "role": "count"},
    # ====================
    # REVENUES & GROWTH - Market value columns
    # ====================
    "total_revenues_fy": {"dtype": "float", "role": "market_value"},
    "total_revenues_ltm": {"dtype": "float", "role": "market_value"},
    "total_revenues_fq": {"dtype": "float", "role": "market_value"},
    "total_revenues_1fy": {"dtype": "float", "role": "market_value"},
    "total_revenues_cagr_5y_fy": {"dtype": "float", "role": "percentage"},
    "total_revenues_5yavgfq": {"dtype": "float", "role": "market_value"},
    "total_revenues_5yavgltm": {"dtype": "float", "role": "market_value"},
    "revenues_est_avg_ntm": {"dtype": "float", "role": "market_value"},
    "revenues_est_avg_fy1e": {"dtype": "float", "role": "market_value"},
    "revenues_est_med_ntm": {"dtype": "float", "role": "market_value"},
    "revenues_est_med_fy1e": {"dtype": "float", "role": "market_value"},
    "revenues_est_yoy_pct_fy1e": {"dtype": "float", "role": "percentage"},
    "total_operating_expenses_ltm": {"dtype": "float", "role": "market_value"},
    # ====================
    # PROFITABILITY - EBITDA (market_value)
    # ====================
    "ebitda_fy": {"dtype": "float", "role": "market_value"},
    "ebitda_ltm": {"dtype": "float", "role": "market_value"},
    "ebitda_fq": {"dtype": "float", "role": "market_value"},
    "ebitda_1fy": {"dtype": "float", "role": "market_value"},
    "ebitda_adj_ltm": {"dtype": "float", "role": "market_value"},
    "ebitda_adj_fy": {"dtype": "float", "role": "market_value"},
    "ebitda_adj_1fy": {"dtype": "float", "role": "market_value"},
    "ebitda_5yavgfq": {"dtype": "float", "role": "market_value"},
    "ebitda_5yavgltm": {"dtype": "float", "role": "market_value"},
    # Forward estimates
    "ebitda_est_avg_fy1e": {"dtype": "float", "role": "market_value"},
    "ebitda_est_avg_ntm": {"dtype": "float", "role": "market_value"},
    # ====================
    # PROFITABILITY - EBIT (market_value)
    # ====================
    "ebit_fy": {"dtype": "float", "role": "market_value"},
    "ebit_ltm": {"dtype": "float", "role": "market_value"},
    "ebit_fq": {"dtype": "float", "role": "market_value"},
    "ebit_1fy": {"dtype": "float", "role": "market_value"},
    "ebit_adj_ltm": {"dtype": "float", "role": "market_value"},
    "ebit_adj_fy": {"dtype": "float", "role": "market_value"},
    "ebit_adj_1fy": {"dtype": "float", "role": "market_value"},
    "ebit_est_med_fy1e": {"dtype": "float", "role": "market_value"},
    "ebit_est_med_ntm": {"dtype": "float", "role": "market_value"},
    "ebit_5yavgfq": {"dtype": "float", "role": "market_value"},
    "ebit_5yavgltm": {"dtype": "float", "role": "market_value"},
    # ====================
    # PROFITABILITY - Net Income (market_value)
    # ====================
    "net_income_is_fy": {"dtype": "float", "role": "market_value"},
    "net_income_is_ltm": {"dtype": "float", "role": "market_value"},
    "net_income_is_fq": {"dtype": "float", "role": "market_value"},
    "net_income_is_1fy": {"dtype": "float", "role": "market_value"},
    "net_income_is_5yavgfq": {"dtype": "float", "role": "market_value"},
    "net_income_is_5yavgltm": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_fy": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_ltm": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_fq": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_1fy": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_5yavgfq": {"dtype": "float", "role": "market_value"},
    "normalized_net_income_5yavgltm": {"dtype": "float", "role": "market_value"},
    "net_income_adj_fy": {"dtype": "float", "role": "market_value"},
    "net_income_adj_ltm": {"dtype": "float", "role": "market_value"},
    "net_income_adj_fq": {"dtype": "float", "role": "market_value"},
    "net_income_adj_1fy": {"dtype": "float", "role": "market_value"},
    "net_income_adj_5yavgfq": {"dtype": "float", "role": "market_value"},
    "operating_income_ltm": {"dtype": "float", "role": "market_value"},
    "operating_income_fy": {"dtype": "float", "role": "market_value"},
    "operating_income_fq": {"dtype": "float", "role": "market_value"},
    "operating_income_5yavgfq": {"dtype": "float", "role": "market_value"},
    # ====================
    # MARGINS - Percentage metrics
    # ====================
    "net_income_margin_pct_fy": {"dtype": "float", "role": "percentage"},
    "net_income_margin_pct_ltm": {"dtype": "float", "role": "percentage"},
    "gross_profit_margin_pct_fy": {"dtype": "float", "role": "percentage"},
    "gross_profit_margin_pct_ltm": {"dtype": "float", "role": "percentage"},
    "gross_profit_ltm": {"dtype": "float", "role": "market_value"},
    "gross_profit_fy": {"dtype": "float", "role": "market_value"},
    # ====================
    # RETURNS ON CAPITAL - Percentage metrics
    # ====================
    "return_on_equity_pct_ltm": {"dtype": "float", "role": "percentage"},
    "return_on_equity_pct_fy": {"dtype": "float", "role": "percentage"},
    "return_on_assets_roa_pct_ltm": {"dtype": "float", "role": "percentage"},
    "return_on_assets_roa_pct_fy": {"dtype": "float", "role": "percentage"},
    # ====================
    # CASH FLOW - Market value columns
    # ====================
    "cfo_ltm": {"dtype": "float", "role": "market_value"},
    "cfo_fy": {"dtype": "float", "role": "market_value"},
    "cfo_fq": {"dtype": "float", "role": "market_value"},
    "cfo_1fy": {"dtype": "float", "role": "market_value"},
    "fcf_ltm": {"dtype": "float", "role": "market_value"},
    "fcf_fy": {"dtype": "float", "role": "market_value"},
    "fcf_fq": {"dtype": "float", "role": "market_value"},
    "fcf_5yavgfq": {"dtype": "float", "role": "market_value"},
    "cfi_ltm": {"dtype": "float", "role": "market_value"},
    "cfi_fy": {"dtype": "float", "role": "market_value"},
    "cfi_fq": {"dtype": "float", "role": "market_value"},
    "cfi_1fy": {"dtype": "float", "role": "market_value"},
    "cff_ltm": {"dtype": "float", "role": "market_value"},
    "cff_fy": {"dtype": "float", "role": "market_value"},
    "cff_fq": {"dtype": "float", "role": "market_value"},
    "cff_1fy": {"dtype": "float", "role": "market_value"},
    # ====================
    # BALANCE SHEET - Market value columns
    # ====================
    "total_assets_ltm": {"dtype": "float", "role": "market_value"},
    "total_assets_fy": {"dtype": "float", "role": "market_value"},
    "total_equity_fy": {"dtype": "float", "role": "market_value"},
    "total_equity_ltm": {"dtype": "float", "role": "market_value"},
    "total_debt_fy": {"dtype": "float", "role": "market_value"},
    "total_debt_ltm": {"dtype": "float", "role": "market_value"},
    "total_current_assets_ltm": {"dtype": "float", "role": "market_value"},
    "total_current_liabilities_ltm": {"dtype": "float", "role": "market_value"},
    "current_ratio_fy": {"dtype": "float", "role": "ratio"},
    "current_ratio_ltm": {"dtype": "float", "role": "ratio"},
    "working_capital_ltm": {"dtype": "float", "role": "market_value"},
    "working_capital_fq": {"dtype": "float", "role": "market_value"},
    "working_capital_fy": {"dtype": "float", "role": "market_value"},
    "working_capital_5yavgfy": {"dtype": "float", "role": "market_value"},
    "tbv_fy": {"dtype": "float", "role": "market_value"},
    "tbv_ltm": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_ltm": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_fq": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_fy": {"dtype": "float", "role": "market_value"},
    "cash_and_equivalents_5yavgfq": {"dtype": "float", "role": "market_value"},
    "retained_earnings_ltm": {"dtype": "float", "role": "market_value"},
    "retained_earnings_fq": {"dtype": "float", "role": "market_value"},
    "retained_earnings_fy": {"dtype": "float", "role": "market_value"},
    "retained_earnings_5yavgfq": {"dtype": "float", "role": "market_value"},
    # Asset Details
    "inventory_ltm": {"dtype": "float", "role": "market_value"},
    "inventory_fq": {"dtype": "float", "role": "market_value"},
    "inventory_fy": {"dtype": "float", "role": "market_value"},
    "inventory_5yavgfq": {"dtype": "float", "role": "market_value"},
    "goodwill_fq": {"dtype": "float", "role": "market_value"},
    "goodwill_ltm": {"dtype": "float", "role": "market_value"},
    "goodwill_fy": {"dtype": "float", "role": "market_value"},
    "goodwill_1fy": {"dtype": "float", "role": "market_value"},
    "goodwill_5yavgfq": {"dtype": "float", "role": "market_value"},
    "intangible_assets": {
        "dtype": "float",
        "role": "feature",
    },  # Base column (no time suffix)
    "gross_intangible_assets_ltm": {"dtype": "float", "role": "market_value"},
    "gross_intangible_assets_fy": {"dtype": "float", "role": "market_value"},
    "gross_intangible_assets_5yavgfq": {"dtype": "float", "role": "market_value"},
    # Capex & Investments
    "capital_expenditure_ltm": {"dtype": "float", "role": "market_value"},
    "capital_expenditure_fy": {"dtype": "float", "role": "market_value"},
    "capital_expenditure_fq": {"dtype": "float", "role": "market_value"},
    "capital_expenditure_1fy": {"dtype": "float", "role": "market_value"},
    "capital_expenditure_5yavgfq": {"dtype": "float", "role": "market_value"},
    "asset_turnover_fy": {"dtype": "float", "role": "ratio"},
    "asset_turnover_ltm": {"dtype": "float", "role": "ratio"},
    "cash_acquisitions_ltm": {"dtype": "float", "role": "market_value"},
    "cash_acquisitions_fy": {"dtype": "float", "role": "market_value"},
    "cash_acquisitions_fq": {"dtype": "float", "role": "market_value"},
    "cash_acquisitions_1fy": {"dtype": "float", "role": "market_value"},
    "cash_acquisitions_5yavgfq": {"dtype": "float", "role": "market_value"},
    # Exceptional Items
    "impairment_of_goodwill_fq": {"dtype": "float", "role": "market_value"},
    "impairment_of_goodwill_ltm": {"dtype": "float", "role": "market_value"},
    "impairment_of_goodwill_1fy": {"dtype": "float", "role": "market_value"},
    "impairment_of_goodwill_fy": {"dtype": "float", "role": "market_value"},
    "impairment_of_goodwill_5yavgfq": {"dtype": "float", "role": "market_value"},
    "asset_writedown_ltm": {"dtype": "float", "role": "market_value"},
    "asset_writedown_fy": {"dtype": "float", "role": "market_value"},
    "asset_writedown_fq": {"dtype": "float", "role": "market_value"},
    "asset_writedown_1fy": {"dtype": "float", "role": "market_value"},
    "asset_writedown_5yavgfq": {"dtype": "float", "role": "market_value"},
    "restructuring_charges_ltm": {"dtype": "float", "role": "market_value"},
    "restructuring_charges_fq": {"dtype": "float", "role": "market_value"},
    "restructuring_charges_1fy": {"dtype": "float", "role": "market_value"},
    "restructuring_charges_fy": {"dtype": "float", "role": "market_value"},
    "restructuring_charges_5yavgfq": {"dtype": "float", "role": "market_value"},
    # Merger & Restructuring Charges - correct normalization with "and" (from CSV "Merger & Restructuring Charges")
    "merger_and_restructuring_charges_ltm": {"dtype": "float", "role": "market_value"},
    "merger_and_restructuring_charges_fq": {"dtype": "float", "role": "market_value"},
    "merger_and_restructuring_charges_fy": {"dtype": "float", "role": "market_value"},
    "merger_and_restructuring_charges_5yavgfq": {"dtype": "float", "role": "market_value"},
    # Legacy aliases (without "and") - kept for backward compatibility
    "merger_restructuring_charges_ltm": {"dtype": "float", "role": "auxiliary"},
    "merger_restructuring_charges_fq": {"dtype": "float", "role": "auxiliary"},
    "merger_restructuring_charges_fy": {"dtype": "float", "role": "auxiliary"},
    "merger_restructuring_charges_5yavgfq": {"dtype": "float", "role": "auxiliary"},
    "other_unusual_items_total_ltm": {"dtype": "float", "role": "market_value"},
    "gain_loss_on_sale_of_assets_ltm": {"dtype": "float", "role": "market_value"},
    # Operating Expenses
    "cost_of_revenues_ltm": {"dtype": "float", "role": "market_value"},
    # R&D Expenses - correct normalization (from CSV "R&D Expenses (LTM)" where & becomes "and")
    "randd_expenses_ltm": {"dtype": "float", "role": "market_value"},
    # R&D generic base column and legacy alias
    "r_d_expenses": {
        "dtype": "float",
        "role": "feature",
    },  # Generic base column (no time suffix)
    "r_d_expenses_ltm": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for randd_expenses_ltm
    "selling_general_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy - use sga_expenses_fq
    "selling_general_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy - use sga_expenses_fy
    "selling_general_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy - use sga_expenses_1fy
    "selling_general_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy
    # SG&A with "and" (correct normalization from "Selling General & Admin Expenses/Total")
    "selling_general_and_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "feature",
    },
    "selling_general_and_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "feature",
    },
    "selling_general_and_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "feature",
    },
    "selling_general_and_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "feature",
    },
    "accounts_receivable_total_fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy - use accounts_receivable_fy
    "accounts_receivable_total_1fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy - use accounts_receivable_1fy
    "accounts_receivable_total_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy
    "marketing_expenses": {
        "dtype": "float",
        "role": "feature",
    },  # Base column (no time suffix)
    "marketing_expenses_fq": {"dtype": "float", "role": "market_value"},
    "marketing_expenses_fy": {"dtype": "float", "role": "market_value"},
    "marketing_expenses_1fy": {"dtype": "float", "role": "market_value"},
    "marketing_expenses_5yavgltm": {"dtype": "float", "role": "market_value"},
    # Earnings Per Share
    "eps_adj_1fy": {"dtype": "float", "role": "ratio"},
    "eps_adj_fy": {"dtype": "float", "role": "ratio"},
    "eps_adj_ltm": {"dtype": "float", "role": "ratio"},
    # Basic EPS Historical (GAAP-based)
    "net_eps_basic_ltm": {"dtype": "float", "role": "ratio"},
    "net_eps_basic_fq": {"dtype": "float", "role": "ratio"},
    "net_eps_basic_fy": {"dtype": "float", "role": "ratio"},
    "net_eps_basic_1fqfq": {"dtype": "float", "role": "ratio"},  # 1 quarter ago
    "net_eps_basic_2fqfq": {"dtype": "float", "role": "ratio"},  # 2 quarters ago
    "net_eps_basic_3fqfq": {"dtype": "float", "role": "ratio"},  # 3 quarters ago
    "net_eps_basic_4fqfq": {"dtype": "float", "role": "ratio"},  # 4 quarters ago
    # Forward fiscal year EPS
    "net_eps_basic_1fy": {"dtype": "float", "role": "ratio"},  # 1 fiscal year
    "net_eps_basic_2fy": {"dtype": "float", "role": "ratio"},  # 2 fiscal years
    "net_eps_basic_3fy": {"dtype": "float", "role": "ratio"},  # 3 fiscal years
    "net_eps_basic_4fy": {"dtype": "float", "role": "ratio"},  # 4 fiscal years
    "net_eps_basic_5fy": {"dtype": "float", "role": "ratio"},  # 5 fiscal years
    # Forward EPS Estimates
    "eps_norm_est_avg_ntm": {"dtype": "float", "role": "ratio"},
    "eps_norm_est_avg_fy1e": {"dtype": "float", "role": "ratio"},
    "eps_norm_est_num_fy1e": {"dtype": "float", "role": "count"},  # Number of analysts
    # EPS Normalized Estimate Revisions (percentage changes over time periods)
    "eps_est_avg_rev_pct_fy1e_1w": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_1m": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_3m": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_6m": {"dtype": "float", "role": "percentage"},
    "eps_est_avg_rev_pct_fy1e_1y": {"dtype": "float", "role": "percentage"},
    # EPS GAAP Estimates & Revisions
    "eps_gaap_est_avg_fy1e": {"dtype": "float", "role": "ratio"},
    "eps_gaap_est_avg_ntm": {"dtype": "float", "role": "ratio"},
    "eps_gaap_est_avg_rev_pct_fy1e_1m": {"dtype": "float", "role": "percentage"},
    "eps_gaap_est_avg_rev_pct_fy1e_3m": {"dtype": "float", "role": "percentage"},
    "eps_gaap_est_avg_rev_pct_fy1e_6m": {"dtype": "float", "role": "percentage"},
    "eps_gaap_est_avg_rev_pct_fy1e_1y": {"dtype": "float", "role": "percentage"},
    "eps_previous_year": {
        "dtype": "float",
        "role": "feature",
    },  # Base column for YoY calculations
    # Dividends
    "dividend_per_share_ltm": {"dtype": "float", "role": "price"},
    "div_yield_ind": {"dtype": "float", "role": "percentage"},
    "div_yield_ltm": {"dtype": "float", "role": "percentage"},
    "div_yield_ttm": {"dtype": "float", "role": "percentage"},
    "div_yield_ntm": {"dtype": "float", "role": "percentage"},
    "div_yield_1fyind": {"dtype": "float", "role": "percentage"},
    "div_yield_2fyind": {"dtype": "float", "role": "percentage"},
    "div_yield_3fyind": {"dtype": "float", "role": "percentage"},
    "div_yield_4fyind": {"dtype": "float", "role": "percentage"},
    "div_yield_5fyind": {"dtype": "float", "role": "percentage"},
    "div_yield_5yavgltm": {"dtype": "float", "role": "percentage"},
    "common_dividends_paid_ltm": {"dtype": "float", "role": "market_value"},
    "common_dividends_paid_fy": {"dtype": "float", "role": "market_value"},
    "dividend_record_frequency": {"dtype": "string", "role": "categorical"},
    "dividend_record_currency": {"dtype": "string", "role": "categorical"},
    "dividend_record_amount": {"dtype": "float", "role": "price"},
    "dividend_streak": {"dtype": "float", "role": "count"},
    "days_to_dividend": {"dtype": "float", "role": "feature"},  # Days until next ex-date
    "buyback_yield_ltm": {"dtype": "float", "role": "percentage"},
    # Interest & Financing
    "interest_expense_total_ltm": {"dtype": "float", "role": "market_value"},
    "interest_income_on_investments_ltm": {"dtype": "float", "role": "market_value"},
    # Employees
    "employees": {
        "dtype": "float",
        "role": "count",
    },  # Base column (current employee count) - float for NULL handling
    "avg_employees_ltm": {"dtype": "float", "role": "count"},
    "avg_employees_fy": {"dtype": "float", "role": "count"},
    "avg_employees_5yavgfy": {"dtype": "float", "role": "count"},
    "total_employees_fy": {"dtype": "float", "role": "count"},
    "total_employees_fq": {"dtype": "float", "role": "count"},
    "full_time_employees_fq": {
        "dtype": "float",
        "role": "count",
    },  # Full time employees (Fiscal Quarter) - float for NULL handling
    "full_time_employees_fy": {
        "dtype": "float",
        "role": "count",
    },  # Full time employees (Fiscal Year) - float for NULL handling
    "full_time_employees_1fy": {
        "dtype": "float",
        "role": "count",
    },  # Full time employees (Previous FY) - float for NULL handling
    "full_time_employees_2fy": {
        "dtype": "float",
        "role": "count",
    },  # Full time employees (2 Years Ago) - float for NULL handling
    "full_time_employees_3fy": {
        "dtype": "float",
        "role": "count",
    },  # Full time employees (3 Years Ago) - float for NULL handling
    # Country-specific
    # ==================================================================================
    # NORMALIZATION VARIANTS & SIMPLIFIED ALIASES
    # Added to resolve unknown column warnings from dtype_diagnostics.json
    # These columns exist in the data pipeline but use different naming conventions
    # ==================================================================================
    # Analyst Ratings (normalized names without "num_" prefix) - Legacy aliases
    # Alias for price_target_num
    "strong_sell_ratings": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for num_strong_sell_ratings
    "strong_buys_ratings": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for num_strong_buys_ratings
    "hold_ratings": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for num_hold_ratings
    "buys_ratings": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for num_buys_ratings
    "sell_ratings": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for num_sell_ratings
    # Simplified Base Columns (without time suffixes - used as generic references)
    "p_e": {"dtype": "float", "role": "ratio"},  # Generic P/E ratio
    "p_b": {"dtype": "float", "role": "ratio"},  # Generic P/B ratio
    "revenue": {"dtype": "float", "role": "market_value"},  # Generic revenue
    "ebitda": {"dtype": "float", "role": "market_value"},  # Generic EBITDA
    "ebit": {"dtype": "float", "role": "market_value"},  # Generic EBIT
    "net_income": {"dtype": "float", "role": "market_value"},  # Generic net income
    "net_income_ltm": {
        "dtype": "float",
        "role": "market_value",
    },  # Duplicate of net_income_is_ltm
    "gross_margin": {"dtype": "float", "role": "percentage"},  # Generic gross margin
    "eps": {"dtype": "float", "role": "ratio"},  # Generic EPS
    "total_equity": {"dtype": "float", "role": "market_value"},  # Generic total equity
    "total_assets": {"dtype": "float", "role": "market_value"},  # Generic total assets
    "total_debt": {"dtype": "float", "role": "market_value"},  # Generic total debt
    "inventory": {"dtype": "float", "role": "market_value"},  # Generic inventory
    "capex": {"dtype": "float", "role": "market_value"},  # Generic capital expenditure
    "cash_and_equivalents": {"dtype": "float", "role": "market_value"},  # Generic cash
    "current_assets": {"dtype": "float", "role": "market_value"},  # Generic current assets
    "accounts_receivable": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias - use accounts_receivable_fy
    "current_liabilities": {
        "dtype": "float",
        "role": "market_value",
    },  # Generic current liabilities
    "working_capital": {"dtype": "float", "role": "market_value"},  # Generic working capital
    "retained_earnings": {
        "dtype": "float",
        "role": "market_value",
    },  # Generic retained earnings
    "cfo": {"dtype": "float", "role": "market_value"},  # Generic cash flow from operations
    "cfi": {"dtype": "float", "role": "market_value"},  # Generic cash flow from investing
    "cff": {"dtype": "float", "role": "market_value"},  # Generic cash flow from financing
    "fcf": {"dtype": "float", "role": "market_value"},  # Generic free cash flow
    "gross_profit": {"dtype": "float", "role": "market_value"},  # Generic gross profit
    "operating_income": {
        "dtype": "float",
        "role": "market_value",
    },  # Generic operating income
    "interest_expense": {
        "dtype": "float",
        "role": "market_value",
    },  # Generic interest expense
    "goodwill": {"dtype": "float", "role": "market_value"},  # Generic goodwill
    "dividend_per_share": {
        "dtype": "float",
        "role": "price",
    },  # Generic dividend per share
    "operating_expenses": {
        "dtype": "float",
        "role": "market_value",
    },  # Generic operating expenses
    "operating_cash_flow": {"dtype": "float", "role": "market_value"},  # Alias for cfo
    "dividends_paid": {"dtype": "float", "role": "market_value"},  # Generic dividends paid
    "dividends_paid_ltm": {"dtype": "float", "role": "market_value"},  # Dividends paid LTM
    # Additional normalized names
    "price_target_number": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Alias for price_target_num
    # SG&A Expenses (normalized naming) - Legacy aliases
    "sga_expenses": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias - use selling_general_and_admin_expenses_*
    "sga_expenses_fq": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for selling_general_and_admin_expenses_total_fq
    "sga_expenses_fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for selling_general_and_admin_expenses_total_fy
    "sga_expenses_1fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for selling_general_and_admin_expenses_total_1fy
    "sga_expenses_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for selling_general_and_admin_expenses_total_5yavgfq
    # Accounts Receivable (normalized naming) - Legacy aliases
    "accounts_receivable_fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for accounts_receivable_total_fy
    "accounts_receivable_1fy": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for accounts_receivable_total_1fy
    "accounts_receivable_5yavgfq": {
        "dtype": "float",
        "role": "auxiliary",
    },  # Legacy alias for accounts_receivable_total_5yavgfq
    # ==================================================================================
    # DERIVED & COMPUTED COLUMNS (Created during preprocessing/feature engineering)
    # ==================================================================================
    # Volatility percentage variants
    "volatility_1y_pct": {
        "dtype": "float",
        "role": "percentage",
    },  # 1-year volatility as percentage
    # --------------------------------------------------------------------------
    # Sector-specific and advanced engineered features (from financial_metrics_etl.py and advanced.py)
    # --------------------------------------------------------------------------
    "tangible_book_value": {"dtype": "float", "role": "market_value"},  # Total equity - intangibles
    "marketing_efficiency": {"dtype": "float", "role": "ratio"},  # Revenue / SG&A
    "r_d_intensity": {"dtype": "float", "role": "percentage"},  # R&D / Revenue ratio
    "rule_of_40": {"dtype": "float", "role": "percentage"},  # Revenue growth + margin (SaaS)
    "operating_leverage": {"dtype": "float", "role": "ratio"},  # Operating leverage ratio
    "one_day_chg": {"dtype": "float", "role": "percentage"},  # One-day price change
    # --------------------------------------------------------------------------
    # Interaction features (multiplicative combinations from create_feature_interactions)
    # --------------------------------------------------------------------------
    "market_cap_x_debt_to_equity": {"dtype": "float", "role": "feature"},
    "market_cap_x_roe": {"dtype": "float", "role": "feature"},
    "p_e_ratio_x_debt_to_equity": {"dtype": "float", "role": "feature"},
    "p_e_ratio_x_roe": {"dtype": "float", "role": "feature"},
    "roe_x_debt_to_equity": {"dtype": "float", "role": "feature"},
    # --------------------------------------------------------------------------
    # Log-transformed semantic metrics (created by ETL semantic transforms)
    # These are log1p transforms of market values for better ML distribution
    # --------------------------------------------------------------------------
    "log_operating_income": {"dtype": "float", "role": "market_value"},
    "log_ebitda": {"dtype": "float", "role": "market_value"},
    "log_net_income": {"dtype": "float", "role": "market_value"},
    "log_capex": {"dtype": "float", "role": "market_value"},
    "log_operating_cash_flow": {"dtype": "float", "role": "market_value"},
    "log_total_equity": {"dtype": "float", "role": "market_value"},
    "log_market_cap": {"dtype": "float", "role": "market_value"},
    "log_total_assets": {"dtype": "float", "role": "market_value"},
    "log_gross_profit": {"dtype": "float", "role": "market_value"},
    "log_cash_and_equivalents": {"dtype": "float", "role": "market_value"},
    "log_total_debt": {"dtype": "float", "role": "market_value"},
    "log_revenue": {"dtype": "float", "role": "market_value"},
    "log_enterprise_value": {"dtype": "float", "role": "market_value"},
    # Additional log-transformed columns (time-series variants)
    "log_gross_profit_previous_year": {"dtype": "float", "role": "market_value"},
    "log_operating_income_fq": {"dtype": "float", "role": "market_value"},
    "log_ebitda_ltm": {"dtype": "float", "role": "market_value"},
    "log_total_revenues_5yavgfq": {"dtype": "float", "role": "market_value"},
    "log_cash_acquisitions_fq": {"dtype": "float", "role": "market_value"},
    "log_total_revenues_5yavgltm": {"dtype": "float", "role": "market_value"},
    "log_ebitda_fy": {"dtype": "float", "role": "market_value"},
    "log_total_assets_ltm": {"dtype": "float", "role": "market_value"},
    "log_ebitda_previous_year": {"dtype": "float", "role": "market_value"},
    "log_operating_income_fy": {"dtype": "float", "role": "market_value"},
    "log_cash_acquisitions_ltm": {"dtype": "float", "role": "market_value"},
    "log_revenues_est_avg_ntm": {"dtype": "float", "role": "market_value"},
    "log_total_revenues_fy": {"dtype": "float", "role": "market_value"},
    "log_net_income_is_1fy": {"dtype": "float", "role": "market_value"},
    "log_fcf_fq": {"dtype": "float", "role": "market_value"},
    "log_total_equity_ltm": {"dtype": "float", "role": "market_value"},
    "log_total_revenues_ltm": {"dtype": "float", "role": "market_value"},
    "log_net_income_adj_1fy": {"dtype": "float", "role": "market_value"},
    "log_total_equity_fy": {"dtype": "float", "role": "market_value"},
    "log_total_debt_fy": {"dtype": "float", "role": "market_value"},
    "log_revenue_previous_year": {"dtype": "float", "role": "market_value"},
    "log_revenue_fy": {"dtype": "float", "role": "market_value"},
    "log_cash_acquisitions_5yavgfq": {"dtype": "float", "role": "market_value"},
    "log_net_income_is_5yavgltm": {"dtype": "float", "role": "market_value"},
    "log_cash_acquisitions_fy": {"dtype": "float", "role": "market_value"},
    "log_total_assets_fy": {"dtype": "float", "role": "market_value"},
    "log_net_income_adj_fy": {"dtype": "float", "role": "market_value"},
    "log_ebitda_5yavgltm": {"dtype": "float", "role": "market_value"},
    "log_revenues_est_avg_fy1e": {"dtype": "float", "role": "market_value"},
    "log_ebitda_fq": {"dtype": "float", "role": "market_value"},
    "log_ebitda_1fy": {"dtype": "float", "role": "market_value"},
    "log_revenues_est_med_ntm": {"dtype": "float", "role": "market_value"},
    "log_cash_and_equivalents_fy": {"dtype": "float", "role": "market_value"},
    "log_net_income_is_5yavgfq": {"dtype": "float", "role": "market_value"},
    "log_cash_and_equivalents_5yavgfq": {"dtype": "float", "role": "market_value"},
    "log_fcf_ltm": {"dtype": "float", "role": "market_value"},
    "log_total_debt_ltm": {"dtype": "float", "role": "market_value"},
    "log_fcf": {"dtype": "float", "role": "market_value"},
    "log_gross_profit_fy": {"dtype": "float", "role": "market_value"},
    "log_market_cap_country_r": {"dtype": "float", "role": "market_value"},
    "log_cash_and_equivalents_ltm": {"dtype": "float", "role": "market_value"},
    "log_fcf_5yavgfq": {"dtype": "float", "role": "market_value"},
    "log_ebitda_5yavgfq": {"dtype": "float", "role": "market_value"},
    "log_fcf_fy": {"dtype": "float", "role": "market_value"},
    "log_revenues_est_med_fy1e": {"dtype": "float", "role": "market_value"},
    "log_total_assets_previous_year": {"dtype": "float", "role": "market_value"},
    "log_operating_income_ltm": {"dtype": "float", "role": "market_value"},
    "log_net_income_is_fq": {"dtype": "float", "role": "market_value"},
    "log_ebitda_adj_ltm": {"dtype": "float", "role": "market_value"},
    "log_gross_profit_ltm": {"dtype": "float", "role": "market_value"},
    # --------------------------------------------------------------------------
    # Valuation / profitability / leverage ratios (Phase 9.3 semantic metrics)
    # Created during ETL semantic transformation stage
    # --------------------------------------------------------------------------
    "p_e_ratio": {"dtype": "float", "role": "ratio"},  # Price-to-Earnings ratio
    "p_s_ratio": {"dtype": "float", "role": "ratio"},  # Price-to-Sales ratio
    "ev_ebitda_ratio": {"dtype": "float", "role": "ratio"},  # EV/EBITDA ratio
    "ev_sales_ratio": {"dtype": "float", "role": "ratio"},  # EV/Sales ratio
    "gross_margin_pct": {
        "dtype": "float",
        "role": "percentage",
    },  # Gross margin percentage
    "operating_margin_pct": {
        "dtype": "float",
        "role": "percentage",
    },  # Operating margin percentage
    "net_margin_pct": {"dtype": "float", "role": "percentage"},  # Net margin percentage
    "roe": {"dtype": "float", "role": "ratio"},  # Return on Equity
    "roa": {"dtype": "float", "role": "ratio"},  # Return on Assets
    "revenue_growth": {"dtype": "float", "role": "percentage"},  # Revenue growth rate
    "ebitda_growth": {"dtype": "float", "role": "percentage"},  # EBITDA growth rate
    "earnings_growth": {"dtype": "float", "role": "percentage"},  # Earnings growth rate
    "debt_to_equity": {"dtype": "float", "role": "ratio"},  # Debt-to-Equity ratio
    "debt_to_assets": {"dtype": "float", "role": "ratio"},  # Debt-to-Assets ratio
    "target_vs_price": {
        "dtype": "float",
        "role": "ratio",
    },  # Price target vs last price ratio
    "target_vs_price_median": {
        "dtype": "float",
        "role": "ratio",
    },  # Median price target vs last price
    "peg_ratio": {
        "dtype": "float",
        "role": "ratio",
    },  # Price/Earnings-to-Growth ratio
    "dividend_yield": {
        "dtype": "float",
        "role": "percentage",
    },  # Dividend yield percentage
    "roic": {"dtype": "float", "role": "ratio"},  # Return on Invested Capital
    # --------------------------------------------------------------------------
    # Year-over-Year (YoY) comparison columns (_previous_year suffix)
    "revenue_previous_year": {
        "dtype": "float",
        "role": "market_value",
    },  # Revenue from previous year
    "ebitda_previous_year": {
        "dtype": "float",
        "role": "market_value",
    },  # EBITDA from previous year
    "total_equity_previous_year": {
        "dtype": "float",
        "role": "market_value",
    },  # Total equity from previous year
    "total_assets_previous_year": {
        "dtype": "float",
        "role": "market_value",
    },  # Total assets from previous year
    "gross_profit_previous_year": {
        "dtype": "float",
        "role": "market_value",
    },  # Gross profit from previous year
    "accounts_receivable_previous_year": {
        "dtype": "float",
        "role": "market_value",
    },  # AR from previous year
    "roa_previous_year": {
        "dtype": "float",
        "role": "ratio",
    },  # ROA from previous year
    "current_ratio_previous_year": {
        "dtype": "float",
        "role": "ratio",
    },  # Current ratio from previous year
    "shares_outstanding_previous_year": {
        "dtype": "float",
        "role": "count",
    },  # Shares outstanding from previous year
    "gross_margin_pct_previous_year": {
        "dtype": "float",
        "role": "percentage",
    },  # Gross margin % from previous year
    "asset_turnover_previous_year": {
        "dtype": "float",
        "role": "ratio",
    },  # Asset turnover from previous year
    # Fiscal year variants (alternative naming)
    "revenue_fy": {"dtype": "float", "role": "market_value"},  # Alias for total_revenues_fy
    "working_capital_1fy": {
        "dtype": "float",
        "role": "market_value",
    },  # Working capital 1 fiscal year
    # ==================================================================================
    # CONDITIONAL METRICS (Only valid for specific business conditions)
    # ==================================================================================
    # Cash burn rate - only valid for companies with negative CFO (burning cash)
    "cash_burn_rate": {"dtype": "float", "role": "feature"},
    "cash_burn_rate_applicable": {
        "dtype": "bool",
        "role": "auxiliary",
    },  # True if company has negative CFO
    # Employee productivity metrics - only valid when employee data is available
    "revenue_per_employee": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_ltm": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_ltm_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_fy": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_fy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_trend": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_trend_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_vs_5y_pct": {"dtype": "float", "role": "feature"},
    "revenue_per_employee_vs_5y_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "assets_per_employee": {"dtype": "float", "role": "feature"},
    "assets_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "ebitda_per_employee": {"dtype": "float", "role": "feature"},
    "ebitda_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "operating_income_per_employee": {"dtype": "float", "role": "feature"},
    "operating_income_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "profit_per_employee": {"dtype": "float", "role": "feature"},
    "profit_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy": {"dtype": "float", "role": "feature"},
    "employee_growth_yoy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy_pct": {"dtype": "float", "role": "feature"},
    "employee_growth_yoy_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_qoq": {"dtype": "float", "role": "feature"},
    "employee_growth_qoq_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_cagr_5y": {"dtype": "float", "role": "feature"},
    "employee_growth_cagr_5y_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_acceleration": {"dtype": "float", "role": "feature"},
    "employee_growth_acceleration_applicable": {"dtype": "bool", "role": "auxiliary"},
    "workforce_volatility": {"dtype": "float", "role": "feature"},
    "workforce_volatility_applicable": {"dtype": "bool", "role": "auxiliary"},
    "hiring_intensity_score": {"dtype": "float", "role": "feature"},
    "hiring_intensity_score_applicable": {"dtype": "bool", "role": "auxiliary"},
    # ==================================================================================
    # PHASE 9.3 COMPOSITE QUALITY SCORES (Advanced feature engineering)
    # ==================================================================================
    "altman_z_score": {
        "dtype": "float",
        "role": "feature",
    },  # Composite bankruptcy risk score
    "beneish_m_score": {
        "dtype": "float",
        "role": "feature",
    },  # Earnings manipulation detection score
    "composite_quality_score": {
        "dtype": "float",
        "role": "feature",
    },  # Multi-factor quality composite
    "momentum_score": {
        "dtype": "float",
        "role": "feature",
    },  # Technical momentum composite
    # ==================================================================================
    # ESTIMATED VS. ACTUAL ANALYTICS (Phase 9.3 - Earnings Analytics Enhancement)
    # ==================================================================================
    "eps_surprise_pct": {"dtype": "float", "role": "percentage"},
    "eps_surprise_magnitude": {"dtype": "category", "role": "categorical"},
    "revenue_surprise_pct": {"dtype": "float", "role": "percentage"},
    "revenue_beat_indicator": {"dtype": "bool", "role": "feature"},
    "ebitda_surprise_pct": {"dtype": "float", "role": "percentage"},
    "earnings_beat_indicator": {"dtype": "bool", "role": "feature"},
    "surprise_momentum_score": {"dtype": "float", "role": "feature"},
    "positive_revision_momentum": {"dtype": "bool", "role": "feature"},
    "consensus_uncertainty_score": {"dtype": "float", "role": "feature"},
    "estimate_revision_acceleration": {"dtype": "float", "role": "percentage"},
    "accelerating_upgrades_flag": {"dtype": "bool", "role": "feature"},
    # ==================================================================================
    # GAAP VS. ADJUSTED ANALYTICS (Phase 9.3 - Earnings Quality)
    # ==================================================================================
    "eps_adjustment_spread_ltm": {"dtype": "float", "role": "feature"},
    "eps_adjustment_ratio_ltm": {"dtype": "float", "role": "ratio"},
    "eps_adjustment_pct_ltm": {"dtype": "float", "role": "percentage"},
    "eps_quality_flag_ltm": {"dtype": "bool", "role": "feature"},
    "eps_adjustment_spread_fy": {"dtype": "float", "role": "feature"},
    "eps_adjustment_ratio_fy": {"dtype": "float", "role": "ratio"},
    "eps_adjustment_pct_fy": {"dtype": "float", "role": "percentage"},
    "net_income_adjustment_spread_ltm": {"dtype": "float", "role": "market_value"},
    "net_income_adjustment_ratio_ltm": {"dtype": "float", "role": "ratio"},
    "net_income_adjustment_pct_ltm": {"dtype": "float", "role": "percentage"},
    "net_income_adjustment_spread_fy": {"dtype": "float", "role": "market_value"},
    "net_income_adjustment_ratio_fy": {"dtype": "float", "role": "ratio"},
    "ebitda_adjustment_spread_ltm": {"dtype": "float", "role": "market_value"},
    "ebitda_adjustment_pct_ltm": {"dtype": "float", "role": "percentage"},
    "ebitda_adjustment_spread_fy": {"dtype": "float", "role": "market_value"},
    "ebit_adjustment_spread_ltm": {"dtype": "float", "role": "market_value"},
    "ebit_adjustment_pct_ltm": {"dtype": "float", "role": "percentage"},
    "ebit_adjustment_spread_fy": {"dtype": "float", "role": "market_value"},
    "adjustment_consistency_score": {"dtype": "float", "role": "feature"},
    "earnings_quality_warning_flag": {"dtype": "bool", "role": "feature"},
    "earnings_quality_score": {"dtype": "float", "role": "feature"},
    "exceptional_items_impact_ratio": {"dtype": "float", "role": "ratio"},
}


# Phase 9.3 Feature Input Categorization
# Maps feature engineering buckets to required input columns
PHASE93_FEATURE_CATEGORIES: Dict[str, List[str]] = {
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
    "Valuation Ratios": [
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
        "p_b_ratio",  # Added: from engineer_valuation_ratios
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
        "book_value_per_share",  # Added: from engineer_valuation_ratios
    ],
    "Profitability": [
        "earnings_quality_score",
        "ebit_adjustment_ratio_ltm",
        "ebit_adjustment_ratio_fy",
        "ebitda_adjustment_ratio_fy",
        "ebitda_adjustment_ratio_ltm",
        "ebitda_margin_trend",
        "gross_margin_pct",
        "gross_margin_trend",
        "net_margin_pct",
        "operating_margin_pct",
        "roa",
        "roe",
        "roic",
        "operating_leverage",  # Added: from engineer_margin_trends
    ],
    "Quality & Risk": [
        "accounting_quality_score",
        "altman_z_trend",
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
    "Cash Flow": [
        "cfo_growth_yoy",
        "cfo_to_net_income",
        "fcf_margin",
        "fcf_stability",
        "fcf_to_net_income",
    ],
    "Capital Allocation": [
        "acquisition_intensity",
        "capex_growth_rate",
        "capex_intensity",
        "capex_volatility",
        "currency_risk_flag",
        "days_since_ex_date",
        "div_yield_ltm",
        "dividend_aristocrat_flag",
        "dividend_consistency_score",
        "dividend_frequency_encoded",
        "dividend_growth_trend",
        "dividend_payout_ratio",
        "dividend_safety_score",
        "dividend_streak_years",
        "dividend_yield_vs_sector",
        "fcf_dividend_coverage",
        "income_stock_flag",
        "reinvestment_rate",
        "total_shareholder_return_yield",
        "working_capital_efficiency",
        "working_capital_trend",
        "payout_ratio",  # Added: from engineer_capital_allocation_features
        "capex_to_depreciation",  # Added: from engineer_sector_specific_features (Industrials)
    ],
    "Analyst Sentiment": [
        "analyst_bearish_pct",
        "analyst_bullish_pct",
        "analyst_conviction",
        "analyst_coverage_quality",
        "consensus_strength",
        "price_target_range",
        "price_target_revision",
        "price_target_spread_pct",
        "target_price_upside_pct",
        "upside_potential",
    ],
    "Market Sentiment": [
        "beta_stability",
        "momentum_20d",
        "price_range_pct",
        "systematic_risk_trend",
        "one_day_chg",  # Added: from engineer_market_sentiment_features
    ],
    "Leverage & Liquidity": [
        "cash_ratio",
        "current_ratio",
        "debt_to_assets",
        "debt_to_equity",
        "equity_ratio",
        "interest_coverage",
        "quick_ratio",
        "working_capital_to_sales",
        "net_debt_to_ebitda",  # Added: from engineer_leverage_ratios
    ],
    "Temporal Patterns": [
        "days_to_earnings",
        "earnings_report_recency",
        "ebitda_5yavgfq",
        "ebitda_fq",
        "fiscal_quarter",
        "fq_vs_5yavg_ebitda",
        "income_statement_report_date",
        "last_updated",
        "ltm_vs_5yavg_revenue",
        "month",
        "next_earnings",
        "quarterly_volatility_score",
        "reporting_lag",
        "total_revenues_ltm",
        "year",
        "_reference_date",  # Added: from engineer_temporal_features
        "days_to_dividend",  # Added: from engineer_temporal_features
    ],
    "Composite Scores": [
        "altman_z_score",
        "beneish_m_score",
        "composite_quality_score",
        "momentum_score",
        "piotroski_f_score",
    ],
    "Growth Metrics": [
        "earnings_growth",
        "ebitda_growth",
        "ebitda_growth_yoy",
        "eps_growth_yoy",
        "revenue_growth",
        "revenue_growth_yoy",
        "debt_growth_rate",  # Added: from engineer_balance_sheet_trends
        "equity_growth_rate",  # Added: from engineer_balance_sheet_trends
        "asset_growth_rate",  # Added: from engineer_balance_sheet_trends
    ],
    "Efficiency Ratios": [
        "asset_turnover",
        "inventory_turnover",
        "receivables_turnover",
        "revenue_per_employee",
    ],
    "Employee Productivity": [
        "assets_per_employee",
        "ebitda_per_employee",
        "employee_base_scale_flag",
        "employee_growth_acceleration",
        "employee_growth_cagr_5y",
        "employee_growth_yoy",
        "employee_growth_yoy_pct",
        "hiring_intensity_score",
        "operating_income_per_employee",
        "profit_per_employee",
        "revenue_per_employee_fy",
        "revenue_per_employee_1fy",
        "revenue_per_employee_trend",
        "revenue_per_employee_vs_5y_pct",
        "workforce_volatility",
        # Added: Full Time Employees (FTE) growth features
        "fte_growth_1y_pct",
        "fte_growth_2y_pct",
        "fte_growth_3y_pct",
        "fte_cagr_3y_pct",
        "fte_volatility",
        "fte_quarterly_momentum",
    ],
    "Balance Sheet Dynamics": [
        "asset_growth_rate",
        "balance_sheet_expansion",
        "current_ratio_trend",
        "debt_growth_rate",
        "earnings_retention_rate",
        "equity_growth_rate",
        "retained_earnings_growth",
        "working_capital_ratio",
        "book_value_per_share",
    ],
    "Revenue Forecasting": [
        "avg_vs_median_bias",
        "estimate_confidence_flag",
        "growth_surprise_potential",
        "revenue_consensus_uncertainty_score",
        "revenue_estimate_spread_fy1e",
        "revenue_estimate_spread_ntm",
        "revenue_growth_acceleration",
        "revenue_growth_implied_fy1e",
        "revenue_growth_implied_ntm",
    ],
    "Earnings Quality": [
        # From engineer_estimated_vs_actual_analytics() - 11 features
        "eps_surprise_pct",
        "earnings_beat_indicator",
        "eps_surprise_magnitude",
        "revenue_surprise_pct",
        "revenue_beat_indicator",
        "ebitda_surprise_pct",
        "surprise_momentum_score",
        "positive_revision_momentum",
        "consensus_uncertainty_score",
        "estimate_revision_acceleration",
        "accelerating_upgrades_flag",
        # From engineer_gaap_vs_adjusted_analytics() - 22 features
        "eps_adjustment_spread_ltm",
        "eps_adjustment_ratio_ltm",
        "eps_adjustment_pct_ltm",
        "eps_quality_flag_ltm",
        "eps_adjustment_spread_fy",
        "eps_adjustment_ratio_fy",
        "eps_adjustment_pct_fy",
        "net_income_adjustment_spread_ltm",
        "net_income_adjustment_ratio_ltm",
        "net_income_adjustment_pct_ltm",
        "net_income_adjustment_spread_fy",
        "net_income_adjustment_ratio_fy",
        "ebitda_adjustment_spread_ltm",
        "ebitda_adjustment_pct_ltm",
        "ebitda_adjustment_spread_fy",
        "ebit_adjustment_spread_ltm",
        "ebit_adjustment_pct_ltm",
        "ebit_adjustment_spread_fy",
        "adjustment_consistency_score",
        "earnings_quality_warning_flag",
        "earnings_quality_score",
        "exceptional_items_impact_ratio",
    ],
    # NEW CATEGORIES (Phase 9.3 Schema 1.3 enhancements)
    "Technical Analysis": [
        "ema_crossover_20_50",
        "ema_crossover_50_250",
        "price_vs_ema_20d",
        "price_vs_ema_250d",
        "ema_slope_20d",
        "ema_trend_consistency",
        "pct_off_52w_high",
        "pct_above_52w_low",
        "52w_range_position",
        "near_52w_high_flag",
        "near_52w_low_flag",
        "volume_momentum_score",
        "breakout_signal",
        "rsi_14d",  # Added: from engineer_momentum_features
        "rsi_30d",  # Added: from engineer_momentum_features
    ],
    "Valuation Timeseries": [
        "ev_sales_trend_1y",
        "ev_sales_trend_3y",
        "ev_ebitda_momentum",
        "p_e_momentum_yoy",
        "p_e_momentum_qoq",
        "ev_sales_vs_3y_avg",
        "ev_ebitda_vs_3y_avg",
        "p_e_vs_3y_avg",
        "valuation_extreme_flag",
        "ev_sales_forward_discount",
        "ev_ebitda_forward_discount",
        "p_e_forward_discount",
        "growth_implied_by_valuation",
        "ev_sales_quarterly_volatility",
        "valuation_stability_score",
        "valuation_trend_consistency",
    ],
    "Dividend Reliability": [
        "dividend_streak_years",
        "dividend_consistency_score",
        "income_stock_flag",
        "dividend_payout_ratio",
        "fcf_dividend_coverage",
        "dividend_safety_score",
        "dividend_growth_trend",
        "dividend_yield_vs_sector",
        "dividend_aristocrat_flag",
        "days_since_ex_date",
        "dividend_frequency_encoded",
        "currency_risk_flag",
    ],
    "Employment Dynamics": [
        "employee_growth_yoy",
        "employee_growth_cagr_5y",
        "employee_growth_acceleration",
        "revenue_per_employee_fy",
        "revenue_per_employee_1fy",
        "revenue_per_employee_trend",
        "profit_per_employee",
        "employee_base_scale_flag",
        "workforce_volatility",
        "hiring_intensity_score",
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

    Special handling:
    - "R&D" -> "randd" (not "r_and_d")
    - "Merger & Restructuring" -> "merger_and_restructuring"
    - "Selling General & Admin" -> "selling_general_and_admin"

    Args:
        column: Original column name (e.g., "Last Price" or "P/E (LTM)")

    Returns:
        Normalized column name (e.g., "last_price" or "p_e_ltm")
    """
    # Special case: R&D should become randd (not r_and_d)
    # Must be done before general & -> and replacement
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
    # Remove consecutive underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    # Remove leading/trailing underscores
    return normalized.strip("_")


def list_etl_generated_column_patterns() -> List[str]:
    """List regex patterns for columns legitimately generated during ETL.

    These patterns are used by schema alignment validation and diagnostics to
    distinguish between truly unknown columns (data quality / upstream drift)
    and expected outputs created by the ETL / feature engineering workflow.

    Returns:
        List of regex pattern strings.

    Example:
        >>> import re
        >>> patterns = list_etl_generated_column_patterns()
        >>> any(re.match(p, "log_market_cap") for p in patterns)
        True
    """

    # NOTE: Keep these patterns conservative. Where possible, the ETL validator
    # also checks that the underlying base column exists in `COLUMN_SCHEMA`.
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


def list_required_schema_columns_for_etl(
    include_extended_financials: bool = False,
) -> List[str]:
    """
    List the canonical set of *required* raw columns for the unified ETL pipeline.

    This function centralizes the minimum schema that must be present in the
    source extract (CSV / DB) for Phase 9.1–9.3 ETL to run correctly and for
    downstream quality checks to be meaningful.

    It is intended for use in:
        - validate_schema(..., required_columns=list_required_schema_columns_for_etl())
        - interpreting dtype diagnostics "missing_expected_columns"
        - notebook/script post-ETL assertions

    The list focuses on:
        - Identifiers and group keys (ticker, sector, region, country, trading_country, isin)
        - Core business-critical price/target columns
        - Core market value columns needed for valuation metrics

    Args:
        include_extended_financials:
            When True, include a small set of additional raw financial
            columns that are highly recommended for modeling but not strictly
            mandatory for a minimal ETL run (e.g. EBITDA LTM, total_assets_ltm).

    Returns:
        Ordered list of normalized column names that must be present in
        the *raw* dataset before ETL.

    Notes:
        - All returned names are guaranteed to exist in COLUMN_SCHEMA; an
          AssertionError is raised during development if the schema drifts.
        - Derived / engineered columns (e.g. log_* metrics, ratios constructed
          in ETL, Phase 9.3 features) are *not* included here, as they are
          created by the pipeline, not required from the source.

    Example:
        >>> required = list_required_schema_columns_for_etl()
        >>> assert "ticker" in required
        >>> assert "last_price" in required
        >>> # Use with validate_schema
        >>> from finance_ml.ml_workflow.validation.validators import validate_schema
        >>> result = validate_schema(df, required_columns=required)
    """
    required: List[str] = [
        # Identifiers / group keys
        "ticker",
        "isin",
        "sector",
        "region",
        "country",
        "trading_country",
        # Core business-critical price & targets
        "last_price",
        "price_target",
        "price_target_median",
        "price_target_ytd_ago",
        # Core market value columns used across ETL & features
        "market_cap",
        "enterprise_value",
    ]

    if include_extended_financials:
        extended = [
            # High-importance financials commonly used by ETL/feature presets
            "total_revenues_ltm",
            "ebitda_ltm",
            "net_income_is_ltm",
            "total_assets_ltm",
            "total_debt_ltm",
            "total_equity_ltm",
        ]
        for col in extended:
            if col not in required:
                required.append(col)

    # Defensive check: ensure all required columns are present in COLUMN_SCHEMA
    missing_from_schema = [col for col in required if col not in COLUMN_SCHEMA]
    if missing_from_schema:
        raise AssertionError(
            f"Required ETL columns not found in COLUMN_SCHEMA: {missing_from_schema}"
        )

    return required
