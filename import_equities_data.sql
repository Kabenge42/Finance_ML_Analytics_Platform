-- noinspection LongLineForFile

-- ===================================================================
-- Equities Data Import Script (Fixed with Explicit Column Names)
-- ===================================================================
-- This script imports financial data from CSV files into the equities table.
-- 
-- Key features:
-- - Explicit column names in INSERT to avoid positional misalignment
-- - Column aliases matching COLUMN_SCHEMA in schema.py
-- - Uses staging tables with TEXT columns to avoid type conversion errors during COPY
-- - Properly casts TEXT to NUMERIC/DATE in INSERT statements
-- - Handles empty strings as NULL values
--
-- Column Alias Reference (SQL → Python normalized name):
-- ======================================================
-- Identifiers:
--   "Ticker"                              → ticker
--   "ISIN"                                → isin
--   "Name"                                → name
--   "Description"                         → description
--   "Exchange"                            → exchange
--   "Unit"                                → unit
--   "Sector"                              → sector
--   "Industry"                            → industry
--   "Last Updated"                        → last_updated
--   "Income Statement Report Date"        → income_statement_report_date
--   "FY End"                              → fy_end
--   "Next Earnings"                       → next_earnings
--   "Next Earnings (When)"                → next_earnings_when
--   "Style Class"                         → style_class
--   "Next Earnings (Status)"              → next_earnings_status
--   "Size Class"                          → size_class
--   "Region"                              → region
--   "Country"                             → country
--   "Trading Country"                     → trading_country
--
-- Market Values:
--   "Market Cap"                          → market_cap
--   "Enterprise Value"                    → enterprise_value
--
-- Price Columns (NEVER transform):
--   "Last Price"                          → last_price
--   "Price Target"                        → price_target
--   "Price Target (YTD Ago)"              → price_target_ytd_ago
--   "Price Target - Low"                  → price_target_low
--   "Price Target - Median"               → price_target_median
--   "Price Target - High"                 → price_target_high
--   "Price Target - #"                    → price_target_count
--   "Price (5D Ago)"                      → price_5d_ago
--   "Price (1W Ago)"                      → price_1w_ago
--   "Price (1M Ago)"                      → price_1m_ago
--   "Price (3M Ago)"                      → price_3m_ago
--   "Price (6M Ago)"                      → price_6m_ago
--   "Price (1Y Ago)"                      → price_1y_ago
--   "Price (3Y Ago)"                      → price_3y_ago
--   "Price (5Y Ago)"                      → price_5y_ago
--   "Price (QTD Ago)"                     → price_qtd_ago
--   "52W High/Adj"                        → 52w_high_adj
--   "52W Low/Adj"                         → 52w_low_adj
--   "EMA (20D)"                           → ema_20d
--   "EMA (50D)"                           → ema_50d
--   "EMA (100D)"                          → ema_100d
--   "EMA (250D)"                          → ema_250d
--
-- Ratio Columns:
--   "P/E (NTM)"                           → p_e_ntm
--   "P/E (LTM)"                           → p_e_ltm
--   "P/E (EST FY1)"                       → p_e_est_fy1
--   "P/E (-1FYLTM)"                       → p_e_1fyltm
--   "P/E (-2FYLTM)"                       → p_e_2fyltm
--   "P/E (-3FYLTM)"                       → p_e_3fyltm
--   "P/E (3YAVGLTM)"                      → p_e_3yavgltm
--   "P/E (-1FQLTM)"                       → p_e_1fqltm
--   "P/E (-2FQLTM)"                       → p_e_2fqltm
--   "P/E (-3FQLTM)"                       → p_e_3fqltm
--   "P/E (5YAVGLTM)"                      → p_e_5yavgltm
--   "P/E (-0FQQoQLTM)"                    → p_e_0fqqoqltm
--   "P/E (-0FYYoYLTM)"                    → p_e_0fyyoyltm
--   "P/E (-1FYYoYLTM)"                    → p_e_1fyyoyltm
--   "P/E (-0FQYoYLTM)"                    → p_e_0fqyoyltm
--   "P/B (LTM)"                           → p_b_ltm
--   "P/B (-1FY)"                          → p_b_1fy
--   "P/B (5YAVG)"                         → p_b_5yavg
--   "P/TBV (LTM)"                         → p_tbv_ltm
--   "EV/Sales (LTM)"                      → ev_sales_ltm
--   "EV/Sales (NTM)"                      → ev_sales_ntm
--   "EV/Sales (EST FY1)"                  → ev_sales_est_fy1
--   "EV/Sales (-1FYLTM)"                  → ev_sales_1fyltm
--   "EV/Sales (-2FYLTM)"                  → ev_sales_2fyltm
--   "EV/Sales (-3FYLTM)"                  → ev_sales_3fyltm
--   "EV/Sales (3YAVGLTM)"                 → ev_sales_3yavgltm
--   "EV/Sales (-1FQLTM)"                  → ev_sales_1fqltm
--   "EV/Sales (-2FQLTM)"                  → ev_sales_2fqltm
--   "EV/Sales (-3FQLTM)"                  → ev_sales_3fqltm
--   "EV/Sales (-4FQLTM)"                  → ev_sales_4fqltm
--   "EV/EBITDA (LTM)"                     → ev_ebitda_ltm
--   "EV/EBITDA (NTM)"                     → ev_ebitda_ntm
--   "EV/EBITDA (EST FY1)"                 → ev_ebitda_est_fy1
--   "EV/EBITDA (-1FYLTM)"                 → ev_ebitda_1fyltm
--   "EV/EBITDA (-1FQLTM)"                 → ev_ebitda_1fqltm
--   "EV/EBITDA (3YAVGLTM)"                → ev_ebitda_3yavgltm
--   "Altman Z-Score (FY)"                 → altman_z_score_fy
--   "Altman Z-Score (FQ)"                 → altman_z_score_fq
--   "Altman Z-Score (LTM)"                → altman_z_score_ltm
--   "Current Ratio (FY)"                  → current_ratio_fy
--   "Current Ratio (LTM)"                 → current_ratio_ltm
--   "Asset Turnover (FY)"                 → asset_turnover_fy
--   "Asset Turnover (LTM)"                → asset_turnover_ltm
--
-- Percentage Columns:
--   "Beta (1Y)"                           → beta_1y
--   "Beta (2Y)"                           → beta_2y
--   "Beta (5Y)"                           → beta_5y
--   "Total Return (YTD)"                  → total_return_ytd
--   "Total Return (5Y)"                   → total_return_5y
--   "Total Return (10Y)"                  → total_return_10y
--   "Tot. Return %/CAGR (3Y)"             → tot_return_pct_cagr_3y
--   "Tot. Return %/CAGR (10Y)"            → tot_return_pct_cagr_10y
--   "Price Chg. % (1M)"                   → price_chg_pct_1m
--   "Price Chg. % (3M)"                   → price_chg_pct_3m
--   "1-Day %"                             → one_day_pct
--   "Volatility (1M)"                     → volatility_1m
--   "Volatility (3M)"                     → volatility_3m
--   "Volatility (6M)"                     → volatility_6m
--   "Volatility (1Y)"                     → volatility_1y
--   "Net Income Margin % (FY)"            → net_income_margin_pct_fy
--   "Net Income Margin % (LTM)"           → net_income_margin_pct_ltm
--   "Gross Profit Margin % (FY)"          → gross_profit_margin_pct_fy
--   "Gross Profit Margin % (LTM)"         → gross_profit_margin_pct_ltm
--   "Return On Equity % (LTM)"            → return_on_equity_pct_ltm
--   "Return On Equity % (FY)"             → return_on_equity_pct_fy
--   "Return on Assets (ROA) % (LTM)"      → return_on_assets_roa_pct_ltm
--   "Return on Assets (ROA) % (FY)"       → return_on_assets_roa_pct_fy
--   "Total Revenues/CAGR (5Y FY)"         → total_revenues_cagr_5y_fy
--   "Revenues - Est YoY % (FY1E)"         → revenues_est_yoy_pct_fy1e
--
-- Count Columns:
--   "Analyst Rating"                      → analyst_rating
--   "# Strong Sell Ratings"               → num_strong_sell_ratings
--   "# Strong Buys Ratings"               → num_strong_buys_ratings
--   "# Hold Ratings"                      → num_hold_ratings
--   "# Buys Ratings"                      → num_buys_ratings
--   "# Sell Ratings"                      → num_sell_ratings
--   "Shrs Out"                            → shares_outstanding
--   "Shrs Out (-1FY)"                     → shrs_out_1fy
--   "Full Time Employees (FQ)"            → full_time_employees_fq
--   "Full Time Employees (FY)"            → full_time_employees_fy
--   "Full Time Employees (-1FY)"          → full_time_employees_1fy
--   "Full Time Employees (-2FY)"          → full_time_employees_2fy
--   "Full Time Employees (-3FY)"          → full_time_employees_3fy
--   "Avg Employees (5YAVGFY)"             → avg_employees_5yavgfy
--   "Dividend Streak"                     → dividend_streak
--   "EPS Norm - Est # (FY1E)"             → eps_norm_est_num_fy1e
--
-- Market Value Columns (Log-transform recommended):
--   "TBV (FY)"                            → tbv_fy
--   "TBV (LTM)"                           → tbv_ltm
--   "Market Cap (Country R)"              → market_cap_country_r
--   "Total Revenues (FQ)"                 → total_revenues_fq
--   "Total Revenues (-1FY)"               → total_revenues_1fy
--   "Total Revenues (FY)"                 → total_revenues_fy
--   "Total Revenues (LTM)"                → total_revenues_ltm
--   "Total Revenues (5YAVGFQ)"            → total_revenues_5yavgfq
--   "Total Revenues (5YAVGLTM)"           → total_revenues_5yavgltm
--   "Total Operating Expenses (LTM)"      → total_operating_expenses_ltm
--   "EBITDA (FQ)"                         → ebitda_fq
--   "EBITDA (LTM)"                        → ebitda_ltm
--   "EBITDA (FY)"                         → ebitda_fy
--   "EBITDA (-1FY)"                       → ebitda_1fy
--   "EBITDA/Adj. (LTM)"                   → ebitda_adj_ltm
--   "EBITDA/Adj. (FY)"                    → ebitda_adj_fy
--   "EBITDA/Adj. (-1FY)"                  → ebitda_adj_1fy
--   "EBITDA (5YAVGFQ)"                    → ebitda_5yavgfq
--   "EBITDA (5YAVGLTM)"                   → ebitda_5yavgltm
--   "EBIT (FQ)"                           → ebit_fq
--   "EBIT (LTM)"                          → ebit_ltm
--   "EBIT (FY)"                           → ebit_fy
--   "EBIT (-1FY)"                         → ebit_1fy
--   "EBIT/Adj. (-1FY)"                    → ebit_adj_1fy
--   "EBIT/Adj. (FY)"                      → ebit_adj_fy
--   "EBIT/Adj. (LTM)"                     → ebit_adj_ltm
--   "EBIT - Est Med (FY1E)"               → ebit_est_med_fy1e
--   "EBIT - Est Med (NTM)"                → ebit_est_med_ntm
--   "EBIT (5YAVGFQ)"                      → ebit_5yavgfq
--   "EBIT (5YAVGLTM)"                     → ebit_5yavgltm
--   "Net Income - (IS) (FY)"              → net_income_is_fy
--   "Net Income - (IS) (LTM)"             → net_income_is_ltm
--   "Net Income - (IS) (FQ)"              → net_income_is_fq
--   "Net Income - (IS) (-1FY)"            → net_income_is_1fy
--   "Net Income - (IS) (5YAVGFQ)"         → net_income_is_5yavgfq
--   "Net Income - (IS) (5YAVGLTM)"        → net_income_is_5yavgltm
--   "Normalized Net Income (FY)"          → normalized_net_income_fy
--   "Normalized Net Income (LTM)"         → normalized_net_income_ltm
--   "Normalized Net Income (FQ)"          → normalized_net_income_fq
--   "Normalized Net Income (-1FY)"        → normalized_net_income_1fy
--   "Normalized Net Income (5YAVGFQ)"     → normalized_net_income_5yavgfq
--   "Normalized Net Income (5YAVGLTM)"    → normalized_net_income_5yavgltm
--   "Net Income/Adj. (FY)"                → net_income_adj_fy
--   "Net Income/Adj. (LTM)"               → net_income_adj_ltm
--   "Net Income/Adj. (FQ)"                → net_income_adj_fq
--   "Net Income/Adj. (-1FY)"              → net_income_adj_1fy
--   "Net Income/Adj. (5YAVGFQ)"           → net_income_adj_5yavgfq
--   "Operating Income (LTM)"              → operating_income_ltm
--   "Operating Income (FY)"               → operating_income_fy
--   "Operating Income (FQ)"               → operating_income_fq
--   "Operating Income (5YAVGFQ)"          → operating_income_5yavgfq
--   "Gross Profit (LTM)"                  → gross_profit_ltm
--   "Gross Profit (FY)"                   → gross_profit_fy
--   "Total Debt (FY)"                     → total_debt_fy
--   "Total Debt (LTM)"                    → total_debt_ltm
--   "Total Equity (FY)"                   → total_equity_fy
--   "Total Equity (LTM)"                  → total_equity_ltm
--   "Total Assets (LTM)"                  → total_assets_ltm
--   "Total Assets (FY)"                   → total_assets_fy
--   "Total Current Assets (LTM)"          → total_current_assets_ltm
--   "Total Current Liabilities (LTM)"     → total_current_liabilities_ltm
--   "Working Capital (LTM)"               → working_capital_ltm
--   "Working Capital (FQ)"                → working_capital_fq
--   "Working Capital (FY)"                → working_capital_fy
--   "Working Capital (5YAVGFY)"           → working_capital_5yavgfy
--   "Cash And Equivalents (LTM)"          → cash_and_equivalents_ltm
--   "Cash And Equivalents (FQ)"           → cash_and_equivalents_fq
--   "Cash And Equivalents (FY)"           → cash_and_equivalents_fy
--   "Cash And Equivalents (5YAVGFQ)"      → cash_and_equivalents_5yavgfq
--   "Retained Earnings (LTM)"             → retained_earnings_ltm
--   "Retained Earnings (FQ)"              → retained_earnings_fq
--   "Retained Earnings (FY)"              → retained_earnings_fy
--   "Retained Earnings (5YAVGFQ)"         → retained_earnings_5yavgfq
--   "CFO (LTM)"                           → cfo_ltm
--   "CFO (FY)"                            → cfo_fy
--   "CFO (FQ)"                            → cfo_fq
--   "CFO (-1FY)"                          → cfo_1fy
--   "FCF (LTM)"                           → fcf_ltm
--   "FCF (FY)"                            → fcf_fy
--   "FCF (FQ)"                            → fcf_fq
--   "FCF (5YAVGFQ)"                       → fcf_5yavgfq
--   "CFI (LTM)"                           → cfi_ltm
--   "CFI (FY)"                            → cfi_fy
--   "CFI (FQ)"                            → cfi_fq
--   "CFI (-1FY)"                          → cfi_1fy
--   "CFF (LTM)"                           → cff_ltm
--   "CFF (FY)"                            → cff_fy
--   "CFF (FQ)"                            → cff_fq
--   "CFF (-1FY)"                          → cff_1fy
--   "Capital Expenditure (LTM)"           → capital_expenditure_ltm
--   "Capital Expenditure (FY)"            → capital_expenditure_fy
--   "Capital Expenditure (FQ)"            → capital_expenditure_fq
--   "Capital Expenditure (-1FY)"          → capital_expenditure_1fy
--   "Capital Expenditure (5YAVGFQ)"       → capital_expenditure_5yavgfq
--   "R&D Expenses (LTM)"                  → randd_expenses_ltm
--   "Revenues - Est Avg (NTM)"            → revenues_est_avg_ntm
--   "Revenues - Est Avg (FY1E)"           → revenues_est_avg_fy1e
--   "Revenues - Est Med (NTM)"            → revenues_est_med_ntm
--   "Revenues - Est Med (FY1E)"           → revenues_est_med_fy1e
--   "EBITDA - Est Avg (NTM)"              → ebitda_est_avg_ntm
--   "EBITDA - Est Avg (FY1E)"             → ebitda_est_avg_fy1e
--
-- Feature Columns:
--   "Volume (Shrs)"                       → volume_shrs
--   "Rel. Volume"                         → rel_volume
--   "Dividend Per Share (LTM)"            → dividend_per_share_ltm
--   "Div Yield (Ind)"                     → div_yield_ind
--   "Div Yield (LTM)"                     → div_yield_ltm
--   "Div Yield (TTM)"                     → div_yield_ttm
--   "Div Yield (NTM)"                     → div_yield_ntm
--   "Div Yield (-1FYInd)"                 → div_yield_1fyind
--   "Div Yield (-2FYInd)"                 → div_yield_2fyind
--   "Div Yield (-3FYInd)"                 → div_yield_3fyind
--   "Div Yield (-4FYInd)"                 → div_yield_4fyind
--   "Div Yield (-5FYInd)"                 → div_yield_5fyind
--   "Div Yield (5YAVGLTM)"                → div_yield_5yavgltm
--   "Common Dividends Paid (LTM)"         → common_dividends_paid_ltm
--   "Common Dividends Paid (FY)"          → common_dividends_paid_fy
--   "Buyback Yield (LTM)"                 → buyback_yield_ltm
--   "EPS Norm - Est Avg (NTM)"            → eps_norm_est_avg_ntm
--   "EPS Norm - Est Avg (FY1E)"           → eps_norm_est_avg_fy1e
--   "EPS/Adj. (-1FY)"                     → eps_adj_1fy
--   "EPS/Adj. (FY)"                       → eps_adj_fy
--   "EPS/Adj. (LTM)"                      → eps_adj_ltm
--   "Net EPS - Basic (LTM)"               → net_eps_basic_ltm
--   "Net EPS - Basic (FQ)"                → net_eps_basic_fq
--   "Net EPS - Basic (FY)"                → net_eps_basic_fy
--   "Net EPS - Basic (-1FQFQ)"            → net_eps_basic_1fqfq
--   "Net EPS - Basic (-2FQFQ)"            → net_eps_basic_2fqfq
--   "Net EPS - Basic (-3FQFQ)"            → net_eps_basic_3fqfq
--   "Net EPS - Basic (-4FQFQ)"            → net_eps_basic_4fqfq
--   "Net EPS - Basic (-1FY)"              → net_eps_basic_1fy
--   "Net EPS - Basic (-2FY)"              → net_eps_basic_2fy
--   "Net EPS - Basic (-3FY)"              → net_eps_basic_3fy
--   "Net EPS - Basic (-4FY)"              → net_eps_basic_4fy
--   "Net EPS - Basic (-5FY)"              → net_eps_basic_5fy
--   "EPS Est Avg Rev % (FY1E - 1W)"       → eps_est_avg_rev_pct_fy1e_1w
--   "EPS Est Avg Rev % (FY1E - 1M)"       → eps_est_avg_rev_pct_fy1e_1m
--   "EPS Est Avg Rev % (FY1E - 3M)"       → eps_est_avg_rev_pct_fy1e_3m
--   "EPS Est Avg Rev % (FY1E - 6M)"       → eps_est_avg_rev_pct_fy1e_6m
--   "EPS Est Avg Rev % (FY1E - 1Y)"       → eps_est_avg_rev_pct_fy1e_1y
--   "EPS GAAP - Est Avg (NTM)"            → eps_gaap_est_avg_ntm
--   "EPS GAAP - Est Avg (FY1E)"           → eps_gaap_est_avg_fy1e
--   "EPS GAAP Est Avg Rev % (FY1E - 1M)"  → eps_gaap_est_avg_rev_pct_fy1e_1m
--   "EPS GAAP Est Avg Rev % (FY1E - 3M)"  → eps_gaap_est_avg_rev_pct_fy1e_3m
--   "EPS GAAP Est Avg Rev % (FY1E - 6M)"  → eps_gaap_est_avg_rev_pct_fy1e_6m
--   "EPS GAAP Est Avg Rev % (FY1E - 1Y)"  → eps_gaap_est_avg_rev_pct_fy1e_1y
--   "Inventory (LTM)"                     → inventory_ltm
--   "Inventory (FQ)"                      → inventory_fq
--   "Inventory (FY)"                      → inventory_fy
--   "Inventory (5YAVGFQ)"                 → inventory_5yavgfq
--   "Goodwill (FQ)"                       → goodwill_fq
--   "Goodwill (LTM)"                      → goodwill_ltm
--   "Goodwill (FY)"                       → goodwill_fy
--   "Goodwill (-1FY)"                     → goodwill_1fy
--   "Goodwill (5YAVGFQ)"                  → goodwill_5yavgfq
--   "Impairment of Goodwill (FQ)"         → impairment_of_goodwill_fq
--   "Impairment of Goodwill (LTM)"        → impairment_of_goodwill_ltm
--   "Impairment of Goodwill (-1FY)"       → impairment_of_goodwill_1fy
--   "Impairment of Goodwill (FY)"         → impairment_of_goodwill_fy
--   "Impairment of Goodwill (5YAVGFQ)"    → impairment_of_goodwill_5yavgfq
--   "Asset Writedown (LTM)"               → asset_writedown_ltm
--   "Asset Writedown (FY)"                → asset_writedown_fy
--   "Asset Writedown (FQ)"                → asset_writedown_fq
--   "Asset Writedown (-1FY)"              → asset_writedown_1fy
--   "Asset Writedown (5YAVGFQ)"           → asset_writedown_5yavgfq
--   "Restructuring Charges (LTM)"         → restructuring_charges_ltm
--   "Restructuring Charges (FQ)"          → restructuring_charges_fq
--   "Restructuring Charges (-1FY)"        → restructuring_charges_1fy
--   "Restructuring Charges (FY)"          → restructuring_charges_fy
--   "Restructuring Charges (5YAVGFQ)"     → restructuring_charges_5yavgfq
--   "Merger & Restructuring Charges (LTM)" → merger_and_restructuring_charges_ltm
--   "Merger & Restructuring Charges (FQ)"  → merger_and_restructuring_charges_fq
--   "Merger & Restructuring Charges (FY)"  → merger_and_restructuring_charges_fy
--   "Merger & Restructuring Charges (5YAVGFQ)" → merger_and_restructuring_charges_5yavgfq
--   "Interest Expense/Total (LTM)"        → interest_expense_total_ltm
--   "Interest Income On Investments (LTM)" → interest_income_on_investments_ltm
--   "Gain (Loss) On Sale Of Assets (LTM)" → gain_loss_on_sale_of_assets_ltm
--   "Cost Of Revenues (LTM)"              → cost_of_revenues_ltm
--   "Other Unusual Items/Total (LTM)"     → other_unusual_items_total_ltm
--   "Cash Acquisitions (LTM)"             → cash_acquisitions_ltm
--   "Cash Acquisitions (FY)"              → cash_acquisitions_fy
--   "Cash Acquisitions (FQ)"              → cash_acquisitions_fq
--   "Cash Acquisitions (-1FY)"            → cash_acquisitions_1fy
--   "Cash Acquisitions (5YAVGFQ)"         → cash_acquisitions_5yavgfq
--   "Gross Intangible Assets (LTM)"       → gross_intangible_assets_ltm
--   "Gross Intangible Assets (FY)"        → gross_intangible_assets_fy
--   "Gross Intangible Assets (5YAVGFQ)"   → gross_intangible_assets_5yavgfq
--   "Selling General & Admin Expenses/Total (FQ)"      → selling_general_and_admin_expenses_total_fq
--   "Selling General & Admin Expenses/Total (FY)"      → selling_general_and_admin_expenses_total_fy
--   "Selling General & Admin Expenses/Total (-1FY)"    → selling_general_and_admin_expenses_total_1fy
--   "Selling General & Admin Expenses/Total (5YAVGFQ)" → selling_general_and_admin_expenses_total_5yavgfq
--   "Accounts Receivable/Total (FY)"      → accounts_receivable_total_fy
--   "Accounts Receivable/Total (-1FY)"    → accounts_receivable_total_1fy
--   "Accounts Receivable/Total (5YAVGFQ)" → accounts_receivable_total_5yavgfq
--   "Marketing Expenses (FQ)"             → marketing_expenses_fq
--   "Marketing Expenses (FY)"             → marketing_expenses_fy
--   "Marketing Expenses (-1FY)"           → marketing_expenses_1fy
--   "Marketing Expenses (5YAVGLTM)"       → marketing_expenses_5yavgltm
--
-- Dividend Record Columns:
--   "Dividend Record (Announce Date)"     → dividend_record_announce_date
--   "Dividend Record (Ex Date)"           → dividend_record_ex_date
--   "Dividend Record (Payable Date)"      → dividend_record_payable_date
--   "Dividend Record (Record Date)"       → dividend_record_record_date
--   "Dividend Record (Frequency)"         → dividend_record_frequency
--   "Dividend Record (Currency)"          → dividend_record_currency
--   "Dividend Record (Amount)"            → dividend_record_amount
--
-- Usage:psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
--
--
-- ===================================================================

-- Enable verbose output
\echo 'Starting equities data import...'

DO
$$
    BEGIN
        RAISE NOTICE 'Import started at %', NOW();
    END
$$;

-- Show current table status
SELECT 'Current equities table row count:' AS status, COUNT(*) AS row_count
FROM equities;

-- ===================================================================
-- HELPER FUNCTION: Safe Numeric Conversion
-- ===================================================================
-- Converts TEXT to NUMERIC, treating common non-numeric patterns as NULL
-- Handles: "-", "--", "N/A", "n/a", "NA", empty strings, whitespace-only

CREATE OR REPLACE FUNCTION safe_to_numeric(input_text TEXT)
RETURNS NUMERIC AS $$
BEGIN
    -- Return NULL for common non-numeric patterns
    IF input_text IS NULL 
       OR TRIM(input_text) = '' 
       OR TRIM(input_text) = '-'
       OR TRIM(input_text) = '--'
       OR UPPER(TRIM(input_text)) = 'N/A'
       OR UPPER(TRIM(input_text)) = 'NA'
       OR UPPER(TRIM(input_text)) = 'NULL'
       OR UPPER(TRIM(input_text)) = 'NONE'
    THEN
        RETURN NULL;
    END IF;
    
    -- Attempt numeric conversion
    RETURN TRIM(input_text)::NUMERIC;
EXCEPTION
    WHEN invalid_text_representation THEN
        RETURN NULL;
    WHEN numeric_value_out_of_range THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- Importing US Region Data...
-- ===================================================================
\echo 'Importing regional data (US, EU, APAC, ROTW)...'

-- Drop staging table if exists
DROP TABLE IF EXISTS screening_staging;

-- Create staging table with all columns as TEXT


-- ===================================================================
-- STAGING TABLE CREATION
-- ===================================================================
-- Create staging table with TEXT columns to avoid type conversion errors
-- Column order matches CSV file exactly

DROP TABLE IF EXISTS screening_staging;

CREATE TEMP TABLE screening_staging
(
    "Ticker"                                           TEXT,
    "ISIN"                                             TEXT,
    "Name"                                             TEXT,
    "Description"                                      TEXT,
    "Region"                                           TEXT,
    "Country"                                          TEXT,
    "Trading Country"                                  TEXT,
    "Exchange"                                         TEXT,
    "Unit"                                             TEXT,
    "Sector"                                           TEXT,
    "Industry"                                         TEXT,
    "Style Class"                                      TEXT,
    "Size Class"                                       TEXT,
    "Last Updated"                                     TEXT,
    "Income Statement Report Date"                     TEXT,
    "FY End"                                           TEXT,
    "Next Earnings"                                    TEXT,
    "Next Earnings (When)"                             TEXT,
    "Next Earnings (Status)"                           TEXT,
    "Dividend Record (Currency)"                       TEXT,
    "Dividend Record (Amount)"                         TEXT,
    "Dividend Record (Frequency)"                      TEXT,
    "Dividend Streak"                                  TEXT,
    "Dividend Record (Announce Date)"                  TEXT,
    "Dividend Record (Payable Date)"                   TEXT,
    "Dividend Record (Record Date)"                    TEXT,
    "Dividend Record (Ex Date)"                        TEXT,
    "Market Cap"                                       TEXT,
    "Enterprise Value"                                 TEXT,
    "Last Price"                                       TEXT,
    "Price Target (YTD Ago)"                           TEXT,
    "Total Return (YTD)"                               TEXT,
    "Price Target"                                     TEXT,
    "Price Target - Low"                               TEXT,
    "Price Target - Median"                            TEXT,
    "Price Target - High"                              TEXT,
    "Price Target - #"                                 TEXT,
    "P/E (NTM)"                                        TEXT,
    "P/E (LTM)"                                        TEXT,
    "Altman Z-Score (FY)"                              TEXT,
    "Altman Z-Score (FQ)"                              TEXT,
    "Altman Z-Score (LTM)"                             TEXT,
    "Beta (1Y)"                                        TEXT,
    "Beta (2Y)"                                        TEXT,
    "Beta (5Y)"                                        TEXT,
    "Analyst Rating"                                   TEXT,
    "# Strong Sell Ratings"                            TEXT,
    "# Strong Buys Ratings"                            TEXT,
    "# Hold Ratings"                                   TEXT,
    "# Buys Ratings"                                   TEXT,
    "# Sell Ratings"                                   TEXT,
    "Total Revenues/CAGR (5Y FY)"                      TEXT,
    "Total Revenues (FQ)"                              TEXT,
    "Total Revenues (-1FY)"                            TEXT,
    "Total Revenues (FY)"                              TEXT,
    "Total Revenues (LTM)"                             TEXT,
    "Total Operating Expenses (LTM)"                   TEXT,
    "P/TBV (LTM)"                                      TEXT,
    "TBV (FY)"                                         TEXT,
    "TBV (LTM)"                                        TEXT,
    "Market Cap (Country R)"                           TEXT,
    "Tot. Return %/CAGR (3Y)"                          TEXT,
    "Tot. Return %/CAGR (10Y)"                         TEXT,
    "Total Return (5Y)"                                TEXT,
    "Total Return (10Y)"                               TEXT,
    "Net Income/Adj. (-1FY)"                           TEXT,
    "CFF (LTM)"                                        TEXT,
    "CFI (LTM)"                                        TEXT,
    "FCF (LTM)"                                        TEXT,
    "CFO (LTM)"                                        TEXT,
    "EBITDA (FQ)"                                      TEXT,
    "EBITDA (LTM)"                                     TEXT,
    "EBITDA (FY)"                                      TEXT,
    "EBITDA (-1FY)"                                    TEXT,
    "EBITDA/Adj. (LTM)"                                TEXT,
    "EBITDA/Adj. (FY)"                                 TEXT,
    "EBITDA/Adj. (-1FY)"                               TEXT,
    "EBIT (FQ)"                                        TEXT,
    "EBIT (LTM)"                                       TEXT,
    "EBIT (FY)"                                        TEXT,
    "EBIT (-1FY)"                                      TEXT,
    "EBIT/Adj. (-1FY)"                                 TEXT,
    "EBIT/Adj. (FY)"                                   TEXT,
    "EBIT/Adj. (LTM)"                                  TEXT,
    "EBIT - Est Med (FY1E)"                            TEXT,
    "EBIT - Est Med (NTM)"                             TEXT,
    "Return On Equity % (LTM)"                         TEXT,
    "Return On Equity % (FY)"                          TEXT,
    "Net Income - (IS) (FY)"                           TEXT,
    "Net Income - (IS) (LTM)"                          TEXT,
    "Normalized Net Income (FY)"                       TEXT,
    "Normalized Net Income (LTM)"                      TEXT,
    "Net Income/Adj. (FY)"                             TEXT,
    "Net Income/Adj. (LTM)"                            TEXT,
    "Net Income Margin % (FY)"                         TEXT,
    "Net Income Margin % (LTM)"                        TEXT,
    "Volatility (1M)"                                  TEXT,
    "Volatility (3M)"                                  TEXT,
    "Volatility (6M)"                                  TEXT,
    "Volatility (1Y)"                                  TEXT,
    "Volume (Shrs)"                                    TEXT,
    "Dividend Per Share (LTM)"                         TEXT,
    "Div Yield (Ind)"                                  TEXT,
    "Div Yield (LTM)"                                  TEXT,
    "Total Debt (FY)"                                  TEXT,
    "Total Equity (FY)"                                TEXT,
    "Total Equity (LTM)"                               TEXT,
    "Total Debt (LTM)"                                 TEXT,
    "Total Assets (LTM)"                               TEXT,
    "Total Assets (FY)"                                TEXT,
    "Current Ratio (FY)"                               TEXT,
    "Current Ratio (LTM)"                              TEXT,
    "Gross Profit Margin % (FY)"                       TEXT,
    "Gross Profit Margin % (LTM)"                      TEXT,
    "Asset Turnover (FY)"                              TEXT,
    "Asset Turnover (LTM)"                             TEXT,
    "Gross Profit (LTM)"                               TEXT,
    "Gross Profit (FY)"                                TEXT,
    "EPS Norm - Est Avg (NTM)"                         TEXT,
    "EPS/Adj. (-1FY)"                                  TEXT,
    "EPS/Adj. (FY)"                                    TEXT,
    "EPS/Adj. (LTM)"                                   TEXT,
    "EPS Norm - Est Avg (FY1E)"                        TEXT,
    "Gain (Loss) On Sale Of Assets (LTM)"              TEXT,
    "Cost Of Revenues (LTM)"                           TEXT,
    "Cash Acquisitions (LTM)"                          TEXT,
    "Cash Acquisitions (FY)"                           TEXT,
    "Cash Acquisitions (-1FY)"                         TEXT,
    "Inventory (LTM)"                                  TEXT,
    "Goodwill (FQ)"                                    TEXT,
    "Goodwill (LTM)"                                   TEXT,
    "Goodwill (FY)"                                    TEXT,
    "Goodwill (-1FY)"                                  TEXT,
    "Impairment of Goodwill (FQ)"                      TEXT,
    "Impairment of Goodwill (LTM)"                     TEXT,
    "Impairment of Goodwill (-1FY)"                    TEXT,
    "Impairment of Goodwill (FY)"                      TEXT,
    "Operating Income (LTM)"                           TEXT,
    "Asset Writedown (LTM)"                            TEXT,
    "Asset Writedown (FY)"                             TEXT,
    "Asset Writedown (-1FY)"                           TEXT,
    "Operating Income (FY)"                            TEXT,
    "Capital Expenditure (LTM)"                        TEXT,
    "Capital Expenditure (-1FY)"                       TEXT,
    "Capital Expenditure (FY)"                         TEXT,
    "Retained Earnings (LTM)"                          TEXT,
    "Total Current Assets (LTM)"                       TEXT,
    "Total Current Liabilities (LTM)"                  TEXT,
    "R&D Expenses (LTM)"                               TEXT,
    "Restructuring Charges (LTM)"                      TEXT,
    "Restructuring Charges (FQ)"                       TEXT,
    "Restructuring Charges (-1FY)"                     TEXT,
    "Restructuring Charges (FY)"                       TEXT,
    "Interest Expense/Total (LTM)"                     TEXT,
    "Merger & Restructuring Charges (LTM)"             TEXT,
    "Working Capital (LTM)"                            TEXT,
    "Other Unusual Items/Total (LTM)"                  TEXT,
    "Interest Income On Investments (LTM)"             TEXT,
    "Buyback Yield (LTM)"                              TEXT,
    "Return on Assets (ROA) % (LTM)"                   TEXT,
    "Return on Assets (ROA) % (FY)"                    TEXT,
    "Net Income - (IS) (-1FY)"                         TEXT,
    "Normalized Net Income (-1FY)"                     TEXT,
    "CFF (FY)"                                         TEXT,
    "CFF (-1FY)"                                       TEXT,
    "CFI (FY)"                                         TEXT,
    "CFI (-1FY)"                                       TEXT,
    "CFO (FY)"                                         TEXT,
    "CFO (-1FY)"                                       TEXT,
    "Div Yield (-1FYInd)"                              TEXT,
    "FCF (FY)"                                         TEXT,
    "Capital Expenditure (FQ)"                         TEXT,
    "Capital Expenditure (5YAVGFQ)"                    TEXT,
    "CFF (FQ)"                                         TEXT,
    "CFI (FQ)"                                         TEXT,
    "CFO (FQ)"                                         TEXT,
    "FCF (FQ)"                                         TEXT,
    "Total Revenues (5YAVGFQ)"                         TEXT,
    "EBITDA (5YAVGFQ)"                                 TEXT,
    "EBIT (5YAVGFQ)"                                   TEXT,
    "FCF (5YAVGFQ)"                                    TEXT,
    "Cash Acquisitions (FQ)"                           TEXT,
    "Cash Acquisitions (5YAVGFQ)"                      TEXT,
    "Asset Writedown (FQ)"                             TEXT,
    "Asset Writedown (5YAVGFQ)"                        TEXT,
    "Impairment of Goodwill (5YAVGFQ)"                 TEXT,
    "Operating Income (FQ)"                            TEXT,
    "Operating Income (5YAVGFQ)"                       TEXT,
    "P/B (LTM)"                                        TEXT,
    "P/B (-1FY)"                                       TEXT,
    "P/B (5YAVG)"                                      TEXT,
    "Cash And Equivalents (LTM)"                       TEXT,
    "Cash And Equivalents (FQ)"                        TEXT,
    "Cash And Equivalents (FY)"                        TEXT,
    "Cash And Equivalents (5YAVGFQ)"                   TEXT,
    "Inventory (FQ)"                                   TEXT,
    "Inventory (FY)"                                   TEXT,
    "Goodwill (5YAVGFQ)"                               TEXT,
    "Inventory (5YAVGFQ)"                              TEXT,
    "Retained Earnings (FQ)"                           TEXT,
    "Retained Earnings (FY)"                           TEXT,
    "Retained Earnings (5YAVGFQ)"                      TEXT,
    "Working Capital (FQ)"                             TEXT,
    "Working Capital (FY)"                             TEXT,
    "Working Capital (5YAVGFY)"                        TEXT,
    "Div Yield (TTM)"                                  TEXT,
    "Div Yield (NTM)"                                  TEXT,
    "Div Yield (5YAVGLTM)"                             TEXT,
    "Gross Intangible Assets (LTM)"                    TEXT,
    "Gross Intangible Assets (FY)"                     TEXT,
    "Gross Intangible Assets (5YAVGFQ)"                TEXT,
    "Restructuring Charges (5YAVGFQ)"                  TEXT,
    "Merger & Restructuring Charges (FQ)"              TEXT,
    "Merger & Restructuring Charges (FY)"              TEXT,
    "Merger & Restructuring Charges (5YAVGFQ)"         TEXT,
    "Normalized Net Income (FQ)"                       TEXT,
    "Normalized Net Income (5YAVGFQ)"                  TEXT,
    "Net Income/Adj. (FQ)"                             TEXT,
    "Net Income/Adj. (5YAVGFQ)"                        TEXT,
    "Net Income - (IS) (FQ)"                           TEXT,
    "Net Income - (IS) (5YAVGFQ)"                      TEXT,
    "Net Income - (IS) (5YAVGLTM)"                     TEXT,
    "Normalized Net Income (5YAVGLTM)"                 TEXT,
    "EBITDA (5YAVGLTM)"                                TEXT,
    "EBIT (5YAVGLTM)"                                  TEXT,
    "Total Revenues (5YAVGLTM)"                        TEXT,
    "Revenues - Est YoY % (FY1E)"                      TEXT,
    "Price Chg. % (1M)"                                TEXT,
    "Price Chg. % (3M)"                                TEXT,
    "1-Day %"                                          TEXT,
    "Price (5D Ago)"                                   TEXT,
    "Price (1W Ago)"                                   TEXT,
    "Price (1M Ago)"                                   TEXT,
    "Price (3M Ago)"                                   TEXT,
    "Price (6M Ago)"                                   TEXT,
    "Price (1Y Ago)"                                   TEXT,
    "Price (3Y Ago)"                                   TEXT,
    "Price (5Y Ago)"                                   TEXT,
    "Price (QTD Ago)"                                  TEXT,
    "Rel. Volume"                                      TEXT,
    "Shrs Out"                                         TEXT,
    "Shrs Out (-1FY)"                                  TEXT,
    "Common Dividends Paid (LTM)"                      TEXT,
    "Common Dividends Paid (FY)"                       TEXT,
    "Selling General & Admin Expenses/Total (FQ)"      TEXT,
    "Selling General & Admin Expenses/Total (FY)"      TEXT,
    "Selling General & Admin Expenses/Total (-1FY)"    TEXT,
    "Selling General & Admin Expenses/Total (5YAVGFQ)" TEXT,
    "Accounts Receivable/Total (FY)"                   TEXT,
    "Accounts Receivable/Total (-1FY)"                 TEXT,
    "Accounts Receivable/Total (5YAVGFQ)"              TEXT,
    "Marketing Expenses (FQ)"                          TEXT,
    "Marketing Expenses (FY)"                          TEXT,
    "Marketing Expenses (-1FY)"                        TEXT,
    "Marketing Expenses (5YAVGLTM)"                    TEXT,
    "Revenues - Est Avg (NTM)"                         TEXT,
    "Revenues - Est Avg (FY1E)"                        TEXT,
    "Revenues - Est Med (NTM)"                         TEXT,
    "Revenues - Est Med (FY1E)"                        TEXT,
    "EV/Sales (EST FY1)"                               TEXT,
    "EV/Sales (LTM)"                                   TEXT,
    "EV/Sales (NTM)"                                   TEXT,
    "EV/Sales (-1FYLTM)"                               TEXT,
    "EV/Sales (-2FYLTM)"                               TEXT,
    "EV/Sales (-3FYLTM)"                               TEXT,
    "EV/Sales (3YAVGLTM)"                              TEXT,
    "EV/Sales (-1FQLTM)"                               TEXT,
    "EV/Sales (-2FQLTM)"                               TEXT,
    "EV/Sales (-3FQLTM)"                               TEXT,
    "EV/Sales (-4FQLTM)"                               TEXT,
    "52W High/Adj"                                     TEXT,
    "52W Low/Adj"                                      TEXT,
    "EMA (20D)"                                        TEXT,
    "EMA (50D)"                                        TEXT,
    "EMA (100D)"                                       TEXT,
    "EMA (250D)"                                       TEXT,
    "EV/EBITDA (LTM)"                                  TEXT,
    "EV/EBITDA (NTM)"                                  TEXT,
    "EV/EBITDA (-1FYLTM)"                              TEXT,
    "EV/EBITDA (-1FQLTM)"                              TEXT,
    "EV/EBITDA (3YAVGLTM)"                             TEXT,
    "EV/EBITDA (EST FY1)"                              TEXT,
    "P/E (EST FY1)"                                    TEXT,
    "P/E (-1FYLTM)"                                    TEXT,
    "P/E (-2FYLTM)"                                    TEXT,
    "P/E (-3FYLTM)"                                    TEXT,
    "P/E (3YAVGLTM)"                                   TEXT,
    "P/E (-1FQLTM)"                                    TEXT,
    "P/E (-2FQLTM)"                                    TEXT,
    "P/E (-3FQLTM)"                                    TEXT,
    "P/E (5YAVGLTM)"                                   TEXT,
    "P/E (-0FQQoQLTM)"                                 TEXT,
    "P/E (-0FYYoYLTM)"                                 TEXT,
    "P/E (-1FYYoYLTM)"                                 TEXT,
    "P/E (-0FQYoYLTM)"                                 TEXT,
    "Full Time Employees (FQ)"                         TEXT,
    "Full Time Employees (FY)"                         TEXT,
    "Full Time Employees (-1FY)"                       TEXT,
    "Full Time Employees (-2FY)"                       TEXT,
    "Full Time Employees (-3FY)"                       TEXT,
    "Avg Employees (5YAVGFY)"                          TEXT,
    "Net EPS - Basic (LTM)"                            TEXT,
    "Net EPS - Basic (FQ)"                             TEXT,
    "Net EPS - Basic (FY)"                             TEXT,
    "Net EPS - Basic (-1FQFQ)"                         TEXT,
    "Net EPS - Basic (-2FQFQ)"                         TEXT,
    "Net EPS - Basic (-3FQFQ)"                         TEXT,
    "Net EPS - Basic (-4FQFQ)"                         TEXT,
    "Net EPS - Basic (-1FY)"                           TEXT,
    "Net EPS - Basic (-2FY)"                           TEXT,
    "Net EPS - Basic (-3FY)"                           TEXT,
    "Net EPS - Basic (-4FY)"                           TEXT,
    "Net EPS - Basic (-5FY)"                           TEXT,
    "EPS Est Avg Rev % (FY1E - 1W)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 1M)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 3M)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 6M)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 1Y)"                    TEXT,
    "Div Yield (-2FYInd)"                              TEXT,
    "Div Yield (-3FYInd)"                              TEXT,
    "Div Yield (-4FYInd)"                              TEXT,
    "Div Yield (-5FYInd)"                              TEXT,
    "EBITDA - Est Avg (NTM)"                           TEXT,
    "EBITDA - Est Avg (FY1E)"                          TEXT,
    "EPS GAAP - Est Avg (NTM)"                         TEXT,
    "EPS GAAP - Est Avg (FY1E)"                        TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 1M)"               TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 3M)"               TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 6M)"               TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 1Y)"               TEXT,
    "EPS Norm - Est # (FY1E)"                          TEXT
);

-- ===================================================================
-- DATA IMPORT FROM CSV FILES
-- ===================================================================
-- Import data from regional CSV files into staging table
-- Uses \copy for client-side import (works with relative paths)
-- NULL '' treats empty strings as NULL values
-- ENCODING 'UTF8' ensures proper character handling

-- US Region
\echo 'Importing US data...'
\copy screening_staging FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');

-- EU Region
\echo 'Importing EU data...'
\copy screening_staging FROM 'data/screening_eu.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');

-- APAC Region
\echo 'Importing APAC data...'
\copy screening_staging FROM 'data/screening_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');

-- ROTW Region
\echo 'Importing ROTW data...'
\copy screening_staging FROM 'data/screening_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');

-- ===================================================================
-- DATA VALIDATION
-- ===================================================================
\echo 'Validating imported data...'
SELECT 'Total rows in staging:' AS info, COUNT(*) AS count
FROM screening_staging;
SELECT 'Rows by Region:' AS info, "Region", COUNT(*) AS count
FROM screening_staging
GROUP BY "Region"
ORDER BY "Region";
SELECT 'Rows with missing Ticker:' AS info, COUNT(*) AS count
FROM screening_staging
WHERE "Ticker" IS NULL
   OR "Ticker" = '';
SELECT 'Rows with missing Sector:' AS info, COUNT(*) AS count
FROM screening_staging
WHERE "Sector" IS NULL
   OR "Sector" = '';

-- ===================================================================
-- INSERT INTO MAIN TABLE
-- ===================================================================
-- Insert data from staging into main equities table
-- Cast TEXT columns to appropriate types (NUMERIC, DATE)
-- NULLIF handles empty strings

\echo 'Inserting data into equities table...'

INSERT INTO equities ("Ticker",
                      "ISIN",
                      "Name",
                      "Description",
                      "Region",
                      "Country",
                      "Trading Country",
                      "Exchange",
                      "Unit",
                      "Sector",
                      "Industry",
                      "Style Class",
                      "Size Class",
                      "Last Updated",
                      "Income Statement Report Date",
                      "FY End",
                      "Next Earnings",
                      "Next Earnings (When)",
                      "Next Earnings (Status)",
                      "Dividend Record (Currency)",
                      "Dividend Record (Amount)",
                      "Dividend Record (Frequency)",
                      "Dividend Streak",
                      "Dividend Record (Announce Date)",
                      "Dividend Record (Payable Date)",
                      "Dividend Record (Record Date)",
                      "Dividend Record (Ex Date)",
                      "Market Cap",
                      "Enterprise Value",
                      "Last Price",
                      "Price Target (YTD Ago)",
                      "Total Return (YTD)",
                      "Price Target",
                      "Price Target - Low",
                      "Price Target - Median",
                      "Price Target - High",
                      "Price Target - #",
                      "P/E (NTM)",
                      "P/E (LTM)",
                      "Altman Z-Score (FY)",
                      "Altman Z-Score (FQ)",
                      "Altman Z-Score (LTM)",
                      "Beta (1Y)",
                      "Beta (2Y)",
                      "Beta (5Y)",
                      "Analyst Rating",
                      "# Strong Sell Ratings",
                      "# Strong Buys Ratings",
                      "# Hold Ratings",
                      "# Buys Ratings",
                      "# Sell Ratings",
                      "Total Revenues/CAGR (5Y FY)",
                      "Total Revenues (FQ)",
                      "Total Revenues (-1FY)",
                      "Total Revenues (FY)",
                      "Total Revenues (LTM)",
                      "Total Operating Expenses (LTM)",
                      "P/TBV (LTM)",
                      "TBV (FY)",
                      "TBV (LTM)",
                      "Market Cap (Country R)",
                      "Tot. Return %/CAGR (3Y)",
                      "Tot. Return %/CAGR (10Y)",
                      "Total Return (5Y)",
                      "Total Return (10Y)",
                      "Net Income/Adj. (-1FY)",
                      "CFF (LTM)",
                      "CFI (LTM)",
                      "FCF (LTM)",
                      "CFO (LTM)",
                      "EBITDA (FQ)",
                      "EBITDA (LTM)",
                      "EBITDA (FY)",
                      "EBITDA (-1FY)",
                      "EBITDA/Adj. (LTM)",
                      "EBITDA/Adj. (FY)",
                      "EBITDA/Adj. (-1FY)",
                      "EBIT (FQ)",
                      "EBIT (LTM)",
                      "EBIT (FY)",
                      "EBIT (-1FY)",
                      "EBIT/Adj. (-1FY)",
                      "EBIT/Adj. (FY)",
                      "EBIT/Adj. (LTM)",
                      "EBIT - Est Med (FY1E)",
                      "EBIT - Est Med (NTM)",
                      "Return On Equity % (LTM)",
                      "Return On Equity % (FY)",
                      "Net Income - (IS) (FY)",
                      "Net Income - (IS) (LTM)",
                      "Normalized Net Income (FY)",
                      "Normalized Net Income (LTM)",
                      "Net Income/Adj. (FY)",
                      "Net Income/Adj. (LTM)",
                      "Net Income Margin % (FY)",
                      "Net Income Margin % (LTM)",
                      "Volatility (1M)",
                      "Volatility (3M)",
                      "Volatility (6M)",
                      "Volatility (1Y)",
                      "Volume (Shrs)",
                      "Dividend Per Share (LTM)",
                      "Div Yield (Ind)",
                      "Div Yield (LTM)",
                      "Total Debt (FY)",
                      "Total Equity (FY)",
                      "Total Equity (LTM)",
                      "Total Debt (LTM)",
                      "Total Assets (LTM)",
                      "Total Assets (FY)",
                      "Current Ratio (FY)",
                      "Current Ratio (LTM)",
                      "Gross Profit Margin % (FY)",
                      "Gross Profit Margin % (LTM)",
                      "Asset Turnover (FY)",
                      "Asset Turnover (LTM)",
                      "Gross Profit (LTM)",
                      "Gross Profit (FY)",
                      "EPS Norm - Est Avg (NTM)",
                      "EPS/Adj. (-1FY)",
                      "EPS/Adj. (FY)",
                      "EPS/Adj. (LTM)",
                      "EPS Norm - Est Avg (FY1E)",
                      "Gain (Loss) On Sale Of Assets (LTM)",
                      "Cost Of Revenues (LTM)",
                      "Cash Acquisitions (LTM)",
                      "Cash Acquisitions (FY)",
                      "Cash Acquisitions (-1FY)",
                      "Inventory (LTM)",
                      "Goodwill (FQ)",
                      "Goodwill (LTM)",
                      "Goodwill (FY)",
                      "Goodwill (-1FY)",
                      "Impairment of Goodwill (FQ)",
                      "Impairment of Goodwill (LTM)",
                      "Impairment of Goodwill (-1FY)",
                      "Impairment of Goodwill (FY)",
                      "Operating Income (LTM)",
                      "Asset Writedown (LTM)",
                      "Asset Writedown (FY)",
                      "Asset Writedown (-1FY)",
                      "Operating Income (FY)",
                      "Capital Expenditure (LTM)",
                      "Capital Expenditure (-1FY)",
                      "Capital Expenditure (FY)",
                      "Retained Earnings (LTM)",
                      "Total Current Assets (LTM)",
                      "Total Current Liabilities (LTM)",
                      "R&D Expenses (LTM)",
                      "Restructuring Charges (LTM)",
                      "Restructuring Charges (FQ)",
                      "Restructuring Charges (-1FY)",
                      "Restructuring Charges (FY)",
                      "Interest Expense/Total (LTM)",
                      "Merger & Restructuring Charges (LTM)",
                      "Working Capital (LTM)",
                      "Other Unusual Items/Total (LTM)",
                      "Interest Income On Investments (LTM)",
                      "Buyback Yield (LTM)",
                      "Return on Assets (ROA) % (LTM)",
                      "Return on Assets (ROA) % (FY)",
                      "Net Income - (IS) (-1FY)",
                      "Normalized Net Income (-1FY)",
                      "CFF (FY)",
                      "CFF (-1FY)",
                      "CFI (FY)",
                      "CFI (-1FY)",
                      "CFO (FY)",
                      "CFO (-1FY)",
                      "Div Yield (-1FYInd)",
                      "FCF (FY)",
                      "Capital Expenditure (FQ)",
                      "Capital Expenditure (5YAVGFQ)",
                      "CFF (FQ)",
                      "CFI (FQ)",
                      "CFO (FQ)",
                      "FCF (FQ)",
                      "Total Revenues (5YAVGFQ)",
                      "EBITDA (5YAVGFQ)",
                      "EBIT (5YAVGFQ)",
                      "FCF (5YAVGFQ)",
                      "Cash Acquisitions (FQ)",
                      "Cash Acquisitions (5YAVGFQ)",
                      "Asset Writedown (FQ)",
                      "Asset Writedown (5YAVGFQ)",
                      "Impairment of Goodwill (5YAVGFQ)",
                      "Operating Income (FQ)",
                      "Operating Income (5YAVGFQ)",
                      "P/B (LTM)",
                      "P/B (-1FY)",
                      "P/B (5YAVG)",
                      "Cash And Equivalents (LTM)",
                      "Cash And Equivalents (FQ)",
                      "Cash And Equivalents (FY)",
                      "Cash And Equivalents (5YAVGFQ)",
                      "Inventory (FQ)",
                      "Inventory (FY)",
                      "Goodwill (5YAVGFQ)",
                      "Inventory (5YAVGFQ)",
                      "Retained Earnings (FQ)",
                      "Retained Earnings (FY)",
                      "Retained Earnings (5YAVGFQ)",
                      "Working Capital (FQ)",
                      "Working Capital (FY)",
                      "Working Capital (5YAVGFY)",
                      "Div Yield (TTM)",
                      "Div Yield (NTM)",
                      "Div Yield (5YAVGLTM)",
                      "Gross Intangible Assets (LTM)",
                      "Gross Intangible Assets (FY)",
                      "Gross Intangible Assets (5YAVGFQ)",
                      "Restructuring Charges (5YAVGFQ)",
                      "Merger & Restructuring Charges (FQ)",
                      "Merger & Restructuring Charges (FY)",
                      "Merger & Restructuring Charges (5YAVGFQ)",
                      "Normalized Net Income (FQ)",
                      "Normalized Net Income (5YAVGFQ)",
                      "Net Income/Adj. (FQ)",
                      "Net Income/Adj. (5YAVGFQ)",
                      "Net Income - (IS) (FQ)",
                      "Net Income - (IS) (5YAVGFQ)",
                      "Net Income - (IS) (5YAVGLTM)",
                      "Normalized Net Income (5YAVGLTM)",
                      "EBITDA (5YAVGLTM)",
                      "EBIT (5YAVGLTM)",
                      "Total Revenues (5YAVGLTM)",
                      "Revenues - Est YoY % (FY1E)",
                      "Price Chg. % (1M)",
                      "Price Chg. % (3M)",
                      "1-Day %",
                      "Price (5D Ago)",
                      "Price (1W Ago)",
                      "Price (1M Ago)",
                      "Price (3M Ago)",
                      "Price (6M Ago)",
                      "Price (1Y Ago)",
                      "Price (3Y Ago)",
                      "Price (5Y Ago)",
                      "Price (QTD Ago)",
                      "Rel. Volume",
                      "Shrs Out",
                      "Shrs Out (-1FY)",
                      "Common Dividends Paid (LTM)",
                      "Common Dividends Paid (FY)",
                      "Selling General & Admin Expenses/Total (FQ)",
                      "Selling General & Admin Expenses/Total (FY)",
                      "Selling General & Admin Expenses/Total (-1FY)",
                      "Selling General & Admin Expenses/Total (5YAVGFQ)",
                      "Accounts Receivable/Total (FY)",
                      "Accounts Receivable/Total (-1FY)",
                      "Accounts Receivable/Total (5YAVGFQ)",
                      "Marketing Expenses (FQ)",
                      "Marketing Expenses (FY)",
                      "Marketing Expenses (-1FY)",
                      "Marketing Expenses (5YAVGLTM)",
                      "Revenues - Est Avg (NTM)",
                      "Revenues - Est Avg (FY1E)",
                      "Revenues - Est Med (NTM)",
                      "Revenues - Est Med (FY1E)",
                      "EV/Sales (EST FY1)",
                      "EV/Sales (LTM)",
                      "EV/Sales (NTM)",
                      "EV/Sales (-1FYLTM)",
                      "EV/Sales (-2FYLTM)",
                      "EV/Sales (-3FYLTM)",
                      "EV/Sales (3YAVGLTM)",
                      "EV/Sales (-1FQLTM)",
                      "EV/Sales (-2FQLTM)",
                      "EV/Sales (-3FQLTM)",
                      "EV/Sales (-4FQLTM)",
                      "52W High/Adj",
                      "52W Low/Adj",
                      "EMA (20D)",
                      "EMA (50D)",
                      "EMA (100D)",
                      "EMA (250D)",
                      "EV/EBITDA (LTM)",
                      "EV/EBITDA (NTM)",
                      "EV/EBITDA (-1FYLTM)",
                      "EV/EBITDA (-1FQLTM)",
                      "EV/EBITDA (3YAVGLTM)",
                      "EV/EBITDA (EST FY1)",
                      "P/E (EST FY1)",
                      "P/E (-1FYLTM)",
                      "P/E (-2FYLTM)",
                      "P/E (-3FYLTM)",
                      "P/E (3YAVGLTM)",
                      "P/E (-1FQLTM)",
                      "P/E (-2FQLTM)",
                      "P/E (-3FQLTM)",
                      "P/E (5YAVGLTM)",
                      "P/E (-0FQQoQLTM)",
                      "P/E (-0FYYoYLTM)",
                      "P/E (-1FYYoYLTM)",
                      "P/E (-0FQYoYLTM)",
                      "Full Time Employees (FQ)",
                      "Full Time Employees (FY)",
                      "Full Time Employees (-1FY)",
                      "Full Time Employees (-2FY)",
                      "Full Time Employees (-3FY)",
                      "Avg Employees (5YAVGFY)",
                      "Net EPS - Basic (LTM)",
                      "Net EPS - Basic (FQ)",
                      "Net EPS - Basic (FY)",
                      "Net EPS - Basic (-1FQFQ)",
                      "Net EPS - Basic (-2FQFQ)",
                      "Net EPS - Basic (-3FQFQ)",
                      "Net EPS - Basic (-4FQFQ)",
                      "Net EPS - Basic (-1FY)",
                      "Net EPS - Basic (-2FY)",
                      "Net EPS - Basic (-3FY)",
                      "Net EPS - Basic (-4FY)",
                      "Net EPS - Basic (-5FY)",
                      "EPS Est Avg Rev % (FY1E - 1W)",
                      "EPS Est Avg Rev % (FY1E - 1M)",
                      "EPS Est Avg Rev % (FY1E - 3M)",
                      "EPS Est Avg Rev % (FY1E - 6M)",
                      "EPS Est Avg Rev % (FY1E - 1Y)",
                      "Div Yield (-2FYInd)",
                      "Div Yield (-3FYInd)",
                      "Div Yield (-4FYInd)",
                      "Div Yield (-5FYInd)",
                      "EBITDA - Est Avg (NTM)",
                      "EBITDA - Est Avg (FY1E)",
                      "EPS GAAP - Est Avg (NTM)",
                      "EPS GAAP - Est Avg (FY1E)",
                      "EPS GAAP Est Avg Rev % (FY1E - 1M)",
                      "EPS GAAP Est Avg Rev % (FY1E - 3M)",
                      "EPS GAAP Est Avg Rev % (FY1E - 6M)",
                      "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
                      "EPS Norm - Est # (FY1E)")
SELECT NULLIF(TRIM("Ticker"), ''),                                                    -- "Ticker"
       NULLIF(TRIM("ISIN"), ''),                                                      -- "ISIN"
       NULLIF(TRIM("Name"), ''),                                                      -- "Name"
       NULLIF(TRIM("Description"), ''),                                               -- "Description"
       NULLIF(TRIM("Region"), ''),                                                    -- "Region"
       NULLIF(TRIM("Country"), ''),                                                   -- "Country"
       NULLIF(TRIM("Trading Country"), ''),                                           -- "Trading Country"
       NULLIF(TRIM("Exchange"), ''),                                                  -- "Exchange"
       NULLIF(TRIM("Unit"), ''),                                                      -- "Unit"
       NULLIF(TRIM("Sector"), ''),                                                    -- "Sector"
       NULLIF(TRIM("Industry"), ''),                                                  -- "Industry"
       NULLIF(TRIM("Style Class"), ''),                                               -- "Style Class"
       NULLIF(TRIM("Size Class"), ''),                                                -- "Size Class"
       NULLIF(TRIM("Last Updated"), '')::DATE,                                        -- "Last Updated"
       NULLIF(TRIM("Income Statement Report Date"), '')::DATE,                        -- "Income Statement Report Date"
       NULLIF(TRIM("FY End"), ''),                                                    -- "FY End"
       NULLIF(TRIM("Next Earnings"), '')::DATE,                                       -- "Next Earnings"
       NULLIF(TRIM("Next Earnings (When)"), ''),                                      -- "Next Earnings (When)"
       NULLIF(TRIM("Next Earnings (Status)"), ''),                                    -- "Next Earnings (Status)"
       NULLIF(TRIM("Dividend Record (Currency)"), ''),                                -- "Dividend Record (Currency)"
       safe_to_numeric("Dividend Record (Amount)"),                                   -- "Dividend Record (Amount)"
       NULLIF(TRIM("Dividend Record (Frequency)"), ''),                               -- "Dividend Record (Frequency)"
       safe_to_numeric("Dividend Streak"),                                            -- "Dividend Streak"
       NULLIF(TRIM("Dividend Record (Announce Date)"), '')::DATE,                     -- "Dividend Record (Announce Date)"
       NULLIF(TRIM("Dividend Record (Payable Date)"), '')::DATE,                      -- "Dividend Record (Payable Date)"
       NULLIF(TRIM("Dividend Record (Record Date)"), '')::DATE,                       -- "Dividend Record (Record Date)"
       NULLIF(TRIM("Dividend Record (Ex Date)"), '')::DATE,                           -- "Dividend Record (Ex Date)"
       safe_to_numeric("Market Cap"),                                                 -- "Market Cap"
       safe_to_numeric("Enterprise Value"),                                           -- "Enterprise Value"
       safe_to_numeric("Last Price"),                                                 -- "Last Price"
       safe_to_numeric("Price Target (YTD Ago)"),                                     -- "Price Target (YTD Ago)"
       safe_to_numeric("Total Return (YTD)"),                                         -- "Total Return (YTD)"
       safe_to_numeric("Price Target"),                                               -- "Price Target"
       safe_to_numeric("Price Target - Low"),                                         -- "Price Target - Low"
       safe_to_numeric("Price Target - Median"),                                      -- "Price Target - Median"
       safe_to_numeric("Price Target - High"),                                        -- "Price Target - High"
       safe_to_numeric("Price Target - #"),                                           -- "Price Target - #"
       safe_to_numeric("P/E (NTM)"),                                                  -- "P/E (NTM)"
       safe_to_numeric("P/E (LTM)"),                                                  -- "P/E (LTM)"
       safe_to_numeric("Altman Z-Score (FY)"),                                        -- "Altman Z-Score (FY)"
       safe_to_numeric("Altman Z-Score (FQ)"),                                        -- "Altman Z-Score (FQ)"
       safe_to_numeric("Altman Z-Score (LTM)"),                                       -- "Altman Z-Score (LTM)"
       safe_to_numeric("Beta (1Y)"),                                                  -- "Beta (1Y)"
       safe_to_numeric("Beta (2Y)"),                                                  -- "Beta (2Y)"
       safe_to_numeric("Beta (5Y)"),                                                  -- "Beta (5Y)"
       safe_to_numeric("Analyst Rating"),                                             -- "Analyst Rating"
       safe_to_numeric("# Strong Sell Ratings"),                                      -- "# Strong Sell Ratings"
       safe_to_numeric("# Strong Buys Ratings"),                                      -- "# Strong Buys Ratings"
       safe_to_numeric("# Hold Ratings"),                                             -- "# Hold Ratings"
       safe_to_numeric("# Buys Ratings"),                                             -- "# Buys Ratings"
       safe_to_numeric("# Sell Ratings"),                                             -- "# Sell Ratings"
       safe_to_numeric("Total Revenues/CAGR (5Y FY)"),                                -- "Total Revenues/CAGR (5Y FY)"
       safe_to_numeric("Total Revenues (FQ)"),                                        -- "Total Revenues (FQ)"
       safe_to_numeric("Total Revenues (-1FY)"),                                      -- "Total Revenues (-1FY)"
       safe_to_numeric("Total Revenues (FY)"),                                        -- "Total Revenues (FY)"
       safe_to_numeric("Total Revenues (LTM)"),                                       -- "Total Revenues (LTM)"
       safe_to_numeric("Total Operating Expenses (LTM)"),                             -- "Total Operating Expenses (LTM)"
       safe_to_numeric("P/TBV (LTM)"),                                                -- "P/TBV (LTM)"
       safe_to_numeric("TBV (FY)"),                                                   -- "TBV (FY)"
       safe_to_numeric("TBV (LTM)"),                                                  -- "TBV (LTM)"
       safe_to_numeric("Market Cap (Country R)"),                                     -- "Market Cap (Country R)"
       safe_to_numeric("Tot. Return %/CAGR (3Y)"),                                    -- "Tot. Return %/CAGR (3Y)"
       safe_to_numeric("Tot. Return %/CAGR (10Y)"),                                   -- "Tot. Return %/CAGR (10Y)"
       safe_to_numeric("Total Return (5Y)"),                                          -- "Total Return (5Y)"
       safe_to_numeric("Total Return (10Y)"),                                         -- "Total Return (10Y)"
       safe_to_numeric("Net Income/Adj. (-1FY)"),                                     -- "Net Income/Adj. (-1FY)"
       safe_to_numeric("CFF (LTM)"),                                                  -- "CFF (LTM)"
       safe_to_numeric("CFI (LTM)"),                                                  -- "CFI (LTM)"
       safe_to_numeric("FCF (LTM)"),                                                  -- "FCF (LTM)"
       safe_to_numeric("CFO (LTM)"),                                                  -- "CFO (LTM)"
       safe_to_numeric("EBITDA (FQ)"),                                                -- "EBITDA (FQ)"
       safe_to_numeric("EBITDA (LTM)"),                                               -- "EBITDA (LTM)"
       safe_to_numeric("EBITDA (FY)"),                                                -- "EBITDA (FY)"
       safe_to_numeric("EBITDA (-1FY)"),                                              -- "EBITDA (-1FY)"
       safe_to_numeric("EBITDA/Adj. (LTM)"),                                          -- "EBITDA/Adj. (LTM)"
       safe_to_numeric("EBITDA/Adj. (FY)"),                                           -- "EBITDA/Adj. (FY)"
       safe_to_numeric("EBITDA/Adj. (-1FY)"),                                         -- "EBITDA/Adj. (-1FY)"
       safe_to_numeric("EBIT (FQ)"),                                                  -- "EBIT (FQ)"
       safe_to_numeric("EBIT (LTM)"),                                                 -- "EBIT (LTM)"
       safe_to_numeric("EBIT (FY)"),                                                  -- "EBIT (FY)"
       safe_to_numeric("EBIT (-1FY)"),                                                -- "EBIT (-1FY)"
       safe_to_numeric("EBIT/Adj. (-1FY)"),                                           -- "EBIT/Adj. (-1FY)"
       safe_to_numeric("EBIT/Adj. (FY)"),                                             -- "EBIT/Adj. (FY)"
       safe_to_numeric("EBIT/Adj. (LTM)"),                                            -- "EBIT/Adj. (LTM)"
       safe_to_numeric("EBIT - Est Med (FY1E)"),                                      -- "EBIT - Est Med (FY1E)"
       safe_to_numeric("EBIT - Est Med (NTM)"),                                       -- "EBIT - Est Med (NTM)"
       safe_to_numeric("Return On Equity % (LTM)"),                                   -- "Return On Equity % (LTM)"
       safe_to_numeric("Return On Equity % (FY)"),                                    -- "Return On Equity % (FY)"
       safe_to_numeric("Net Income - (IS) (FY)"),                                     -- "Net Income - (IS) (FY)"
       safe_to_numeric("Net Income - (IS) (LTM)"),                                    -- "Net Income - (IS) (LTM)"
       safe_to_numeric("Normalized Net Income (FY)"),                                 -- "Normalized Net Income (FY)"
       safe_to_numeric("Normalized Net Income (LTM)"),                                -- "Normalized Net Income (LTM)"
       safe_to_numeric("Net Income/Adj. (FY)"),                                       -- "Net Income/Adj. (FY)"
       safe_to_numeric("Net Income/Adj. (LTM)"),                                      -- "Net Income/Adj. (LTM)"
       safe_to_numeric("Net Income Margin % (FY)"),                                   -- "Net Income Margin % (FY)"
       safe_to_numeric("Net Income Margin % (LTM)"),                                  -- "Net Income Margin % (LTM)"
       safe_to_numeric("Volatility (1M)"),                                            -- "Volatility (1M)"
       safe_to_numeric("Volatility (3M)"),                                            -- "Volatility (3M)"
       safe_to_numeric("Volatility (6M)"),                                            -- "Volatility (6M)"
       safe_to_numeric("Volatility (1Y)"),                                            -- "Volatility (1Y)"
       safe_to_numeric("Volume (Shrs)"),                                              -- "Volume (Shrs)"
       safe_to_numeric("Dividend Per Share (LTM)"),                                   -- "Dividend Per Share (LTM)"
       safe_to_numeric("Div Yield (Ind)"),                                            -- "Div Yield (Ind)"
       safe_to_numeric("Div Yield (LTM)"),                                            -- "Div Yield (LTM)"
       safe_to_numeric("Total Debt (FY)"),                                            -- "Total Debt (FY)"
       safe_to_numeric("Total Equity (FY)"),                                          -- "Total Equity (FY)"
       safe_to_numeric("Total Equity (LTM)"),                                         -- "Total Equity (LTM)"
       safe_to_numeric("Total Debt (LTM)"),                                           -- "Total Debt (LTM)"
       safe_to_numeric("Total Assets (LTM)"),                                         -- "Total Assets (LTM)"
       safe_to_numeric("Total Assets (FY)"),                                          -- "Total Assets (FY)"
       safe_to_numeric("Current Ratio (FY)"),                                         -- "Current Ratio (FY)"
       safe_to_numeric("Current Ratio (LTM)"),                                        -- "Current Ratio (LTM)"
       safe_to_numeric("Gross Profit Margin % (FY)"),                                 -- "Gross Profit Margin % (FY)"
       safe_to_numeric("Gross Profit Margin % (LTM)"),                                -- "Gross Profit Margin % (LTM)"
       safe_to_numeric("Asset Turnover (FY)"),                                        -- "Asset Turnover (FY)"
       safe_to_numeric("Asset Turnover (LTM)"),                                       -- "Asset Turnover (LTM)"
       safe_to_numeric("Gross Profit (LTM)"),                                         -- "Gross Profit (LTM)"
       safe_to_numeric("Gross Profit (FY)"),                                          -- "Gross Profit (FY)"
       safe_to_numeric("EPS Norm - Est Avg (NTM)"),                                   -- "EPS Norm - Est Avg (NTM)"
       safe_to_numeric("EPS/Adj. (-1FY)"),                                            -- "EPS/Adj. (-1FY)"
       safe_to_numeric("EPS/Adj. (FY)"),                                              -- "EPS/Adj. (FY)"
       safe_to_numeric("EPS/Adj. (LTM)"),                                             -- "EPS/Adj. (LTM)"
       safe_to_numeric("EPS Norm - Est Avg (FY1E)"),                                  -- "EPS Norm - Est Avg (FY1E)"
       safe_to_numeric("Gain (Loss) On Sale Of Assets (LTM)"),                        -- "Gain (Loss) On Sale Of Assets (LTM)"
       safe_to_numeric("Cost Of Revenues (LTM)"),                                     -- "Cost Of Revenues (LTM)"
       safe_to_numeric("Cash Acquisitions (LTM)"),                                    -- "Cash Acquisitions (LTM)"
       safe_to_numeric("Cash Acquisitions (FY)"),                                     -- "Cash Acquisitions (FY)"
       safe_to_numeric("Cash Acquisitions (-1FY)"),                                   -- "Cash Acquisitions (-1FY)"
       safe_to_numeric("Inventory (LTM)"),                                            -- "Inventory (LTM)"
       safe_to_numeric("Goodwill (FQ)"),                                              -- "Goodwill (FQ)"
       safe_to_numeric("Goodwill (LTM)"),                                             -- "Goodwill (LTM)"
       safe_to_numeric("Goodwill (FY)"),                                              -- "Goodwill (FY)"
       safe_to_numeric("Goodwill (-1FY)"),                                            -- "Goodwill (-1FY)"
       safe_to_numeric("Impairment of Goodwill (FQ)"),                                -- "Impairment of Goodwill (FQ)"
       safe_to_numeric("Impairment of Goodwill (LTM)"),                               -- "Impairment of Goodwill (LTM)"
       safe_to_numeric("Impairment of Goodwill (-1FY)"),                              -- "Impairment of Goodwill (-1FY)"
       safe_to_numeric("Impairment of Goodwill (FY)"),                                -- "Impairment of Goodwill (FY)"
       safe_to_numeric("Operating Income (LTM)"),                                     -- "Operating Income (LTM)"
       safe_to_numeric("Asset Writedown (LTM)"),                                      -- "Asset Writedown (LTM)"
       safe_to_numeric("Asset Writedown (FY)"),                                       -- "Asset Writedown (FY)"
       safe_to_numeric("Asset Writedown (-1FY)"),                                     -- "Asset Writedown (-1FY)"
       safe_to_numeric("Operating Income (FY)"),                                      -- "Operating Income (FY)"
       safe_to_numeric("Capital Expenditure (LTM)"),                                  -- "Capital Expenditure (LTM)"
       safe_to_numeric("Capital Expenditure (-1FY)"),                                 -- "Capital Expenditure (-1FY)"
       safe_to_numeric("Capital Expenditure (FY)"),                                   -- "Capital Expenditure (FY)"
       safe_to_numeric("Retained Earnings (LTM)"),                                    -- "Retained Earnings (LTM)"
       safe_to_numeric("Total Current Assets (LTM)"),                                 -- "Total Current Assets (LTM)"
       safe_to_numeric("Total Current Liabilities (LTM)"),                            -- "Total Current Liabilities (LTM)"
       safe_to_numeric("R&D Expenses (LTM)"),                                         -- "R&D Expenses (LTM)"
       safe_to_numeric("Restructuring Charges (LTM)"),                                -- "Restructuring Charges (LTM)"
       safe_to_numeric("Restructuring Charges (FQ)"),                                 -- "Restructuring Charges (FQ)"
       safe_to_numeric("Restructuring Charges (-1FY)"),                               -- "Restructuring Charges (-1FY)"
       safe_to_numeric("Restructuring Charges (FY)"),                                 -- "Restructuring Charges (FY)"
       safe_to_numeric("Interest Expense/Total (LTM)"),                               -- "Interest Expense/Total (LTM)"
       safe_to_numeric("Merger & Restructuring Charges (LTM)"),                       -- "Merger & Restructuring Charges (LTM)"
       safe_to_numeric("Working Capital (LTM)"),                                      -- "Working Capital (LTM)"
       safe_to_numeric("Other Unusual Items/Total (LTM)"),                            -- "Other Unusual Items/Total (LTM)"
       safe_to_numeric("Interest Income On Investments (LTM)"),                       -- "Interest Income On Investments (LTM)"
       safe_to_numeric("Buyback Yield (LTM)"),                                        -- "Buyback Yield (LTM)"
       safe_to_numeric("Return on Assets (ROA) % (LTM)"),                             -- "Return on Assets (ROA) % (LTM)"
       safe_to_numeric("Return on Assets (ROA) % (FY)"),                              -- "Return on Assets (ROA) % (FY)"
       safe_to_numeric("Net Income - (IS) (-1FY)"),                                   -- "Net Income - (IS) (-1FY)"
       safe_to_numeric("Normalized Net Income (-1FY)"),                               -- "Normalized Net Income (-1FY)"
       safe_to_numeric("CFF (FY)"),                                                   -- "CFF (FY)"
       safe_to_numeric("CFF (-1FY)"),                                                 -- "CFF (-1FY)"
       safe_to_numeric("CFI (FY)"),                                                   -- "CFI (FY)"
       safe_to_numeric("CFI (-1FY)"),                                                 -- "CFI (-1FY)"
       safe_to_numeric("CFO (FY)"),                                                   -- "CFO (FY)"
       safe_to_numeric("CFO (-1FY)"),                                                 -- "CFO (-1FY)"
       safe_to_numeric("Div Yield (-1FYInd)"),                                        -- "Div Yield (-1FYInd)"
       safe_to_numeric("FCF (FY)"),                                                   -- "FCF (FY)"
       safe_to_numeric("Capital Expenditure (FQ)"),                         -- "Capital Expenditure (FQ)"
       safe_to_numeric("Capital Expenditure (5YAVGFQ)"),                    -- "Capital Expenditure (5YAVGFQ)"
       safe_to_numeric("CFF (FQ)"),                                         -- "CFF (FQ)"
       safe_to_numeric("CFI (FQ)"),                                         -- "CFI (FQ)"
       safe_to_numeric("CFO (FQ)"),                                         -- "CFO (FQ)"
       safe_to_numeric("FCF (FQ)"),                                         -- "FCF (FQ)"
       safe_to_numeric("Total Revenues (5YAVGFQ)"),                         -- "Total Revenues (5YAVGFQ)"
       safe_to_numeric("EBITDA (5YAVGFQ)"),                                 -- "EBITDA (5YAVGFQ)"
       safe_to_numeric("EBIT (5YAVGFQ)"),                                   -- "EBIT (5YAVGFQ)"
       safe_to_numeric("FCF (5YAVGFQ)"),                                    -- "FCF (5YAVGFQ)"
       safe_to_numeric("Cash Acquisitions (FQ)"),                           -- "Cash Acquisitions (FQ)"
       safe_to_numeric("Cash Acquisitions (5YAVGFQ)"),                      -- "Cash Acquisitions (5YAVGFQ)"
       safe_to_numeric("Asset Writedown (FQ)"),                             -- "Asset Writedown (FQ)"
       safe_to_numeric("Asset Writedown (5YAVGFQ)"),                        -- "Asset Writedown (5YAVGFQ)"
       safe_to_numeric("Impairment of Goodwill (5YAVGFQ)"),                 -- "Impairment of Goodwill (5YAVGFQ)"
       safe_to_numeric("Operating Income (FQ)"),                            -- "Operating Income (FQ)"
       safe_to_numeric("Operating Income (5YAVGFQ)"),                       -- "Operating Income (5YAVGFQ)"
       safe_to_numeric("P/B (LTM)"),                                        -- "P/B (LTM)"
       safe_to_numeric("P/B (-1FY)"),                                       -- "P/B (-1FY)"
       safe_to_numeric("P/B (5YAVG)"),                                      -- "P/B (5YAVG)"
       safe_to_numeric("Cash And Equivalents (LTM)"),                       -- "Cash And Equivalents (LTM)"
       safe_to_numeric("Cash And Equivalents (FQ)"),                        -- "Cash And Equivalents (FQ)"
       safe_to_numeric("Cash And Equivalents (FY)"),                        -- "Cash And Equivalents (FY)"
       safe_to_numeric("Cash And Equivalents (5YAVGFQ)"),                   -- "Cash And Equivalents (5YAVGFQ)"
       safe_to_numeric("Inventory (FQ)"),                                   -- "Inventory (FQ)"
       safe_to_numeric("Inventory (FY)"),                                   -- "Inventory (FY)"
       safe_to_numeric("Goodwill (5YAVGFQ)"),                               -- "Goodwill (5YAVGFQ)"
       safe_to_numeric("Inventory (5YAVGFQ)"),                              -- "Inventory (5YAVGFQ)"
       safe_to_numeric("Retained Earnings (FQ)"),                           -- "Retained Earnings (FQ)"
       safe_to_numeric("Retained Earnings (FY)"),                           -- "Retained Earnings (FY)"
       safe_to_numeric("Retained Earnings (5YAVGFQ)"),                      -- "Retained Earnings (5YAVGFQ)"
       safe_to_numeric("Working Capital (FQ)"),                             -- "Working Capital (FQ)"
       safe_to_numeric("Working Capital (FY)"),                             -- "Working Capital (FY)"
       safe_to_numeric("Working Capital (5YAVGFY)"),                        -- "Working Capital (5YAVGFY)"
       safe_to_numeric("Div Yield (TTM)"),                                  -- "Div Yield (TTM)"
       safe_to_numeric("Div Yield (NTM)"),                                  -- "Div Yield (NTM)"
       safe_to_numeric("Div Yield (5YAVGLTM)"),                             -- "Div Yield (5YAVGLTM)"
       safe_to_numeric("Gross Intangible Assets (LTM)"),                    -- "Gross Intangible Assets (LTM)"
       safe_to_numeric("Gross Intangible Assets (FY)"),                     -- "Gross Intangible Assets (FY)"
       safe_to_numeric("Gross Intangible Assets (5YAVGFQ)"),                -- "Gross Intangible Assets (5YAVGFQ)"
       safe_to_numeric("Restructuring Charges (5YAVGFQ)"),                  -- "Restructuring Charges (5YAVGFQ)"
       safe_to_numeric("Merger & Restructuring Charges (FQ)"),              -- "Merger & Restructuring Charges (FQ)"
       safe_to_numeric("Merger & Restructuring Charges (FY)"),              -- "Merger & Restructuring Charges (FY)"
       safe_to_numeric("Merger & Restructuring Charges (5YAVGFQ)"),         -- "Merger & Restructuring Charges (5YAVGFQ)"
       safe_to_numeric("Normalized Net Income (FQ)"),                       -- "Normalized Net Income (FQ)"
       safe_to_numeric("Normalized Net Income (5YAVGFQ)"),                  -- "Normalized Net Income (5YAVGFQ)"
       safe_to_numeric("Net Income/Adj. (FQ)"),                             -- "Net Income/Adj. (FQ)"
       safe_to_numeric("Net Income/Adj. (5YAVGFQ)"),                        -- "Net Income/Adj. (5YAVGFQ)"
       safe_to_numeric("Net Income - (IS) (FQ)"),                           -- "Net Income - (IS) (FQ)"
       safe_to_numeric("Net Income - (IS) (5YAVGFQ)"),                      -- "Net Income - (IS) (5YAVGFQ)"
       safe_to_numeric("Net Income - (IS) (5YAVGLTM)"),                     -- "Net Income - (IS) (5YAVGLTM)"
       safe_to_numeric("Normalized Net Income (5YAVGLTM)"),                 -- "Normalized Net Income (5YAVGLTM)"
       safe_to_numeric("EBITDA (5YAVGLTM)"),                                -- "EBITDA (5YAVGLTM)"
       safe_to_numeric("EBIT (5YAVGLTM)"),                                  -- "EBIT (5YAVGLTM)"
       safe_to_numeric("Total Revenues (5YAVGLTM)"),                        -- "Total Revenues (5YAVGLTM)"
       safe_to_numeric("Revenues - Est YoY % (FY1E)"),                      -- "Revenues - Est YoY % (FY1E)"
       safe_to_numeric("Price Chg. % (1M)"),                                -- "Price Chg. % (1M)"
       safe_to_numeric("Price Chg. % (3M)"),                                -- "Price Chg. % (3M)"
       safe_to_numeric("1-Day %"),                                          -- "1-Day %"
       safe_to_numeric("Price (5D Ago)"),                                   -- "Price (5D Ago)"
       safe_to_numeric("Price (1W Ago)"),                                   -- "Price (1W Ago)"
       safe_to_numeric("Price (1M Ago)"),                                   -- "Price (1M Ago)"
       safe_to_numeric("Price (3M Ago)"),                                   -- "Price (3M Ago)"
       safe_to_numeric("Price (6M Ago)"),                                   -- "Price (6M Ago)"
       safe_to_numeric("Price (1Y Ago)"),                                   -- "Price (1Y Ago)"
       safe_to_numeric("Price (3Y Ago)"),                                   -- "Price (3Y Ago)"
       safe_to_numeric("Price (5Y Ago)"),                                   -- "Price (5Y Ago)"
       safe_to_numeric("Price (QTD Ago)"),                                  -- "Price (QTD Ago)"
       safe_to_numeric("Rel. Volume"),                                      -- "Rel. Volume"
       safe_to_numeric("Shrs Out"),                                         -- "Shrs Out"
       safe_to_numeric("Shrs Out (-1FY)"),                                  -- "Shrs Out (-1FY)"
       safe_to_numeric("Common Dividends Paid (LTM)"),                      -- "Common Dividends Paid (LTM)"
       safe_to_numeric("Common Dividends Paid (FY)"),                       -- "Common Dividends Paid (FY)"
       safe_to_numeric("Selling General & Admin Expenses/Total (FQ)"),      -- "Selling General & Admin Expenses/Total (FQ)"
       safe_to_numeric("Selling General & Admin Expenses/Total (FY)"),      -- "Selling General & Admin Expenses/Total (FY)"
       safe_to_numeric("Selling General & Admin Expenses/Total (-1FY)"),    -- "Selling General & Admin Expenses/Total (-1FY)"
       safe_to_numeric("Selling General & Admin Expenses/Total (5YAVGFQ)"), -- "Selling General & Admin Expenses/Total (5YAVGFQ)"
       safe_to_numeric("Accounts Receivable/Total (FY)"),                   -- "Accounts Receivable/Total (FY)"
       safe_to_numeric("Accounts Receivable/Total (-1FY)"),                 -- "Accounts Receivable/Total (-1FY)"
       safe_to_numeric("Accounts Receivable/Total (5YAVGFQ)"),              -- "Accounts Receivable/Total (5YAVGFQ)"
       safe_to_numeric("Marketing Expenses (FQ)"),                          -- "Marketing Expenses (FQ)"
       safe_to_numeric("Marketing Expenses (FY)"),                          -- "Marketing Expenses (FY)"
       safe_to_numeric("Marketing Expenses (-1FY)"),                        -- "Marketing Expenses (-1FY)"
       safe_to_numeric("Marketing Expenses (5YAVGLTM)"),                    -- "Marketing Expenses (5YAVGLTM)"
       safe_to_numeric("Revenues - Est Avg (NTM)"),                         -- "Revenues - Est Avg (NTM)"
       safe_to_numeric("Revenues - Est Avg (FY1E)"),                        -- "Revenues - Est Avg (FY1E)"
       safe_to_numeric("Revenues - Est Med (NTM)"),                         -- "Revenues - Est Med (NTM)"
       safe_to_numeric("Revenues - Est Med (FY1E)"),                        -- "Revenues - Est Med (FY1E)"
       safe_to_numeric("EV/Sales (EST FY1)"),                               -- "EV/Sales (EST FY1)"
       safe_to_numeric("EV/Sales (LTM)"),                                   -- "EV/Sales (LTM)"
       safe_to_numeric("EV/Sales (NTM)"),                                   -- "EV/Sales (NTM)"
       safe_to_numeric("EV/Sales (-1FYLTM)"),                               -- "EV/Sales (-1FYLTM)"
       safe_to_numeric("EV/Sales (-2FYLTM)"),                               -- "EV/Sales (-2FYLTM)"
       safe_to_numeric("EV/Sales (-3FYLTM)"),                               -- "EV/Sales (-3FYLTM)"
       safe_to_numeric("EV/Sales (3YAVGLTM)"),                              -- "EV/Sales (3YAVGLTM)"
       safe_to_numeric("EV/Sales (-1FQLTM)"),                               -- "EV/Sales (-1FQLTM)"
       safe_to_numeric("EV/Sales (-2FQLTM)"),                               -- "EV/Sales (-2FQLTM)"
       safe_to_numeric("EV/Sales (-3FQLTM)"),                               -- "EV/Sales (-3FQLTM)"
       safe_to_numeric("EV/Sales (-4FQLTM)"),                               -- "EV/Sales (-4FQLTM)"
       safe_to_numeric("52W High/Adj"),                                     -- "52W High/Adj"
       safe_to_numeric("52W Low/Adj"),                                      -- "52W Low/Adj"
       safe_to_numeric("EMA (20D)"),                                        -- "EMA (20D)"
       safe_to_numeric("EMA (50D)"),                                        -- "EMA (50D)"
       safe_to_numeric("EMA (100D)"),                                       -- "EMA (100D)"
       safe_to_numeric("EMA (250D)"),                                       -- "EMA (250D)"
       safe_to_numeric("EV/EBITDA (LTM)"),                                  -- "EV/EBITDA (LTM)"
       safe_to_numeric("EV/EBITDA (NTM)"),                                  -- "EV/EBITDA (NTM)"
       safe_to_numeric("EV/EBITDA (-1FYLTM)"),                              -- "EV/EBITDA (-1FYLTM)"
       safe_to_numeric("EV/EBITDA (-1FQLTM)"),                              -- "EV/EBITDA (-1FQLTM)"
       safe_to_numeric("EV/EBITDA (3YAVGLTM)"),                             -- "EV/EBITDA (3YAVGLTM)"
       safe_to_numeric("EV/EBITDA (EST FY1)"),                              -- "EV/EBITDA (EST FY1)"
       safe_to_numeric("P/E (EST FY1)"),                                    -- "P/E (EST FY1)"
       safe_to_numeric("P/E (-1FYLTM)"),                                    -- "P/E (-1FYLTM)"
       safe_to_numeric("P/E (-2FYLTM)"),                                    -- "P/E (-2FYLTM)"
       safe_to_numeric("P/E (-3FYLTM)"),                                    -- "P/E (-3FYLTM)"
       safe_to_numeric("P/E (3YAVGLTM)"),                                   -- "P/E (3YAVGLTM)"
       safe_to_numeric("P/E (-1FQLTM)"),                                    -- "P/E (-1FQLTM)"
       safe_to_numeric("P/E (-2FQLTM)"),                                    -- "P/E (-2FQLTM)"
       safe_to_numeric("P/E (-3FQLTM)"),                                    -- "P/E (-3FQLTM)"
       safe_to_numeric("P/E (5YAVGLTM)"),                                   -- "P/E (5YAVGLTM)"
       safe_to_numeric("P/E (-0FQQoQLTM)"),                                 -- "P/E (-0FQQoQLTM)"
       safe_to_numeric("P/E (-0FYYoYLTM)"),                                 -- "P/E (-0FYYoYLTM)"
       safe_to_numeric("P/E (-1FYYoYLTM)"),                                 -- "P/E (-1FYYoYLTM)"
       safe_to_numeric("P/E (-0FQYoYLTM)"),                                 -- "P/E (-0FQYoYLTM)"
       safe_to_numeric("Full Time Employees (FQ)"),                         -- "Full Time Employees (FQ)"
       safe_to_numeric("Full Time Employees (FY)"),                         -- "Full Time Employees (FY)"
       safe_to_numeric("Full Time Employees (-1FY)"),                       -- "Full Time Employees (-1FY)"
       safe_to_numeric("Full Time Employees (-2FY)"),                       -- "Full Time Employees (-2FY)"
       safe_to_numeric("Full Time Employees (-3FY)"),                       -- "Full Time Employees (-3FY)"
       safe_to_numeric("Avg Employees (5YAVGFY)"),                          -- "Avg Employees (5YAVGFY)"
       safe_to_numeric("Net EPS - Basic (LTM)"),                            -- "Net EPS - Basic (LTM)"
       safe_to_numeric("Net EPS - Basic (FQ)"),                             -- "Net EPS - Basic (FQ)"
       safe_to_numeric("Net EPS - Basic (FY)"),                             -- "Net EPS - Basic (FY)"
       safe_to_numeric("Net EPS - Basic (-1FQFQ)"),                         -- "Net EPS - Basic (-1FQFQ)"
       safe_to_numeric("Net EPS - Basic (-2FQFQ)"),                         -- "Net EPS - Basic (-2FQFQ)"
       safe_to_numeric("Net EPS - Basic (-3FQFQ)"),                         -- "Net EPS - Basic (-3FQFQ)"
       safe_to_numeric("Net EPS - Basic (-4FQFQ)"),                         -- "Net EPS - Basic (-4FQFQ)"
       safe_to_numeric("Net EPS - Basic (-1FY)"),                           -- "Net EPS - Basic (-1FY)"
       safe_to_numeric("Net EPS - Basic (-2FY)"),                           -- "Net EPS - Basic (-2FY)"
       safe_to_numeric("Net EPS - Basic (-3FY)"),                           -- "Net EPS - Basic (-3FY)"
       safe_to_numeric("Net EPS - Basic (-4FY)"),                           -- "Net EPS - Basic (-4FY)"
       safe_to_numeric("Net EPS - Basic (-5FY)"),                           -- "Net EPS - Basic (-5FY)"
       safe_to_numeric("EPS Est Avg Rev % (FY1E - 1W)"),                    -- "EPS Est Avg Rev % (FY1E - 1W)"
       safe_to_numeric("EPS Est Avg Rev % (FY1E - 1M)"),                    -- "EPS Est Avg Rev % (FY1E - 1M)"
       safe_to_numeric("EPS Est Avg Rev % (FY1E - 3M)"),                    -- "EPS Est Avg Rev % (FY1E - 3M)"
       safe_to_numeric("EPS Est Avg Rev % (FY1E - 6M)"),                    -- "EPS Est Avg Rev % (FY1E - 6M)"
       safe_to_numeric("EPS Est Avg Rev % (FY1E - 1Y)"),                    -- "EPS Est Avg Rev % (FY1E - 1Y)"
       safe_to_numeric("Div Yield (-2FYInd)"),                              -- "Div Yield (-2FYInd)"
       safe_to_numeric("Div Yield (-3FYInd)"),                              -- "Div Yield (-3FYInd)"
       safe_to_numeric("Div Yield (-4FYInd)"),                              -- "Div Yield (-4FYInd)"
       safe_to_numeric("Div Yield (-5FYInd)"),                              -- "Div Yield (-5FYInd)"
       safe_to_numeric("EBITDA - Est Avg (NTM)"),                           -- "EBITDA - Est Avg (NTM)"
       safe_to_numeric("EBITDA - Est Avg (FY1E)"),                          -- "EBITDA - Est Avg (FY1E)"
       safe_to_numeric("EPS GAAP - Est Avg (NTM)"),                         -- "EPS GAAP - Est Avg (NTM)"
       safe_to_numeric("EPS GAAP - Est Avg (FY1E)"),                        -- "EPS GAAP - Est Avg (FY1E)"
       safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 1M)"),               -- "EPS GAAP Est Avg Rev % (FY1E - 1M)"
       safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 3M)"),               -- "EPS GAAP Est Avg Rev % (FY1E - 3M)"
       safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 6M)"),               -- "EPS GAAP Est Avg Rev % (FY1E - 6M)"
       safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 1Y)"),               -- "EPS GAAP Est Avg Rev % (FY1E - 1Y)"
       safe_to_numeric("EPS Norm - Est # (FY1E)")                           -- "EPS Norm - Est # (FY1E)"
FROM screening_staging
ON CONFLICT DO NOTHING;;

-- ===================================================================
-- FINAL VALIDATION
-- ===================================================================
\echo 'Final validation...'
SELECT 'Total rows in equities:' AS info, COUNT(*) AS count
FROM equities;
SELECT 'Rows by Region:' AS info, "Region", COUNT(*) AS count
FROM equities
GROUP BY "Region"
ORDER BY "Region";
SELECT 'Rows by Sector (top 10):' AS info, "Sector", COUNT(*) AS count
FROM equities
GROUP BY "Sector"
ORDER BY COUNT(*) DESC
LIMIT 10;

-- ===================================================================
-- CLEANUP
-- ===================================================================
DROP TABLE IF EXISTS screening_staging;
DROP FUNCTION IF EXISTS safe_to_numeric(TEXT);

\echo 'Import complete!'
