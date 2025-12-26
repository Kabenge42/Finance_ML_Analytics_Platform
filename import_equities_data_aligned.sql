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
SELECT NULLIF(TRIM("Ticker"), '') -- "Ticker",
           NULLIF(TRIM("ISIN"), '')                                              -- "ISIN",
    NULLIF(TRIM("Name"), '')                                              -- "Name",
    NULLIF(TRIM("Description"), '')                                       -- "Description",
    NULLIF(TRIM("Region"), '')                                            -- "Region",
    NULLIF(TRIM("Country"), '')                                           -- "Country",
    NULLIF(TRIM("Trading Country"), '')                                   -- "Trading Country",
    NULLIF(TRIM("Exchange"), '')                                          -- "Exchange",
    NULLIF(TRIM("Unit"), '')                                              -- "Unit",
    NULLIF(TRIM("Sector"), '')                                            -- "Sector",
    NULLIF(TRIM("Industry"), '')                                          -- "Industry",
    NULLIF(TRIM("Style Class"), '')                                       -- "Style Class",
    NULLIF(TRIM("Size Class"), '')                                        -- "Size Class",
    NULLIF(TRIM("Last Updated"), '')::DATE                                      -- "Last Updated",
    NULLIF(TRIM("Income Statement Report Date"), '')::DATE                      -- "Income Statement Report Date",
    NULLIF(TRIM("FY End"), '')::DATE                                            -- "FY End",
    NULLIF(TRIM("Next Earnings"), '')::DATE                                     -- "Next Earnings",
    NULLIF(TRIM("Next Earnings (When)"), '')::DATE                              -- "Next Earnings (When)",
    NULLIF(TRIM("Next Earnings (Status)"), '')::DATE                            -- "Next Earnings (Status)",
    NULLIF(TRIM("Dividend Record (Currency)"), '')::DATE                        -- "Dividend Record (Currency)",
    NULLIF(TRIM("Dividend Record (Amount)"), '')::DATE                          -- "Dividend Record (Amount)",
    NULLIF(TRIM("Dividend Record (Frequency)"), '')::DATE                       -- "Dividend Record (Frequency)",
    NULLIF(TRIM("Dividend Streak"), '')::DATE                                   -- "Dividend Streak",
    NULLIF(TRIM("Dividend Record (Announce Date)"), '')::DATE                   -- "Dividend Record (Announce Date)",
    NULLIF(TRIM("Dividend Record (Payable Date)"), '')::DATE                    -- "Dividend Record (Payable Date)",
    NULLIF(TRIM("Dividend Record (Record Date)"), '')::DATE                     -- "Dividend Record (Record Date)",
    NULLIF(TRIM("Dividend Record (Ex Date)"), '')::DATE                         -- "Dividend Record (Ex Date)",
    NULLIF(TRIM("Market Cap"), '')::NUMERIC                                        -- "Market Cap",
    NULLIF(TRIM("Enterprise Value"), '')::NUMERIC                                  -- "Enterprise Value",
    NULLIF(TRIM("Last Price"), '')::NUMERIC                                        -- "Last Price",
    NULLIF(TRIM("Price Target (YTD Ago)"), '')::NUMERIC                            -- "Price Target (YTD Ago)",
    NULLIF(TRIM("Total Return (YTD)"), '')::NUMERIC                                -- "Total Return (YTD)",
    NULLIF(TRIM("Price Target"), '')::NUMERIC                                      -- "Price Target",
    NULLIF(TRIM("Price Target - Low"), '')::NUMERIC                                -- "Price Target - Low",
    NULLIF(TRIM("Price Target - Median"), '')::NUMERIC                             -- "Price Target - Median",
    NULLIF(TRIM("Price Target - High"), '')::NUMERIC                               -- "Price Target - High",
    NULLIF(TRIM("Price Target - #"), '')::NUMERIC                                  -- "Price Target - #",
    NULLIF(TRIM("P/E (NTM)"), '')::NUMERIC                                         -- "P/E (NTM)",
    NULLIF(TRIM("P/E (LTM)"), '')::NUMERIC                                         -- "P/E (LTM)",
    NULLIF(TRIM("Altman Z-Score (FY)"), '')::NUMERIC                               -- "Altman Z-Score (FY)",
    NULLIF(TRIM("Altman Z-Score (FQ)"), '')::NUMERIC                               -- "Altman Z-Score (FQ)",
    NULLIF(TRIM("Altman Z-Score (LTM)"), '')::NUMERIC                              -- "Altman Z-Score (LTM)",
    NULLIF(TRIM("Beta (1Y)"), '')::NUMERIC                                         -- "Beta (1Y)",
    NULLIF(TRIM("Beta (2Y)"), '')::NUMERIC                                         -- "Beta (2Y)",
    NULLIF(TRIM("Beta (5Y)"), '')::NUMERIC                                         -- "Beta (5Y)",
    NULLIF(TRIM("Analyst Rating"), '')::NUMERIC                                    -- "Analyst Rating",
    NULLIF(TRIM("# Strong Sell Ratings"), '')::NUMERIC                             -- "# Strong Sell Ratings",
    NULLIF(TRIM("# Strong Buys Ratings"), '')::NUMERIC                             -- "# Strong Buys Ratings",
    NULLIF(TRIM("# Hold Ratings"), '')::NUMERIC                                    -- "# Hold Ratings",
    NULLIF(TRIM("# Buys Ratings"), '')::NUMERIC                                    -- "# Buys Ratings",
    NULLIF(TRIM("# Sell Ratings"), '')::NUMERIC                                    -- "# Sell Ratings",
    NULLIF(TRIM("Total Revenues/CAGR (5Y FY)"), '')::NUMERIC                       -- "Total Revenues/CAGR (5Y FY)",
    NULLIF(TRIM("Total Revenues (FQ)"), '')::NUMERIC                               -- "Total Revenues (FQ)",
    NULLIF(TRIM("Total Revenues (-1FY)"), '')::NUMERIC                             -- "Total Revenues (-1FY)",
    NULLIF(TRIM("Total Revenues (FY)"), '')::NUMERIC                               -- "Total Revenues (FY)",
    NULLIF(TRIM("Total Revenues (LTM)"), '')::NUMERIC                              -- "Total Revenues (LTM)",
    NULLIF(TRIM("Total Operating Expenses (LTM)"), '')::NUMERIC                    -- "Total Operating Expenses (LTM)",
    NULLIF(TRIM("P/TBV (LTM)"), '')::NUMERIC                                       -- "P/TBV (LTM)",
    NULLIF(TRIM("TBV (FY)"), '')::NUMERIC                                          -- "TBV (FY)",
    NULLIF(TRIM("TBV (LTM)"), '')::NUMERIC                                         -- "TBV (LTM)",
    NULLIF(TRIM("Market Cap (Country R)"), '')                            -- "Market Cap (Country R)",
    NULLIF(TRIM("Tot. Return %/CAGR (3Y)"), '')::NUMERIC                           -- "Tot. Return %/CAGR (3Y)",
    NULLIF(TRIM("Tot. Return %/CAGR (10Y)"), '')::NUMERIC                          -- "Tot. Return %/CAGR (10Y)",
    NULLIF(TRIM("Total Return (5Y)"), '')::NUMERIC                                 -- "Total Return (5Y)",
    NULLIF(TRIM("Total Return (10Y)"), '')::NUMERIC                                -- "Total Return (10Y)",
    NULLIF(TRIM("Net Income/Adj. (-1FY)"), '')::NUMERIC                            -- "Net Income/Adj. (-1FY)",
    NULLIF(TRIM("CFF (LTM)"), '')::NUMERIC                                         -- "CFF (LTM)",
    NULLIF(TRIM("CFI (LTM)"), '')::NUMERIC                                         -- "CFI (LTM)",
    NULLIF(TRIM("FCF (LTM)"), '')::NUMERIC                                         -- "FCF (LTM)",
    NULLIF(TRIM("CFO (LTM)"), '')::NUMERIC                                         -- "CFO (LTM)",
    NULLIF(TRIM("EBITDA (FQ)"), '')::NUMERIC                                       -- "EBITDA (FQ)",
    NULLIF(TRIM("EBITDA (LTM)"), '')::NUMERIC                                      -- "EBITDA (LTM)",
    NULLIF(TRIM("EBITDA (FY)"), '')::NUMERIC                                       -- "EBITDA (FY)",
    NULLIF(TRIM("EBITDA (-1FY)"), '')::NUMERIC                                     -- "EBITDA (-1FY)",
    NULLIF(TRIM("EBITDA/Adj. (LTM)"), '')::NUMERIC                                 -- "EBITDA/Adj. (LTM)",
    NULLIF(TRIM("EBITDA/Adj. (FY)"), '')::NUMERIC                                  -- "EBITDA/Adj. (FY)",
    NULLIF(TRIM("EBITDA/Adj. (-1FY)"), '')::NUMERIC                                -- "EBITDA/Adj. (-1FY)",
    NULLIF(TRIM("EBIT (FQ)"), '')::NUMERIC                                         -- "EBIT (FQ)",
    NULLIF(TRIM("EBIT (LTM)"), '')::NUMERIC                                        -- "EBIT (LTM)",
    NULLIF(TRIM("EBIT (FY)"), '')::NUMERIC                                         -- "EBIT (FY)",
    NULLIF(TRIM("EBIT (-1FY)"), '')::NUMERIC                                       -- "EBIT (-1FY)",
    NULLIF(TRIM("EBIT/Adj. (-1FY)"), '')::NUMERIC                                  -- "EBIT/Adj. (-1FY)",
    NULLIF(TRIM("EBIT/Adj. (FY)"), '')::NUMERIC                                    -- "EBIT/Adj. (FY)",
    NULLIF(TRIM("EBIT/Adj. (LTM)"), '')::NUMERIC                                   -- "EBIT/Adj. (LTM)",
    NULLIF(TRIM("EBIT - Est Med (FY1E)"), '')::NUMERIC                             -- "EBIT - Est Med (FY1E)",
    NULLIF(TRIM("EBIT - Est Med (NTM)"), '')::NUMERIC                              -- "EBIT - Est Med (NTM)",
    NULLIF(TRIM("Return On Equity % (LTM)"), '')::NUMERIC                          -- "Return On Equity % (LTM)",
    NULLIF(TRIM("Return On Equity % (FY)"), '')::NUMERIC                           -- "Return On Equity % (FY)",
    NULLIF(TRIM("Net Income - (IS) (FY)"), '')::NUMERIC                            -- "Net Income - (IS) (FY)",
    NULLIF(TRIM("Net Income - (IS) (LTM)"), '')::NUMERIC                           -- "Net Income - (IS) (LTM)",
    NULLIF(TRIM("Normalized Net Income (FY)"), '')::NUMERIC                        -- "Normalized Net Income (FY)",
    NULLIF(TRIM("Normalized Net Income (LTM)"), '')::NUMERIC                       -- "Normalized Net Income (LTM)",
    NULLIF(TRIM("Net Income/Adj. (FY)"), '')::NUMERIC                              -- "Net Income/Adj. (FY)",
    NULLIF(TRIM("Net Income/Adj. (LTM)"), '')::NUMERIC                             -- "Net Income/Adj. (LTM)",
    NULLIF(TRIM("Net Income Margin % (FY)"), '')::NUMERIC                          -- "Net Income Margin % (FY)",
    NULLIF(TRIM("Net Income Margin % (LTM)"), '')::NUMERIC                         -- "Net Income Margin % (LTM)",
    NULLIF(TRIM("Volatility (1M)"), '')::NUMERIC                                   -- "Volatility (1M)",
    NULLIF(TRIM("Volatility (3M)"), '')::NUMERIC                                   -- "Volatility (3M)",
    NULLIF(TRIM("Volatility (6M)"), '')::NUMERIC                                   -- "Volatility (6M)",
    NULLIF(TRIM("Volatility (1Y)"), '')::NUMERIC                                   -- "Volatility (1Y)",
    NULLIF(TRIM("Volume (Shrs)"), '')::NUMERIC                                     -- "Volume (Shrs)",
    NULLIF(TRIM("Dividend Per Share (LTM)"), '')::DATE                          -- "Dividend Per Share (LTM)",
    NULLIF(TRIM("Div Yield (Ind)"), '')::NUMERIC                                   -- "Div Yield (Ind)",
    NULLIF(TRIM("Div Yield (LTM)"), '')::NUMERIC                                   -- "Div Yield (LTM)",
    NULLIF(TRIM("Total Debt (FY)"), '')::NUMERIC                                   -- "Total Debt (FY)",
    NULLIF(TRIM("Total Equity (FY)"), '')::NUMERIC                                 -- "Total Equity (FY)",
    NULLIF(TRIM("Total Equity (LTM)"), '')::NUMERIC                                -- "Total Equity (LTM)",
    NULLIF(TRIM("Total Debt (LTM)"), '')::NUMERIC                                  -- "Total Debt (LTM)",
    NULLIF(TRIM("Total Assets (LTM)"), '')::NUMERIC                                -- "Total Assets (LTM)",
    NULLIF(TRIM("Total Assets (FY)"), '')::NUMERIC                                 -- "Total Assets (FY)",
    NULLIF(TRIM("Current Ratio (FY)"), '')::NUMERIC                                -- "Current Ratio (FY)",
    NULLIF(TRIM("Current Ratio (LTM)"), '')::NUMERIC                               -- "Current Ratio (LTM)",
    NULLIF(TRIM("Gross Profit Margin % (FY)"), '')::NUMERIC                        -- "Gross Profit Margin % (FY)",
    NULLIF(TRIM("Gross Profit Margin % (LTM)"), '')::NUMERIC                       -- "Gross Profit Margin % (LTM)",
    NULLIF(TRIM("Asset Turnover (FY)"), '')::NUMERIC                               -- "Asset Turnover (FY)",
    NULLIF(TRIM("Asset Turnover (LTM)"), '')::NUMERIC                              -- "Asset Turnover (LTM)",
    NULLIF(TRIM("Gross Profit (LTM)"), '')::NUMERIC                                -- "Gross Profit (LTM)",
    NULLIF(TRIM("Gross Profit (FY)"), '')::NUMERIC                                 -- "Gross Profit (FY)",
    NULLIF(TRIM("EPS Norm - Est Avg (NTM)"), '')::NUMERIC                          -- "EPS Norm - Est Avg (NTM)",
    NULLIF(TRIM("EPS/Adj. (-1FY)"), '')::NUMERIC                                   -- "EPS/Adj. (-1FY)",
    NULLIF(TRIM("EPS/Adj. (FY)"), '')::NUMERIC                                     -- "EPS/Adj. (FY)",
    NULLIF(TRIM("EPS/Adj. (LTM)"), '')::NUMERIC                                    -- "EPS/Adj. (LTM)",
    NULLIF(TRIM("EPS Norm - Est Avg (FY1E)"), '')::NUMERIC                         -- "EPS Norm - Est Avg (FY1E)",
    NULLIF(TRIM("Gain (Loss) On Sale Of Assets (LTM)"), '')::NUMERIC               -- "Gain (Loss) On Sale Of Assets (LTM)",
    NULLIF(TRIM("Cost Of Revenues (LTM)"), '')::NUMERIC                            -- "Cost Of Revenues (LTM)",
    NULLIF(TRIM("Cash Acquisitions (LTM)"), '')::NUMERIC                           -- "Cash Acquisitions (LTM)",
    NULLIF(TRIM("Cash Acquisitions (FY)"), '')::NUMERIC                            -- "Cash Acquisitions (FY)",
    NULLIF(TRIM("Cash Acquisitions (-1FY)"), '')::NUMERIC                          -- "Cash Acquisitions (-1FY)",
    NULLIF(TRIM("Inventory (LTM)"), '')::NUMERIC                                   -- "Inventory (LTM)",
    NULLIF(TRIM("Goodwill (FQ)"), '')::NUMERIC                                     -- "Goodwill (FQ)",
    NULLIF(TRIM("Goodwill (LTM)"), '')::NUMERIC                                    -- "Goodwill (LTM)",
    NULLIF(TRIM("Goodwill (FY)"), '')::NUMERIC                                     -- "Goodwill (FY)",
    NULLIF(TRIM("Goodwill (-1FY)"), '')::NUMERIC                                   -- "Goodwill (-1FY)",
    NULLIF(TRIM("Impairment of Goodwill (FQ)"), '')::NUMERIC                       -- "Impairment of Goodwill (FQ)",
    NULLIF(TRIM("Impairment of Goodwill (LTM)"), '')::NUMERIC                      -- "Impairment of Goodwill (LTM)",
    NULLIF(TRIM("Impairment of Goodwill (-1FY)"), '')::NUMERIC                     -- "Impairment of Goodwill (-1FY)",
    NULLIF(TRIM("Impairment of Goodwill (FY)"), '')::NUMERIC                       -- "Impairment of Goodwill (FY)",
    NULLIF(TRIM("Operating Income (LTM)"), '')::NUMERIC                            -- "Operating Income (LTM)",
    NULLIF(TRIM("Asset Writedown (LTM)"), '')::NUMERIC                             -- "Asset Writedown (LTM)",
    NULLIF(TRIM("Asset Writedown (FY)"), '')::NUMERIC                              -- "Asset Writedown (FY)",
    NULLIF(TRIM("Asset Writedown (-1FY)"), '')::NUMERIC                            -- "Asset Writedown (-1FY)",
    NULLIF(TRIM("Operating Income (FY)"), '')::NUMERIC                             -- "Operating Income (FY)",
    NULLIF(TRIM("Capital Expenditure (LTM)"), '')::DATE                         -- "Capital Expenditure (LTM)",
    NULLIF(TRIM("Capital Expenditure (-1FY)"), '')::DATE                        -- "Capital Expenditure (-1FY)",
    NULLIF(TRIM("Capital Expenditure (FY)"), '')::DATE                          -- "Capital Expenditure (FY)",
    NULLIF(TRIM("Retained Earnings (LTM)"), '')::DATE                           -- "Retained Earnings (LTM)",
    NULLIF(TRIM("Total Current Assets (LTM)"), '')::NUMERIC                        -- "Total Current Assets (LTM)",
    NULLIF(TRIM("Total Current Liabilities (LTM)"), '')::NUMERIC                   -- "Total Current Liabilities (LTM)",
    NULLIF(TRIM("R&D Expenses (LTM)"), '')::NUMERIC                                -- "R&D Expenses (LTM)",
    NULLIF(TRIM("Restructuring Charges (LTM)"), '')::NUMERIC                       -- "Restructuring Charges (LTM)",
    NULLIF(TRIM("Restructuring Charges (FQ)"), '')::NUMERIC                        -- "Restructuring Charges (FQ)",
    NULLIF(TRIM("Restructuring Charges (-1FY)"), '')::NUMERIC                      -- "Restructuring Charges (-1FY)",
    NULLIF(TRIM("Restructuring Charges (FY)"), '')::NUMERIC                        -- "Restructuring Charges (FY)",
    NULLIF(TRIM("Interest Expense/Total (LTM)"), '')::NUMERIC                      -- "Interest Expense/Total (LTM)",
    NULLIF(TRIM("Merger & Restructuring Charges (LTM)"), '')::NUMERIC              -- "Merger & Restructuring Charges (LTM)",
    NULLIF(TRIM("Working Capital (LTM)"), '')::NUMERIC                             -- "Working Capital (LTM)",
    NULLIF(TRIM("Other Unusual Items/Total (LTM)"), '')::NUMERIC                   -- "Other Unusual Items/Total (LTM)",
    NULLIF(TRIM("Interest Income On Investments (LTM)"), '')::NUMERIC              -- "Interest Income On Investments (LTM)",
    NULLIF(TRIM("Buyback Yield (LTM)"), '')::NUMERIC                               -- "Buyback Yield (LTM)",
    NULLIF(TRIM("Return on Assets (ROA) % (LTM)"), '')::NUMERIC                    -- "Return on Assets (ROA) % (LTM)",
    NULLIF(TRIM("Return on Assets (ROA) % (FY)"), '')::NUMERIC                     -- "Return on Assets (ROA) % (FY)",
    NULLIF(TRIM("Net Income - (IS) (-1FY)"), '')::NUMERIC                          -- "Net Income - (IS) (-1FY)",
    NULLIF(TRIM("Normalized Net Income (-1FY)"), '')::NUMERIC                      -- "Normalized Net Income (-1FY)",
    NULLIF(TRIM("CFF (FY)"), '')::NUMERIC                                          -- "CFF (FY)",
    NULLIF(TRIM("CFF (-1FY)"), '')::NUMERIC                                        -- "CFF (-1FY)",
    NULLIF(TRIM("CFI (FY)"), '')::NUMERIC                                          -- "CFI (FY)",
    NULLIF(TRIM("CFI (-1FY)"), '')::NUMERIC                                        -- "CFI (-1FY)",
    NULLIF(TRIM("CFO (FY)"), '')::NUMERIC                                          -- "CFO (FY)",
    NULLIF(TRIM("CFO (-1FY)"), '')::NUMERIC                                        -- "CFO (-1FY)",
    NULLIF(TRIM("Div Yield (-1FYInd)"), '')::NUMERIC                               -- "Div Yield (-1FYInd)",
    NULLIF(TRIM("FCF (FY)"), '')::NUMERIC                                          -- "FCF (FY)",
    NULLIF(TRIM("Capital Expenditure (FQ)"), '')::DATE                          -- "Capital Expenditure (FQ)",
    NULLIF(TRIM("Capital Expenditure (5YAVGFQ)"), '')::DATE                     -- "Capital Expenditure (5YAVGFQ)",
    NULLIF(TRIM("CFF (FQ)"), '')::NUMERIC                                          -- "CFF (FQ)",
    NULLIF(TRIM("CFI (FQ)"), '')::NUMERIC                                          -- "CFI (FQ)",
    NULLIF(TRIM("CFO (FQ)"), '')::NUMERIC                                          -- "CFO (FQ)",
    NULLIF(TRIM("FCF (FQ)"), '')::NUMERIC                                          -- "FCF (FQ)",
    NULLIF(TRIM("Total Revenues (5YAVGFQ)"), '')::NUMERIC                          -- "Total Revenues (5YAVGFQ)",
    NULLIF(TRIM("EBITDA (5YAVGFQ)"), '')::NUMERIC                                  -- "EBITDA (5YAVGFQ)",
    NULLIF(TRIM("EBIT (5YAVGFQ)"), '')::NUMERIC                                    -- "EBIT (5YAVGFQ)",
    NULLIF(TRIM("FCF (5YAVGFQ)"), '')::NUMERIC                                     -- "FCF (5YAVGFQ)",
    NULLIF(TRIM("Cash Acquisitions (FQ)"), '')::NUMERIC                            -- "Cash Acquisitions (FQ)",
    NULLIF(TRIM("Cash Acquisitions (5YAVGFQ)"), '')::NUMERIC                       -- "Cash Acquisitions (5YAVGFQ)",
    NULLIF(TRIM("Asset Writedown (FQ)"), '')::NUMERIC                              -- "Asset Writedown (FQ)",
    NULLIF(TRIM("Asset Writedown (5YAVGFQ)"), '')::NUMERIC                         -- "Asset Writedown (5YAVGFQ)",
    NULLIF(TRIM("Impairment of Goodwill (5YAVGFQ)"), '')::NUMERIC                  -- "Impairment of Goodwill (5YAVGFQ)",
    NULLIF(TRIM("Operating Income (FQ)"), '')::NUMERIC                             -- "Operating Income (FQ)",
    NULLIF(TRIM("Operating Income (5YAVGFQ)"), '')::NUMERIC                        -- "Operating Income (5YAVGFQ)",
    NULLIF(TRIM("P/B (LTM)"), '')::NUMERIC                                         -- "P/B (LTM)",
    NULLIF(TRIM("P/B (-1FY)"), '')::NUMERIC                                        -- "P/B (-1FY)",
    NULLIF(TRIM("P/B (5YAVG)"), '')::NUMERIC                                       -- "P/B (5YAVG)",
    NULLIF(TRIM("Cash And Equivalents (LTM)"), '')::NUMERIC                        -- "Cash And Equivalents (LTM)",
    NULLIF(TRIM("Cash And Equivalents (FQ)"), '')::NUMERIC                         -- "Cash And Equivalents (FQ)",
    NULLIF(TRIM("Cash And Equivalents (FY)"), '')::NUMERIC                         -- "Cash And Equivalents (FY)",
    NULLIF(TRIM("Cash And Equivalents (5YAVGFQ)"), '')::NUMERIC                    -- "Cash And Equivalents (5YAVGFQ)",
    NULLIF(TRIM("Inventory (FQ)"), '')::NUMERIC                                    -- "Inventory (FQ)",
    NULLIF(TRIM("Inventory (FY)"), '')::NUMERIC                                    -- "Inventory (FY)",
    NULLIF(TRIM("Goodwill (5YAVGFQ)"), '')::NUMERIC                                -- "Goodwill (5YAVGFQ)",
    NULLIF(TRIM("Inventory (5YAVGFQ)"), '')::NUMERIC                               -- "Inventory (5YAVGFQ)",
    NULLIF(TRIM("Retained Earnings (FQ)"), '')::DATE                            -- "Retained Earnings (FQ)",
    NULLIF(TRIM("Retained Earnings (FY)"), '')::DATE                            -- "Retained Earnings (FY)",
    NULLIF(TRIM("Retained Earnings (5YAVGFQ)"), '')::DATE                       -- "Retained Earnings (5YAVGFQ)",
    NULLIF(TRIM("Working Capital (FQ)"), '')::NUMERIC                              -- "Working Capital (FQ)",
    NULLIF(TRIM("Working Capital (FY)"), '')::NUMERIC                              -- "Working Capital (FY)",
    NULLIF(TRIM("Working Capital (5YAVGFY)"), '')::NUMERIC                         -- "Working Capital (5YAVGFY)",
    NULLIF(TRIM("Div Yield (TTM)"), '')::NUMERIC                                   -- "Div Yield (TTM)",
    NULLIF(TRIM("Div Yield (NTM)"), '')::NUMERIC                                   -- "Div Yield (NTM)",
    NULLIF(TRIM("Div Yield (5YAVGLTM)"), '')::NUMERIC                              -- "Div Yield (5YAVGLTM)",
    NULLIF(TRIM("Gross Intangible Assets (LTM)"), '')::NUMERIC                     -- "Gross Intangible Assets (LTM)",
    NULLIF(TRIM("Gross Intangible Assets (FY)"), '')::NUMERIC                      -- "Gross Intangible Assets (FY)",
    NULLIF(TRIM("Gross Intangible Assets (5YAVGFQ)"), '')::NUMERIC                 -- "Gross Intangible Assets (5YAVGFQ)",
    NULLIF(TRIM("Restructuring Charges (5YAVGFQ)"), '')::NUMERIC                   -- "Restructuring Charges (5YAVGFQ)",
    NULLIF(TRIM("Merger & Restructuring Charges (FQ)"), '')::NUMERIC               -- "Merger & Restructuring Charges (FQ)",
    NULLIF(TRIM("Merger & Restructuring Charges (FY)"), '')::NUMERIC               -- "Merger & Restructuring Charges (FY)",
    NULLIF(TRIM("Merger & Restructuring Charges (5YAVGFQ)"), '')::NUMERIC          -- "Merger & Restructuring Charges (5YAVGFQ)",
    NULLIF(TRIM("Normalized Net Income (FQ)"), '')::NUMERIC                        -- "Normalized Net Income (FQ)",
    NULLIF(TRIM("Normalized Net Income (5YAVGFQ)"), '')::NUMERIC                   -- "Normalized Net Income (5YAVGFQ)",
    NULLIF(TRIM("Net Income/Adj. (FQ)"), '')::NUMERIC                              -- "Net Income/Adj. (FQ)",
    NULLIF(TRIM("Net Income/Adj. (5YAVGFQ)"), '')::NUMERIC                         -- "Net Income/Adj. (5YAVGFQ)",
    NULLIF(TRIM("Net Income - (IS) (FQ)"), '')::NUMERIC                            -- "Net Income - (IS) (FQ)",
    NULLIF(TRIM("Net Income - (IS) (5YAVGFQ)"), '')::NUMERIC                       -- "Net Income - (IS) (5YAVGFQ)",
    NULLIF(TRIM("Net Income - (IS) (5YAVGLTM)"), '')::NUMERIC                      -- "Net Income - (IS) (5YAVGLTM)",
    NULLIF(TRIM("Normalized Net Income (5YAVGLTM)"), '')::NUMERIC                  -- "Normalized Net Income (5YAVGLTM)",
    NULLIF(TRIM("EBITDA (5YAVGLTM)"), '')::NUMERIC                                 -- "EBITDA (5YAVGLTM)",
    NULLIF(TRIM("EBIT (5YAVGLTM)"), '')::NUMERIC                                   -- "EBIT (5YAVGLTM)",
    NULLIF(TRIM("Total Revenues (5YAVGLTM)"), '')::NUMERIC                         -- "Total Revenues (5YAVGLTM)",
    NULLIF(TRIM("Revenues - Est YoY % (FY1E)"), '')::NUMERIC                       -- "Revenues - Est YoY % (FY1E)",
    NULLIF(TRIM("Price Chg. % (1M)"), '')::NUMERIC                                 -- "Price Chg. % (1M)",
    NULLIF(TRIM("Price Chg. % (3M)"), '')::NUMERIC                                 -- "Price Chg. % (3M)",
    NULLIF(TRIM("1-Day %"), '')::NUMERIC                                           -- "1-Day %",
    NULLIF(TRIM("Price (5D Ago)"), '')::NUMERIC                                    -- "Price (5D Ago)",
    NULLIF(TRIM("Price (1W Ago)"), '')::NUMERIC                                    -- "Price (1W Ago)",
    NULLIF(TRIM("Price (1M Ago)"), '')::NUMERIC                                    -- "Price (1M Ago)",
    NULLIF(TRIM("Price (3M Ago)"), '')::NUMERIC                                    -- "Price (3M Ago)",
    NULLIF(TRIM("Price (6M Ago)"), '')::NUMERIC                                    -- "Price (6M Ago)",
    NULLIF(TRIM("Price (1Y Ago)"), '')::NUMERIC                                    -- "Price (1Y Ago)",
    NULLIF(TRIM("Price (3Y Ago)"), '')::NUMERIC                                    -- "Price (3Y Ago)",
    NULLIF(TRIM("Price (5Y Ago)"), '')::NUMERIC                                    -- "Price (5Y Ago)",
    NULLIF(TRIM("Price (QTD Ago)"), '')::NUMERIC                                   -- "Price (QTD Ago)",
    NULLIF(TRIM("Rel. Volume"), '')::NUMERIC                                       -- "Rel. Volume",
    NULLIF(TRIM("Shrs Out"), '')::NUMERIC                                          -- "Shrs Out",
    NULLIF(TRIM("Shrs Out (-1FY)"), '')::NUMERIC                                   -- "Shrs Out (-1FY)",
    NULLIF(TRIM("Common Dividends Paid (LTM)"), '')::DATE                       -- "Common Dividends Paid (LTM)",
    NULLIF(TRIM("Common Dividends Paid (FY)"), '')::DATE                        -- "Common Dividends Paid (FY)",
    NULLIF(TRIM("Selling General & Admin Expenses/Total (FQ)"), '')::NUMERIC       -- "Selling General & Admin Expenses/Total (FQ)",
    NULLIF(TRIM("Selling General & Admin Expenses/Total (FY)"), '')::NUMERIC       -- "Selling General & Admin Expenses/Total (FY)",
    NULLIF(TRIM("Selling General & Admin Expenses/Total (-1FY)"), '')::NUMERIC     -- "Selling General & Admin Expenses/Total (-1FY)",
    NULLIF(TRIM("Selling General & Admin Expenses/Total (5YAVGFQ)"), '')::NUMERIC  -- "Selling General & Admin Expenses/Total (5YAVGFQ)",
    NULLIF(TRIM("Accounts Receivable/Total (FY)"), '')::NUMERIC                    -- "Accounts Receivable/Total (FY)",
    NULLIF(TRIM("Accounts Receivable/Total (-1FY)"), '')::NUMERIC                  -- "Accounts Receivable/Total (-1FY)",
    NULLIF(TRIM("Accounts Receivable/Total (5YAVGFQ)"), '')::NUMERIC               -- "Accounts Receivable/Total (5YAVGFQ)",
    NULLIF(TRIM("Marketing Expenses (FQ)"), '')::NUMERIC                           -- "Marketing Expenses (FQ)",
    NULLIF(TRIM("Marketing Expenses (FY)"), '')::NUMERIC                           -- "Marketing Expenses (FY)",
    NULLIF(TRIM("Marketing Expenses (-1FY)"), '')::NUMERIC                         -- "Marketing Expenses (-1FY)",
    NULLIF(TRIM("Marketing Expenses (5YAVGLTM)"), '')::NUMERIC                     -- "Marketing Expenses (5YAVGLTM)",
    NULLIF(TRIM("Revenues - Est Avg (NTM)"), '')::NUMERIC                          -- "Revenues - Est Avg (NTM)",
    NULLIF(TRIM("Revenues - Est Avg (FY1E)"), '')::NUMERIC                         -- "Revenues - Est Avg (FY1E)",
    NULLIF(TRIM("Revenues - Est Med (NTM)"), '')::NUMERIC                          -- "Revenues - Est Med (NTM)",
    NULLIF(TRIM("Revenues - Est Med (FY1E)"), '')::NUMERIC                         -- "Revenues - Est Med (FY1E)",
    NULLIF(TRIM("EV/Sales (EST FY1)"), '')::NUMERIC                                -- "EV/Sales (EST FY1)",
    NULLIF(TRIM("EV/Sales (LTM)"), '')::NUMERIC                                    -- "EV/Sales (LTM)",
    NULLIF(TRIM("EV/Sales (NTM)"), '')::NUMERIC                                    -- "EV/Sales (NTM)",
    NULLIF(TRIM("EV/Sales (-1FYLTM)"), '')::NUMERIC                                -- "EV/Sales (-1FYLTM)",
    NULLIF(TRIM("EV/Sales (-2FYLTM)"), '')::NUMERIC                                -- "EV/Sales (-2FYLTM)",
    NULLIF(TRIM("EV/Sales (-3FYLTM)"), '')::NUMERIC                                -- "EV/Sales (-3FYLTM)",
    NULLIF(TRIM("EV/Sales (3YAVGLTM)"), '')::NUMERIC                               -- "EV/Sales (3YAVGLTM)",
    NULLIF(TRIM("EV/Sales (-1FQLTM)"), '')::NUMERIC                                -- "EV/Sales (-1FQLTM)",
    NULLIF(TRIM("EV/Sales (-2FQLTM)"), '')::NUMERIC                                -- "EV/Sales (-2FQLTM)",
    NULLIF(TRIM("EV/Sales (-3FQLTM)"), '')::NUMERIC                                -- "EV/Sales (-3FQLTM)",
    NULLIF(TRIM("EV/Sales (-4FQLTM)"), '')::NUMERIC                                -- "EV/Sales (-4FQLTM)",
    NULLIF(TRIM("52W High/Adj"), '')::NUMERIC                                      -- "52W High/Adj",
    NULLIF(TRIM("52W Low/Adj"), '')::NUMERIC                                       -- "52W Low/Adj",
    NULLIF(TRIM("EMA (20D)"), '')::NUMERIC                                         -- "EMA (20D)",
    NULLIF(TRIM("EMA (50D)"), '')::NUMERIC                                         -- "EMA (50D)",
    NULLIF(TRIM("EMA (100D)"), '')::NUMERIC                                        -- "EMA (100D)",
    NULLIF(TRIM("EMA (250D)"), '')::NUMERIC                                        -- "EMA (250D)",
    NULLIF(TRIM("EV/EBITDA (LTM)"), '')::NUMERIC                                   -- "EV/EBITDA (LTM)",
    NULLIF(TRIM("EV/EBITDA (NTM)"), '')::NUMERIC                                   -- "EV/EBITDA (NTM)",
    NULLIF(TRIM("EV/EBITDA (-1FYLTM)"), '')::NUMERIC                               -- "EV/EBITDA (-1FYLTM)",
    NULLIF(TRIM("EV/EBITDA (-1FQLTM)"), '')::NUMERIC                               -- "EV/EBITDA (-1FQLTM)",
    NULLIF(TRIM("EV/EBITDA (3YAVGLTM)"), '')::NUMERIC                              -- "EV/EBITDA (3YAVGLTM)",
    NULLIF(TRIM("EV/EBITDA (EST FY1)"), '')::NUMERIC                               -- "EV/EBITDA (EST FY1)",
    NULLIF(TRIM("P/E (EST FY1)"), '')::NUMERIC                                     -- "P/E (EST FY1)",
    NULLIF(TRIM("P/E (-1FYLTM)"), '')::NUMERIC                                     -- "P/E (-1FYLTM)",
    NULLIF(TRIM("P/E (-2FYLTM)"), '')::NUMERIC                                     -- "P/E (-2FYLTM)",
    NULLIF(TRIM("P/E (-3FYLTM)"), '')::NUMERIC                                     -- "P/E (-3FYLTM)",
    NULLIF(TRIM("P/E (3YAVGLTM)"), '')::NUMERIC                                    -- "P/E (3YAVGLTM)",
    NULLIF(TRIM("P/E (-1FQLTM)"), '')::NUMERIC                                     -- "P/E (-1FQLTM)",
    NULLIF(TRIM("P/E (-2FQLTM)"), '')::NUMERIC                                     -- "P/E (-2FQLTM)",
    NULLIF(TRIM("P/E (-3FQLTM)"), '')::NUMERIC                                     -- "P/E (-3FQLTM)",
    NULLIF(TRIM("P/E (5YAVGLTM)"), '')::NUMERIC                                    -- "P/E (5YAVGLTM)",
    NULLIF(TRIM("P/E (-0FQQoQLTM)"), '')::NUMERIC                                  -- "P/E (-0FQQoQLTM)",
    NULLIF(TRIM("P/E (-0FYYoYLTM)"), '')::NUMERIC                                  -- "P/E (-0FYYoYLTM)",
    NULLIF(TRIM("P/E (-1FYYoYLTM)"), '')::NUMERIC                                  -- "P/E (-1FYYoYLTM)",
    NULLIF(TRIM("P/E (-0FQYoYLTM)"), '')::NUMERIC                                  -- "P/E (-0FQYoYLTM)",
    NULLIF(TRIM("Full Time Employees (FQ)"), '')::NUMERIC                          -- "Full Time Employees (FQ)",
    NULLIF(TRIM("Full Time Employees (FY)"), '')::NUMERIC                          -- "Full Time Employees (FY)",
    NULLIF(TRIM("Full Time Employees (-1FY)"), '')::NUMERIC                        -- "Full Time Employees (-1FY)",
    NULLIF(TRIM("Full Time Employees (-2FY)"), '')::NUMERIC                        -- "Full Time Employees (-2FY)",
    NULLIF(TRIM("Full Time Employees (-3FY)"), '')::NUMERIC                        -- "Full Time Employees (-3FY)",
    NULLIF(TRIM("Avg Employees (5YAVGFY)"), '')::NUMERIC                           -- "Avg Employees (5YAVGFY)",
    NULLIF(TRIM("Net EPS - Basic (LTM)"), '')::NUMERIC                             -- "Net EPS - Basic (LTM)",
    NULLIF(TRIM("Net EPS - Basic (FQ)"), '')::NUMERIC                              -- "Net EPS - Basic (FQ)",
    NULLIF(TRIM("Net EPS - Basic (FY)"), '')::NUMERIC                              -- "Net EPS - Basic (FY)",
    NULLIF(TRIM("Net EPS - Basic (-1FQFQ)"), '')::NUMERIC                          -- "Net EPS - Basic (-1FQFQ)",
    NULLIF(TRIM("Net EPS - Basic (-2FQFQ)"), '')::NUMERIC                          -- "Net EPS - Basic (-2FQFQ)",
    NULLIF(TRIM("Net EPS - Basic (-3FQFQ)"), '')::NUMERIC                          -- "Net EPS - Basic (-3FQFQ)",
    NULLIF(TRIM("Net EPS - Basic (-4FQFQ)"), '')::NUMERIC                          -- "Net EPS - Basic (-4FQFQ)",
    NULLIF(TRIM("Net EPS - Basic (-1FY)"), '')::NUMERIC                            -- "Net EPS - Basic (-1FY)",
    NULLIF(TRIM("Net EPS - Basic (-2FY)"), '')::NUMERIC                            -- "Net EPS - Basic (-2FY)",
    NULLIF(TRIM("Net EPS - Basic (-3FY)"), '')::NUMERIC                            -- "Net EPS - Basic (-3FY)",
    NULLIF(TRIM("Net EPS - Basic (-4FY)"), '')::NUMERIC                            -- "Net EPS - Basic (-4FY)",
    NULLIF(TRIM("Net EPS - Basic (-5FY)"), '')::NUMERIC                            -- "Net EPS - Basic (-5FY)",
    NULLIF(TRIM("EPS Est Avg Rev % (FY1E - 1W)"), '')::NUMERIC                     -- "EPS Est Avg Rev % (FY1E - 1W)",
    NULLIF(TRIM("EPS Est Avg Rev % (FY1E - 1M)"), '')::NUMERIC                     -- "EPS Est Avg Rev % (FY1E - 1M)",
    NULLIF(TRIM("EPS Est Avg Rev % (FY1E - 3M)"), '')::NUMERIC                     -- "EPS Est Avg Rev % (FY1E - 3M)",
    NULLIF(TRIM("EPS Est Avg Rev % (FY1E - 6M)"), '')::NUMERIC                     -- "EPS Est Avg Rev % (FY1E - 6M)",
    NULLIF(TRIM("EPS Est Avg Rev % (FY1E - 1Y)"), '')::NUMERIC                     -- "EPS Est Avg Rev % (FY1E - 1Y)",
    NULLIF(TRIM("Div Yield (-2FYInd)"), '')::NUMERIC                               -- "Div Yield (-2FYInd)",
    NULLIF(TRIM("Div Yield (-3FYInd)"), '')::NUMERIC                               -- "Div Yield (-3FYInd)",
    NULLIF(TRIM("Div Yield (-4FYInd)"), '')::NUMERIC                               -- "Div Yield (-4FYInd)",
    NULLIF(TRIM("Div Yield (-5FYInd)"), '')::NUMERIC                               -- "Div Yield (-5FYInd)",
    NULLIF(TRIM("EBITDA - Est Avg (NTM)"), '')::NUMERIC                            -- "EBITDA - Est Avg (NTM)",
    NULLIF(TRIM("EBITDA - Est Avg (FY1E)"), '')::NUMERIC                           -- "EBITDA - Est Avg (FY1E)",
    NULLIF(TRIM("EPS GAAP - Est Avg (NTM)"), '')::NUMERIC                          -- "EPS GAAP - Est Avg (NTM)",
    NULLIF(TRIM("EPS GAAP - Est Avg (FY1E)"), '')::NUMERIC                         -- "EPS GAAP - Est Avg (FY1E)",
    NULLIF(TRIM("EPS GAAP Est Avg Rev % (FY1E - 1M)"), '')::NUMERIC                -- "EPS GAAP Est Avg Rev % (FY1E - 1M)",
    NULLIF(TRIM("EPS GAAP Est Avg Rev % (FY1E - 3M)"), '')::NUMERIC                -- "EPS GAAP Est Avg Rev % (FY1E - 3M)",
    NULLIF(TRIM("EPS GAAP Est Avg Rev % (FY1E - 6M)"), '')::NUMERIC                -- "EPS GAAP Est Avg Rev % (FY1E - 6M)",
    NULLIF(TRIM("EPS GAAP Est Avg Rev % (FY1E - 1Y)"), '')::NUMERIC                -- "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
    NULLIF(TRIM("EPS Norm - Est # (FY1E)"), '')::NUMERIC                           -- "EPS Norm - Est # (FY1E)"
FROM screening_staging
ON CONFLICT DO NOTHING;

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

\echo 'Import complete!'
