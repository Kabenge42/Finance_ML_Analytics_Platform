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
    RETURNS NUMERIC AS
$$
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
-- HELPER FUNCTION: Parse FY End to Date
-- ===================================================================
-- Converts "FY End" text (e.g., "Dec 2024", "Mar 2025") to a DATE
-- Returns the last day of the specified month/year

CREATE OR REPLACE FUNCTION parse_fy_end_to_date(fy_end_text TEXT)
    RETURNS DATE AS
$$
DECLARE
    month_name TEXT;
    year_text  TEXT;
    month_num  INTEGER;
BEGIN
    IF fy_end_text IS NULL OR TRIM(fy_end_text) = '' THEN
        RETURN NULL;
    END IF;

    -- Handle various input formats
    fy_end_text := TRIM(fy_end_text);

    -- Try "Mon YYYY" format first
    month_name := SPLIT_PART(fy_end_text, ' ', 1);
    year_text := SPLIT_PART(fy_end_text, ' ', 2);

    -- Validate year is numeric and reasonable
    IF year_text !~ '^\d{4}$' OR year_text::INTEGER < 1900 OR year_text::INTEGER > 2100 THEN
        RETURN NULL;
    END IF;

    month_num := CASE UPPER(LEFT(month_name, 3))
                     WHEN 'JAN' THEN 1
                     WHEN 'FEB' THEN 2
                     WHEN 'MAR' THEN 3
                     WHEN 'APR' THEN 4
                     WHEN 'MAY' THEN 5
                     WHEN 'JUN' THEN 6
                     WHEN 'JUL' THEN 7
                     WHEN 'AUG' THEN 8
                     WHEN 'SEP' THEN 9
                     WHEN 'OCT' THEN 10
                     WHEN 'NOV' THEN 11
                     WHEN 'DEC' THEN 12
        END;

    IF month_num IS NULL THEN
        RETURN NULL;
    END IF;

    -- Last day of month using cleaner syntax
    RETURN (MAKE_DATE(year_text::INTEGER, month_num, 1) + INTERVAL '1 month - 1 day')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Calculate Fiscal Info
-- ===================================================================
-- Unified fiscal date calculator that returns all fiscal metrics at once
CREATE OR REPLACE FUNCTION calculate_fiscal_info(
    reference_date DATE,
    fy_end_date DATE,
    OUT fiscal_month INTEGER,
    OUT fiscal_quarter INTEGER,
    OUT fiscal_year INTEGER,
    OUT next_quarter INTEGER,
    OUT next_quarter_year INTEGER
) AS
$$
DECLARE
    months_since_fy_end INTEGER;
BEGIN
    IF reference_date IS NULL OR fy_end_date IS NULL THEN
        RETURN;
    END IF;

    -- Calculate months since FY end using precise date math
    months_since_fy_end := (EXTRACT(YEAR FROM reference_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
        + (EXTRACT(MONTH FROM reference_date) - EXTRACT(MONTH FROM fy_end_date));

    -- Fiscal month (1-12)
    fiscal_month := ((months_since_fy_end - 1) % 12) + 1;
    IF fiscal_month <= 0 THEN
        fiscal_month := fiscal_month + 12;
    END IF;

    -- Current quarter
    fiscal_quarter := CEIL(fiscal_month / 3.0)::INTEGER;

    -- Fiscal year
    fiscal_year := EXTRACT(YEAR FROM fy_end_date)::INTEGER + 1 + ((months_since_fy_end - 1) / 12);

    -- Next quarter
    next_quarter := CASE WHEN fiscal_quarter = 4 THEN 1 ELSE fiscal_quarter + 1 END;
    next_quarter_year := CASE WHEN fiscal_quarter = 4 THEN fiscal_year + 1 ELSE fiscal_year END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Determine Next Earnings Report Type
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_earnings_report(next_fiscal_quarter TEXT)
    RETURNS TEXT AS
$$
BEGIN
    RETURN CASE WHEN next_fiscal_quarter LIKE 'Q4%' THEN 'Full Year' ELSE 'Interim' END;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next FY End Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fy_end_date(fy_end_date DATE)
    RETURNS DATE AS
$$
BEGIN
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;

    -- Handle Feb 29 → Feb 28 transition properly
    -- Adding '1 year' interval handles this automatically in PostgreSQL
    RETURN (fy_end_date + INTERVAL '1 year')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Validate Fiscal Dates
-- ===================================================================
CREATE OR REPLACE FUNCTION validate_fiscal_dates(
    fy_end_date DATE,
    report_date DATE,
    reference_date DATE DEFAULT CURRENT_DATE
)
    RETURNS TABLE
            (
                issue    TEXT,
                severity TEXT
            )
AS
$$
BEGIN
    -- FY End in future relative to data
    IF fy_end_date > reference_date THEN
        RETURN QUERY SELECT 'FY End Date is in the future'::TEXT, 'WARNING'::TEXT;
    END IF;

    -- Report date before FY End (impossible)
    IF report_date IS NOT NULL AND report_date < fy_end_date - INTERVAL '1 year' THEN
        RETURN QUERY SELECT 'Report date predates fiscal year'::TEXT, 'ERROR'::TEXT;
    END IF;

    -- Report date too far in future
    IF report_date > reference_date + INTERVAL '1 day' THEN
        RETURN QUERY SELECT 'Report date is in the future'::TEXT, 'WARNING'::TEXT;
    END IF;

    -- FY End not on month boundary
    IF fy_end_date != (DATE_TRUNC('month', fy_end_date) + INTERVAL '1 month - 1 day')::DATE THEN
        RETURN QUERY SELECT 'FY End is not last day of month'::TEXT, 'INFO'::TEXT;
    END IF;
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
                      "FY End Date",
                      "Fiscal Month",
                      "Fiscal Quarter",
                      "Fiscal Year",
                      "Current Fiscal Quarter",
                      "Next Fiscal Quarter",
                      "Next Earnings (Report)",
                      "Next FY End Date",
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
                      "Reference Date",
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
SELECT NULLIF(TRIM("Ticker"), '')                                                AS "Ticker",
       NULLIF(TRIM("ISIN"), '')                                                  AS "ISIN",
       NULLIF(TRIM("Name"), '')                                                  AS "Name",
       NULLIF(TRIM("Description"), '')                                           AS "Description",
       NULLIF(TRIM("Region"), '')                                                AS "Region",
       NULLIF(TRIM("Country"), '')                                               AS "Country",
       NULLIF(TRIM("Trading Country"), '')                                       AS "Trading Country",
       NULLIF(TRIM("Exchange"), '')                                              AS "Exchange",
       NULLIF(TRIM("Unit"), '')                                                  AS "Unit",
       NULLIF(TRIM("Sector"), '')                                                AS "Sector",
       NULLIF(TRIM("Industry"), '')                                              AS "Industry",
       NULLIF(TRIM("Style Class"), '')                                           AS "Style Class",
       NULLIF(TRIM("Size Class"), '')                                            AS "Size Class",
       NULLIF(TRIM("Last Updated"), '')::DATE                                    AS "Last Updated",
       NULLIF(TRIM("Income Statement Report Date"), '')::DATE                    AS "Income Statement Report Date",
       COALESCE(NULLIF(TRIM("FY End"), ''), 'Dec 2024')                          AS "FY End",
       parsed.fy_end_date                                                        AS "FY End Date",
       report_fiscal.fiscal_month                                                AS "Fiscal Month",
       report_fiscal.fiscal_quarter                                              AS "Fiscal Quarter",
       report_fiscal.fiscal_year                                                 AS "Fiscal Year",
       'Q' || current_fiscal.fiscal_quarter || ' ' || current_fiscal.fiscal_year AS "Current Fiscal Quarter",
       'Q' || report_fiscal.next_quarter || ' ' ||
       report_fiscal.next_quarter_year                                           AS "Next Fiscal Quarter",
       calculate_next_earnings_report(
               'Q' || report_fiscal.next_quarter || ' ' || report_fiscal.next_quarter_year
       )                                                                         AS "Next Earnings (Report)",
       next_fy.next_fy_end_date                                                  AS "Next FY End Date",
       NULLIF(TRIM("Next Earnings"), '')::DATE                                   AS "Next Earnings",
       NULLIF(TRIM("Next Earnings (When)"), '')                                  AS "Next Earnings (When)",
       NULLIF(TRIM("Next Earnings (Status)"), '')                                AS "Next Earnings (Status)",
       NULLIF(TRIM("Dividend Record (Currency)"), '')                            AS "Dividend Record (Currency)",
       safe_to_numeric("Dividend Record (Amount)")                               AS "Dividend Record (Amount)",
       NULLIF(TRIM("Dividend Record (Frequency)"), '')                           AS "Dividend Record (Frequency)",
       COALESCE(safe_to_numeric("Dividend Streak"), 0)                           AS "Dividend Streak",
       NULLIF(TRIM("Dividend Record (Announce Date)"), '')::DATE                 AS "Dividend Record (Announce Date)",
       NULLIF(TRIM("Dividend Record (Payable Date)"), '')::DATE                  AS "Dividend Record (Payable Date)",
       NULLIF(TRIM("Dividend Record (Record Date)"), '')::DATE                   AS "Dividend Record (Record Date)",
       NULLIF(TRIM("Dividend Record (Ex Date)"), '')::DATE                       AS "Dividend Record (Ex Date)",
       CURRENT_DATE                                                              AS "Reference Date",
       safe_to_numeric("Market Cap")                                             AS "Market Cap",
       safe_to_numeric("Enterprise Value")                                       AS "Enterprise Value",
       safe_to_numeric("Last Price")                                             AS "Last Price",
       safe_to_numeric("Price Target (YTD Ago)")                                 AS "Price Target (YTD Ago)",
       safe_to_numeric("Total Return (YTD)")                                     AS "Total Return (YTD)",
       safe_to_numeric("Price Target")                                           AS "Price Target",
       safe_to_numeric("Price Target - Low")                                     AS "Price Target - Low",
       safe_to_numeric("Price Target - Median")                                  AS "Price Target - Median",
       safe_to_numeric("Price Target - High")                                    AS "Price Target - High",
       COALESCE(safe_to_numeric("Price Target - #"), 0)                          AS "Price Target - #",
       safe_to_numeric("P/E (NTM)")                                              AS "P/E (NTM)",
       safe_to_numeric("P/E (LTM)")                                              AS "P/E (LTM)",
       safe_to_numeric("Altman Z-Score (FY)")                                    AS "Altman Z-Score (FY)",
       safe_to_numeric("Altman Z-Score (FQ)")                                    AS "Altman Z-Score (FQ)",
       safe_to_numeric("Altman Z-Score (LTM)")                                   AS "Altman Z-Score (LTM)",
       safe_to_numeric("Beta (1Y)")                                              AS "Beta (1Y)",
       safe_to_numeric("Beta (2Y)")                                              AS "Beta (2Y)",
       safe_to_numeric("Beta (5Y)")                                              AS "Beta (5Y)",
       COALESCE(safe_to_numeric("Analyst Rating"), 0)                            AS "Analyst Rating",
       COALESCE(safe_to_numeric("# Strong Sell Ratings"), 0)                     AS "# Strong Sell Ratings",
       COALESCE(safe_to_numeric("# Strong Buys Ratings"), 0)                     AS "# Strong Buys Ratings",
       COALESCE(safe_to_numeric("# Hold Ratings"), 0)                            AS "# Hold Ratings",
       COALESCE(safe_to_numeric("# Buys Ratings"), 0)                            AS "# Buys Ratings",
       COALESCE(safe_to_numeric("# Sell Ratings"), 0)                            AS "# Sell Ratings",
       COALESCE(safe_to_numeric("Total Revenues/CAGR (5Y FY)"), 0)        AS "Total Revenues/CAGR (5Y FY)",
       COALESCE(safe_to_numeric("Total Revenues (FQ)"), 0)                       AS "Total Revenues (FQ)",
       COALESCE(safe_to_numeric("Total Revenues (-1FY)"), 0)                     AS "Total Revenues (-1FY)",
       COALESCE(safe_to_numeric("Total Revenues (FY)"), 0)                       AS "Total Revenues (FY)",
       COALESCE(safe_to_numeric("Total Revenues (LTM)"), 0)                      AS "Total Revenues (LTM)",
       COALESCE(safe_to_numeric("Total Operating Expenses (LTM)"), 0)            AS "Total Operating Expenses (LTM)",
       safe_to_numeric("P/TBV (LTM)")                                            AS "P/TBV (LTM)",
       COALESCE(safe_to_numeric("TBV (FY)"), 0)                                  AS "TBV (FY)",
       COALESCE(safe_to_numeric("TBV (LTM)"), 0)                                 AS "TBV (LTM)",
       safe_to_numeric("Market Cap (Country R)")                                 AS "Market Cap (Country R)",
       safe_to_numeric("Tot. Return %/CAGR (3Y)")                                AS "Tot. Return %/CAGR (3Y)",
       safe_to_numeric("Tot. Return %/CAGR (10Y)")                               AS "Tot. Return %/CAGR (10Y)",
       safe_to_numeric("Total Return (5Y)")                                      AS "Total Return (5Y)",
       safe_to_numeric("Total Return (10Y)")                                     AS "Total Return (10Y)",
       COALESCE(safe_to_numeric("Net Income/Adj. (-1FY)"), 0)                    AS "Net Income/Adj. (-1FY)",
       COALESCE(safe_to_numeric("CFF (LTM)"), 0)                                 AS "CFF (LTM)",
       COALESCE(safe_to_numeric("CFI (LTM)"), 0)                                 AS "CFI (LTM)",
       COALESCE(safe_to_numeric("FCF (LTM)"), 0)                                 AS "FCF (LTM)",
       COALESCE(safe_to_numeric("CFO (LTM)"), 0)                                 AS "CFO (LTM)",
       COALESCE(safe_to_numeric("EBITDA (FQ)"), 0)                               AS "EBITDA (FQ)",
       COALESCE(safe_to_numeric("EBITDA (LTM)"), 0)                              AS "EBITDA (LTM)",
       COALESCE(safe_to_numeric("EBITDA (FY)"), 0)                               AS "EBITDA (FY)",
       COALESCE(safe_to_numeric("EBITDA (-1FY)"), 0)                             AS "EBITDA (-1FY)",
       COALESCE(safe_to_numeric("EBITDA/Adj. (LTM)"), 0)                         AS "EBITDA/Adj. (LTM)",
       COALESCE(safe_to_numeric("EBITDA/Adj. (FY)"), 0)                          AS "EBITDA/Adj. (FY)",
       COALESCE(safe_to_numeric("EBITDA/Adj. (-1FY)"), 0)                        AS "EBITDA/Adj. (-1FY)",
       COALESCE(safe_to_numeric("EBIT (FQ)"), 0)                                 AS "EBIT (FQ)",
       COALESCE(safe_to_numeric("EBIT (LTM)"), 0)                                AS "EBIT (LTM)",
       COALESCE(safe_to_numeric("EBIT (FY)"), 0)                                 AS "EBIT (FY)",
       COALESCE(safe_to_numeric("EBIT (-1FY)"), 0)                               AS "EBIT (-1FY)",
       COALESCE(safe_to_numeric("EBIT/Adj. (-1FY)"), 0)                          AS "EBIT/Adj. (-1FY)",
       COALESCE(safe_to_numeric("EBIT/Adj. (FY)"), 0)                            AS "EBIT/Adj. (FY)",
       COALESCE(safe_to_numeric("EBIT/Adj. (LTM)"), 0)                           AS "EBIT/Adj. (LTM)",
       COALESCE(safe_to_numeric("EBIT - Est Med (FY1E)"), 0)                     AS "EBIT - Est Med (FY1E)",
       COALESCE(safe_to_numeric("EBIT - Est Med (NTM)"), 0)                      AS "EBIT - Est Med (NTM)",
       safe_to_numeric("Return On Equity % (LTM)")                               AS "Return On Equity % (LTM)",
       safe_to_numeric("Return On Equity % (FY)")                                AS "Return On Equity % (FY)",
       COALESCE(safe_to_numeric("Net Income - (IS) (FY)"), 0)                    AS "Net Income - (IS) (FY)",
       COALESCE(safe_to_numeric("Net Income - (IS) (LTM)"), 0)                   AS "Net Income - (IS) (LTM)",
       COALESCE(safe_to_numeric("Normalized Net Income (FY)"), 0)                AS "Normalized Net Income (FY)",
       COALESCE(safe_to_numeric("Normalized Net Income (LTM)"), 0)               AS "Normalized Net Income (LTM)",
       COALESCE(safe_to_numeric("Net Income/Adj. (FY)"), 0)                      AS "Net Income/Adj. (FY)",
       COALESCE(safe_to_numeric("Net Income/Adj. (LTM)"), 0)                     AS "Net Income/Adj. (LTM)",
       safe_to_numeric("Net Income Margin % (FY)")                               AS "Net Income Margin % (FY)",
       safe_to_numeric("Net Income Margin % (LTM)")                              AS "Net Income Margin % (LTM)",
       COALESCE(safe_to_numeric("Volatility (1M)"), 0)                    AS "Volatility (1M)",
       COALESCE(safe_to_numeric("Volatility (3M)"), 0)                    AS "Volatility (3M)",
       COALESCE(safe_to_numeric("Volatility (6M)"), 0)                    AS "Volatility (6M)",
       COALESCE(safe_to_numeric("Volatility (1Y)"), 0)                    AS "Volatility (1Y)",
       COALESCE(safe_to_numeric("Volume (Shrs)"), 0)                      AS "Volume (Shrs)",
       COALESCE(safe_to_numeric("Dividend Per Share (LTM)"), 0)           AS "Dividend Per Share (LTM)",
       COALESCE(safe_to_numeric("Div Yield (Ind)"), 0)                    AS "Div Yield (Ind)",
       COALESCE(safe_to_numeric("Div Yield (LTM)"), 0)                    AS "Div Yield (LTM)",
       COALESCE(safe_to_numeric("Total Debt (FY)"), 0)                           AS "Total Debt (FY)",
       COALESCE(safe_to_numeric("Total Equity (FY)"), 0)                         AS "Total Equity (FY)",
       COALESCE(safe_to_numeric("Total Equity (LTM)"), 0)                        AS "Total Equity (LTM)",
       COALESCE(safe_to_numeric("Total Debt (LTM)"), 0)                          AS "Total Debt (LTM)",
       COALESCE(safe_to_numeric("Total Assets (LTM)"), 0)                        AS "Total Assets (LTM)",
       COALESCE(safe_to_numeric("Total Assets (FY)"), 0)                         AS "Total Assets (FY)",
       safe_to_numeric("Current Ratio (FY)")                                     AS "Current Ratio (FY)",
       safe_to_numeric("Current Ratio (LTM)")                                    AS "Current Ratio (LTM)",
       safe_to_numeric("Gross Profit Margin % (FY)")                             AS "Gross Profit Margin % (FY)",
       safe_to_numeric("Gross Profit Margin % (LTM)")                            AS "Gross Profit Margin % (LTM)",
       safe_to_numeric("Asset Turnover (FY)")                                    AS "Asset Turnover (FY)",
       safe_to_numeric("Asset Turnover (LTM)")                                   AS "Asset Turnover (LTM)",
       COALESCE(safe_to_numeric("Gross Profit (LTM)"), 0)                        AS "Gross Profit (LTM)",
       COALESCE(safe_to_numeric("Gross Profit (FY)"), 0)                         AS "Gross Profit (FY)",
       safe_to_numeric("EPS Norm - Est Avg (NTM)")                               AS "EPS Norm - Est Avg (NTM)",
       safe_to_numeric("EPS/Adj. (-1FY)")                                        AS "EPS/Adj. (-1FY)",
       safe_to_numeric("EPS/Adj. (FY)")                                          AS "EPS/Adj. (FY)",
       safe_to_numeric("EPS/Adj. (LTM)")                                         AS "EPS/Adj. (LTM)",
       safe_to_numeric("EPS Norm - Est Avg (FY1E)")                              AS "EPS Norm - Est Avg (FY1E)",
       COALESCE(safe_to_numeric("Gain (Loss) On Sale Of Assets (LTM)"), 0)       AS "Gain (Loss) On Sale Of Assets (LTM)",
       COALESCE(safe_to_numeric("Cost Of Revenues (LTM)"), 0)                    AS "Cost Of Revenues (LTM)",
       COALESCE(safe_to_numeric("Cash Acquisitions (LTM)"), 0)                   AS "Cash Acquisitions (LTM)",
       COALESCE(safe_to_numeric("Cash Acquisitions (FY)"), 0)                    AS "Cash Acquisitions (FY)",
       COALESCE(safe_to_numeric("Cash Acquisitions (-1FY)"), 0)                  AS "Cash Acquisitions (-1FY)",
       COALESCE(safe_to_numeric("Inventory (LTM)"), 0)                           AS "Inventory (LTM)",
       COALESCE(safe_to_numeric("Goodwill (FQ)"), 0)                             AS "Goodwill (FQ)",
       COALESCE(safe_to_numeric("Goodwill (LTM)"), 0)                            AS "Goodwill (LTM)",
       COALESCE(safe_to_numeric("Goodwill (FY)"), 0)                             AS "Goodwill (FY)",
       COALESCE(safe_to_numeric("Goodwill (-1FY)"), 0)                           AS "Goodwill (-1FY)",
       COALESCE(safe_to_numeric("Impairment of Goodwill (FQ)"), 0)               AS "Impairment of Goodwill (FQ)",
       COALESCE(safe_to_numeric("Impairment of Goodwill (LTM)"), 0)              AS "Impairment of Goodwill (LTM)",
       COALESCE(safe_to_numeric("Impairment of Goodwill (-1FY)"), 0)             AS "Impairment of Goodwill (-1FY)",
       COALESCE(safe_to_numeric("Impairment of Goodwill (FY)"), 0)               AS "Impairment of Goodwill (FY)",
       COALESCE(safe_to_numeric("Operating Income (LTM)"), 0)                    AS "Operating Income (LTM)",
       COALESCE(safe_to_numeric("Asset Writedown (LTM)"), 0)                     AS "Asset Writedown (LTM)",
       COALESCE(safe_to_numeric("Asset Writedown (FY)"), 0)                      AS "Asset Writedown (FY)",
       COALESCE(safe_to_numeric("Asset Writedown (-1FY)"), 0)                    AS "Asset Writedown (-1FY)",
       COALESCE(safe_to_numeric("Operating Income (FY)"), 0)                     AS "Operating Income (FY)",
       COALESCE(safe_to_numeric("Capital Expenditure (LTM)"), 0)                 AS "Capital Expenditure (LTM)",
       COALESCE(safe_to_numeric("Capital Expenditure (-1FY)"), 0)                AS "Capital Expenditure (-1FY)",
       COALESCE(safe_to_numeric("Capital Expenditure (FY)"), 0)                  AS "Capital Expenditure (FY)",
       COALESCE(safe_to_numeric("Retained Earnings (LTM)"), 0)                   AS "Retained Earnings (LTM)",
       COALESCE(safe_to_numeric("Total Current Assets (LTM)"), 0)                AS "Total Current Assets (LTM)",
       COALESCE(safe_to_numeric("Total Current Liabilities (LTM)"), 0)           AS "Total Current Liabilities (LTM)",
       COALESCE(safe_to_numeric("R&D Expenses (LTM)"), 0)                        AS "R&D Expenses (LTM)",
       COALESCE(safe_to_numeric("Restructuring Charges (LTM)"), 0)               AS "Restructuring Charges (LTM)",
       COALESCE(safe_to_numeric("Restructuring Charges (FQ)"), 0)                AS "Restructuring Charges (FQ)",
       COALESCE(safe_to_numeric("Restructuring Charges (-1FY)"), 0)              AS "Restructuring Charges (-1FY)",
       COALESCE(safe_to_numeric("Restructuring Charges (FY)"), 0)                AS "Restructuring Charges (FY)",
       COALESCE(safe_to_numeric("Interest Expense/Total (LTM)"), 0)              AS "Interest Expense/Total (LTM)",
       COALESCE(safe_to_numeric("Merger & Restructuring Charges (LTM)"), 0)      AS "Merger & Restructuring Charges (LTM)",
       COALESCE(safe_to_numeric("Working Capital (LTM)"), 0)                     AS "Working Capital (LTM)",
       COALESCE(safe_to_numeric("Other Unusual Items/Total (LTM)"), 0)           AS "Other Unusual Items/Total (LTM)",
       COALESCE(safe_to_numeric("Interest Income On Investments (LTM)"), 0)      AS "Interest Income On Investments (LTM)",
       COALESCE(safe_to_numeric("Buyback Yield (LTM)"), 0)                AS "Buyback Yield (LTM)",
       safe_to_numeric("Return on Assets (ROA) % (LTM)")                         AS "Return on Assets (ROA) % (LTM)",
       safe_to_numeric("Return on Assets (ROA) % (FY)")                          AS "Return on Assets (ROA) % (FY)",
       COALESCE(safe_to_numeric("Net Income - (IS) (-1FY)"), 0)                  AS "Net Income - (IS) (-1FY)",
       COALESCE(safe_to_numeric("Normalized Net Income (-1FY)"), 0)              AS "Normalized Net Income (-1FY)",
       COALESCE(safe_to_numeric("CFF (FY)"), 0)                                  AS "CFF (FY)",
       COALESCE(safe_to_numeric("CFF (-1FY)"), 0)                                AS "CFF (-1FY)",
       COALESCE(safe_to_numeric("CFI (FY)"), 0)                                  AS "CFI (FY)",
       COALESCE(safe_to_numeric("CFI (-1FY)"), 0)                                AS "CFI (-1FY)",
       COALESCE(safe_to_numeric("CFO (FY)"), 0)                                  AS "CFO (FY)",
       COALESCE(safe_to_numeric("CFO (-1FY)"), 0)                                AS "CFO (-1FY)",
       safe_to_numeric("Div Yield (-1FYInd)")                                    AS "Div Yield (-1FYInd)",
       COALESCE(safe_to_numeric("FCF (FY)"), 0)                                  AS "FCF (FY)",
       COALESCE(safe_to_numeric("Capital Expenditure (FQ)"), 0)                  AS "Capital Expenditure (FQ)",
       COALESCE(safe_to_numeric("Capital Expenditure (5YAVGFQ)"), 0)             AS "Capital Expenditure (5YAVGFQ)",
       COALESCE(safe_to_numeric("CFF (FQ)"), 0)                                  AS "CFF (FQ)",
       COALESCE(safe_to_numeric("CFI (FQ)"), 0)                                  AS "CFI (FQ)",
       COALESCE(safe_to_numeric("CFO (FQ)"), 0)                                  AS "CFO (FQ)",
       COALESCE(safe_to_numeric("FCF (FQ)"), 0)                                  AS "FCF (FQ)",
       COALESCE(safe_to_numeric("Total Revenues (5YAVGFQ)"), 0)                  AS "Total Revenues (5YAVGFQ)",
       COALESCE(safe_to_numeric("EBITDA (5YAVGFQ)"), 0)                          AS "EBITDA (5YAVGFQ)",
       COALESCE(safe_to_numeric("EBIT (5YAVGFQ)"), 0)                            AS "EBIT (5YAVGFQ)",
       COALESCE(safe_to_numeric("FCF (5YAVGFQ)"), 0)                             AS "FCF (5YAVGFQ)",
       COALESCE(safe_to_numeric("Cash Acquisitions (FQ)"), 0)                    AS "Cash Acquisitions (FQ)",
       COALESCE(safe_to_numeric("Cash Acquisitions (5YAVGFQ)"), 0)               AS "Cash Acquisitions (5YAVGFQ)",
       COALESCE(safe_to_numeric("Asset Writedown (FQ)"), 0)                      AS "Asset Writedown (FQ)",
       COALESCE(safe_to_numeric("Asset Writedown (5YAVGFQ)"), 0)                 AS "Asset Writedown (5YAVGFQ)",
       COALESCE(safe_to_numeric("Impairment of Goodwill (5YAVGFQ)"), 0)          AS "Impairment of Goodwill (5YAVGFQ)",
       COALESCE(safe_to_numeric("Operating Income (FQ)"), 0)                     AS "Operating Income (FQ)",
       COALESCE(safe_to_numeric("Operating Income (5YAVGFQ)"), 0)                AS "Operating Income (5YAVGFQ)",
       safe_to_numeric("P/B (LTM)")                                              AS "P/B (LTM)",
       safe_to_numeric("P/B (-1FY)")                                             AS "P/B (-1FY)",
       safe_to_numeric("P/B (5YAVG)")                                            AS "P/B (5YAVG)",
       COALESCE(safe_to_numeric("Cash And Equivalents (LTM)"), 0)                AS "Cash And Equivalents (LTM)",
       COALESCE(safe_to_numeric("Cash And Equivalents (FQ)"), 0)                 AS "Cash And Equivalents (FQ)",
       COALESCE(safe_to_numeric("Cash And Equivalents (FY)"), 0)                 AS "Cash And Equivalents (FY)",
       COALESCE(safe_to_numeric("Cash And Equivalents (5YAVGFQ)"), 0)            AS "Cash And Equivalents (5YAVGFQ)",
       COALESCE(safe_to_numeric("Inventory (FQ)"), 0)                            AS "Inventory (FQ)",
       COALESCE(safe_to_numeric("Inventory (FY)"), 0)                            AS "Inventory (FY)",
       COALESCE(safe_to_numeric("Goodwill (5YAVGFQ)"), 0)                        AS "Goodwill (5YAVGFQ)",
       COALESCE(safe_to_numeric("Inventory (5YAVGFQ)"), 0)                       AS "Inventory (5YAVGFQ)",
       COALESCE(safe_to_numeric("Retained Earnings (FQ)"), 0)                    AS "Retained Earnings (FQ)",
       COALESCE(safe_to_numeric("Retained Earnings (FY)"), 0)                    AS "Retained Earnings (FY)",
       COALESCE(safe_to_numeric("Retained Earnings (5YAVGFQ)"), 0)               AS "Retained Earnings (5YAVGFQ)",
       COALESCE(safe_to_numeric("Working Capital (FQ)"), 0)                      AS "Working Capital (FQ)",
       COALESCE(safe_to_numeric("Working Capital (FY)"), 0)                      AS "Working Capital (FY)",
       COALESCE(safe_to_numeric("Working Capital (5YAVGFY)"), 0)                 AS "Working Capital (5YAVGFY)",
       COALESCE(safe_to_numeric("Div Yield (TTM)"), 0)                    AS "Div Yield (TTM)",
       COALESCE(safe_to_numeric("Div Yield (NTM)"), 0)                    AS "Div Yield (NTM)",
       COALESCE(safe_to_numeric("Div Yield (5YAVGLTM)"), 0)               AS "Div Yield (5YAVGLTM)",
       COALESCE(safe_to_numeric("Gross Intangible Assets (LTM)"), 0)             AS "Gross Intangible Assets (LTM)",
       COALESCE(safe_to_numeric("Gross Intangible Assets (FY)"), 0)              AS "Gross Intangible Assets (FY)",
       COALESCE(safe_to_numeric("Gross Intangible Assets (5YAVGFQ)"), 0)         AS "Gross Intangible Assets (5YAVGFQ)",
       COALESCE(safe_to_numeric("Restructuring Charges (5YAVGFQ)"), 0)           AS "Restructuring Charges (5YAVGFQ)",
       COALESCE(safe_to_numeric("Merger & Restructuring Charges (FQ)"), 0)       AS "Merger & Restructuring Charges (FQ)",
       COALESCE(safe_to_numeric("Merger & Restructuring Charges (FY)"), 0)       AS "Merger & Restructuring Charges (FY)",
       COALESCE(safe_to_numeric("Merger & Restructuring Charges (5YAVGFQ)"),
                0)                                                               AS "Merger & Restructuring Charges (5YAVGFQ)",
       COALESCE(safe_to_numeric("Normalized Net Income (FQ)"), 0)                AS "Normalized Net Income (FQ)",
       COALESCE(safe_to_numeric("Normalized Net Income (5YAVGFQ)"), 0)           AS "Normalized Net Income (5YAVGFQ)",
       COALESCE(safe_to_numeric("Net Income/Adj. (FQ)"), 0)                      AS "Net Income/Adj. (FQ)",
       COALESCE(safe_to_numeric("Net Income/Adj. (5YAVGFQ)"), 0)                 AS "Net Income/Adj. (5YAVGFQ)",
       COALESCE(safe_to_numeric("Net Income - (IS) (FQ)"), 0)                    AS "Net Income - (IS) (FQ)",
       COALESCE(safe_to_numeric("Net Income - (IS) (5YAVGFQ)"), 0)               AS "Net Income - (IS) (5YAVGFQ)",
       COALESCE(safe_to_numeric("Net Income - (IS) (5YAVGLTM)"), 0)              AS "Net Income - (IS) (5YAVGLTM)",
       COALESCE(safe_to_numeric("Normalized Net Income (5YAVGLTM)"), 0)          AS "Normalized Net Income (5YAVGLTM)",
       COALESCE(safe_to_numeric("EBITDA (5YAVGLTM)"), 0)                         AS "EBITDA (5YAVGLTM)",
       COALESCE(safe_to_numeric("EBIT (5YAVGLTM)"), 0)                           AS "EBIT (5YAVGLTM)",
       COALESCE(safe_to_numeric("Total Revenues (5YAVGLTM)"), 0)                 AS "Total Revenues (5YAVGLTM)",
       COALESCE(safe_to_numeric("Revenues - Est YoY % (FY1E)"), 0)        AS "Revenues - Est YoY % (FY1E)",
       COALESCE(safe_to_numeric("Price Chg. % (1M)"), 0)                  AS "Price Chg. % (1M)",
       COALESCE(safe_to_numeric("Price Chg. % (3M)"), 0)                  AS "Price Chg. % (3M)",
       COALESCE(safe_to_numeric("1-Day %"), 0)                            AS "1-Day %",
       COALESCE(safe_to_numeric("Price (5D Ago)"), 0)                     AS "Price (5D Ago)",
       COALESCE(safe_to_numeric("Price (1W Ago)"), 0)                     AS "Price (1W Ago)",
       COALESCE(safe_to_numeric("Price (1M Ago)"), 0)                     AS "Price (1M Ago)",
       COALESCE(safe_to_numeric("Price (3M Ago)"), 0)                     AS "Price (3M Ago)",
       COALESCE(safe_to_numeric("Price (6M Ago)"), 0)                     AS "Price (6M Ago)",
       COALESCE(safe_to_numeric("Price (1Y Ago)"), 0)                     AS "Price (1Y Ago)",
       COALESCE(safe_to_numeric("Price (3Y Ago)"), 0)                     AS "Price (3Y Ago)",
       COALESCE(safe_to_numeric("Price (5Y Ago)"), 0)                     AS "Price (5Y Ago)",
       COALESCE(safe_to_numeric("Price (QTD Ago)"), 0)                    AS "Price (QTD Ago)",
       COALESCE(safe_to_numeric("Rel. Volume"), 0)                        AS "Rel. Volume",
       COALESCE(safe_to_numeric("Shrs Out"), 0)                                  AS "Shrs Out",
       COALESCE(safe_to_numeric("Shrs Out (-1FY)"), 0)                           AS "Shrs Out (-1FY)",
       COALESCE(safe_to_numeric("Common Dividends Paid (LTM)"), 0)               AS "Common Dividends Paid (LTM)",
       COALESCE(safe_to_numeric("Common Dividends Paid (FY)"), 0)                AS "Common Dividends Paid (FY)",
       COALESCE(safe_to_numeric("Selling General & Admin Expenses/Total (FQ)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (FQ)",
       COALESCE(safe_to_numeric("Selling General & Admin Expenses/Total (FY)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (FY)",
       COALESCE(safe_to_numeric("Selling General & Admin Expenses/Total (-1FY)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (-1FY)",
       COALESCE(safe_to_numeric("Selling General & Admin Expenses/Total (5YAVGFQ)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (5YAVGFQ)",
       COALESCE(safe_to_numeric("Accounts Receivable/Total (FY)"), 0)            AS "Accounts Receivable/Total (FY)",
       COALESCE(safe_to_numeric("Accounts Receivable/Total (-1FY)"), 0)          AS "Accounts Receivable/Total (-1FY)",
       COALESCE(safe_to_numeric("Accounts Receivable/Total (5YAVGFQ)"), 0)       AS "Accounts Receivable/Total (5YAVGFQ)",
       COALESCE(safe_to_numeric("Marketing Expenses (FQ)"), 0)                   AS "Marketing Expenses (FQ)",
       COALESCE(safe_to_numeric("Marketing Expenses (FY)"), 0)                   AS "Marketing Expenses (FY)",
       COALESCE(safe_to_numeric("Marketing Expenses (-1FY)"), 0)                 AS "Marketing Expenses (-1FY)",
       COALESCE(safe_to_numeric("Marketing Expenses (5YAVGLTM)"), 0)             AS "Marketing Expenses (5YAVGLTM)",
       COALESCE(safe_to_numeric("Revenues - Est Avg (NTM)"), 0)                  AS "Revenues - Est Avg (NTM)",
       COALESCE(safe_to_numeric("Revenues - Est Avg (FY1E)"), 0)                 AS "Revenues - Est Avg (FY1E)",
       COALESCE(safe_to_numeric("Revenues - Est Med (NTM)"), 0)                  AS "Revenues - Est Med (NTM)",
       COALESCE(safe_to_numeric("Revenues - Est Med (FY1E)"), 0)                 AS "Revenues - Est Med (FY1E)",
       safe_to_numeric("EV/Sales (EST FY1)")                                     AS "EV/Sales (EST FY1)",
       safe_to_numeric("EV/Sales (LTM)")                                         AS "EV/Sales (LTM)",
       safe_to_numeric("EV/Sales (NTM)")                                         AS "EV/Sales (NTM)",
       safe_to_numeric("EV/Sales (-1FYLTM)")                                     AS "EV/Sales (-1FYLTM)",
       safe_to_numeric("EV/Sales (-2FYLTM)")                                     AS "EV/Sales (-2FYLTM)",
       safe_to_numeric("EV/Sales (-3FYLTM)")                                     AS "EV/Sales (-3FYLTM)",
       safe_to_numeric("EV/Sales (3YAVGLTM)")                                    AS "EV/Sales (3YAVGLTM)",
       safe_to_numeric("EV/Sales (-1FQLTM)")                                     AS "EV/Sales (-1FQLTM)",
       safe_to_numeric("EV/Sales (-2FQLTM)")                                     AS "EV/Sales (-2FQLTM)",
       safe_to_numeric("EV/Sales (-3FQLTM)")                                     AS "EV/Sales (-3FQLTM)",
       safe_to_numeric("EV/Sales (-4FQLTM)")                                     AS "EV/Sales (-4FQLTM)",
       safe_to_numeric("52W High/Adj")                                           AS "52W High/Adj",
       safe_to_numeric("52W Low/Adj")                                            AS "52W Low/Adj",
       safe_to_numeric("EMA (20D)")                                              AS "EMA (20D)",
       safe_to_numeric("EMA (50D)")                                              AS "EMA (50D)",
       safe_to_numeric("EMA (100D)")                                             AS "EMA (100D)",
       safe_to_numeric("EMA (250D)")                                             AS "EMA (250D)",
       safe_to_numeric("EV/EBITDA (LTM)")                                        AS "EV/EBITDA (LTM)",
       safe_to_numeric("EV/EBITDA (NTM)")                                        AS "EV/EBITDA (NTM)",
       safe_to_numeric("EV/EBITDA (-1FYLTM)")                                    AS "EV/EBITDA (-1FYLTM)",
       safe_to_numeric("EV/EBITDA (-1FQLTM)")                                    AS "EV/EBITDA (-1FQLTM)",
       safe_to_numeric("EV/EBITDA (3YAVGLTM)")                                   AS "EV/EBITDA (3YAVGLTM)",
       safe_to_numeric("EV/EBITDA (EST FY1)")                                    AS "EV/EBITDA (EST FY1)",
       safe_to_numeric("P/E (EST FY1)")                                          AS "P/E (EST FY1)",
       safe_to_numeric("P/E (-1FYLTM)")                                          AS "P/E (-1FYLTM)",
       safe_to_numeric("P/E (-2FYLTM)")                                          AS "P/E (-2FYLTM)",
       safe_to_numeric("P/E (-3FYLTM)")                                          AS "P/E (-3FYLTM)",
       safe_to_numeric("P/E (3YAVGLTM)")                                         AS "P/E (3YAVGLTM)",
       safe_to_numeric("P/E (-1FQLTM)")                                          AS "P/E (-1FQLTM)",
       safe_to_numeric("P/E (-2FQLTM)")                                          AS "P/E (-2FQLTM)",
       safe_to_numeric("P/E (-3FQLTM)")                                          AS "P/E (-3FQLTM)",
       safe_to_numeric("P/E (5YAVGLTM)")                                         AS "P/E (5YAVGLTM)",
       safe_to_numeric("P/E (-0FQQoQLTM)")                                       AS "P/E (-0FQQoQLTM)",
       safe_to_numeric("P/E (-0FYYoYLTM)")                                       AS "P/E (-0FYYoYLTM)",
       safe_to_numeric("P/E (-1FYYoYLTM)")                                       AS "P/E (-1FYYoYLTM)",
       safe_to_numeric("P/E (-0FQYoYLTM)")                                       AS "P/E (-0FQYoYLTM)",
       COALESCE(safe_to_numeric("Full Time Employees (FQ)"), 0)                  AS "Full Time Employees (FQ)",
       COALESCE(safe_to_numeric("Full Time Employees (FY)"), 0)                  AS "Full Time Employees (FY)",
       COALESCE(safe_to_numeric("Full Time Employees (-1FY)"), 0)                AS "Full Time Employees (-1FY)",
       COALESCE(safe_to_numeric("Full Time Employees (-2FY)"), 0)                AS "Full Time Employees (-2FY)",
       COALESCE(safe_to_numeric("Full Time Employees (-3FY)"), 0)                AS "Full Time Employees (-3FY)",
       COALESCE(safe_to_numeric("Avg Employees (5YAVGFY)"), 0)                   AS "Avg Employees (5YAVGFY)",
       safe_to_numeric("Net EPS - Basic (LTM)")                                  AS "Net EPS - Basic (LTM)",
       safe_to_numeric("Net EPS - Basic (FQ)")                                   AS "Net EPS - Basic (FQ)",
       safe_to_numeric("Net EPS - Basic (FY)")                                   AS "Net EPS - Basic (FY)",
       safe_to_numeric("Net EPS - Basic (-1FQFQ)")                               AS "Net EPS - Basic (-1FQFQ)",
       safe_to_numeric("Net EPS - Basic (-2FQFQ)")                               AS "Net EPS - Basic (-2FQFQ)",
       safe_to_numeric("Net EPS - Basic (-3FQFQ)")                               AS "Net EPS - Basic (-3FQFQ)",
       safe_to_numeric("Net EPS - Basic (-4FQFQ)")                               AS "Net EPS - Basic (-4FQFQ)",
       safe_to_numeric("Net EPS - Basic (-1FY)")                                 AS "Net EPS - Basic (-1FY)",
       safe_to_numeric("Net EPS - Basic (-2FY)")                                 AS "Net EPS - Basic (-2FY)",
       safe_to_numeric("Net EPS - Basic (-3FY)")                                 AS "Net EPS - Basic (-3FY)",
       safe_to_numeric("Net EPS - Basic (-4FY)")                                 AS "Net EPS - Basic (-4FY)",
       safe_to_numeric("Net EPS - Basic (-5FY)")                                 AS "Net EPS - Basic (-5FY)",
       COALESCE(safe_to_numeric("EPS Est Avg Rev % (FY1E - 1W)"), 0)      AS "EPS Est Avg Rev % (FY1E - 1W)",
       COALESCE(safe_to_numeric("EPS Est Avg Rev % (FY1E - 1M)"), 0)      AS "EPS Est Avg Rev % (FY1E - 1M)",
       COALESCE(safe_to_numeric("EPS Est Avg Rev % (FY1E - 3M)"), 0)      AS "EPS Est Avg Rev % (FY1E - 3M)",
       COALESCE(safe_to_numeric("EPS Est Avg Rev % (FY1E - 6M)"), 0)      AS "EPS Est Avg Rev % (FY1E - 6M)",
       COALESCE(safe_to_numeric("EPS Est Avg Rev % (FY1E - 1Y)"), 0)      AS "EPS Est Avg Rev % (FY1E - 1Y)",
       COALESCE(safe_to_numeric("Div Yield (-2FYInd)"), 0)                AS "Div Yield (-2FYInd)",
       COALESCE(safe_to_numeric("Div Yield (-3FYInd)"), 0)                AS "Div Yield (-3FYInd)",
       COALESCE(safe_to_numeric("Div Yield (-4FYInd)"), 0)                AS "Div Yield (-4FYInd)",
       COALESCE(safe_to_numeric("Div Yield (-5FYInd)"), 0)                AS "Div Yield (-5FYInd)",
       COALESCE(safe_to_numeric("EBITDA - Est Avg (NTM)"), 0)                    AS "EBITDA - Est Avg (NTM)",
       COALESCE(safe_to_numeric("EBITDA - Est Avg (FY1E)"), 0)                   AS "EBITDA - Est Avg (FY1E)",
       safe_to_numeric("EPS GAAP - Est Avg (NTM)")                               AS "EPS GAAP - Est Avg (NTM)",
       safe_to_numeric("EPS GAAP - Est Avg (FY1E)")                              AS "EPS GAAP - Est Avg (FY1E)",
       COALESCE(safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 1M)"), 0) AS "EPS GAAP Est Avg Rev % (FY1E - 1M)",
       COALESCE(safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 3M)"), 0) AS "EPS GAAP Est Avg Rev % (FY1E - 3M)",
       COALESCE(safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 6M)"), 0) AS "EPS GAAP Est Avg Rev % (FY1E - 6M)",
       COALESCE(safe_to_numeric("EPS GAAP Est Avg Rev % (FY1E - 1Y)"), 0) AS "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
       COALESCE(safe_to_numeric("EPS Norm - Est # (FY1E)"), 0)                   AS "EPS Norm - Est # (FY1E)"
FROM screening_staging s,
     LATERAL (
         SELECT parse_fy_end_to_date(COALESCE(NULLIF(TRIM(s."FY End"), ''), 'Dec 2024')) AS fy_end_date
         ) parsed,
     LATERAL (
         SELECT calculate_next_fy_end_date(parsed.fy_end_date) AS next_fy_end_date
         ) next_fy,
     LATERAL (
         SELECT * FROM calculate_fiscal_info(CURRENT_DATE, parsed.fy_end_date)
         ) current_fiscal,
     LATERAL (
         SELECT *
         FROM calculate_fiscal_info(NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE, parsed.fy_end_date)
         ) report_fiscal
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
DROP FUNCTION IF EXISTS safe_to_numeric(TEXT);
DROP FUNCTION IF EXISTS parse_fy_end_to_date(TEXT);
DROP FUNCTION IF EXISTS calculate_fiscal_info(DATE, DATE);
DROP FUNCTION IF EXISTS calculate_next_earnings_report(TEXT);
DROP FUNCTION IF EXISTS calculate_next_fy_end_date(DATE);
DROP FUNCTION IF EXISTS validate_fiscal_dates(DATE, DATE, DATE);

\echo 'Import complete!'

-- Auto-generated from COLUMN_SCHEMA
COPY equities ("Ticker", "ISIN", "Name", "Description", "Sector", "Industry", "Region", "Country", "Trading Country",
               "P/E (NTM)", "Last Price", "Price Target", "Market Cap")
    FROM '/path/to/data.csv'
    WITH (FORMAT csv, HEADER true, NULL '');
