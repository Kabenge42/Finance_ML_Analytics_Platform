# Column Mapping Reference

This document provides the column alias mapping between SQL column names and Python normalized names used in the Finance
ML Analytics Platform.

> **Note**: This reference was extracted from `import_equities_data.sql` to improve maintainability.
> For the authoritative schema definition, see `create_equities_schema.sql`.

## Column Alias Reference (SQL → Python normalized name)

### Identifiers

| SQL Column                       | Python Name                    |
|----------------------------------|--------------------------------|
| `"Ticker"`                       | `ticker`                       |
| `"ISIN"`                         | `isin`                         |
| `"Name"`                         | `name`                         |
| `"Description"`                  | `description`                  |
| `"Exchange"`                     | `exchange`                     |
| `"Unit"`                         | `unit`                         |
| `"Sector"`                       | `sector`                       |
| `"Industry"`                     | `industry`                     |
| `"Last Updated"`                 | `last_updated`                 |
| `"Income Statement Report Date"` | `income_statement_report_date` |
| `"FY End"`                       | `fy_end`                       |
| `"Next Earnings"`                | `next_earnings`                |
| `"Next Earnings (When)"`         | `next_earnings_when`           |
| `"Style Class"`                  | `style_class`                  |
| `"Next Earnings (Status)"`       | `next_earnings_status`         |
| `"Size Class"`                   | `size_class`                   |
| `"Region"`                       | `region`                       |
| `"Country"`                      | `country`                      |
| `"Trading Country"`              | `trading_country`              |

### Market Values

| SQL Column           | Python Name        |
|----------------------|--------------------|
| `"Market Cap"`       | `market_cap`       |
| `"Enterprise Value"` | `enterprise_value` |

### Price Columns (NEVER transform)

| SQL Column                 | Python Name            |
|----------------------------|------------------------|
| `"Last Price"`             | `last_price`           |
| `"Price Target"`           | `price_target`         |
| `"Price Target (YTD Ago)"` | `price_target_ytd_ago` |
| `"Price Target - Low"`     | `price_target_low`     |
| `"Price Target - Median"`  | `price_target_median`  |
| `"Price Target - High"`    | `price_target_high`    |
| `"Price Target - #"`       | `price_target_count`   |
| `"Price (5D Ago)"`         | `price_5d_ago`         |
| `"Price (1W Ago)"`         | `price_1w_ago`         |
| `"Price (1M Ago)"`         | `price_1m_ago`         |
| `"Price (3M Ago)"`         | `price_3m_ago`         |
| `"Price (6M Ago)"`         | `price_6m_ago`         |
| `"Price (1Y Ago)"`         | `price_1y_ago`         |
| `"Price (3Y Ago)"`         | `price_3y_ago`         |
| `"Price (5Y Ago)"`         | `price_5y_ago`         |
| `"Price (QTD Ago)"`        | `price_qtd_ago`        |
| `"52W High/Adj"`           | `52w_high_adj`         |
| `"52W Low/Adj"`            | `52w_low_adj`          |
| `"EMA (20D)"`              | `ema_20d`              |
| `"EMA (50D)"`              | `ema_50d`              |
| `"EMA (100D)"`             | `ema_100d`             |
| `"EMA (250D)"`             | `ema_250d`             |

### Ratio Columns

| SQL Column               | Python Name          |
|--------------------------|----------------------|
| `"P/E (NTM)"`            | `p_e_ntm`            |
| `"P/E (LTM)"`            | `p_e_ltm`            |
| `"P/E (EST FY1)"`        | `p_e_est_fy1`        |
| `"P/E (-1FYLTM)"`        | `p_e_1fyltm`         |
| `"P/E (-2FYLTM)"`        | `p_e_2fyltm`         |
| `"P/E (-3FYLTM)"`        | `p_e_3fyltm`         |
| `"P/E (3YAVGLTM)"`       | `p_e_3yavgltm`       |
| `"P/E (-1FQLTM)"`        | `p_e_1fqltm`         |
| `"P/E (-2FQLTM)"`        | `p_e_2fqltm`         |
| `"P/E (-3FQLTM)"`        | `p_e_3fqltm`         |
| `"P/E (5YAVGLTM)"`       | `p_e_5yavgltm`       |
| `"P/E (-0FQQoQLTM)"`     | `p_e_0fqqoqltm`      |
| `"P/E (-0FYYoYLTM)"`     | `p_e_0fyyoyltm`      |
| `"P/E (-1FYYoYLTM)"`     | `p_e_1fyyoyltm`      |
| `"P/E (-0FQYoYLTM)"`     | `p_e_0fqyoyltm`      |
| `"P/B (LTM)"`            | `p_b_ltm`            |
| `"P/B (-1FY)"`           | `p_b_1fy`            |
| `"P/B (5YAVG)"`          | `p_b_5yavg`          |
| `"P/TBV (LTM)"`          | `p_tbv_ltm`          |
| `"EV/Sales (LTM)"`       | `ev_sales_ltm`       |
| `"EV/Sales (NTM)"`       | `ev_sales_ntm`       |
| `"EV/Sales (EST FY1)"`   | `ev_sales_est_fy1`   |
| `"EV/Sales (-1FYLTM)"`   | `ev_sales_1fyltm`    |
| `"EV/Sales (-2FYLTM)"`   | `ev_sales_2fyltm`    |
| `"EV/Sales (-3FYLTM)"`   | `ev_sales_3fyltm`    |
| `"EV/Sales (3YAVGLTM)"`  | `ev_sales_3yavgltm`  |
| `"EV/Sales (-1FQLTM)"`   | `ev_sales_1fqltm`    |
| `"EV/Sales (-2FQLTM)"`   | `ev_sales_2fqltm`    |
| `"EV/Sales (-3FQLTM)"`   | `ev_sales_3fqltm`    |
| `"EV/Sales (-4FQLTM)"`   | `ev_sales_4fqltm`    |
| `"EV/EBITDA (LTM)"`      | `ev_ebitda_ltm`      |
| `"EV/EBITDA (NTM)"`      | `ev_ebitda_ntm`      |
| `"EV/EBITDA (EST FY1)"`  | `ev_ebitda_est_fy1`  |
| `"EV/EBITDA (-1FYLTM)"`  | `ev_ebitda_1fyltm`   |
| `"EV/EBITDA (-1FQLTM)"`  | `ev_ebitda_1fqltm`   |
| `"EV/EBITDA (3YAVGLTM)"` | `ev_ebitda_3yavgltm` |
| `"Altman Z-Score (FY)"`  | `altman_z_score_fy`  |
| `"Altman Z-Score (FQ)"`  | `altman_z_score_fq`  |
| `"Altman Z-Score (LTM)"` | `altman_z_score_ltm` |
| `"Current Ratio (FY)"`   | `current_ratio_fy`   |
| `"Current Ratio (LTM)"`  | `current_ratio_ltm`  |
| `"Asset Turnover (FY)"`  | `asset_turnover_fy`  |
| `"Asset Turnover (LTM)"` | `asset_turnover_ltm` |

### Percentage Columns

| SQL Column                         | Python Name                    |
|------------------------------------|--------------------------------|
| `"Beta (1Y)"`                      | `beta_1y`                      |
| `"Beta (2Y)"`                      | `beta_2y`                      |
| `"Beta (5Y)"`                      | `beta_5y`                      |
| `"Total Return (YTD)"`             | `total_return_ytd`             |
| `"Total Return (5Y)"`              | `total_return_5y`              |
| `"Total Return (10Y)"`             | `total_return_10y`             |
| `"Tot. Return %/CAGR (3Y)"`        | `tot_return_pct_cagr_3y`       |
| `"Tot. Return %/CAGR (10Y)"`       | `tot_return_pct_cagr_10y`      |
| `"Price Chg. % (1M)"`              | `price_chg_pct_1m`             |
| `"Price Chg. % (3M)"`              | `price_chg_pct_3m`             |
| `"1-Day %"`                        | `one_day_pct`                  |
| `"Volatility (1M)"`                | `volatility_1m`                |
| `"Volatility (3M)"`                | `volatility_3m`                |
| `"Volatility (6M)"`                | `volatility_6m`                |
| `"Volatility (1Y)"`                | `volatility_1y`                |
| `"Net Income Margin % (FY)"`       | `net_income_margin_pct_fy`     |
| `"Net Income Margin % (LTM)"`      | `net_income_margin_pct_ltm`    |
| `"Gross Profit Margin % (FY)"`     | `gross_profit_margin_pct_fy`   |
| `"Gross Profit Margin % (LTM)"`    | `gross_profit_margin_pct_ltm`  |
| `"Return On Equity % (LTM)"`       | `return_on_equity_pct_ltm`     |
| `"Return On Equity % (FY)"`        | `return_on_equity_pct_fy`      |
| `"Return on Assets (ROA) % (LTM)"` | `return_on_assets_roa_pct_ltm` |
| `"Return on Assets (ROA) % (FY)"`  | `return_on_assets_roa_pct_fy`  |
| `"Total Revenues/CAGR (5Y FY)"`    | `total_revenues_cagr_5y_fy`    |
| `"Revenues - Est YoY % (FY1E)"`    | `revenues_est_yoy_pct_fy1e`    |

### Count Columns

| SQL Column                     | Python Name               |
|--------------------------------|---------------------------|
| `"Analyst Rating"`             | `analyst_rating`          |
| `"# Strong Sell Ratings"`      | `num_strong_sell_ratings` |
| `"# Strong Buys Ratings"`      | `num_strong_buys_ratings` |
| `"# Hold Ratings"`             | `num_hold_ratings`        |
| `"# Buys Ratings"`             | `num_buys_ratings`        |
| `"# Sell Ratings"`             | `num_sell_ratings`        |
| `"Shrs Out"`                   | `shares_outstanding`      |
| `"Shrs Out (-1FY)"`            | `shrs_out_1fy`            |
| `"Full Time Employees (FQ)"`   | `full_time_employees_fq`  |
| `"Full Time Employees (FY)"`   | `full_time_employees_fy`  |
| `"Full Time Employees (-1FY)"` | `full_time_employees_1fy` |
| `"Full Time Employees (-2FY)"` | `full_time_employees_2fy` |
| `"Full Time Employees (-3FY)"` | `full_time_employees_3fy` |
| `"Avg Employees (5YAVGFY)"`    | `avg_employees_5yavgfy`   |
| `"Dividend Streak"`            | `dividend_streak`         |
| `"EPS Norm - Est # (FY1E)"`    | `eps_norm_est_num_fy1e`   |

### Financial Statement Columns (Log-transform recommended)

| SQL Column                           | Python Name                      |
|--------------------------------------|----------------------------------|
| `"TBV (FY)"`                         | `tbv_fy`                         |
| `"TBV (LTM)"`                        | `tbv_ltm`                        |
| `"Market Cap (Country R)"`           | `market_cap_country_r`           |
| `"Total Revenues (FQ)"`              | `total_revenues_fq`              |
| `"Total Revenues (-1FY)"`            | `total_revenues_1fy`             |
| `"Total Revenues (FY)"`              | `total_revenues_fy`              |
| `"Total Revenues (LTM)"`             | `total_revenues_ltm`             |
| `"Total Revenues (5YAVGFQ)"`         | `total_revenues_5yavgfq`         |
| `"Total Revenues (5YAVGLTM)"`        | `total_revenues_5yavgltm`        |
| `"Total Operating Expenses (LTM)"`   | `total_operating_expenses_ltm`   |
| `"EBITDA (FQ)"`                      | `ebitda_fq`                      |
| `"EBITDA (LTM)"`                     | `ebitda_ltm`                     |
| `"EBITDA (FY)"`                      | `ebitda_fy`                      |
| `"EBITDA (-1FY)"`                    | `ebitda_1fy`                     |
| `"EBITDA/Adj. (LTM)"`                | `ebitda_adj_ltm`                 |
| `"EBITDA/Adj. (FY)"`                 | `ebitda_adj_fy`                  |
| `"EBITDA/Adj. (-1FY)"`               | `ebitda_adj_1fy`                 |
| `"EBITDA (5YAVGFQ)"`                 | `ebitda_5yavgfq`                 |
| `"EBITDA (5YAVGLTM)"`                | `ebitda_5yavgltm`                |
| `"EBIT (FQ)"`                        | `ebit_fq`                        |
| `"EBIT (LTM)"`                       | `ebit_ltm`                       |
| `"EBIT (FY)"`                        | `ebit_fy`                        |
| `"EBIT (-1FY)"`                      | `ebit_1fy`                       |
| `"EBIT/Adj. (-1FY)"`                 | `ebit_adj_1fy`                   |
| `"EBIT/Adj. (FY)"`                   | `ebit_adj_fy`                    |
| `"EBIT/Adj. (LTM)"`                  | `ebit_adj_ltm`                   |
| `"EBIT - Est Med (FY1E)"`            | `ebit_est_med_fy1e`              |
| `"EBIT - Est Med (NTM)"`             | `ebit_est_med_ntm`               |
| `"EBIT (5YAVGFQ)"`                   | `ebit_5yavgfq`                   |
| `"EBIT (5YAVGLTM)"`                  | `ebit_5yavgltm`                  |
| `"Net Income - (IS) (FY)"`           | `net_income_is_fy`               |
| `"Net Income - (IS) (LTM)"`          | `net_income_is_ltm`              |
| `"Net Income - (IS) (FQ)"`           | `net_income_is_fq`               |
| `"Net Income - (IS) (-1FY)"`         | `net_income_is_1fy`              |
| `"Net Income - (IS) (5YAVGFQ)"`      | `net_income_is_5yavgfq`          |
| `"Net Income - (IS) (5YAVGLTM)"`     | `net_income_is_5yavgltm`         |
| `"Normalized Net Income (FY)"`       | `normalized_net_income_fy`       |
| `"Normalized Net Income (LTM)"`      | `normalized_net_income_ltm`      |
| `"Normalized Net Income (FQ)"`       | `normalized_net_income_fq`       |
| `"Normalized Net Income (-1FY)"`     | `normalized_net_income_1fy`      |
| `"Normalized Net Income (5YAVGFQ)"`  | `normalized_net_income_5yavgfq`  |
| `"Normalized Net Income (5YAVGLTM)"` | `normalized_net_income_5yavgltm` |
| `"Net Income/Adj. (FY)"`             | `net_income_adj_fy`              |
| `"Net Income/Adj. (LTM)"`            | `net_income_adj_ltm`             |
| `"Net Income/Adj. (FQ)"`             | `net_income_adj_fq`              |
| `"Net Income/Adj. (-1FY)"`           | `net_income_adj_1fy`             |
| `"Net Income/Adj. (5YAVGFQ)"`        | `net_income_adj_5yavgfq`         |
| `"Operating Income (LTM)"`           | `operating_income_ltm`           |
| `"Operating Income (FY)"`            | `operating_income_fy`            |
| `"Operating Income (FQ)"`            | `operating_income_fq`            |
| `"Operating Income (5YAVGFQ)"`       | `operating_income_5yavgfq`       |
| `"Gross Profit (LTM)"`               | `gross_profit_ltm`               |
| `"Gross Profit (FY)"`                | `gross_profit_fy`                |
| `"Total Debt (FY)"`                  | `total_debt_fy`                  |
| `"Total Debt (LTM)"`                 | `total_debt_ltm`                 |
| `"Total Equity (FY)"`                | `total_equity_fy`                |
| `"Total Equity (LTM)"`               | `total_equity_ltm`               |
| `"Total Assets (LTM)"`               | `total_assets_ltm`               |
| `"Total Assets (FY)"`                | `total_assets_fy`                |
| `"Total Current Assets (LTM)"`       | `total_current_assets_ltm`       |
| `"Total Current Liabilities (LTM)"`  | `total_current_liabilities_ltm`  |
| `"Working Capital (LTM)"`            | `working_capital_ltm`            |
| `"Working Capital (FQ)"`             | `working_capital_fq`             |
| `"Working Capital (FY)"`             | `working_capital_fy`             |
| `"Working Capital (5YAVGFY)"`        | `working_capital_5yavgfy`        |
| `"Cash And Equivalents (LTM)"`       | `cash_and_equivalents_ltm`       |
| `"Cash And Equivalents (FQ)"`        | `cash_and_equivalents_fq`        |
| `"Cash And Equivalents (FY)"`        | `cash_and_equivalents_fy`        |
| `"Cash And Equivalents (5YAVGFQ)"`   | `cash_and_equivalents_5yavgfq`   |
| `"Retained Earnings (LTM)"`          | `retained_earnings_ltm`          |
| `"Retained Earnings (FQ)"`           | `retained_earnings_fq`           |
| `"Retained Earnings (FY)"`           | `retained_earnings_fy`           |
| `"Retained Earnings (5YAVGFQ)"`      | `retained_earnings_5yavgfq`      |

### Cash Flow Columns

| SQL Column                        | Python Name                   |
|-----------------------------------|-------------------------------|
| `"CFO (LTM)"`                     | `cfo_ltm`                     |
| `"CFO (FY)"`                      | `cfo_fy`                      |
| `"CFO (FQ)"`                      | `cfo_fq`                      |
| `"CFO (-1FY)"`                    | `cfo_1fy`                     |
| `"FCF (LTM)"`                     | `fcf_ltm`                     |
| `"FCF (FY)"`                      | `fcf_fy`                      |
| `"FCF (FQ)"`                      | `fcf_fq`                      |
| `"FCF (5YAVGFQ)"`                 | `fcf_5yavgfq`                 |
| `"CFI (LTM)"`                     | `cfi_ltm`                     |
| `"CFI (FY)"`                      | `cfi_fy`                      |
| `"CFI (FQ)"`                      | `cfi_fq`                      |
| `"CFI (-1FY)"`                    | `cfi_1fy`                     |
| `"CFF (LTM)"`                     | `cff_ltm`                     |
| `"CFF (FY)"`                      | `cff_fy`                      |
| `"CFF (FQ)"`                      | `cff_fq`                      |
| `"CFF (-1FY)"`                    | `cff_1fy`                     |
| `"Capital Expenditure (LTM)"`     | `capital_expenditure_ltm`     |
| `"Capital Expenditure (FY)"`      | `capital_expenditure_fy`      |
| `"Capital Expenditure (FQ)"`      | `capital_expenditure_fq`      |
| `"Capital Expenditure (-1FY)"`    | `capital_expenditure_1fy`     |
| `"Capital Expenditure (5YAVGFQ)"` | `capital_expenditure_5yavgfq` |
| `"R&D Expenses (LTM)"`            | `randd_expenses_ltm`          |
| `"Revenues - Est Avg (NTM)"`      | `revenues_est_avg_ntm`        |
| `"Revenues - Est Avg (FY1E)"`     | `revenues_est_avg_fy1e`       |
| `"Revenues - Est Med (NTM)"`      | `revenues_est_med_ntm`        |
| `"Revenues - Est Med (FY1E)"`     | `revenues_est_med_fy1e`       |
| `"EBITDA - Est Avg (NTM)"`        | `ebitda_est_avg_ntm`          |
| `"EBITDA - Est Avg (FY1E)"`       | `ebitda_est_avg_fy1e`         |

### Feature Columns

| SQL Column                      | Python Name                 |
|---------------------------------|-----------------------------|
| `"Volume (Shrs)"`               | `volume_shrs`               |
| `"Rel. Volume"`                 | `rel_volume`                |
| `"Dividend Per Share (LTM)"`    | `dividend_per_share_ltm`    |
| `"Div Yield (Ind)"`             | `div_yield_ind`             |
| `"Div Yield (LTM)"`             | `div_yield_ltm`             |
| `"Div Yield (TTM)"`             | `div_yield_ttm`             |
| `"Div Yield (NTM)"`             | `div_yield_ntm`             |
| `"Div Yield (-1FYInd)"`         | `div_yield_1fyind`          |
| `"Div Yield (-2FYInd)"`         | `div_yield_2fyind`          |
| `"Div Yield (-3FYInd)"`         | `div_yield_3fyind`          |
| `"Div Yield (-4FYInd)"`         | `div_yield_4fyind`          |
| `"Div Yield (-5FYInd)"`         | `div_yield_5fyind`          |
| `"Div Yield (5YAVGLTM)"`        | `div_yield_5yavgltm`        |
| `"Common Dividends Paid (LTM)"` | `common_dividends_paid_ltm` |
| `"Common Dividends Paid (FY)"`  | `common_dividends_paid_fy`  |
| `"Buyback Yield (LTM)"`         | `buyback_yield_ltm`         |
| `"EPS Norm - Est Avg (NTM)"`    | `eps_norm_est_avg_ntm`      |
| `"EPS Norm - Est Avg (FY1E)"`   | `eps_norm_est_avg_fy1e`     |
| `"EPS/Adj. (-1FY)"`             | `eps_adj_1fy`               |
| `"EPS/Adj. (FY)"`               | `eps_adj_fy`                |
| `"EPS/Adj. (LTM)"`              | `eps_adj_ltm`               |
| `"Net EPS - Basic (LTM)"`       | `net_eps_basic_ltm`         |
| `"Net EPS - Basic (FQ)"`        | `net_eps_basic_fq`          |
| `"Net EPS - Basic (FY)"`        | `net_eps_basic_fy`          |

### Dividend Record Columns

| SQL Column                          | Python Name                     |
|-------------------------------------|---------------------------------|
| `"Dividend Record (Announce Date)"` | `dividend_record_announce_date` |
| `"Dividend Record (Ex Date)"`       | `dividend_record_ex_date`       |
| `"Dividend Record (Payable Date)"`  | `dividend_record_payable_date`  |
| `"Dividend Record (Record Date)"`   | `dividend_record_record_date`   |
| `"Dividend Record (Frequency)"`     | `dividend_record_frequency`     |
| `"Dividend Record (Currency)"`      | `dividend_record_currency`      |
| `"Dividend Record (Amount)"`        | `dividend_record_amount`        |

---

## Related Files

- **Schema Definition**: `create_equities_schema.sql`
- **Import Script**: `import_equities_data.sql`
- **Python Schema**: `finance_ml/core/schema.py`
