create materialized view mv_all_stock_features as
SELECT "Ticker"                          AS ticker,
       "ISIN"                            AS isin,
       "Name"                                                                                                     AS name,
       "Region"                                                                                                   AS region,
       "Country"                                                                                                  AS country,
       "Trading Country"                                                                                          AS trading_country,
       "Exchange"                                                                                                 AS exchange,
       "Sector"                                                                                                   AS sector,
       "Industry"                                                                                                 AS industry,
       "Next Earnings (When)"                                                                                     AS next_earnings_when,
       "Next Earnings (Status)"                                                                                   AS next_earnings_status,
       "Dividend Record (Currency)"      AS dividend_record_currency,
       "Dividend Record (Frequency)"     AS dividend_record_frequency,
       "Current Fiscal Quarter"                                                                                   AS current_fiscal_quarter,
       "Next Fiscal Quarter"                                                                                      AS next_fiscal_quarter,
       "Next Earnings (Report)"                                                                                   AS next_earnings_report,
       "Earnings Report (Frequency)"                                                                              AS earnings_report_frequency,
       "Last Updated"                    AS last_updated,
       "Income Statement Report Date"                                                                             AS income_statement_report_date,
       "Next Earnings"                   AS next_earnings,
       "FY End Date"                     AS fy_end_date,
       "Next FY End Date"                AS next_fy_end_date,
       "Next Income Statement Report Date"                                                                        AS next_income_statement_report_date,
       "Price Target"                    AS price_target,
       "Price Target - Median"           AS price_target_median,
       "Dividend Record (Amount)"                                                                                 AS dividend_record_amount,
       "Market Cap"                                                                                               AS market_cap,
       "Enterprise Value"                                                                                         AS enterprise_value,
       "Last Price"                                                                                               AS last_price,
       "Price Target - Low"              AS price_target_low,
       "Price Target - High"             AS price_target_high,
       "Price Target - High (1W Ago)"    AS price_target_high_1w_ago,
       "Price Target - High (1M Ago)"    AS price_target_high_1m_ago,
       "Price Target - High (6M Ago)"    AS price_target_high_6m_ago,
       "Price Target - High (MTD Ago)"   AS price_target_high_mtd_ago,
       "Price Target - High (3M Ago)"    AS price_target_high_3m_ago,
       "Price Target - High (QTD Ago)"   AS price_target_high_qtd_ago,
       "Price Target - High (1Y Ago)"    AS price_target_high_1y_ago,
       "Price Target - High (YTD Ago)"   AS price_target_high_ytd_ago,
       "Price Target - Low (1W Ago)"     AS price_target_low_1w_ago,
       "Price Target - Low (1M Ago)"     AS price_target_low_1m_ago,
       "Price Target - Low (3M Ago)"     AS price_target_low_3m_ago,
       "Price Target - Low (6M Ago)"     AS price_target_low_6m_ago,
       "Price Target - Low (MTD Ago)"    AS price_target_low_mtd_ago,
       "Price Target - Low (QTD Ago)"    AS price_target_low_qtd_ago,
       "Price Target - Low (YTD Ago)"    AS price_target_low_ytd_ago,
       "Price Target - Low (1Y Ago)"     AS price_target_low_1y_ago,
       "Price Target - Median (1W Ago)"  AS price_target_median_1w_ago,
       "Price Target - Median (1M Ago)"  AS price_target_median_1m_ago,
       "Price Target - Median (3M Ago)"  AS price_target_median_3m_ago,
       "Price Target - Median (6M Ago)"  AS price_target_median_6m_ago,
       "Price Target - Median (MTD Ago)" AS price_target_median_mtd_ago,
       "Price Target - Median (QTD Ago)" AS price_target_median_qtd_ago,
       "Price Target - Median (YTD Ago)" AS price_target_median_ytd_ago,
       "Price Target - Median (1Y Ago)"  AS price_target_median_1y_ago,
       "Price Target - #"                AS price_target_count,
       "Price Target - # (3M Ago)"       AS price_target_count_3m_ago,
       "Price Target - # (6M Ago)"       AS price_target_count_6m_ago,
       "Price Target - # (YTD Ago)"      AS price_target_count_ytd_ago,
       "Price Target - # (1Y Ago)"       AS price_target_count_1y_ago,
       "Price Target - # (1W Ago)"       AS price_target_count_1w_ago,
       "Price Target - # (1M Ago)"       AS price_target_count_1m_ago,
       "Price Target - # (MTD Ago)"      AS price_target_count_mtd_ago,
       "Price Target - # (QTD Ago)"      AS price_target_count_qtd_ago,
       "P/E (LTM)"                                                                                                AS p_e_ratio,
       "P/B (LTM)"                                                                                                AS p_b_ratio,
       "EV/EBITDA (LTM)"                                                                                          AS ev_ebitda_ratio,
       "EV/Sales (LTM)"                                                                                           AS ev_sales_ratio,
       "Div Yield (LTM)"                                                                                          AS dividend_yield,
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0::numeric THEN safe_divide("P/E (LTM)", "Total Revenues/CAGR (5Y FY)")
           ELSE NULL::numeric
           END                                                                                                    AS peg_ratio,
       calc_change_ratio("EV/Sales (LTM)", "EV/Sales (-1FYLTM)")                                                  AS ev_sales_trend_1y,
       calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (-1FYLTM)")                                                AS ev_ebitda_momentum,
       calc_change_ratio("P/E (LTM)", "P/E (-1FYLTM)")                                                            AS p_e_momentum_yoy,
       calc_change_ratio("P/E (LTM)", "P/E (-1FQLTM)")                                                            AS p_e_momentum_qoq,
       calc_change_ratio("EV/Sales (LTM)", "EV/Sales (3YAVGLTM)")                                                 AS ev_sales_vs_3y_avg,
       calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)")                                               AS ev_ebitda_vs_3y_avg,
       calc_change_ratio("P/E (LTM)", "P/E (3YAVGLTM)")                                                           AS p_e_vs_3y_avg,
       calc_change_ratio("EV/Sales (NTM)", "EV/Sales (LTM)")                                                      AS ev_sales_forward_discount,
       calc_change_ratio("EV/EBITDA (NTM)", "EV/EBITDA (LTM)")                                                    AS ev_ebitda_forward_discount,
       calc_change_ratio("P/E (EST FY1)", "P/E (LTM)")                                                            AS p_e_forward_discount,
       safe_divide("P/B (LTM)", "P/B (5YAVG)")                                                                    AS p_b_vs_5y_avg,
       ("EV/Sales (LTM)" - "EV/Sales (-1FQLTM)") /
       NULLIF("EV/Sales (-1FQLTM)", 0::numeric)                                                                   AS ev_sales_qoq_1q,
       ("P/E (LTM)" - "P/E (5YAVGLTM)") /
       NULLIF("P/E (5YAVGLTM)", 0::numeric)                                                                       AS p_e_vs_5y_avg_ext,
       ("P/B (LTM)" - "P/B (-1FY)") / NULLIF("P/B (-1FY)", 0::numeric)                                            AS p_b_momentum_yoy,
       ("P/E (EST FY1)" - "P/E (LTM)") / NULLIF(abs("P/E (LTM)"), 0::numeric) *
       100::numeric                                                                                               AS forward_pe_premium,
       pct_change("Last Price", "Price (1M Ago)")                                                                 AS price_momentum_1m,
       pct_change("Last Price", "Price (3M Ago)")                                                                 AS price_momentum_3m,
       pct_change("Last Price", "Price (6M Ago)")                                                                 AS price_momentum_6m,
       pct_change("Last Price", "Price (1Y Ago)")                                                                 AS price_momentum_1y,
       pct_change("Last Price", "Price (5D Ago)")                                                                 AS price_momentum_5d,
       ema_crossover_signal("EMA (20D)", "EMA (50D)")                                                             AS ema_crossover_20_50,
       ema_crossover_signal("EMA (50D)", "EMA (250D)")                                                            AS ema_crossover_50_250,
       calc_change_ratio("Last Price", "EMA (20D)")                                                               AS price_vs_ema_20d,
       calc_change_ratio("Last Price", "EMA (250D)")                                                              AS price_vs_ema_250d,
       calc_change_ratio("52W High/Adj" - "Last Price", "52W High/Adj")                                           AS pct_off_52w_high,
       calc_change_ratio("Last Price" - "52W Low/Adj", "52W Low/Adj")                                             AS pct_above_52w_low,
       clamp_score(safe_divide("Last Price" - "52W Low/Adj", "52W High/Adj" - "52W Low/Adj"), 0::numeric,
                   1::numeric)                                                                                    AS range_52w_position,
       "Beta (1Y)" - "Beta (5Y)"                                                                                  AS beta_momentum,
       safe_divide("Volatility (1M)", "Volatility (1Y)")                                                          AS volatility_regime,
       ("EMA (20D)" - "EMA (50D)") / NULLIF("EMA (50D)", 0::numeric)                                              AS ema_slope_20d,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" AND "EMA (50D)" > "EMA (100D)" AND "EMA (100D)" > "EMA (250D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" AND "EMA (50D)" < "EMA (100D)" AND "EMA (100D)" < "EMA (250D)" THEN '-1'::integer
           ELSE 0
           END                                                                                                    AS ema_trend_consistency,
       ("Last Price" - "EMA (100D)") / NULLIF("EMA (100D)", 0::numeric) *
       100::numeric                                                                                               AS price_vs_ema_100d,
       CASE
           WHEN (("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0::numeric)) <= 0.05 THEN 1
           ELSE 0
           END                                                                                                    AS near_52w_high_flag,
       CASE
           WHEN (("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0::numeric)) <= 0.05 THEN 1
           ELSE 0
           END                                                                                                    AS near_52w_low_flag,
       "Rel. Volume" * "Price Chg. % (1M)"                                                                        AS volume_momentum_score,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" AND
                (("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0::numeric)) <= 0.05 THEN 1
           ELSE 0
           END                                                                                                    AS breakout_signal,
       CASE
           WHEN "Rel. Volume" > 1.5 THEN 1
           ELSE 0
           END                                                                                                    AS high_volume_flag,
       CASE
           WHEN "Rel. Volume" < 0.5 THEN 1
           ELSE 0
           END                                                                                                    AS low_volume_flag,
       "Volatility (1Y)" - "Volatility (1M)"                                                                      AS volatility_compression,
       "Volatility (3M)" - "Volatility (6M)"                                                                      AS volatility_term_structure,
       "Return On Equity % (LTM)"                                                                                 AS roe,
       "Return on Assets (ROA) % (LTM)"                                                                           AS roa,
       "Gross Profit Margin % (LTM)"                                                                              AS gross_margin_pct,
       "Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS operating_margin_pct,
       "Net Income Margin % (LTM)"                                                                                AS net_margin_pct,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS ebitda_margin_pct,
       "Net Income - (IS) (LTM)" / NULLIF("Total Equity (LTM)" + "Total Debt (LTM)", 0::numeric) *
       100::numeric                                                                                               AS roic,
       "R&D Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric)                                          AS rnd_intensity,
       "Total Assets (LTM)" / NULLIF("Total Equity (LTM)", 0::numeric)                                            AS equity_multiplier,
       "Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)"                                               AS gross_margin_trend_yoy,
       ("Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) -
        "Operating Income (FY)" / NULLIF("Total Revenues (FY)", 0::numeric)) *
       100::numeric                                                                                               AS operating_margin_trend,
       "Net Income Margin % (LTM)" - "Net Income Margin % (FY)"                                                   AS net_margin_trend_yoy,
       ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) -
        "EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0::numeric)) *
       100::numeric                                                                                               AS ebitda_margin_trend,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" AND
                "Net Income Margin % (LTM)" > "Net Income Margin % (FY)" AND
                ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric)) >
                ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0::numeric)) THEN 1
           ELSE 0
           END                                                                                                    AS margin_expansion_flag,
       CASE
           WHEN "Impairment of Goodwill (LTM)" <> 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS has_goodwill_impairment,
       CASE
           WHEN "Asset Writedown (LTM)" <> 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS has_asset_writedown,
       CASE
           WHEN "Restructuring Charges (LTM)" <> 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS has_restructuring,
       "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric) *
       100::numeric                                                                                               AS goodwill_to_assets_pct,
       "Gross Intangible Assets (LTM)" /
       NULLIF("Total Assets (LTM)", 0::numeric)                                                                   AS intangible_intensity,
       (abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") + abs("Restructuring Charges (LTM)")) /
       NULLIF(abs("EBITDA (LTM)"), 0::numeric)                                                                    AS exceptional_items_to_ebitda,
       "Altman Z-Score (LTM)"                                                                                     AS altman_z_score,
       "Altman Z-Score (FY)" - "Altman Z-Score (LTM)"                                                             AS altman_z_trend,
       "Current Ratio (LTM)"                                                                                      AS current_ratio,
       ("Total Current Assets (LTM)" - "Inventory (LTM)") /
       NULLIF("Total Current Liabilities (LTM)", 0::numeric)                                                      AS quick_ratio,
       GREATEST(0::numeric, LEAST(100::numeric, ("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0::numeric) *
                                                100::numeric))                                                    AS distress_risk_score,
       CASE
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
           ELSE 0.0
           END                                                                                                    AS liquidity_stress_score,
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(abs("Working Capital (FY)"), 0::numeric)                                                            AS working_capital_trend,
       "Cash And Equivalents (FQ)" /
       NULLIF("Total Operating Expenses (LTM)" / 12.0, 0::numeric)                                                AS cash_runway_months,
       CASE
           WHEN "Retained Earnings (FQ)" < 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS accumulated_deficit_flag,
       CASE
           WHEN ("Cash And Equivalents (FQ)" / NULLIF("Total Operating Expenses (LTM)" / 12.0, 0::numeric)) > 6::numeric
               THEN 1
           ELSE 0
           END                                                                                                    AS adequate_cash_buffer,
       ("Goodwill (LTM)" - "Goodwill (-1FY)") /
       NULLIF("Goodwill (-1FY)", 0::numeric)                                                                      AS goodwill_change_rate,
       "Restructuring Charges (LTM)" /
       NULLIF("Total Assets (LTM)", 0::numeric)                                                                   AS restructuring_intensity,
       CASE
           WHEN abs("Impairment of Goodwill (FQ)") > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN abs("Asset Writedown (FQ)") > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN abs("Restructuring Charges (FQ)") > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS exceptional_items_frequency,
       "Merger & Restructuring Charges (LTM)" /
       NULLIF("Market Cap", 0::numeric)                                                                           AS merger_impact_ratio,
       "Interest Income On Investments (LTM)" /
       NULLIF(abs("Net Income - (IS) (LTM)"), 0::numeric)                                                         AS non_operating_income_share,
       CASE
           WHEN "Gain (Loss) On Sale Of Assets (LTM)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS asset_sale_boost,
       GREATEST(0, LEAST(100, 100 -
                              CASE
                                  WHEN "Impairment of Goodwill (LTM)" <> 0::numeric THEN 25
                                  ELSE 0
                                  END -
                              CASE
                                  WHEN "Asset Writedown (LTM)" <> 0::numeric THEN 10
                                  ELSE 0
                                  END -
                              CASE
                                  WHEN "Restructuring Charges (LTM)" <> 0::numeric THEN 15
                                  ELSE 0
                                  END -
                              CASE
                                  WHEN ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)) > 0.30 THEN 15
                                  ELSE 0
                                  END -
                              CASE
                                  WHEN ((abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") +
                                         abs("Restructuring Charges (LTM)")) /
                                        NULLIF(abs("Net Income - (IS) (LTM)"), 0::numeric)) > 0.10 THEN 15
                                  ELSE 0
                                  END))                                                                           AS accounting_quality_score,
       "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0::numeric)                                              AS debt_to_equity,
       "Total Debt (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)                                              AS debt_to_assets,
       "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)                                            AS equity_ratio,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0::numeric)                                          AS interest_coverage,
       "Cash And Equivalents (LTM)" /
       NULLIF("Total Current Liabilities (LTM)", 0::numeric)                                                      AS cash_ratio,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)                                         AS working_capital_ratio,
       "Total Revenues (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)                                          AS asset_turnover,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0::numeric)                                           AS inventory_turnover,
       "Accounts Receivable/Total (FY)" /
       NULLIF("Total Revenues (FY)" / 365.0, 0::numeric)                                                          AS receivables_days,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0::numeric)                                       AS working_capital_turns,
       "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric) *
       100::numeric                                                                                               AS cash_to_assets_pct,
       ("Cash And Equivalents (FQ)" - "Cash And Equivalents (FY)") /
       NULLIF(abs("Cash And Equivalents (FY)"), 0::numeric)                                                       AS cash_change_qoq,
       "Cash And Equivalents (FQ)" /
       NULLIF("Cash And Equivalents (5YAVGFQ)", 0::numeric)                                                       AS cash_vs_5y_avg,
       ("Inventory (FY)" - "Inventory (FQ)") /
       NULLIF(abs("Inventory (FQ)"), 0::numeric)                                                                  AS inventory_change_yoy,
       "Inventory (FQ)" / NULLIF("Inventory (5YAVGFQ)", 0::numeric)                                               AS inventory_vs_5y_avg,
       "Working Capital (FQ)" /
       NULLIF("Working Capital (5YAVGFY)", 0::numeric)                                                            AS working_capital_vs_5y_avg,
       "Retained Earnings (FQ)" /
       NULLIF("Retained Earnings (5YAVGFQ)", 0::numeric)                                                          AS retained_earnings_vs_5y,
       CASE
           WHEN ("Gross Intangible Assets (FY)" / NULLIF("Gross Intangible Assets (5YAVGFQ)", 0::numeric)) > 1.5 THEN 1
           ELSE 0
           END                                                                                                    AS intangibles_growth_flag,
       GREATEST(0::numeric, LEAST(100::numeric, 50::numeric + "Cash And Equivalents (LTM)" /
                                                              NULLIF("Total Assets (LTM)", 0::numeric) * 100::numeric -
                                                "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric) *
                                                100::numeric))                                                    AS asset_quality_score,
       GREATEST(0, LEAST(100,
                         CASE
                             WHEN ("Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)) > 0.10
                                 THEN 25
                             ELSE 0
                             END +
                         CASE
                             WHEN ("Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)) > 0.40 THEN 25
                             ELSE 0
                             END +
                         CASE
                             WHEN "Working Capital (LTM)" > 0::numeric THEN 25
                             ELSE 0
                             END +
                         CASE
                             WHEN "Current Ratio (LTM)" > 1.5 THEN 25
                             ELSE 0
                             END))                                                                                AS balance_sheet_strength,
       "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0::numeric)                                                    AS debt_maturity_risk,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# Sell Ratings" +
                 "# Strong Sell Ratings") > 0::numeric THEN ("# Strong Buys Ratings" + "# Buys Ratings") / NULLIF(
                   "# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# Sell Ratings" +
                   "# Strong Sell Ratings", 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS analyst_bullish_pct,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# Sell Ratings" +
                 "# Strong Sell Ratings") > 0::numeric THEN ("# Sell Ratings" + "# Strong Sell Ratings") / NULLIF(
                   "# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# Sell Ratings" +
                   "# Strong Sell Ratings", 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS analyst_bearish_pct,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# Sell Ratings" +
                 "# Strong Sell Ratings") > 0::numeric THEN "# Hold Ratings" / NULLIF(
                   "# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# Sell Ratings" +
                   "# Strong Sell Ratings", 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS analyst_neutral_pct,
       ("Price Target - Median" - "Last Price") / NULLIF("Last Price", 0::numeric) *
       100::numeric                                                                                               AS upside_potential,
       ("Price Target - High" - "Price Target - Low") / NULLIF("Price Target - Median", 0::numeric) *
       100::numeric                                                                                               AS price_target_spread_pct,
       ("Price Target" - "Price Target (1M Ago)") /
       NULLIF("Price Target (1M Ago)", 0::numeric)                                                                AS price_target_revision_1m,
       ("Price Target" - "Price Target (3M Ago)") /
       NULLIF("Price Target (3M Ago)", 0::numeric)                                                                AS price_target_revision_3m,
       COALESCE("EPS Est Avg Rev % (FY1E - 1W)", 0::numeric) * 0.30 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1M)", 0::numeric) * 0.25 +
       COALESCE("EPS Est Avg Rev % (FY1E - 3M)", 0::numeric) * 0.20 +
       COALESCE("EPS Est Avg Rev % (FY1E - 6M)", 0::numeric) * 0.15 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1Y)", 0::numeric) *
       0.10                                                                                                       AS eps_revision_momentum,
       ("Analyst Rating" - 1::numeric) * 25::numeric                                                              AS analyst_rating_normalized,
       "Price Target - #" /
       NULLIF(ln(1::numeric + "Market Cap"), 0::numeric)                                                          AS analyst_coverage_quality,
       ("Price Target" - "Price Target (1W Ago)") /
       NULLIF("Price Target (1W Ago)", 0::numeric)                                                                AS pt_momentum_1w,
       ("Price Target" - "Price Target (1M Ago)") /
       NULLIF("Price Target (1M Ago)", 0::numeric)                                                                AS pt_momentum_1m,
       ("Price Target" - "Price Target (3M Ago)") /
       NULLIF("Price Target (3M Ago)", 0::numeric)                                                                AS pt_momentum_3m,
       ("Price Target" - "Price Target (6M Ago)") /
       NULLIF("Price Target (6M Ago)", 0::numeric)                                                                AS pt_momentum_6m,
       ("Price Target" - "Price Target (1Y Ago)") /
       NULLIF("Price Target (1Y Ago)", 0::numeric)                                                                AS pt_momentum_1y,
       ("Price Target - #" - "Price Target - # (1M Ago)")::integer                                                AS analyst_coverage_change_1m,
       ("Price Target - #" - "Price Target - # (3M Ago)")::integer                                                AS analyst_coverage_change_3m,
       ("Price Target - #" - "Price Target - # (1Y Ago)")::integer                                                AS analyst_coverage_change_1y,
       CASE
           WHEN abs("EPS Norm - Est Avg (FY1E)") > 0::numeric THEN
               ("EPS/Adj. (LTM)" - "EPS Norm - Est Avg (FY1E)") / NULLIF(abs("EPS Norm - Est Avg (FY1E)"), 0::numeric) *
               100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS eps_surprise_pct,
       CASE
           WHEN abs("Revenues - Est Avg (FY1E)") > 0::numeric THEN
               ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
               NULLIF(abs("Revenues - Est Avg (FY1E)"), 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS revenue_surprise_pct,
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0::numeric)                                             AS eps_adjustment_ratio,
       CASE
           WHEN abs("EPS Norm - Est Avg (FY1E)") > 0::numeric THEN
               ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") /
               NULLIF(abs("EPS Norm - Est Avg (FY1E)"), 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS gaap_adj_eps_gap_pct,
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0::numeric)                                                   AS ebitda_adjustment_ratio,
       CASE
           WHEN abs("Net EPS - Basic (-4FQFQ)") > 0::numeric THEN
               ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
               NULLIF(abs("Net EPS - Basic (-4FQFQ)"), 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS eps_quarterly_trend,
       CASE
           WHEN abs("Net EPS - Basic (-1FY)") > 0::numeric THEN
               ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") / NULLIF(abs("Net EPS - Basic (-1FY)"), 0::numeric) *
               100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS eps_yoy_growth,
       CASE
           WHEN abs("Net EPS - Basic (-1FQFQ)") > 0::numeric THEN
               ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
               NULLIF(abs("Net EPS - Basic (-1FQFQ)"), 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS eps_qoq_growth,
       CASE
           WHEN abs("Net EPS - Basic (-4FQFQ)") > 0::numeric THEN
               ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
               NULLIF(abs("Net EPS - Basic (-4FQFQ)"), 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS eps_yoy_quarterly,
       CASE
           WHEN "Net EPS - Basic (FQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-1FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-2FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-3FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-4FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS eps_positive_streak,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0::numeric AND "Net EPS - Basic (FY)" > 0::numeric THEN
               (power("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0::numeric), 1.0 / 3.0) - 1::numeric) *
               100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS eps_cagr_3y,
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0::numeric AND "Net EPS - Basic (FY)" > 0::numeric THEN
               (power("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0::numeric), 1.0 / 5.0) - 1::numeric) *
               100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS eps_cagr_5y,
       CASE
           WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1
           ELSE 0
           END                                                                                                    AS eps_improvement_count,
       (
           CASE
               WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1
               ELSE 0
               END +
           CASE
               WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1
               ELSE 0
               END +
           CASE
               WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1
               ELSE 0
               END +
           CASE
               WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1
               ELSE 0
               END +
           CASE
               WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1
               ELSE 0
               END)::numeric / 5.0 *
       100::numeric                                                                                               AS eps_trajectory_score,
       "EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)"                                                                 AS eps_adjustment_spread_ltm,
       "EPS/Adj. (FY)" - "Net EPS - Basic (FY)"                                                                   AS eps_adjustment_spread_fy,
       "EPS/Adj. (-1FY)" - "Net EPS - Basic (-1FY)"                                                               AS eps_adjustment_spread_1fy,
       "EPS/Adj. (FQ)" - "Net EPS - Basic (FQ)"                                                                   AS eps_adjustment_spread_fq,
       "EPS/Adj. (-1FQFQ)" - "Net EPS - Basic (-1FQFQ)"                                                           AS eps_adjustment_spread_1fqfq,
       "EPS/Adj. (-2FQFQ)" - "Net EPS - Basic (-2FQFQ)"                                                           AS eps_adjustment_spread_2fqfq,
       "EPS/Adj. (-3FQFQ)" - "Net EPS - Basic (-3FQFQ)"                                                           AS eps_adjustment_spread_3fqfq,
       "EPS/Adj. (-4FQFQ)" - "Net EPS - Basic (-4FQFQ)"                                                           AS eps_adjustment_spread_4fqfq,
       "EPS/Adj. (-2FY)" - "Net EPS - Basic (-2FY)"                                                               AS eps_adjustment_spread_2fy,
       "EPS/Adj. (-3FY)" - "Net EPS - Basic (-3FY)"                                                               AS eps_adjustment_spread_3fy,
       "EPS/Adj. (-4FY)" - "Net EPS - Basic (-4FY)"                                                               AS eps_adjustment_spread_4fy,
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") / NULLIF(abs("Net EPS - Basic (LTM)"), 0::numeric) *
       100::numeric                                                                                               AS eps_adjustment_pct,
       "Net Income/Adj. (LTM)" /
       NULLIF("Net Income - (IS) (LTM)", 0::numeric)                                                              AS net_income_adjustment_ratio_ltm,
       "Net Income/Adj. (FY)" / NULLIF("Net Income - (IS) (FY)", 0::numeric)                                      AS net_income_adjustment_ratio_fy,
       "Net Income/Adj. (-1FY)" /
       NULLIF("Net Income - (IS) (-1FY)", 0::numeric)                                                             AS net_income_adjustment_ratio_1fy,
       "Net Income/Adj. (FQ)" / NULLIF("Net Income - (IS) (FQ)", 0::numeric)                                      AS net_income_adjustment_ratio_fq,
       "Net Income/Adj. (5YAVGFQ)" /
       NULLIF("Net Income - (IS) (5YAVGFQ)", 0::numeric)                                                          AS net_income_adjustment_ratio_5yavgfq,
       "Net Income/Adj. (-1FQFQ)" /
       NULLIF("Net Income - (IS) (-1FQFQ)", 0::numeric)                                                           AS net_income_adjustment_ratio_1fqfq,
       "Net Income/Adj. (-2FQFQ)" /
       NULLIF("Net Income - (IS) (-2FQFQ)", 0::numeric)                                                           AS net_income_adjustment_ratio_2fqfq,
       "Net Income/Adj. (-3FQFQ)" /
       NULLIF("Net Income - (IS) (-3FQFQ)", 0::numeric)                                                           AS net_income_adjustment_ratio_3fqfq,
       "Net Income/Adj. (-4FQFQ)" /
       NULLIF("Net Income - (IS) (-4FQFQ)", 0::numeric)                                                           AS net_income_adjustment_ratio_4fqfq,
       "Net Income/Adj. (-2FY)" /
       NULLIF("Net Income - (IS) (-2FY)", 0::numeric)                                                             AS net_income_adjustment_ratio_2fy,
       "Net Income/Adj. (-3FY)" /
       NULLIF("Net Income - (IS) (-3FY)", 0::numeric)                                                             AS net_income_adjustment_ratio_3fy,
       "Net Income/Adj. (-4FY)" /
       NULLIF("Net Income - (IS) (-4FY)", 0::numeric)                                                             AS net_income_adjustment_ratio_4fy,
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") / NULLIF(abs("Net Income - (IS) (LTM)"), 0::numeric) *
       100::numeric                                                                                               AS net_income_adjustment_pct,
       ("EBITDA/Adj. (LTM)" - "EBITDA (LTM)") / NULLIF(abs("EBITDA (LTM)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_ltm,
       ("EBITDA/Adj. (FY)" - "EBITDA (FY)") / NULLIF(abs("EBITDA (FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_fy,
       ("EBITDA/Adj. (-1FY)" - "EBITDA (-1FY)") / NULLIF(abs("EBITDA (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_1fy,
       ("EBITDA/Adj. (FQ)" - "EBITDA (FQ)") / NULLIF(abs("EBITDA (FQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_fq,
       ("EBITDA/Adj. (-1FQFQ)" - "EBITDA (-1FQFQ)") / NULLIF(abs("EBITDA (-1FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_1fqfq,
       ("EBITDA/Adj. (-2FQFQ)" - "EBITDA (-2FQFQ)") / NULLIF(abs("EBITDA (-2FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_2fqfq,
       ("EBITDA/Adj. (-3FQFQ)" - "EBITDA (-3FQFQ)") / NULLIF(abs("EBITDA (-3FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_3fqfq,
       ("EBITDA/Adj. (-4FQFQ)" - "EBITDA (-4FQFQ)") / NULLIF(abs("EBITDA (-4FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_4fqfq,
       ("EBITDA/Adj. (-2FY)" - "EBITDA (-2FY)") / NULLIF(abs("EBITDA (-2FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_2fy,
       ("EBITDA/Adj. (-3FY)" - "EBITDA (-3FY)") / NULLIF(abs("EBITDA (-3FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_3fy,
       ("EBITDA/Adj. (-4FY)" - "EBITDA (-4FY)") / NULLIF(abs("EBITDA (-4FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_adjustment_pct_4fy,
       ("EBIT/Adj. (LTM)" - "EBIT (LTM)") / NULLIF(abs("EBIT (LTM)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_ltm,
       ("EBIT/Adj. (FY)" - "EBIT (FY)") / NULLIF(abs("EBIT (FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_fy,
       ("EBIT/Adj. (-1FY)" - "EBIT (-1FY)") / NULLIF(abs("EBIT (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_1fy,
       ("EBIT/Adj. (FQ)" - "EBIT (FQ)") / NULLIF(abs("EBIT (FQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_fq,
       ("EBIT/Adj. (-1FQFQ)" - "EBIT (-1FQFQ)") / NULLIF(abs("EBIT (-1FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_1fqfq,
       ("EBIT/Adj. (-2FQFQ)" - "EBIT (-2FQFQ)") / NULLIF(abs("EBIT (-2FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_2fqfq,
       ("EBIT/Adj. (-3FQFQ)" - "EBIT (-3FQFQ)") / NULLIF(abs("EBIT (-3FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_3fqfq,
       ("EBIT/Adj. (-4FQFQ)" - "EBIT (-4FQFQ)") / NULLIF(abs("EBIT (-4FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_4fqfq,
       ("EBIT/Adj. (-2FY)" - "EBIT (-2FY)") / NULLIF(abs("EBIT (-2FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_2fy,
       ("EBIT/Adj. (-3FY)" - "EBIT (-3FY)") / NULLIF(abs("EBIT (-3FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_3fy,
       ("EBIT/Adj. (-4FY)" - "EBIT (-4FY)") / NULLIF(abs("EBIT (-4FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_adjustment_pct_4fy,
       GREATEST(0::numeric, LEAST(100::numeric, 100::numeric - abs(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                                                                   NULLIF(abs("Net EPS - Basic (LTM)"), 0::numeric) *
                                                                   100::numeric)))                                AS earnings_quality_score,
       CASE
           WHEN abs(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") / NULLIF(abs("Net EPS - Basic (LTM)"), 0::numeric) *
                    100::numeric) > 15::numeric THEN 1
           ELSE 0
           END                                                                                                    AS earnings_quality_warning,
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"                                                  AS forward_eps_gaap_adj_spread,
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1M)", 0::numeric) * 0.35 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 3M)", 0::numeric) * 0.30 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 6M)", 0::numeric) * 0.20 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0::numeric) *
       0.15                                                                                                       AS gaap_revision_momentum,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)"                                                                       AS gaap_revision_1m,
       "EPS GAAP Est Avg Rev % (FY1E - 3M)"                                                                       AS gaap_revision_3m,
       "EPS GAAP Est Avg Rev % (FY1E - 6M)"                                                                       AS gaap_revision_6m,
       "EPS GAAP Est Avg Rev % (FY1E - 1Y)"                                                                       AS gaap_revision_1y,
       "EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)"                                     AS gaap_vs_norm_revision_spread,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)" -
       "EPS GAAP Est Avg Rev % (FY1E - 6M)"                                                                       AS gaap_revision_acceleration,
       CASE
           WHEN "EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0::numeric AND
                "EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0::numeric AND "EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0::numeric
               THEN 1
           ELSE 0
           END                                                                                                    AS gaap_positive_revision_flag,
       CASE
           WHEN abs("Total Revenues (-1FY)") > 0::numeric THEN
               ("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(abs("Total Revenues (-1FY)"), 0::numeric) *
               100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS revenue_growth_yoy,
       CASE
           WHEN abs("EBITDA (-1FY)") > 0::numeric THEN
               ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(abs("EBITDA (-1FY)"), 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS ebitda_growth_yoy,
       CASE
           WHEN abs("Operating Income (FY)") > 0::numeric THEN
               ("Operating Income (LTM)" - "Operating Income (FY)") / NULLIF(abs("Operating Income (FY)"), 0::numeric) *
               100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS operating_income_growth,
       CASE
           WHEN abs("FCF (FY)") > 0::numeric THEN ("FCF (LTM)" - "FCF (FY)") / NULLIF(abs("FCF (FY)"), 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS fcf_growth,
       "Total Revenues/CAGR (5Y FY)"                                                                              AS revenue_cagr_5y,
       "Revenues - Est YoY % (FY1E)"                                                                              AS forward_revenue_growth,
       "Total Revenues (LTM)" /
       NULLIF("Total Revenues (5YAVGLTM)", 0::numeric)                                                            AS revenue_vs_5y_avg,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") / NULLIF("Revenues - Est Med (FY1E)", 0::numeric) *
       100::numeric                                                                                               AS revenue_est_spread,
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") / NULLIF(abs("Revenues - Est Avg (FY1E)"), 0::numeric) *
       100::numeric                                                                                               AS revenue_beat_potential,
       "Revenues - Est YoY % (FY1E)"                                                                              AS revenue_est_revision_trend,
       ("EBITDA (LTM)" - "EBITDA - Est Avg (FY1E)") / NULLIF(abs("EBITDA - Est Avg (FY1E)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_est_vs_actual,
       "Enterprise Value" / NULLIF("Revenues - Est Avg (FY1E)", 0::numeric)                                       AS forward_revenue_multiple,
       "EPS Norm - Est # (FY1E)"                                                                                  AS revenue_estimate_count,
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(abs("Revenues - Est Avg (FY1E)"), 0::numeric) *
       100::numeric                                                                                               AS revenue_guidance_gap,
       ("Revenues - Est Avg (FY1E)" - "Total Revenues (FY)") / NULLIF(abs("Total Revenues (FY)"), 0::numeric) *
       100::numeric                                                                                               AS consensus_revenue_growth,
       "EBITDA - Est Avg (FY1E)" / NULLIF("Revenues - Est Avg (FY1E)", 0::numeric) *
       100::numeric                                                                                               AS forward_ebitda_margin,
       "Revenues - Est YoY % (FY1E)" - "Total Revenues/CAGR (5Y FY)"                                              AS revenue_acceleration,
       GREATEST(0::numeric, LEAST(100::numeric, 100::numeric -
                                                abs(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                                    NULLIF("Revenues - Est Med (FY1E)", 0::numeric) *
                                                    100::numeric)))                                               AS estimate_confidence_score,
       "Dividend Streak"::integer                                                                                 AS dividend_streak,
       "Div Yield (LTM)"                                                                                          AS dividend_yield_ltm,
       "Div Yield (NTM)"                                                                                          AS dividend_yield_ntm,
       abs("Common Dividends Paid (LTM)") /
       NULLIF("Net Income/Adj. (LTM)", 0::numeric)                                                                AS dividend_payout_ratio,
       CASE
           WHEN abs("Common Dividends Paid (LTM)") > 0::numeric
               THEN "FCF (LTM)" / NULLIF(abs("Common Dividends Paid (LTM)"), 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS fcf_dividend_coverage,
       "Buyback Yield (LTM)"                                                                                      AS buyback_yield,
       COALESCE("Buyback Yield (LTM)", 0::numeric) +
       COALESCE("Div Yield (LTM)", 0::numeric)                                                                    AS total_shareholder_yield,
       "Div Yield (NTM)" - "Div Yield (LTM)"                                                                      AS dividend_growth_expectation,
       CURRENT_DATE - "Dividend Record (Ex Date)"                                                                 AS days_since_ex_date,
       "Dividend Record (Payable Date)" - CURRENT_DATE                                                            AS days_to_payment,
       CASE
           WHEN (CURRENT_DATE - "Dividend Record (Announce Date)") <= 30 THEN 1
           ELSE 0
           END                                                                                                    AS dividend_announced_flag,
       CASE
           WHEN ("Dividend Record (Ex Date)" - CURRENT_DATE) >= 0 AND ("Dividend Record (Ex Date)" - CURRENT_DATE) <= 14
               THEN 1
           ELSE 0
           END                                                                                                    AS ex_date_approaching_flag,
       CASE "Dividend Record (Frequency)"
           WHEN 'Quarterly'::text THEN 4
           WHEN 'Semi-Annual'::text THEN 2
           WHEN 'Annual'::text THEN 1
           WHEN 'Monthly'::text THEN 12
           ELSE 0
           END                                                                                                    AS dividend_frequency_score,
       LEAST(1.0, "Dividend Streak" / 10.0)                                                                       AS dividend_consistency,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0::numeric)                                             AS dividend_yield_vs_5y_avg,
       CASE
           WHEN "Full Time Employees (FY)" > 0::numeric
               THEN "Total Revenues (FY)" / NULLIF("Full Time Employees (FY)", 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS revenue_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0::numeric THEN "Normalized Net Income (FY)" /
                                                             NULLIF("Full Time Employees (FY)", 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS profit_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0::numeric
               THEN "EBITDA (FY)" / NULLIF("Full Time Employees (FY)", 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS ebitda_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0::numeric
               THEN "Total Assets (FY)" / NULLIF("Full Time Employees (FY)", 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS assets_per_employee,
       CASE
           WHEN "Full Time Employees (-1FY)" > 0::numeric THEN
               ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
               NULLIF("Full Time Employees (-1FY)", 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS fte_growth_1y_pct,
       CASE
           WHEN "Full Time Employees (-3FY)" > 0::numeric THEN
               ("Full Time Employees (FY)" - "Full Time Employees (-3FY)") /
               NULLIF("Full Time Employees (-3FY)", 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS fte_growth_3y_pct,
       CASE
           WHEN "Avg Employees (5YAVGFY)" > 0::numeric THEN "Full Time Employees (FY)" /
                                                            NULLIF("Avg Employees (5YAVGFY)", 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS workforce_stability,
       CASE
           WHEN "Full Time Employees (-2FY)" > 0::numeric THEN
               ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
               NULLIF("Full Time Employees (-2FY)", 0::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS fte_growth_2y_pct,
       CASE
           WHEN "Full Time Employees (FY)" < "Full Time Employees (-1FY)" AND
                "Total Revenues (FY)" < "Total Revenues (-1FY)" THEN 1
           ELSE 0
           END                                                                                                    AS layoff_risk_flag,
       CASE
           WHEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                 NULLIF("Full Time Employees (-1FY)", 0::numeric)) > 0.20 THEN 1
           ELSE 0
           END                                                                                                    AS rapid_hiring_flag,
       CASE
           WHEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(abs("Total Revenues (-1FY)"), 0::numeric)) >
                (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                 NULLIF("Full Time Employees (-1FY)", 0::numeric)) AND
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS sustainable_growth_flag,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0::numeric)                                                AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0::numeric)                                                AS fcf_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric)                                                   AS fcf_margin,
       ("CFO (LTM)" - "CFO (-1FY)") / NULLIF("CFO (-1FY)", 0::numeric)                                            AS cfo_growth_yoy,
       (
           CASE
               WHEN "FCF (FQ)" > 0::numeric THEN 1
               ELSE 0
               END +
           CASE
               WHEN "FCF (-1FQFQ)" > 0::numeric THEN 1
               ELSE 0
               END +
           CASE
               WHEN "FCF (-2FQFQ)" > 0::numeric THEN 1
               ELSE 0
               END +
           CASE
               WHEN "FCF (-3FQFQ)" > 0::numeric THEN 1
               ELSE 0
               END +
           CASE
               WHEN "FCF (-4FQFQ)" > 0::numeric THEN 1
               ELSE 0
               END)::numeric /
       5.0                                                                                                        AS fcf_positive_ratio,
       abs(COALESCE("Cash Acquisitions (FQ)", 0::numeric)) + abs(COALESCE("Cash Acquisitions (-1FQFQ)", 0::numeric)) +
       abs(COALESCE("Cash Acquisitions (-2FQFQ)", 0::numeric)) +
       abs(COALESCE("Cash Acquisitions (-3FQFQ)", 0::numeric))                                                    AS acquisition_intensity,
       CASE
           WHEN abs("CFI (LTM)") > 0::numeric THEN "CFO (LTM)" / NULLIF(abs("CFI (LTM)"), 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS self_funding_ratio,
       CASE
           WHEN "FCF (FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "FCF (-1FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "FCF (-2FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "FCF (-3FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "FCF (-4FY)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS fcf_positive_years,
       CASE
           WHEN "FCF (FY)" > 0::numeric AND "FCF (-1FY)" > 0::numeric AND "FCF (-2FY)" > 0::numeric AND
                "FCF (-3FY)" > 0::numeric AND "FCF (-4FY)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS fcf_always_positive,
       abs("Capital Expenditure (FQ)") /
       NULLIF(abs("Capital Expenditure (5YAVGFQ)"), 0::numeric)                                                   AS capex_vs_5y_avg,
       CASE
           WHEN (abs("Capital Expenditure (FQ)") / NULLIF(abs("Capital Expenditure (5YAVGFQ)"), 0::numeric)) < 0.7
               THEN 1
           ELSE 0
           END                                                                                                    AS underinvestment_flag,
       abs("CFO (LTM)") / NULLIF(abs("CFO (LTM)") + abs("CFI (LTM)") + abs("CFF (LTM)"),
                                 0::numeric)                                                                      AS cfo_share_of_cf,
       abs("CFI (LTM)") / NULLIF(abs("CFO (LTM)") + abs("CFI (LTM)") + abs("CFF (LTM)"),
                                 0::numeric)                                                                      AS cfi_share_of_cf,
       abs("CFF (LTM)") / NULLIF(abs("CFO (LTM)") + abs("CFI (LTM)") + abs("CFF (LTM)"),
                                 0::numeric)                                                                      AS cff_share_of_cf,
       CASE
           WHEN ("CFO (LTM)" / NULLIF(abs("CFI (LTM)"), 0::numeric)) > 1::numeric THEN 1
           ELSE 0
           END                                                                                                    AS self_funding_flag,
       ("FCF (FQ)" - "FCF (-4FQFQ)") /
       NULLIF(abs("FCF (-4FQFQ)"), 0::numeric)                                                                    AS fcf_4q_improvement,
       (
           CASE
               WHEN ("CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0::numeric)) > 1::numeric THEN 25
               ELSE 0
               END +
           CASE
               WHEN "FCF (FY)" > 0::numeric AND "FCF (-1FY)" > 0::numeric AND "FCF (-2FY)" > 0::numeric AND
                    "FCF (-3FY)" > 0::numeric AND "FCF (-4FY)" > 0::numeric THEN 25
               ELSE 0
               END +
           CASE
               WHEN "CFO (LTM)" > abs("CFI (LTM)") THEN 25
               ELSE 0
               END +
           CASE
               WHEN "FCF (LTM)" > 0::numeric THEN 25
               ELSE 0
               END)::numeric                                                                                      AS cash_flow_quality_score,
       (abs(COALESCE("Cash Acquisitions (FQ)", 0::numeric)) + abs(COALESCE("Cash Acquisitions (-1FQFQ)", 0::numeric)) +
        abs(COALESCE("Cash Acquisitions (-2FQFQ)", 0::numeric)) +
        abs(COALESCE("Cash Acquisitions (-3FQFQ)", 0::numeric))) /
       NULLIF(abs("FCF (LTM)"), 0::numeric)                                                                       AS acquisition_to_fcf,
       CASE
           WHEN ((abs(COALESCE("Cash Acquisitions (FQ)", 0::numeric)) +
                  abs(COALESCE("Cash Acquisitions (-1FQFQ)", 0::numeric)) +
                  abs(COALESCE("Cash Acquisitions (-2FQFQ)", 0::numeric)) +
                  abs(COALESCE("Cash Acquisitions (-3FQFQ)", 0::numeric))) / NULLIF(abs("FCF (LTM)"), 0::numeric)) < 0.5
               THEN 1
           ELSE 0
           END                                                                                                    AS sustainable_ma_flag,
       (abs("Capital Expenditure (FY)") - abs("Capital Expenditure (-1FY)")) /
       NULLIF(abs("Capital Expenditure (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS capex_yoy_growth,
       (abs("Capital Expenditure (FQ)") - abs("Capital Expenditure (-1FQFQ)")) /
       NULLIF(abs("Capital Expenditure (-1FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS capex_qoq_growth,
       (abs("Capital Expenditure (FY)") - abs("Capital Expenditure (-3FY)")) /
       NULLIF(abs("Capital Expenditure (-3FY)"), 0::numeric) *
       100::numeric                                                                                               AS capex_3y_trend,
       (abs(abs("Capital Expenditure (FQ)") - abs("Capital Expenditure (-1FQFQ)")) +
        abs(abs("Capital Expenditure (-1FQFQ)") - abs("Capital Expenditure (-2FQFQ)")) +
        abs(abs("Capital Expenditure (-2FQFQ)") - abs("Capital Expenditure (-3FQFQ)")) +
        abs(abs("Capital Expenditure (-3FQFQ)") - abs("Capital Expenditure (-4FQFQ)"))) / NULLIF(
               (abs("Capital Expenditure (FQ)") + abs("Capital Expenditure (-1FQFQ)") +
                abs("Capital Expenditure (-2FQFQ)") + abs("Capital Expenditure (-3FQFQ)") +
                abs("Capital Expenditure (-4FQFQ)")) / 5.0,
               0::numeric)                                                                                        AS capex_volatility,
       CASE
           WHEN abs("Capital Expenditure (FY)") > abs("Capital Expenditure (-1FY)") AND
                abs("Capital Expenditure (-1FY)") > abs("Capital Expenditure (-2FY)") THEN 1
           ELSE 0
           END                                                                                                    AS capex_acceleration,
       CASE
           WHEN ((abs("Capital Expenditure (FY)") - abs("Capital Expenditure (-1FY)")) /
                 NULLIF(abs("Capital Expenditure (-1FY)"), 0::numeric)) < '-0.25'::numeric THEN 1
           ELSE 0
           END                                                                                                    AS capex_cut_flag,
       CASE
           WHEN (abs("Capital Expenditure (FQ)") / NULLIF(abs("Capital Expenditure (5YAVGFQ)"), 0::numeric)) > 1.5
               THEN 1
           ELSE 0
           END                                                                                                    AS overinvestment_flag,
       (abs(COALESCE("Cash Acquisitions (FY)", 0::numeric)) - abs(COALESCE("Cash Acquisitions (-1FY)", 0::numeric))) /
       NULLIF(abs(COALESCE("Cash Acquisitions (-1FY)", 0::numeric)), 0::numeric) *
       100::numeric                                                                                               AS acquisitions_yoy_growth,
       abs(COALESCE("Cash Acquisitions (FQ)", 0::numeric)) /
       NULLIF(abs(COALESCE("Cash Acquisitions (5YAVGFQ)", 0::numeric)),
              0::numeric)                                                                                         AS acquisitions_vs_5y_avg,
       abs(COALESCE("Cash Acquisitions (LTM)", 0::numeric))                                                       AS acquisitions_ltm_total,
       abs(COALESCE("Cash Acquisitions (LTM)", 0::numeric)) / NULLIF("Total Assets (LTM)", 0::numeric) *
       100::numeric                                                                                               AS ma_intensity_score,
       CASE
           WHEN (
                    CASE
                        WHEN abs(COALESCE("Cash Acquisitions (FY)", 0::numeric)) > 0::numeric THEN 1
                        ELSE 0
                        END +
                    CASE
                        WHEN abs(COALESCE("Cash Acquisitions (-1FY)", 0::numeric)) > 0::numeric THEN 1
                        ELSE 0
                        END +
                    CASE
                        WHEN abs(COALESCE("Cash Acquisitions (-2FY)", 0::numeric)) > 0::numeric THEN 1
                        ELSE 0
                        END +
                    CASE
                        WHEN abs(COALESCE("Cash Acquisitions (-3FY)", 0::numeric)) > 0::numeric THEN 1
                        ELSE 0
                        END) >= 3 THEN 1
           ELSE 0
           END                                                                                                    AS serial_acquirer_flag,
       CASE
           WHEN abs(COALESCE("Cash Acquisitions (FY)", 0::numeric)) = 0::numeric AND
                (abs(COALESCE("Cash Acquisitions (-1FY)", 0::numeric)) > 0::numeric OR
                 abs(COALESCE("Cash Acquisitions (-2FY)", 0::numeric)) > 0::numeric) THEN 1
           ELSE 0
           END                                                                                                    AS acquisition_pause_flag,
       (abs(COALESCE("Capital Expenditure (LTM)", 0::numeric)) + abs(COALESCE("Cash Acquisitions (LTM)", 0::numeric))) /
       NULLIF(abs("CFO (LTM)"), 0::numeric)                                                                       AS total_investment_to_cfo,
       abs(COALESCE("Capital Expenditure (LTM)", 0::numeric)) /
       NULLIF(abs(COALESCE("Cash Acquisitions (LTM)", 0::numeric)),
              0::numeric)                                                                                         AS organic_vs_inorganic,
       CASE
           WHEN (abs(COALESCE("Capital Expenditure (-1FY)", 0::numeric)) +
                 abs(COALESCE("Cash Acquisitions (-1FY)", 0::numeric))) > 0::numeric THEN
               ("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(
                       abs(COALESCE("Capital Expenditure (-1FY)", 0::numeric)) +
                       abs(COALESCE("Cash Acquisitions (-1FY)", 0::numeric)), 0::numeric)
           ELSE NULL::numeric
           END                                                                                                    AS investment_efficiency,
       ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(abs("CFO (-4FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS cfo_quarterly_trend,
       ("CFI (FQ)" - "CFI (-4FQFQ)") / NULLIF(abs("CFI (-4FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS cfi_quarterly_trend,
       ("CFF (FQ)" - "CFF (-4FQFQ)") / NULLIF(abs("CFF (-4FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS cff_quarterly_trend,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(abs("FCF (-4FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS fcf_quarterly_trend,
       CASE
           WHEN "CFO (FQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-1FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-2FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-3FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-4FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS cfo_positive_quarters,
       CASE
           WHEN "CFI (FQ)" < 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFI (-1FQFQ)" < 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFI (-2FQFQ)" < 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFI (-3FQFQ)" < 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFI (-4FQFQ)" < 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS cfi_negative_quarters,
       CASE
           WHEN "FCF (LTM)" < 0::numeric THEN abs("FCF (LTM)") / NULLIF("Cash And Equivalents (FQ)", 0::numeric) / 12.0
           ELSE 0::numeric
           END                                                                                                    AS cash_burn_rate,
       abs("CFF (LTM)") / NULLIF(abs("CFO (LTM)"), 0::numeric)                                                    AS financing_dependency,
       "Fiscal Quarter"                                                                                           AS fiscal_quarter,
       "Fiscal Month"                                                                                             AS fiscal_month,
       "Fiscal Year"                                                                                              AS fiscal_year,
       "Next Earnings" - CURRENT_DATE                                                                             AS days_to_earnings,
       CURRENT_DATE - "Income Statement Report Date"                                                              AS earnings_report_recency,
       "Reporting Lag"                                                                                            AS reporting_lag,
       "Fiscal Month"::numeric / 12.0                                                                             AS fiscal_year_progress,
       CURRENT_DATE - "Income Statement Report Date"                                                              AS days_since_last_report,
       "FY End Date" - CURRENT_DATE                                                                               AS days_to_fy_end,
       CASE
           WHEN EXTRACT(month FROM CURRENT_DATE) = ANY (ARRAY [3::numeric, 6::numeric, 9::numeric, 12::numeric]) THEN 1
           ELSE 0
           END                                                                                                    AS is_quarter_end_month,
       CASE
           WHEN EXTRACT(month FROM CURRENT_DATE) = EXTRACT(month FROM "FY End Date") THEN 1
           ELSE 0
           END                                                                                                    AS is_fy_end_month,
       CASE
           WHEN EXTRACT(month FROM CURRENT_DATE) = ANY
                (ARRAY [1::numeric, 2::numeric, 4::numeric, 5::numeric, 7::numeric, 8::numeric, 10::numeric, 11::numeric])
               THEN 1
           ELSE 0
           END                                                                                                    AS earnings_season_flag,
       CASE
           WHEN ("Next Earnings" - CURRENT_DATE) >= 0 AND ("Next Earnings" - CURRENT_DATE) <= 14 THEN 1
           ELSE 0
           END                                                                                                    AS pre_earnings_window,
       CASE
           WHEN (CURRENT_DATE - "Income Statement Report Date") >= 0 AND
                (CURRENT_DATE - "Income Statement Report Date") <= 7 THEN 1
           ELSE 0
           END                                                                                                    AS post_earnings_window,
       GREATEST(0::numeric, LEAST(100::numeric, 100::numeric -
                                                (CURRENT_DATE - "Income Statement Report Date")::numeric / 90.0 *
                                                100::numeric))                                                    AS reporting_freshness_score,
       CASE
           WHEN "Return on Assets (ROA) % (LTM)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (LTM)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Return on Assets (ROA) % (LTM)" > "Return on Assets (ROA) % (FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (LTM)" > "Net Income - (IS) (LTM)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN ("Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0::numeric)) <
                ("Total Debt (FY)" / NULLIF("Total Equity (FY)", 0::numeric)) THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Current Ratio (LTM)" > "Current Ratio (FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Shrs Out" <= "Shrs Out (-1FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Asset Turnover (LTM)" > "Asset Turnover (FY)" THEN 1
           ELSE 0
           END                                                                                                    AS piotroski_f_score,
       GREATEST(0::numeric, LEAST(100::numeric, 50::numeric - ("Shrs Out" - "Shrs Out (-1FY)") /
                                                              NULLIF("Shrs Out (-1FY)", 0::numeric) *
                                                              100::numeric))                                      AS dilution_score,
       "EBIT (FQ)"                                                                                                AS ebit_fq,
       "EBIT (LTM)"                                                                                               AS ebit_ltm,
       "EBIT (FY)"                                                                                                AS ebit_fy,
       "EBIT (-1FY)"                                                                                              AS ebit_1fy,
       "EBITDA (FQ)"                                                                                              AS ebitda_fq,
       "EBITDA (LTM)"                                                                                             AS ebitda_ltm,
       "EBITDA (FY)"                                                                                              AS ebitda_fy,
       "EBITDA (-1FY)"                                                                                            AS ebitda_1fy,
       "EBIT (-2FY)"                                                                                              AS ebit_2fy,
       "EBIT (-3FY)"                                                                                              AS ebit_3fy,
       "EBIT (-4FY)"                                                                                              AS ebit_4fy,
       "EBITDA (-2FY)"                                                                                            AS ebitda_2fy,
       "EBITDA (-3FY)"                                                                                            AS ebitda_3fy,
       "EBITDA (-4FY)"                                                                                            AS ebitda_4fy,
       "EBIT (-1FQFQ)"                                                                                            AS ebit_1fqfq,
       "EBIT (-2FQFQ)"                                                                                            AS ebit_2fqfq,
       "EBIT (-3FQFQ)"                                                                                            AS ebit_3fqfq,
       "EBIT (-4FQFQ)"                                                                                            AS ebit_4fqfq,
       "EBITDA (-1FQFQ)"                                                                                          AS ebitda_1fqfq,
       "EBITDA (-2FQFQ)"                                                                                          AS ebitda_2fqfq,
       "EBITDA (-3FQFQ)"                                                                                          AS ebitda_3fqfq,
       "EBITDA (-4FQFQ)"                                                                                          AS ebitda_4fqfq,
       "EBIT (5YAVGFQ)"                                                                                           AS ebit_5yavgfq,
       "EBIT (5YAVGLTM)"                                                                                          AS ebit_5yavgltm,
       "EBITDA (5YAVGFQ)"                                                                                         AS ebitda_5yavgfq,
       "EBITDA (5YAVGLTM)"                                                                                        AS ebitda_5yavgltm,
       "EBIT/Adj. (FQ)"                                                                                           AS ebit_adj_fq,
       "EBIT/Adj. (LTM)"                                                                                          AS ebit_adj_ltm,
       "EBIT/Adj. (FY)"                                                                                           AS ebit_adj_fy,
       "EBITDA/Adj. (FQ)"                                                                                         AS ebitda_adj_fq,
       "EBITDA/Adj. (LTM)"                                                                                        AS ebitda_adj_ltm,
       "EBITDA/Adj. (FY)"                                                                                         AS ebitda_adj_fy,
       ("EBIT (FY)" - "EBIT (-1FY)") / NULLIF(abs("EBIT (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_growth_yoy,
       ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(abs("EBITDA (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_growth_yoy_comp,
       "EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS ebit_margin_ltm,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS ebitda_margin_ltm,
       CASE
           WHEN "EBIT (FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBIT (-1FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBIT (-2FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBIT (-3FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBIT (-4FY)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS ebit_positive_years,
       CASE
           WHEN "EBITDA (FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBITDA (-1FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBITDA (-2FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBITDA (-3FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "EBITDA (-4FY)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS ebitda_positive_years,
       ("EBIT (FQ)" - "EBIT (-1FQFQ)") / NULLIF(abs("EBIT (-1FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebit_qoq_growth,
       ("EBITDA (FQ)" - "EBITDA (-1FQFQ)") / NULLIF(abs("EBITDA (-1FQFQ)"), 0::numeric) *
       100::numeric                                                                                               AS ebitda_qoq_growth,
       CASE
           WHEN "EBIT (-3FY)" > 0::numeric AND "EBIT (FY)" > 0::numeric THEN
               (power("EBIT (FY)" / NULLIF("EBIT (-3FY)", 0::numeric), 1.0 / 3.0) - 1::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS ebit_cagr_3y,
       CASE
           WHEN "EBITDA (-3FY)" > 0::numeric AND "EBITDA (FY)" > 0::numeric THEN
               (power("EBITDA (FY)" / NULLIF("EBITDA (-3FY)", 0::numeric), 1.0 / 3.0) - 1::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS ebitda_cagr_3y,
       "EBIT (LTM)" / NULLIF("EBIT (5YAVGLTM)", 0::numeric)                                                       AS ebit_vs_5y_avg,
       "EBITDA (LTM)" / NULLIF("EBITDA (5YAVGLTM)", 0::numeric)                                                   AS ebitda_vs_5y_avg,
       "Net Income - (IS) (FQ)"                                                                                   AS net_income_is_fq,
       "Net Income - (IS) (LTM)"                                                                                  AS net_income_is_ltm,
       "Net Income - (IS) (FY)"                                                                                   AS net_income_is_fy,
       "Net Income/Adj. (LTM)"                                                                                    AS net_income_adj_ltm,
       "Normalized Net Income (LTM)"                                                                              AS normalized_ni_ltm,
       "Net Income - (IS) (-1FQFQ)"                                                                               AS net_income_is_1fqfq,
       "Net Income - (IS) (-2FQFQ)"                                                                               AS net_income_is_2fqfq,
       "Net Income - (IS) (-3FQFQ)"                                                                               AS net_income_is_3fqfq,
       "Net Income - (IS) (-4FQFQ)"                                                                               AS net_income_is_4fqfq,
       "Net Income - (IS) (-1FY)"                                                                                 AS net_income_is_1fy,
       "Net Income - (IS) (-2FY)"                                                                                 AS net_income_is_2fy,
       "Net Income - (IS) (-3FY)"                                                                                 AS net_income_is_3fy,
       "Net Income - (IS) (-4FY)"                                                                                 AS net_income_is_4fy,
       "Net Income - (IS) (5YAVGFQ)"                                                                              AS net_income_is_5yavgfq,
       "Net Income - (IS) (5YAVGLTM)"                                                                             AS net_income_is_5yavgltm,
       "Normalized Net Income (5YAVGFQ)"                                                                          AS normalized_ni_5yavgfq,
       "Normalized Net Income (5YAVGLTM)"                                                                         AS normalized_ni_5yavgltm,
       pct_change("Net Income - (IS) (FY)", "Net Income - (IS) (-1FY)")                                           AS net_income_growth_yoy,
       "Net Income Margin % (LTM)"                                                                                AS net_income_margin_ltm,
       safe_divide("Net Income/Adj. (LTM)", "Net Income - (IS) (LTM)")                                            AS ni_adjustment_ratio,
       CASE
           WHEN "Net Income - (IS) (FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net Income - (IS) (-1FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net Income - (IS) (-2FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net Income - (IS) (-3FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Net Income - (IS) (-4FY)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS net_income_positive_years,
       clamp_score((50 +
                    CASE
                        WHEN "Net Income - (IS) (FY)" > 0::numeric THEN 10
                        ELSE '-10'::integer
                        END +
                    CASE
                        WHEN "Net Income - (IS) (-1FY)" > 0::numeric THEN 5
                        ELSE '-5'::integer
                        END +
                    CASE
                        WHEN "Net Income - (IS) (-2FY)" > 0::numeric THEN 5
                        ELSE '-5'::integer
                        END +
                    CASE
                        WHEN abs(safe_divide("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)",
                                             "Net Income - (IS) (LTM)")) < 0.10 THEN 15
                        ELSE '-15'::integer
                        END +
                    CASE
                        WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 10
                        ELSE '-5'::integer
                        END +
                    CASE
                        WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 5
                        ELSE '-5'::integer
                        END)::numeric)                                                                            AS earnings_quality_composite_comp,
       pct_change("Net Income - (IS) (FQ)", "Net Income - (IS) (-1FQFQ)")                                         AS net_income_qoq_growth,
       pct_change("Net Income - (IS) (FQ)", "Net Income - (IS) (-4FQFQ)")                                         AS net_income_yoy_quarterly,
       safe_divide("Net Income - (IS) (LTM)", "Net Income - (IS) (5YAVGLTM)")                                     AS net_income_vs_5y_avg,
       safe_divide("Normalized Net Income (LTM)",
                   "Normalized Net Income (5YAVGLTM)")                                                            AS normalized_ni_vs_5y_avg,
       "CFO (FQ)"                                                                                                 AS cfo_fq,
       "CFO (LTM)"                                                                                                AS cfo_ltm,
       "CFO (FY)"                                                                                                 AS cfo_fy,
       "FCF (FQ)"                                                                                                 AS fcf_fq,
       "FCF (LTM)"                                                                                                AS fcf_ltm,
       "FCF (FY)"                                                                                                 AS fcf_fy,
       ("CFO (FY)" - "CFO (-1FY)") / NULLIF(abs("CFO (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS cfo_growth_yoy_comp,
       ("FCF (FY)" - "FCF (-1FY)") / NULLIF(abs("FCF (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS fcf_growth_yoy,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS fcf_margin_pct,
       "FCF (LTM)" / NULLIF("Market Cap", 0::numeric) * 100::numeric                                              AS fcf_yield,
       CASE
           WHEN "CFO (FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-1FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-2FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-3FY)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "CFO (-4FY)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS cfo_positive_years,
       "Beta (1Y)"                                                                                                AS beta_1y,
       "Beta (5Y)"                                                                                                AS beta_5y,
       "Beta (1Y)" - "Beta (5Y)"                                                                                  AS beta_spread,
       ("Beta (1Y)" - "Beta (5Y)") / NULLIF(abs("Beta (5Y)"), 0::numeric) *
       100::numeric                                                                                               AS beta_trend,
       CASE
           WHEN "Beta (1Y)" > 1.5 THEN 1
           ELSE 0
           END                                                                                                    AS high_beta_flag,
       CASE
           WHEN "Beta (1Y)" < 0.5 THEN 1
           ELSE 0
           END                                                                                                    AS low_beta_flag,
       GREATEST(0::numeric, LEAST(100::numeric, 100::numeric - abs("Beta (1Y)" - "Beta (5Y)") *
                                                               50::numeric))                                      AS beta_stability_score,
       safe_divide("Cost Of Revenues (LTM)", "Total Revenues (LTM)") *
       100::numeric                                                                                               AS cogs_to_revenue,
       safe_divide("Total Operating Expenses (LTM)", "Total Revenues (LTM)") *
       100::numeric                                                                                               AS opex_to_revenue,
       safe_divide("Selling General & Admin Expenses/Total (FY)", "Total Revenues (FY)") *
       100::numeric                                                                                               AS sga_to_revenue,
       safe_divide("R&D Expenses (LTM)", "Total Revenues (LTM)") *
       100::numeric                                                                                               AS rnd_to_revenue,
       safe_divide("Interest Expense/Total (LTM)", "Total Revenues (LTM)") *
       100::numeric                                                                                               AS interest_to_revenue,
       "Interest Income On Investments (LTM)"                                                                     AS interest_income_ltm,
       "Interest Expense/Total (LTM)"                                                                             AS interest_expense_ltm,
       COALESCE("Interest Income On Investments (LTM)", 0::numeric) -
       COALESCE("Interest Expense/Total (LTM)", 0::numeric)                                                       AS net_interest_income,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0::numeric)                                          AS interest_coverage_ratio,
       "Interest Income On Investments (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS interest_income_to_revenue,
       "Interest Expense/Total (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS interest_expense_to_revenue,
       pct_change("Last Price", "Price (3Y Ago)")                                                                 AS price_momentum_3y,
       pct_change("Last Price", "Price (5Y Ago)")                                                                 AS price_momentum_5y,
       (COALESCE(pct_change("Last Price", "Price (1Y Ago)"), 0::numeric) * 0.50 +
        COALESCE(pct_change("Last Price", "Price (3Y Ago)"), 0::numeric) * 0.30 +
        COALESCE(pct_change("Last Price", "Price (5Y Ago)"), 0::numeric) * 0.20) /
       100::numeric                                                                                               AS long_term_trend_score,
       CASE
           WHEN calc_change_ratio("52W High/Adj" - "Last Price", "52W High/Adj") <= 0.10 AND
                calc_change_ratio("Last Price", "Price (3Y Ago)") > 0.5 THEN 1
           ELSE 0
           END                                                                                                    AS multi_year_high_flag,
       CASE
           WHEN calc_change_ratio("Last Price", "Price (3Y Ago)") > 0.20 AND
                calc_change_ratio("Last Price", "Price (1Y Ago)") > 0::numeric AND "EMA (50D)" > "EMA (250D)" THEN 1
           ELSE 0
           END                                                                                                    AS secular_trend_flag,
       "Total Equity (LTM)" - COALESCE("Goodwill (LTM)", 0::numeric) -
       COALESCE("Gross Intangible Assets (LTM)", 0::numeric)                                                      AS tangible_book_value,
       ("Total Equity (LTM)" - COALESCE("Goodwill (LTM)", 0::numeric) -
        COALESCE("Gross Intangible Assets (LTM)", 0::numeric)) /
       NULLIF("Shrs Out", 0::numeric)                                                                             AS tangible_book_per_share,
       "Last Price" * "Shrs Out" / NULLIF("Total Equity (LTM)" - COALESCE("Goodwill (LTM)", 0::numeric) -
                                          COALESCE("Gross Intangible Assets (LTM)", 0::numeric),
                                          0::numeric)                                                             AS price_to_tangible_book,
       ("Total Equity (LTM)" - COALESCE("Goodwill (LTM)", 0::numeric) -
        COALESCE("Gross Intangible Assets (LTM)", 0::numeric)) / NULLIF("Total Assets (LTM)", 0::numeric) *
       100::numeric                                                                                               AS tangible_equity_ratio,
       COALESCE("Gross Intangible Assets (LTM)", 0::numeric) / NULLIF("Total Equity (LTM)", 0::numeric) *
       100::numeric                                                                                               AS intangibles_to_equity,
       COALESCE("Goodwill (LTM)", 0::numeric) / NULLIF("Total Equity (LTM)", 0::numeric) *
       100::numeric                                                                                               AS goodwill_to_equity,
       "Working Capital (LTM)"                                                                                    AS working_capital_ltm,
       "Working Capital (FQ)"                                                                                     AS working_capital_fq,
       "Working Capital (FY)"                                                                                     AS working_capital_fy,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                                               AS wc_to_revenue,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric) *
       100::numeric                                                                                               AS wc_to_assets,
       ("Working Capital (FQ)" - "Working Capital (FY)") / NULLIF(abs("Working Capital (FY)"), 0::numeric) *
       100::numeric                                                                                               AS wc_change_qoq,
       ("Working Capital (FY)" - "Working Capital (-1FY)") / NULLIF(abs("Working Capital (-1FY)"), 0::numeric) *
       100::numeric                                                                                               AS wc_change_yoy,
       "Working Capital (LTM)" /
       NULLIF("Total Revenues (LTM)" / 365.0, 0::numeric)                                                         AS days_working_capital,
       CASE
           WHEN "Working Capital (LTM)" < 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS negative_wc_flag,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (FY)" AND "Working Capital (FY)" > "Working Capital (-1FY)"
               THEN 1
           ELSE 0
           END                                                                                                    AS wc_improvement_flag,
       "Other Unusual Items/Total (LTM)"                                                                          AS other_unusual_items_ltm,
       COALESCE("Other Unusual Items/Total (LTM)", 0::numeric) + COALESCE("Impairment of Goodwill (LTM)", 0::numeric) +
       COALESCE("Asset Writedown (LTM)", 0::numeric) +
       COALESCE("Restructuring Charges (LTM)", 0::numeric)                                                        AS total_unusual_items,
       safe_divide(abs(COALESCE("Other Unusual Items/Total (LTM)", 0::numeric) +
                       COALESCE("Impairment of Goodwill (LTM)", 0::numeric) +
                       COALESCE("Asset Writedown (LTM)", 0::numeric) +
                       COALESCE("Restructuring Charges (LTM)", 0::numeric)), "Total Revenues (LTM)") *
       100::numeric                                                                                               AS unusual_items_to_revenue,
       safe_divide(abs(COALESCE("Other Unusual Items/Total (LTM)", 0::numeric) +
                       COALESCE("Impairment of Goodwill (LTM)", 0::numeric) +
                       COALESCE("Asset Writedown (LTM)", 0::numeric) +
                       COALESCE("Restructuring Charges (LTM)", 0::numeric)), abs("EBITDA (LTM)")) *
       100::numeric                                                                                               AS unusual_items_to_ebitda,
       CASE
           WHEN (abs(COALESCE("Other Unusual Items/Total (LTM)", 0::numeric)) +
                 abs(COALESCE("Impairment of Goodwill (LTM)", 0::numeric)) +
                 abs(COALESCE("Asset Writedown (LTM)", 0::numeric)) +
                 abs(COALESCE("Restructuring Charges (LTM)", 0::numeric))) > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS has_unusual_items_flag,
       "Revenues - Est Avg (FY1E)"                                                                                AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"                                                                                AS revenue_est_med_fy1e,
       "Revenues - Est Avg (NTM)"                                                                                 AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"                                                                                 AS revenue_est_med_ntm,
       safe_divide("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)", "Revenues - Est Med (FY1E)") *
       100::numeric                                                                                               AS revenue_avg_med_diff_pct,
       clamp_score(100::numeric - abs(safe_divide("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)",
                                                  "Revenues - Est Med (FY1E)") * 100::numeric) *
                                  2::numeric)                                                                     AS revenue_consensus_strength,
       safe_divide("Revenues - Est Avg (FY1E)", "Total Revenues (LTM)")                                           AS revenue_vs_current,
       "Total Revenues (FQ)"                                                                                      AS revenue_fq,
       "Total Revenues (-1FQFQ)"                                                                                  AS revenue_1fq,
       "Total Revenues (-2FQFQ)"                                                                                  AS revenue_2fq,
       "Total Revenues (-3FQFQ)"                                                                                  AS revenue_3q,
       "Total Revenues (-4FQFQ)"                                                                                  AS revenue_4q,
       "Total Revenues (FY)"                                                                                      AS revenue_fy,
       "Total Revenues (-1FY)"                                                                                    AS revenue_1fy,
       "Total Revenues (-2FY)"                                                                                    AS revenue_2fy,
       "Total Revenues (-3FY)"                                                                                    AS revenue_3fy,
       "Total Revenues (-4FY)"                                                                                    AS revenue_4fy,
       "Total Revenues (LTM)"                                                                                     AS revenue_ltm,
       "Total Revenues (5YAVGLTM)"                                                                                AS revenue_5y_avg,
       pct_change("Total Revenues (FY)", "Total Revenues (-1FY)")                                                 AS revenue_yoy_growth,
       pct_change("Total Revenues (-1FY)", "Total Revenues (-2FY)")                                               AS revenue_1fy_vs_2fy,
       pct_change("Total Revenues (-2FY)", "Total Revenues (-3FY)")                                               AS revenue_2fy_vs_3fy,
       pct_change("Total Revenues (-3FY)", "Total Revenues (-4FY)")                                               AS revenue_3fy_vs_4fy,
       CASE
           WHEN "Total Revenues (FY)" > "Total Revenues (-1FY)" THEN 1
           ELSE 0
           END                                                                                                    AS revenue_growth_flag,
       "Total Revenues (5YAVGFQ)"                                                                                 AS revenue_5yavgfq,
       "Total Revenues (5YAVGLTM)"                                                                                AS revenue_5yavgltm,
       pct_change("Total Revenues (FY)", "Total Revenues (-1FY)")                                                 AS revenue_growth_yoy_temp,
       safe_divide("Total Revenues (FQ)", "Total Revenues (5YAVGFQ)")                                             AS revenue_vs_5y_avg_fq_temp,
       safe_divide("Total Revenues (LTM)", "Total Revenues (5YAVGLTM)")                                           AS revenue_vs_5y_avg_ltm_temp,
       safe_divide("Total Revenues (FQ)" - "Total Revenues (5YAVGFQ)", "Total Revenues (5YAVGFQ)") *
       100::numeric                                                                                               AS revenue_fq_vs_avg_temp,
       calc_change_ratio("Total Revenues (LTM)", "Total Revenues (-1FY)") *
       100::numeric                                                                                               AS revenue_momentum_temp,
       "Working Capital (FQ)"                                                                                     AS wc_fq_temp,
       "Working Capital (FY)"                                                                                     AS wc_fy_temp,
       "Working Capital (LTM)"                                                                                    AS wc_ltm_temp,
       "Working Capital (5YAVGFY)"                                                                                AS wc_5yavgfy,
       "Working Capital (-1FQ)"                                                                                   AS wc_1fq,
       "Working Capital (-2FQ)"                                                                                   AS wc_2fq,
       "Working Capital (-3FQ)"                                                                                   AS wc_3fq,
       "Working Capital (-4FQ)"                                                                                   AS wc_4fq,
       "Working Capital (-1FY)"                                                                                   AS wc_1fy,
       "Working Capital (-2FY)"                                                                                   AS wc_2fy,
       "Working Capital (-3FY)"                                                                                   AS wc_3fy,
       "Working Capital (-4FY)"                                                                                   AS wc_4fy,
       pct_change("Working Capital (FQ)", "Working Capital (-1FQ)")                                               AS wc_qoq_change,
       pct_change("Working Capital (FY)", "Working Capital (-1FY)")                                               AS wc_yoy_change,
       pct_change("Working Capital (FQ)", "Working Capital (-4FQ)")                                               AS wc_4q_trend,
       safe_divide("Working Capital (FQ)", "Working Capital (5YAVGFY)")                                           AS wc_vs_5y_avg_temp,
       CASE
           WHEN "Working Capital (FQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Working Capital (-1FQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Working Capital (-2FQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Working Capital (-3FQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Working Capital (-4FQ)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS wc_positive_quarters,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (-1FQ)" AND
                "Working Capital (-1FQ)" > "Working Capital (-2FQ)" THEN 1
           ELSE 0
           END                                                                                                    AS wc_improving_flag_temp,
       (abs("Working Capital (FQ)" - "Working Capital (-1FQ)") +
        abs("Working Capital (-1FQ)" - "Working Capital (-2FQ)") +
        abs("Working Capital (-2FQ)" - "Working Capital (-3FQ)") +
        abs("Working Capital (-3FQ)" - "Working Capital (-4FQ)")) / NULLIF(
               abs(("Working Capital (FQ)" + "Working Capital (-1FQ)" + "Working Capital (-2FQ)" +
                    "Working Capital (-3FQ)" + "Working Capital (-4FQ)") / 5.0),
               0::numeric)                                                                                        AS wc_volatility,
       "Total Debt (FQ)"                                                                                          AS debt_fq,
       "Total Debt (FY)"                                                                                          AS debt_fy,
       "Total Debt (LTM)"                                                                                         AS debt_ltm,
       "Total Debt (-1FQ)"                                                                                        AS debt_1fq,
       "Total Debt (-2FQ)"                                                                                        AS debt_2fq,
       "Total Debt (-3FQ)"                                                                                        AS debt_3fq,
       "Total Debt (-4FQ)"                                                                                        AS debt_4fq,
       "Total Debt (-1FY)"                                                                                        AS debt_1fy_temp,
       "Total Debt (-2FY)"                                                                                        AS debt_2fy,
       "Total Debt (-3FY)"                                                                                        AS debt_3fy,
       "Total Debt (-4FY)"                                                                                        AS debt_4fy,
       pct_change("Total Debt (FQ)", "Total Debt (-1FQ)")                                                         AS debt_qoq_change,
       pct_change("Total Debt (FY)", "Total Debt (-1FY)")                                                         AS debt_yoy_change,
       pct_change("Total Debt (FQ)", "Total Debt (-4FQ)")                                                         AS debt_4q_trend,
       CASE
           WHEN "Total Debt (-3FY)" > 0::numeric THEN
               (power(safe_divide("Total Debt (FY)", "Total Debt (-3FY)"), 1.0 / 3.0) - 1::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS debt_3y_cagr,
       CASE
           WHEN "Total Debt (FQ)" < "Total Debt (-1FQ)" AND "Total Debt (-1FQ)" < "Total Debt (-2FQ)" THEN 1
           ELSE 0
           END                                                                                                    AS debt_deleveraging,
       safe_divide("Total Debt (FY)", "Total Equity (FY)") - safe_divide("Total Debt (-1FY)",
                                                                         NULLIF("Total Equity (FY)", 0::numeric)) AS debt_to_equity_trend,
       "Total Assets (FQ)"                                                                                        AS assets_fq,
       "Total Assets (FY)"                                                                                        AS assets_fy,
       "Total Assets (LTM)"                                                                                       AS assets_ltm_temp,
       "Total Assets (-1FQ)"                                                                                      AS assets_1fq,
       "Total Assets (-2FQ)"                                                                                      AS assets_2fq,
       "Total Assets (-3FQ)"                                                                                      AS assets_3fq,
       "Total Assets (-4FQ)"                                                                                      AS assets_4fq,
       "Total Assets (-1FY)"                                                                                      AS assets_1fy,
       "Total Assets (-2FY)"                                                                                      AS assets_2fy,
       "Total Assets (-3FY)"                                                                                      AS assets_3fy,
       "Total Assets (-4FY)"                                                                                      AS assets_4fy,
       pct_change("Total Assets (FQ)", "Total Assets (-1FQ)")                                                     AS assets_qoq_growth,
       pct_change("Total Assets (FY)", "Total Assets (-1FY)")                                                     AS assets_yoy_growth,
       CASE
           WHEN "Total Assets (-3FY)" > 0::numeric THEN
               (power(safe_divide("Total Assets (FY)", "Total Assets (-3FY)"), 1.0 / 3.0) - 1::numeric) * 100::numeric
           ELSE NULL::numeric
           END                                                                                                    AS assets_3y_cagr,
       pct_change("Total Assets (FY)", "Total Assets (-1FY)") -
       pct_change("Total Assets (-1FY)", "Total Assets (-2FY)")                                                   AS asset_growth_accel,
       CASE
           WHEN "Total Assets (FY)" >= "Total Assets (-1FY)" AND "Total Assets (-1FY)" >= "Total Assets (-2FY)" AND
                "Total Assets (-2FY)" >= "Total Assets (-3FY)" THEN 1
           ELSE 0
           END                                                                                                    AS asset_base_stable,
       "Gross Profit (FQ)"                                                                                        AS gp_fq,
       "Gross Profit (FY)"                                                                                        AS gp_fy,
       "Gross Profit (LTM)"                                                                                       AS gp_ltm_temp,
       "Gross Profit (-1FQFQ)"                                                                                    AS gp_1fqfq,
       "Gross Profit (-2FQFQ)"                                                                                    AS gp_2fqfq,
       "Gross Profit (-3FQFQ)"                                                                                    AS gp_3fqfq,
       "Gross Profit (-4FQFQ)"                                                                                    AS gp_4fqfq,
       "Gross Profit (-1FY)"                                                                                      AS gp_1fy,
       "Gross Profit (-2FY)"                                                                                      AS gp_2fy,
       "Gross Profit (-3FY)"                                                                                      AS gp_3fy,
       "Gross Profit (-4FY)"                                                                                      AS gp_4fy,
       pct_change("Gross Profit (FQ)", "Gross Profit (-1FQFQ)")                                                   AS gp_qoq_growth,
       pct_change("Gross Profit (FY)", "Gross Profit (-1FY)")                                                     AS gp_yoy_growth,
       safe_divide("Gross Profit (FQ)", "Total Revenues (FQ)") * 100::numeric                                     AS gp_margin_fq,
       (safe_divide("Gross Profit (FQ)", "Total Revenues (FQ)") -
        safe_divide("Gross Profit (-4FQFQ)", "Total Revenues (5YAVGFQ)")) *
       100::numeric                                                                                               AS gp_margin_trend,
       CASE
           WHEN "Gross Profit (FQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Gross Profit (-1FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Gross Profit (-2FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Gross Profit (-3FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END +
       CASE
           WHEN "Gross Profit (-4FQFQ)" > 0::numeric THEN 1
           ELSE 0
           END                                                                                                    AS gp_positive_quarters,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" THEN 1
           ELSE 0
           END                                                                                                    AS gp_margin_expansion_temp,
       "Div Yield (Ind)"                                                                                          AS div_yield_ind,
       "Div Yield (-1FYInd)"                                                                                      AS div_yield_1fy_ind,
       "Div Yield (5YAVGLTM)"                                                                                     AS div_yield_5y_avg,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0::numeric)                                             AS div_yield_vs_5y_avg,
       ("Div Yield (NTM)" - "Div Yield (LTM)") / NULLIF("Div Yield (LTM)", 0::numeric) *
       100::numeric                                                                                               AS div_yield_growth_expected,
       CASE
           WHEN "Div Yield (LTM)" > 4::numeric THEN 1
           ELSE 0
           END                                                                                                    AS high_yield_flag,
       CASE
           WHEN "Div Yield (LTM)" > 0::numeric AND
                "FCF (LTM)" > abs(COALESCE("Common Dividends Paid (LTM)", 0::numeric)) AND "Dividend Streak" >= 5::numeric
               THEN 1
           ELSE 0
           END                                                                                                    AS sustainable_dividend_flag,
       "Basic EPS - Cont (LTM)"                                                                                   AS eps_cont_ltm,
       "Basic EPS - Cont (FQ)"                                                                                    AS eps_cont_fq,
       "Basic EPS - Cont (FY)"                                                                                    AS eps_cont_fy,
       "Basic EPS - Cont (-1FY)"                                                                                  AS eps_cont_1fy,
       "Basic EPS - Cont (-2FY)"                                                                                  AS eps_cont_2fy,
       "Basic EPS - Cont (-3FY)"                                                                                  AS eps_cont_3fy,
       "Basic EPS - Cont (-4FY)"                                                                                  AS eps_cont_4fy,
       pct_change("Basic EPS - Cont (FQ)", "Basic EPS - Cont (-1FQFQ)")                                           AS eps_cont_qoq_growth,
       pct_change("Basic EPS - Cont (FY)", "Basic EPS - Cont (-1FY)")                                             AS eps_cont_yoy_growth,
       safe_divide("Basic EPS - Cont (LTM)", "Net EPS - Basic (LTM)")                                             AS eps_cont_vs_total_eps,
       ("Net EPS - Basic (LTM)" - "Basic EPS - Cont (LTM)") / NULLIF(abs("Net EPS - Basic (LTM)"), 0::numeric) *
       100::numeric                                                                                               AS discontinued_ops_impact,
       "R&D Expenses (FQ)"                                                                                        AS rnd_fq,
       "R&D Expenses (FY)"                                                                                        AS rnd_fy,
       "R&D Expenses (-1FY)"                                                                                      AS rnd_1fy,
       "R&D Expenses (-2FY)"                                                                                      AS rnd_2fy,
       "R&D Expenses (-3FY)"                                                                                      AS rnd_3fy,
       "R&D Expenses (-4FY)"                                                                                      AS rnd_4fy,
       safe_divide("R&D Expenses (LTM)", "Total Revenues (LTM)") *
       100::numeric                                                                                               AS rnd_intensity_ltm,
       pct_change("R&D Expenses (FY)", "R&D Expenses (-1FY)")                                                     AS rnd_yoy_growth,
       safe_divide("R&D Expenses (FY)", "Full Time Employees (FY)")                                               AS rnd_per_employee,
       CASE
           WHEN safe_divide("R&D Expenses (LTM)", "Total Revenues (LTM)") > 0.10 THEN 1
           ELSE 0
           END                                                                                                    AS high_rnd_intensity_flag,
       "Inventory (-1FQ)"                                                                                         AS inventory_1fq,
       "Inventory (-2FQ)"                                                                                         AS inventory_2fq,
       "Inventory (-3FQ)"                                                                                         AS inventory_3fq,
       "Inventory (-4FQ)"                                                                                         AS inventory_4fq,
       "Inventory (-1FY)"                                                                                         AS inventory_1fy,
       "Inventory (-2FY)"                                                                                         AS inventory_2fy,
       "Inventory (-3FY)"                                                                                         AS inventory_3fy,
       "Inventory (-4FY)"                                                                                         AS inventory_4fy,
       pct_change("Inventory (FQ)", "Inventory (-1FQ)")                                                           AS inventory_qoq_change,
       pct_change("Inventory (FY)", "Inventory (-1FY)")                                                           AS inventory_yoy_change,
       "Inventory (LTM)" /
       NULLIF("Cost Of Revenues (LTM)" / 365.0, 0::numeric)                                                       AS inventory_days,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0::numeric)                                           AS inventory_turnover_mv,
       "Goodwill (-1FQ)"                                                                                          AS goodwill_1fq,
       "Goodwill (-2FQ)"                                                                                          AS goodwill_2fq,
       "Goodwill (-3FQ)"                                                                                          AS goodwill_3fq,
       "Goodwill (-4FQ)"                                                                                          AS goodwill_4fq,
       "Goodwill (-1FY)"                                                                                          AS goodwill_1fy,
       "Goodwill (-2FY)"                                                                                          AS goodwill_2fy,
       "Goodwill (-3FY)"                                                                                          AS goodwill_3fy,
       "Goodwill (-4FY)"                                                                                          AS goodwill_4fy,
       pct_change("Goodwill (FQ)", "Goodwill (-1FQ)")                                                             AS goodwill_qoq_change,
       pct_change("Goodwill (FY)", "Goodwill (-1FY)")                                                             AS goodwill_yoy_change,
       pct_change("Goodwill (FY)", "Goodwill (-3FY)")                                                             AS goodwill_3y_growth,
       safe_divide("Goodwill (LTM)", "Total Equity (LTM)") * 100::numeric                                         AS goodwill_concentration,
       CASE
           WHEN pct_change("Goodwill (FQ)", "Goodwill (-1FQ)") > 20::numeric THEN 1
           ELSE 0
           END                                                                                                    AS recent_acquisition_flag,
       safe_divide("Marketing Expenses (FY)", "Total Revenues (FY)") *
       100::numeric                                                                                               AS marketing_to_revenue,
       pct_change("Marketing Expenses (FY)", "Marketing Expenses (-1FY)")                                         AS marketing_trend_yoy,
       safe_divide("Marketing Expenses (FY)",
                   "Marketing Expenses (5YAVGLTM)")                                                               AS marketing_vs_5y_avg,
       safe_divide("Selling General & Admin Expenses/Total (FQ)",
                   "Selling General & Admin Expenses/Total (5YAVGFQ)")                                            AS sga_vs_5y_avg,
       "TBV (FY)"                                                                                                 AS tangible_book_value_fy,
       "TBV (LTM)"                                                                                                AS tangible_book_value_ltm,
       pct_change("TBV (LTM)", "TBV (FY)")                                                                        AS tbv_yoy_growth,
       CURRENT_TIMESTAMP                                                                                          AS calculated_at
FROM equities e;

alter materialized view mv_all_stock_features owner to postgres;

create index idx_mv_all_stock_features_isin
    on mv_all_stock_features (isin);

create index idx_mv_all_stock_features_ticker
    on mv_all_stock_features (ticker);

create index idx_mv_all_stock_features_piotroski
    on mv_all_stock_features (piotroski_f_score);

create index idx_mv_all_stock_features_quality
    on mv_all_stock_features (accounting_quality_score);

create index idx_mv_all_stock_features_momentum
    on mv_all_stock_features (price_momentum_1y);

create unique index idx_mv_all_stock_features_isin_unique
    on mv_all_stock_features (isin);

