create materialized view mv_interest_analysis as
SELECT "Ticker"                                                                AS ticker,
       "ISIN"                                                                  AS isin,
       "Name"                                                                  AS name,
       "Sector"                                                                AS sector,
       "Industry"                                                              AS industry,
       "Country"                                                               AS country,
       "Market Cap"                                                            AS market_cap,
       "Interest Income On Investments (LTM)"                                  AS interest_income_ltm,
       "Interest Expense/Total (LTM)"                                          AS interest_expense_ltm,
       "EBIT (LTM)"                                                            AS ebit_ltm,
       "EBITDA (LTM)"                                                          AS ebitda_ltm,
       "Total Assets (LTM)"                                                    AS total_assets_ltm,
       "Operating Income (LTM)"                                                AS operating_income_ltm,
       "Net Income - (IS) (LTM)"                                               AS net_income_ltm,
       "Gain (Loss) On Sale Of Assets (LTM)"                                   AS asset_sale_gain_ltm,
       "Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)" AS net_interest_income,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0::numeric)       AS interest_coverage_ebit,
       "EBITDA (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0::numeric)     AS interest_coverage_ebitda,
       "Interest Income On Investments (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                            AS interest_income_to_revenue,
       ("Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)") /
       NULLIF("Total Assets (LTM)", 0::numeric) *
       100::numeric                                                            AS net_interest_margin,
       ("Interest Income On Investments (LTM)" + "Gain (Loss) On Sale Of Assets (LTM)") /
       NULLIF(abs("Net Income - (IS) (LTM)"), 0::numeric)                      AS non_operating_income_ratio,
       "Operating Income (LTM)" / NULLIF("Operating Income (LTM)" + "Interest Income On Investments (LTM)",
                                         0::numeric)                           AS financial_income_quality,
       "Interest Expense/Total (LTM)" / NULLIF("EBIT (LTM)", 0::numeric)       AS interest_burden_ratio
FROM equities e;

alter materialized view mv_interest_analysis owner to postgres;

