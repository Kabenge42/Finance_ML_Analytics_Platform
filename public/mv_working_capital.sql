create materialized view mv_working_capital as
SELECT "Ticker"                                                             AS ticker,
       "ISIN"                                                               AS isin,
       "Name"                                                               AS name,
       "Sector"                                                             AS sector,
       "Industry"                                                           AS industry,
       "Country"                                                            AS country,
       "Market Cap"                                                         AS market_cap,
       "Total Current Assets (LTM)"                                         AS current_assets_ltm,
       "Total Current Liabilities (LTM)"                                    AS current_liabilities_ltm,
       "Working Capital (LTM)"                                              AS working_capital_ltm,
       "Cash And Equivalents (LTM)"                                         AS cash_ltm,
       "Inventory (LTM)"                                                    AS inventory_ltm,
       "Total Assets (LTM)"                                                 AS total_assets_ltm,
       "Total Operating Expenses (LTM)"                                     AS total_opex_ltm,
       "Current Ratio (LTM)"                                                AS current_ratio_base,
       "Total Current Assets (LTM)" - "Total Current Liabilities (LTM)"     AS net_working_capital,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) AS working_capital_to_revenue,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0::numeric)   AS working_capital_to_assets,
       "Total Current Assets (LTM)" /
       NULLIF("Total Current Liabilities (LTM)", 0::numeric)                AS current_ratio,
       ("Total Current Assets (LTM)" - "Inventory (LTM)") /
       NULLIF("Total Current Liabilities (LTM)", 0::numeric)                AS quick_ratio,
       "Cash And Equivalents (LTM)" /
       NULLIF("Total Current Liabilities (LTM)", 0::numeric)                AS cash_ratio,
       "Total Current Assets (LTM)" /
       NULLIF("Total Operating Expenses (LTM)" / 365::numeric, 0::numeric)  AS defensive_interval,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0::numeric) AS working_capital_turnover,
       GREATEST(0, LEAST(100,
                         CASE
                             WHEN ("Total Current Assets (LTM)" /
                                   NULLIF("Total Current Liabilities (LTM)", 0::numeric)) >= 2::numeric THEN 40
                             WHEN ("Total Current Assets (LTM)" /
                                   NULLIF("Total Current Liabilities (LTM)", 0::numeric)) >= 1.5 THEN 30
                             WHEN ("Total Current Assets (LTM)" /
                                   NULLIF("Total Current Liabilities (LTM)", 0::numeric)) >= 1::numeric THEN 20
                             ELSE 0
                             END +
                         CASE
                             WHEN (("Total Current Assets (LTM)" - "Inventory (LTM)") /
                                   NULLIF("Total Current Liabilities (LTM)", 0::numeric)) >= 1::numeric THEN 30
                             ELSE 15
                             END +
                         CASE
                             WHEN ("Cash And Equivalents (LTM)" /
                                   NULLIF("Total Current Liabilities (LTM)", 0::numeric)) >= 0.5 THEN 30
                             WHEN ("Cash And Equivalents (LTM)" /
                                   NULLIF("Total Current Liabilities (LTM)", 0::numeric)) >= 0.2 THEN 15
                             ELSE 0
                             END))                                          AS liquidity_score,
       "Total Revenues (LTM)" /
       NULLIF(abs("Working Capital (LTM)"), 0::numeric)                     AS working_capital_efficiency
FROM equities e;

alter materialized view mv_working_capital owner to postgres;

