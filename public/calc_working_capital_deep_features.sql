create function calc_working_capital_deep_features()
    returns TABLE
            (
                ticker                     text,
                current_assets_ltm         numeric,
                current_liabilities_ltm    numeric,
                net_working_capital        numeric,
                working_capital_to_revenue numeric,
                working_capital_to_assets  numeric,
                current_ratio              numeric,
                quick_ratio                numeric,
                cash_ratio                 numeric,
                defensive_interval         numeric,
                working_capital_turnover   numeric,
                liquidity_score            numeric,
                working_capital_efficiency numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                         AS ticker,
       "Total Current Assets (LTM)"                                     AS current_assets_ltm,
       "Total Current Liabilities (LTM)"                                AS current_liabilities_ltm,
       -- Net Working Capital
       "Total Current Assets (LTM)" - "Total Current Liabilities (LTM)" AS net_working_capital,
       -- Working Capital to Revenue
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0)      AS working_capital_to_revenue,
       -- Working Capital to Assets
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0)        AS working_capital_to_assets,
       -- Current Ratio
       "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0)
                                                                        AS current_ratio,
       -- Quick Ratio (exclude inventory)
       ("Total Current Assets (LTM)" - "Inventory (LTM)") /
       NULLIF("Total Current Liabilities (LTM)", 0)                     AS quick_ratio,
       -- Cash Ratio
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0)
                                                                        AS cash_ratio,
       -- Defensive Interval (days of OpEx covered by current assets)
       ("Total Current Assets (LTM)" /
        NULLIF("Total Operating Expenses (LTM)" / 365, 0))              AS defensive_interval,
       -- Working Capital Turnover
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)      AS working_capital_turnover,
       -- Liquidity Score (weighted average of ratios, 0-100)
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 2
                                  THEN 40
                              WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1.5
                                  THEN 30
                              WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1
                                  THEN 20
                              ELSE 0
                             END) +
                         (CASE
                              WHEN ("Total Current Assets (LTM)" - "Inventory (LTM)") /
                                   NULLIF("Total Current Liabilities (LTM)", 0) >= 1 THEN 30
                              ELSE 15
                             END) +
                         (CASE
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.5
                                  THEN 30
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.2
                                  THEN 15
                              ELSE 0
                             END)
                   ))                                                   AS liquidity_score,
       -- Working Capital Efficiency (Revenue generated per $ of WC)
       "Total Revenues (LTM)" / NULLIF(ABS("Working Capital (LTM)"), 0) AS working_capital_efficiency
FROM postgres.public.equities;
$$;

alter function calc_working_capital_deep_features() owner to postgres;

