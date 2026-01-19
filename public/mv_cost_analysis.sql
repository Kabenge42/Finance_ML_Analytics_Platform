create materialized view mv_cost_analysis as
SELECT "Ticker"                                                              AS ticker,
       "ISIN"                                                                AS isin,
       "Name"                                                                AS name,
       "Sector"                                                              AS sector,
       "Industry"                                                            AS industry,
       "Country"                                                             AS country,
       "Market Cap"                                                          AS market_cap,
       "Selling General & Admin Expenses/Total (FQ)"                         AS sga_fq,
       "Selling General & Admin Expenses/Total (FY)"                         AS sga_fy,
       "Selling General & Admin Expenses/Total (-1FY)"                       AS sga_1fy,
       "Selling General & Admin Expenses/Total (5YAVGFQ)"                    AS sga_5yavg_fq,
       "Marketing Expenses (FQ)"                                             AS marketing_fq,
       "Marketing Expenses (FY)"                                             AS marketing_fy,
       "Marketing Expenses (-1FY)"                                           AS marketing_1fy,
       "Marketing Expenses (5YAVGLTM)"                                       AS marketing_5yavg_ltm,
       "Total Operating Expenses (LTM)"                                      AS total_opex_ltm,
       "Cost Of Revenues (LTM)"                                              AS cost_of_revenues_ltm,
       "Selling General & Admin Expenses/Total (FQ)" / NULLIF("Total Revenues (FQ)", 0::numeric) *
       100::numeric                                                          AS sga_to_revenue_fq,
       "Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0::numeric) *
       100::numeric                                                          AS sga_to_revenue_fy,
       ("Selling General & Admin Expenses/Total (FY)" - "Selling General & Admin Expenses/Total (-1FY)") /
       NULLIF(abs("Selling General & Admin Expenses/Total (-1FY)"), 0::numeric) *
       100::numeric                                                          AS sga_trend_yoy,
       "Selling General & Admin Expenses/Total (FQ)" / NULLIF("Selling General & Admin Expenses/Total (5YAVGFQ)",
                                                              0::numeric)    AS sga_vs_5yavg,
       "Marketing Expenses (FQ)" / NULLIF("Total Revenues (FQ)", 0::numeric) *
       100::numeric                                                          AS marketing_to_revenue_fq,
       "Marketing Expenses (FY)" / NULLIF("Total Revenues (FY)", 0::numeric) *
       100::numeric                                                          AS marketing_to_revenue_fy,
       ("Marketing Expenses (FY)" - "Marketing Expenses (-1FY)") /
       NULLIF(abs("Marketing Expenses (-1FY)"), 0::numeric) *
       100::numeric                                                          AS marketing_trend_yoy,
       ("Marketing Expenses (FY)" + "Marketing Expenses (-1FY)") / 2::numeric /
       NULLIF("Marketing Expenses (5YAVGLTM)", 0::numeric)                   AS marketing_vs_5yavg,
       "Total Operating Expenses (LTM)" /
       NULLIF("Total Revenues (LTM)", 0::numeric)                            AS operating_expense_ratio,
       "Cost Of Revenues (LTM)" / NULLIF("Total Revenues (LTM)", 0::numeric) AS cost_of_revenue_ratio,
       CASE
           WHEN abs("Total Revenues (-1FY)") > 0::numeric AND abs("Total Operating Expenses (LTM)") > 0::numeric THEN
               ("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(abs("Total Revenues (-1FY)"), 0::numeric) /
               NULLIF(("Selling General & Admin Expenses/Total (FY)" -
                       "Selling General & Admin Expenses/Total (-1FY)") /
                      NULLIF(abs("Selling General & Admin Expenses/Total (-1FY)"), 0::numeric), 0::numeric)
           ELSE NULL::numeric
           END                                                               AS operating_leverage_score,
       "Selling General & Admin Expenses/Total (-1FY)" / NULLIF("Total Revenues (-1FY)", 0::numeric) -
       "Selling General & Admin Expenses/Total (FY)" /
       NULLIF("Total Revenues (FY)", 0::numeric)                             AS cost_efficiency_trend
FROM equities e;

alter materialized view mv_cost_analysis owner to postgres;

