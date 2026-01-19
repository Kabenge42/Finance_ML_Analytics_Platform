create function calc_cost_structure_features()
    returns TABLE
            (
                ticker                   text,
                sga_to_revenue_fq        numeric,
                sga_to_revenue_fy        numeric,
                sga_trend_yoy            numeric,
                sga_vs_5yavg             numeric,
                marketing_to_revenue_fq  numeric,
                marketing_to_revenue_fy  numeric,
                marketing_trend_yoy      numeric,
                marketing_vs_5yavg       numeric,
                operating_expense_ratio  numeric,
                cost_of_revenue_ratio    numeric,
                operating_leverage_score numeric,
                cost_efficiency_trend    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                              AS ticker,
       -- SG&A as % of Revenue (FQ)
       "Selling General & Admin Expenses/Total (FQ)" /
       NULLIF("Total Revenues (FQ)", 0) * 100                                AS sga_to_revenue_fq,
       -- SG&A as % of Revenue (FY)
       "Selling General & Admin Expenses/Total (FY)" /
       NULLIF("Total Revenues (FY)", 0) * 100                                AS sga_to_revenue_fy,
       -- SG&A YoY Trend
       ("Selling General & Admin Expenses/Total (FY)" -
        "Selling General & Admin Expenses/Total (-1FY)") /
       NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0) * 100 AS sga_trend_yoy,
       -- SG&A vs 5Y Average
       "Selling General & Admin Expenses/Total (FQ)" /
       NULLIF("Selling General & Admin Expenses/Total (5YAVGFQ)", 0)         AS sga_vs_5yavg,
       -- Marketing as % of Revenue (FQ)
       "Marketing Expenses (FQ)" / NULLIF("Total Revenues (FQ)", 0) * 100    AS marketing_to_revenue_fq,
       -- Marketing as % of Revenue (FY)
       "Marketing Expenses (FY)" / NULLIF("Total Revenues (FY)", 0) * 100    AS marketing_to_revenue_fy,
       -- Marketing YoY Trend
       ("Marketing Expenses (FY)" - "Marketing Expenses (-1FY)") /
       NULLIF(ABS("Marketing Expenses (-1FY)"), 0) * 100                     AS marketing_trend_yoy,
       -- Marketing vs 5Y Average
       ("Marketing Expenses (FY)" + "Marketing Expenses (-1FY)") / 2 /
       NULLIF("Marketing Expenses (5YAVGLTM)", 0)                            AS marketing_vs_5yavg,
       -- Total Operating Expense Ratio
       "Total Operating Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)  AS operating_expense_ratio,
       -- Cost of Revenue Ratio (COGS/Revenue)
       "Cost Of Revenues (LTM)" / NULLIF("Total Revenues (LTM)", 0)          AS cost_of_revenue_ratio,
       -- Operating Leverage Score (Revenue growth - OpEx growth sensitivity)
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0 AND ABS("Total Operating Expenses (LTM)") > 0
               THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                     NULLIF(ABS("Total Revenues (-1FY)"), 0)) /
                    NULLIF((("Selling General & Admin Expenses/Total (FY)" -
                             "Selling General & Admin Expenses/Total (-1FY)") /
                            NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0)), 0)
           END                                                               AS operating_leverage_score,
       -- Cost Efficiency Trend (declining SG&A ratio = improving efficiency)
       ("Selling General & Admin Expenses/Total (-1FY)" / NULLIF("Total Revenues (-1FY)", 0)) -
       ("Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0))
                                                                             AS cost_efficiency_trend
FROM postgres.public.equities;
$$;

alter function calc_cost_structure_features() owner to postgres;

