create function calc_revenue_quarterly_features()
    returns TABLE
            (
                ticker                     text,
                revenue_fq                 numeric,
                revenue_ltm                numeric,
                revenue_fy                 numeric,
                revenue_1fy                numeric,
                revenue_5yavg_fq           numeric,
                revenue_5yavg_ltm          numeric,
                revenue_fq_vs_5yavg        numeric,
                revenue_ltm_vs_5yavg       numeric,
                revenue_qoq_growth         numeric,
                revenue_yoy_growth         numeric,
                revenue_quarterly_run_rate numeric,
                revenue_seasonality_factor numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                        AS ticker,
       "Total Revenues (FQ)"                                           AS revenue_fq,
       "Total Revenues (LTM)"                                          AS revenue_ltm,
       "Total Revenues (FY)"                                           AS revenue_fy,
       "Total Revenues (-1FY)"                                         AS revenue_1fy,
       "Total Revenues (5YAVGFQ)"                                      AS revenue_5yavg_fq,
       "Total Revenues (5YAVGLTM)"                                     AS revenue_5yavg_ltm,
       -- FQ vs 5Y Average (seasonality-adjusted benchmark)
       "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0)   AS revenue_fq_vs_5yavg,
       -- LTM vs 5Y Average
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_ltm_vs_5yavg,
       -- QoQ Growth (FQ annualized vs LTM)
       ("Total Revenues (FQ)" * 4 - "Total Revenues (LTM)") /
       NULLIF("Total Revenues (LTM)", 0) * 100                         AS revenue_qoq_growth,
       -- YoY Growth
       ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
       NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100                   AS revenue_yoy_growth,
       -- Quarterly Run Rate (annualized FQ)
       "Total Revenues (FQ)" * 4                                       AS revenue_quarterly_run_rate,
       -- Seasonality Factor (FQ as % of typical quarter)
       "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4, 0)   AS revenue_seasonality_factor
FROM postgres.public.equities;
$$;

alter function calc_revenue_quarterly_features() owner to postgres;

