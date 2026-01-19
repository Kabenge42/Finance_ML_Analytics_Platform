create materialized view mv_revenue_analysis as
SELECT "Ticker"                                                                        AS ticker,
       "ISIN"                                                                          AS isin,
       "Name"                                                                          AS name,
       "Sector"                                                                        AS sector,
       "Industry"                                                                      AS industry,
       "Country"                                                                       AS country,
       "Market Cap"                                                                    AS market_cap,
       "Total Revenues (FQ)"                                                           AS revenue_fq,
       "Total Revenues (FY)"                                                           AS revenue_fy,
       "Total Revenues (LTM)"                                                          AS revenue_ltm,
       "Total Revenues (-1FY)"                                                         AS revenue_1fy,
       "Total Revenues (5YAVGFQ)"                                                      AS revenue_5yavg_fq,
       "Total Revenues (5YAVGLTM)"                                                     AS revenue_5yavg_ltm,
       "Total Revenues/CAGR (5Y FY)"                                                   AS revenue_cagr_5y,
       "Revenue Growth (3Y)"                                                           AS revenue_growth_3y,
       "Revenue Growth (5Y)"                                                           AS revenue_growth_5y,
       "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0::numeric)          AS revenue_fq_vs_5yavg,
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0::numeric)        AS revenue_ltm_vs_5yavg,
       ("Total Revenues (FQ)" * 4::numeric - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                                    AS revenue_qoq_growth,
       ("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(abs("Total Revenues (-1FY)"), 0::numeric) *
       100::numeric                                                                    AS revenue_yoy_growth,
       "Total Revenues (FQ)" * 4::numeric                                              AS revenue_quarterly_run_rate,
       "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4::numeric, 0::numeric) AS revenue_seasonality_factor
FROM equities e;

alter materialized view mv_revenue_analysis owner to postgres;

create index idx_mv_rev_ticker
    on mv_revenue_analysis (ticker)
    where (ticker IS NOT NULL);

create index idx_mv_rev_sector
    on mv_revenue_analysis (sector);

create index idx_mv_rev_growth
    on mv_revenue_analysis (revenue_yoy_growth desc);

