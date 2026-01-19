create view v_revenue_analysis
            (ticker, isin, name, sector, industry, country, market_cap, revenue_fq, revenue_fy, revenue_ltm,
             revenue_1fy, revenue_5yavg_fq, revenue_5yavg_ltm, revenue_cagr_5y, revenue_growth_3y, revenue_growth_5y,
             revenue_fq_vs_5yavg, revenue_ltm_vs_5yavg, revenue_qoq_growth, revenue_yoy_growth,
             revenue_quarterly_run_rate, revenue_seasonality_factor)
as
SELECT ticker,
       isin,
       name,
       sector,
       industry,
       country,
       market_cap,
       revenue_fq,
       revenue_fy,
       revenue_ltm,
       revenue_1fy,
       revenue_5yavg_fq,
       revenue_5yavg_ltm,
       revenue_cagr_5y,
       revenue_growth_3y,
       revenue_growth_5y,
       revenue_fq_vs_5yavg,
       revenue_ltm_vs_5yavg,
       revenue_qoq_growth,
       revenue_yoy_growth,
       revenue_quarterly_run_rate,
       revenue_seasonality_factor
FROM mv_revenue_analysis;

alter table v_revenue_analysis
    owner to postgres;

