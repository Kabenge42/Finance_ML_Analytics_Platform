create view v_cost_analysis
            (ticker, isin, name, sector, industry, country, market_cap, sga_fq, sga_fy, sga_1fy, sga_5yavg_fq,
             marketing_fq, marketing_fy, marketing_1fy, marketing_5yavg_ltm, total_opex_ltm, cost_of_revenues_ltm,
             sga_to_revenue_fq, sga_to_revenue_fy, sga_trend_yoy, sga_vs_5yavg, marketing_to_revenue_fq,
             marketing_to_revenue_fy, marketing_trend_yoy, marketing_vs_5yavg, operating_expense_ratio,
             cost_of_revenue_ratio, operating_leverage_score, cost_efficiency_trend)
as
SELECT ticker,
       isin,
       name,
       sector,
       industry,
       country,
       market_cap,
       sga_fq,
       sga_fy,
       sga_1fy,
       sga_5yavg_fq,
       marketing_fq,
       marketing_fy,
       marketing_1fy,
       marketing_5yavg_ltm,
       total_opex_ltm,
       cost_of_revenues_ltm,
       sga_to_revenue_fq,
       sga_to_revenue_fy,
       sga_trend_yoy,
       sga_vs_5yavg,
       marketing_to_revenue_fq,
       marketing_to_revenue_fy,
       marketing_trend_yoy,
       marketing_vs_5yavg,
       operating_expense_ratio,
       cost_of_revenue_ratio,
       operating_leverage_score,
       cost_efficiency_trend
FROM mv_cost_analysis;

alter table v_cost_analysis
    owner to postgres;

