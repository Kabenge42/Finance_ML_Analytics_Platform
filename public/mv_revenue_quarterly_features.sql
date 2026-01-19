create materialized view mv_revenue_quarterly_features as
SELECT ticker,
       revenue_fq,
       revenue_ltm,
       revenue_fy,
       revenue_1fy,
       revenue_5yavg_fq,
       revenue_5yavg_ltm,
       revenue_fq_vs_5yavg,
       revenue_ltm_vs_5yavg,
       revenue_qoq_growth,
       revenue_yoy_growth,
       revenue_quarterly_run_rate,
       revenue_seasonality_factor
FROM v_revenue_quarterly_features;

alter materialized view mv_revenue_quarterly_features owner to postgres;

create index idx_mv_rev_q_ticker
    on mv_revenue_quarterly_features (ticker);

create index idx_mv_rev_q_growth
    on mv_revenue_quarterly_features (revenue_yoy_growth desc);

