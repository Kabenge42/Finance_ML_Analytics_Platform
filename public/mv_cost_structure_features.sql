create materialized view mv_cost_structure_features as
SELECT ticker,
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
FROM v_cost_structure_features;

alter materialized view mv_cost_structure_features owner to postgres;

create index idx_mv_cost_ticker
    on mv_cost_structure_features (ticker);

create index idx_mv_cost_efficiency
    on mv_cost_structure_features (cost_efficiency_trend desc);

