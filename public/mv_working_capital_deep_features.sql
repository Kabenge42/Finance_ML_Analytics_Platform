create materialized view mv_working_capital_deep_features as
SELECT ticker,
       current_assets_ltm,
       current_liabilities_ltm,
       net_working_capital,
       working_capital_to_revenue,
       working_capital_to_assets,
       current_ratio,
       quick_ratio,
       cash_ratio,
       defensive_interval,
       working_capital_turnover,
       liquidity_score,
       working_capital_efficiency
FROM v_working_capital_deep_features;

alter materialized view mv_working_capital_deep_features owner to postgres;

create index idx_mv_wc_ticker
    on mv_working_capital_deep_features (ticker);

create index idx_mv_wc_liquidity
    on mv_working_capital_deep_features (liquidity_score desc);

