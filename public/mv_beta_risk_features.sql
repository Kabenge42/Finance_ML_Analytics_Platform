create materialized view mv_beta_risk_features as
SELECT ticker,
       beta_1y,
       beta_2y,
       beta_5y,
       beta_trend_short,
       beta_trend_long,
       beta_stability,
       beta_regime_change,
       systematic_risk_score,
       defensive_stock_flag,
       high_beta_flag
FROM v_beta_risk_features;

alter materialized view mv_beta_risk_features owner to postgres;

create index idx_mv_beta_ticker
    on mv_beta_risk_features (ticker);

create index idx_mv_beta_defensive
    on mv_beta_risk_features (defensive_stock_flag)
    where (defensive_stock_flag = 1);

