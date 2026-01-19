create materialized view mv_long_term_momentum_features as
SELECT ticker,
       price_momentum_qtd,
       price_momentum_3y,
       price_momentum_5y,
       momentum_acceleration_1y,
       momentum_acceleration_3y,
       long_term_trend_score,
       price_vs_3y_avg,
       price_vs_5y_avg,
       momentum_consistency,
       secular_trend_flag
FROM v_long_term_momentum_features;

alter materialized view mv_long_term_momentum_features owner to postgres;

create index idx_mv_ltm_ticker
    on mv_long_term_momentum_features (ticker);

create index idx_mv_ltm_secular
    on mv_long_term_momentum_features (secular_trend_flag)
    where (secular_trend_flag = 1);

