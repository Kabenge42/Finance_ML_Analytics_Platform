create materialized view mv_unusual_items_features as
SELECT ticker,
       other_unusual_items_ltm,
       total_unusual_items,
       unusual_to_revenue_ratio,
       unusual_to_ebitda_ratio,
       unusual_to_net_income_ratio,
       clean_earnings_flag,
       recurring_unusual_flag,
       earnings_noise_score,
       quality_adjusted_ni,
       exceptional_items_impact
FROM v_unusual_items_features;

alter materialized view mv_unusual_items_features owner to postgres;

create index idx_mv_unusual_ticker
    on mv_unusual_items_features (ticker);

create index idx_mv_unusual_clean
    on mv_unusual_items_features (clean_earnings_flag)
    where (clean_earnings_flag = 1);

