create materialized view mv_tangible_book_features as
SELECT ticker,
       tbv_fy,
       tbv_ltm,
       price_to_tbv,
       tbv_per_share,
       tbv_growth_yoy,
       tangible_equity_ratio,
       intangible_to_tbv_ratio,
       tbv_vs_market_cap,
       net_tangible_assets,
       tbv_margin_of_safety
FROM v_tangible_book_features;

alter materialized view mv_tangible_book_features owner to postgres;

create index idx_mv_tbv_ticker
    on mv_tangible_book_features (ticker);

create index idx_mv_tbv_ratio
    on mv_tangible_book_features (price_to_tbv);

