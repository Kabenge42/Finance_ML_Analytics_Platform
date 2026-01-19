create view v_tangible_book_features
            (ticker, tbv_fy, tbv_ltm, price_to_tbv, tbv_per_share, tbv_growth_yoy, tangible_equity_ratio,
             intangible_to_tbv_ratio, tbv_vs_market_cap, net_tangible_assets, tbv_margin_of_safety)
as
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
FROM calc_tangible_book_features() calc_tangible_book_features(ticker, tbv_fy, tbv_ltm, price_to_tbv, tbv_per_share,
                                                               tbv_growth_yoy, tangible_equity_ratio,
                                                               intangible_to_tbv_ratio, tbv_vs_market_cap,
                                                               net_tangible_assets, tbv_margin_of_safety);

alter table v_tangible_book_features
    owner to postgres;

