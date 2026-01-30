create view vw_features_analyst_sentiment
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, analyst_bullish_pct,
             analyst_bearish_pct, analyst_neutral_pct, analyst_conviction, upside_potential, price_target_spread_pct,
             price_target_revision_1m, price_target_revision_3m, eps_revision_momentum, analyst_rating_normalized,
             analyst_coverage_quality, pt_momentum_1w, pt_momentum_1m, pt_momentum_3m, pt_momentum_6m, pt_momentum_1y,
             pt_median_momentum_1m, pt_median_momentum_3m, pt_acceleration_short, pt_acceleration_long,
             pt_consensus_convergence, analyst_coverage_change_1m, analyst_coverage_change_3m,
             analyst_coverage_change_1y, pt_vs_price_momentum, analyst_coverage_trend)
as
SELECT id.isin,
       id.ticker,
       id.name,
       id.industry,
       id.sector,
       id.trading_country,
       id.region,
       id.country,
       id.exchange,
       sf.analyst_bullish_pct,
       sf.analyst_bearish_pct,
       sf.analyst_neutral_pct,
       sf.analyst_conviction,
       sf.upside_potential,
       sf.price_target_spread_pct,
       sf.price_target_revision_1m,
       sf.price_target_revision_3m,
       sf.eps_revision_momentum,
       sf.analyst_rating_normalized,
       sf.analyst_coverage_quality,
       ptd.pt_momentum_1w,
       ptd.pt_momentum_1m,
       ptd.pt_momentum_3m,
       ptd.pt_momentum_6m,
       ptd.pt_momentum_1y,
       ptd.pt_median_momentum_1m,
       ptd.pt_median_momentum_3m,
       ptd.pt_acceleration_short,
       ptd.pt_acceleration_long,
       ptd.pt_consensus_convergence,
       ptd.analyst_coverage_change_1m,
       ptd.analyst_coverage_change_3m,
       ptd.analyst_coverage_change_1y,
       ptd.pt_vs_price_momentum,
       ptd.analyst_coverage_trend
FROM vw_identifier_columns                      id
         LEFT JOIN calc_sentiment_features()    sf(isin, analyst_bullish_pct, analyst_bearish_pct, analyst_neutral_pct,
                                                   analyst_conviction, upside_potential, price_target_spread_pct,
                                                   price_target_revision_1m, price_target_revision_3m,
                                                   eps_revision_momentum, analyst_rating_normalized,
                                                   analyst_coverage_quality) USING (isin)
         LEFT JOIN calc_price_target_dynamics() ptd(isin, pt_momentum_1w, pt_momentum_1m, pt_momentum_3m,
                                                    pt_momentum_6m, pt_momentum_1y, pt_median_momentum_1m,
                                                    pt_median_momentum_3m, pt_acceleration_short, pt_acceleration_long,
                                                    pt_consensus_convergence, analyst_coverage_change_1m,
                                                    analyst_coverage_change_3m, analyst_coverage_change_1y,
                                                    pt_vs_price_momentum, analyst_coverage_trend) USING (isin);

comment on view vw_features_analyst_sentiment is 'Analyst sentiment metrics including ratings distribution and price target dynamics.
    Source functions: calc_sentiment_features, calc_price_target_dynamics';

alter table vw_features_analyst_sentiment
    owner to postgres;

