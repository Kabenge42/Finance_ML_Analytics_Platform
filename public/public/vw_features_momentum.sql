create view vw_features_momentum
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, price_momentum_1m,
             price_momentum_3m, price_momentum_6m, price_momentum_1y, price_momentum_5d, ema_crossover_20_50,
             ema_crossover_50_250, price_vs_ema_20d, price_vs_ema_250d, pct_off_52w_high, pct_above_52w_low,
             range_52w_position, beta_momentum, volatility_regime, price_momentum_1y_long, price_momentum_3y,
             price_momentum_5y, long_term_trend_score, price_vs_ema_250d_long, multi_year_high_flag, secular_trend_flag)
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
       mf.price_momentum_1m,
       mf.price_momentum_3m,
       mf.price_momentum_6m,
       mf.price_momentum_1y,
       mf.price_momentum_5d,
       mf.ema_crossover_20_50,
       mf.ema_crossover_50_250,
       mf.price_vs_ema_20d,
       mf.price_vs_ema_250d,
       mf.pct_off_52w_high,
       mf.pct_above_52w_low,
       mf.range_52w_position,
       mf.beta_momentum,
       mf.volatility_regime,
       ltm.price_momentum_1y AS price_momentum_1y_long,
       ltm.price_momentum_3y,
       ltm.price_momentum_5y,
       ltm.long_term_trend_score,
       ltm.price_vs_ema_250d AS price_vs_ema_250d_long,
       ltm.multi_year_high_flag,
       ltm.secular_trend_flag
FROM vw_identifier_columns                            id
         LEFT JOIN calc_momentum_features()           mf(isin, price_momentum_1m, price_momentum_3m, price_momentum_6m,
                                                         price_momentum_1y, price_momentum_5d, ema_crossover_20_50,
                                                         ema_crossover_50_250, price_vs_ema_20d, price_vs_ema_250d,
                                                         pct_off_52w_high, pct_above_52w_low, range_52w_position,
                                                         beta_momentum, volatility_regime) USING (isin)
         LEFT JOIN calc_long_term_momentum_features() ltm(isin, price_momentum_1y, price_momentum_3y, price_momentum_5y,
                                                          long_term_trend_score, price_vs_ema_250d,
                                                          multi_year_high_flag, secular_trend_flag) USING (isin);

comment on view vw_features_momentum is 'Price momentum and trend indicators across multiple timeframes.
    Source functions: calc_momentum_features, calc_long_term_momentum_features';

alter table vw_features_momentum
    owner to postgres;

