create view vw_features_technical_analysis
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, ema_slope_20d,
             ema_trend_consistency, price_vs_ema_100d, near_52w_high_flag, near_52w_low_flag, volume_momentum_score,
             breakout_signal, high_volume_flag, low_volume_flag, volatility_compression, volatility_term_structure)
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
       ta.ema_slope_20d,
       ta.ema_trend_consistency,
       ta.price_vs_ema_100d,
       ta.near_52w_high_flag,
       ta.near_52w_low_flag,
       ta.volume_momentum_score,
       ta.breakout_signal,
       ta.high_volume_flag,
       ta.low_volume_flag,
       ta.volatility_compression,
       ta.volatility_term_structure
FROM vw_identifier_columns                            id
         LEFT JOIN calc_technical_analysis_features() ta(isin, ema_slope_20d, ema_trend_consistency, price_vs_ema_100d,
                                                         near_52w_high_flag, near_52w_low_flag, volume_momentum_score,
                                                         breakout_signal, high_volume_flag, low_volume_flag,
                                                         volatility_compression, volatility_term_structure)
                   USING (isin);

comment on view vw_features_technical_analysis is 'Technical analysis indicators including EMA trends, volume signals, and volatility patterns.
    Source function: calc_technical_analysis_features';

alter table vw_features_technical_analysis
    owner to postgres;

