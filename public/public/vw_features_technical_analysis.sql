create view vw_features_technical_analysis
            (isin, ema_slope_20d, ema_trend_consistency, price_vs_ema_100d, near_52w_high_flag, near_52w_low_flag,
             volume_momentum_score, breakout_signal, high_volume_flag, low_volume_flag, volatility_compression,
             volatility_term_structure)
as
SELECT isin,
       ema_slope_20d,
       ema_trend_consistency,
       price_vs_ema_100d,
       near_52w_high_flag,
       near_52w_low_flag,
       volume_momentum_score,
       breakout_signal,
       high_volume_flag,
       low_volume_flag,
       volatility_compression,
       volatility_term_structure
FROM calc_technical_analysis_features() calc_technical_analysis_features(isin, ema_slope_20d, ema_trend_consistency,
                                                                         price_vs_ema_100d, near_52w_high_flag,
                                                                         near_52w_low_flag, volume_momentum_score,
                                                                         breakout_signal, high_volume_flag,
                                                                         low_volume_flag, volatility_compression,
                                                                         volatility_term_structure);

alter table vw_features_technical_analysis
    owner to postgres;

