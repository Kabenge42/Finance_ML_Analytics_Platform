create table analytics.technical_analysis_distributions
(
    ema_slope_20d             text,
    ema_trend_consistency     text,
    price_vs_ema_100d         text,
    near_52w_high_flag        text,
    near_52w_low_flag         text,
    volume_momentum_score     text,
    breakout_signal           text,
    high_volume_flag          text,
    low_volume_flag           text,
    volatility_compression    text,
    volatility_term_structure text
);

alter table analytics.technical_analysis_distributions
    owner to postgres;

