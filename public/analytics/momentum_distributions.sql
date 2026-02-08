create table analytics.momentum_distributions
(
    price_momentum_1m      text,
    price_momentum_3m      text,
    price_momentum_6m      text,
    price_momentum_1y      text,
    price_momentum_5d      text,
    ema_crossover_20_50    text,
    ema_crossover_50_250   text,
    price_vs_ema_20d       text,
    price_vs_ema_250d      text,
    pct_off_52w_high       text,
    pct_above_52w_low      text,
    range_52w_position     text,
    beta_momentum          text,
    volatility_regime      text,
    price_momentum_1y_long text,
    price_momentum_3y      text,
    price_momentum_5y      text,
    long_term_trend_score  text,
    price_vs_ema_250d_long text,
    multi_year_high_flag   text,
    secular_trend_flag     text
);

alter table analytics.momentum_distributions
    owner to postgres;

