create function calc_technical_analysis_features()
    returns TABLE
            (
                ticker                    text,
                ema_slope_20d             numeric,
                ema_trend_consistency     integer,
                price_vs_ema_100d         numeric,
                near_52w_high_flag        integer,
                near_52w_low_flag         integer,
                volume_momentum_score     numeric,
                breakout_signal           integer,
                high_volume_flag          integer,
                low_volume_flag           integer,
                volatility_compression    numeric,
                volatility_term_structure numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                      AS ticker,
       -- EMA Slope (20D vs 50D as proxy for short-term trend direction)
       ("EMA (20D)" - "EMA (50D)") / NULLIF("EMA (50D)", 0)          AS ema_slope_20d,

       -- EMA Trend Consistency (all EMAs aligned: bullish=1, bearish=-1, mixed=0)
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" AND "EMA (50D)" > "EMA (100D)"
               AND "EMA (100D)" > "EMA (250D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" AND "EMA (50D)" < "EMA (100D)"
               AND "EMA (100D)" < "EMA (250D)" THEN -1
           ELSE 0
           END                                                       AS ema_trend_consistency,

       -- Price vs EMA 100D (medium-term deviation)
       ("Last Price" - "EMA (100D)") / NULLIF("EMA (100D)", 0) * 100 AS price_vs_ema_100d,

       -- Near 52W High Flag (within 5% of high)
       CASE
           WHEN ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_high_flag,

       -- Near 52W Low Flag (within 5% of low)
       CASE
           WHEN ("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_low_flag,

       -- Volume Momentum Score (Relative Volume × 1M Price Change)
       "Rel. Volume" * "Price Chg. % (1M)"                           AS volume_momentum_score,

       -- Breakout Signal (EMA bullish crossover + near 52W high)
       CASE
           WHEN "EMA (20D)" > "EMA (50D)"
               AND ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS breakout_signal,

       -- High Volume Flag (Relative Volume > 1.5x average)
       CASE WHEN "Rel. Volume" > 1.5 THEN 1 ELSE 0 END               AS high_volume_flag,

       -- Low Volume Flag (Relative Volume < 0.5x average)
       CASE WHEN "Rel. Volume" < 0.5 THEN 1 ELSE 0 END               AS low_volume_flag,

       -- Volatility Compression (1Y - 1M, positive = vol decreasing)
       "Volatility (1Y)" - "Volatility (1M)"                         AS volatility_compression,

       -- Volatility Term Structure (3M vs 6M)
       "Volatility (3M)" - "Volatility (6M)"                         AS volatility_term_structure

FROM postgres.public.equities;
$$;

alter function calc_technical_analysis_features() owner to postgres;

