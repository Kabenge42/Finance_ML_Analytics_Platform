create function calc_momentum_features()
    returns TABLE
            (
                ticker               text,
                price_momentum_1m    numeric,
                price_momentum_3m    numeric,
                price_momentum_6m    numeric,
                price_momentum_1y    numeric,
                price_momentum_5d    numeric,
                ema_crossover_20_50  integer,
                ema_crossover_50_250 integer,
                price_vs_ema_20d     numeric,
                price_vs_ema_250d    numeric,
                pct_off_52w_high     numeric,
                pct_above_52w_low    numeric,
                range_52w_position   numeric,
                beta_momentum        numeric,
                volatility_regime    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                              AS ticker,
       -- Price Momentum (NULLIF handles zero division, returning NULL)
       ("Last Price" - "Price (1M Ago)") / NULLIF("Price (1M Ago)", 0) * 100 AS price_momentum_1m,
       ("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100 AS price_momentum_3m,
       ("Last Price" - "Price (6M Ago)") / NULLIF("Price (6M Ago)", 0) * 100 AS price_momentum_6m,
       ("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 100 AS price_momentum_1y,
       ("Last Price" - "Price (5D Ago)") / NULLIF("Price (5D Ago)", 0) * 100 AS price_momentum_5d,
       -- EMA Crossovers
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" THEN -1
           ELSE 0
           END                                                               AS ema_crossover_20_50,
       CASE
           WHEN "EMA (50D)" > "EMA (250D)" THEN 1
           WHEN "EMA (50D)" < "EMA (250D)" THEN -1
           ELSE 0
           END                                                               AS ema_crossover_50_250,
       -- Price vs EMA
       ("Last Price" - "EMA (20D)") / NULLIF("EMA (20D)", 0)                 AS price_vs_ema_20d,
       ("Last Price" - "EMA (250D)") / NULLIF("EMA (250D)", 0)               AS price_vs_ema_250d,
       -- 52W Range Position
       ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0)           AS pct_off_52w_high,
       ("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0)             AS pct_above_52w_low,
       LEAST(1, GREATEST(0, ("Last Price" - "52W Low/Adj") /
                            NULLIF("52W High/Adj" - "52W Low/Adj", 0)))      AS range_52w_position,
       -- Beta Momentum
       "Beta (1Y)" - "Beta (5Y)"                                             AS beta_momentum,
       -- Volatility Regime
       "Volatility (1M)" / NULLIF("Volatility (1Y)", 0)                      AS volatility_regime
FROM postgres.public.equities;
$$;

alter function calc_momentum_features() owner to postgres;

