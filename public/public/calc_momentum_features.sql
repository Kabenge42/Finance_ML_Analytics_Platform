create function calc_momentum_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                 text,
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
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                           AS isin,
       pct_change("Last Price", "Price (1M Ago)")                       AS price_momentum_1m,
       pct_change("Last Price", "Price (3M Ago)")                       AS price_momentum_3m,
       pct_change("Last Price", "Price (6M Ago)")                       AS price_momentum_6m,
       pct_change("Last Price", "Price (1Y Ago)")                       AS price_momentum_1y,
       pct_change("Last Price", "Price (5D Ago)")                       AS price_momentum_5d,
       ema_crossover_signal("EMA (20D)", "EMA (50D)")                   AS ema_crossover_20_50,
       ema_crossover_signal("EMA (50D)", "EMA (250D)")                  AS ema_crossover_50_250,
       calc_change_ratio("Last Price", "EMA (20D)")                     AS price_vs_ema_20d,
       calc_change_ratio("Last Price", "EMA (250D)")                    AS price_vs_ema_250d,
       calc_change_ratio("52W High/Adj" - "Last Price", "52W High/Adj") AS pct_off_52w_high,
       calc_change_ratio("Last Price" - "52W Low/Adj", "52W Low/Adj")   AS pct_above_52w_low,
       clamp_score(safe_divide("Last Price" - "52W Low/Adj",
                               "52W High/Adj" - "52W Low/Adj"), 0, 1)   AS range_52w_position,
       "Beta (1Y)" - "Beta (5Y)"                                        AS beta_momentum,
       safe_divide("Volatility (1M)", "Volatility (1Y)")                AS volatility_regime
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_momentum_features(text) owner to postgres;

