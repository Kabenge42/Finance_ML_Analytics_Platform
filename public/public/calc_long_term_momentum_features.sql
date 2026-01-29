create function calc_long_term_momentum_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                  text,
                price_momentum_1y     numeric,
                price_momentum_3y     numeric,
                price_momentum_5y     numeric,
                long_term_trend_score numeric,
                price_vs_ema_250d     numeric,
                multi_year_high_flag  integer,
                secular_trend_flag    integer
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                     AS isin,
       pct_change("Last Price", "Price (1Y Ago)") AS price_momentum_1y,
       pct_change("Last Price", "Price (3Y Ago)") AS price_momentum_3y,
       pct_change("Last Price", "Price (5Y Ago)") AS price_momentum_5y,
       -- Weighted trend score using available periods (1Y: 50%, 3Y: 30%, 5Y: 20%)
       (COALESCE(pct_change("Last Price", "Price (1Y Ago)"), 0) * 0.50 +
        COALESCE(pct_change("Last Price", "Price (3Y Ago)"), 0) * 0.30 +
        COALESCE(pct_change("Last Price", "Price (5Y Ago)"), 0) * 0.20) / 100
                                                  AS long_term_trend_score,
       pct_change("Last Price", "EMA (250D)")     AS price_vs_ema_250d,
       CASE
           WHEN calc_change_ratio("52W High/Adj" - "Last Price", "52W High/Adj") <= 0.10
               AND calc_change_ratio("Last Price", "Price (3Y Ago)") > 0.5
               THEN 1
           ELSE 0
           END                                    AS multi_year_high_flag,
       CASE
           WHEN calc_change_ratio("Last Price", "Price (3Y Ago)") > 0.20
               AND calc_change_ratio("Last Price", "Price (1Y Ago)") > 0
               AND "EMA (50D)" > "EMA (250D)"
               THEN 1
           ELSE 0
           END                                    AS secular_trend_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_long_term_momentum_features(text) owner to postgres;

