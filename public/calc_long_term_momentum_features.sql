create function calc_long_term_momentum_features()
    returns TABLE
            (
                ticker                   text,
                price_momentum_qtd       numeric,
                price_momentum_3y        numeric,
                price_momentum_5y        numeric,
                momentum_acceleration_1y numeric,
                momentum_acceleration_3y numeric,
                long_term_trend_score    numeric,
                price_vs_3y_avg          numeric,
                price_vs_5y_avg          numeric,
                momentum_consistency     numeric,
                secular_trend_flag       integer
            )
    language sql
as
$$
SELECT "Ticker"                                                            AS ticker,
       -- QTD Momentum
       ("Last Price" - "Price (QTD Ago)") /
       NULLIF("Price (QTD Ago)", 0) * 100                                  AS price_momentum_qtd,
       -- 3Y Momentum
       ("Last Price" - "Price (3Y Ago)") /
       NULLIF("Price (3Y Ago)", 0) * 100                                   AS price_momentum_3y,
       -- 5Y Momentum
       ("Last Price" - "Price (5Y Ago)") /
       NULLIF("Price (5Y Ago)", 0) * 100                                   AS price_momentum_5y,
       -- Momentum Acceleration (1Y vs 3Y CAGR)
       (POWER("Last Price" / NULLIF("Price (1Y Ago)", 0), 1.0) - 1) -
       (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1)  AS momentum_acceleration_1y,
       -- Momentum Acceleration (3Y vs 5Y CAGR)
       (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1) -
       (POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1)  AS momentum_acceleration_3y,
       -- Long-Term Trend Score (weighted momentum)
       (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 0.5 +
        ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 0.3 +
        ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 0.2) * 100
                                                                           AS long_term_trend_score,
       -- Price vs 3Y Simple Average (proxy)
       "Last Price" / NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)") / 2, 0) AS price_vs_3y_avg,
       -- Price vs 5Y Simple Average (proxy)
       "Last Price" / NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)" + "Price (5Y Ago)") / 3, 0)
                                                                           AS price_vs_5y_avg,
       -- Momentum Consistency (all timeframes positive)
       CASE
           WHEN ("Last Price" > "Price (1M Ago)") AND
                ("Last Price" > "Price (3M Ago)") AND
                ("Last Price" > "Price (1Y Ago)") AND
                ("Last Price" > "Price (3Y Ago)")
               THEN 1.0
           WHEN ("Last Price" > "Price (1M Ago)") AND
                ("Last Price" > "Price (3M Ago)") AND
                ("Last Price" > "Price (1Y Ago)")
               THEN 0.75
           WHEN ("Last Price" > "Price (1M Ago)") AND
                ("Last Price" > "Price (3M Ago)")
               THEN 0.5
           WHEN ("Last Price" > "Price (1M Ago)")
               THEN 0.25
           ELSE 0
           END                                                             AS momentum_consistency,
       -- Secular Trend Flag (5Y CAGR > 10%)
       CASE
           WHEN POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1 > 0.10
               THEN 1
           ELSE 0
           END                                                             AS secular_trend_flag
FROM postgres.public.equities;
$$;

alter function calc_long_term_momentum_features() owner to postgres;

