create materialized view mv_momentum_analysis as
SELECT "Ticker"                          AS ticker,
       "ISIN"                            AS isin,
       "Name"                            AS name,
       "Sector"                          AS sector,
       "Industry"                        AS industry,
       "Country"                         AS country,
       "Market Cap"                      AS market_cap,
       "Last Price"                      AS last_price,
       "Price (1M Ago)"                  AS price_1m_ago,
       "Price (3M Ago)"                  AS price_3m_ago,
       "Price (6M Ago)"                  AS price_6m_ago,
       "Price (1Y Ago)"                  AS price_1y_ago,
       "Price (3Y Ago)"                  AS price_3y_ago,
       "Price (5Y Ago)"                  AS price_5y_ago,
       "Price (QTD Ago)"                 AS price_qtd_ago,
       "Total Return (YTD)"              AS total_return_ytd,
       "Total Return (5Y)"               AS total_return_5y,
       ("Last Price" - "Price (QTD Ago)") / NULLIF("Price (QTD Ago)", 0::numeric) *
       100::numeric                      AS price_momentum_qtd,
       ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0::numeric) *
       100::numeric                      AS price_momentum_3y,
       ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0::numeric) *
       100::numeric                      AS price_momentum_5y,
       power("Last Price" / NULLIF("Price (1Y Ago)", 0::numeric), 1.0) - 1::numeric -
       (power("Last Price" / NULLIF("Price (3Y Ago)", 0::numeric), 1.0 / 3.0) -
        1::numeric)                      AS momentum_acceleration_1y,
       power("Last Price" / NULLIF("Price (3Y Ago)", 0::numeric), 1.0 / 3.0) - 1::numeric -
       (power("Last Price" / NULLIF("Price (5Y Ago)", 0::numeric), 1.0 / 5.0) -
        1::numeric)                      AS momentum_acceleration_3y,
       (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0::numeric) * 0.5 +
        ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0::numeric) * 0.3 +
        ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0::numeric) * 0.2) *
       100::numeric                      AS long_term_trend_score,
       "Last Price" / NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)") / 2::numeric,
                             0::numeric) AS price_vs_3y_avg,
       "Last Price" / NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)" + "Price (5Y Ago)") / 3::numeric,
                             0::numeric) AS price_vs_5y_avg,
       CASE
           WHEN "Last Price" > "Price (1M Ago)" AND "Last Price" > "Price (3M Ago)" AND
                "Last Price" > "Price (1Y Ago)" AND "Last Price" > "Price (3Y Ago)" THEN 1.0
           WHEN "Last Price" > "Price (1M Ago)" AND "Last Price" > "Price (3M Ago)" AND "Last Price" > "Price (1Y Ago)"
               THEN 0.75
           WHEN "Last Price" > "Price (1M Ago)" AND "Last Price" > "Price (3M Ago)" THEN 0.5
           WHEN "Last Price" > "Price (1M Ago)" THEN 0.25
           ELSE 0::numeric
           END                           AS momentum_consistency,
       CASE
           WHEN (power("Last Price" / NULLIF("Price (5Y Ago)", 0::numeric), 1.0 / 5.0) - 1::numeric) > 0.10 THEN 1
           ELSE 0
           END                           AS secular_trend_flag
FROM equities e;

alter materialized view mv_momentum_analysis owner to postgres;

