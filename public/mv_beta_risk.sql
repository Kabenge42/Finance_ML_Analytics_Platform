create materialized view mv_beta_risk as
SELECT "Ticker"                                                  AS ticker,
       "ISIN"                                                    AS isin,
       "Name"                                                    AS name,
       "Sector"                                                  AS sector,
       "Industry"                                                AS industry,
       "Country"                                                 AS country,
       "Market Cap"                                              AS market_cap,
       "Beta (1Y)"                                               AS beta_1y,
       "Beta (2Y)"                                               AS beta_2y,
       "Beta (5Y)"                                               AS beta_5y,
       "Volatility (1M)"                                         AS volatility_1m,
       "Volatility (3M)"                                         AS volatility_3m,
       "Volatility (6M)"                                         AS volatility_6m,
       "Volatility (1Y)"                                         AS volatility_1y,
       "Beta (1Y)" - "Beta (2Y)"                                 AS beta_trend_short,
       "Beta (2Y)" - "Beta (5Y)"                                 AS beta_trend_long,
       CASE
           WHEN (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") - LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)")) > 0::numeric
               THEN 1.0 /
                    (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") - LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)"))
           ELSE 1.0
           END                                                   AS beta_stability,
       CASE
           WHEN abs("Beta (1Y)" - "Beta (5Y)") > 0.3 THEN 1
           ELSE 0
           END                                                   AS beta_regime_change,
       "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2 AS systematic_risk_score,
       CASE
           WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8 THEN 1
           ELSE 0
           END                                                   AS defensive_stock_flag,
       CASE
           WHEN "Beta (1Y)" > 1.3 AND "Beta (2Y)" > 1.3 THEN 1
           ELSE 0
           END                                                   AS high_beta_flag
FROM equities e;

alter materialized view mv_beta_risk owner to postgres;

