create function calc_beta_risk_features()
    returns TABLE
            (
                ticker                text,
                beta_1y               numeric,
                beta_2y               numeric,
                beta_5y               numeric,
                beta_trend_short      numeric,
                beta_trend_long       numeric,
                beta_stability        numeric,
                beta_regime_change    integer,
                systematic_risk_score numeric,
                defensive_stock_flag  integer,
                high_beta_flag        integer
            )
    language sql
as
$$
SELECT "Ticker"                                                  AS ticker,
       "Beta (1Y)"                                               AS beta_1y,
       "Beta (2Y)"                                               AS beta_2y,
       "Beta (5Y)"                                               AS beta_5y,
       -- Beta Trend Short-Term (1Y vs 2Y)
       "Beta (1Y)" - "Beta (2Y)"                                 AS beta_trend_short,
       -- Beta Trend Long-Term (2Y vs 5Y)
       "Beta (2Y)" - "Beta (5Y)"                                 AS beta_trend_long,
       -- Beta Stability (inverse of range)
       CASE
           WHEN GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
                LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") > 0
               THEN 1.0 / (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
                           LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)"))
           ELSE 1.0
           END                                                   AS beta_stability,
       -- Beta Regime Change Flag (significant change between periods)
       CASE
           WHEN ABS("Beta (1Y)" - "Beta (5Y)") > 0.3 THEN 1
           ELSE 0
           END                                                   AS beta_regime_change,
       -- Systematic Risk Score (weighted beta)
       "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2 AS systematic_risk_score,
       -- Defensive Stock Flag (beta consistently < 0.8)
       CASE
           WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8
               THEN 1
           ELSE 0
           END                                                   AS defensive_stock_flag,
       -- High Beta Flag (beta consistently > 1.3)
       CASE
           WHEN "Beta (1Y)" > 1.3 AND "Beta (2Y)" > 1.3
               THEN 1
           ELSE 0
           END                                                   AS high_beta_flag
FROM postgres.public.equities;
$$;

alter function calc_beta_risk_features() owner to postgres;

