create materialized view mv_estimate_consensus as
SELECT "Ticker"                                                     AS ticker,
       "ISIN"                                                       AS isin,
       "Name"                                                       AS name,
       "Sector"                                                     AS sector,
       "Industry"                                                   AS industry,
       "Country"                                                    AS country,
       "Market Cap"                                                 AS market_cap,
       "Revenues - Est Avg (NTM)"                                   AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"                                   AS revenue_est_med_ntm,
       "Revenues - Est Avg (FY1E)"                                  AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"                                  AS revenue_est_med_fy1e,
       "Total Revenues (LTM)"                                       AS revenue_ltm,
       "Revenues - Est YoY % (FY1E)"                                AS revenue_est_yoy,
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Med (NTM)") / NULLIF("Revenues - Est Med (NTM)", 0::numeric) *
       100::numeric                                                 AS estimate_skew_ntm,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") / NULLIF("Revenues - Est Med (FY1E)", 0::numeric) *
       100::numeric                                                 AS estimate_skew_fy1e,
       GREATEST(0::numeric, LEAST(100::numeric, 100::numeric -
                                                abs(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                                    NULLIF("Revenues - Est Med (FY1E)", 0::numeric) *
                                                    100::numeric))) AS consensus_confidence,
       ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                 AS upside_to_consensus,
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") / NULLIF("Revenues - Est Avg (FY1E)", 0::numeric) *
       100::numeric                                                 AS estimate_vs_actual_ltm,
       ("Revenues - Est Med (NTM)" - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                                 AS forward_revenue_growth,
       CASE
           WHEN "Total Revenues (LTM)" > "Revenues - Est Avg (FY1E)" THEN 1.0
           ELSE 0.0
           END                                                      AS revenue_beat_history
FROM equities e;

alter materialized view mv_estimate_consensus owner to postgres;

