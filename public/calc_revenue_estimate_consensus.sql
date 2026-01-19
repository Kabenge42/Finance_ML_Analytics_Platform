create function calc_revenue_estimate_consensus()
    returns TABLE
            (
                ticker                 text,
                revenue_est_avg_ntm    numeric,
                revenue_est_med_ntm    numeric,
                revenue_est_avg_fy1e   numeric,
                revenue_est_med_fy1e   numeric,
                estimate_skew_ntm      numeric,
                estimate_skew_fy1e     numeric,
                consensus_confidence   numeric,
                upside_to_consensus    numeric,
                estimate_vs_actual_ltm numeric,
                forward_revenue_growth numeric,
                revenue_beat_history   numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                    AS ticker,
       "Revenues - Est Avg (NTM)"                                                  AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"                                                  AS revenue_est_med_ntm,
       "Revenues - Est Avg (FY1E)"                                                 AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"                                                 AS revenue_est_med_fy1e,
       -- Estimate Skew NTM (Avg vs Median, positive = optimistic outliers)
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Med (NTM)") /
       NULLIF("Revenues - Est Med (NTM)", 0) * 100                                 AS estimate_skew_ntm,
       -- Estimate Skew FY1E
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                                AS estimate_skew_fy1e,
       -- Consensus Confidence (lower skew = higher confidence)
       GREATEST(0, LEAST(100,
                         100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                   NULLIF("Revenues - Est Med (FY1E)", 0) * 100))) AS consensus_confidence,
       -- Upside to Consensus (current run-rate vs estimate)
       ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") /
       NULLIF("Total Revenues (LTM)", 0) * 100                                     AS upside_to_consensus,
       -- Estimate vs Actual LTM (how close were estimates)
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF("Revenues - Est Avg (FY1E)", 0) * 100                                AS estimate_vs_actual_ltm,
       -- Forward Revenue Growth (NTM vs LTM)
       ("Revenues - Est Med (NTM)" - "Total Revenues (LTM)") /
       NULLIF("Total Revenues (LTM)", 0) * 100                                     AS forward_revenue_growth,
       -- Revenue Beat History Proxy (actual > estimate)
       CASE
           WHEN "Total Revenues (LTM)" > "Revenues - Est Avg (FY1E)"
               THEN 1.0
           ELSE 0.0
           END                                                                     AS revenue_beat_history
FROM postgres.public.equities;
$$;

alter function calc_revenue_estimate_consensus() owner to postgres;

