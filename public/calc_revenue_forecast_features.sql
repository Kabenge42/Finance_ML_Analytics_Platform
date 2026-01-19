create function calc_revenue_forecast_features()
    returns TABLE
            (
                ticker                     text,
                revenue_est_spread         numeric,
                revenue_beat_potential     numeric,
                revenue_est_revision_trend numeric,
                ebitda_est_vs_actual       numeric,
                forward_revenue_multiple   numeric,
                revenue_estimate_count     numeric,
                revenue_guidance_gap       numeric,
                consensus_revenue_growth   numeric,
                ebit_estimate_spread       numeric,
                forward_ebitda_margin      numeric,
                revenue_acceleration       numeric,
                estimate_confidence_score  numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                 AS ticker,
       -- Revenue Estimate Spread (High - Low) / Median
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                             AS revenue_est_spread,

       -- Revenue Beat Potential (Current vs Estimate)
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_beat_potential,

       -- Revenue Estimate Revision Trend (use forward growth as proxy)
       "Revenues - Est YoY % (FY1E)"                                            AS revenue_est_revision_trend,

       -- EBITDA Estimate vs Actual
       ("EBITDA (LTM)" - "EBITDA - Est Avg (FY1E)") /
       NULLIF(ABS("EBITDA - Est Avg (FY1E)"), 0) * 100                          AS ebitda_est_vs_actual,

       -- Forward Revenue Multiple (EV / Forward Revenue)
       "Enterprise Value" / NULLIF("Revenues - Est Avg (FY1E)", 0)              AS forward_revenue_multiple,

       -- Revenue Estimate Count (proxy: use EPS estimate count)
       "EPS Norm - Est # (FY1E)"                                                AS revenue_estimate_count,

       -- Revenue Guidance Gap (NTM vs FY1E difference)
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_guidance_gap,

       -- Consensus Revenue Growth Expectation
       ("Revenues - Est Avg (FY1E)" - "Total Revenues (FY)") /
       NULLIF(ABS("Total Revenues (FY)"), 0) * 100                              AS consensus_revenue_growth,

       -- EBIT Estimate Spread
       ("EBIT - Est Med (FY1E)" - "EBIT - Est Med (NTM)") /
       NULLIF(ABS("EBIT - Est Med (NTM)"), 0) * 100                             AS ebit_estimate_spread,

       -- Forward EBITDA Margin (Estimated EBITDA / Estimated Revenue)
       "EBITDA - Est Avg (FY1E)" / NULLIF("Revenues - Est Avg (FY1E)", 0) * 100 AS forward_ebitda_margin,

       -- Revenue Acceleration (Forward growth vs historical growth)
       "Revenues - Est YoY % (FY1E)" - "Total Revenues/CAGR (5Y FY)"            AS revenue_acceleration,

       -- Estimate Confidence Score (narrower spread = higher confidence)
       GREATEST(0, LEAST(100,
                         100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                   NULLIF("Revenues - Est Med (FY1E)", 0) * 100)
                   ))                                                           AS estimate_confidence_score

FROM postgres.public.equities;
$$;

alter function calc_revenue_forecast_features() owner to postgres;

