create function calc_eps_comprehensive()
    returns TABLE
            (
                ticker                  text,
                eps_basic_fq            numeric,
                eps_basic_ltm           numeric,
                eps_basic_fy            numeric,
                eps_basic_1fy           numeric,
                eps_basic_2fy           numeric,
                eps_basic_3fy           numeric,
                eps_basic_4fy           numeric,
                eps_basic_5fy           numeric,
                eps_basic_1fqfq         numeric,
                eps_basic_2fqfq         numeric,
                eps_basic_3fqfq         numeric,
                eps_basic_4fqfq         numeric,
                eps_cont_fq             numeric,
                eps_cont_ltm            numeric,
                eps_cont_fy             numeric,
                eps_cont_1fy            numeric,
                eps_cont_2fy            numeric,
                eps_cont_3fy            numeric,
                eps_cont_4fy            numeric,
                eps_cont_1fqfq          numeric,
                eps_cont_2fqfq          numeric,
                eps_cont_3fqfq          numeric,
                eps_cont_4fqfq          numeric,
                eps_adj_fq              numeric,
                eps_adj_ltm             numeric,
                eps_adj_fy              numeric,
                eps_adj_1fy             numeric,
                eps_adj_2fy             numeric,
                eps_adj_3fy             numeric,
                eps_adj_4fy             numeric,
                eps_adj_1fqfq           numeric,
                eps_adj_2fqfq           numeric,
                eps_adj_3fqfq           numeric,
                eps_adj_4fqfq           numeric,
                eps_norm_est_ntm        numeric,
                eps_norm_est_fy1e       numeric,
                eps_gaap_est_ntm        numeric,
                eps_gaap_est_fy1e       numeric,
                eps_estimate_count      numeric,
                eps_growth_yoy          numeric,
                eps_growth_qoq          numeric,
                eps_growth_2y           numeric,
                eps_cagr_3y             numeric,
                eps_cagr_5y             numeric,
                eps_growth_acceleration numeric,
                eps_adjustment_ratio    numeric,
                eps_adjustment_pct      numeric,
                gaap_vs_norm_est_spread numeric,
                eps_positive_years      integer,
                eps_positive_quarters   integer,
                eps_improvement_years   integer,
                eps_trajectory_score    numeric,
                eps_stability_score     numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                              AS ticker,
       -- Net EPS Basic Values
       "Net EPS - Basic (FQ)"                                                AS eps_basic_fq,
       "Net EPS - Basic (LTM)"                                               AS eps_basic_ltm,
       "Net EPS - Basic (FY)"                                                AS eps_basic_fy,
       "Net EPS - Basic (-1FY)"                                              AS eps_basic_1fy,
       "Net EPS - Basic (-2FY)"                                              AS eps_basic_2fy,
       "Net EPS - Basic (-3FY)"                                              AS eps_basic_3fy,
       "Net EPS - Basic (-4FY)"                                              AS eps_basic_4fy,
       "Net EPS - Basic (-5FY)"                                              AS eps_basic_5fy,
       "Net EPS - Basic (-1FQFQ)"                                            AS eps_basic_1fqfq,
       "Net EPS - Basic (-2FQFQ)"                                            AS eps_basic_2fqfq,
       "Net EPS - Basic (-3FQFQ)"                                            AS eps_basic_3fqfq,
       "Net EPS - Basic (-4FQFQ)"                                            AS eps_basic_4fqfq,
       -- Basic EPS Continuing Values
       "Basic EPS - Cont (FQ)"                                               AS eps_cont_fq,
       "Basic EPS - Cont (LTM)"                                              AS eps_cont_ltm,
       "Basic EPS - Cont (FY)"                                               AS eps_cont_fy,
       "Basic EPS - Cont (-1FY)"                                             AS eps_cont_1fy,
       "Basic EPS - Cont (-2FY)"                                             AS eps_cont_2fy,
       "Basic EPS - Cont (-3FY)"                                             AS eps_cont_3fy,
       "Basic EPS - Cont (-4FY)"                                             AS eps_cont_4fy,
       "Basic EPS - Cont (-1FQFQ)"                                           AS eps_cont_1fqfq,
       "Basic EPS - Cont (-2FQFQ)"                                           AS eps_cont_2fqfq,
       "Basic EPS - Cont (-3FQFQ)"                                           AS eps_cont_3fqfq,
       "Basic EPS - Cont (-4FQFQ)"                                           AS eps_cont_4fqfq,
       -- EPS Adjusted Values
       "EPS/Adj. (FQ)"                                                       AS eps_adj_fq,
       "EPS/Adj. (LTM)"                                                      AS eps_adj_ltm,
       "EPS/Adj. (FY)"                                                       AS eps_adj_fy,
       "EPS/Adj. (-1FY)"                                                     AS eps_adj_1fy,
       "EPS/Adj. (-2FY)"                                                     AS eps_adj_2fy,
       "EPS/Adj. (-3FY)"                                                     AS eps_adj_3fy,
       "EPS/Adj. (-4FY)"                                                     AS eps_adj_4fy,
       "EPS/Adj. (-1FQFQ)"                                                   AS eps_adj_1fqfq,
       "EPS/Adj. (-2FQFQ)"                                                   AS eps_adj_2fqfq,
       "EPS/Adj. (-3FQFQ)"                                                   AS eps_adj_3fqfq,
       "EPS/Adj. (-4FQFQ)"                                                   AS eps_adj_4fqfq,
       -- Estimates
       "EPS Norm - Est Avg (NTM)"                                            AS eps_norm_est_ntm,
       "EPS Norm - Est Avg (FY1E)"                                           AS eps_norm_est_fy1e,
       "EPS GAAP - Est Avg (NTM)"                                            AS eps_gaap_est_ntm,
       "EPS GAAP - Est Avg (FY1E)"                                           AS eps_gaap_est_fy1e,
       "EPS Norm - Est # (FY1E)"                                             AS eps_estimate_count,
       -- Growth Analytics
       ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
       NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100                        AS eps_growth_yoy,
       ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
       NULLIF(ABS("Net EPS - Basic (-1FQFQ)"), 0) * 100                      AS eps_growth_qoq,
       ("Net EPS - Basic (FY)" - "Net EPS - Basic (-2FY)") /
       NULLIF(ABS("Net EPS - Basic (-2FY)"), 0) * 100                        AS eps_growth_2y,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                               AS eps_cagr_3y,
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                               AS eps_cagr_5y,
       -- Growth Acceleration (3Y CAGR - 5Y CAGR)
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN ((POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                     (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
           END                                                               AS eps_growth_acceleration,
       -- Adjustment Analytics
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0)                 AS eps_adjustment_ratio,
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
       NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100                         AS eps_adjustment_pct,
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"             AS gaap_vs_norm_est_spread,
       -- Consistency Metrics
       (CASE WHEN "Net EPS - Basic (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER   AS eps_positive_years,
       (CASE WHEN "Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_quarters,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                             AS eps_improvement_years,
       -- Trajectory Score (0-100)
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                     AS eps_trajectory_score,
       -- Stability Score (based on coefficient of variation proxy)
       GREATEST(0, LEAST(100,
                         100 - ABS("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
                               NULLIF(GREATEST(ABS("Net EPS - Basic (FY)"), 0.01), 0) * 20
                   ))                                                        AS eps_stability_score
FROM postgres.public.equities;
$$;

alter function calc_eps_comprehensive() owner to postgres;

