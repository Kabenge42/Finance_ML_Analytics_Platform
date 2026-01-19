create function calc_eps_trajectory_features()
    returns TABLE
            (
                ticker                text,
                eps_qoq_growth        numeric,
                eps_yoy_quarterly     numeric,
                eps_positive_streak   integer,
                eps_cagr_3y           numeric,
                eps_cagr_5y           numeric,
                eps_growth_accel      numeric,
                eps_vs_5y_avg         numeric,
                eps_improvement_count integer,
                eps_trajectory_score  numeric,
                eps_stability         numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                              AS ticker,
       -- Quarter-over-Quarter EPS Growth
       CASE
           WHEN ABS("Net EPS - Basic (-1FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-1FQFQ)"), 0) * 100
           END                                                               AS eps_qoq_growth,

       -- Year-over-Year Quarterly EPS Growth
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0) * 100
           END                                                               AS eps_yoy_quarterly,

       -- EPS Positive Streak (count of positive quarters out of last 5)
       (CASE WHEN "Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_streak,

       -- EPS CAGR 3Y (compound annual growth rate)
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                               AS eps_cagr_3y,

       -- EPS CAGR 5Y
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                               AS eps_cagr_5y,

       -- Growth Acceleration (3Y CAGR - 5Y CAGR, positive = accelerating growth)
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (-5FY)" > 0
               AND "Net EPS - Basic (FY)" > 0
               THEN ((POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                     (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
           END                                                               AS eps_growth_accel,

       -- EPS vs 5-Year Average (current EPS as percentage deviation from 5Y average)
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN ("Net EPS - Basic (FY)" -
                     (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                       "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0)) /
                    NULLIF(ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                                "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0), 0) * 100
           END                                                               AS eps_vs_5y_avg,

       -- EPS Improvement Count (years with YoY improvement out of last 5)
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                             AS eps_improvement_count,

       -- EPS Trajectory Score (% of improving years, scaled to 0-100)
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                     AS eps_trajectory_score,

       -- EPS Stability (inverse of coefficient of variation, higher = more stable)
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN 1 - (STDDEV(val) / NULLIF(ABS(AVG(val)), 0))
           END                                                               AS eps_stability

FROM postgres.public.equities,
     LATERAL (VALUES ("Net EPS - Basic (FY)"),
                     ("Net EPS - Basic (-1FY)"),
                     ("Net EPS - Basic (-2FY)"),
                     ("Net EPS - Basic (-3FY)"),
                     ("Net EPS - Basic (-4FY)")) AS t(val)
GROUP BY "Ticker", "Net EPS - Basic (FQ)", "Net EPS - Basic (-1FQFQ)", "Net EPS - Basic (-2FQFQ)",
         "Net EPS - Basic (-3FQFQ)", "Net EPS - Basic (-4FQFQ)", "Net EPS - Basic (FY)",
         "Net EPS - Basic (-1FY)", "Net EPS - Basic (-2FY)", "Net EPS - Basic (-3FY)",
         "Net EPS - Basic (-4FY)", "Net EPS - Basic (-5FY)";
$$;

alter function calc_eps_trajectory_features() owner to postgres;

