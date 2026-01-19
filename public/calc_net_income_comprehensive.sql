create function calc_net_income_comprehensive()
    returns TABLE
            (
                ticker                       text,
                net_income_is_fq             numeric,
                net_income_is_ltm            numeric,
                net_income_is_fy             numeric,
                net_income_is_1fy            numeric,
                net_income_is_2fy            numeric,
                net_income_is_3fy            numeric,
                net_income_is_4fy            numeric,
                net_income_is_1fqfq          numeric,
                net_income_is_2fqfq          numeric,
                net_income_is_3fqfq          numeric,
                net_income_is_4fqfq          numeric,
                net_income_is_5yavg_fq       numeric,
                net_income_is_5yavg_ltm      numeric,
                net_income_adj_fq            numeric,
                net_income_adj_ltm           numeric,
                net_income_adj_fy            numeric,
                net_income_adj_1fy           numeric,
                net_income_adj_2fy           numeric,
                net_income_adj_3fy           numeric,
                net_income_adj_4fy           numeric,
                net_income_adj_1fqfq         numeric,
                net_income_adj_2fqfq         numeric,
                net_income_adj_3fqfq         numeric,
                net_income_adj_4fqfq         numeric,
                net_income_adj_5yavg         numeric,
                normalized_ni_fq             numeric,
                normalized_ni_ltm            numeric,
                normalized_ni_fy             numeric,
                normalized_ni_1fy            numeric,
                normalized_ni_2fy            numeric,
                normalized_ni_3fy            numeric,
                normalized_ni_4fy            numeric,
                normalized_ni_1fqfq          numeric,
                normalized_ni_2fqfq          numeric,
                normalized_ni_3fqfq          numeric,
                normalized_ni_4fqfq          numeric,
                normalized_ni_5yavg_fq       numeric,
                normalized_ni_5yavg_ltm      numeric,
                net_income_growth_yoy        numeric,
                net_income_growth_qoq        numeric,
                normalized_ni_growth_yoy     numeric,
                net_income_cagr_3y           numeric,
                net_income_margin_ltm        numeric,
                net_income_margin_fy         numeric,
                net_income_margin_trend      numeric,
                ni_adjustment_ratio          numeric,
                ni_normalization_ratio       numeric,
                gaap_vs_adj_spread_pct       numeric,
                net_income_positive_years    integer,
                net_income_improvement_count integer,
                net_income_positive_quarters integer,
                earnings_quality_composite   numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                AS ticker,
       -- Net Income (IS) Values
       "Net Income - (IS) (FQ)"                                                AS net_income_is_fq,
       "Net Income - (IS) (LTM)"                                               AS net_income_is_ltm,
       "Net Income - (IS) (FY)"                                                AS net_income_is_fy,
       "Net Income - (IS) (-1FY)"                                              AS net_income_is_1fy,
       "Net Income - (IS) (-2FY)"                                              AS net_income_is_2fy,
       "Net Income - (IS) (-3FY)"                                              AS net_income_is_3fy,
       "Net Income - (IS) (-4FY)"                                              AS net_income_is_4fy,
       "Net Income - (IS) (-1FQFQ)"                                            AS net_income_is_1fqfq,
       "Net Income - (IS) (-2FQFQ)"                                            AS net_income_is_2fqfq,
       "Net Income - (IS) (-3FQFQ)"                                            AS net_income_is_3fqfq,
       "Net Income - (IS) (-4FQFQ)"                                            AS net_income_is_4fqfq,
       "Net Income - (IS) (5YAVGFQ)"                                           AS net_income_is_5yavg_fq,
       "Net Income - (IS) (5YAVGLTM)"                                          AS net_income_is_5yavg_ltm,
       -- Net Income Adjusted Values
       "Net Income/Adj. (FQ)"                                                  AS net_income_adj_fq,
       "Net Income/Adj. (LTM)"                                                 AS net_income_adj_ltm,
       "Net Income/Adj. (FY)"                                                  AS net_income_adj_fy,
       "Net Income/Adj. (-1FY)"                                                AS net_income_adj_1fy,
       "Net Income/Adj. (-2FY)"                                                AS net_income_adj_2fy,
       "Net Income/Adj. (-3FY)"                                                AS net_income_adj_3fy,
       "Net Income/Adj. (-4FY)"                                                AS net_income_adj_4fy,
       "Net Income/Adj. (-1FQFQ)"                                              AS net_income_adj_1fqfq,
       "Net Income/Adj. (-2FQFQ)"                                              AS net_income_adj_2fqfq,
       "Net Income/Adj. (-3FQFQ)"                                              AS net_income_adj_3fqfq,
       "Net Income/Adj. (-4FQFQ)"                                              AS net_income_adj_4fqfq,
       "Net Income/Adj. (5YAVGFQ)"                                             AS net_income_adj_5yavg,
       -- Normalized Net Income Values
       "Normalized Net Income (FQ)"                                            AS normalized_ni_fq,
       "Normalized Net Income (LTM)"                                           AS normalized_ni_ltm,
       "Normalized Net Income (FY)"                                            AS normalized_ni_fy,
       "Normalized Net Income (-1FY)"                                          AS normalized_ni_1fy,
       "Normalized Net Income (-2FY)"                                          AS normalized_ni_2fy,
       "Normalized Net Income (-3FY)"                                          AS normalized_ni_3fy,
       "Normalized Net Income (-4FY)"                                          AS normalized_ni_4fy,
       "Normalized Net Income (-1FQFQ)"                                        AS normalized_ni_1fqfq,
       "Normalized Net Income (-2FQFQ)"                                        AS normalized_ni_2fqfq,
       "Normalized Net Income (-3FQFQ)"                                        AS normalized_ni_3fqfq,
       "Normalized Net Income (-4FQFQ)"                                        AS normalized_ni_4fqfq,
       "Normalized Net Income (5YAVGFQ)"                                       AS normalized_ni_5yavg_fq,
       "Normalized Net Income (5YAVGLTM)"                                      AS normalized_ni_5yavg_ltm,
       -- Growth Analytics
       ("Net Income - (IS) (FY)" - "Net Income - (IS) (-1FY)") /
       NULLIF(ABS("Net Income - (IS) (-1FY)"), 0) * 100                        AS net_income_growth_yoy,
       ("Net Income - (IS) (FQ)" - "Net Income - (IS) (-1FQFQ)") /
       NULLIF(ABS("Net Income - (IS) (-1FQFQ)"), 0) * 100                      AS net_income_growth_qoq,
       ("Normalized Net Income (FY)" - "Normalized Net Income (-1FY)") /
       NULLIF(ABS("Normalized Net Income (-1FY)"), 0) * 100                    AS normalized_ni_growth_yoy,
       CASE
           WHEN "Net Income - (IS) (-3FY)" > 0 AND "Net Income - (IS) (FY)" > 0
               THEN (POWER("Net Income - (IS) (FY)" / NULLIF("Net Income - (IS) (-3FY)", 0), 1.0 / 3.0) - 1) *
                    100
           END                                                                 AS net_income_cagr_3y,
       -- Margins
       "Net Income Margin % (LTM)"                                             AS net_income_margin_ltm,
       "Net Income Margin % (FY)"                                              AS net_income_margin_fy,
       "Net Income Margin % (LTM)" - "Net Income Margin % (FY)"                AS net_income_margin_trend,
       -- Adjustment Analytics
       "Net Income/Adj. (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)          AS ni_adjustment_ratio,
       "Normalized Net Income (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)    AS ni_normalization_ratio,
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100                         AS gaap_vs_adj_spread_pct,
       -- Consistency Metrics
       (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER   AS net_income_positive_years,
       (CASE WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FY)" > "Net Income - (IS) (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FY)" > "Net Income - (IS) (-4FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                               AS net_income_improvement_count,
       (CASE WHEN "Net Income - (IS) (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS net_income_positive_quarters,
       -- Earnings Quality Composite (100 = best)
       GREATEST(0, LEAST(100,
                         50 +
                         (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 10 ELSE -10 END) +
                         (CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 5 ELSE -5 END) +
                         (CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 5 ELSE -5 END) +
                         (CASE
                              WHEN ABS(("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
                                       NULLIF(ABS("Net Income - (IS) (LTM)"), 0)) < 0.10 THEN 15
                              ELSE -15 END) +
                         (CASE WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 10 ELSE -5 END) +
                         (CASE WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 5 ELSE -5 END)
                   ))                                                          AS earnings_quality_composite
FROM postgres.public.equities;
$$;

alter function calc_net_income_comprehensive() owner to postgres;

