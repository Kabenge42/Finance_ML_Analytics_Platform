create function calc_ebit_ebitda_comprehensive()
    returns TABLE
            (
                ticker                   text,
                ebit_fq                  numeric,
                ebit_ltm                 numeric,
                ebit_fy                  numeric,
                ebit_1fy                 numeric,
                ebit_2fy                 numeric,
                ebit_3fy                 numeric,
                ebit_4fy                 numeric,
                ebit_1fqfq               numeric,
                ebit_2fqfq               numeric,
                ebit_3fqfq               numeric,
                ebit_4fqfq               numeric,
                ebit_5yavg               numeric,
                ebit_adj_fq              numeric,
                ebit_adj_ltm             numeric,
                ebit_adj_fy              numeric,
                ebit_adj_1fy             numeric,
                ebit_adj_2fy             numeric,
                ebit_adj_3fy             numeric,
                ebit_adj_4fy             numeric,
                ebit_adj_1fqfq           numeric,
                ebit_adj_2fqfq           numeric,
                ebit_adj_3fqfq           numeric,
                ebit_adj_4fqfq           numeric,
                ebitda_fq                numeric,
                ebitda_ltm               numeric,
                ebitda_fy                numeric,
                ebitda_1fy               numeric,
                ebitda_2fy               numeric,
                ebitda_3fy               numeric,
                ebitda_4fy               numeric,
                ebitda_1fqfq             numeric,
                ebitda_2fqfq             numeric,
                ebitda_3fqfq             numeric,
                ebitda_4fqfq             numeric,
                ebitda_5yavg_fq          numeric,
                ebitda_5yavg_ltm         numeric,
                ebitda_adj_fq            numeric,
                ebitda_adj_ltm           numeric,
                ebitda_adj_fy            numeric,
                ebitda_adj_1fy           numeric,
                ebitda_adj_2fy           numeric,
                ebitda_adj_3fy           numeric,
                ebitda_adj_4fy           numeric,
                ebitda_adj_1fqfq         numeric,
                ebitda_adj_2fqfq         numeric,
                ebitda_adj_3fqfq         numeric,
                ebitda_adj_4fqfq         numeric,
                ebit_growth_yoy          numeric,
                ebit_growth_qoq          numeric,
                ebitda_growth_yoy        numeric,
                ebitda_growth_qoq        numeric,
                ebit_cagr_3y             numeric,
                ebitda_cagr_3y           numeric,
                ebit_margin_ltm          numeric,
                ebit_margin_fy           numeric,
                ebitda_margin_ltm        numeric,
                ebitda_margin_fy         numeric,
                ebit_margin_trend        numeric,
                ebitda_margin_trend      numeric,
                ebit_adjustment_ratio    numeric,
                ebitda_adjustment_ratio  numeric,
                ebit_positive_years      integer,
                ebitda_positive_years    integer,
                ebit_improvement_count   integer,
                ebitda_improvement_count integer
            )
    language sql
as
$$
SELECT "Ticker"                                                                  AS ticker,
       -- EBIT Raw Values
       "EBIT (FQ)"                                                               AS ebit_fq,
       "EBIT (LTM)"                                                              AS ebit_ltm,
       "EBIT (FY)"                                                               AS ebit_fy,
       "EBIT (-1FY)"                                                             AS ebit_1fy,
       "EBIT (-2FY)"                                                             AS ebit_2fy,
       "EBIT (-3FY)"                                                             AS ebit_3fy,
       "EBIT (-4FY)"                                                             AS ebit_4fy,
       "EBIT (-1FQFQ)"                                                           AS ebit_1fqfq,
       "EBIT (-2FQFQ)"                                                           AS ebit_2fqfq,
       "EBIT (-3FQFQ)"                                                           AS ebit_3fqfq,
       "EBIT (-4FQFQ)"                                                           AS ebit_4fqfq,
       "EBIT (5YAVGFQ)"                                                          AS ebit_5yavg,
       -- EBIT Adjusted Values
       "EBIT/Adj. (FQ)"                                                          AS ebit_adj_fq,
       "EBIT/Adj. (LTM)"                                                         AS ebit_adj_ltm,
       "EBIT/Adj. (FY)"                                                          AS ebit_adj_fy,
       "EBIT/Adj. (-1FY)"                                                        AS ebit_adj_1fy,
       "EBIT/Adj. (-2FY)"                                                        AS ebit_adj_2fy,
       "EBIT/Adj. (-3FY)"                                                        AS ebit_adj_3fy,
       "EBIT/Adj. (-4FY)"                                                        AS ebit_adj_4fy,
       "EBIT/Adj. (-1FQFQ)"                                                      AS ebit_adj_1fqfq,
       "EBIT/Adj. (-2FQFQ)"                                                      AS ebit_adj_2fqfq,
       "EBIT/Adj. (-3FQFQ)"                                                      AS ebit_adj_3fqfq,
       "EBIT/Adj. (-4FQFQ)"                                                      AS ebit_adj_4fqfq,
       -- EBITDA Raw Values
       "EBITDA (FQ)"                                                             AS ebitda_fq,
       "EBITDA (LTM)"                                                            AS ebitda_ltm,
       "EBITDA (FY)"                                                             AS ebitda_fy,
       "EBITDA (-1FY)"                                                           AS ebitda_1fy,
       "EBITDA (-2FY)"                                                           AS ebitda_2fy,
       "EBITDA (-3FY)"                                                           AS ebitda_3fy,
       "EBITDA (-4FY)"                                                           AS ebitda_4fy,
       "EBITDA (-1FQFQ)"                                                         AS ebitda_1fqfq,
       "EBITDA (-2FQFQ)"                                                         AS ebitda_2fqfq,
       "EBITDA (-3FQFQ)"                                                         AS ebitda_3fqfq,
       "EBITDA (-4FQFQ)"                                                         AS ebitda_4fqfq,
       "EBITDA (5YAVGFQ)"                                                        AS ebitda_5yavg_fq,
       "EBITDA (5YAVGLTM)"                                                       AS ebitda_5yavg_ltm,
       -- EBITDA Adjusted Values
       "EBITDA/Adj. (FQ)"                                                        AS ebitda_adj_fq,
       "EBITDA/Adj. (LTM)"                                                       AS ebitda_adj_ltm,
       "EBITDA/Adj. (FY)"                                                        AS ebitda_adj_fy,
       "EBITDA/Adj. (-1FY)"                                                      AS ebitda_adj_1fy,
       "EBITDA/Adj. (-2FY)"                                                      AS ebitda_adj_2fy,
       "EBITDA/Adj. (-3FY)"                                                      AS ebitda_adj_3fy,
       "EBITDA/Adj. (-4FY)"                                                      AS ebitda_adj_4fy,
       "EBITDA/Adj. (-1FQFQ)"                                                    AS ebitda_adj_1fqfq,
       "EBITDA/Adj. (-2FQFQ)"                                                    AS ebitda_adj_2fqfq,
       "EBITDA/Adj. (-3FQFQ)"                                                    AS ebitda_adj_3fqfq,
       "EBITDA/Adj. (-4FQFQ)"                                                    AS ebitda_adj_4fqfq,
       -- Growth Trends
       ("EBIT (FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) * 100       AS ebit_growth_yoy,
       ("EBIT (FQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) * 100   AS ebit_growth_qoq,
       ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100 AS ebitda_growth_yoy,
       ("EBITDA (FQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) *
       100                                                                       AS ebitda_growth_qoq,
       -- CAGR (3Y)
       CASE
           WHEN "EBIT (-3FY)" > 0 AND "EBIT (FY)" > 0
               THEN (POWER("EBIT (FY)" / NULLIF("EBIT (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebit_cagr_3y,
       CASE
           WHEN "EBITDA (-3FY)" > 0 AND "EBITDA (FY)" > 0
               THEN (POWER("EBITDA (FY)" / NULLIF("EBITDA (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebitda_cagr_3y,
       -- Margins
       "EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                    AS ebit_margin_ltm,
       "EBIT (FY)" / NULLIF("Total Revenues (FY)", 0) * 100                      AS ebit_margin_fy,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                  AS ebitda_margin_ltm,
       "EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0) * 100                    AS ebitda_margin_fy,
       ("EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
       ("EBIT (FY)" / NULLIF("Total Revenues (FY)", 0))                          AS ebit_margin_trend,
       ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
       ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))                        AS ebitda_margin_trend,
       -- Adjustment Analytics
       "EBIT/Adj. (LTM)" / NULLIF("EBIT (LTM)", 0)                               AS ebit_adjustment_ratio,
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0)                           AS ebitda_adjustment_ratio,
       -- Consistency Metrics
       (CASE WHEN "EBIT (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                  AS ebit_positive_years,
       (CASE WHEN "EBITDA (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                AS ebitda_positive_years,
       (CASE WHEN "EBIT (FY)" > "EBIT (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-1FY)" > "EBIT (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-2FY)" > "EBIT (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-3FY)" > "EBIT (-4FY)" THEN 1 ELSE 0 END)::INTEGER      AS ebit_improvement_count,
       (CASE WHEN "EBITDA (FY)" > "EBITDA (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-1FY)" > "EBITDA (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-2FY)" > "EBITDA (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-3FY)" > "EBITDA (-4FY)" THEN 1 ELSE 0 END)::INTEGER  AS ebitda_improvement_count
FROM postgres.public.equities;
$$;

alter function calc_ebit_ebitda_comprehensive() owner to postgres;

