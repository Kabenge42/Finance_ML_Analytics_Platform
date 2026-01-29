create function calc_total_debt_temporal(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                 text,
                debt_fq              numeric,
                debt_fy              numeric,
                debt_ltm             numeric,
                debt_1fq             numeric,
                debt_2fq             numeric,
                debt_3fq             numeric,
                debt_4fq             numeric,
                debt_1fy             numeric,
                debt_2fy             numeric,
                debt_3fy             numeric,
                debt_4fy             numeric,
                debt_qoq_change      numeric,
                debt_yoy_change      numeric,
                debt_4q_trend        numeric,
                debt_3y_cagr         numeric,
                debt_deleveraging    integer,
                debt_to_equity_trend numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                           AS isin,
       -- Current values
       "Total Debt (FQ)"                                                AS debt_fq,
       "Total Debt (FY)"                                                AS debt_fy,
       "Total Debt (LTM)"                                               AS debt_ltm,
       -- Quarterly historical
       "Total Debt (-1FQ)"                                              AS debt_1fq,
       "Total Debt (-2FQ)"                                              AS debt_2fq,
       "Total Debt (-3FQ)"                                              AS debt_3fq,
       "Total Debt (-4FQ)"                                              AS debt_4fq,
       -- Yearly historical
       "Total Debt (-1FY)"                                              AS debt_1fy,
       "Total Debt (-2FY)"                                              AS debt_2fy,
       "Total Debt (-3FY)"                                              AS debt_3fy,
       "Total Debt (-4FY)"                                              AS debt_4fy,
       -- Trend metrics
       pct_change("Total Debt (FQ)", "Total Debt (-1FQ)")               AS debt_qoq_change,
       pct_change("Total Debt (FY)", "Total Debt (-1FY)")               AS debt_yoy_change,
       pct_change("Total Debt (FQ)", "Total Debt (-4FQ)")               AS debt_4q_trend,
       CASE
           WHEN "Total Debt (-3FY)" > 0
               THEN (POWER(safe_divide("Total Debt (FY)", "Total Debt (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                          AS debt_3y_cagr,
       CASE
           WHEN "Total Debt (FQ)" < "Total Debt (-1FQ)"
               AND "Total Debt (-1FQ)" < "Total Debt (-2FQ)"
               THEN 1
           ELSE 0 END                                                   AS debt_deleveraging,
       safe_divide("Total Debt (FY)", "Total Equity (FY)") -
       safe_divide("Total Debt (-1FY)", NULLIF("Total Equity (FY)", 0)) AS debt_to_equity_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_total_debt_temporal(text) owner to postgres;

