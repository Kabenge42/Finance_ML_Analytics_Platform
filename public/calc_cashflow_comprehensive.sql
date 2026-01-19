create function calc_cashflow_comprehensive()
    returns TABLE
            (
                ticker                   text,
                cfo_fq                   numeric,
                cfo_ltm                  numeric,
                cfo_fy                   numeric,
                cfo_1fy                  numeric,
                cfo_2fy                  numeric,
                cfo_3fy                  numeric,
                cfo_4fy                  numeric,
                cfo_1fqfq                numeric,
                cfo_2fqfq                numeric,
                cfo_3fqfq                numeric,
                cfo_4fqfq                numeric,
                cfi_fq                   numeric,
                cfi_ltm                  numeric,
                cfi_fy                   numeric,
                cfi_1fy                  numeric,
                cfi_2fy                  numeric,
                cfi_3fy                  numeric,
                cfi_4fy                  numeric,
                cfi_1fqfq                numeric,
                cfi_2fqfq                numeric,
                cfi_3fqfq                numeric,
                cfi_4fqfq                numeric,
                cff_fq                   numeric,
                cff_ltm                  numeric,
                cff_fy                   numeric,
                cff_1fy                  numeric,
                cff_2fy                  numeric,
                cff_3fy                  numeric,
                cff_4fy                  numeric,
                cff_1fqfq                numeric,
                cff_2fqfq                numeric,
                cff_3fqfq                numeric,
                cff_4fqfq                numeric,
                fcf_fq                   numeric,
                fcf_ltm                  numeric,
                fcf_fy                   numeric,
                fcf_1fy                  numeric,
                fcf_2fy                  numeric,
                fcf_3fy                  numeric,
                fcf_4fy                  numeric,
                fcf_1fqfq                numeric,
                fcf_2fqfq                numeric,
                fcf_3fqfq                numeric,
                fcf_4fqfq                numeric,
                fcf_5yavg                numeric,
                acquisitions_fq          numeric,
                acquisitions_ltm         numeric,
                acquisitions_fy          numeric,
                acquisitions_1fy         numeric,
                acquisitions_1fqfq       numeric,
                acquisitions_2fqfq       numeric,
                acquisitions_3fqfq       numeric,
                acquisitions_4fqfq       numeric,
                acquisitions_5yavg       numeric,
                capex_fq                 numeric,
                capex_ltm                numeric,
                capex_fy                 numeric,
                capex_1fy                numeric,
                capex_5yavg              numeric,
                cfo_growth_yoy           numeric,
                cfo_growth_qoq           numeric,
                fcf_growth_yoy           numeric,
                fcf_growth_qoq           numeric,
                cfo_cagr_3y              numeric,
                fcf_cagr_3y              numeric,
                cfo_to_net_income        numeric,
                fcf_to_net_income        numeric,
                fcf_margin               numeric,
                fcf_yield                numeric,
                cfo_positive_years       integer,
                cfo_positive_quarters    integer,
                fcf_positive_years       integer,
                fcf_positive_quarters    integer,
                cfi_negative_years       integer,
                cff_capital_return_flag  integer,
                self_funding_ratio       numeric,
                acquisition_intensity_4q numeric,
                capex_vs_5y_avg          numeric,
                cash_flow_quality_score  numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                             AS ticker,
       -- CFO Values
       "CFO (FQ)"                                                           AS cfo_fq,
       "CFO (LTM)"                                                          AS cfo_ltm,
       "CFO (FY)"                                                           AS cfo_fy,
       "CFO (-1FY)"                                                         AS cfo_1fy,
       "CFO (-2FY)"                                                         AS cfo_2fy,
       "CFO (-3FY)"                                                         AS cfo_3fy,
       "CFO (-4FY)"                                                         AS cfo_4fy,
       "CFO (-1FQFQ)"                                                       AS cfo_1fqfq,
       "CFO (-2FQFQ)"                                                       AS cfo_2fqfq,
       "CFO (-3FQFQ)"                                                       AS cfo_3fqfq,
       "CFO (-4FQFQ)"                                                       AS cfo_4fqfq,
       -- CFI Values
       "CFI (FQ)"                                                           AS cfi_fq,
       "CFI (LTM)"                                                          AS cfi_ltm,
       "CFI (FY)"                                                           AS cfi_fy,
       "CFI (-1FY)"                                                         AS cfi_1fy,
       "CFI (-2FY)"                                                         AS cfi_2fy,
       "CFI (-3FY)"                                                         AS cfi_3fy,
       "CFI (-4FY)"                                                         AS cfi_4fy,
       "CFI (-1FQFQ)"                                                       AS cfi_1fqfq,
       "CFI (-2FQFQ)"                                                       AS cfi_2fqfq,
       "CFI (-3FQFQ)"                                                       AS cfi_3fqfq,
       "CFI (-4FQFQ)"                                                       AS cfi_4fqfq,
       -- CFF Values
       "CFF (FQ)"                                                           AS cff_fq,
       "CFF (LTM)"                                                          AS cff_ltm,
       "CFF (FY)"                                                           AS cff_fy,
       "CFF (-1FY)"                                                         AS cff_1fy,
       "CFF (-2FY)"                                                         AS cff_2fy,
       "CFF (-3FY)"                                                         AS cff_3fy,
       "CFF (-4FY)"                                                         AS cff_4fy,
       "CFF (-1FQFQ)"                                                       AS cff_1fqfq,
       "CFF (-2FQFQ)"                                                       AS cff_2fqfq,
       "CFF (-3FQFQ)"                                                       AS cff_3fqfq,
       "CFF (-4FQFQ)"                                                       AS cff_4fqfq,
       -- FCF Values
       "FCF (FQ)"                                                           AS fcf_fq,
       "FCF (LTM)"                                                          AS fcf_ltm,
       "FCF (FY)"                                                           AS fcf_fy,
       "FCF (-1FY)"                                                         AS fcf_1fy,
       "FCF (-2FY)"                                                         AS fcf_2fy,
       "FCF (-3FY)"                                                         AS fcf_3fy,
       "FCF (-4FY)"                                                         AS fcf_4fy,
       "FCF (-1FQFQ)"                                                       AS fcf_1fqfq,
       "FCF (-2FQFQ)"                                                       AS fcf_2fqfq,
       "FCF (-3FQFQ)"                                                       AS fcf_3fqfq,
       "FCF (-4FQFQ)"                                                       AS fcf_4fqfq,
       "FCF (5YAVGFQ)"                                                      AS fcf_5yavg,
       -- Cash Acquisitions
       "Cash Acquisitions (FQ)"                                             AS acquisitions_fq,
       "Cash Acquisitions (LTM)"                                            AS acquisitions_ltm,
       "Cash Acquisitions (FY)"                                             AS acquisitions_fy,
       "Cash Acquisitions (-1FY)"                                           AS acquisitions_1fy,
       "Cash Acquisitions (-1FQFQ)"                                         AS acquisitions_1fqfq,
       "Cash Acquisitions (-2FQFQ)"                                         AS acquisitions_2fqfq,
       "Cash Acquisitions (-3FQFQ)"                                         AS acquisitions_3fqfq,
       "Cash Acquisitions (-4FQFQ)"                                         AS acquisitions_4fqfq,
       "Cash Acquisitions (5YAVGFQ)"                                        AS acquisitions_5yavg,
       -- CapEx
       "Capital Expenditure (FQ)"                                           AS capex_fq,
       "Capital Expenditure (LTM)"                                          AS capex_ltm,
       "Capital Expenditure (FY)"                                           AS capex_fy,
       "Capital Expenditure (-1FY)"                                         AS capex_1fy,
       "Capital Expenditure (5YAVGFQ)"                                      AS capex_5yavg,
       -- Growth Trends
       ("CFO (FY)" - "CFO (-1FY)") / NULLIF(ABS("CFO (-1FY)"), 0) * 100     AS cfo_growth_yoy,
       ("CFO (FQ)" - "CFO (-1FQFQ)") / NULLIF(ABS("CFO (-1FQFQ)"), 0) * 100 AS cfo_growth_qoq,
       ("FCF (FY)" - "FCF (-1FY)") / NULLIF(ABS("FCF (-1FY)"), 0) * 100     AS fcf_growth_yoy,
       ("FCF (FQ)" - "FCF (-1FQFQ)") / NULLIF(ABS("FCF (-1FQFQ)"), 0) * 100 AS fcf_growth_qoq,
       CASE
           WHEN "CFO (-3FY)" > 0 AND "CFO (FY)" > 0
               THEN (POWER("CFO (FY)" / NULLIF("CFO (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                              AS cfo_cagr_3y,
       CASE
           WHEN "FCF (-3FY)" > 0 AND "FCF (FY)" > 0
               THEN (POWER("FCF (FY)" / NULLIF("FCF (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                              AS fcf_cagr_3y,
       -- Quality Metrics
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)                   AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)                   AS fcf_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0)                      AS fcf_margin,
       "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                          AS fcf_yield,
       -- Consistency Metrics
       (CASE WHEN "CFO (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER              AS cfo_positive_years,
       (CASE WHEN "CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS cfo_positive_quarters,
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER              AS fcf_positive_years,
       (CASE WHEN "FCF (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS fcf_positive_quarters,
       (CASE WHEN "CFI (FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-1FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-2FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-3FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-4FY)" < 0 THEN 1 ELSE 0 END)::INTEGER              AS cfi_negative_years,
       -- Pattern Analysis
       CASE
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") < 0 THEN 1
           ELSE 0
           END                                                              AS cff_capital_return_flag,
       "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0)                            AS self_funding_ratio,
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) + ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))                       AS acquisition_intensity_4q,
       ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0)
                                                                            AS capex_vs_5y_avg,
       -- Cash Flow Quality Score (0-100)
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0 AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0
                THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC              AS cash_flow_quality_score
FROM postgres.public.equities;
$$;

alter function calc_cashflow_comprehensive() owner to postgres;

