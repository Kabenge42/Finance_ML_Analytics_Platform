create function calc_enhanced_cashflow_features()
    returns TABLE
            (
                ticker                  text,
                fcf_positive_years      integer,
                fcf_always_positive     integer,
                capex_vs_5y_avg         numeric,
                underinvestment_flag    integer,
                cfo_share_of_cf         numeric,
                cfi_share_of_cf         numeric,
                cff_share_of_cf         numeric,
                self_funding_flag       integer,
                acquisition_to_fcf      numeric,
                sustainable_ma_flag     integer,
                fcf_4q_improvement      numeric,
                cash_flow_quality_score numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                          AS ticker,
       -- FCF Positive Years (count over 5 years)
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER           AS fcf_positive_years,

       -- FCF Always Positive Flag (all 5 years positive)
       CASE
           WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
               AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0
               THEN 1
           ELSE 0
           END                                                           AS fcf_always_positive,

       -- CapEx vs 5Y Average (investment consistency)
       ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0)
                                                                         AS capex_vs_5y_avg,

       -- Underinvestment Flag (CapEx < 70% of historical average)
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) < 0.7
               THEN 1
           ELSE 0
           END                                                           AS underinvestment_flag,

       -- CFO Share of Total Cash Flow
       ABS("CFO (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfo_share_of_cf,

       -- CFI Share of Total Cash Flow
       ABS("CFI (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfi_share_of_cf,

       -- CFF Share of Total Cash Flow
       ABS("CFF (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cff_share_of_cf,

       -- Self-Funding Flag (CFO covers CFI needs)
       CASE
           WHEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0) > 1
               THEN 1
           ELSE 0
           END                                                           AS self_funding_flag,

       -- Acquisition to FCF Ratio (M&A sustainability)
       (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
       NULLIF(ABS("FCF (LTM)"), 0)                                       AS acquisition_to_fcf,

       -- Sustainable M&A Flag (acquisitions < 50% of FCF)
       CASE
           WHEN (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
                NULLIF(ABS("FCF (LTM)"), 0) < 0.5
               THEN 1
           ELSE 0
           END                                                           AS sustainable_ma_flag,

       -- FCF 4Q Improvement (most recent vs oldest quarter)
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0)    AS fcf_4q_improvement,

       -- Composite Cash Flow Quality Score (0-100)
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC           AS cash_flow_quality_score

FROM postgres.public.equities;
$$;

alter function calc_enhanced_cashflow_features() owner to postgres;

