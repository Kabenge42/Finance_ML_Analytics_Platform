create function calc_cashflow_temporal_features()
    returns TABLE
            (
                ticker                text,
                cfo_quarterly_trend   numeric,
                cfo_yoy_quarterly     numeric,
                cfi_quarterly_trend   numeric,
                cff_quarterly_trend   numeric,
                fcf_quarterly_trend   numeric,
                cfo_positive_quarters integer,
                cfi_negative_quarters integer,
                cff_pattern_score     numeric,
                cash_burn_rate        numeric,
                cf_volatility_score   numeric,
                operating_cf_momentum numeric,
                financing_dependency  numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                             AS ticker,
       -- CFO Quarterly Trend (FQ vs -4FQFQ YoY)
       ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100 AS cfo_quarterly_trend,

       -- CFO YoY Quarterly Growth
       CASE
           WHEN ABS("CFO (-4FQFQ)") > 0
               THEN ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100
           END                                                              AS cfo_yoy_quarterly,

       -- CFI Quarterly Trend
       ("CFI (FQ)" - "CFI (-4FQFQ)") / NULLIF(ABS("CFI (-4FQFQ)"), 0) * 100 AS cfi_quarterly_trend,

       -- CFF Quarterly Trend
       ("CFF (FQ)" - "CFF (-4FQFQ)") / NULLIF(ABS("CFF (-4FQFQ)"), 0) * 100 AS cff_quarterly_trend,

       -- FCF Quarterly Trend
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0) * 100 AS fcf_quarterly_trend,

       -- CFO Positive Quarters (count of last 5)
       (CASE WHEN "CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS cfo_positive_quarters,

       -- CFI Negative Quarters (normal for investing companies)
       (CASE WHEN "CFI (FQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-1FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-2FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-3FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-4FQFQ)" < 0 THEN 1 ELSE 0 END)::INTEGER            AS cfi_negative_quarters,

       -- CFF Pattern Score (positive = raising capital, negative = returning capital)
       CASE
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") > 0
               THEN -1 -- Capital raising (potentially dilutive)
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") < 0
               THEN 1 -- Capital return (buybacks/dividends)
           ELSE 0
           END::NUMERIC                                                     AS cff_pattern_score,

       -- Cash Burn Rate (negative FCF / Cash, monthly)
       CASE
           WHEN "FCF (LTM)" < 0
               THEN ABS("FCF (LTM)") / NULLIF("Cash And Equivalents (FQ)", 0) / 12.0
           ELSE 0
           END                                                              AS cash_burn_rate,

       -- CF Volatility Score (std dev proxy across quarters)
       (ABS("CFO (FQ)" - "CFO (-1FQFQ)") + ABS("CFO (-1FQFQ)" - "CFO (-2FQFQ)") +
        ABS("CFO (-2FQFQ)" - "CFO (-3FQFQ)") + ABS("CFO (-3FQFQ)" - "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (FQ)" + "CFO (-1FQFQ)" + "CFO (-2FQFQ)" +
                  "CFO (-3FQFQ)" + "CFO (-4FQFQ)") / 5.0, 0)                AS cf_volatility_score,

       -- Operating CF Momentum (recent 2Q vs older 2Q)
       (("CFO (FQ)" + "CFO (-1FQFQ)") - ("CFO (-3FQFQ)" + "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (-3FQFQ)" + "CFO (-4FQFQ)"), 0) * 100                AS operating_cf_momentum,

       -- Financing Dependency (CFF / CFO, higher = more dependent)
       ABS("CFF (LTM)") / NULLIF(ABS("CFO (LTM)"), 0)                       AS financing_dependency

FROM postgres.public.equities;
$$;

alter function calc_cashflow_temporal_features() owner to postgres;

