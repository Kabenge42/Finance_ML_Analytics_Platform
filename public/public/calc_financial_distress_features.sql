create function calc_financial_distress_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                     text,
                distress_risk_score      numeric,
                liquidity_stress_score   numeric,
                working_capital_trend    numeric,
                cash_runway_months       numeric,
                combined_distress_score  numeric,
                wc_deteriorating_flag    integer,
                retained_earnings_growth numeric,
                accumulated_deficit_flag integer,
                adequate_cash_buffer     integer
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                            AS isin,
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100))) AS distress_risk_score,
       CASE
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
           ELSE 0.0
           END                                                                           AS liquidity_stress_score,
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0)                                            AS working_capital_trend,
       "Cash And Equivalents (FQ)" /
       NULLIF("Total Operating Expenses (LTM)" / 12.0, 0)                                AS cash_runway_months,
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 70) +
                         (100 - CASE
                                    WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
                                    WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
                                    ELSE 0.0
                             END) * 0.30))                                               AS combined_distress_score,
       CASE
           WHEN ("Working Capital (FQ)" - "Working Capital (FY)") /
                NULLIF(ABS("Working Capital (FY)"), 0) < -0.2
               THEN 1
           ELSE 0
           END                                                                           AS wc_deteriorating_flag,
       ("Retained Earnings (FQ)" - "Retained Earnings (FY)") /
       NULLIF(ABS("Retained Earnings (FY)"), 0)                                          AS retained_earnings_growth,
       CASE WHEN "Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END                          AS accumulated_deficit_flag,
       CASE
           WHEN "Cash And Equivalents (FQ)" /
                NULLIF("Total Operating Expenses (LTM)" / 12.0, 0) > 6
               THEN 1
           ELSE 0
           END                                                                           AS adequate_cash_buffer
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_financial_distress_features(text) owner to postgres;

