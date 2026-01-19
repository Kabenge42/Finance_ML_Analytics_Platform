create function calc_financial_distress_features()
    returns TABLE
            (
                ticker                   text,
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
    language sql
as
$$
SELECT "Ticker"                                                                          AS ticker,
       -- Distress Risk Score (map Z-score to 0-100: z<=1.8 → 0, z>=3.0 → 100)
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100))) AS distress_risk_score,

       -- Liquidity Stress Score (higher = more stress)
       CASE
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
           ELSE 0.0
           END                                                                           AS liquidity_stress_score,

       -- Working Capital Trend (QoQ change)
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0)                                            AS working_capital_trend,

       -- Cash Runway (months of OpEx coverage)
       "Cash And Equivalents (FQ)" /
       NULLIF("Total Operating Expenses (LTM)" / 12.0, 0)                                AS cash_runway_months,

       -- Combined Distress Score (70% Z-score + 30% liquidity)
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 70) +
                         (100 - CASE
                                    WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
                                    WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
                                    ELSE 0.0
                             END) * 0.30))                                               AS combined_distress_score,

       -- Working Capital Deteriorating Flag
       CASE
           WHEN ("Working Capital (FQ)" - "Working Capital (FY)") /
                NULLIF(ABS("Working Capital (FY)"), 0) < -0.2
               THEN 1
           ELSE 0
           END                                                                           AS wc_deteriorating_flag,

       -- Retained Earnings Growth (FQ vs FY)
       ("Retained Earnings (FQ)" - "Retained Earnings (FY)") /
       NULLIF(ABS("Retained Earnings (FY)"), 0)                                          AS retained_earnings_growth,

       -- Accumulated Deficit Flag (negative retained earnings)
       CASE WHEN "Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END                          AS accumulated_deficit_flag,

       -- Adequate Cash Buffer (> 6 months runway)
       CASE
           WHEN "Cash And Equivalents (FQ)" /
                NULLIF("Total Operating Expenses (LTM)" / 12.0, 0) > 6
               THEN 1
           ELSE 0
           END                                                                           AS adequate_cash_buffer

FROM postgres.public.equities;
$$;

alter function calc_financial_distress_features() owner to postgres;

