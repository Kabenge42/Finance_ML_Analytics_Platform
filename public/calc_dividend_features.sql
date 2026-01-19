create function calc_dividend_features()
    returns TABLE
            (
                ticker                      text,
                dividend_streak             integer,
                dividend_yield_ltm          numeric,
                dividend_yield_ntm          numeric,
                dividend_payout_ratio       numeric,
                fcf_dividend_coverage       numeric,
                buyback_yield               numeric,
                total_shareholder_yield     numeric,
                dividend_growth_expectation numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                AS ticker,
       "Dividend Streak"::INTEGER                                              AS dividend_streak,
       "Div Yield (LTM)"                                                       AS dividend_yield_ltm,
       "Div Yield (NTM)"                                                       AS dividend_yield_ntm,
       -- Dividend Payout Ratio (NULLIF handles zero division)
       ABS("Common Dividends Paid (LTM)") / NULLIF("Net Income/Adj. (LTM)", 0) AS dividend_payout_ratio,
       -- FCF Dividend Coverage
       CASE
           WHEN ABS("Common Dividends Paid (LTM)") > 0
               THEN "FCF (LTM)" / NULLIF(ABS("Common Dividends Paid (LTM)"), 0)
           END                                                                 AS fcf_dividend_coverage,
       -- Buyback Yield
       "Buyback Yield (LTM)"                                                   AS buyback_yield,
       -- Total Shareholder Yield
       COALESCE("Buyback Yield (LTM)", 0) + COALESCE("Div Yield (LTM)", 0)     AS total_shareholder_yield,
       -- Dividend Growth Expectation
       "Div Yield (NTM)" - "Div Yield (LTM)"                                   AS dividend_growth_expectation
FROM postgres.public.equities;
$$;

alter function calc_dividend_features() owner to postgres;

