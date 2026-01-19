create function calc_enhanced_valuation_ratios()
    returns TABLE
            (
                ticker                    text,
                forward_pe                numeric,
                trailing_pe               numeric,
                pe_forward_discount       numeric,
                peg_ratio                 numeric,
                peg_adjusted              numeric,
                peg_forward               numeric,
                pe_to_growth              numeric,
                ev_to_fcf                 numeric,
                earnings_yield            numeric,
                fcf_yield                 numeric,
                shareholder_yield_total   numeric,
                valuation_composite_score numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                            AS ticker,
       "Forward P/E"                                                       AS forward_pe,
       "P/E (LTM)"                                                         AS trailing_pe,
       -- Forward P/E Discount vs Trailing
       ("P/E (LTM)" - "Forward P/E") / NULLIF("P/E (LTM)", 0) * 100        AS pe_forward_discount,
       -- PEG Ratio (pre-calculated)
       "PEG Ratio"                                                         AS peg_ratio,
       -- PEG Adjusted (using 5Y revenue CAGR instead)
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0
               THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0)
           END                                                             AS peg_adjusted,
       -- PEG Forward (Forward P/E / Forward Growth)
       CASE
           WHEN "Revenues - Est YoY % (FY1E)" > 0
               THEN "Forward P/E" / NULLIF("Revenues - Est YoY % (FY1E)", 0)
           END                                                             AS peg_forward,
       -- P/E to Growth (simpler ratio)
       "P/E (LTM)" / NULLIF("EPS Growth (TTM)", 0)                         AS pe_to_growth,
       -- EV to FCF
       "EV/FCF"                                                            AS ev_to_fcf,
       -- Earnings Yield (inverse of P/E)
       CASE
           WHEN "P/E (LTM)" > 0
               THEN 1.0 / "P/E (LTM)" * 100
           END                                                             AS earnings_yield,
       -- FCF Yield
       "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                         AS fcf_yield,
       -- Total Shareholder Yield (Dividend + Buyback)
       COALESCE("Div Yield (LTM)", 0) + COALESCE("Buyback Yield (LTM)", 0) AS shareholder_yield_total,
       -- Valuation Composite Score (lower = cheaper, 0-100)
       GREATEST(0, LEAST(100,
                         50 -
                         (CASE WHEN "P/E (LTM)" < 15 THEN 10 WHEN "P/E (LTM)" < 25 THEN 0 ELSE -10 END) -
                         (CASE WHEN "PEG Ratio" < 1 THEN 15 WHEN "PEG Ratio" < 2 THEN 5 ELSE -5 END) -
                         (CASE WHEN "EV/EBITDA (LTM)" < 10 THEN 10 WHEN "EV/EBITDA (LTM)" < 15 THEN 0 ELSE -10 END) +
                         (CASE WHEN "FCF (LTM)" / NULLIF("Market Cap", 0) * 100 > 5 THEN 15 ELSE 0 END)
                   ))                                                      AS valuation_composite_score
FROM postgres.public.equities;
$$;

alter function calc_enhanced_valuation_ratios() owner to postgres;

