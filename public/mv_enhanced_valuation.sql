create materialized view mv_enhanced_valuation as
SELECT "Ticker"                                                                              AS ticker,
       "ISIN"                                                                                AS isin,
       "Name"                                                                                AS name,
       "Sector"                                                                              AS sector,
       "Industry"                                                                            AS industry,
       "Country"                                                                             AS country,
       "Market Cap"                                                                          AS market_cap,
       "Last Price"                                                                          AS last_price,
       "Forward P/E"                                                                         AS forward_pe,
       "P/E (LTM)"                                                                           AS trailing_pe,
       "PEG Ratio"                                                                           AS peg_ratio,
       "EV/FCF"                                                                              AS ev_to_fcf,
       "EV/EBITDA (LTM)"                                                                     AS ev_ebitda_ltm,
       "FCF (LTM)"                                                                           AS fcf_ltm,
       "Div Yield (LTM)"                                                                     AS div_yield_ltm,
       "Buyback Yield (LTM)"                                                                 AS buyback_yield_ltm,
       "Total Revenues/CAGR (5Y FY)"                                                         AS revenue_cagr_5y,
       "Revenues - Est YoY % (FY1E)"                                                         AS revenue_est_yoy,
       "EPS Growth (TTM)"                                                                    AS eps_growth_ttm,
       ("P/E (LTM)" - "Forward P/E") / NULLIF("P/E (LTM)", 0::numeric) * 100::numeric        AS pe_forward_discount,
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0::numeric
               THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0::numeric)
           ELSE NULL::numeric
           END                                                                               AS peg_adjusted,
       CASE
           WHEN "Revenues - Est YoY % (FY1E)" > 0::numeric
               THEN "Forward P/E" / NULLIF("Revenues - Est YoY % (FY1E)", 0::numeric)
           ELSE NULL::numeric
           END                                                                               AS peg_forward,
       "P/E (LTM)" / NULLIF("EPS Growth (TTM)", 0::numeric)                                  AS pe_to_growth,
       CASE
           WHEN "P/E (LTM)" > 0::numeric THEN 1.0 / "P/E (LTM)" * 100::numeric
           ELSE NULL::numeric
           END                                                                               AS earnings_yield,
       "FCF (LTM)" / NULLIF("Market Cap", 0::numeric) * 100::numeric                         AS fcf_yield,
       COALESCE("Div Yield (LTM)", 0::numeric) + COALESCE("Buyback Yield (LTM)", 0::numeric) AS shareholder_yield_total,
       GREATEST(0, LEAST(100, 50 -
                              CASE
                                  WHEN "P/E (LTM)" < 15::numeric THEN 10
                                  WHEN "P/E (LTM)" < 25::numeric THEN 0
                                  ELSE '-10'::integer
                                  END -
                              CASE
                                  WHEN "PEG Ratio" < 1::numeric THEN 15
                                  WHEN "PEG Ratio" < 2::numeric THEN 5
                                  ELSE '-5'::integer
                                  END -
                              CASE
                                  WHEN "EV/EBITDA (LTM)" < 10::numeric THEN 10
                                  WHEN "EV/EBITDA (LTM)" < 15::numeric THEN 0
                                  ELSE '-10'::integer
                                  END +
                              CASE
                                  WHEN ("FCF (LTM)" / NULLIF("Market Cap", 0::numeric) * 100::numeric) > 5::numeric
                                      THEN 15
                                  ELSE 0
                                  END))                                                      AS valuation_composite_score
FROM equities e;

alter materialized view mv_enhanced_valuation owner to postgres;

