create function calc_profitability_features()
    returns TABLE
            (
                ticker               text,
                roe                  numeric,
                roa                  numeric,
                gross_margin_pct     numeric,
                operating_margin_pct numeric,
                net_margin_pct       numeric,
                ebitda_margin_pct    numeric,
                roic                 numeric,
                rnd_intensity        numeric,
                equity_multiplier    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                               AS ticker,
       "Return On Equity % (LTM)"                                                             AS roe,
       "Return on Assets (ROA) % (LTM)"                                                       AS roa,
       "Gross Profit Margin % (LTM)"                                                          AS gross_margin_pct,
       -- Operating Margin (NULLIF handles zero division)
       "Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                     AS operating_margin_pct,
       "Net Income Margin % (LTM)"                                                            AS net_margin_pct,
       -- EBITDA Margin
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                               AS ebitda_margin_pct,
       -- ROIC: Net Income / (Total Equity + Total Debt)
       "Net Income - (IS) (LTM)" / NULLIF("Total Equity (LTM)" + "Total Debt (LTM)", 0) * 100 AS roic,
       -- R&D Intensity
       "R&D Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)                               AS rnd_intensity,
       -- Equity Multiplier (DuPont)
       "Total Assets (LTM)" / NULLIF("Total Equity (LTM)", 0)                                 AS equity_multiplier
FROM postgres.public.equities;
$$;

alter function calc_profitability_features() owner to postgres;

