create function calc_margin_trends()
    returns TABLE
            (
                ticker                 text,
                gross_margin_trend_yoy numeric,
                operating_margin_trend numeric,
                net_margin_trend_yoy   numeric,
                ebitda_margin_trend    numeric,
                margin_expansion_flag  integer,
                margin_stability_score numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                             AS ticker,
       -- Gross Margin Trend YoY
       ("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)")       AS gross_margin_trend_yoy,

       -- Operating Margin Trend (LTM vs FY)
       (("Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("Operating Income (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100 AS operating_margin_trend,

       -- Net Margin Trend YoY
       ("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")           AS net_margin_trend_yoy,

       -- EBITDA Margin Trend
       (("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100           AS ebitda_margin_trend,

       -- Margin Expansion Flag (all margins improving)
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               AND "Net Income Margin % (LTM)" > "Net Income Margin % (FY)"
               AND ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) >
                   ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))
               THEN 1
           ELSE 0
           END                                                              AS margin_expansion_flag,

       -- Margin Stability Score (inverse of margin volatility, 0-100)
       GREATEST(0, LEAST(100,
                         100 - (ABS("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)") +
                                ABS("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")) / 2
                   ))                                                       AS margin_stability_score

FROM postgres.public.equities;
$$;

alter function calc_margin_trends() owner to postgres;

