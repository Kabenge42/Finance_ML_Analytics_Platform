create function calc_growth_features()
    returns TABLE
            (
                ticker                  text,
                revenue_growth_yoy      numeric,
                ebitda_growth_yoy       numeric,
                operating_income_growth numeric,
                fcf_growth              numeric,
                revenue_cagr_5y         numeric,
                forward_revenue_growth  numeric,
                revenue_vs_5y_avg       numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                        AS ticker,
       -- Revenue Growth YoY
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100
           END                                                         AS revenue_growth_yoy,
       -- EBITDA Growth YoY
       CASE
           WHEN ABS("EBITDA (-1FY)") > 0
               THEN ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100
           END                                                         AS ebitda_growth_yoy,
       -- Operating Income Growth
       CASE
           WHEN ABS("Operating Income (FY)") > 0
               THEN ("Operating Income (LTM)" - "Operating Income (FY)") /
                    NULLIF(ABS("Operating Income (FY)"), 0) * 100
           END                                                         AS operating_income_growth,
       -- FCF Growth
       CASE
           WHEN ABS("FCF (FY)") > 0
               THEN ("FCF (LTM)" - "FCF (FY)") / NULLIF(ABS("FCF (FY)"), 0) * 100
           END                                                         AS fcf_growth,
       -- Revenue CAGR 5Y
       "Total Revenues/CAGR (5Y FY)"                                   AS revenue_cagr_5y,
       -- Forward Revenue Growth (Estimate)
       "Revenues - Est YoY % (FY1E)"                                   AS forward_revenue_growth,
       -- Revenue vs 5Y Average (NULLIF handles zero division)
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_vs_5y_avg
FROM postgres.public.equities;
$$;

alter function calc_growth_features() owner to postgres;

