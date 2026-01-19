create function calc_earnings_features()
    returns TABLE
            (
                ticker                  text,
                eps_surprise_pct        numeric,
                revenue_surprise_pct    numeric,
                eps_adjustment_ratio    numeric,
                gaap_adj_eps_gap_pct    numeric,
                ebitda_adjustment_ratio numeric,
                eps_quarterly_trend     numeric,
                eps_yoy_growth          numeric
            )
    language sql
as
$$
SELECT "Ticker"                                              AS ticker,
       -- EPS Surprise (Actual vs Estimate)
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS/Adj. (LTM)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS eps_surprise_pct,
       -- Revenue Surprise
       CASE
           WHEN ABS("Revenues - Est Avg (FY1E)") > 0
               THEN ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
                    NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100
           END                                               AS revenue_surprise_pct,
       -- EPS Adjustment Ratio (Adjusted / GAAP) - NULLIF handles zero division
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0) AS eps_adjustment_ratio,
       -- GAAP vs Adjusted EPS Gap
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS gaap_adj_eps_gap_pct,
       -- EBITDA Adjustment Ratio
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0)       AS ebitda_adjustment_ratio,
       -- EPS Quarterly Trend (FQ vs -4FQFQ for YoY)
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0)
           END                                               AS eps_quarterly_trend,
       -- EPS YoY Growth
       CASE
           WHEN ABS("Net EPS - Basic (-1FY)") > 0
               THEN ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
                    NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100
           END                                               AS eps_yoy_growth
FROM postgres.public.equities;
$$;

alter function calc_earnings_features() owner to postgres;

