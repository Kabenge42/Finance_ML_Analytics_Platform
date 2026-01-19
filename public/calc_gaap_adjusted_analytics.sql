create function calc_gaap_adjusted_analytics()
    returns TABLE
            (
                ticker                      text,
                eps_adjustment_spread       numeric,
                eps_adjustment_pct          numeric,
                net_income_adjustment_ratio numeric,
                net_income_adjustment_pct   numeric,
                ebitda_adjustment_pct       numeric,
                earnings_quality_score      numeric,
                earnings_quality_warning    integer,
                forward_eps_gaap_adj_spread numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                     AS ticker,
       -- EPS Adjustment Spread (Adjusted - GAAP, dollar difference)
       "EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)"                                   AS eps_adjustment_spread,

       -- EPS Adjustment Percentage
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
       NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100                                AS eps_adjustment_pct,

       -- Net Income Adjustment Ratio (Adjusted / GAAP)
       "Net Income/Adj. (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS net_income_adjustment_ratio,

       -- Net Income Adjustment Percentage
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100                              AS net_income_adjustment_pct,

       -- EBITDA Adjustment Percentage
       ("EBITDA/Adj. (LTM)" - "EBITDA (LTM)") /
       NULLIF(ABS("EBITDA (LTM)"), 0) * 100                                         AS ebitda_adjustment_pct,

       -- Earnings Quality Score (100 - adjustment %, capped at 0-100, higher = better)
       GREATEST(0, LEAST(100,
                         100 - ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                                   NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100))) AS earnings_quality_score,

       -- Warning flag if adjustment exceeds 15%
       CASE
           WHEN ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                    NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100) > 15
               THEN 1
           ELSE 0
           END                                                                      AS earnings_quality_warning,

       -- Forward EPS GAAP vs Adjusted Spread
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"                    AS forward_eps_gaap_adj_spread

FROM postgres.public.equities;
$$;

alter function calc_gaap_adjusted_analytics() owner to postgres;

