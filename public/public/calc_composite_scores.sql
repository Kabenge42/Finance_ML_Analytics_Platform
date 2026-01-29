create function calc_composite_scores(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                   text,
                piotroski_f_score      integer,
                eps_trajectory_score   numeric,
                dilution_score         numeric,
                quality_momentum_score numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"            AS isin,
       (CASE WHEN "Return on Assets (ROA) % (LTM)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (LTM)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Return on Assets (ROA) % (LTM)" > "Return on Assets (ROA) % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "CFO (LTM)" > "Net Income - (IS) (LTM)" THEN 1 ELSE 0 END +
        CASE
            WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) <
                 "Total Debt (FY)" / NULLIF("Total Equity (FY)", 0) THEN 1
            ELSE 0 END +
        CASE WHEN "Current Ratio (LTM)" > "Current Ratio (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Shrs Out" <= "Shrs Out (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Asset Turnover (LTM)" > "Asset Turnover (FY)" THEN 1 ELSE 0 END
           )::INTEGER    AS piotroski_f_score,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100 AS eps_trajectory_score,
       GREATEST(0, LEAST(100,
                         50 - (("Shrs Out" - "Shrs Out (-1FY)") / NULLIF("Shrs Out (-1FY)", 0)) * 100
                   ))    AS dilution_score,
       (((CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
          CASE WHEN "Return On Equity % (LTM)" > 15 THEN 25 ELSE 0 END +
          CASE WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) < 1 THEN 25 ELSE 0 END +
          CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END) * 0.40) +
        (LEAST(100, GREATEST(0,
                             (("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100 + 50))) * 0.30) +
        (CASE
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 20 THEN 100
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 10 THEN 75
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 0 THEN 50
             ELSE 25
             END * 0.30)
           )             AS quality_momentum_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_composite_scores(text) owner to postgres;

