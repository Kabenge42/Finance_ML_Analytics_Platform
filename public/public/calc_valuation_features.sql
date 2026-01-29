create function calc_valuation_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin            text,
                p_e_ratio       numeric,
                p_b_ratio       numeric,
                ev_ebitda_ratio numeric,
                ev_sales_ratio  numeric,
                dividend_yield  numeric,
                peg_ratio       numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"            AS isin,
       "P/E (LTM)"       AS p_e_ratio,
       "P/B (LTM)"       AS p_b_ratio,
       "EV/EBITDA (LTM)" AS ev_ebitda_ratio,
       "EV/Sales (LTM)"  AS ev_sales_ratio,
       "Div Yield (LTM)" AS dividend_yield,
       CASE
           WHEN "Net EPS - Basic (FY)" > 0 AND "Net EPS - Basic (-3FY)" > 0
               THEN safe_divide(
                   "P/E (LTM)",
                   (POWER(
                            safe_divide("Net EPS - Basic (FY)", "Net EPS - Basic (-3FY)"),
                            1.0 / 3.0
                    ) - 1) * 100
                    )
           END           AS peg_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_valuation_features(text) owner to postgres;

