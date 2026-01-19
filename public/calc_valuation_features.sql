create function calc_valuation_features()
    returns TABLE
            (
                ticker          text,
                p_e_ratio       numeric,
                p_b_ratio       numeric,
                ev_ebitda_ratio numeric,
                ev_sales_ratio  numeric,
                dividend_yield  numeric,
                peg_ratio       numeric
            )
    language sql
as
$$
SELECT "Ticker"          AS ticker,
       "P/E (LTM)"       AS p_e_ratio,
       "P/B (LTM)"       AS p_b_ratio,
       "EV/EBITDA (LTM)" AS ev_ebitda_ratio,
       "EV/Sales (LTM)"  AS ev_sales_ratio,
       "Div Yield (LTM)" AS dividend_yield,
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0
               THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0)
           END           AS peg_ratio
FROM postgres.public.equities;
$$;

alter function calc_valuation_features() owner to postgres;

