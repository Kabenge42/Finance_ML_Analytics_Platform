create function calc_valuation_timeseries_features()
    returns TABLE
            (
                ticker                     text,
                ev_sales_trend_1y          numeric,
                ev_ebitda_momentum         numeric,
                p_e_momentum_yoy           numeric,
                p_e_momentum_qoq           numeric,
                ev_sales_vs_3y_avg         numeric,
                ev_ebitda_vs_3y_avg        numeric,
                p_e_vs_3y_avg              numeric,
                ev_sales_forward_discount  numeric,
                ev_ebitda_forward_discount numeric,
                p_e_forward_discount       numeric,
                p_b_vs_5y_avg              numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                         AS ticker,
       -- EV/Sales 1Y Trend (NULLIF handles zero division, returning NULL)
       ("EV/Sales (LTM)" - "EV/Sales (-1FYLTM)") / NULLIF("EV/Sales (-1FYLTM)", 0)      AS ev_sales_trend_1y,
       -- EV/EBITDA Momentum
       ("EV/EBITDA (LTM)" - "EV/EBITDA (-1FYLTM)") / NULLIF("EV/EBITDA (-1FYLTM)", 0)   AS ev_ebitda_momentum,
       -- P/E Momentum YoY
       ("P/E (LTM)" - "P/E (-1FYLTM)") / NULLIF("P/E (-1FYLTM)", 0)                     AS p_e_momentum_yoy,
       -- P/E Momentum QoQ
       ("P/E (LTM)" - "P/E (-1FQLTM)") / NULLIF("P/E (-1FQLTM)", 0)                     AS p_e_momentum_qoq,
       -- Mean Reversion Features
       ("EV/Sales (LTM)" - "EV/Sales (3YAVGLTM)") / NULLIF("EV/Sales (3YAVGLTM)", 0)    AS ev_sales_vs_3y_avg,
       ("EV/EBITDA (LTM)" - "EV/EBITDA (3YAVGLTM)") / NULLIF("EV/EBITDA (3YAVGLTM)", 0) AS ev_ebitda_vs_3y_avg,
       ("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF("P/E (3YAVGLTM)", 0)                   AS p_e_vs_3y_avg,
       -- Forward vs Trailing Discount
       ("EV/Sales (NTM)" - "EV/Sales (LTM)") / NULLIF("EV/Sales (LTM)", 0)              AS ev_sales_forward_discount,
       ("EV/EBITDA (NTM)" - "EV/EBITDA (LTM)") / NULLIF("EV/EBITDA (LTM)", 0)           AS ev_ebitda_forward_discount,
       ("P/E (EST FY1)" - "P/E (LTM)") / NULLIF("P/E (LTM)", 0)                         AS p_e_forward_discount,
       -- P/B vs 5Y Average
       "P/B (LTM)" / NULLIF("P/B (5YAVG)", 0)                                           AS p_b_vs_5y_avg
FROM postgres.public.equities;
$$;

alter function calc_valuation_timeseries_features() owner to postgres;

