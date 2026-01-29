create function calc_valuation_timeseries_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                       text,
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
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                       AS isin,
       calc_change_ratio("EV/Sales (LTM)", "EV/Sales (-1FYLTM)")    AS ev_sales_trend_1y,
       calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (-1FYLTM)")  AS ev_ebitda_momentum,
       calc_change_ratio("P/E (LTM)", "P/E (-1FYLTM)")              AS p_e_momentum_yoy,
       calc_change_ratio("P/E (LTM)", "P/E (-1FQLTM)")              AS p_e_momentum_qoq,
       calc_change_ratio("EV/Sales (LTM)", "EV/Sales (3YAVGLTM)")   AS ev_sales_vs_3y_avg,
       calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)") AS ev_ebitda_vs_3y_avg,
       calc_change_ratio("P/E (LTM)", "P/E (3YAVGLTM)")             AS p_e_vs_3y_avg,
       calc_change_ratio("EV/Sales (NTM)", "EV/Sales (LTM)")        AS ev_sales_forward_discount,
       calc_change_ratio("EV/EBITDA (NTM)", "EV/EBITDA (LTM)")      AS ev_ebitda_forward_discount,
       calc_change_ratio("P/E (EST FY1)", "P/E (LTM)")              AS p_e_forward_discount,
       safe_divide("P/B (LTM)", "P/B (5YAVG)")                      AS p_b_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_valuation_timeseries_features(text) owner to postgres;

