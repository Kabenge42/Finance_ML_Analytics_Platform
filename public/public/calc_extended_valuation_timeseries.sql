create function calc_extended_valuation_timeseries(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                     text,
                ev_sales_qoq_1q          numeric,
                ev_sales_qoq_2q          numeric,
                ev_sales_qoq_3q          numeric,
                ev_sales_qoq_4q          numeric,
                p_e_vs_5y_avg            numeric,
                p_e_percentile_proxy     numeric,
                valuation_mean_reversion numeric,
                ev_ebitda_qoq_trend      numeric,
                p_b_momentum_yoy         numeric,
                valuation_compression    numeric,
                forward_pe_premium       numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                      AS isin,
       ("EV/Sales (LTM)" - "EV/Sales (-1FQLTM)") / NULLIF("EV/Sales (-1FQLTM)", 0) AS ev_sales_qoq_1q,
       ("EV/Sales (-1FQLTM)" - "EV/Sales (-2FQLTM)") / NULLIF("EV/Sales (-2FQLTM)", 0)
                                                                                   AS ev_sales_qoq_2q,
       ("EV/Sales (-2FQLTM)" - "EV/Sales (-3FQLTM)") / NULLIF("EV/Sales (-3FQLTM)", 0)
                                                                                   AS ev_sales_qoq_3q,
       ("EV/Sales (-3FQLTM)" - "EV/Sales (-4FQLTM)") / NULLIF("EV/Sales (-4FQLTM)", 0)
                                                                                   AS ev_sales_qoq_4q,
       ("P/E (LTM)" - "P/E (5YAVGLTM)") / NULLIF("P/E (5YAVGLTM)", 0)              AS p_e_vs_5y_avg,
       CASE
           WHEN "P/E (LTM)" IS NOT NULL AND "P/E (3YAVGLTM)" IS NOT NULL
               THEN ("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF(ABS("P/E (3YAVGLTM)") * 0.5, 0)
           END                                                                     AS p_e_percentile_proxy,
       (("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF("P/E (3YAVGLTM)", 0) +
        ("EV/Sales (LTM)" - "EV/Sales (3YAVGLTM)") / NULLIF("EV/Sales (3YAVGLTM)", 0) +
        ("EV/EBITDA (LTM)" - "EV/EBITDA (3YAVGLTM)") / NULLIF("EV/EBITDA (3YAVGLTM)", 0)) / 3.0
                                                                                   AS valuation_mean_reversion,
       ("EV/EBITDA (LTM)" - "EV/EBITDA (-1FQLTM)") / NULLIF("EV/EBITDA (-1FQLTM)", 0)
                                                                                   AS ev_ebitda_qoq_trend,
       ("P/B (LTM)" - "P/B (-1FY)") / NULLIF("P/B (-1FY)", 0)                      AS p_b_momentum_yoy,
       (("P/E (LTM)" / NULLIF("P/E (3YAVGLTM)", 0)) +
        ("EV/EBITDA (LTM)" / NULLIF("EV/EBITDA (3YAVGLTM)", 0))) / 2.0 - 1.0       AS valuation_compression,
       ("P/E (EST FY1)" - "P/E (LTM)") / NULLIF(ABS("P/E (LTM)"), 0) * 100         AS forward_pe_premium
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_extended_valuation_timeseries(text) owner to postgres;

