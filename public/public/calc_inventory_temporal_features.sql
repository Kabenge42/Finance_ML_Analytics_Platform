create function calc_inventory_temporal_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                     text,
                inventory_ltm            numeric,
                inventory_fq             numeric,
                inventory_fy             numeric,
                inventory_1fq            numeric,
                inventory_2fq            numeric,
                inventory_3fq            numeric,
                inventory_4fq            numeric,
                inventory_1fy            numeric,
                inventory_2fy            numeric,
                inventory_3fy            numeric,
                inventory_4fy            numeric,
                inventory_qoq_change     numeric,
                inventory_yoy_change     numeric,
                inventory_4q_trend       numeric,
                inventory_vs_5y_avg      numeric,
                inventory_days           numeric,
                inventory_turnover       numeric,
                inventory_to_revenue     numeric,
                inventory_to_assets      numeric,
                inventory_buildup_flag   integer,
                inventory_reduction_flag integer,
                inventory_volatility     numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                          AS isin,
       -- Current values
       "Inventory (LTM)"                                               AS inventory_ltm,
       "Inventory (FQ)"                                                AS inventory_fq,
       "Inventory (FY)"                                                AS inventory_fy,
       -- Quarterly historical
       "Inventory (-1FQ)"                                              AS inventory_1fq,
       "Inventory (-2FQ)"                                              AS inventory_2fq,
       "Inventory (-3FQ)"                                              AS inventory_3fq,
       "Inventory (-4FQ)"                                              AS inventory_4fq,
       -- Yearly historical
       "Inventory (-1FY)"                                              AS inventory_1fy,
       "Inventory (-2FY)"                                              AS inventory_2fy,
       "Inventory (-3FY)"                                              AS inventory_3fy,
       "Inventory (-4FY)"                                              AS inventory_4fy,
       -- Trend metrics
       pct_change("Inventory (FQ)", "Inventory (-1FQ)")                AS inventory_qoq_change,
       pct_change("Inventory (FY)", "Inventory (-1FY)")                AS inventory_yoy_change,
       pct_change("Inventory (FQ)", "Inventory (-4FQ)")                AS inventory_4q_trend,
       safe_divide("Inventory (FQ)", "Inventory (5YAVGFQ)")            AS inventory_vs_5y_avg,
       -- Efficiency metrics
       "Inventory (LTM)" / NULLIF("Cost Of Revenues (LTM)" / 365.0, 0) AS inventory_days,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)         AS inventory_turnover,
       safe_divide("Inventory (LTM)", "Total Revenues (LTM)") * 100    AS inventory_to_revenue,
       safe_divide("Inventory (LTM)", "Total Assets (LTM)") * 100      AS inventory_to_assets,
       -- Inventory buildup flag (rising faster than revenue)
       CASE
           WHEN pct_change("Inventory (FQ)", "Inventory (-4FQ)") >
                pct_change("Total Revenues (FQ)", "Total Revenues (-4FQFQ)") + 10
               THEN 1
           ELSE 0 END                                                  AS inventory_buildup_flag,
       -- Inventory reduction flag (declining)
       CASE
           WHEN "Inventory (FQ)" < "Inventory (-1FQ)"
               AND "Inventory (-1FQ)" < "Inventory (-2FQ)"
               THEN 1
           ELSE 0 END                                                  AS inventory_reduction_flag,
       -- Volatility (coefficient of variation)
       (ABS("Inventory (FQ)" - "Inventory (-1FQ)") +
        ABS("Inventory (-1FQ)" - "Inventory (-2FQ)") +
        ABS("Inventory (-2FQ)" - "Inventory (-3FQ)") +
        ABS("Inventory (-3FQ)" - "Inventory (-4FQ)")) /
       NULLIF(ABS(("Inventory (FQ)" + "Inventory (-1FQ)" + "Inventory (-2FQ)" +
                   "Inventory (-3FQ)" + "Inventory (-4FQ)") / 5.0), 0) AS inventory_volatility
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_inventory_temporal_features(text) owner to postgres;

