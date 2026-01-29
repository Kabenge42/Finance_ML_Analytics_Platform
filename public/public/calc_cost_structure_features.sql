create function calc_cost_structure_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                     text,
                cogs_to_revenue          numeric,
                opex_to_revenue          numeric,
                sga_to_revenue           numeric,
                rnd_to_revenue           numeric,
                interest_to_revenue      numeric,
                sga_trend_yoy            numeric,
                operating_leverage_proxy numeric,
                cost_efficiency_score    numeric,
                marketing_to_revenue     numeric,
                marketing_trend_yoy      numeric,
                marketing_vs_5y_avg      numeric,
                sga_vs_5y_avg            numeric,
                sga_efficiency_trend     numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                      AS isin,
       safe_divide("Cost Of Revenues (LTM)", "Total Revenues (LTM)") * 100         AS cogs_to_revenue,
       safe_divide("Total Operating Expenses (LTM)", "Total Revenues (LTM)") * 100 AS opex_to_revenue,
       safe_divide("Selling General & Admin Expenses/Total (FY)", "Total Revenues (FY)") * 100
                                                                                   AS sga_to_revenue,
       safe_divide("R&D Expenses (LTM)", "Total Revenues (LTM)") * 100             AS rnd_to_revenue,
       safe_divide("Interest Expense/Total (LTM)", "Total Revenues (LTM)") * 100   AS interest_to_revenue,
       -- SG&A trend using available FY columns
       (safe_divide("Selling General & Admin Expenses/Total (FY)", "Total Revenues (FY)") -
        safe_divide("Selling General & Admin Expenses/Total (-1FY)", "Total Revenues (-1FY)")) * 100
                                                                                   AS sga_trend_yoy,
       CASE
           WHEN calc_change_ratio("Total Revenues (FY)", "Total Revenues (-1FY)") > 0
               THEN safe_divide(
                   calc_change_ratio("Operating Income (FY)", "Operating Income (-1FY)"),
                   calc_change_ratio("Total Revenues (FY)", "Total Revenues (-1FY)")
                    )
           END                                                                     AS operating_leverage_proxy,
       clamp_score(
               100 - safe_divide("Cost Of Revenues (LTM)", "Total Revenues (LTM)") * 100 * 0.5 -
               safe_divide("Total Operating Expenses (LTM)", "Total Revenues (LTM)") * 100 * 0.3
       )                                                                           AS cost_efficiency_score,
       -- NEW: Marketing efficiency metrics using schema columns
       safe_divide("Marketing Expenses (FY)", "Total Revenues (FY)") * 100         AS marketing_to_revenue,
       pct_change("Marketing Expenses (FY)", "Marketing Expenses (-1FY)")          AS marketing_trend_yoy,
       safe_divide("Marketing Expenses (FY)", "Marketing Expenses (5YAVGLTM)")     AS marketing_vs_5y_avg,
       -- NEW: SG&A vs 5Y average
       safe_divide("Selling General & Admin Expenses/Total (FQ)",
                   "Selling General & Admin Expenses/Total (5YAVGFQ)")             AS sga_vs_5y_avg,
       -- NEW: SG&A efficiency trend (lower ratio = better efficiency)
       (safe_divide("Selling General & Admin Expenses/Total (-1FY)", "Total Revenues (-1FY)") -
        safe_divide("Selling General & Admin Expenses/Total (FY)", "Total Revenues (FY)")) * 100
                                                                                   AS sga_efficiency_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_cost_structure_features(text) owner to postgres;

