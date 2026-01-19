create materialized view mv_unusual_items as
SELECT "Ticker"                                           AS ticker,
       "ISIN"                                             AS isin,
       "Name"                                             AS name,
       "Sector"                                           AS sector,
       "Industry"                                         AS industry,
       "Country"                                          AS country,
       "Market Cap"                                       AS market_cap,
       "Impairment of Goodwill (LTM)"                     AS impairment_ltm,
       "Impairment of Goodwill (FY)"                      AS impairment_fy,
       "Impairment of Goodwill (-1FY)"                    AS impairment_1fy,
       "Asset Writedown (LTM)"                            AS writedown_ltm,
       "Restructuring Charges (LTM)"                      AS restructuring_ltm,
       "Restructuring Charges (FY)"                       AS restructuring_fy,
       "Restructuring Charges (-1FY)"                     AS restructuring_1fy,
       "Merger & Restructuring Charges (LTM)"             AS merger_charges_ltm,
       "Gain (Loss) On Sale Of Assets (LTM)"              AS asset_sale_gain_ltm,
       "Other Unusual Items/Total (LTM)"                  AS other_unusual_ltm,
       abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") + abs("Restructuring Charges (LTM)") +
       abs("Merger & Restructuring Charges (LTM)") + abs("Gain (Loss) On Sale Of Assets (LTM)") +
       abs("Other Unusual Items/Total (LTM)")             AS total_unusual_items,
       (abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") + abs("Restructuring Charges (LTM)") +
        abs("Other Unusual Items/Total (LTM)")) / NULLIF("Total Revenues (LTM)", 0::numeric) *
       100::numeric                                       AS unusual_to_revenue_ratio,
       (abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") + abs("Restructuring Charges (LTM)") +
        abs("Other Unusual Items/Total (LTM)")) /
       NULLIF(abs("EBITDA (LTM)"), 0::numeric)            AS unusual_to_ebitda_ratio,
       (abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") + abs("Restructuring Charges (LTM)") +
        abs("Other Unusual Items/Total (LTM)")) /
       NULLIF(abs("Net Income - (IS) (LTM)"), 0::numeric) AS unusual_to_net_income_ratio,
       CASE
           WHEN abs("Impairment of Goodwill (LTM)") < 1::numeric AND abs("Asset Writedown (LTM)") < 1::numeric AND
                abs("Restructuring Charges (LTM)") < 1::numeric AND abs("Other Unusual Items/Total (LTM)") < 1::numeric
               THEN 1
           ELSE 0
           END                                            AS clean_earnings_flag,
       CASE
           WHEN (
                    CASE
                        WHEN abs("Impairment of Goodwill (FY)") > 0::numeric THEN 1
                        ELSE 0
                        END +
                    CASE
                        WHEN abs("Impairment of Goodwill (-1FY)") > 0::numeric THEN 1
                        ELSE 0
                        END +
                    CASE
                        WHEN abs("Restructuring Charges (FY)") > 0::numeric THEN 1
                        ELSE 0
                        END +
                    CASE
                        WHEN abs("Restructuring Charges (-1FY)") > 0::numeric THEN 1
                        ELSE 0
                        END) >= 3 THEN 1
           ELSE 0
           END                                            AS recurring_unusual_flag,
       LEAST(100::numeric,
             (abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") + abs("Restructuring Charges (LTM)") +
              abs("Other Unusual Items/Total (LTM)")) / NULLIF(abs("Net Income - (IS) (LTM)"), 0::numeric) *
             100::numeric)                                AS earnings_noise_score,
       "Net Income - (IS) (LTM)" + "Impairment of Goodwill (LTM)" + "Asset Writedown (LTM)" +
       "Restructuring Charges (LTM)"                      AS quality_adjusted_ni,
       (abs("Impairment of Goodwill (LTM)") + abs("Asset Writedown (LTM)") + abs("Restructuring Charges (LTM)")) /
       NULLIF("Shrs Out", 0::numeric)                     AS exceptional_items_impact
FROM equities e;

alter materialized view mv_unusual_items owner to postgres;

