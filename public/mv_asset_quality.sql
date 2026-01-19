create materialized view mv_asset_quality as
SELECT "Ticker"                                                                        AS ticker,
       "ISIN"                                                                          AS isin,
       "Name"                                                                          AS name,
       "Sector"                                                                        AS sector,
       "Industry"                                                                      AS industry,
       "Country"                                                                       AS country,
       "Market Cap"                                                                    AS market_cap,
       "Last Price"                                                                    AS last_price,
       "TBV (FY)"                                                                      AS tbv_fy,
       "TBV (LTM)"                                                                     AS tbv_ltm,
       "Total Equity (FY)"                                                             AS total_equity_fy,
       "Total Equity (LTM)"                                                            AS total_equity_ltm,
       "Goodwill (LTM)"                                                                AS goodwill_ltm,
       "Gross Intangible Assets (LTM)"                                                 AS intangible_assets_ltm,
       "Total Debt (LTM)"                                                              AS total_debt_ltm,
       "Shrs Out"                                                                      AS shares_outstanding,
       "P/TBV (LTM)"                                                                   AS ptbv_ltm,
       "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0::numeric), 0::numeric) AS price_to_tbv,
       "TBV (LTM)" / NULLIF("Shrs Out", 0::numeric)                                    AS tbv_per_share,
       ("TBV (LTM)" - "TBV (FY)") / NULLIF(abs("TBV (FY)"), 0::numeric) * 100::numeric AS tbv_growth_yoy,
       "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0::numeric)                          AS tangible_equity_ratio,
       ("Goodwill (LTM)" + "Gross Intangible Assets (LTM)") /
       NULLIF("TBV (LTM)", 0::numeric)                                                 AS intangible_to_tbv_ratio,
       "TBV (LTM)" / NULLIF("Market Cap", 0::numeric)                                  AS tbv_vs_market_cap,
       "TBV (LTM)" - "Total Debt (LTM)"                                                AS net_tangible_assets,
       ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0::numeric) * 100::numeric   AS tbv_margin_of_safety
FROM equities e;

alter materialized view mv_asset_quality owner to postgres;

