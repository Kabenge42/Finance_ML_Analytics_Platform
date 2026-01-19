create function calc_tangible_book_features()
    returns TABLE
            (
                ticker                  text,
                tbv_fy                  numeric,
                tbv_ltm                 numeric,
                price_to_tbv            numeric,
                tbv_per_share           numeric,
                tbv_growth_yoy          numeric,
                tangible_equity_ratio   numeric,
                intangible_to_tbv_ratio numeric,
                tbv_vs_market_cap       numeric,
                net_tangible_assets     numeric,
                tbv_margin_of_safety    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                      AS ticker,
       "TBV (FY)"                                                    AS tbv_fy,
       "TBV (LTM)"                                                   AS tbv_ltm,
       -- Price to Tangible Book Value
       "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0), 0) AS price_to_tbv,
       -- TBV Per Share
       "TBV (LTM)" / NULLIF("Shrs Out", 0)                           AS tbv_per_share,
       -- TBV Growth YoY
       ("TBV (LTM)" - "TBV (FY)") / NULLIF(ABS("TBV (FY)"), 0) * 100 AS tbv_growth_yoy,
       -- Tangible Equity as % of Total Equity
       "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0)                 AS tangible_equity_ratio,
       -- Intangibles to TBV Ratio (lower = better asset quality)
       ("Goodwill (LTM)" + "Gross Intangible Assets (LTM)") /
       NULLIF("TBV (LTM)", 0)                                        AS intangible_to_tbv_ratio,
       -- TBV vs Market Cap (margin of safety)
       "TBV (LTM)" / NULLIF("Market Cap", 0)                         AS tbv_vs_market_cap,
       -- Net Tangible Assets (TBV - Total Debt)
       "TBV (LTM)" - "Total Debt (LTM)"                              AS net_tangible_assets,
       -- TBV Margin of Safety (discount to TBV)
       ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0) * 100   AS tbv_margin_of_safety
FROM postgres.public.equities;
$$;

alter function calc_tangible_book_features() owner to postgres;

