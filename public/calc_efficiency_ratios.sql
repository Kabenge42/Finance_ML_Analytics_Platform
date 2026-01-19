create function calc_efficiency_ratios()
    returns TABLE
            (
                ticker                text,
                asset_turnover        numeric,
                inventory_turnover    numeric,
                receivables_days      numeric,
                working_capital_turns numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                      AS ticker,
       -- Asset Turnover (Revenue / Total Assets)
       "Total Revenues (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS asset_turnover,

       -- Inventory Turnover (COGS / Average Inventory)
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)                       AS inventory_turnover,

       -- Receivables Days (Accounts Receivable / Daily Revenue * 365)
       ("Accounts Receivable/Total (FY)" / NULLIF("Total Revenues (FY)" / 365.0, 0)) AS receivables_days,

       -- Working Capital Turnover (Revenue / Working Capital)
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turns

FROM postgres.public.equities;
$$;

alter function calc_efficiency_ratios() owner to postgres;

