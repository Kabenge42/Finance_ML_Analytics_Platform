create function calc_unusual_items_features()
    returns TABLE
            (
                ticker                      text,
                other_unusual_items_ltm     numeric,
                total_unusual_items         numeric,
                unusual_to_revenue_ratio    numeric,
                unusual_to_ebitda_ratio     numeric,
                unusual_to_net_income_ratio numeric,
                clean_earnings_flag         integer,
                recurring_unusual_flag      integer,
                earnings_noise_score        numeric,
                quality_adjusted_ni         numeric,
                exceptional_items_impact    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                     AS ticker,
       "Other Unusual Items/Total (LTM)"                            AS other_unusual_items_ltm,
       -- Total Unusual Items (sum of all exceptional items)
       ABS("Impairment of Goodwill (LTM)") +
       ABS("Asset Writedown (LTM)") +
       ABS("Restructuring Charges (LTM)") +
       ABS("Merger & Restructuring Charges (LTM)") +
       ABS("Gain (Loss) On Sale Of Assets (LTM)") +
       ABS("Other Unusual Items/Total (LTM)")                       AS total_unusual_items,
       -- Unusual Items to Revenue Ratio
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF("Total Revenues (LTM)", 0) * 100                      AS unusual_to_revenue_ratio,
       -- Unusual Items to EBITDA Ratio
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)                               AS unusual_to_ebitda_ratio,
       -- Unusual Items to Net Income Ratio
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0)                    AS unusual_to_net_income_ratio,
       -- Clean Earnings Flag (no unusual items)
       CASE
           WHEN ABS("Impairment of Goodwill (LTM)") < 1 AND
                ABS("Asset Writedown (LTM)") < 1 AND
                ABS("Restructuring Charges (LTM)") < 1 AND
                ABS("Other Unusual Items/Total (LTM)") < 1
               THEN 1
           ELSE 0
           END                                                      AS clean_earnings_flag,
       -- Recurring Unusual Items Flag (unusual in multiple years)
       CASE
           WHEN (CASE WHEN ABS("Impairment of Goodwill (FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Impairment of Goodwill (-1FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Restructuring Charges (FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Restructuring Charges (-1FY)") > 0 THEN 1 ELSE 0 END) >= 3
               THEN 1
           ELSE 0
           END                                                      AS recurring_unusual_flag,
       -- Earnings Noise Score (0-100, lower = cleaner)
       LEAST(100,
             (ABS("Impairment of Goodwill (LTM)") +
              ABS("Asset Writedown (LTM)") +
              ABS("Restructuring Charges (LTM)") +
              ABS("Other Unusual Items/Total (LTM)")) /
             NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100)       AS earnings_noise_score,
       -- Quality Adjusted Net Income
       "Net Income - (IS) (LTM)" +
       "Impairment of Goodwill (LTM)" +
       "Asset Writedown (LTM)" +
       "Restructuring Charges (LTM)"                                AS quality_adjusted_ni,
       -- Exceptional Items Impact on EPS (per share)
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF("Shrs Out", 0) AS exceptional_items_impact
FROM postgres.public.equities;
$$;

alter function calc_unusual_items_features() owner to postgres;

