create function calc_accounting_quality_features()
    returns TABLE
            (
                ticker                      text,
                goodwill_change_rate        numeric,
                restructuring_intensity     numeric,
                exceptional_items_frequency integer,
                merger_impact_ratio         numeric,
                non_operating_income_share  numeric,
                asset_sale_boost            integer,
                accounting_quality_score    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                              AS ticker,
       -- Goodwill Change Rate (YoY)
       ("Goodwill (LTM)" - "Goodwill (-1FY)") / NULLIF("Goodwill (-1FY)", 0) AS goodwill_change_rate,

       -- Restructuring Intensity (to Total Assets)
       "Restructuring Charges (LTM)" / NULLIF("Total Assets (LTM)", 0)       AS restructuring_intensity,

       -- Exceptional Items Frequency (count of non-zero exceptional items)
       (CASE WHEN ABS("Impairment of Goodwill (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Asset Writedown (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Restructuring Charges (FQ)") > 0 THEN 1 ELSE 0 END)   AS exceptional_items_frequency,

       -- Merger Impact Ratio (Merger Charges / Market Cap)
       "Merger & Restructuring Charges (LTM)" / NULLIF("Market Cap", 0)      AS merger_impact_ratio,

       -- Non-Operating Income Share (Interest Income / Net Income)
       "Interest Income On Investments (LTM)" / NULLIF(ABS("Net Income - (IS) (LTM)"), 0)
                                                                             AS non_operating_income_share,

       -- Asset Sale Boost Flag (gain on sale of assets > 0)
       CASE WHEN "Gain (Loss) On Sale Of Assets (LTM)" > 0 THEN 1 ELSE 0 END AS asset_sale_boost,

       -- Composite Accounting Quality Score (100 = highest quality)
       GREATEST(0, LEAST(100,
                         100 -
                         (CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 25 ELSE 0 END) -
                         (CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 10 ELSE 0 END) -
                         (CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 15 ELSE 0 END) -
                         (CASE WHEN "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.30 THEN 15 ELSE 0 END) -
                         (CASE
                              WHEN (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
                                    ABS("Restructuring Charges (LTM)")) /
                                   NULLIF(ABS("Net Income - (IS) (LTM)"), 0) > 0.10 THEN 15
                              ELSE 0 END)
                   ))                                                        AS accounting_quality_score

FROM postgres.public.equities;
$$;

alter function calc_accounting_quality_features() owner to postgres;

