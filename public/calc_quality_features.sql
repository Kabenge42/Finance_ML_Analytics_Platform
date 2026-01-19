create function calc_quality_features()
    returns TABLE
            (
                ticker                      text,
                has_goodwill_impairment     integer,
                has_asset_writedown         integer,
                has_restructuring           integer,
                goodwill_to_assets_pct      numeric,
                intangible_intensity        numeric,
                exceptional_items_to_ebitda numeric,
                altman_z_score              numeric,
                altman_z_trend              numeric,
                current_ratio               numeric,
                quick_ratio                 numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                                          AS ticker,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END                                   AS has_goodwill_impairment,
       CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 1 ELSE 0 END                                          AS has_asset_writedown,
       CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 1 ELSE 0 END                                    AS has_restructuring,
       -- Goodwill to Assets (NULLIF handles zero division)
       "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                                          AS goodwill_to_assets_pct,
       -- Intangible Intensity
       "Gross Intangible Assets (LTM)" / NULLIF("Total Assets (LTM)", 0)                                 AS intangible_intensity,
       -- Exceptional Items to EBITDA
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)                                                                    AS exceptional_items_to_ebitda,
       -- Altman Z-Score
       "Altman Z-Score (LTM)"                                                                            AS altman_z_score,
       -- Altman Z Trend
       "Altman Z-Score (FY)" - "Altman Z-Score (LTM)"                                                    AS altman_z_trend,
       -- Liquidity Ratios
       "Current Ratio (LTM)"                                                                             AS current_ratio,
       -- Quick Ratio
       ("Total Current Assets (LTM)" - "Inventory (LTM)") / NULLIF("Total Current Liabilities (LTM)", 0) AS quick_ratio
FROM postgres.public.equities;
$$;

alter function calc_quality_features() owner to postgres;

