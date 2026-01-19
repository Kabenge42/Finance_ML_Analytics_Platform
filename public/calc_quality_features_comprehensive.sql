create function calc_quality_features_comprehensive()
    returns TABLE
            (
                ticker                        text,
                goodwill_impairment_fq        numeric,
                goodwill_impairment_ltm       numeric,
                goodwill_impairment_fy        numeric,
                goodwill_impairment_1fy       numeric,
                goodwill_impairment_2fy       numeric,
                goodwill_impairment_3fy       numeric,
                goodwill_impairment_4fy       numeric,
                goodwill_impairment_1fqfq     numeric,
                goodwill_impairment_2fqfq     numeric,
                goodwill_impairment_3fqfq     numeric,
                goodwill_impairment_4fqfq     numeric,
                goodwill_impairment_5yavg     numeric,
                has_goodwill_impairment_ltm   integer,
                has_goodwill_impairment_fy    integer,
                has_goodwill_impairment_1fy   integer,
                has_goodwill_impairment_2fy   integer,
                has_goodwill_impairment_3fy   integer,
                has_goodwill_impairment_4fy   integer,
                goodwill_impairment_frequency integer,
                asset_writedown_fq            numeric,
                asset_writedown_ltm           numeric,
                asset_writedown_fy            numeric,
                asset_writedown_1fy           numeric,
                asset_writedown_2fy           numeric,
                asset_writedown_3fy           numeric,
                asset_writedown_4fy           numeric,
                asset_writedown_5fy           numeric,
                asset_writedown_1fqfq         numeric,
                asset_writedown_2fqfq         numeric,
                asset_writedown_3fqfq         numeric,
                asset_writedown_4fqfq         numeric,
                asset_writedown_5yavg         numeric,
                asset_writedown_frequency     integer,
                restructuring_fq              numeric,
                restructuring_ltm             numeric,
                restructuring_fy              numeric,
                restructuring_1fy             numeric,
                restructuring_2fy             numeric,
                restructuring_3fy             numeric,
                restructuring_4fy             numeric,
                restructuring_1fqfq           numeric,
                restructuring_2fqfq           numeric,
                restructuring_3fqfq           numeric,
                restructuring_4fqfq           numeric,
                restructuring_5yavg           numeric,
                restructuring_frequency       integer,
                merger_restructuring_fq       numeric,
                merger_restructuring_ltm      numeric,
                merger_restructuring_fy       numeric,
                merger_restructuring_5yavg    numeric,
                asset_sale_gain_fq            numeric,
                asset_sale_gain_ltm           numeric,
                asset_sale_gain_fy            numeric,
                asset_sale_gain_1fy           numeric,
                asset_sale_gain_2fy           numeric,
                asset_sale_gain_3fy           numeric,
                asset_sale_gain_4fy           numeric,
                asset_sale_gain_1fqfq         numeric,
                asset_sale_gain_2fqfq         numeric,
                asset_sale_gain_3fqfq         numeric,
                asset_sale_gain_4fqfq         numeric,
                goodwill_impairment_trend_yoy numeric,
                goodwill_impairment_trend_qoq numeric,
                asset_writedown_trend_yoy     numeric,
                restructuring_trend_yoy       numeric,
                restructuring_trend_qoq       numeric,
                exceptional_items_total_ltm   numeric,
                exceptional_items_to_revenue  numeric,
                exceptional_items_to_ebitda   numeric,
                quality_issues_count_5y       integer,
                accounting_quality_score      numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                     AS ticker,
       -- Goodwill Impairment Values (ALL periods)
       "Impairment of Goodwill (FQ)"                                                AS goodwill_impairment_fq,
       "Impairment of Goodwill (LTM)"                                               AS goodwill_impairment_ltm,
       "Impairment of Goodwill (FY)"                                                AS goodwill_impairment_fy,
       "Impairment of Goodwill (-1FY)"                                              AS goodwill_impairment_1fy,
       "Impairment of Goodwill (-2FY)"                                              AS goodwill_impairment_2fy,
       "Impairment of Goodwill (-3FY)"                                              AS goodwill_impairment_3fy,
       "Impairment of Goodwill (-4FY)"                                              AS goodwill_impairment_4fy,
       "Impairment of Goodwill (-1FQFQ)"                                            AS goodwill_impairment_1fqfq,
       "Impairment of Goodwill (-2FQFQ)"                                            AS goodwill_impairment_2fqfq,
       "Impairment of Goodwill (-3FQFQ)"                                            AS goodwill_impairment_3fqfq,
       "Impairment of Goodwill (-4FQFQ)"                                            AS goodwill_impairment_4fqfq,
       "Impairment of Goodwill (5YAVGFQ)"                                           AS goodwill_impairment_5yavg,
       -- Goodwill Impairment Flags
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END              AS has_goodwill_impairment_ltm,
       CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END               AS has_goodwill_impairment_fy,
       CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_1fy,
       CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_2fy,
       CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_3fy,
       CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_4fy,
       (CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER  AS goodwill_impairment_frequency,
       -- Asset Writedown Values (ALL periods)
       "Asset Writedown (FQ)"                                                       AS asset_writedown_fq,
       "Asset Writedown (LTM)"                                                      AS asset_writedown_ltm,
       "Asset Writedown (FY)"                                                       AS asset_writedown_fy,
       "Asset Writedown (-1FY)"                                                     AS asset_writedown_1fy,
       "Asset Writedown (-2FY)"                                                     AS asset_writedown_2fy,
       "Asset Writedown (-3FY)"                                                     AS asset_writedown_3fy,
       "Asset Writedown (-4FY)"                                                     AS asset_writedown_4fy,
       "Asset Writedown (-5FY)"                                                     AS asset_writedown_5fy,
       "Asset Writedown (-1FQFQ)"                                                   AS asset_writedown_1fqfq,
       "Asset Writedown (-2FQFQ)"                                                   AS asset_writedown_2fqfq,
       "Asset Writedown (-3FQFQ)"                                                   AS asset_writedown_3fqfq,
       "Asset Writedown (-4FQFQ)"                                                   AS asset_writedown_4fqfq,
       "Asset Writedown (5YAVGFQ)"                                                  AS asset_writedown_5yavg,
       (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER         AS asset_writedown_frequency,
       -- Restructuring Charges Values (ALL periods)
       "Restructuring Charges (FQ)"                                                 AS restructuring_fq,
       "Restructuring Charges (LTM)"                                                AS restructuring_ltm,
       "Restructuring Charges (FY)"                                                 AS restructuring_fy,
       "Restructuring Charges (-1FY)"                                               AS restructuring_1fy,
       "Restructuring Charges (-2FY)"                                               AS restructuring_2fy,
       "Restructuring Charges (-3FY)"                                               AS restructuring_3fy,
       "Restructuring Charges (-4FY)"                                               AS restructuring_4fy,
       "Restructuring Charges (-1FQFQ)"                                             AS restructuring_1fqfq,
       "Restructuring Charges (-2FQFQ)"                                             AS restructuring_2fqfq,
       "Restructuring Charges (-3FQFQ)"                                             AS restructuring_3fqfq,
       "Restructuring Charges (-4FQFQ)"                                             AS restructuring_4fqfq,
       "Restructuring Charges (5YAVGFQ)"                                            AS restructuring_5yavg,
       (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER   AS restructuring_frequency,
       -- Merger & Restructuring
       "Merger & Restructuring Charges (FQ)"                                        AS merger_restructuring_fq,
       "Merger & Restructuring Charges (LTM)"                                       AS merger_restructuring_ltm,
       "Merger & Restructuring Charges (FY)"                                        AS merger_restructuring_fy,
       "Merger & Restructuring Charges (5YAVGFQ)"                                   AS merger_restructuring_5yavg,
       -- Gain/Loss on Asset Sales
       "Gain (Loss) On Sale Of Assets (FQ)"                                         AS asset_sale_gain_fq,
       "Gain (Loss) On Sale Of Assets (LTM)"                                        AS asset_sale_gain_ltm,
       "Gain (Loss) On Sale Of Assets (FY)"                                         AS asset_sale_gain_fy,
       "Gain (Loss) On Sale Of Assets (-1FY)"                                       AS asset_sale_gain_1fy,
       "Gain (Loss) On Sale Of Assets (-2FY)"                                       AS asset_sale_gain_2fy,
       "Gain (Loss) On Sale Of Assets (-3FY)"                                       AS asset_sale_gain_3fy,
       "Gain (Loss) On Sale Of Assets (-4FY)"                                       AS asset_sale_gain_4fy,
       "Gain (Loss) On Sale Of Assets (-1FQFQ)"                                     AS asset_sale_gain_1fqfq,
       "Gain (Loss) On Sale Of Assets (-2FQFQ)"                                     AS asset_sale_gain_2fqfq,
       "Gain (Loss) On Sale Of Assets (-3FQFQ)"                                     AS asset_sale_gain_3fqfq,
       "Gain (Loss) On Sale Of Assets (-4FQFQ)"                                     AS asset_sale_gain_4fqfq,
       -- Trends
       ("Impairment of Goodwill (FY)" - "Impairment of Goodwill (-1FY)") /
       NULLIF(ABS("Impairment of Goodwill (-1FY)"), 0)                              AS goodwill_impairment_trend_yoy,
       ("Impairment of Goodwill (FQ)" - "Impairment of Goodwill (-1FQFQ)") /
       NULLIF(ABS("Impairment of Goodwill (-1FQFQ)"), 0)                            AS goodwill_impairment_trend_qoq,
       ("Asset Writedown (FY)" - "Asset Writedown (-1FY)") /
       NULLIF(ABS("Asset Writedown (-1FY)"), 0)                                     AS asset_writedown_trend_yoy,
       ("Restructuring Charges (FY)" - "Restructuring Charges (-1FY)") /
       NULLIF(ABS("Restructuring Charges (-1FY)"), 0)                               AS restructuring_trend_yoy,
       ("Restructuring Charges (FQ)" - "Restructuring Charges (-1FQFQ)") /
       NULLIF(ABS("Restructuring Charges (-1FQFQ)"), 0)                             AS restructuring_trend_qoq,
       -- Aggregate Metrics
       ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
       ABS("Restructuring Charges (LTM)")                                           AS exceptional_items_total_ltm,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF("Total Revenues (LTM)", 0)     AS exceptional_items_to_revenue,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF(ABS("EBITDA (LTM)"), 0)        AS exceptional_items_to_ebitda,
       -- Combined 5Y Quality Issues Count
       ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END))::INTEGER AS quality_issues_count_5y,
       -- Comprehensive Quality Score (100 = best)
       GREATEST(0, LEAST(100,
                         100 -
                         ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) * 8) -
                         ((CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4) -
                         ((CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4)
                   ))                                                               AS accounting_quality_score
FROM postgres.public.equities;
$$;

alter function calc_quality_features_comprehensive() owner to postgres;

