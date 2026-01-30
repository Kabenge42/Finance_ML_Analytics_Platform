create view vw_features_unusual_items
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, other_unusual_items_ltm,
             impairment_goodwill_ltm, asset_writedown_ltm, restructuring_charges_ltm, total_unusual_items,
             unusual_items_to_revenue, unusual_items_to_ebitda, has_unusual_items_flag, earnings_quality_impact)
as
SELECT id.isin,
       id.ticker,
       id.name,
       id.industry,
       id.sector,
       id.trading_country,
       id.region,
       id.country,
       id.exchange,
       uif.other_unusual_items_ltm,
       uif.impairment_goodwill_ltm,
       uif.asset_writedown_ltm,
       uif.restructuring_charges_ltm,
       uif.total_unusual_items,
       uif.unusual_items_to_revenue,
       uif.unusual_items_to_ebitda,
       uif.has_unusual_items_flag,
       uif.earnings_quality_impact
FROM vw_identifier_columns                       id
         LEFT JOIN calc_unusual_items_features() uif(isin, other_unusual_items_ltm, impairment_goodwill_ltm,
                                                     asset_writedown_ltm, restructuring_charges_ltm,
                                                     total_unusual_items, unusual_items_to_revenue,
                                                     unusual_items_to_ebitda, has_unusual_items_flag,
                                                     earnings_quality_impact) USING (isin);

comment on view vw_features_unusual_items is 'Non-recurring and unusual items analysis for earnings quality assessment.
    Source function: calc_unusual_items_features';

alter table vw_features_unusual_items
    owner to postgres;

