create view vw_features_composite(isin, feature_count, reference_date) as
SELECT isin,
       feature_count,
       reference_date
FROM calc_all_enhanced_features() calc_all_enhanced_features(isin, feature_count, reference_date);

alter table vw_features_composite
    owner to postgres;

