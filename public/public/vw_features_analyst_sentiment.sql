create view vw_features_analyst_sentiment
            (isin, analyst_bullish_pct, analyst_bearish_pct, analyst_neutral_pct, analyst_conviction,
             upside_potential,
             price_target_spread_pct, price_target_revision_1m, price_target_revision_3m, eps_revision_momentum,
             analyst_rating_normalized, analyst_coverage_quality)
as
SELECT isin,

       analyst_bullish_pct,
       analyst_bearish_pct,
       analyst_neutral_pct,
       analyst_conviction,
       upside_potential,
       price_target_spread_pct,
       price_target_revision_1m,
       price_target_revision_3m,
       eps_revision_momentum,
       analyst_rating_normalized,
       analyst_coverage_quality
FROM calc_sentiment_features() calc_sentiment_features(isin, analyst_bullish_pct, analyst_bearish_pct,
                                                       analyst_neutral_pct, analyst_conviction, upside_potential,
                                                       price_target_spread_pct, price_target_revision_1m,
                                                       price_target_revision_3m, eps_revision_momentum,
                                                       analyst_rating_normalized, analyst_coverage_quality);

alter table vw_features_analyst_sentiment
    owner to postgres;

