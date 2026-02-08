create table analytics.analyst_sentiment_distributions
(
    analyst_bullish_pct        text,
    analyst_bearish_pct        text,
    analyst_neutral_pct        text,
    analyst_conviction         text,
    upside_potential           text,
    price_target_spread_pct    text,
    price_target_revision_1m   text,
    price_target_revision_3m   text,
    eps_revision_momentum      text,
    analyst_rating_normalized  text,
    analyst_coverage_quality   text,
    pt_momentum_1w             text,
    pt_momentum_1m             text,
    pt_momentum_3m             text,
    pt_momentum_6m             text,
    pt_momentum_1y             text,
    pt_median_momentum_1m      text,
    pt_median_momentum_3m      text,
    pt_acceleration_short      text,
    pt_acceleration_long       text,
    pt_consensus_convergence   text,
    analyst_coverage_change_1m text,
    analyst_coverage_change_3m text,
    analyst_coverage_change_1y text,
    pt_vs_price_momentum       text,
    analyst_coverage_trend     text
);

alter table analytics.analyst_sentiment_distributions
    owner to postgres;

