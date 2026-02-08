create table analytics.prob_vw_features_earnings
(
    isin              text,
    ticker            text,
    name              text,
    industry          text,
    sector            text,
    feature           text,
    value             double precision,
    percentile        double precision,
    z_score           double precision,
    prob_above_median double precision
);

alter table analytics.prob_vw_features_earnings
    owner to postgres;

