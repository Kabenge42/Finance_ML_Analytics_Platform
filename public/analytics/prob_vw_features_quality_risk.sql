create table analytics.prob_vw_features_quality_risk
(
    isin              text,
    ticker            text,
    name              text,
    industry          text,
    sector            text,
    feature           text,
    value             text,
    percentile        double precision,
    z_score           double precision,
    prob_above_median double precision
);

alter table analytics.prob_vw_features_quality_risk
    owner to postgres;

