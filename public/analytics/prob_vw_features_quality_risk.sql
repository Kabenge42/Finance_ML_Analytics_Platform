create table analytics.prob_vw_features_quality_risk
(
    feature                  text,
    value_mean               double precision,
    value_median             double precision,
    value_std                double precision,
    value_min                text,
    value_max                text,
    value_count              bigint,
    percentile_mean          double precision,
    percentile_median        double precision,
    percentile_std           double precision,
    percentile_min           double precision,
    percentile_max           double precision,
    percentile_count         bigint,
    z_score_mean             double precision,
    z_score_median           double precision,
    z_score_std              double precision,
    z_score_min              double precision,
    z_score_max              double precision,
    z_score_count            bigint,
    prob_above_median_mean   double precision,
    prob_above_median_median double precision,
    prob_above_median_std    double precision,
    prob_above_median_min    double precision,
    prob_above_median_max    double precision,
    prob_above_median_count  bigint
);

alter table analytics.prob_vw_features_quality_risk
    owner to postgres;

