create table analytics.earnings_probability_analysis
(
    ticker               text,
    name                 text,
    sector               text,
    historical_beats     bigint,
    total_reports        bigint,
    historical_beat_rate double precision,
    posterior_beat_prob  double precision,
    posterior_std        double precision,
    ci_90_lower          double precision,
    ci_90_upper          double precision,
    ci_95_lower          double precision,
    ci_95_upper          double precision,
    confidence_score     double precision,
    beat_classification  text
);

alter table analytics.earnings_probability_analysis
    owner to postgres;

