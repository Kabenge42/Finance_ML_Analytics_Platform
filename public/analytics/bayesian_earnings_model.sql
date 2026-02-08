create table analytics.bayesian_earnings_model
(
    ticker              text,
    name                text,
    sector              text,
    industry            text,
    region              text,
    country             text,
    exchange            text,
    eps_positive_streak bigint,
    posterior_beat_prob double precision,
    model_confidence    double precision,
    map_estimate        double precision
);

alter table analytics.bayesian_earnings_model
    owner to postgres;

