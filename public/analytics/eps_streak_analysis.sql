create table analytics.eps_streak_analysis
(
    ticker                     text,
    name                       text,
    sector                     text,
    industry                   text,
    country                    text,
    exchange                   text,
    current_streak             bigint,
    streak_type                text,
    continuation_probability   double precision,
    mean_reversion_probability double precision,
    expected_next_outcome      text,
    prediction_confidence      double precision
);

alter table analytics.eps_streak_analysis
    owner to postgres;

