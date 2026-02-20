create table analytics.kalman_filtered_price_targets
(
    ticker          text,
    name            text,
    country         text,
    exchange        text,
    sector          text,
    industry        text,
    kalman_estimate double precision,
    kalman_variance double precision,
    kalman_gain     double precision,
    signal_strength double precision,
    original_price  double precision,
    original_target double precision,
    filtered_upside double precision
);

alter table analytics.kalman_filtered_price_targets
    owner to postgres;

