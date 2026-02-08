create table analytics.leverage_liquidity_statistics
(
    mean      double precision,
    median    double precision,
    std       double precision,
    skewness  double precision,
    kurtosis  double precision,
    view_name text,
    category  text
);

alter table analytics.leverage_liquidity_statistics
    owner to postgres;

