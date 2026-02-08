create table analytics.valuation_ratios_statistics
(
    mean      double precision,
    median    double precision,
    std       double precision,
    skewness  double precision,
    kurtosis  double precision,
    view_name text,
    category  text
);

alter table analytics.valuation_ratios_statistics
    owner to postgres;

