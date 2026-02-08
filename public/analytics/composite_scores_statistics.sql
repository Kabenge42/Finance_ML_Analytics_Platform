create table analytics.composite_scores_statistics
(
    mean      double precision,
    median    double precision,
    std       double precision,
    skewness  double precision,
    kurtosis  double precision,
    view_name text,
    category  text
);

alter table analytics.composite_scores_statistics
    owner to postgres;

