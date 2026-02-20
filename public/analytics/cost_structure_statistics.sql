create table analytics.cost_structure_statistics
(
    mean         double precision,
    median       double precision,
    std          double precision,
    skewness     double precision,
    kurtosis     double precision,
    view_name    text,
    category     text,
    feature_cols text
);

alter table analytics.cost_structure_statistics
    owner to postgres;

