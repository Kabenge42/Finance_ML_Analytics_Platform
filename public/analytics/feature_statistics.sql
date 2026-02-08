create table analytics.feature_statistics
(
    count        bigint,
    mean         double precision,
    median       double precision,
    std          double precision,
    min          double precision,
    max          double precision,
    q25          double precision,
    q75          double precision,
    positive_pct double precision,
    missing_pct  double precision,
    category     text,
    feature      text
);

alter table analytics.feature_statistics
    owner to postgres;

