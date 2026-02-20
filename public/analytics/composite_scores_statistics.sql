create table analytics.composite_scores_statistics
(
    ticker          text,
    name            text,
    sector          text,
    industry        text,
    composite_score double precision
);

alter table analytics.composite_scores_statistics
    owner to postgres;

