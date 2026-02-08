create table analytics.composite_scores_distributions
(
    piotroski_f_score          text,
    eps_trajectory_score       text,
    dilution_score             text,
    quality_momentum_score     text,
    net_income_is_fq           text,
    net_income_is_ltm          text,
    net_income_is_fy           text,
    net_income_adj_ltm         text,
    normalized_ni_ltm          text,
    net_income_is_1fqfq        text,
    net_income_is_2fqfq        text,
    net_income_is_3fqfq        text,
    net_income_is_4fqfq        text,
    net_income_is_1fy          text,
    net_income_is_2fy          text,
    net_income_is_3fy          text,
    net_income_is_4fy          text,
    net_income_is_5yavgfq      text,
    net_income_is_5yavgltm     text,
    normalized_ni_5yavgfq      text,
    normalized_ni_5yavgltm     text,
    net_income_growth_yoy      text,
    net_income_margin_ltm      text,
    ni_adjustment_ratio        text,
    net_income_positive_years  text,
    earnings_quality_composite text,
    net_income_qoq_growth      text,
    net_income_yoy_quarterly   text,
    net_income_vs_5y_avg       text,
    normalized_ni_vs_5y_avg    text
);

alter table analytics.composite_scores_distributions
    owner to postgres;

