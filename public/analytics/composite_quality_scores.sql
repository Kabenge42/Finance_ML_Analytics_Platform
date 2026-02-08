create table analytics.composite_quality_scores
(
    ticker                     text,
    name                       text,
    sector                     text,
    industry                   text,
    region                     text,
    country                    text,
    exchange                   text,
    market_cap                 double precision,
    enterprise_value           double precision,
    last_price                 double precision,
    price_target               double precision,
    piotroski_f_score          bigint,
    earnings_quality_composite double precision,
    cash_flow_quality_score    double precision,
    distress_risk_score        double precision,
    accounting_quality_score   double precision,
    dilution_score             double precision,
    beta_stability_score       double precision,
    long_term_trend_score      double precision,
    eps_trajectory_score       double precision,
    composite_quality_score    double precision,
    quality_tier               text
);

alter table analytics.composite_quality_scores
    owner to postgres;

