create table analytics.expected_returns_tri_model
(
    ticker                        text,
    name                          text,
    region                        text,
    country                       text,
    exchange                      text,
    sector                        text,
    industry                      text,
    prob_positive_upside          double precision,
    price_target_mc               double precision,
    expected_upside_pct           double precision,
    risk_reward_ratio             double precision,
    var_5_pct                     double precision,
    filtered_upside               double precision,
    kalman_estimate               double precision,
    kalman_variance               double precision,
    expected_return_prob_weighted double precision,
    achievement_probability       double precision,
    price_target_prob_weighted    double precision,
    confidence_level              text,
    analyst_conviction            double precision,
    eps_revision_momentum         double precision,
    analyst_rating_normalized     double precision,
    mc_bullish                    boolean,
    kal_bullish                   boolean,
    pt_bullish                    boolean,
    agreement_score               bigint,
    signal                        text
);

alter table analytics.expected_returns_tri_model
    owner to postgres;

