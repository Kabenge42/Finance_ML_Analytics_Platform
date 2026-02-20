create table analytics.expected_returns_tri_model
(
    ticker                        text,
    name                          text,
    sector                        text,
    industry                      text,
    expected_upside_pct           double precision,
    prob_positive_upside          double precision,
    filtered_upside               double precision,
    expected_return_prob_weighted double precision,
    achievement_probability       double precision,
    confidence_level              text,
    mc_bullish                    boolean,
    kal_bullish                   boolean,
    pt_bullish                    boolean,
    agreement_score               bigint,
    signal                        text
);

alter table analytics.expected_returns_tri_model
    owner to postgres;

