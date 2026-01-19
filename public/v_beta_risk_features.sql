create view v_beta_risk_features
            (ticker, beta_1y, beta_2y, beta_5y, beta_trend_short, beta_trend_long, beta_stability, beta_regime_change,
             systematic_risk_score, defensive_stock_flag, high_beta_flag)
as
SELECT ticker,
       beta_1y,
       beta_2y,
       beta_5y,
       beta_trend_short,
       beta_trend_long,
       beta_stability,
       beta_regime_change,
       systematic_risk_score,
       defensive_stock_flag,
       high_beta_flag
FROM calc_beta_risk_features() calc_beta_risk_features(ticker, beta_1y, beta_2y, beta_5y, beta_trend_short,
                                                       beta_trend_long, beta_stability, beta_regime_change,
                                                       systematic_risk_score, defensive_stock_flag, high_beta_flag);

alter table v_beta_risk_features
    owner to postgres;

