create view vw_features_financial_distress
            (isin, distress_risk_score, liquidity_stress_score, working_capital_trend, cash_runway_months,
             combined_distress_score, wc_deteriorating_flag, retained_earnings_growth, accumulated_deficit_flag,
             adequate_cash_buffer)
as
SELECT isin,
       distress_risk_score,
       liquidity_stress_score,
       working_capital_trend,
       cash_runway_months,
       combined_distress_score,
       wc_deteriorating_flag,
       retained_earnings_growth,
       accumulated_deficit_flag,
       adequate_cash_buffer
FROM calc_financial_distress_features() calc_financial_distress_features(isin, distress_risk_score,
                                                                         liquidity_stress_score, working_capital_trend,
                                                                         cash_runway_months, combined_distress_score,
                                                                         wc_deteriorating_flag,
                                                                         retained_earnings_growth,
                                                                         accumulated_deficit_flag,
                                                                         adequate_cash_buffer);

alter table vw_features_financial_distress
    owner to postgres;

