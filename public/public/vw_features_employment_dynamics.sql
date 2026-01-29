create view vw_features_employment_dynamics
            (isin, fte_growth_2y_pct, fte_acceleration, workforce_volatility, hiring_intensity, productivity_trend,
             headcount_vs_revenue, workforce_efficiency_gain, layoff_risk_flag, rapid_hiring_flag,
             sustainable_growth_flag)
as
SELECT isin,
       fte_growth_2y_pct,
       fte_acceleration,
       workforce_volatility,
       hiring_intensity,
       productivity_trend,
       headcount_vs_revenue,
       workforce_efficiency_gain,
       layoff_risk_flag,
       rapid_hiring_flag,
       sustainable_growth_flag
FROM calc_employment_dynamics() calc_employment_dynamics(isin, fte_growth_2y_pct, fte_acceleration,
                                                         workforce_volatility, hiring_intensity, productivity_trend,
                                                         headcount_vs_revenue, workforce_efficiency_gain,
                                                         layoff_risk_flag, rapid_hiring_flag, sustainable_growth_flag);

alter table vw_features_employment_dynamics
    owner to postgres;

