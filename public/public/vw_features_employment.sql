create view vw_features_employment
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, revenue_per_employee,
             profit_per_employee, ebitda_per_employee, assets_per_employee, fte_growth_1y_pct, fte_growth_3y_pct,
             workforce_stability, fte_growth_2y_pct, fte_acceleration, workforce_volatility, hiring_intensity,
             productivity_trend, headcount_vs_revenue, workforce_efficiency_gain, layoff_risk_flag, rapid_hiring_flag,
             sustainable_growth_flag)
as
SELECT id.isin,
       id.ticker,
       id.name,
       id.industry,
       id.sector,
       id.trading_country,
       id.region,
       id.country,
       id.exchange,
       ef.revenue_per_employee,
       ef.profit_per_employee,
       ef.ebitda_per_employee,
       ef.assets_per_employee,
       ef.fte_growth_1y_pct,
       ef.fte_growth_3y_pct,
       ef.workforce_stability,
       ed.fte_growth_2y_pct,
       ed.fte_acceleration,
       ed.workforce_volatility,
       ed.hiring_intensity,
       ed.productivity_trend,
       ed.headcount_vs_revenue,
       ed.workforce_efficiency_gain,
       ed.layoff_risk_flag,
       ed.rapid_hiring_flag,
       ed.sustainable_growth_flag
FROM vw_identifier_columns                    id
         LEFT JOIN calc_employment_features() ef(isin, revenue_per_employee, profit_per_employee, ebitda_per_employee,
                                                 assets_per_employee, fte_growth_1y_pct, fte_growth_3y_pct,
                                                 workforce_stability) USING (isin)
         LEFT JOIN calc_employment_dynamics() ed(isin, fte_growth_2y_pct, fte_acceleration, workforce_volatility,
                                                 hiring_intensity, productivity_trend, headcount_vs_revenue,
                                                 workforce_efficiency_gain, layoff_risk_flag, rapid_hiring_flag,
                                                 sustainable_growth_flag) USING (isin);

comment on view vw_features_employment is 'Employment metrics including productivity, workforce trends, and efficiency.
    Source functions: calc_employment_features, calc_employment_dynamics';

alter table vw_features_employment
    owner to postgres;

