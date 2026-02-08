create table analytics.employment_distributions
(
    revenue_per_employee      text,
    profit_per_employee       text,
    ebitda_per_employee       text,
    assets_per_employee       text,
    fte_growth_1y_pct         text,
    fte_growth_3y_pct         text,
    workforce_stability       text,
    fte_growth_2y_pct         text,
    fte_acceleration          text,
    workforce_volatility      text,
    hiring_intensity          text,
    productivity_trend        text,
    headcount_vs_revenue      text,
    workforce_efficiency_gain text,
    layoff_risk_flag          text,
    rapid_hiring_flag         text,
    sustainable_growth_flag   text
);

alter table analytics.employment_distributions
    owner to postgres;

