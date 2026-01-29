create view vw_features_employee_productivity
            (isin, revenue_per_employee, profit_per_employee, ebitda_per_employee, assets_per_employee,
             fte_growth_1y_pct, fte_growth_3y_pct, workforce_stability)
as
SELECT isin,
       revenue_per_employee,
       profit_per_employee,
       ebitda_per_employee,
       assets_per_employee,
       fte_growth_1y_pct,
       fte_growth_3y_pct,
       workforce_stability
FROM calc_employment_features() calc_employment_features(isin, revenue_per_employee, profit_per_employee,
                                                         ebitda_per_employee, assets_per_employee, fte_growth_1y_pct,
                                                         fte_growth_3y_pct, workforce_stability);

alter table vw_features_employee_productivity
    owner to postgres;

