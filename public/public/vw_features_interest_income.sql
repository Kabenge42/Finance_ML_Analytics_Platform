create view vw_features_interest_income
            (isin, interest_income_ltm, interest_expense_ltm, net_interest_income, interest_coverage_ratio,
             interest_income_to_revenue, interest_expense_to_revenue, net_interest_margin_proxy)
as
SELECT isin,
       interest_income_ltm,
       interest_expense_ltm,
       net_interest_income,
       interest_coverage_ratio,
       interest_income_to_revenue,
       interest_expense_to_revenue,
       net_interest_margin_proxy
FROM calc_interest_income_features() calc_interest_income_features(isin, interest_income_ltm, interest_expense_ltm,
                                                                   net_interest_income, interest_coverage_ratio,
                                                                   interest_income_to_revenue,
                                                                   interest_expense_to_revenue,
                                                                   net_interest_margin_proxy);

alter table vw_features_interest_income
    owner to postgres;

