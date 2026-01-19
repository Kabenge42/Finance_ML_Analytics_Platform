create view v_interest_income_features
            (ticker, interest_income_ltm, interest_expense_ltm, net_interest_income, interest_coverage_ebit,
             interest_coverage_ebitda, interest_income_to_revenue, net_interest_margin, non_operating_income_ratio,
             financial_income_quality, interest_burden_ratio)
as
SELECT ticker,
       interest_income_ltm,
       interest_expense_ltm,
       net_interest_income,
       interest_coverage_ebit,
       interest_coverage_ebitda,
       interest_income_to_revenue,
       net_interest_margin,
       non_operating_income_ratio,
       financial_income_quality,
       interest_burden_ratio
FROM calc_interest_income_features() calc_interest_income_features(ticker, interest_income_ltm, interest_expense_ltm,
                                                                   net_interest_income, interest_coverage_ebit,
                                                                   interest_coverage_ebitda, interest_income_to_revenue,
                                                                   net_interest_margin, non_operating_income_ratio,
                                                                   financial_income_quality, interest_burden_ratio);

alter table v_interest_income_features
    owner to postgres;

