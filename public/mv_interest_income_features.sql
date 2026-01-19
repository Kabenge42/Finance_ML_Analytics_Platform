create materialized view mv_interest_income_features as
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
FROM v_interest_income_features;

alter materialized view mv_interest_income_features owner to postgres;

create index idx_mv_int_ticker
    on mv_interest_income_features (ticker);

create index idx_mv_int_coverage
    on mv_interest_income_features (interest_coverage_ebitda desc);

