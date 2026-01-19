create function calc_interest_income_features()
    returns TABLE
            (
                ticker                     text,
                interest_income_ltm        numeric,
                interest_expense_ltm       numeric,
                net_interest_income        numeric,
                interest_coverage_ebit     numeric,
                interest_coverage_ebitda   numeric,
                interest_income_to_revenue numeric,
                net_interest_margin        numeric,
                non_operating_income_ratio numeric,
                financial_income_quality   numeric,
                interest_burden_ratio      numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                AS ticker,
       "Interest Income On Investments (LTM)"                                  AS interest_income_ltm,
       "Interest Expense/Total (LTM)"                                          AS interest_expense_ltm,
       -- Net Interest Income
       "Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)" AS net_interest_income,
       -- Interest Coverage (EBIT)
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                AS interest_coverage_ebit,
       -- Interest Coverage (EBITDA)
       "EBITDA (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)              AS interest_coverage_ebitda,
       -- Interest Income as % of Revenue
       "Interest Income On Investments (LTM)" /
       NULLIF("Total Revenues (LTM)", 0) * 100                                 AS interest_income_to_revenue,
       -- Net Interest Margin (Net Interest / Avg Assets)
       ("Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)") /
       NULLIF("Total Assets (LTM)", 0) * 100                                   AS net_interest_margin,
       -- Non-Operating Income Ratio
       ("Interest Income On Investments (LTM)" +
        "Gain (Loss) On Sale Of Assets (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0)                               AS non_operating_income_ratio,
       -- Financial Income Quality (operating income / total income)
       "Operating Income (LTM)" /
       NULLIF("Operating Income (LTM)" + "Interest Income On Investments (LTM)", 0)
                                                                               AS financial_income_quality,
       -- Interest Burden Ratio (Interest Expense / EBIT)
       "Interest Expense/Total (LTM)" / NULLIF("EBIT (LTM)", 0)                AS interest_burden_ratio
FROM postgres.public.equities;
$$;

alter function calc_interest_income_features() owner to postgres;

