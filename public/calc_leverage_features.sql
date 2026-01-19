create function calc_leverage_features()
    returns TABLE
            (
                ticker                text,
                debt_to_equity        numeric,
                debt_to_assets        numeric,
                equity_ratio          numeric,
                interest_coverage     numeric,
                current_ratio         numeric,
                cash_ratio            numeric,
                working_capital_ratio numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                    AS ticker,
       -- Debt to Equity (NULLIF handles zero division)
       "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0)                        AS debt_to_equity,
       -- Debt to Assets
       "Total Debt (LTM)" / NULLIF("Total Assets (LTM)", 0)                        AS debt_to_assets,
       -- Equity Ratio
       "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS equity_ratio,
       -- Interest Coverage
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                    AS interest_coverage,
       -- Current Ratio
       "Current Ratio (LTM)"                                                       AS current_ratio,
       -- Cash Ratio
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) AS cash_ratio,
       -- Working Capital Ratio
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0)                   AS working_capital_ratio
FROM postgres.public.equities;
$$;

alter function calc_leverage_features() owner to postgres;

