-- =============================================================================
-- NEW FEATURE FUNCTIONS FOR EXPANDED COVERAGE
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1: Profitability Trend Features (ROE/ROA analysis over time)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_profitability_trend_features()
    RETURNS TABLE
            (
                isin                 TEXT,
                ticker               TEXT,
                roe_ltm              NUMERIC,
                roe_fy               NUMERIC,
                roa_ltm              NUMERIC,
                roa_fy               NUMERIC,
                roe_roa_spread       NUMERIC,
                gross_margin_ltm     NUMERIC,
                net_margin_ltm       NUMERIC,
                margin_expansion     NUMERIC,
                operating_margin_ltm NUMERIC,
                profitability_score  NUMERIC
            )
AS
$$
SELECT "ISIN"                                                             AS isin,
       "Ticker"                                                           AS ticker,
       "Return On Equity % (LTM)"                                         AS roe_ltm,
       "Return On Equity % (FY)"                                          AS roe_fy,
       "Return on Assets (ROA) % (LTM)"                                   AS roa_ltm,
       "Return on Assets (ROA) % (FY)"                                    AS roa_fy,
       "Return On Equity % (LTM)" - "Return on Assets (ROA) % (LTM)"      AS roe_roa_spread,
       "Gross Profit Margin % (LTM)"                                      AS gross_margin_ltm,
       "Net Income Margin % (LTM)"                                        AS net_margin_ltm,
       "Net Income Margin % (LTM)" - "Net Income Margin % (FY)"           AS margin_expansion,
       "Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100 AS operating_margin_ltm,
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Return On Equity % (LTM)" > 15 THEN 30
                              WHEN "Return On Equity % (LTM)" > 10 THEN 20
                              ELSE 10 END) +
                         (CASE
                              WHEN "Gross Profit Margin % (LTM)" > 40 THEN 35
                              WHEN "Gross Profit Margin % (LTM)" > 25 THEN 25
                              ELSE 10 END) +
                         (CASE
                              WHEN "Net Income Margin % (LTM)" > 10 THEN 35
                              WHEN "Net Income Margin % (LTM)" > 5 THEN 25
                              ELSE 10 END)
                   ))                                                     AS profitability_score
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- 2: Debt & Leverage Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_debt_leverage_features()
    RETURNS TABLE
            (
                isin                  TEXT,
                ticker                TEXT,
                total_debt_ltm        NUMERIC,
                total_equity_ltm      NUMERIC,
                debt_to_equity        NUMERIC,
                debt_to_assets        NUMERIC,
                debt_to_ebitda        NUMERIC,
                net_debt              NUMERIC,
                net_debt_to_ebitda    NUMERIC,
                financial_leverage    NUMERIC,
                debt_service_coverage NUMERIC,
                leverage_risk_score   INTEGER
            )
AS
$$
SELECT "ISIN"                                                 AS isin,
       "Ticker"                                               AS ticker,
       "Total Debt (LTM)"                                     AS total_debt_ltm,
       "Total Equity (LTM)"                                   AS total_equity_ltm,
       "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0)   AS debt_to_equity,
       "Total Debt (LTM)" / NULLIF("Total Assets (LTM)", 0)   AS debt_to_assets,
       "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0)         AS debt_to_ebitda,
       "Total Debt (LTM)" - "Cash And Equivalents (LTM)"      AS net_debt,
       ("Total Debt (LTM)" - "Cash And Equivalents (LTM)") /
       NULLIF("EBITDA (LTM)", 0)                              AS net_debt_to_ebitda,
       "Total Assets (LTM)" / NULLIF("Total Equity (LTM)", 0) AS financial_leverage,
       ("EBITDA (LTM)" - "Capital Expenditure (LTM)") /
       NULLIF("Interest Expense/Total (LTM)", 0)              AS debt_service_coverage,
       CASE
           WHEN "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0) > 4 THEN 3
           WHEN "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0) > 2 THEN 2
           WHEN "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0) > 0 THEN 1
           ELSE 0
           END                                                AS leverage_risk_score
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- 3: Cash Flow Quality Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_cashflow_quality_features()
    RETURNS TABLE
            (
                isin                  TEXT,
                ticker                TEXT,
                cfo_ltm               NUMERIC,
                fcf_ltm               NUMERIC,
                cfo_to_net_income     NUMERIC,
                fcf_to_cfo            NUMERIC,
                capex_to_cfo          NUMERIC,
                cash_conversion_cycle NUMERIC,
                fcf_growth_yoy        NUMERIC,
                operating_cash_margin NUMERIC,
                reinvestment_rate     NUMERIC,
                cash_quality_score    NUMERIC
            )
AS
$$
SELECT "ISIN"                                                            AS isin,
       "Ticker"                                                          AS ticker,
       "CFO (LTM)"                                                       AS cfo_ltm,
       "FCF (LTM)"                                                       AS fcf_ltm,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)                AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("CFO (LTM)", 0)                              AS fcf_to_cfo,
       ABS("Capital Expenditure (LTM)") / NULLIF("CFO (LTM)", 0)         AS capex_to_cfo,
       ("Inventory (LTM)" + "Working Capital (LTM)") /
       NULLIF("Total Revenues (LTM)" / 365, 0)                           AS cash_conversion_cycle,
       ("FCF (LTM)" - "FCF (-1FY)") / NULLIF(ABS("FCF (-1FY)"), 0) * 100 AS fcf_growth_yoy,
       "CFO (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100             AS operating_cash_margin,
       ABS("Capital Expenditure (LTM)") / NULLIF("CFO (LTM)", 0)         AS reinvestment_rate,
       GREATEST(0, LEAST(100,
                         (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 40 ELSE 20 END) +
                         (CASE WHEN "FCF (LTM)" > 0 THEN 30 ELSE 0 END) +
                         (CASE WHEN "FCF (LTM)" / NULLIF("CFO (LTM)", 0) > 0.5 THEN 30 ELSE 15 END)
                   ))                                                    AS cash_quality_score
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- 4: Dividend Sustainability Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_dividend_sustainability_features()
    RETURNS TABLE
            (
                isin                     TEXT,
                ticker                   TEXT,
                div_yield_ltm            NUMERIC,
                payout_ratio             NUMERIC,
                dividend_coverage_fcf    NUMERIC,
                dividend_coverage_eps    NUMERIC,
                dividend_streak          NUMERIC,
                dps_growth               NUMERIC,
                buyback_yield_ltm        NUMERIC,
                total_shareholder_return NUMERIC,
                dividend_safety_score    NUMERIC,
                income_investor_flag     INTEGER
            )
AS
$$
SELECT "ISIN"                                                           AS isin,
       "Ticker"                                                         AS ticker,
       "Div Yield (LTM)"                                                AS div_yield_ltm,
       "Payout Ratio (TTM)"                                             AS payout_ratio,
       "Dividend Per Share (LTM)" * "Shrs Out" / NULLIF("FCF (LTM)", 0) AS dividend_coverage_fcf,
       "Dividend Per Share (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0)  AS dividend_coverage_eps,
       "Dividend Streak"                                                AS dividend_streak,
       "DPS Growth"                                                     AS dps_growth,
       "Buyback Yield (LTM)"                                            AS buyback_yield_ltm,
       COALESCE("Div Yield (LTM)", 0) + COALESCE("Buyback Yield (LTM)", 0) +
       (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 100)
                                                                        AS total_shareholder_return,
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Payout Ratio (TTM)" < 60 THEN 40
                              WHEN "Payout Ratio (TTM)" < 80 THEN 25
                              ELSE 10 END) +
                         (CASE
                              WHEN "Dividend Streak" > 10 THEN 30
                              WHEN "Dividend Streak" > 5 THEN 20
                              ELSE 5 END) +
                         (CASE
                              WHEN "FCF (LTM)" > "Dividend Per Share (LTM)" * "Shrs Out" THEN 30
                              ELSE 10 END)
                   ))                                                   AS dividend_safety_score,
       CASE
           WHEN "Div Yield (LTM)" > 3 AND "Dividend Streak" > 5 AND "Payout Ratio (TTM)" < 80
               THEN 1
           ELSE 0
           END                                                          AS income_investor_flag
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- 5: EPS Revision Momentum Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_eps_revision_features()
    RETURNS TABLE
            (
                isin                     TEXT,
                ticker                   TEXT,
                eps_rev_1w               NUMERIC,
                eps_rev_1m               NUMERIC,
                eps_rev_3m               NUMERIC,
                eps_rev_6m               NUMERIC,
                eps_rev_acceleration     NUMERIC,
                revision_momentum_score  NUMERIC,
                estimate_trend_direction INTEGER,
                revision_breadth         NUMERIC,
                earnings_surprise_proxy  NUMERIC,
                analyst_sentiment_flag   INTEGER
            )
AS
$$
SELECT "ISIN"                                                            AS isin,
       "Ticker"                                                          AS ticker,
       "EPS Est Avg Rev % (FY1E - 1W)"                                   AS eps_rev_1w,
       "EPS Est Avg Rev % (FY1E - 1M)"                                   AS eps_rev_1m,
       "EPS Est Avg Rev % (FY1E - 3M)"                                   AS eps_rev_3m,
       "EPS Est Avg Rev % (FY1E - 6M)"                                   AS eps_rev_6m,
       "EPS Est Avg Rev % (FY1E - 1M)" - "EPS Est Avg Rev % (FY1E - 3M)" AS eps_rev_acceleration,
       (COALESCE("EPS Est Avg Rev % (FY1E - 1W)", 0) * 0.4 +
        COALESCE("EPS Est Avg Rev % (FY1E - 1M)", 0) * 0.3 +
        COALESCE("EPS Est Avg Rev % (FY1E - 3M)", 0) * 0.2 +
        COALESCE("EPS Est Avg Rev % (FY1E - 6M)", 0) * 0.1)              AS revision_momentum_score,
       CASE
           WHEN "EPS Est Avg Rev % (FY1E - 1M)" > 0 AND "EPS Est Avg Rev % (FY1E - 3M)" > 0 THEN 1
           WHEN "EPS Est Avg Rev % (FY1E - 1M)" < 0 AND "EPS Est Avg Rev % (FY1E - 3M)" < 0 THEN -1
           ELSE 0
           END                                                           AS estimate_trend_direction,
       (CASE WHEN "EPS Est Avg Rev % (FY1E - 1W)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EPS Est Avg Rev % (FY1E - 1M)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EPS Est Avg Rev % (FY1E - 3M)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EPS Est Avg Rev % (FY1E - 6M)" > 0 THEN 1 ELSE 0 END) / 4.0
                                                                         AS revision_breadth,
       "EPS/Adj. (LTM)" / NULLIF("EPS Norm - Est Avg (FY1E)", 0) - 1     AS earnings_surprise_proxy,
       CASE
           WHEN "EPS Est Avg Rev % (FY1E - 1M)" > 2 AND "EPS Est Avg Rev % (FY1E - 3M)" > 0
               THEN 1
           ELSE 0
           END                                                           AS analyst_sentiment_flag
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- 6: Employee Productivity Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_employee_productivity_features()
    RETURNS TABLE
            (
                isin                        TEXT,
                ticker                      TEXT,
                employees_fy                NUMERIC,
                revenue_per_employee        NUMERIC,
                profit_per_employee         NUMERIC,
                employee_growth_yoy         NUMERIC,
                employee_growth_2y          NUMERIC,
                revenue_growth_vs_headcount NUMERIC,
                productivity_trend          NUMERIC,
                human_capital_roi           NUMERIC,
                efficiency_flag             INTEGER
            )
AS
$$
SELECT "ISIN"                                                              AS isin,
       "Ticker"                                                            AS ticker,
       "Full Time Employees (FY)"                                          AS employees_fy,
       "Total Revenues (LTM)" / NULLIF("Full Time Employees (FY)", 0)      AS revenue_per_employee,
       "Net Income - (IS) (LTM)" / NULLIF("Full Time Employees (FY)", 0)   AS profit_per_employee,
       ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
       NULLIF("Full Time Employees (-1FY)", 0) * 100                       AS employee_growth_yoy,
       ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
       NULLIF("Full Time Employees (-2FY)", 0) * 100                       AS employee_growth_2y,
       (("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF("Total Revenues (-1FY)", 0)) -
       (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
        NULLIF("Full Time Employees (-1FY)", 0))                           AS revenue_growth_vs_headcount,
       ("Total Revenues (FY)" / NULLIF("Full Time Employees (FY)", 0)) -
       ("Total Revenues (-1FY)" / NULLIF("Full Time Employees (-1FY)", 0)) AS productivity_trend,
       "EBIT (LTM)" / NULLIF("Selling General & Admin Expenses/Total (FY)", 0)
                                                                           AS human_capital_roi,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF("Total Revenues (-1FY)", 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               THEN 1
           ELSE 0
           END                                                             AS efficiency_flag
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- INSERT NEW FUNCTIONS INTO FEATURE REGISTRY
-- =============================================================================
INSERT INTO feature_registry_metadata (function_name, category, feature_count, description, python_equivalent)
VALUES ('calc_profitability_trend_features', 'Profitability', 10,
        'ROE, ROA, margins, profitability scoring',
        'engineer_profitability_features'),
       ('calc_debt_leverage_features', 'Leverage', 10,
        'Debt ratios, leverage metrics, debt service coverage',
        'engineer_leverage_features'),
       ('calc_cashflow_quality_features', 'Cash Flow Quality', 10,
        'CFO quality, FCF conversion, cash cycle analysis',
        'engineer_cashflow_quality'),
       ('calc_dividend_sustainability_features', 'Dividend Analysis', 10,
        'Dividend safety, coverage, sustainability metrics',
        'engineer_dividend_features'),
       ('calc_eps_revision_features', 'EPS Revisions', 10,
        'Analyst revision momentum, estimate trends',
        'engineer_eps_revision_features'),
       ('calc_employee_productivity_features', 'Productivity', 9,
        'Revenue/profit per employee, productivity trends',
        'engineer_productivity_features')
ON CONFLICT (function_name) DO UPDATE
    SET category          = EXCLUDED.category,
        feature_count     = EXCLUDED.feature_count,
        description       = EXCLUDED.description,
        python_equivalent = EXCLUDED.python_equivalent;
