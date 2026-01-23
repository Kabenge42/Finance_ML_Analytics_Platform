-- ... existing code ...

-- =============================================================================
-- SECTION 2B: LONG-TERM MOMENTUM FEATURES (NEW)
-- =============================================================================

-- Long-Term Momentum Features (engineer_long_term_momentum_features)
CREATE OR REPLACE FUNCTION calc_long_term_momentum_features()
    RETURNS TABLE
            (
                ticker                   TEXT,
                price_momentum_qtd       NUMERIC,
                price_momentum_3y        NUMERIC,
                price_momentum_5y        NUMERIC,
                momentum_acceleration_1y NUMERIC,
                momentum_acceleration_3y NUMERIC,
                long_term_trend_score    NUMERIC,
                price_vs_3y_avg          NUMERIC,
                price_vs_5y_avg          NUMERIC,
                momentum_consistency     NUMERIC,
                secular_trend_flag       INTEGER
            )
AS
$$
SELECT "Ticker"                                                            AS ticker,
       -- QTD Momentum
       ("Last Price" - "Price (QTD Ago)") /
       NULLIF("Price (QTD Ago)", 0) * 100                                  AS price_momentum_qtd,
       -- 3Y Momentum
       ("Last Price" - "Price (3Y Ago)") /
       NULLIF("Price (3Y Ago)", 0) * 100                                   AS price_momentum_3y,
       -- 5Y Momentum
       ("Last Price" - "Price (5Y Ago)") /
       NULLIF("Price (5Y Ago)", 0) * 100                                   AS price_momentum_5y,
       -- Momentum Acceleration (1Y vs 3Y CAGR)
       (POWER("Last Price" / NULLIF("Price (1Y Ago)", 0), 1.0) - 1) -
       (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1)  AS momentum_acceleration_1y,
       -- Momentum Acceleration (3Y vs 5Y CAGR)
       (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1) -
       (POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1)  AS momentum_acceleration_3y,
       -- Long-Term Trend Score (weighted momentum)
       (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 0.5 +
        ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 0.3 +
        ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 0.2) * 100
                                                                           AS long_term_trend_score,
       -- Price vs 3Y Simple Average (proxy)
       "Last Price" / NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)") / 2, 0) AS price_vs_3y_avg,
       -- Price vs 5Y Simple Average (proxy)
       "Last Price" / NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)" + "Price (5Y Ago)") / 3, 0)
                                                                           AS price_vs_5y_avg,
       -- Momentum Consistency (all timeframes positive)
       CASE
           WHEN ("Last Price" > "Price (1M Ago)") AND
                ("Last Price" > "Price (3M Ago)") AND
                ("Last Price" > "Price (1Y Ago)") AND
                ("Last Price" > "Price (3Y Ago)")
               THEN 1.0
           WHEN ("Last Price" > "Price (1M Ago)") AND
                ("Last Price" > "Price (3M Ago)") AND
                ("Last Price" > "Price (1Y Ago)")
               THEN 0.75
           WHEN ("Last Price" > "Price (1M Ago)") AND
                ("Last Price" > "Price (3M Ago)")
               THEN 0.5
           WHEN ("Last Price" > "Price (1M Ago)")
               THEN 0.25
           ELSE 0
           END                                                             AS momentum_consistency,
       -- Secular Trend Flag (5Y CAGR > 10%)
       CASE
           WHEN POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1 > 0.10
               THEN 1
           ELSE 0
           END                                                             AS secular_trend_flag
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 2C: BETA RISK FEATURES (NEW)
-- =============================================================================

-- Beta Risk Features (engineer_beta_risk_features)
CREATE OR REPLACE FUNCTION calc_beta_risk_features()
    RETURNS TABLE
            (
                ticker                TEXT,
                beta_1y               NUMERIC,
                beta_2y               NUMERIC,
                beta_5y               NUMERIC,
                beta_trend_short      NUMERIC,
                beta_trend_long       NUMERIC,
                beta_stability        NUMERIC,
                beta_regime_change    INTEGER,
                systematic_risk_score NUMERIC,
                defensive_stock_flag  INTEGER,
                high_beta_flag        INTEGER
            )
AS
$$
SELECT "Ticker"                                                  AS ticker,
       "Beta (1Y)"                                               AS beta_1y,
       "Beta (2Y)"                                               AS beta_2y,
       "Beta (5Y)"                                               AS beta_5y,
       -- Beta Trend Short-Term (1Y vs 2Y)
       "Beta (1Y)" - "Beta (2Y)"                                 AS beta_trend_short,
       -- Beta Trend Long-Term (2Y vs 5Y)
       "Beta (2Y)" - "Beta (5Y)"                                 AS beta_trend_long,
       -- Beta Stability (inverse of range)
       CASE
           WHEN GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
                LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") > 0
               THEN 1.0 / (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
                           LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)"))
           ELSE 1.0
           END                                                   AS beta_stability,
       -- Beta Regime Change Flag (significant change between periods)
       CASE
           WHEN ABS("Beta (1Y)" - "Beta (5Y)") > 0.3 THEN 1
           ELSE 0
           END                                                   AS beta_regime_change,
       -- Systematic Risk Score (weighted beta)
       "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2 AS systematic_risk_score,
       -- Defensive Stock Flag (beta consistently < 0.8)
       CASE
           WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8
               THEN 1
           ELSE 0
           END                                                   AS defensive_stock_flag,
       -- High Beta Flag (beta consistently > 1.3)
       CASE
           WHEN "Beta (1Y)" > 1.3 AND "Beta (2Y)" > 1.3
               THEN 1
           ELSE 0
           END                                                   AS high_beta_flag
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 3A: INTEREST INCOME FEATURES (NEW)
-- =============================================================================

-- Interest Income Features (engineer_interest_income_features)
CREATE OR REPLACE FUNCTION calc_interest_income_features()
    RETURNS TABLE
            (
                ticker                     TEXT,
                interest_income_ltm        NUMERIC,
                interest_expense_ltm       NUMERIC,
                net_interest_income        NUMERIC,
                interest_coverage_ebit     NUMERIC,
                interest_coverage_ebitda   NUMERIC,
                interest_income_to_revenue NUMERIC,
                net_interest_margin        NUMERIC,
                non_operating_income_ratio NUMERIC,
                financial_income_quality   NUMERIC,
                interest_burden_ratio      NUMERIC
            )
AS
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
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 5A: TANGIBLE BOOK FEATURES (NEW)
-- =============================================================================

-- Tangible Book Features (engineer_tangible_book_features)
CREATE OR REPLACE FUNCTION calc_tangible_book_features()
    RETURNS TABLE
            (
                ticker                  TEXT,
                tbv_fy                  NUMERIC,
                tbv_ltm                 NUMERIC,
                price_to_tbv            NUMERIC,
                tbv_per_share           NUMERIC,
                tbv_growth_yoy          NUMERIC,
                tangible_equity_ratio   NUMERIC,
                intangible_to_tbv_ratio NUMERIC,
                tbv_vs_market_cap       NUMERIC,
                net_tangible_assets     NUMERIC,
                tbv_margin_of_safety    NUMERIC
            )
AS
$$
SELECT "Ticker"                                                      AS ticker,
       "TBV (FY)"                                                    AS tbv_fy,
       "TBV (LTM)"                                                   AS tbv_ltm,
       -- Price to Tangible Book Value
       "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0), 0) AS price_to_tbv,
       -- TBV Per Share
       "TBV (LTM)" / NULLIF("Shrs Out", 0)                           AS tbv_per_share,
       -- TBV Growth YoY
       ("TBV (LTM)" - "TBV (FY)") / NULLIF(ABS("TBV (FY)"), 0) * 100 AS tbv_growth_yoy,
       -- Tangible Equity as % of Total Equity
       "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0)                 AS tangible_equity_ratio,
       -- Intangibles to TBV Ratio (lower = better asset quality)
       ("Goodwill (LTM)" + "Gross Intangible Assets (LTM)") /
       NULLIF("TBV (LTM)", 0)                                        AS intangible_to_tbv_ratio,
       -- TBV vs Market Cap (margin of safety)
       "TBV (LTM)" / NULLIF("Market Cap", 0)                         AS tbv_vs_market_cap,
       -- Net Tangible Assets (TBV - Total Debt)
       "TBV (LTM)" - "Total Debt (LTM)"                              AS net_tangible_assets,
       -- TBV Margin of Safety (discount to TBV)
       ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0) * 100   AS tbv_margin_of_safety
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 5B: WORKING CAPITAL DEEP FEATURES (NEW)
-- =============================================================================

-- Working Capital Deep Features (engineer_working_capital_deep_features)
CREATE OR REPLACE FUNCTION calc_working_capital_deep_features()
    RETURNS TABLE
            (
                ticker                     TEXT,
                current_assets_ltm         NUMERIC,
                current_liabilities_ltm    NUMERIC,
                net_working_capital        NUMERIC,
                working_capital_to_revenue NUMERIC,
                working_capital_to_assets  NUMERIC,
                current_ratio              NUMERIC,
                quick_ratio                NUMERIC,
                cash_ratio                 NUMERIC,
                defensive_interval         NUMERIC,
                working_capital_turnover   NUMERIC,
                liquidity_score            NUMERIC,
                working_capital_efficiency NUMERIC
            )
AS
$$
SELECT "Ticker"                                                         AS ticker,
       "Total Current Assets (LTM)"                                     AS current_assets_ltm,
       "Total Current Liabilities (LTM)"                                AS current_liabilities_ltm,
       -- Net Working Capital
       "Total Current Assets (LTM)" - "Total Current Liabilities (LTM)" AS net_working_capital,
       -- Working Capital to Revenue
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0)      AS working_capital_to_revenue,
       -- Working Capital to Assets
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0)        AS working_capital_to_assets,
       -- Current Ratio
       "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0)
                                                                        AS current_ratio,
       -- Quick Ratio (exclude inventory)
       ("Total Current Assets (LTM)" - "Inventory (LTM)") /
       NULLIF("Total Current Liabilities (LTM)", 0)                     AS quick_ratio,
       -- Cash Ratio
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0)
                                                                        AS cash_ratio,
       -- Defensive Interval (days of OpEx covered by current assets)
       ("Total Current Assets (LTM)" /
        NULLIF("Total Operating Expenses (LTM)" / 365, 0))              AS defensive_interval,
       -- Working Capital Turnover
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)      AS working_capital_turnover,
       -- Liquidity Score (weighted average of ratios, 0-100)
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 2
                                  THEN 40
                              WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1.5
                                  THEN 30
                              WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1
                                  THEN 20
                              ELSE 0
                             END) +
                         (CASE
                              WHEN ("Total Current Assets (LTM)" - "Inventory (LTM)") /
                                   NULLIF("Total Current Liabilities (LTM)", 0) >= 1 THEN 30
                              ELSE 15
                             END) +
                         (CASE
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.5
                                  THEN 30
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.2
                                  THEN 15
                              ELSE 0
                             END)
                   ))                                                   AS liquidity_score,
       -- Working Capital Efficiency (Revenue generated per $ of WC)
       "Total Revenues (LTM)" / NULLIF(ABS("Working Capital (LTM)"), 0) AS working_capital_efficiency
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 7C: UNUSUAL ITEMS FEATURES (NEW)
-- =============================================================================

-- Unusual Items Features (engineer_unusual_items_features)
CREATE OR REPLACE FUNCTION calc_unusual_items_features()
    RETURNS TABLE
            (
                ticker                      TEXT,
                other_unusual_items_ltm     NUMERIC,
                total_unusual_items         NUMERIC,
                unusual_to_revenue_ratio    NUMERIC,
                unusual_to_ebitda_ratio     NUMERIC,
                unusual_to_net_income_ratio NUMERIC,
                clean_earnings_flag         INTEGER,
                recurring_unusual_flag      INTEGER,
                earnings_noise_score        NUMERIC,
                quality_adjusted_ni         NUMERIC,
                exceptional_items_impact    NUMERIC
            )
AS
$$
SELECT "Ticker"                                                     AS ticker,
       "Other Unusual Items/Total (LTM)"                            AS other_unusual_items_ltm,
       -- Total Unusual Items (sum of all exceptional items)
       ABS("Impairment of Goodwill (LTM)") +
       ABS("Asset Writedown (LTM)") +
       ABS("Restructuring Charges (LTM)") +
       ABS("Merger & Restructuring Charges (LTM)") +
       ABS("Gain (Loss) On Sale Of Assets (LTM)") +
       ABS("Other Unusual Items/Total (LTM)")                       AS total_unusual_items,
       -- Unusual Items to Revenue Ratio
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF("Total Revenues (LTM)", 0) * 100                      AS unusual_to_revenue_ratio,
       -- Unusual Items to EBITDA Ratio
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)                               AS unusual_to_ebitda_ratio,
       -- Unusual Items to Net Income Ratio
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0)                    AS unusual_to_net_income_ratio,
       -- Clean Earnings Flag (no unusual items)
       CASE
           WHEN ABS("Impairment of Goodwill (LTM)") < 1 AND
                ABS("Asset Writedown (LTM)") < 1 AND
                ABS("Restructuring Charges (LTM)") < 1 AND
                ABS("Other Unusual Items/Total (LTM)") < 1
               THEN 1
           ELSE 0
           END                                                      AS clean_earnings_flag,
       -- Recurring Unusual Items Flag (unusual in multiple years)
       CASE
           WHEN (CASE WHEN ABS("Impairment of Goodwill (FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Impairment of Goodwill (-1FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Restructuring Charges (FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Restructuring Charges (-1FY)") > 0 THEN 1 ELSE 0 END) >= 3
               THEN 1
           ELSE 0
           END                                                      AS recurring_unusual_flag,
       -- Earnings Noise Score (0-100, lower = cleaner)
       LEAST(100,
             (ABS("Impairment of Goodwill (LTM)") +
              ABS("Asset Writedown (LTM)") +
              ABS("Restructuring Charges (LTM)") +
              ABS("Other Unusual Items/Total (LTM)")) /
             NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100)       AS earnings_noise_score,
       -- Quality Adjusted Net Income
       "Net Income - (IS) (LTM)" +
       "Impairment of Goodwill (LTM)" +
       "Asset Writedown (LTM)" +
       "Restructuring Charges (LTM)"                                AS quality_adjusted_ni,
       -- Exceptional Items Impact on EPS (per share)
       (ABS("Impairment of Goodwill (LTM)") +
        ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF("Shrs Out", 0) AS exceptional_items_impact
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 8A: REVENUE QUARTERLY FEATURES (NEW)
-- =============================================================================

-- Revenue Quarterly Features (engineer_revenue_quarterly_features)
CREATE OR REPLACE FUNCTION calc_revenue_quarterly_features()
    RETURNS TABLE
            (
                ticker                     TEXT,
                revenue_fq                 NUMERIC,
                revenue_ltm                NUMERIC,
                revenue_fy                 NUMERIC,
                revenue_1fy                NUMERIC,
                revenue_5yavg_fq           NUMERIC,
                revenue_5yavg_ltm          NUMERIC,
                revenue_fq_vs_5yavg        NUMERIC,
                revenue_ltm_vs_5yavg       NUMERIC,
                revenue_qoq_growth         NUMERIC,
                revenue_yoy_growth         NUMERIC,
                revenue_quarterly_run_rate NUMERIC,
                revenue_seasonality_factor NUMERIC
            )
AS
$$
SELECT "Ticker"                                                        AS ticker,
       "Total Revenues (FQ)"                                           AS revenue_fq,
       "Total Revenues (LTM)"                                          AS revenue_ltm,
       "Total Revenues (FY)"                                           AS revenue_fy,
       "Total Revenues (-1FY)"                                         AS revenue_1fy,
       "Total Revenues (5YAVGFQ)"                                      AS revenue_5yavg_fq,
       "Total Revenues (5YAVGLTM)"                                     AS revenue_5yavg_ltm,
       -- FQ vs 5Y Average (seasonality-adjusted benchmark)
       "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0)   AS revenue_fq_vs_5yavg,
       -- LTM vs 5Y Average
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_ltm_vs_5yavg,
       -- QoQ Growth (FQ annualized vs LTM)
       ("Total Revenues (FQ)" * 4 - "Total Revenues (LTM)") /
       NULLIF("Total Revenues (LTM)", 0) * 100                         AS revenue_qoq_growth,
       -- YoY Growth
       ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
       NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100                   AS revenue_yoy_growth,
       -- Quarterly Run Rate (annualized FQ)
       "Total Revenues (FQ)" * 4                                       AS revenue_quarterly_run_rate,
       -- Seasonality Factor (FQ as % of typical quarter)
       "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4, 0)   AS revenue_seasonality_factor
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 8B: REVENUE ESTIMATE CONSENSUS (NEW)
-- =============================================================================

-- Revenue Estimate Consensus Features (engineer_revenue_estimate_consensus)
CREATE OR REPLACE FUNCTION calc_revenue_estimate_consensus()
    RETURNS TABLE
            (
                ticker                 TEXT,
                revenue_est_avg_ntm    NUMERIC,
                revenue_est_med_ntm    NUMERIC,
                revenue_est_avg_fy1e   NUMERIC,
                revenue_est_med_fy1e   NUMERIC,
                estimate_skew_ntm      NUMERIC,
                estimate_skew_fy1e     NUMERIC,
                consensus_confidence   NUMERIC,
                upside_to_consensus    NUMERIC,
                estimate_vs_actual_ltm NUMERIC,
                forward_revenue_growth NUMERIC,
                revenue_beat_history   NUMERIC
            )
AS
$$
SELECT "Ticker"                                                                    AS ticker,
       "Revenues - Est Avg (NTM)"                                                  AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"                                                  AS revenue_est_med_ntm,
       "Revenues - Est Avg (FY1E)"                                                 AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"                                                 AS revenue_est_med_fy1e,
       -- Estimate Skew NTM (Avg vs Median, positive = optimistic outliers)
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Med (NTM)") /
       NULLIF("Revenues - Est Med (NTM)", 0) * 100                                 AS estimate_skew_ntm,
       -- Estimate Skew FY1E
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                                AS estimate_skew_fy1e,
       -- Consensus Confidence (lower skew = higher confidence)
       GREATEST(0, LEAST(100,
                         100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                   NULLIF("Revenues - Est Med (FY1E)", 0) * 100))) AS consensus_confidence,
       -- Upside to Consensus (current run-rate vs estimate)
       ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") /
       NULLIF("Total Revenues (LTM)", 0) * 100                                     AS upside_to_consensus,
       -- Estimate vs Actual LTM (how close were estimates)
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF("Revenues - Est Avg (FY1E)", 0) * 100                                AS estimate_vs_actual_ltm,
       -- Forward Revenue Growth (NTM vs LTM)
       ("Revenues - Est Med (NTM)" - "Total Revenues (LTM)") /
       NULLIF("Total Revenues (LTM)", 0) * 100                                     AS forward_revenue_growth,
       -- Revenue Beat History Proxy (actual > estimate)
       CASE
           WHEN "Total Revenues (LTM)" > "Revenues - Est Avg (FY1E)"
               THEN 1.0
           ELSE 0.0
           END                                                                     AS revenue_beat_history
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 8C: COST STRUCTURE FEATURES (NEW)
-- =============================================================================

-- Cost Structure Features (engineer_cost_structure_features)
CREATE OR REPLACE FUNCTION calc_cost_structure_features()
    RETURNS TABLE
            (
                ticker                   TEXT,
                sga_to_revenue_fq        NUMERIC,
                sga_to_revenue_fy        NUMERIC,
                sga_trend_yoy            NUMERIC,
                sga_vs_5yavg             NUMERIC,
                marketing_to_revenue_fq  NUMERIC,
                marketing_to_revenue_fy  NUMERIC,
                marketing_trend_yoy      NUMERIC,
                marketing_vs_5yavg       NUMERIC,
                operating_expense_ratio  NUMERIC,
                cost_of_revenue_ratio    NUMERIC,
                operating_leverage_score NUMERIC,
                cost_efficiency_trend    NUMERIC
            )
AS
$$
SELECT "Ticker"                                                              AS ticker,
       -- SG&A as % of Revenue (FQ)
       "Selling General & Admin Expenses/Total (FQ)" /
       NULLIF("Total Revenues (FQ)", 0) * 100                                AS sga_to_revenue_fq,
       -- SG&A as % of Revenue (FY)
       "Selling General & Admin Expenses/Total (FY)" /
       NULLIF("Total Revenues (FY)", 0) * 100                                AS sga_to_revenue_fy,
       -- SG&A YoY Trend
       ("Selling General & Admin Expenses/Total (FY)" -
        "Selling General & Admin Expenses/Total (-1FY)") /
       NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0) * 100 AS sga_trend_yoy,
       -- SG&A vs 5Y Average
       "Selling General & Admin Expenses/Total (FQ)" /
       NULLIF("Selling General & Admin Expenses/Total (5YAVGFQ)", 0)         AS sga_vs_5yavg,
       -- Marketing as % of Revenue (FQ)
       "Marketing Expenses (FQ)" / NULLIF("Total Revenues (FQ)", 0) * 100    AS marketing_to_revenue_fq,
       -- Marketing as % of Revenue (FY)
       "Marketing Expenses (FY)" / NULLIF("Total Revenues (FY)", 0) * 100    AS marketing_to_revenue_fy,
       -- Marketing YoY Trend
       ("Marketing Expenses (FY)" - "Marketing Expenses (-1FY)") /
       NULLIF(ABS("Marketing Expenses (-1FY)"), 0) * 100                     AS marketing_trend_yoy,
       -- Marketing vs 5Y Average
       ("Marketing Expenses (FY)" + "Marketing Expenses (-1FY)") / 2 /
       NULLIF("Marketing Expenses (5YAVGLTM)", 0)                            AS marketing_vs_5yavg,
       -- Total Operating Expense Ratio
       "Total Operating Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)  AS operating_expense_ratio,
       -- Cost of Revenue Ratio (COGS/Revenue)
       "Cost Of Revenues (LTM)" / NULLIF("Total Revenues (LTM)", 0)          AS cost_of_revenue_ratio,
       -- Operating Leverage Score (Revenue growth - OpEx growth sensitivity)
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0 AND ABS("Total Operating Expenses (LTM)") > 0
               THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                     NULLIF(ABS("Total Revenues (-1FY)"), 0)) /
                    NULLIF((("Selling General & Admin Expenses/Total (FY)" -
                             "Selling General & Admin Expenses/Total (-1FY)") /
                            NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0)), 0)
           END                                                               AS operating_leverage_score,
       -- Cost Efficiency Trend (declining SG&A ratio = improving efficiency)
       ("Selling General & Admin Expenses/Total (-1FY)" / NULLIF("Total Revenues (-1FY)", 0)) -
       ("Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0))
                                                                             AS cost_efficiency_trend
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 1A: ENHANCED VALUATION RATIOS (NEW)
-- =============================================================================

-- Enhanced Valuation Ratios (engineer_enhanced_valuation_ratios)
CREATE OR REPLACE FUNCTION calc_enhanced_valuation_ratios()
    RETURNS TABLE
            (
                ticker                    TEXT,
                forward_pe                NUMERIC,
                trailing_pe               NUMERIC,
                pe_forward_discount       NUMERIC,
                peg_ratio                 NUMERIC,
                peg_adjusted              NUMERIC,
                peg_forward               NUMERIC,
                pe_to_growth              NUMERIC,
                ev_to_fcf                 NUMERIC,
                earnings_yield            NUMERIC,
                fcf_yield                 NUMERIC,
                shareholder_yield_total   NUMERIC,
                valuation_composite_score NUMERIC
            )
AS
$$
SELECT "Ticker"                                                            AS ticker,
       "Forward P/E"                                                       AS forward_pe,
       "P/E (LTM)"                                                         AS trailing_pe,
       -- Forward P/E Discount vs Trailing
       ("P/E (LTM)" - "Forward P/E") / NULLIF("P/E (LTM)", 0) * 100        AS pe_forward_discount,
       -- PEG Ratio (pre-calculated)
       "PEG Ratio"                                                         AS peg_ratio,
       -- PEG Adjusted (using 5Y revenue CAGR instead)
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0
               THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0)
           END                                                             AS peg_adjusted,
       -- PEG Forward (Forward P/E / Forward Growth)
       CASE
           WHEN "Revenues - Est YoY % (FY1E)" > 0
               THEN "Forward P/E" / NULLIF("Revenues - Est YoY % (FY1E)", 0)
           END                                                             AS peg_forward,
       -- P/E to Growth (simpler ratio)
       "P/E (LTM)" / NULLIF("EPS Growth (TTM)", 0)                         AS pe_to_growth,
       -- EV to FCF
       "EV/FCF"                                                            AS ev_to_fcf,
       -- Earnings Yield (inverse of P/E)
       CASE
           WHEN "P/E (LTM)" > 0
               THEN 1.0 / "P/E (LTM)" * 100
           END                                                             AS earnings_yield,
       -- FCF Yield
       "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                         AS fcf_yield,
       -- Total Shareholder Yield (Dividend + Buyback)
       COALESCE("Div Yield (LTM)", 0) + COALESCE("Buyback Yield (LTM)", 0) AS shareholder_yield_total,
       -- Valuation Composite Score (lower = cheaper, 0-100)
       GREATEST(0, LEAST(100,
                         50 -
                         (CASE WHEN "P/E (LTM)" < 15 THEN 10 WHEN "P/E (LTM)" < 25 THEN 0 ELSE -10 END) -
                         (CASE WHEN "PEG Ratio" < 1 THEN 15 WHEN "PEG Ratio" < 2 THEN 5 ELSE -5 END) -
                         (CASE WHEN "EV/EBITDA (LTM)" < 10 THEN 10 WHEN "EV/EBITDA (LTM)" < 15 THEN 0 ELSE -10 END) +
                         (CASE WHEN "FCF (LTM)" / NULLIF("Market Cap", 0) * 100 > 5 THEN 15 ELSE 0 END)
                   ))                                                      AS valuation_composite_score
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 13E: ALL ENHANCED FEATURES COMPREHENSIVE (NEW)
-- =============================================================================

-- All Enhanced Features (comprehensive composite function)
CREATE OR REPLACE FUNCTION calc_all_enhanced_features()
    RETURNS TABLE
            (
                ticker                       TEXT,
                name                         TEXT,
                sector                       TEXT,
                industry                     TEXT,
                revenue_fq_vs_5yavg          NUMERIC,
                revenue_qoq_growth           NUMERIC,
                revenue_seasonality_factor   NUMERIC,
                sga_to_revenue_fy            NUMERIC,
                sga_trend_yoy                NUMERIC,
                marketing_to_revenue_fy      NUMERIC,
                operating_leverage_score     NUMERIC,
                cost_efficiency_trend        NUMERIC,
                price_to_tbv                 NUMERIC,
                tbv_per_share                NUMERIC,
                tangible_equity_ratio        NUMERIC,
                tbv_margin_of_safety         NUMERIC,
                net_interest_income          NUMERIC,
                interest_coverage_ebitda     NUMERIC,
                financial_income_quality     NUMERIC,
                interest_burden_ratio        NUMERIC,
                price_momentum_3y            NUMERIC,
                price_momentum_5y            NUMERIC,
                momentum_acceleration_3y     NUMERIC,
                long_term_trend_score        NUMERIC,
                momentum_consistency         NUMERIC,
                secular_trend_flag           INTEGER,
                beta_2y                      NUMERIC,
                beta_trend_short             NUMERIC,
                beta_trend_long              NUMERIC,
                beta_stability               NUMERIC,
                beta_regime_change           INTEGER,
                systematic_risk_score        NUMERIC,
                defensive_stock_flag         INTEGER,
                high_beta_flag               INTEGER,
                estimate_skew_fy1e           NUMERIC,
                revenue_consensus_confidence NUMERIC,
                upside_to_consensus          NUMERIC,
                total_unusual_items          NUMERIC,
                unusual_to_net_income_ratio  NUMERIC,
                clean_earnings_flag          INTEGER,
                recurring_unusual_flag       INTEGER,
                earnings_noise_score         NUMERIC,
                quality_adjusted_ni          NUMERIC,
                pe_forward_discount          NUMERIC,
                peg_adjusted                 NUMERIC,
                peg_forward                  NUMERIC,
                earnings_yield               NUMERIC,
                fcf_yield                    NUMERIC,
                shareholder_yield_total      NUMERIC,
                valuation_composite_score    NUMERIC,
                defensive_interval           NUMERIC,
                working_capital_turnover     NUMERIC,
                liquidity_score              NUMERIC,
                working_capital_efficiency   NUMERIC
            )
    STABLE
    PARALLEL SAFE
    LANGUAGE SQL
AS
$$
SELECT
    -- Identity
    "Ticker"                                                                      AS ticker,
    "Name"                                                                        AS name,
    "Sector"                                                                      AS sector,
    "Industry"                                                                    AS industry,

    -- Revenue Quarterly Features
    "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0)                 AS revenue_fq_vs_5yavg,
    ("Total Revenues (FQ)" * 4 - "Total Revenues (LTM)") /
    NULLIF("Total Revenues (LTM)", 0) * 100                                       AS revenue_qoq_growth,
    "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4, 0)                 AS revenue_seasonality_factor,

    -- Cost Structure Features
    "Selling General & Admin Expenses/Total (FY)" /
    NULLIF("Total Revenues (FY)", 0) * 100                                        AS sga_to_revenue_fy,
    ("Selling General & Admin Expenses/Total (FY)" -
     "Selling General & Admin Expenses/Total (-1FY)") /
    NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0) * 100         AS sga_trend_yoy,
    "Marketing Expenses (FY)" / NULLIF("Total Revenues (FY)", 0) * 100            AS marketing_to_revenue_fy,
    CASE
        WHEN ABS("Total Revenues (-1FY)") > 0 AND ABS("Total Operating Expenses (LTM)") > 0
            THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0)) /
                 NULLIF((("Selling General & Admin Expenses/Total (FY)" -
                          "Selling General & Admin Expenses/Total (-1FY)") /
                         NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0)), 0)
        END                                                                       AS operating_leverage_score,
    ("Selling General & Admin Expenses/Total (-1FY)" / NULLIF("Total Revenues (-1FY)", 0)) -
    ("Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0))
                                                                                  AS cost_efficiency_trend,

    -- Tangible Book Features
    "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0), 0)                 AS price_to_tbv,
    "TBV (LTM)" / NULLIF("Shrs Out", 0)                                           AS tbv_per_share,
    "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0)                                 AS tangible_equity_ratio,
    ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0) * 100                   AS tbv_margin_of_safety,

    -- Interest Income Features
    "Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)"       AS net_interest_income,
    "EBITDA (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                    AS interest_coverage_ebitda,
    "Operating Income (LTM)" /
    NULLIF("Operating Income (LTM)" + "Interest Income On Investments (LTM)", 0)  AS financial_income_quality,
    "Interest Expense/Total (LTM)" / NULLIF("EBIT (LTM)", 0)                      AS interest_burden_ratio,

    -- Long-Term Momentum Features
    ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 100         AS price_momentum_3y,
    ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 100         AS price_momentum_5y,
    (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1) -
    (POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1)            AS momentum_acceleration_3y,
    (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 0.5 +
     ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 0.3 +
     ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 0.2) * 100 AS long_term_trend_score,
    CASE
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
             ("Last Price" > "Price (1Y Ago)") AND ("Last Price" > "Price (3Y Ago)") THEN 1.0
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
             ("Last Price" > "Price (1Y Ago)") THEN 0.75
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") THEN 0.5
        WHEN ("Last Price" > "Price (1M Ago)") THEN 0.25
        ELSE 0
        END                                                                       AS momentum_consistency,
    CASE
        WHEN POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1 > 0.10 THEN 1
        ELSE 0
        END                                                                       AS secular_trend_flag,

    -- Beta Risk Features
    "Beta (2Y)"                                                                   AS beta_2y,
    "Beta (1Y)" - "Beta (2Y)"                                                     AS beta_trend_short,
    "Beta (2Y)" - "Beta (5Y)"                                                     AS beta_trend_long,
    CASE
        WHEN GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
             LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") > 0
            THEN 1.0 / (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
                        LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)"))
        ELSE 1.0
        END                                                                       AS beta_stability,
    CASE WHEN ABS("Beta (1Y)" - "Beta (5Y)") > 0.3 THEN 1 ELSE 0 END              AS beta_regime_change,
    "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2                     AS systematic_risk_score,
    CASE
        WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8 THEN 1
        ELSE 0
        END                                                                       AS defensive_stock_flag,
    CASE WHEN "Beta (1Y)" > 1.3 AND "Beta (2Y)" > 1.3 THEN 1 ELSE 0 END           AS high_beta_flag,

    -- Revenue Estimate Consensus
    ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
    NULLIF("Revenues - Est Med (FY1E)", 0) * 100                                  AS estimate_skew_fy1e,
    GREATEST(0, LEAST(100,
                      100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                NULLIF("Revenues - Est Med (FY1E)", 0) * 100)))   AS revenue_consensus_confidence,
    ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") /
    NULLIF("Total Revenues (LTM)", 0) * 100                                       AS upside_to_consensus,

    -- Unusual Items Features
    ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
    ABS("Restructuring Charges (LTM)") + ABS("Merger & Restructuring Charges (LTM)") +
    ABS("Gain (Loss) On Sale Of Assets (LTM)") +
    ABS("Other Unusual Items/Total (LTM)")                                        AS total_unusual_items,
    (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
     ABS("Restructuring Charges (LTM)") + ABS("Other Unusual Items/Total (LTM)")) /
    NULLIF(ABS("Net Income - (IS) (LTM)"), 0)                                     AS unusual_to_net_income_ratio,
    CASE
        WHEN ABS("Impairment of Goodwill (LTM)") < 1 AND ABS("Asset Writedown (LTM)") < 1 AND
             ABS("Restructuring Charges (LTM)") < 1 AND ABS("Other Unusual Items/Total (LTM)") < 1 THEN 1
        ELSE 0
        END                                                                       AS clean_earnings_flag,
    CASE
        WHEN (CASE WHEN ABS("Impairment of Goodwill (FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Impairment of Goodwill (-1FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Restructuring Charges (FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Restructuring Charges (-1FY)") > 0 THEN 1 ELSE 0 END) >= 3 THEN 1
        ELSE 0
        END                                                                       AS recurring_unusual_flag,
    LEAST(100, (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
                ABS("Restructuring Charges (LTM)") + ABS("Other Unusual Items/Total (LTM)")) /
               NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100)                   AS earnings_noise_score,
    "Net Income - (IS) (LTM)" + "Impairment of Goodwill (LTM)" +
    "Asset Writedown (LTM)" + "Restructuring Charges (LTM)"                       AS quality_adjusted_ni,

    -- Enhanced Valuation Ratios
    ("P/E (LTM)" - "Forward P/E") / NULLIF("P/E (LTM)", 0) * 100                  AS pe_forward_discount,
    CASE
        WHEN "Total Revenues/CAGR (5Y FY)" > 0
            THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0)
        END                                                                       AS peg_adjusted,
    CASE
        WHEN "Revenues - Est YoY % (FY1E)" > 0
            THEN "Forward P/E" / NULLIF("Revenues - Est YoY % (FY1E)", 0)
        END                                                                       AS peg_forward,
    CASE WHEN "P/E (LTM)" > 0 THEN 1.0 / "P/E (LTM)" * 100 END                    AS earnings_yield,
    "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                                   AS fcf_yield,
    COALESCE("Div Yield (LTM)", 0) + COALESCE("Buyback Yield (LTM)", 0)           AS shareholder_yield_total,
    GREATEST(0, LEAST(100,
                      50 -
                      (CASE WHEN "P/E (LTM)" < 15 THEN 10 WHEN "P/E (LTM)" < 25 THEN 0 ELSE -10 END) -
                      (CASE WHEN "PEG Ratio" < 1 THEN 15 WHEN "PEG Ratio" < 2 THEN 5 ELSE -5 END) -
                      (CASE
                           WHEN "EV/EBITDA (LTM)" < 10 THEN 10
                           WHEN "EV/EBITDA (LTM)" < 15 THEN 0
                           ELSE -10 END) +
                      (CASE
                           WHEN "FCF (LTM)" / NULLIF("Market Cap", 0) * 100 > 5 THEN 15
                           ELSE 0 END)))                                          AS valuation_composite_score,

    -- Working Capital Deep Features
    ("Total Current Assets (LTM)" / NULLIF("Total Operating Expenses (LTM)" / 365, 0))
                                                                                  AS defensive_interval,
    "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turnover,
    GREATEST(0, LEAST(100,
                      (CASE
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 2
                               THEN 40
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1.5
                               THEN 30
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1
                               THEN 20
                           ELSE 0 END) +
                      (CASE
                           WHEN ("Total Current Assets (LTM)" - "Inventory (LTM)") /
                                NULLIF("Total Current Liabilities (LTM)", 0) >= 1 THEN 30
                           ELSE 15 END) +
                      (CASE
                           WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.5
                               THEN 30
                           WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.2
                               THEN 15
                           ELSE 0 END)))                                          AS liquidity_score,
    "Total Revenues (LTM)" / NULLIF(ABS("Working Capital (LTM)"), 0)              AS working_capital_efficiency

FROM postgres.public.equities;
$$;

-- ... existing code ...
