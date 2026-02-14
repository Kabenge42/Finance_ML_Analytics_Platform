-- =============================================================================
-- UNIFIED MATERIALIZED VIEW AND FEATURE REGISTRY
-- Integrates all feature calculation functions from feature_registry.sql
-- Aligned with all 18 vw_features_* views for complete coverage (700+ columns)
-- =============================================================================

-- Pre-flight check: ensure prerequisite functions exist
DO
$$
    BEGIN
        -- Verify critical helper functions exist
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'safe_divide') THEN
            RAISE EXCEPTION 'Required function safe_divide() does not exist. Run feature_registry.sql first.';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'calc_piotroski_f_score') THEN
            RAISE EXCEPTION 'Required function calc_piotroski_f_score() does not exist. Run feature_registry.sql first.';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'pct_change') THEN
            RAISE EXCEPTION 'Required function pct_change() does not exist. Run feature_registry.sql first.';
        END IF;
    END
$$;

-- =============================================================================
-- SECTION 1: UNIFIED MATERIALIZED VIEW - ALL FEATURES
-- =============================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_all_stock_features CASCADE;

CREATE MATERIALIZED VIEW mv_all_stock_features AS
SELECT
    -- =========================================================================
    -- [1/18] IDENTIFIER COLUMNS (vw_identifier_columns)
    -- =========================================================================
    e."ISIN"                                                                                           AS isin,
    e."Ticker"                                                                                         AS ticker,
    e."Name"                                                                                           AS name,
    e."Region"                                                                                         AS region,
    e."Country"                                                                                        AS country,
    e."Trading Country"                                                                                AS trading_country,
    e."Exchange"                                                                                       AS exchange,
    e."Sector"                                                                                         AS sector,
    e."Industry"                                                                                       AS industry,
    e."Next Earnings"               AS next_earnings,
    e."Next Earnings (When)"        AS next_earnings_when,
    e."Next Earnings (Status)"      AS next_earnings_status,
    e."Current Fiscal Quarter"      AS current_fiscal_quarter,
    e."Next Fiscal Quarter"                                                                            AS next_fiscal_quarter,
    e."Next Earnings (Report)"      AS next_earnings_report,
    e."Earnings Report (Frequency)" AS earnings_report_frequency,
    e."Income Statement Report Date"                                                                   AS income_statement_report_date,
    e."Next Income Statement Report Date"                                                              AS next_income_statement_report_date,
    e."Dividend Record (Currency)"  AS dividend_record_currency,
    e."Dividend Record (Amount)"    AS dividend_record_amount,
    e."Dividend Record (Frequency)" AS dividend_record_frequency,
    e."Market Cap"                                                                                     AS market_cap,
    e."Enterprise Value"                                                                               AS enterprise_value,
    e."Last Price"                                                                                     AS last_price,
    e."Price Target"                                                                                   AS price_target,

    -- =========================================================================
    -- [2/18] VALUATION FEATURES (vw_features_valuation_ratios)
    -- calc_valuation_features
    -- =========================================================================
    e."P/E (LTM)"                                                                                      AS p_e_ratio,
    e."P/B (LTM)"                                                                                      AS p_b_ratio,
    e."EV/EBITDA (LTM)"                                                                                AS ev_ebitda_ratio,
    e."EV/Sales (LTM)"                                                                                 AS ev_sales_ratio,
    e."Div Yield (LTM)"                                                                                AS dividend_yield,
    CASE
        WHEN e."Net EPS - Basic (FY)" > 0 AND e."Net EPS - Basic (-3FY)" > 0
            THEN safe_divide(
                e."P/E (LTM)"::NUMERIC,
                ((POWER(safe_divide(e."Net EPS - Basic (FY)"::NUMERIC, e."Net EPS - Basic (-3FY)"::NUMERIC),
                        (1.0 / 3.0)::NUMERIC) - 1) * 100)::NUMERIC
                 )
        END                                                                                            AS peg_ratio,

    -- calc_valuation_timeseries_features
    calc_change_ratio(e."EV/Sales (LTM)", e."EV/Sales (-1FYLTM)")                                      AS ev_sales_trend_1y,
    calc_change_ratio(e."EV/EBITDA (LTM)", e."EV/EBITDA (-1FYLTM)")                                    AS ev_ebitda_momentum,
    calc_change_ratio(e."P/E (LTM)", e."P/E (-1FYLTM)")                                                AS p_e_momentum_yoy,
    calc_change_ratio(e."P/E (LTM)", e."P/E (-1FQLTM)")                                                AS p_e_momentum_qoq,
    calc_change_ratio(e."EV/Sales (LTM)", e."EV/Sales (3YAVGLTM)")                                     AS ev_sales_vs_3y_avg,
    calc_change_ratio(e."EV/EBITDA (LTM)", e."EV/EBITDA (3YAVGLTM)")                                   AS ev_ebitda_vs_3y_avg,
    calc_change_ratio(e."P/E (LTM)", e."P/E (3YAVGLTM)")                                               AS p_e_vs_3y_avg,
    calc_change_ratio(e."EV/Sales (NTM)", e."EV/Sales (LTM)")                                          AS ev_sales_forward_discount,
    calc_change_ratio(e."EV/EBITDA (NTM)", e."EV/EBITDA (LTM)")                                        AS ev_ebitda_forward_discount,
    calc_change_ratio(e."P/E (EST FY1)", e."P/E (LTM)")                                                AS p_e_forward_discount,
    safe_divide(e."P/B (LTM)", e."P/B (5YAVG)")                                                        AS p_b_vs_5y_avg,

    -- calc_extended_valuation_timeseries
    calc_change_ratio(e."EV/Sales (LTM)", e."EV/Sales (-1FQLTM)")                                      AS ev_sales_qoq_1q,
    calc_change_ratio(e."EV/Sales (-1FQLTM)", e."EV/Sales (-2FQLTM)")                                  AS ev_sales_qoq_2q,
    calc_change_ratio(e."EV/Sales (-2FQLTM)", e."EV/Sales (-3FQLTM)")                                  AS ev_sales_qoq_3q,
    calc_change_ratio(e."EV/Sales (-3FQLTM)", e."EV/Sales (-4FQLTM)")                                  AS ev_sales_qoq_4q,
    calc_change_ratio(e."P/E (LTM)", e."P/E (5YAVGLTM)")                                               AS p_e_vs_5y_avg_ext,
    CASE
        WHEN e."P/E (LTM)" IS NOT NULL AND e."P/E (3YAVGLTM)" IS NOT NULL
            THEN (e."P/E (LTM)" - e."P/E (3YAVGLTM)") / NULLIF(ABS(e."P/E (3YAVGLTM)") * 0.5, 0)
        END                                                                                            AS p_e_percentile_proxy,
    (calc_change_ratio(e."P/E (LTM)", e."P/E (3YAVGLTM)") +
     calc_change_ratio(e."EV/Sales (LTM)", e."EV/Sales (3YAVGLTM)") +
     calc_change_ratio(e."EV/EBITDA (LTM)", e."EV/EBITDA (3YAVGLTM)")) /
    3.0                                                                                                AS valuation_mean_reversion,
    calc_change_ratio(e."EV/EBITDA (LTM)", e."EV/EBITDA (-1FQLTM)")                                    AS ev_ebitda_qoq_trend,
    calc_change_ratio(e."P/B (LTM)", e."P/B (-1FY)")                                                   AS p_b_momentum_yoy,
    (safe_divide(e."P/E (LTM)", e."P/E (3YAVGLTM)") +
     safe_divide(e."EV/EBITDA (LTM)", e."EV/EBITDA (3YAVGLTM)")) / 2.0 -
    1.0                                                                                                AS valuation_compression,
    calc_change_ratio(e."P/E (EST FY1)", e."P/E (LTM)") * 100                                          AS forward_pe_premium,

    -- Tangible book features (part of valuation ratios view)
    e."Total Equity (LTM)" - COALESCE(e."Goodwill (LTM)", 0) -
    COALESCE(e."Gross Intangible Assets (LTM)", 0)                                                     AS tangible_book_value,
    (e."Total Equity (LTM)" - COALESCE(e."Goodwill (LTM)", 0) - COALESCE(e."Gross Intangible Assets (LTM)", 0)) /
    NULLIF(e."Shrs Out", 0)                                                                            AS tangible_book_per_share,
    e."Last Price" * e."Shrs Out" /
    NULLIF(e."Total Equity (LTM)" - COALESCE(e."Goodwill (LTM)", 0) - COALESCE(e."Gross Intangible Assets (LTM)", 0),
           0)                                                                                          AS price_to_tangible_book,
    (e."Total Equity (LTM)" - COALESCE(e."Goodwill (LTM)", 0) - COALESCE(e."Gross Intangible Assets (LTM)", 0)) /
    NULLIF(e."Total Assets (LTM)", 0) *
    100                                                                                                AS tangible_equity_ratio,
    COALESCE(e."Gross Intangible Assets (LTM)", 0) / NULLIF(e."Total Equity (LTM)", 0) *
    100                                                                                                AS intangibles_to_equity,
    COALESCE(e."Goodwill (LTM)", 0) / NULLIF(e."Total Equity (LTM)", 0) *
    100                                                                                                AS goodwill_to_equity,

    -- =========================================================================
    -- [3/18] MOMENTUM FEATURES (vw_features_momentum)
    -- calc_momentum_features
    -- =========================================================================
    pct_change(e."Last Price", e."Price (1M Ago)")                                                     AS price_momentum_1m,
    pct_change(e."Last Price", e."Price (3M Ago)")                                                     AS price_momentum_3m,
    pct_change(e."Last Price", e."Price (6M Ago)")                                                     AS price_momentum_6m,
    pct_change(e."Last Price", e."Price (1Y Ago)")                                                     AS price_momentum_1y,
    pct_change(e."Last Price", e."Price (5D Ago)")                                                     AS price_momentum_5d,
    ema_crossover_signal(e."EMA (20D)", e."EMA (50D)")                                                 AS ema_crossover_20_50,
    ema_crossover_signal(e."EMA (50D)", e."EMA (250D)")                                                AS ema_crossover_50_250,
    calc_change_ratio(e."Last Price", e."EMA (20D)")                                                   AS price_vs_ema_20d,
    calc_change_ratio(e."Last Price", e."EMA (250D)")                                                  AS price_vs_ema_250d,
    calc_change_ratio(e."52W High/Adj" - e."Last Price",
                      e."52W High/Adj")                                                                AS pct_off_52w_high,
    calc_change_ratio(e."Last Price" - e."52W Low/Adj", e."52W Low/Adj")                               AS pct_above_52w_low,
    clamp_score(safe_divide(e."Last Price" - e."52W Low/Adj", e."52W High/Adj" - e."52W Low/Adj"), 0,
                1)                                                                                     AS range_52w_position,
    e."Beta (1Y)" - e."Beta (5Y)"                                                                      AS beta_momentum,
    safe_divide(e."Volatility (1M)", e."Volatility (1Y)")                                              AS volatility_regime,

    -- calc_long_term_momentum_features
    pct_change(e."Last Price", e."Price (3Y Ago)")                                                     AS price_momentum_3y,
    pct_change(e."Last Price", e."Price (5Y Ago)")                                                     AS price_momentum_5y,
    (COALESCE(pct_change(e."Last Price", e."Price (1Y Ago)"), 0) * 0.50 +
     COALESCE(pct_change(e."Last Price", e."Price (3Y Ago)"), 0) * 0.30 +
     COALESCE(pct_change(e."Last Price", e."Price (5Y Ago)"), 0) * 0.20) /
    100                                                                                                AS long_term_trend_score,
    CASE
        WHEN calc_change_ratio(e."52W High/Adj" - e."Last Price", e."52W High/Adj") <= 0.10
            AND calc_change_ratio(e."Last Price", e."Price (3Y Ago)") > 0.5 THEN 1
        ELSE 0
        END                                                                                            AS multi_year_high_flag,
    CASE
        WHEN calc_change_ratio(e."Last Price", e."Price (3Y Ago)") > 0.20
            AND calc_change_ratio(e."Last Price", e."Price (1Y Ago)") > 0
            AND e."EMA (50D)" > e."EMA (250D)" THEN 1
        ELSE 0
        END                                                                                            AS secular_trend_flag,

    -- =========================================================================
    -- [4/18] TECHNICAL ANALYSIS FEATURES (vw_features_technical_analysis)
    -- calc_technical_analysis_features
    -- =========================================================================
    (e."EMA (20D)" - e."EMA (50D)") / NULLIF(e."EMA (50D)", 0)                                         AS ema_slope_20d,
    CASE
        WHEN e."EMA (20D)" > e."EMA (50D)" AND e."EMA (50D)" > e."EMA (100D)" AND e."EMA (100D)" > e."EMA (250D)" THEN 1
        WHEN e."EMA (20D)" < e."EMA (50D)" AND e."EMA (50D)" < e."EMA (100D)" AND e."EMA (100D)" < e."EMA (250D)" THEN -1
        ELSE 0
        END                                                                                            AS ema_trend_consistency,
    (e."Last Price" - e."EMA (100D)") / NULLIF(e."EMA (100D)", 0) * 100                                AS price_vs_ema_100d,
    CASE
        WHEN (e."52W High/Adj" - e."Last Price") / NULLIF(e."52W High/Adj", 0) <= 0.05 THEN 1
        ELSE 0 END                                                                                     AS near_52w_high_flag,
    CASE
        WHEN (e."Last Price" - e."52W Low/Adj") / NULLIF(e."52W Low/Adj", 0) <= 0.05 THEN 1
        ELSE 0 END                                                                                     AS near_52w_low_flag,
    e."Rel. Volume" * e."Price Chg. % (1M)"                                                            AS volume_momentum_score,
    CASE
        WHEN e."EMA (20D)" > e."EMA (50D)" AND (e."52W High/Adj" - e."Last Price") / NULLIF(e."52W High/Adj", 0) <= 0.05
            THEN 1
        ELSE 0
        END                                                                                            AS breakout_signal,
    CASE WHEN e."Rel. Volume" > 1.5 THEN 1 ELSE 0 END                                                  AS high_volume_flag,
    CASE WHEN e."Rel. Volume" < 0.5 THEN 1 ELSE 0 END                                                  AS low_volume_flag,
    e."Volatility (1Y)" - e."Volatility (1M)"                                                          AS volatility_compression,
    e."Volatility (3M)" - e."Volatility (6M)"                                                          AS volatility_term_structure,

    -- =========================================================================
    -- [5/18] PROFITABILITY FEATURES (vw_features_profitability)
    -- calc_profitability_features
    -- =========================================================================
    e."Return On Equity % (LTM)"                                                                       AS roe,
    e."Return on Assets (ROA) % (LTM)"                                                                 AS roa,
    e."Gross Profit Margin % (LTM)"                                                                    AS gross_margin_pct,
    e."Operating Income (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                             AS operating_margin_pct,
    e."Net Income Margin % (LTM)"                                                                      AS net_margin_pct,
    e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                       AS ebitda_margin_pct,
    e."EBIT (LTM)" * (1 - 0.25) /
    NULLIF(e."Total Equity (LTM)" + e."Total Debt (LTM)" - e."Cash And Equivalents (LTM)", 0) *
    100                                                                                                AS roic,
    e."R&D Expenses (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)                                       AS rnd_intensity,
    e."Total Assets (LTM)" / NULLIF(e."Total Equity (LTM)", 0)                                         AS equity_multiplier,

    -- calc_margin_trends
    (e."Gross Profit Margin % (LTM)" - e."Gross Profit Margin % (FY)")                                 AS gross_margin_trend_yoy,
    ((e."Operating Income (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)) -
     (e."Operating Income (FY)" / NULLIF(e."Total Revenues (FY)", 0))) *
    100                                                                                                AS operating_margin_trend,
    (e."Net Income Margin % (LTM)" - e."Net Income Margin % (FY)")                                     AS net_margin_trend_yoy,
    ((e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)) -
     (e."EBITDA (FY)" / NULLIF(e."Total Revenues (FY)", 0))) *
    100                                                                                                AS ebitda_margin_trend,
    CASE
        WHEN e."Gross Profit Margin % (LTM)" > e."Gross Profit Margin % (FY)"
            AND e."Net Income Margin % (LTM)" > e."Net Income Margin % (FY)"
            AND (e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)) >
                (e."EBITDA (FY)" / NULLIF(e."Total Revenues (FY)", 0))
            THEN 1
        ELSE 0
        END                                                                                            AS margin_expansion_flag,
    -- *** PREVIOUSLY MISSING: margin_stability_score ***
    GREATEST(0, LEAST(100,
                      100 - (ABS(e."Gross Profit Margin % (LTM)" - e."Gross Profit Margin % (FY)") +
                             ABS(e."Net Income Margin % (LTM)" - e."Net Income Margin % (FY)")) / 2
                ))                                                                                     AS margin_stability_score,

    -- =========================================================================
    -- [6/18] QUALITY & RISK FEATURES (vw_features_quality_risk)
    -- calc_quality_features
    -- =========================================================================
    CASE WHEN e."Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END                                  AS has_goodwill_impairment,
    CASE WHEN e."Asset Writedown (LTM)" <> 0 THEN 1 ELSE 0 END                                         AS has_asset_writedown,
    CASE WHEN e."Restructuring Charges (LTM)" <> 0 THEN 1 ELSE 0 END                                   AS has_restructuring,
    e."Goodwill (LTM)" / NULLIF(e."Total Assets (LTM)", 0) * 100                                       AS goodwill_to_assets_pct,
    e."Gross Intangible Assets (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                              AS intangible_intensity,
    (ABS(e."Impairment of Goodwill (LTM)") + ABS(e."Asset Writedown (LTM)") + ABS(e."Restructuring Charges (LTM)")) /
    NULLIF(ABS(e."EBITDA (LTM)"), 0)                                                                   AS exceptional_items_to_ebitda,
    e."Altman Z-Score (LTM)"                                                                           AS altman_z_score,
    e."Altman Z-Score (FY)" - e."Altman Z-Score (LTM)"                                                 AS altman_z_trend,
    e."Current Ratio (LTM)"                                                                            AS current_ratio,
    (e."Total Current Assets (LTM)" - e."Inventory (LTM)") /
    NULLIF(e."Total Current Liabilities (LTM)", 0)                                                     AS quick_ratio,

    -- calc_financial_distress_features
    GREATEST(0, LEAST(100,
                      ((e."Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100)))                AS distress_risk_score,
    CASE
        WHEN e."Current Ratio (LTM)" < 1.0 THEN 30.0
        WHEN e."Current Ratio (LTM)" < 1.5 THEN 15.0
        ELSE 0.0
        END                                                                                            AS liquidity_stress_score,
    (e."Working Capital (FQ)" - e."Working Capital (FY)") /
    NULLIF(ABS(e."Working Capital (FY)"), 0)                                                           AS working_capital_trend,
    e."Cash And Equivalents (FQ)" /
    NULLIF(e."Total Operating Expenses (LTM)" / 12.0, 0)                                               AS cash_runway_months,
    -- *** PREVIOUSLY MISSING: combined_distress_score ***
    GREATEST(0, LEAST(100,
                      ((e."Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 70) +
                      (100 - CASE
                                 WHEN e."Current Ratio (LTM)" < 1.0 THEN 30.0
                                 WHEN e."Current Ratio (LTM)" < 1.5 THEN 15.0
                                 ELSE 0.0
                          END) *
                      0.30))                                                                           AS combined_distress_score,
    -- *** PREVIOUSLY MISSING: wc_deteriorating_flag ***
    CASE
        WHEN (e."Working Capital (FQ)" - e."Working Capital (FY)") /
             NULLIF(ABS(e."Working Capital (FY)"), 0) < -0.2
            THEN 1
        ELSE 0
        END                                                                                            AS wc_deteriorating_flag,
    -- *** PREVIOUSLY MISSING: retained_earnings_growth ***
    (e."Retained Earnings (FQ)" - e."Retained Earnings (FY)") /
    NULLIF(ABS(e."Retained Earnings (FY)"), 0)                                                         AS retained_earnings_growth,
    CASE WHEN e."Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END                                         AS accumulated_deficit_flag,
    CASE
        WHEN e."Cash And Equivalents (FQ)" / NULLIF(e."Total Operating Expenses (LTM)" / 12.0, 0) > 6 THEN 1
        ELSE 0
        END                                                                                            AS adequate_cash_buffer,

    -- calc_accounting_quality_features
    (e."Goodwill (LTM)" - e."Goodwill (-1FY)") /
    NULLIF(e."Goodwill (-1FY)", 0)                                                                     AS goodwill_change_rate,
    e."Restructuring Charges (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                AS restructuring_intensity,
    (CASE WHEN ABS(e."Impairment of Goodwill (FQ)") > 0 THEN 1 ELSE 0 END +
     CASE WHEN ABS(e."Asset Writedown (FQ)") > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN ABS(e."Restructuring Charges (FQ)") > 0 THEN 1
         ELSE 0 END)                                                                                   AS exceptional_items_frequency,
    e."Merger & Restructuring Charges (LTM)" / NULLIF(e."Market Cap", 0)                               AS merger_impact_ratio,
    e."Interest Income On Investments (LTM)" /
    NULLIF(ABS(e."Net Income - (IS) (LTM)"), 0)                                                        AS non_operating_income_share,
    CASE WHEN e."Gain (Loss) On Sale Of Assets (LTM)" > 0 THEN 1 ELSE 0 END                            AS asset_sale_boost,
    GREATEST(0, LEAST(100,
                      100 -
                      (CASE WHEN e."Impairment of Goodwill (LTM)" <> 0 THEN 25 ELSE 0 END) -
                      (CASE WHEN e."Asset Writedown (LTM)" <> 0 THEN 10 ELSE 0 END) -
                      (CASE WHEN e."Restructuring Charges (LTM)" <> 0 THEN 15 ELSE 0 END) -
                      (CASE WHEN e."Goodwill (LTM)" / NULLIF(e."Total Assets (LTM)", 0) > 0.30 THEN 15 ELSE 0 END) -
                      (CASE
                           WHEN (ABS(e."Impairment of Goodwill (LTM)") + ABS(e."Asset Writedown (LTM)") +
                                 ABS(e."Restructuring Charges (LTM)")) /
                                NULLIF(ABS(e."Net Income - (IS) (LTM)"), 0) > 0.10 THEN 15
                           ELSE 0 END)
                ))                                                                                     AS accounting_quality_score,

    -- calc_beta_risk_features
    e."Beta (1Y)"                                                                                      AS beta_1y,
    e."Beta (5Y)"                                                                                      AS beta_5y,
    e."Beta (1Y)" - e."Beta (5Y)"                                                                      AS beta_spread,
    (e."Beta (1Y)" - e."Beta (5Y)") / NULLIF(ABS(e."Beta (5Y)"), 0) * 100                              AS beta_trend,
    CASE WHEN e."Beta (1Y)" > 1.5 THEN 1 ELSE 0 END                                                    AS high_beta_flag,
    CASE WHEN e."Beta (1Y)" < 0.5 THEN 1 ELSE 0 END                                                    AS low_beta_flag,
    -- *** PREVIOUSLY MISSING: beta_stability_score ***
    GREATEST(0,
             LEAST(100, 100 - ABS(e."Beta (1Y)" - e."Beta (5Y)") * 50))                                AS beta_stability_score,

    -- =========================================================================
    -- [7/18] LEVERAGE & LIQUIDITY FEATURES (vw_features_leverage_liquidity)
    -- calc_leverage_features
    -- =========================================================================
    e."Total Debt (LTM)" / NULLIF(e."Total Equity (LTM)", 0)                                           AS debt_to_equity,
    e."Total Debt (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                           AS debt_to_assets,
    e."Total Equity (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                         AS equity_ratio,
    e."EBIT (LTM)" / NULLIF(e."Interest Expense/Total (LTM)", 0)                                       AS interest_coverage,
    e."Cash And Equivalents (LTM)" /
    NULLIF(e."Total Current Liabilities (LTM)", 0)                                                     AS cash_ratio,
    e."Working Capital (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                      AS working_capital_ratio,

    -- calc_efficiency_ratios
    e."Total Revenues (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                       AS asset_turnover,
    e."Cost Of Revenues (LTM)" / NULLIF(e."Inventory (LTM)", 0)                                        AS inventory_turnover,
    (e."Accounts Receivable/Total (FY)" /
     NULLIF(e."Total Revenues (FY)" / 365.0, 0))                                                       AS receivables_days,
    e."Total Revenues (LTM)" / NULLIF(e."Working Capital (LTM)", 0)                                    AS working_capital_turns,

    -- =========================================================================
    -- [8/18] BALANCE SHEET FEATURES (vw_features_balance_sheet)
    -- calc_balance_sheet_dynamics
    -- =========================================================================
    e."Cash And Equivalents (LTM)" / NULLIF(e."Total Assets (LTM)", 0) *
    100                                                                                                AS cash_to_assets_pct,
    (e."Cash And Equivalents (FQ)" - e."Cash And Equivalents (FY)") /
    NULLIF(ABS(e."Cash And Equivalents (FY)"), 0)                                                      AS cash_change_qoq,
    e."Cash And Equivalents (FQ)" /
    NULLIF(e."Cash And Equivalents (5YAVGFQ)", 0)                                                      AS cash_vs_5y_avg,
    (e."Inventory (FY)" - e."Inventory (FQ)") /
    NULLIF(ABS(e."Inventory (FQ)"), 0)                                                                 AS inventory_change_yoy,
    e."Inventory (FQ)" / NULLIF(e."Inventory (5YAVGFQ)", 0)                                            AS inventory_vs_5y_avg,
    -- *** PREVIOUSLY MISSING: receivables_change_yoy ***
    (e."Accounts Receivable/Total (FY)" - e."Accounts Receivable/Total (-1FY)") /
    NULLIF(ABS(e."Accounts Receivable/Total (-1FY)"), 0)                                               AS receivables_change_yoy,
    -- *** PREVIOUSLY MISSING: receivables_vs_5y_avg ***
    e."Accounts Receivable/Total (FY)" /
    NULLIF(e."Accounts Receivable/Total (5YAVGFQ)", 0)                                                 AS receivables_vs_5y_avg,
    e."Working Capital (FQ)" / NULLIF(e."Working Capital (5YAVGFY)", 0)                                AS working_capital_vs_5y_avg,
    e."Retained Earnings (FQ)" /
    NULLIF(e."Retained Earnings (5YAVGFQ)", 0)                                                         AS retained_earnings_vs_5y,
    CASE
        WHEN e."Gross Intangible Assets (FY)" / NULLIF(e."Gross Intangible Assets (5YAVGFQ)", 0) > 1.5 THEN 1
        ELSE 0 END                                                                                     AS intangibles_growth_flag,
    GREATEST(0, LEAST(100,
                      50 + (e."Cash And Equivalents (LTM)" / NULLIF(e."Total Assets (LTM)", 0) * 100) -
                      (e."Goodwill (LTM)" / NULLIF(e."Total Assets (LTM)", 0) * 100)
                ))                                                                                     AS asset_quality_score,
    GREATEST(0, LEAST(100,
                      (CASE
                           WHEN e."Cash And Equivalents (LTM)" / NULLIF(e."Total Assets (LTM)", 0) > 0.10 THEN 25
                           ELSE 0 END) +
                      (CASE WHEN e."Total Equity (LTM)" / NULLIF(e."Total Assets (LTM)", 0) > 0.40 THEN 25 ELSE 0 END) +
                      (CASE WHEN e."Working Capital (LTM)" > 0 THEN 25 ELSE 0 END) +
                      (CASE WHEN e."Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END)
                ))                                                                                     AS balance_sheet_strength,
    e."Total Debt (LTM)" / NULLIF(e."EBITDA (LTM)", 0)                                                 AS debt_maturity_risk,

    -- Working capital deep features
    e."Working Capital (LTM)"                                                                          AS working_capital_ltm,
    e."Working Capital (FQ)"                                                                           AS working_capital_fq,
    e."Working Capital (FY)"                                                                           AS working_capital_fy,
    e."Working Capital (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                              AS wc_to_revenue,
    e."Working Capital (LTM)" / NULLIF(e."Total Assets (LTM)", 0) * 100                                AS wc_to_assets,
    (e."Working Capital (FQ)" - e."Working Capital (FY)") / NULLIF(ABS(e."Working Capital (FY)"), 0) *
    100                                                                                                AS wc_change_qoq,
    (e."Working Capital (FY)" - e."Working Capital (-1FY)") / NULLIF(ABS(e."Working Capital (-1FY)"), 0) *
    100                                                                                                AS wc_change_yoy,
    e."Working Capital (LTM)" /
    NULLIF(e."Total Revenues (LTM)" / 365.0, 0)                                                        AS days_working_capital,
    CASE WHEN e."Working Capital (LTM)" < 0 THEN 1 ELSE 0 END                                          AS negative_wc_flag,
    CASE
        WHEN e."Working Capital (FQ)" > e."Working Capital (FY)" AND
             e."Working Capital (FY)" > e."Working Capital (-1FY)" THEN 1
        ELSE 0 END                                                                                     AS wc_improvement_flag,

    -- Working capital temporal
    e."Working Capital (5YAVGFY)"                                                                      AS wc_5yavgfy,
    e."Working Capital (-1FQ)"                                                                         AS wc_1fq,
    e."Working Capital (-2FQ)"                                                                         AS wc_2fq,
    e."Working Capital (-3FQ)"                                                                         AS wc_3fq,
    e."Working Capital (-4FQ)"                                                                         AS wc_4fq,
    e."Working Capital (-1FY)"                                                                         AS wc_1fy,
    e."Working Capital (-2FY)"                                                                         AS wc_2fy,
    e."Working Capital (-3FY)"                                                                         AS wc_3fy,
    e."Working Capital (-4FY)"                                                                         AS wc_4fy,
    pct_change(e."Working Capital (FQ)", e."Working Capital (-1FQ)")                                   AS wc_qoq_change,
    pct_change(e."Working Capital (FY)", e."Working Capital (-1FY)")                                   AS wc_yoy_change,
    pct_change(e."Working Capital (FQ)", e."Working Capital (-4FQ)")                                   AS wc_4q_trend,
    safe_divide(e."Working Capital (FQ)", e."Working Capital (5YAVGFY)")                               AS wc_vs_5y_avg_temp,

    -- Total assets temporal
    e."Total Assets (FQ)"                                                                              AS total_assets_fq,
    e."Total Assets (LTM)"                                                                             AS total_assets_ltm,
    e."Total Assets (FY)"                                                                              AS total_assets_fy,
    e."Total Assets (5YAVGFQ)"                                                                         AS total_assets_5yavgfq,
    pct_change(e."Total Assets (FY)", e."Total Assets (-1FY)")                                         AS total_assets_yoy_growth,
    safe_divide(e."Total Assets (FQ)", e."Total Assets (5YAVGFQ)")                                     AS total_assets_vs_5y_avg,

    -- Total debt temporal
    e."Total Debt (FQ)"                                                                                AS total_debt_fq,
    e."Total Debt (LTM)"                                                                               AS total_debt_ltm,
    e."Total Debt (FY)"                                                                                AS total_debt_fy,
    e."Total Debt (5YAVGFQ)"                                                                           AS total_debt_5yavgfq,
    pct_change(e."Total Debt (FY)", e."Total Debt (-1FY)")                                             AS total_debt_yoy_growth,
    safe_divide(e."Total Debt (FQ)", e."Total Debt (5YAVGFQ)")                                         AS total_debt_vs_5y_avg,

    -- Inventory temporal
    e."Inventory (FQ)"                                                                                 AS inventory_fq,
    e."Inventory (FY)"                                                                                 AS inventory_fy,
    e."Inventory (LTM)"                                                                                AS inventory_ltm,
    e."Inventory (5YAVGFQ)"                                                                            AS inventory_5yavgfq,

    -- =========================================================================
    -- [9/18] ANALYST SENTIMENT FEATURES (vw_features_analyst_sentiment)
    -- calc_sentiment_features
    -- =========================================================================
    CASE
        WHEN (e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
              e."# Strong Sell Ratings") > 0
            THEN (e."# Strong Buys Ratings" + e."# Buys Ratings") /
                 NULLIF(e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
                        e."# Strong Sell Ratings", 0) * 100
        END                                                                                            AS analyst_bullish_pct,
    CASE
        WHEN (e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
              e."# Strong Sell Ratings") > 0
            THEN (e."# Sell Ratings" + e."# Strong Sell Ratings") /
                 NULLIF(e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
                        e."# Strong Sell Ratings", 0) * 100
        END                                                                                            AS analyst_bearish_pct,
    -- *** PREVIOUSLY MISSING: analyst_neutral_pct ***
    CASE
        WHEN (e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
              e."# Strong Sell Ratings") > 0
            THEN e."# Hold Ratings" /
                 NULLIF(e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
                        e."# Strong Sell Ratings", 0) * 100
        END                                                                                            AS analyst_neutral_pct,
    -- *** PREVIOUSLY MISSING: analyst_conviction ***
    ABS(
            CASE
                WHEN (e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" +
                      e."# Sell Ratings" + e."# Strong Sell Ratings") > 0
                    THEN ((e."# Strong Buys Ratings" + e."# Buys Ratings") -
                          (e."# Sell Ratings" + e."# Strong Sell Ratings")) /
                         NULLIF(e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" +
                                e."# Sell Ratings" + e."# Strong Sell Ratings", 0) * 100
                END
    )                                                                                                  AS analyst_conviction,
    (e."Price Target - Median" - e."Last Price") / NULLIF(e."Last Price", 0) *
    100                                                                                                AS upside_potential,
    (e."Price Target - High" - e."Price Target - Low") / NULLIF(e."Price Target - Median", 0) *
    100                                                                                                AS price_target_spread_pct,
    (e."Price Target" - e."Price Target (1M Ago)") /
    NULLIF(e."Price Target (1M Ago)", 0)                                                               AS price_target_revision_1m,
    (e."Price Target" - e."Price Target (3M Ago)") /
    NULLIF(e."Price Target (3M Ago)", 0)                                                               AS price_target_revision_3m,
    COALESCE(e."EPS Est Avg Rev % (FY1E - 1W)", 0) * 0.30 +
    COALESCE(e."EPS Est Avg Rev % (FY1E - 1M)", 0) * 0.25 +
    COALESCE(e."EPS Est Avg Rev % (FY1E - 3M)", 0) * 0.20 +
    COALESCE(e."EPS Est Avg Rev % (FY1E - 6M)", 0) * 0.15 +
    COALESCE(e."EPS Est Avg Rev % (FY1E - 1Y)", 0) *
    0.10                                                                                               AS eps_revision_momentum,
    (e."Analyst Rating" - 1) * 25                                                                      AS analyst_rating_normalized,
    e."Price Target - #" / NULLIF(LN(1 + e."Market Cap"), 0)                                           AS analyst_coverage_quality,

    -- calc_price_target_dynamics
    (e."Price Target" - e."Price Target (1W Ago)") /
    NULLIF(e."Price Target (1W Ago)", 0)                                                               AS pt_momentum_1w,
    (e."Price Target" - e."Price Target (1M Ago)") /
    NULLIF(e."Price Target (1M Ago)", 0)                                                               AS pt_momentum_1m,
    (e."Price Target" - e."Price Target (3M Ago)") /
    NULLIF(e."Price Target (3M Ago)", 0)                                                               AS pt_momentum_3m,
    (e."Price Target" - e."Price Target (6M Ago)") /
    NULLIF(e."Price Target (6M Ago)", 0)                                                               AS pt_momentum_6m,
    (e."Price Target" - e."Price Target (1Y Ago)") /
    NULLIF(e."Price Target (1Y Ago)", 0)                                                               AS pt_momentum_1y,
    -- *** PREVIOUSLY MISSING: pt_median_momentum_1m ***
    (e."Price Target - Median" - e."Price Target - Median (1M Ago)") /
    NULLIF(e."Price Target - Median (1M Ago)", 0)                                                      AS pt_median_momentum_1m,
    -- *** PREVIOUSLY MISSING: pt_median_momentum_3m ***
    (e."Price Target - Median" - e."Price Target - Median (3M Ago)") /
    NULLIF(e."Price Target - Median (3M Ago)", 0)                                                      AS pt_median_momentum_3m,
    -- *** PREVIOUSLY MISSING: pt_acceleration_short ***
    ((e."Price Target" - e."Price Target (1M Ago)") / NULLIF(e."Price Target (1M Ago)", 0)) -
    ((e."Price Target" - e."Price Target (3M Ago)") /
     NULLIF(e."Price Target (3M Ago)", 0))                                                             AS pt_acceleration_short,
    -- *** PREVIOUSLY MISSING: pt_acceleration_long ***
    ((e."Price Target" - e."Price Target (3M Ago)") / NULLIF(e."Price Target (3M Ago)", 0)) -
    ((e."Price Target" - e."Price Target (1Y Ago)") /
     NULLIF(e."Price Target (1Y Ago)", 0))                                                             AS pt_acceleration_long,
    -- *** PREVIOUSLY MISSING: pt_consensus_convergence ***
    ((e."Price Target - High (3M Ago)" - e."Price Target - Low (3M Ago)") /
     NULLIF(e."Price Target - Median (3M Ago)", 0)) -
    ((e."Price Target - High" - e."Price Target - Low") /
     NULLIF(e."Price Target - Median", 0))                                                             AS pt_consensus_convergence,
    (e."Price Target - #" - e."Price Target - # (1M Ago)")::INTEGER                                    AS analyst_coverage_change_1m,
    (e."Price Target - #" - e."Price Target - # (3M Ago)")::INTEGER                                    AS analyst_coverage_change_3m,
    (e."Price Target - #" - e."Price Target - # (1Y Ago)")::INTEGER                                    AS analyst_coverage_change_1y,
    -- *** PREVIOUSLY MISSING: pt_vs_price_momentum ***
    ((e."Price Target" / NULLIF(e."Last Price", 0)) -
     (e."Price Target (3M Ago)" / NULLIF(e."Price (3M Ago)", 0))) /
    NULLIF((e."Price Target (3M Ago)" / NULLIF(e."Price (3M Ago)", 0)),
           0)                                                                                          AS pt_vs_price_momentum,
    -- *** PREVIOUSLY MISSING: analyst_coverage_trend ***
    (COALESCE(e."Price Target - #" - e."Price Target - # (1M Ago)", 0) * 0.40 +
     COALESCE(e."Price Target - #" - e."Price Target - # (3M Ago)", 0) * 0.35 +
     COALESCE(e."Price Target - #" - e."Price Target - # (6M Ago)", 0) * 0.25) /
    NULLIF(e."Price Target - #"::NUMERIC, 0)                                                           AS analyst_coverage_trend,

    -- =========================================================================
    -- [10/18] EARNINGS FEATURES (vw_features_earnings)
    -- calc_earnings_features
    -- =========================================================================
    CASE
        WHEN ABS(e."EPS Norm - Est Avg (FY1E)") > 0
            THEN (e."EPS/Adj. (LTM)" - e."EPS Norm - Est Avg (FY1E)") / NULLIF(ABS(e."EPS Norm - Est Avg (FY1E)"), 0) *
                 100
        END                                                                                            AS eps_surprise_pct,
    CASE
        WHEN ABS(e."Revenues - Est Avg (FY1E)") > 0
            THEN
            (e."Total Revenues (LTM)" - e."Revenues - Est Avg (FY1E)") / NULLIF(ABS(e."Revenues - Est Avg (FY1E)"), 0) *
            100
        END                                                                                            AS revenue_surprise_pct,
    e."EPS/Adj. (LTM)" / NULLIF(e."Net EPS - Basic (LTM)", 0)                                          AS eps_adjustment_ratio,
    CASE
        WHEN ABS(e."EPS Norm - Est Avg (FY1E)") > 0
            THEN (e."EPS GAAP - Est Avg (FY1E)" - e."EPS Norm - Est Avg (FY1E)") /
                 NULLIF(ABS(e."EPS Norm - Est Avg (FY1E)"), 0) * 100
        END                                                                                            AS gaap_adj_eps_gap_pct,
    e."EBITDA/Adj. (LTM)" / NULLIF(e."EBITDA (LTM)", 0)                                                AS ebitda_adjustment_ratio,
    CASE
        WHEN ABS(e."Net EPS - Basic (-4FQFQ)") > 0
            THEN (e."Net EPS - Basic (FQ)" - e."Net EPS - Basic (-4FQFQ)") /
                 NULLIF(ABS(e."Net EPS - Basic (-4FQFQ)"), 0)
        END                                                                                            AS eps_quarterly_trend,
    CASE
        WHEN ABS(e."Net EPS - Basic (-1FY)") > 0
            THEN (e."Net EPS - Basic (FY)" - e."Net EPS - Basic (-1FY)") / NULLIF(ABS(e."Net EPS - Basic (-1FY)"), 0) *
                 100
        END                                                                                            AS eps_yoy_growth,

    -- calc_eps_trajectory_features
    CASE
        WHEN ABS(e."Net EPS - Basic (-1FQFQ)") > 0
            THEN
            (e."Net EPS - Basic (FQ)" - e."Net EPS - Basic (-1FQFQ)") / NULLIF(ABS(e."Net EPS - Basic (-1FQFQ)"), 0) *
            100
        END                                                                                            AS eps_qoq_growth,
    CASE
        WHEN ABS(e."Net EPS - Basic (-4FQFQ)") > 0
            THEN
            (e."Net EPS - Basic (FQ)" - e."Net EPS - Basic (-4FQFQ)") / NULLIF(ABS(e."Net EPS - Basic (-4FQFQ)"), 0) *
            100
        END                                                                                            AS eps_yoy_quarterly,
    (CASE WHEN e."Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."Net EPS - Basic (-4FQFQ)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS eps_positive_streak,
    CASE
        WHEN e."Net EPS - Basic (-3FY)" > 0 AND e."Net EPS - Basic (FY)" > 0
            THEN (POWER(e."Net EPS - Basic (FY)" / NULLIF(e."Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
        END                                                                                            AS eps_cagr_3y,
    CASE
        WHEN e."Net EPS - Basic (-5FY)" > 0 AND e."Net EPS - Basic (FY)" > 0
            THEN (POWER(e."Net EPS - Basic (FY)" / NULLIF(e."Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
        END                                                                                            AS eps_cagr_5y,
    -- *** PREVIOUSLY MISSING: eps_growth_accel ***
    CASE
        WHEN e."Net EPS - Basic (-3FY)" > 0 AND e."Net EPS - Basic (-5FY)" > 0
            AND e."Net EPS - Basic (FY)" > 0
            THEN ((POWER(e."Net EPS - Basic (FY)" / NULLIF(e."Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                  (POWER(e."Net EPS - Basic (FY)" / NULLIF(e."Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
        END                                                                                            AS eps_growth_accel,
    -- *** PREVIOUSLY MISSING: eps_vs_5y_avg ***
    CASE
        WHEN ABS((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" + e."Net EPS - Basic (-2FY)" +
                  e."Net EPS - Basic (-3FY)" + e."Net EPS - Basic (-4FY)") / 5.0) > 0
            THEN (e."Net EPS - Basic (FY)" -
                  ((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" + e."Net EPS - Basic (-2FY)" +
                    e."Net EPS - Basic (-3FY)" + e."Net EPS - Basic (-4FY)") / 5.0)) /
                 NULLIF(ABS((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" + e."Net EPS - Basic (-2FY)" +
                             e."Net EPS - Basic (-3FY)" + e."Net EPS - Basic (-4FY)") / 5.0), 0) * 100
        END                                                                                            AS eps_vs_5y_avg,
    (CASE WHEN e."Net EPS - Basic (FY)" > e."Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-1FY)" > e."Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-2FY)" > e."Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-3FY)" > e."Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
     CASE
         WHEN e."Net EPS - Basic (-4FY)" > e."Net EPS - Basic (-5FY)" THEN 1
         ELSE 0 END)::INTEGER                                                                          AS eps_improvement_count,
    (CASE WHEN e."Net EPS - Basic (FY)" > e."Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-1FY)" > e."Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-2FY)" > e."Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-3FY)" > e."Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Net EPS - Basic (-4FY)" > e."Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
        ) / 5.0 *
    100                                                                                                AS eps_trajectory_score,
    -- *** PREVIOUSLY MISSING: eps_stability ***
    CASE
        WHEN ABS((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" + e."Net EPS - Basic (-2FY)" +
                  e."Net EPS - Basic (-3FY)" + e."Net EPS - Basic (-4FY)") / 5.0) > 0
            THEN 1.0 - LEAST(1.0,
                             SQRT(
                                     (POWER(e."Net EPS - Basic (FY)" -
                                            ((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" +
                                              e."Net EPS - Basic (-2FY)" + e."Net EPS - Basic (-3FY)" +
                                              e."Net EPS - Basic (-4FY)") / 5.0), 2) +
                                      POWER(e."Net EPS - Basic (-1FY)" -
                                            ((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" +
                                              e."Net EPS - Basic (-2FY)" + e."Net EPS - Basic (-3FY)" +
                                              e."Net EPS - Basic (-4FY)") / 5.0), 2) +
                                      POWER(e."Net EPS - Basic (-2FY)" -
                                            ((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" +
                                              e."Net EPS - Basic (-2FY)" + e."Net EPS - Basic (-3FY)" +
                                              e."Net EPS - Basic (-4FY)") / 5.0), 2) +
                                      POWER(e."Net EPS - Basic (-3FY)" -
                                            ((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" +
                                              e."Net EPS - Basic (-2FY)" + e."Net EPS - Basic (-3FY)" +
                                              e."Net EPS - Basic (-4FY)") / 5.0), 2) +
                                      POWER(e."Net EPS - Basic (-4FY)" -
                                            ((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" +
                                              e."Net EPS - Basic (-2FY)" + e."Net EPS - Basic (-3FY)" +
                                              e."Net EPS - Basic (-4FY)") / 5.0), 2)
                                         ) / 5.0
                             ) / NULLIF(ABS((e."Net EPS - Basic (FY)" + e."Net EPS - Basic (-1FY)" +
                                             e."Net EPS - Basic (-2FY)" +
                                             e."Net EPS - Basic (-3FY)" + e."Net EPS - Basic (-4FY)") / 5.0), 0)
                       )
        END                                                                                            AS eps_stability,

    -- GAAP Adjusted Analytics (calc_gaap_adjusted_analytics)
    e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)"                                                     AS eps_adjustment_spread_ltm,
    e."EPS/Adj. (FY)" - e."Net EPS - Basic (FY)"                                                       AS eps_adjustment_spread_fy,
    e."EPS/Adj. (-1FY)" - e."Net EPS - Basic (-1FY)"                                                   AS eps_adjustment_spread_1fy,
    e."EPS/Adj. (FQ)" - e."Net EPS - Basic (FQ)"                                                       AS eps_adjustment_spread_fq,
    e."EPS/Adj. (-1FQFQ)" - e."Net EPS - Basic (-1FQFQ)"                                               AS eps_adjustment_spread_1fqfq,
    e."EPS/Adj. (-2FQFQ)" - e."Net EPS - Basic (-2FQFQ)"                                               AS eps_adjustment_spread_2fqfq,
    e."EPS/Adj. (-3FQFQ)" - e."Net EPS - Basic (-3FQFQ)"                                               AS eps_adjustment_spread_3fqfq,
    e."EPS/Adj. (-4FQFQ)" - e."Net EPS - Basic (-4FQFQ)"                                               AS eps_adjustment_spread_4fqfq,
    e."EPS/Adj. (-2FY)" - e."Net EPS - Basic (-2FY)"                                                   AS eps_adjustment_spread_2fy,
    e."EPS/Adj. (-3FY)" - e."Net EPS - Basic (-3FY)"                                                   AS eps_adjustment_spread_3fy,
    e."EPS/Adj. (-4FY)" - e."Net EPS - Basic (-4FY)"                                                   AS eps_adjustment_spread_4fy,
    (e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") / NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) *
    100                                                                                                AS eps_adjustment_pct,
    e."Net Income/Adj. (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0)                                 AS net_income_adjustment_ratio_ltm,
    e."Net Income/Adj. (FY)" / NULLIF(e."Net Income - (IS) (FY)", 0)                                   AS net_income_adjustment_ratio_fy,
    e."Net Income/Adj. (-1FY)" / NULLIF(e."Net Income - (IS) (-1FY)", 0)                               AS net_income_adjustment_ratio_1fy,
    e."Net Income/Adj. (FQ)" / NULLIF(e."Net Income - (IS) (FQ)", 0)                                   AS net_income_adjustment_ratio_fq,
    e."Net Income/Adj. (5YAVGFQ)" /
    NULLIF(e."Net Income - (IS) (5YAVGFQ)", 0)                                                         AS net_income_adjustment_ratio_5yavgfq,
    e."Net Income/Adj. (-1FQFQ)" /
    NULLIF(e."Net Income - (IS) (-1FQFQ)", 0)                                                          AS net_income_adjustment_ratio_1fqfq,
    e."Net Income/Adj. (-2FQFQ)" /
    NULLIF(e."Net Income - (IS) (-2FQFQ)", 0)                                                          AS net_income_adjustment_ratio_2fqfq,
    e."Net Income/Adj. (-3FQFQ)" /
    NULLIF(e."Net Income - (IS) (-3FQFQ)", 0)                                                          AS net_income_adjustment_ratio_3fqfq,
    e."Net Income/Adj. (-4FQFQ)" /
    NULLIF(e."Net Income - (IS) (-4FQFQ)", 0)                                                          AS net_income_adjustment_ratio_4fqfq,
    e."Net Income/Adj. (-2FY)" / NULLIF(e."Net Income - (IS) (-2FY)", 0)                               AS net_income_adjustment_ratio_2fy,
    e."Net Income/Adj. (-3FY)" / NULLIF(e."Net Income - (IS) (-3FY)", 0)                               AS net_income_adjustment_ratio_3fy,
    e."Net Income/Adj. (-4FY)" / NULLIF(e."Net Income - (IS) (-4FY)", 0)                               AS net_income_adjustment_ratio_4fy,
    (e."Net Income/Adj. (LTM)" - e."Net Income - (IS) (LTM)") / NULLIF(ABS(e."Net Income - (IS) (LTM)"), 0) * 100
                                                                                                       AS net_income_adjustment_pct,
    (e."EBITDA/Adj. (LTM)" - e."EBITDA (LTM)") / NULLIF(ABS(e."EBITDA (LTM)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_ltm,
    (e."EBITDA/Adj. (FY)" - e."EBITDA (FY)") / NULLIF(ABS(e."EBITDA (FY)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_fy,
    (e."EBITDA/Adj. (-1FY)" - e."EBITDA (-1FY)") / NULLIF(ABS(e."EBITDA (-1FY)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_1fy,
    (e."EBITDA/Adj. (FQ)" - e."EBITDA (FQ)") / NULLIF(ABS(e."EBITDA (FQ)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_fq,
    (e."EBITDA/Adj. (-1FQFQ)" - e."EBITDA (-1FQFQ)") / NULLIF(ABS(e."EBITDA (-1FQFQ)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_1fqfq,
    (e."EBITDA/Adj. (-2FQFQ)" - e."EBITDA (-2FQFQ)") / NULLIF(ABS(e."EBITDA (-2FQFQ)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_2fqfq,
    (e."EBITDA/Adj. (-3FQFQ)" - e."EBITDA (-3FQFQ)") / NULLIF(ABS(e."EBITDA (-3FQFQ)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_3fqfq,
    (e."EBITDA/Adj. (-4FQFQ)" - e."EBITDA (-4FQFQ)") / NULLIF(ABS(e."EBITDA (-4FQFQ)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_4fqfq,
    (e."EBITDA/Adj. (-2FY)" - e."EBITDA (-2FY)") / NULLIF(ABS(e."EBITDA (-2FY)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_2fy,
    (e."EBITDA/Adj. (-3FY)" - e."EBITDA (-3FY)") / NULLIF(ABS(e."EBITDA (-3FY)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_3fy,
    (e."EBITDA/Adj. (-4FY)" - e."EBITDA (-4FY)") / NULLIF(ABS(e."EBITDA (-4FY)"), 0) *
    100                                                                                                AS ebitda_adjustment_pct_4fy,
    (e."EBIT/Adj. (LTM)" - e."EBIT (LTM)") / NULLIF(ABS(e."EBIT (LTM)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_ltm,
    (e."EBIT/Adj. (FY)" - e."EBIT (FY)") / NULLIF(ABS(e."EBIT (FY)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_fy,
    (e."EBIT/Adj. (-1FY)" - e."EBIT (-1FY)") / NULLIF(ABS(e."EBIT (-1FY)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_1fy,
    (e."EBIT/Adj. (FQ)" - e."EBIT (FQ)") / NULLIF(ABS(e."EBIT (FQ)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_fq,
    (e."EBIT/Adj. (-1FQFQ)" - e."EBIT (-1FQFQ)") / NULLIF(ABS(e."EBIT (-1FQFQ)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_1fqfq,
    (e."EBIT/Adj. (-2FQFQ)" - e."EBIT (-2FQFQ)") / NULLIF(ABS(e."EBIT (-2FQFQ)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_2fqfq,
    (e."EBIT/Adj. (-3FQFQ)" - e."EBIT (-3FQFQ)") / NULLIF(ABS(e."EBIT (-3FQFQ)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_3fqfq,
    (e."EBIT/Adj. (-4FQFQ)" - e."EBIT (-4FQFQ)") / NULLIF(ABS(e."EBIT (-4FQFQ)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_4fqfq,
    (e."EBIT/Adj. (-2FY)" - e."EBIT (-2FY)") / NULLIF(ABS(e."EBIT (-2FY)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_2fy,
    (e."EBIT/Adj. (-3FY)" - e."EBIT (-3FY)") / NULLIF(ABS(e."EBIT (-3FY)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_3fy,
    (e."EBIT/Adj. (-4FY)" - e."EBIT (-4FY)") / NULLIF(ABS(e."EBIT (-4FY)"), 0) *
    100                                                                                                AS ebit_adjustment_pct_4fy,
    GREATEST(0, LEAST(100, 100 - ABS((e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") /
                                     NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) *
                                     100)))                                                            AS earnings_quality_score,
    CASE
        WHEN ABS((e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") / NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) * 100) >
             15
            THEN 1
        ELSE 0 END                                                                                     AS earnings_quality_warning,
    e."EPS Norm - Est Avg (FY1E)" - e."EPS GAAP - Est Avg (FY1E)"                                      AS forward_eps_gaap_adj_spread,

    -- GAAP Revision Features (calc_gaap_revision_features)
    COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
    COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
    COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
    COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) *
    0.15                                                                                               AS gaap_revision_momentum,
    e."EPS GAAP Est Avg Rev % (FY1E - 1M)"                                                             AS gaap_revision_1m,
    e."EPS GAAP Est Avg Rev % (FY1E - 3M)"                                                             AS gaap_revision_3m,
    e."EPS GAAP Est Avg Rev % (FY1E - 6M)"                                                             AS gaap_revision_6m,
    e."EPS GAAP Est Avg Rev % (FY1E - 1Y)"                                                             AS gaap_revision_1y,
    e."EPS Est Avg Rev % (FY1E - 3M)" -
    e."EPS GAAP Est Avg Rev % (FY1E - 3M)"                                                             AS gaap_vs_norm_revision_spread,
    e."EPS GAAP Est Avg Rev % (FY1E - 1M)" -
    e."EPS GAAP Est Avg Rev % (FY1E - 6M)"                                                             AS gaap_revision_acceleration,
    CASE
        WHEN e."EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0 AND e."EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0 AND
             e."EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0 THEN 1
        ELSE 0 END                                                                                     AS gaap_positive_revision_flag,
    -- *** PREVIOUSLY MISSING: revision_quality_divergence ***
    ABS((e."EPS Est Avg Rev % (FY1E - 3M)" - e."EPS GAAP Est Avg Rev % (FY1E - 3M)") -
        (e."EPS Est Avg Rev % (FY1E - 1M)" -
         e."EPS GAAP Est Avg Rev % (FY1E - 1M)"))                                                      AS revision_quality_divergence,

    -- =========================================================================
    -- [11/18] GROWTH FEATURES (vw_features_growth)
    -- calc_growth_features
    -- =========================================================================
    CASE
        WHEN ABS(e."Total Revenues (-1FY)") > 0
            THEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") / NULLIF(ABS(e."Total Revenues (-1FY)"), 0) * 100
        END                                                                                            AS revenue_growth_yoy,
    CASE
        WHEN ABS(e."EBITDA (-1FY)") > 0
            THEN (e."EBITDA (FY)" - e."EBITDA (-1FY)") / NULLIF(ABS(e."EBITDA (-1FY)"), 0) * 100
        END                                                                                            AS ebitda_growth_yoy,
    CASE
        WHEN ABS(e."Operating Income (FY)") > 0
            THEN (e."Operating Income (LTM)" - e."Operating Income (FY)") / NULLIF(ABS(e."Operating Income (FY)"), 0) *
                 100
        END                                                                                            AS operating_income_growth,
    CASE
        WHEN ABS(e."FCF (FY)") > 0
            THEN (e."FCF (LTM)" - e."FCF (FY)") / NULLIF(ABS(e."FCF (FY)"), 0) * 100
        END                                                                                            AS fcf_growth,
    e."Total Revenues/CAGR (5Y FY)"                                                                    AS revenue_cagr_5y,
    e."Revenues - Est YoY % (FY1E)"                                                                    AS forward_revenue_growth,
    e."Total Revenues (LTM)" / NULLIF(e."Total Revenues (5YAVGLTM)", 0)                                AS revenue_vs_5y_avg,

    -- calc_revenue_forecast_features
    (e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)") / NULLIF(e."Revenues - Est Med (FY1E)", 0) * 100
                                                                                                       AS revenue_est_spread,
    (e."Total Revenues (LTM)" - e."Revenues - Est Avg (FY1E)") / NULLIF(ABS(e."Revenues - Est Avg (FY1E)"), 0) * 100
                                                                                                       AS revenue_beat_potential,
    e."Revenues - Est YoY % (FY1E)"                                                                    AS revenue_est_revision_trend,
    (e."EBITDA (LTM)" - e."EBITDA - Est Avg (FY1E)") / NULLIF(ABS(e."EBITDA - Est Avg (FY1E)"), 0) * 100
                                                                                                       AS ebitda_est_vs_actual,
    e."Enterprise Value" / NULLIF(e."Revenues - Est Avg (FY1E)", 0)                                    AS forward_revenue_multiple,
    e."EPS Norm - Est # (FY1E)"                                                                        AS revenue_estimate_count,
    (e."Revenues - Est Avg (NTM)" - e."Revenues - Est Avg (FY1E)") / NULLIF(ABS(e."Revenues - Est Avg (FY1E)"), 0) * 100
                                                                                                       AS revenue_guidance_gap,
    (e."Revenues - Est Avg (FY1E)" - e."Total Revenues (FY)") / NULLIF(ABS(e."Total Revenues (FY)"), 0) * 100
                                                                                                       AS consensus_revenue_growth,
    -- *** PREVIOUSLY MISSING: ebit_estimate_spread ***
    (e."EBIT - Est Med (FY1E)" - e."EBIT - Est Med (NTM)") / NULLIF(ABS(e."EBIT - Est Med (NTM)"), 0) * 100
                                                                                                       AS ebit_estimate_spread,
    e."EBITDA - Est Avg (FY1E)" / NULLIF(e."Revenues - Est Avg (FY1E)", 0) *
    100                                                                                                AS forward_ebitda_margin,
    e."Revenues - Est YoY % (FY1E)" - e."Total Revenues/CAGR (5Y FY)"                                  AS revenue_acceleration,
    GREATEST(0, LEAST(100, 100 - ABS((e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)") /
                                     NULLIF(e."Revenues - Est Med (FY1E)", 0) *
                                     100)))                                                            AS estimate_confidence_score,

    -- =========================================================================
    -- [12/18] DIVIDEND FEATURES (vw_features_dividends)
    -- calc_dividend_features
    -- =========================================================================
    e."Dividend Streak"::INTEGER                                                                       AS dividend_streak,
    e."Div Yield (LTM)"                                                                                AS dividend_yield_ltm,
    e."Div Yield (NTM)"                                                                                AS dividend_yield_ntm,
    ABS(e."Common Dividends Paid (LTM)") /
    NULLIF(e."Net Income/Adj. (LTM)", 0)                                                               AS dividend_payout_ratio,
    CASE
        WHEN ABS(e."Common Dividends Paid (LTM)") > 0
            THEN e."FCF (LTM)" / NULLIF(ABS(e."Common Dividends Paid (LTM)"), 0)
        END                                                                                            AS fcf_dividend_coverage,
    e."Buyback Yield (LTM)"                                                                            AS buyback_yield,
    COALESCE(e."Buyback Yield (LTM)", 0) +
    COALESCE(e."Div Yield (LTM)", 0)                                                                   AS total_shareholder_yield,
    e."Div Yield (NTM)" - e."Div Yield (LTM)"                                                          AS dividend_growth_expectation,

    -- calc_dividend_timing
    (CURRENT_DATE - e."Dividend Record (Ex Date)")::INTEGER                                            AS days_since_ex_date,
    (e."Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER                                       AS days_to_payment,
    CASE
        WHEN (CURRENT_DATE - e."Dividend Record (Announce Date)") <= 30 THEN 1
        ELSE 0 END                                                                                     AS dividend_announced_flag,
    CASE
        WHEN (e."Dividend Record (Ex Date)" - CURRENT_DATE) BETWEEN 0 AND 14 THEN 1
        ELSE 0 END                                                                                     AS ex_date_approaching_flag,
    CASE e."Dividend Record (Frequency)"
        WHEN 'Quarterly' THEN 4
        WHEN 'Semi-Annual' THEN 2
        WHEN 'Annual' THEN 1
        WHEN 'Monthly' THEN 12
        ELSE 0
        END                                                                                            AS dividend_frequency_score,
    LEAST(1.0, e."Dividend Streak"::NUMERIC / 10.0)                                                    AS dividend_consistency,
    -- *** PREVIOUSLY MISSING: recent_dividend_change ***
    CASE
        WHEN e."Div Yield (-1FYInd)" > 0
            THEN (e."Div Yield (Ind)" - e."Div Yield (-1FYInd)") /
                 NULLIF(e."Div Yield (-1FYInd)", 0) * 100
        END                                                                                            AS recent_dividend_change,
    e."Div Yield (LTM)" / NULLIF(e."Div Yield (5YAVGLTM)", 0)                                          AS dividend_yield_vs_5y_avg,

    -- =========================================================================
    -- [13/18] EMPLOYMENT FEATURES (vw_features_employment)
    -- calc_employment_features
    -- =========================================================================
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."Total Revenues (FY)" / NULLIF(e."Full Time Employees (FY)", 0) END                 AS revenue_per_employee,
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."Normalized Net Income (FY)" /
                 NULLIF(e."Full Time Employees (FY)", 0) END                                           AS profit_per_employee,
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."EBITDA (FY)" / NULLIF(e."Full Time Employees (FY)", 0) END                         AS ebitda_per_employee,
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."Total Assets (FY)" / NULLIF(e."Full Time Employees (FY)", 0) END                   AS assets_per_employee,
    CASE
        WHEN e."Full Time Employees (-1FY)" > 0
            THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                 NULLIF(e."Full Time Employees (-1FY)", 0) *
                 100 END                                                                               AS fte_growth_1y_pct,
    CASE
        WHEN e."Full Time Employees (-3FY)" > 0
            THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-3FY)") /
                 NULLIF(e."Full Time Employees (-3FY)", 0) *
                 100 END                                                                               AS fte_growth_3y_pct,
    CASE
        WHEN e."Avg Employees (5YAVGFY)" > 0
            THEN e."Full Time Employees (FY)" / NULLIF(e."Avg Employees (5YAVGFY)", 0) END             AS workforce_stability,

    -- calc_employment_dynamics
    CASE
        WHEN e."Full Time Employees (-2FY)" > 0
            THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-2FY)") /
                 NULLIF(e."Full Time Employees (-2FY)", 0) *
                 100 END                                                                               AS fte_growth_2y_pct,
    -- *** PREVIOUSLY MISSING: fte_acceleration ***
    CASE
        WHEN e."Full Time Employees (-1FY)" > 0 AND e."Full Time Employees (-3FY)" > 0
            THEN (((e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                   NULLIF(e."Full Time Employees (-1FY)", 0)) -
                  (POWER(e."Full Time Employees (FY)" / NULLIF(e."Full Time Employees (-3FY)", 0), 1.0 / 3.0) - 1)) *
                 100
        END                                                                                            AS fte_acceleration,
    -- *** PREVIOUSLY MISSING: workforce_volatility ***
    ABS(((e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
         NULLIF(e."Full Time Employees (-1FY)", 0)) -
        ((e."Full Time Employees (-1FY)" - e."Full Time Employees (-2FY)") /
         NULLIF(e."Full Time Employees (-2FY)", 0))) *
    100                                                                                                AS workforce_volatility,
    -- *** PREVIOUSLY MISSING: hiring_intensity ***
    CASE
        WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") / NULLIF(ABS(e."Total Revenues (-1FY)"), 0) > 0
            THEN ((e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                  NULLIF(e."Full Time Employees (-1FY)", 0)) /
                 NULLIF(((e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
                         NULLIF(ABS(e."Total Revenues (-1FY)"), 0)), 0)
        END                                                                                            AS hiring_intensity,
    -- *** PREVIOUSLY MISSING: productivity_trend ***
    CASE
        WHEN e."Full Time Employees (FY)" > 0 AND e."Full Time Employees (-1FY)" > 0
            THEN ((e."Total Revenues (FY)" / e."Full Time Employees (FY)") -
                  (e."Total Revenues (-1FY)" / e."Full Time Employees (-1FY)")) /
                 NULLIF(ABS(e."Total Revenues (-1FY)" / e."Full Time Employees (-1FY)"), 0) * 100
        END                                                                                            AS productivity_trend,
    -- *** PREVIOUSLY MISSING: headcount_vs_revenue ***
    ((e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
     NULLIF(e."Full Time Employees (-1FY)", 0) * 100) -
    ((e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
     NULLIF(ABS(e."Total Revenues (-1FY)"), 0) *
     100)                                                                                              AS headcount_vs_revenue,
    -- *** PREVIOUSLY MISSING: workforce_efficiency_gain ***
    CASE
        WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") / NULLIF(ABS(e."Total Revenues (-1FY)"), 0) >
             (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") / NULLIF(e."Full Time Employees (-1FY)", 0)
            THEN ((e."Total Revenues (FY)" - e."Total Revenues (-1FY)") / NULLIF(ABS(e."Total Revenues (-1FY)"), 0) -
                  (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                  NULLIF(e."Full Time Employees (-1FY)", 0)) * 100
        ELSE 0
        END                                                                                            AS workforce_efficiency_gain,
    CASE
        WHEN e."Full Time Employees (FY)" < e."Full Time Employees (-1FY)" AND
             e."Total Revenues (FY)" < e."Total Revenues (-1FY)" THEN 1
        ELSE 0 END                                                                                     AS layoff_risk_flag,
    CASE
        WHEN (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
             NULLIF(e."Full Time Employees (-1FY)", 0) > 0.20 THEN 1
        ELSE 0 END                                                                                     AS rapid_hiring_flag,
    CASE
        WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") / NULLIF(ABS(e."Total Revenues (-1FY)"), 0) >
             (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") / NULLIF(e."Full Time Employees (-1FY)", 0)
            AND (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") > 0 THEN 1
        ELSE 0 END                                                                                     AS sustainable_growth_flag,

    -- =========================================================================
    -- [14/18] CASH FLOW FEATURES (vw_features_cashflow)
    -- calc_cashflow_features
    -- =========================================================================
    e."CFO (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0)                                             AS cfo_to_net_income,
    e."FCF (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0)                                             AS fcf_to_net_income,
    e."FCF (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)                                                AS fcf_margin,
    (e."CFO (LTM)" - e."CFO (-1FY)") / NULLIF(e."CFO (-1FY)", 0)                                       AS cfo_growth_yoy,
    (CASE WHEN e."FCF (FQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END) /
    5.0                                                                                                AS fcf_positive_ratio,
    ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) + ABS(COALESCE(e."Cash Acquisitions (-1FQFQ)", 0)) +
    ABS(COALESCE(e."Cash Acquisitions (-2FQFQ)", 0)) + ABS(COALESCE(e."Cash Acquisitions (-3FQFQ)", 0))
                                                                                                       AS acquisition_intensity,
    CASE
        WHEN ABS(e."CFI (LTM)") > 0
            THEN e."CFO (LTM)" / NULLIF(ABS(e."CFI (LTM)"), 0) END                                     AS self_funding_ratio,

    -- calc_enhanced_cashflow_features
    (CASE WHEN e."FCF (FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."FCF (-4FY)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS fcf_positive_years,
    CASE
        WHEN e."FCF (FY)" > 0 AND e."FCF (-1FY)" > 0 AND e."FCF (-2FY)" > 0 AND e."FCF (-3FY)" > 0 AND
             e."FCF (-4FY)" > 0 THEN 1
        ELSE 0 END                                                                                     AS fcf_always_positive,
    ABS(e."Capital Expenditure (FQ)") /
    NULLIF(ABS(e."Capital Expenditure (5YAVGFQ)"), 0)                                                  AS capex_vs_5y_avg,
    CASE
        WHEN ABS(e."Capital Expenditure (FQ)") / NULLIF(ABS(e."Capital Expenditure (5YAVGFQ)"), 0) < 0.7 THEN 1
        ELSE 0 END                                                                                     AS underinvestment_flag,
    ABS(e."CFO (LTM)") / NULLIF(ABS(e."CFO (LTM)") + ABS(e."CFI (LTM)") + ABS(e."CFF (LTM)"),
                                0)                                                                     AS cfo_share_of_cf,
    ABS(e."CFI (LTM)") / NULLIF(ABS(e."CFO (LTM)") + ABS(e."CFI (LTM)") + ABS(e."CFF (LTM)"),
                                0)                                                                     AS cfi_share_of_cf,
    ABS(e."CFF (LTM)") / NULLIF(ABS(e."CFO (LTM)") + ABS(e."CFI (LTM)") + ABS(e."CFF (LTM)"),
                                0)                                                                     AS cff_share_of_cf,
    CASE
        WHEN e."CFO (LTM)" / NULLIF(ABS(e."CFI (LTM)"), 0) > 1 THEN 1
        ELSE 0 END                                                                                     AS self_funding_flag,
    -- *** PREVIOUSLY MISSING: acquisition_to_fcf ***
    (ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) + ABS(COALESCE(e."Cash Acquisitions (-1FQFQ)", 0)) +
     ABS(COALESCE(e."Cash Acquisitions (-2FQFQ)", 0)) + ABS(COALESCE(e."Cash Acquisitions (-3FQFQ)", 0))) /
    NULLIF(ABS(e."FCF (LTM)"), 0)                                                                      AS acquisition_to_fcf,
    -- *** PREVIOUSLY MISSING: sustainable_ma_flag ***
    CASE
        WHEN (ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) + ABS(COALESCE(e."Cash Acquisitions (-1FQFQ)", 0)) +
              ABS(COALESCE(e."Cash Acquisitions (-2FQFQ)", 0)) + ABS(COALESCE(e."Cash Acquisitions (-3FQFQ)", 0))) /
             NULLIF(ABS(e."FCF (LTM)"), 0) < 0.5 THEN 1
        ELSE 0 END                                                                                     AS sustainable_ma_flag,
    (e."FCF (FQ)" - e."FCF (-4FQFQ)") /
    NULLIF(ABS(e."FCF (-4FQFQ)"), 0)                                                                   AS fcf_4q_improvement,
    (CASE WHEN e."CFO (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
     CASE
         WHEN e."FCF (FY)" > 0 AND e."FCF (-1FY)" > 0 AND e."FCF (-2FY)" > 0 AND e."FCF (-3FY)" > 0 AND
              e."FCF (-4FY)" > 0 THEN 25
         ELSE 0 END +
     CASE WHEN e."CFO (LTM)" > ABS(e."CFI (LTM)") THEN 25 ELSE 0 END +
     CASE
         WHEN e."FCF (LTM)" > 0 THEN 25
         ELSE 0 END)::NUMERIC                                                                          AS cash_flow_quality_score,
    -- *** PREVIOUSLY MISSING: capex temporal features ***
    (ABS(e."Capital Expenditure (FY)") - ABS(e."Capital Expenditure (-1FY)")) /
    NULLIF(ABS(e."Capital Expenditure (-1FY)"), 0) *
    100                                                                                                AS capex_yoy_growth,
    (ABS(e."Capital Expenditure (FQ)") - ABS(e."Capital Expenditure (-1FQFQ)")) /
    NULLIF(ABS(e."Capital Expenditure (-1FQFQ)"), 0) *
    100                                                                                                AS capex_qoq_growth,
    (ABS(e."Capital Expenditure (FY)") - ABS(e."Capital Expenditure (-3FY)")) /
    NULLIF(ABS(e."Capital Expenditure (-3FY)"), 0) *
    100                                                                                                AS capex_3y_trend,
    (ABS(ABS(e."Capital Expenditure (FQ)") - ABS(e."Capital Expenditure (-1FQFQ)")) +
     ABS(ABS(e."Capital Expenditure (-1FQFQ)") - ABS(e."Capital Expenditure (-2FQFQ)")) +
     ABS(ABS(e."Capital Expenditure (-2FQFQ)") - ABS(e."Capital Expenditure (-3FQFQ)")) +
     ABS(ABS(e."Capital Expenditure (-3FQFQ)") - ABS(e."Capital Expenditure (-4FQFQ)"))) /
    NULLIF((ABS(e."Capital Expenditure (FQ)") + ABS(e."Capital Expenditure (-1FQFQ)") +
            ABS(e."Capital Expenditure (-2FQFQ)") + ABS(e."Capital Expenditure (-3FQFQ)") +
            ABS(e."Capital Expenditure (-4FQFQ)")) / 5.0,
           0)                                                                                          AS capex_volatility,
    CASE
        WHEN ABS(e."Capital Expenditure (FY)") > ABS(e."Capital Expenditure (-1FY)")
            AND ABS(e."Capital Expenditure (-1FY)") > ABS(e."Capital Expenditure (-2FY)") THEN 1
        ELSE 0 END                                                                                     AS capex_acceleration,
    CASE
        WHEN (ABS(e."Capital Expenditure (FY)") - ABS(e."Capital Expenditure (-1FY)")) /
             NULLIF(ABS(e."Capital Expenditure (-1FY)"), 0) < -0.25 THEN 1
        ELSE 0 END                                                                                     AS capex_cut_flag,
    CASE
        WHEN ABS(e."Capital Expenditure (FQ)") / NULLIF(ABS(e."Capital Expenditure (5YAVGFQ)"), 0) > 1.5 THEN 1
        ELSE 0 END                                                                                     AS overinvestment_flag,
    -- *** PREVIOUSLY MISSING: acquisitions temporal features ***
    (ABS(COALESCE(e."Cash Acquisitions (FY)", 0)) - ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0))) /
    NULLIF(ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0)), 0) *
    100                                                                                                AS acquisitions_yoy_growth,
    ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) /
    NULLIF(ABS(COALESCE(e."Cash Acquisitions (5YAVGFQ)", 0)), 0)                                       AS acquisitions_vs_5y_avg,
    ABS(COALESCE(e."Cash Acquisitions (LTM)", 0))                                                      AS acquisitions_ltm_total,
    ABS(COALESCE(e."Cash Acquisitions (LTM)", 0)) / NULLIF(e."Total Assets (LTM)", 0) *
    100                                                                                                AS ma_intensity_score,
    CASE
        WHEN (CASE WHEN ABS(COALESCE(e."Cash Acquisitions (FY)", 0)) > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0)) > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS(COALESCE(e."Cash Acquisitions (-2FY)", 0)) > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS(COALESCE(e."Cash Acquisitions (-3FY)", 0)) > 0 THEN 1 ELSE 0 END) >= 3 THEN 1
        ELSE 0 END                                                                                     AS serial_acquirer_flag,
    CASE
        WHEN ABS(COALESCE(e."Cash Acquisitions (FY)", 0)) = 0
            AND (ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0)) > 0
                OR ABS(COALESCE(e."Cash Acquisitions (-2FY)", 0)) > 0) THEN 1
        ELSE 0 END                                                                                     AS acquisition_pause_flag,
    (ABS(COALESCE(e."Capital Expenditure (LTM)", 0)) + ABS(COALESCE(e."Cash Acquisitions (LTM)", 0))) /
    NULLIF(ABS(e."CFO (LTM)"), 0)                                                                      AS total_investment_to_cfo,
    ABS(COALESCE(e."Capital Expenditure (LTM)", 0)) /
    NULLIF(ABS(COALESCE(e."Cash Acquisitions (LTM)", 0)), 0)                                           AS organic_vs_inorganic,
    CASE
        WHEN (ABS(COALESCE(e."Capital Expenditure (-1FY)", 0)) + ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0))) > 0
            THEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
                 NULLIF(ABS(COALESCE(e."Capital Expenditure (-1FY)", 0)) +
                        ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0)), 0)
        END                                                                                            AS investment_efficiency,

    -- calc_cashflow_temporal_features
    (e."CFO (FQ)" - e."CFO (-4FQFQ)") / NULLIF(ABS(e."CFO (-4FQFQ)"), 0) *
    100                                                                                                AS cfo_quarterly_trend,
    -- *** PREVIOUSLY MISSING: cfo_yoy_quarterly ***
    CASE
        WHEN ABS(e."CFO (-4FQFQ)") > 0
            THEN (e."CFO (FQ)" - e."CFO (-4FQFQ)") / NULLIF(ABS(e."CFO (-4FQFQ)"), 0) * 100
        END                                                                                            AS cfo_yoy_quarterly,
    (e."CFI (FQ)" - e."CFI (-4FQFQ)") / NULLIF(ABS(e."CFI (-4FQFQ)"), 0) *
    100                                                                                                AS cfi_quarterly_trend,
    (e."CFF (FQ)" - e."CFF (-4FQFQ)") / NULLIF(ABS(e."CFF (-4FQFQ)"), 0) *
    100                                                                                                AS cff_quarterly_trend,
    (e."FCF (FQ)" - e."FCF (-4FQFQ)") / NULLIF(ABS(e."FCF (-4FQFQ)"), 0) *
    100                                                                                                AS fcf_quarterly_trend,
    (CASE WHEN e."CFO (FQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."CFO (-4FQFQ)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS cfo_positive_quarters,
    (CASE WHEN e."CFI (FQ)" < 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFI (-1FQFQ)" < 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFI (-2FQFQ)" < 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFI (-3FQFQ)" < 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."CFI (-4FQFQ)" < 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS cfi_negative_quarters,
    -- *** PREVIOUSLY MISSING: cff_pattern_score ***
    CASE
        WHEN (e."CFF (FQ)" + e."CFF (-1FQFQ)" + e."CFF (-2FQFQ)" + e."CFF (-3FQFQ)") > 0 THEN -1
        WHEN (e."CFF (FQ)" + e."CFF (-1FQFQ)" + e."CFF (-2FQFQ)" + e."CFF (-3FQFQ)") < 0 THEN 1
        ELSE 0 END::NUMERIC                                                                            AS cff_pattern_score,
    CASE
        WHEN e."FCF (LTM)" < 0 THEN ABS(e."FCF (LTM)") / NULLIF(e."Cash And Equivalents (FQ)", 0) / 12.0
        ELSE 0 END                                                                                     AS cash_burn_rate,
    -- *** PREVIOUSLY MISSING: cf_volatility_score ***
    (ABS(e."CFO (FQ)" - e."CFO (-1FQFQ)") + ABS(e."CFO (-1FQFQ)" - e."CFO (-2FQFQ)") +
     ABS(e."CFO (-2FQFQ)" - e."CFO (-3FQFQ)") + ABS(e."CFO (-3FQFQ)" - e."CFO (-4FQFQ)")) /
    NULLIF(ABS(e."CFO (FQ)" + e."CFO (-1FQFQ)" + e."CFO (-2FQFQ)" +
               e."CFO (-3FQFQ)" + e."CFO (-4FQFQ)") / 5.0,
           0)                                                                                          AS cf_volatility_score,
    -- *** PREVIOUSLY MISSING: operating_cf_momentum ***
    ((e."CFO (FQ)" + e."CFO (-1FQFQ)") - (e."CFO (-3FQFQ)" + e."CFO (-4FQFQ)")) /
    NULLIF(ABS(e."CFO (-3FQFQ)" + e."CFO (-4FQFQ)"), 0) *
    100                                                                                                AS operating_cf_momentum,
    ABS(e."CFF (LTM)") / NULLIF(ABS(e."CFO (LTM)"), 0)                                                 AS financing_dependency,

    -- Cashflow comprehensive
    e."CFO (FQ)"                                                                                       AS cfo_fq,
    e."CFO (LTM)"                                                                                      AS cfo_ltm,
    e."CFO (FY)"                                                                                       AS cfo_fy,
    e."FCF (FQ)"                                                                                       AS fcf_fq,
    e."FCF (LTM)"                                                                                      AS fcf_ltm,
    e."FCF (FY)"                                                                                       AS fcf_fy,
    (e."CFO (FY)" - e."CFO (-1FY)") / NULLIF(ABS(e."CFO (-1FY)"), 0) *
    100                                                                                                AS cfo_growth_yoy_comp,
    (e."FCF (FY)" - e."FCF (-1FY)") / NULLIF(ABS(e."FCF (-1FY)"), 0) *
    100                                                                                                AS fcf_growth_yoy,
    e."FCF (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                          AS fcf_margin_pct,
    e."FCF (LTM)" / NULLIF(e."Market Cap", 0) * 100                                                    AS fcf_yield,
    (CASE WHEN e."CFO (FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (-1FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (-2FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (-3FY)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."CFO (-4FY)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS cfo_positive_years,

    -- =========================================================================
    -- [15/18] COST STRUCTURE FEATURES (vw_features_cost_structure)
    -- *** THIS ENTIRE SECTION WAS MISSING — DIRECT CAUSE OF THE CRASH ***
    -- =========================================================================
    safe_divide(e."Cost Of Revenues (LTM)", e."Total Revenues (LTM)") *
    100                                                                                                AS cogs_to_revenue,
    safe_divide(e."Total Operating Expenses (LTM)", e."Total Revenues (LTM)") *
    100                                                                                                AS opex_to_revenue,
    safe_divide(e."Selling General & Admin Expenses/Total (FY)", e."Total Revenues (FY)") *
    100                                                                                                AS sga_to_revenue,
    safe_divide(e."R&D Expenses (LTM)", e."Total Revenues (LTM)") * 100                                AS rnd_to_revenue,
    safe_divide(e."Interest Expense/Total (LTM)", e."Total Revenues (LTM)") *
    100                                                                                                AS interest_to_revenue,
    -- R&D temporal features (calc_rnd_temporal_features) — THE CRASH COLUMNS
    e."R&D Expenses (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                 AS rnd_intensity_ltm,
    CASE
        WHEN ABS(e."R&D Expenses (-1FY)") > 0
            THEN (e."R&D Expenses (FY)" - e."R&D Expenses (-1FY)") / NULLIF(ABS(e."R&D Expenses (-1FY)"), 0) * 100
        END                                                                                            AS rnd_yoy_growth,
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."R&D Expenses (FY)" / NULLIF(e."Full Time Employees (FY)", 0)
        END                                                                                            AS rnd_per_employee,
    CASE
        WHEN e."R&D Expenses (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) > 0.15 THEN 1
        ELSE 0
        END                                                                                            AS high_rnd_intensity_flag,
    -- Cost trend features
    (safe_divide(e."Cost Of Revenues (LTM)", e."Total Revenues (LTM)") -
     safe_divide(e."Cost Of Revenues (FY)", e."Total Revenues (FY)")) *
    100                                                                                                AS cogs_to_revenue_trend,
    (safe_divide(e."Total Operating Expenses (LTM)", e."Total Revenues (LTM)") -
     safe_divide(e."Total Operating Expenses (FY)", e."Total Revenues (FY)")) *
    100                                                                                                AS opex_efficiency_trend,
    safe_divide(e."Selling General & Admin Expenses/Total (FY)",
                e."Total Revenues (FY)") /
    NULLIF(safe_divide(e."Selling General & Admin Expenses/Total (-1FY)",
                       e."Total Revenues (-1FY)"),
           0)                                                                                          AS sga_efficiency,
    -- Interest income features
    e."Interest Income On Investments (LTM)"                                                           AS interest_income_ltm,
    e."Interest Expense/Total (LTM)"                                                                   AS interest_expense_ltm,
    COALESCE(e."Interest Income On Investments (LTM)", 0) -
    COALESCE(e."Interest Expense/Total (LTM)", 0)                                                      AS net_interest_income,
    e."EBIT (LTM)" / NULLIF(e."Interest Expense/Total (LTM)", 0)                                       AS interest_coverage_ratio,
    e."Interest Income On Investments (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) *
    100                                                                                                AS interest_income_to_revenue,
    e."Interest Expense/Total (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) *
    100                                                                                                AS interest_expense_to_revenue,

    -- =========================================================================
    -- [16/18] TEMPORAL FEATURES (vw_features_temporal)
    -- calc_temporal_features
    -- =========================================================================
    e."Fiscal Quarter"                                                                                 AS fiscal_quarter,
    e."Fiscal Month"                                                                                   AS fiscal_month,
    e."Fiscal Year"                                                                                    AS fiscal_year,
    (e."Next Earnings" - CURRENT_DATE)::INTEGER                                                        AS days_to_earnings,
    (CURRENT_DATE - e."Income Statement Report Date")::INTEGER                                         AS earnings_report_recency,
    e."Reporting Lag"                                                                                  AS reporting_lag,
    e."Fiscal Month" / 12.0                                                                            AS fiscal_year_progress,

    -- calc_fiscal_calendar_features
    (CURRENT_DATE - e."Income Statement Report Date")::INTEGER                                         AS days_since_last_report,
    (e."FY End Date" - CURRENT_DATE)::INTEGER                                                          AS days_to_fy_end,
    CASE
        WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12) THEN 1
        ELSE 0 END                                                                                     AS is_quarter_end_month,
    CASE
        WHEN EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM e."FY End Date") THEN 1
        ELSE 0 END                                                                                     AS is_fy_end_month,
    CASE
        WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 2, 4, 5, 7, 8, 10, 11) THEN 1
        ELSE 0 END                                                                                     AS earnings_season_flag,
    CASE
        WHEN (e."Next Earnings" - CURRENT_DATE) BETWEEN 0 AND 14 THEN 1
        ELSE 0 END                                                                                     AS pre_earnings_window,
    CASE
        WHEN (CURRENT_DATE - e."Income Statement Report Date") BETWEEN 0 AND 7 THEN 1
        ELSE 0 END                                                                                     AS post_earnings_window,
    GREATEST(0, LEAST(100, 100 -
                           ((CURRENT_DATE - e."Income Statement Report Date")::NUMERIC / 90.0 * 100))) AS reporting_freshness_score,
    -- *** PREVIOUSLY MISSING: fiscal_quarter_progress ***
    CASE
        WHEN e."Fiscal Month" IS NOT NULL
            THEN ((e."Fiscal Month" - 1) % 3 + 1) / 3.0
        END                                                                                            AS fiscal_quarter_progress,

    -- =========================================================================
    -- [17/18] COMPOSITE SCORES (vw_features_composite_scores)
    -- calc_composite_scores / calc_piotroski_f_score
    -- =========================================================================
    (CASE WHEN e."Return on Assets (ROA) % (LTM)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (LTM)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Return on Assets (ROA) % (LTM)" > e."Return on Assets (ROA) % (FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."CFO (LTM)" > e."Net Income - (IS) (LTM)" THEN 1 ELSE 0 END +
     CASE
         WHEN e."Total Debt (LTM)" / NULLIF(e."Total Equity (LTM)", 0) <
              e."Total Debt (FY)" / NULLIF(e."Total Equity (FY)", 0) THEN 1
         ELSE 0 END +
     CASE WHEN e."Current Ratio (LTM)" > e."Current Ratio (FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Shrs Out" <= e."Shrs Out (-1FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Gross Profit Margin % (LTM)" > e."Gross Profit Margin % (FY)" THEN 1 ELSE 0 END +
     CASE WHEN e."Asset Turnover (LTM)" > e."Asset Turnover (FY)" THEN 1 ELSE 0 END
        )::INTEGER                                                                                     AS piotroski_f_score,
    GREATEST(0, LEAST(100, 50 - ((e."Shrs Out" - e."Shrs Out (-1FY)") / NULLIF(e."Shrs Out (-1FY)", 0)) *
                                100))                                                                  AS dilution_score,

    -- EBIT/EBITDA Comprehensive
    e."EBIT (FQ)"                                                                                      AS ebit_fq,
    e."EBIT (LTM)"                                                                                     AS ebit_ltm,
    e."EBIT (FY)"                                                                                      AS ebit_fy,
    e."EBIT (-1FY)"                                                                                    AS ebit_1fy,
    e."EBITDA (FQ)"                                                                                    AS ebitda_fq,
    e."EBITDA (LTM)"                                                                                   AS ebitda_ltm,
    e."EBITDA (FY)"                                                                                    AS ebitda_fy,
    e."EBITDA (-1FY)"                                                                                  AS ebitda_1fy,
    e."EBIT (-2FY)"                                                                                    AS ebit_2fy,
    e."EBIT (-3FY)"                                                                                    AS ebit_3fy,
    e."EBIT (-4FY)"                                                                                    AS ebit_4fy,
    e."EBITDA (-2FY)"                                                                                  AS ebitda_2fy,
    e."EBITDA (-3FY)"                                                                                  AS ebitda_3fy,
    e."EBITDA (-4FY)"                                                                                  AS ebitda_4fy,
    e."EBIT (-1FQFQ)"                                                                                  AS ebit_1fqfq,
    e."EBIT (-2FQFQ)"                                                                                  AS ebit_2fqfq,
    e."EBIT (-3FQFQ)"                                                                                  AS ebit_3fqfq,
    e."EBIT (-4FQFQ)"                                                                                  AS ebit_4fqfq,
    e."EBITDA (-1FQFQ)"                                                                                AS ebitda_1fqfq,
    e."EBITDA (-2FQFQ)"                                                                                AS ebitda_2fqfq,
    e."EBITDA (-3FQFQ)"                                                                                AS ebitda_3fqfq,
    e."EBITDA (-4FQFQ)"                                                                                AS ebitda_4fqfq,
    e."EBIT (5YAVGFQ)"                                                                                 AS ebit_5yavgfq,
    e."EBIT (5YAVGLTM)"                                                                                AS ebit_5yavgltm,
    e."EBITDA (5YAVGFQ)"                                                                               AS ebitda_5yavgfq,
    e."EBITDA (5YAVGLTM)"                                                                              AS ebitda_5yavgltm,
    e."EBIT/Adj. (FQ)"                                                                                 AS ebit_adj_fq,
    e."EBIT/Adj. (LTM)"                                                                                AS ebit_adj_ltm,
    e."EBIT/Adj. (FY)"                                                                                 AS ebit_adj_fy,
    e."EBITDA/Adj. (FQ)"                                                                               AS ebitda_adj_fq,
    e."EBITDA/Adj. (LTM)"                                                                              AS ebitda_adj_ltm,
    e."EBITDA/Adj. (FY)"                                                                               AS ebitda_adj_fy,
    (e."EBIT (FY)" - e."EBIT (-1FY)") / NULLIF(ABS(e."EBIT (-1FY)"), 0) *
    100                                                                                                AS ebit_growth_yoy,
    (e."EBITDA (FY)" - e."EBITDA (-1FY)") / NULLIF(ABS(e."EBITDA (-1FY)"), 0) *
    100                                                                                                AS ebitda_growth_yoy_comp,
    e."EBIT (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                         AS ebit_margin_ltm,
    e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                       AS ebitda_margin_ltm,
    (CASE WHEN e."EBIT (FY)" > 0 THEN 1 ELSE 0 END + CASE WHEN e."EBIT (-1FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBIT (-2FY)" > 0 THEN 1 ELSE 0 END + CASE WHEN e."EBIT (-3FY)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."EBIT (-4FY)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS ebit_positive_years,
    (CASE WHEN e."EBITDA (FY)" > 0 THEN 1 ELSE 0 END + CASE WHEN e."EBITDA (-1FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBITDA (-2FY)" > 0 THEN 1 ELSE 0 END + CASE WHEN e."EBITDA (-3FY)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."EBITDA (-4FY)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS ebitda_positive_years,
    (e."EBIT (FQ)" - e."EBIT (-1FQFQ)") / NULLIF(ABS(e."EBIT (-1FQFQ)"), 0) *
    100                                                                                                AS ebit_qoq_growth,
    (e."EBITDA (FQ)" - e."EBITDA (-1FQFQ)") / NULLIF(ABS(e."EBITDA (-1FQFQ)"), 0) *
    100                                                                                                AS ebitda_qoq_growth,
    CASE
        WHEN e."EBIT (-3FY)" > 0 AND e."EBIT (FY)" > 0
            THEN (POWER(e."EBIT (FY)" / NULLIF(e."EBIT (-3FY)", 0), 1.0 / 3.0) - 1) * 100 END          AS ebit_cagr_3y,
    CASE
        WHEN e."EBITDA (-3FY)" > 0 AND e."EBITDA (FY)" > 0
            THEN (POWER(e."EBITDA (FY)" / NULLIF(e."EBITDA (-3FY)", 0), 1.0 / 3.0) - 1) *
                 100 END                                                                               AS ebitda_cagr_3y,
    e."EBIT (LTM)" / NULLIF(e."EBIT (5YAVGLTM)", 0)                                                    AS ebit_vs_5y_avg,
    e."EBITDA (LTM)" / NULLIF(e."EBITDA (5YAVGLTM)", 0)                                                AS ebitda_vs_5y_avg,

    -- Net Income Comprehensive
    e."Net Income - (IS) (FQ)"                                                                         AS net_income_is_fq,
    e."Net Income - (IS) (LTM)"                                                                        AS net_income_is_ltm,
    e."Net Income - (IS) (FY)"                                                                         AS net_income_is_fy,
    e."Net Income/Adj. (LTM)"                                                                          AS net_income_adj_ltm,
    e."Normalized Net Income (LTM)"                                                                    AS normalized_ni_ltm,
    e."Net Income - (IS) (-1FQFQ)"                                                                     AS net_income_is_1fqfq,
    e."Net Income - (IS) (-2FQFQ)"                                                                     AS net_income_is_2fqfq,
    e."Net Income - (IS) (-3FQFQ)"                                                                     AS net_income_is_3fqfq,
    e."Net Income - (IS) (-4FQFQ)"                                                                     AS net_income_is_4fqfq,
    e."Net Income - (IS) (-1FY)"                                                                       AS net_income_is_1fy,
    e."Net Income - (IS) (-2FY)"                                                                       AS net_income_is_2fy,
    e."Net Income - (IS) (-3FY)"                                                                       AS net_income_is_3fy,
    e."Net Income - (IS) (-4FY)"                                                                       AS net_income_is_4fy,
    e."Net Income - (IS) (5YAVGFQ)"                                                                    AS net_income_is_5yavgfq,
    e."Net Income - (IS) (5YAVGLTM)"                                                                   AS net_income_is_5yavgltm,
    e."Normalized Net Income (5YAVGFQ)"                                                                AS normalized_ni_5yavgfq,
    e."Normalized Net Income (5YAVGLTM)"                                                               AS normalized_ni_5yavgltm,
    pct_change(e."Net Income - (IS) (FY)", e."Net Income - (IS) (-1FY)")                               AS net_income_growth_yoy,
    e."Net Income Margin % (LTM)"                                                                      AS net_income_margin_ltm,
    safe_divide(e."Net Income/Adj. (LTM)", e."Net Income - (IS) (LTM)")                                AS ni_adjustment_ratio,
    (CASE WHEN e."Net Income - (IS) (FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Net Income - (IS) (-1FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Net Income - (IS) (-2FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Net Income - (IS) (-3FY)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."Net Income - (IS) (-4FY)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS net_income_positive_years,
    clamp_score(
            50 +
            (CASE WHEN e."Net Income - (IS) (FY)" > 0 THEN 10 ELSE -10 END) +
            (CASE WHEN e."Net Income - (IS) (-1FY)" > 0 THEN 5 ELSE -5 END) +
            (CASE WHEN e."Net Income - (IS) (-2FY)" > 0 THEN 5 ELSE -5 END) +
            (CASE
                 WHEN ABS(safe_divide(e."Net Income/Adj. (LTM)" - e."Net Income - (IS) (LTM)",
                                      e."Net Income - (IS) (LTM)")) < 0.10 THEN 15
                 ELSE -15 END) +
            (CASE WHEN e."Net Income - (IS) (FY)" > e."Net Income - (IS) (-1FY)" THEN 10 ELSE -5 END) +
            (CASE WHEN e."Net Income - (IS) (-1FY)" > e."Net Income - (IS) (-2FY)" THEN 5 ELSE -5 END)
    )                                                                                                  AS earnings_quality_composite,
    pct_change(e."Net Income - (IS) (FQ)",
               e."Net Income - (IS) (-1FQFQ)")                                                         AS net_income_qoq_growth,
    pct_change(e."Net Income - (IS) (FQ)",
               e."Net Income - (IS) (-4FQFQ)")                                                         AS net_income_yoy_quarterly,
    safe_divide(e."Net Income - (IS) (LTM)",
                e."Net Income - (IS) (5YAVGLTM)")                                                      AS net_income_vs_5y_avg,
    safe_divide(e."Normalized Net Income (LTM)",
                e."Normalized Net Income (5YAVGLTM)")                                                  AS normalized_ni_vs_5y_avg,

    -- Revenue features
    e."Revenues - Est Avg (FY1E)"                                                                      AS revenue_est_avg_fy1e,
    e."Revenues - Est Med (FY1E)"                                                                      AS revenue_est_med_fy1e,
    e."Revenues - Est Avg (NTM)"                                                                       AS revenue_est_avg_ntm,
    e."Revenues - Est Med (NTM)"                                                                       AS revenue_est_med_ntm,
    safe_divide(e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)", e."Revenues - Est Med (FY1E)") * 100
                                                                                                       AS revenue_avg_med_diff_pct,
    clamp_score(100 - ABS(safe_divide(e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)",
                                      e."Revenues - Est Med (FY1E)") * 100) *
                      2)                                                                               AS revenue_consensus_strength,
    safe_divide(e."Revenues - Est Avg (FY1E)", e."Total Revenues (LTM)")                               AS revenue_vs_current,
    e."Total Revenues (FQ)"                                                                            AS revenue_fq,
    e."Total Revenues (FY)"                                                                            AS revenue_fy,
    e."Total Revenues (LTM)"                                                                           AS revenue_ltm,
    e."Total Revenues (5YAVGLTM)"                                                                      AS revenue_5y_avg,
    pct_change(e."Total Revenues (FY)", e."Total Revenues (-1FY)")                                     AS revenue_yoy_growth,
    safe_divide(e."Total Revenues (LTM)", e."Total Revenues (5YAVGLTM)")                               AS revenue_vs_5y_avg_qtr,
    safe_divide(e."Total Revenues (LTM)", e."Total Revenues (FY)")                                     AS revenue_ltm_vs_fy,
    safe_divide(e."Total Revenues (FQ)", e."Total Revenues (5YAVGFQ)")                                 AS revenue_fq_vs_5y_avg_fq,
    CASE
        WHEN e."Total Revenues (FY)" > e."Total Revenues (-1FY)" THEN 1
        ELSE 0 END                                                                                     AS revenue_growth_flag,
    e."Total Revenues (5YAVGFQ)"                                                                       AS revenue_5yavgfq,
    e."Total Revenues (5YAVGLTM)"                                                                      AS revenue_5yavgltm,
    calc_change_ratio(e."Total Revenues (LTM)", e."Total Revenues (-1FY)") *
    100                                                                                                AS revenue_momentum_temp,

    -- =========================================================================
    -- [18/18] UNUSUAL ITEMS FEATURES (vw_features_unusual_items)
    -- =========================================================================
    e."Other Unusual Items/Total (LTM)"                                                                AS other_unusual_items_ltm,
    COALESCE(e."Other Unusual Items/Total (LTM)", 0) + COALESCE(e."Impairment of Goodwill (LTM)", 0) +
    COALESCE(e."Asset Writedown (LTM)", 0) +
    COALESCE(e."Restructuring Charges (LTM)", 0)                                                       AS total_unusual_items,
    safe_divide(
            ABS(COALESCE(e."Other Unusual Items/Total (LTM)", 0) + COALESCE(e."Impairment of Goodwill (LTM)", 0) +
                COALESCE(e."Asset Writedown (LTM)", 0) + COALESCE(e."Restructuring Charges (LTM)", 0)),
            e."Total Revenues (LTM)") *
    100                                                                                                AS unusual_items_to_revenue,
    safe_divide(
            ABS(COALESCE(e."Other Unusual Items/Total (LTM)", 0) + COALESCE(e."Impairment of Goodwill (LTM)", 0) +
                COALESCE(e."Asset Writedown (LTM)", 0) + COALESCE(e."Restructuring Charges (LTM)", 0)),
            ABS(e."EBITDA (LTM)")) *
    100                                                                                                AS unusual_items_to_ebitda,
    CASE
        WHEN
            ABS(COALESCE(e."Other Unusual Items/Total (LTM)", 0)) + ABS(COALESCE(e."Impairment of Goodwill (LTM)", 0)) +
            ABS(COALESCE(e."Asset Writedown (LTM)", 0)) + ABS(COALESCE(e."Restructuring Charges (LTM)", 0)) > 0 THEN 1
        ELSE 0 END                                                                                     AS has_unusual_items_flag

FROM postgres.public.equities e;

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_all_stock_features_isin
    ON mv_all_stock_features (isin);

CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_ticker
    ON mv_all_stock_features (ticker);

CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_sector
    ON mv_all_stock_features (sector);

CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_country
    ON mv_all_stock_features (country);

-- =============================================================================
-- REFRESH FUNCTION
-- =============================================================================
CREATE OR REPLACE FUNCTION refresh_all_stock_features()
    RETURNS VOID
    LANGUAGE plpgsql
AS
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;
END;
$$;