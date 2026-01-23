-- =============================================================================
-- SQL Feature Registry for Finance ML Analytics Platform
-- Phase 9.3 Feature Engineering - PostgreSQL Implementation (OPTIMIZED)
-- =============================================================================
-- OPTIMIZATIONS APPLIED:
-- 1. STABLE modifier on all functions (enables query optimizer caching)
-- 2. Optional isin parameter for filtered access (uses idx_equities_isin)
-- 3. Materialized views for comprehensive functions
-- 4. PARALLEL SAFE where applicable
-- 5. Helper functions for common calculations (DRY principle)
-- =============================================================================

-- =============================================================================
-- HELPER FUNCTIONS: Extracted Common Calculations
-- =============================================================================

-- Safe division helper (avoids division by zero)
CREATE OR REPLACE FUNCTION safe_divide(numerator NUMERIC, denominator NUMERIC)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$ LANGUAGE SQL;

-- Percentage change helper
CREATE OR REPLACE FUNCTION pct_change(current_val NUMERIC, previous_val NUMERIC)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$ LANGUAGE SQL;

-- Momentum/change ratio helper (without percentage multiplier)
CREATE OR REPLACE FUNCTION calc_change_ratio(current_val NUMERIC, previous_val NUMERIC)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) AS result;
$$ LANGUAGE SQL;

-- Score clamping helper (constrains value between 0 and 100)
CREATE OR REPLACE FUNCTION clamp_score(val NUMERIC, min_val NUMERIC DEFAULT 0, max_val NUMERIC DEFAULT 100)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$ LANGUAGE SQL;

-- EMA crossover signal helper
CREATE OR REPLACE FUNCTION ema_crossover_signal(fast_ema NUMERIC, slow_ema NUMERIC)
    RETURNS INTEGER
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT CASE
           WHEN fast_ema > slow_ema THEN 1
           WHEN fast_ema < slow_ema THEN -1
           ELSE 0
           END AS result;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 1: VALUATION FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_valuation_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin            TEXT,
                p_e_ratio       NUMERIC,
                p_b_ratio       NUMERIC,
                ev_ebitda_ratio NUMERIC,
                ev_sales_ratio  NUMERIC,
                dividend_yield  NUMERIC,
                peg_ratio       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"            AS isin,
       "P/E (LTM)"       AS p_e_ratio,
       "P/B (LTM)"       AS p_b_ratio,
       "EV/EBITDA (LTM)" AS ev_ebitda_ratio,
       "EV/Sales (LTM)"  AS ev_sales_ratio,
       "Div Yield (LTM)" AS dividend_yield,
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0
               THEN safe_divide("P/E (LTM)", "Total Revenues/CAGR (5Y FY)")
           END           AS peg_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_valuation_timeseries_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                ev_sales_trend_1y          NUMERIC,
                ev_ebitda_momentum         NUMERIC,
                p_e_momentum_yoy           NUMERIC,
                p_e_momentum_qoq           NUMERIC,
                ev_sales_vs_3y_avg         NUMERIC,
                ev_ebitda_vs_3y_avg        NUMERIC,
                p_e_vs_3y_avg              NUMERIC,
                ev_sales_forward_discount  NUMERIC,
                ev_ebitda_forward_discount NUMERIC,
                p_e_forward_discount       NUMERIC,
                p_b_vs_5y_avg              NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                       AS isin,
       calc_change_ratio("EV/Sales (LTM)", "EV/Sales (-1FYLTM)")    AS ev_sales_trend_1y,
       calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (-1FYLTM)")  AS ev_ebitda_momentum,
       calc_change_ratio("P/E (LTM)", "P/E (-1FYLTM)")              AS p_e_momentum_yoy,
       calc_change_ratio("P/E (LTM)", "P/E (-1FQLTM)")              AS p_e_momentum_qoq,
       calc_change_ratio("EV/Sales (LTM)", "EV/Sales (3YAVGLTM)")   AS ev_sales_vs_3y_avg,
       calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)") AS ev_ebitda_vs_3y_avg,
       calc_change_ratio("P/E (LTM)", "P/E (3YAVGLTM)")             AS p_e_vs_3y_avg,
       calc_change_ratio("EV/Sales (NTM)", "EV/Sales (LTM)")        AS ev_sales_forward_discount,
       calc_change_ratio("EV/EBITDA (NTM)", "EV/EBITDA (LTM)")      AS ev_ebitda_forward_discount,
       calc_change_ratio("P/E (EST FY1)", "P/E (LTM)")              AS p_e_forward_discount,
       safe_divide("P/B (LTM)", "P/B (5YAVG)")                      AS p_b_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_extended_valuation_timeseries(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                ev_sales_qoq_1q          NUMERIC,
                ev_sales_qoq_2q          NUMERIC,
                ev_sales_qoq_3q          NUMERIC,
                ev_sales_qoq_4q          NUMERIC,
                p_e_vs_5y_avg            NUMERIC,
                p_e_percentile_proxy     NUMERIC,
                valuation_mean_reversion NUMERIC,
                ev_ebitda_qoq_trend      NUMERIC,
                p_b_momentum_yoy         NUMERIC,
                valuation_compression    NUMERIC,
                forward_pe_premium       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,
       ("EV/Sales (LTM)" - "EV/Sales (-1FQLTM)") / NULLIF("EV/Sales (-1FQLTM)", 0) AS ev_sales_qoq_1q,
       ("EV/Sales (-1FQLTM)" - "EV/Sales (-2FQLTM)") / NULLIF("EV/Sales (-2FQLTM)", 0)
                                                                                   AS ev_sales_qoq_2q,
       ("EV/Sales (-2FQLTM)" - "EV/Sales (-3FQLTM)") / NULLIF("EV/Sales (-3FQLTM)", 0)
                                                                                   AS ev_sales_qoq_3q,
       ("EV/Sales (-3FQLTM)" - "EV/Sales (-4FQLTM)") / NULLIF("EV/Sales (-4FQLTM)", 0)
                                                                                   AS ev_sales_qoq_4q,
       ("P/E (LTM)" - "P/E (5YAVGLTM)") / NULLIF("P/E (5YAVGLTM)", 0)              AS p_e_vs_5y_avg,
       CASE
           WHEN "P/E (LTM)" IS NOT NULL AND "P/E (3YAVGLTM)" IS NOT NULL
               THEN ("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF(ABS("P/E (3YAVGLTM)") * 0.5, 0)
           END                                                                     AS p_e_percentile_proxy,
       (("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF("P/E (3YAVGLTM)", 0) +
        ("EV/Sales (LTM)" - "EV/Sales (3YAVGLTM)") / NULLIF("EV/Sales (3YAVGLTM)", 0) +
        ("EV/EBITDA (LTM)" - "EV/EBITDA (3YAVGLTM)") / NULLIF("EV/EBITDA (3YAVGLTM)", 0)) / 3.0
                                                                                   AS valuation_mean_reversion,
       ("EV/EBITDA (LTM)" - "EV/EBITDA (-1FQLTM)") / NULLIF("EV/EBITDA (-1FQLTM)", 0)
                                                                                   AS ev_ebitda_qoq_trend,
       ("P/B (LTM)" - "P/B (-1FY)") / NULLIF("P/B (-1FY)", 0)                      AS p_b_momentum_yoy,
       (("P/E (LTM)" / NULLIF("P/E (3YAVGLTM)", 0)) +
        ("EV/EBITDA (LTM)" / NULLIF("EV/EBITDA (3YAVGLTM)", 0))) / 2.0 - 1.0       AS valuation_compression,
       ("P/E (EST FY1)" - "P/E (LTM)") / NULLIF(ABS("P/E (LTM)"), 0) * 100         AS forward_pe_premium
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 2: MOMENTUM & TECHNICAL FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_momentum_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                price_momentum_1m    NUMERIC,
                price_momentum_3m    NUMERIC,
                price_momentum_6m    NUMERIC,
                price_momentum_1y    NUMERIC,
                price_momentum_5d    NUMERIC,
                ema_crossover_20_50  INTEGER,
                ema_crossover_50_250 INTEGER,
                price_vs_ema_20d     NUMERIC,
                price_vs_ema_250d    NUMERIC,
                pct_off_52w_high     NUMERIC,
                pct_above_52w_low    NUMERIC,
                range_52w_position   NUMERIC,
                beta_momentum        NUMERIC,
                volatility_regime    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                           AS isin,
       pct_change("Last Price", "Price (1M Ago)")                       AS price_momentum_1m,
       pct_change("Last Price", "Price (3M Ago)")                       AS price_momentum_3m,
       pct_change("Last Price", "Price (6M Ago)")                       AS price_momentum_6m,
       pct_change("Last Price", "Price (1Y Ago)")                       AS price_momentum_1y,
       pct_change("Last Price", "Price (5D Ago)")                       AS price_momentum_5d,
       ema_crossover_signal("EMA (20D)", "EMA (50D)")                   AS ema_crossover_20_50,
       ema_crossover_signal("EMA (50D)", "EMA (250D)")                  AS ema_crossover_50_250,
       calc_change_ratio("Last Price", "EMA (20D)")                     AS price_vs_ema_20d,
       calc_change_ratio("Last Price", "EMA (250D)")                    AS price_vs_ema_250d,
       calc_change_ratio("52W High/Adj" - "Last Price", "52W High/Adj") AS pct_off_52w_high,
       calc_change_ratio("Last Price" - "52W Low/Adj", "52W Low/Adj")   AS pct_above_52w_low,
       clamp_score(safe_divide("Last Price" - "52W Low/Adj",
                               "52W High/Adj" - "52W Low/Adj"), 0, 1)   AS range_52w_position,
       "Beta (1Y)" - "Beta (5Y)"                                        AS beta_momentum,
       safe_divide("Volatility (1M)", "Volatility (1Y)")                AS volatility_regime
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_technical_analysis_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                ema_slope_20d             NUMERIC,
                ema_trend_consistency     INTEGER,
                price_vs_ema_100d         NUMERIC,
                near_52w_high_flag        INTEGER,
                near_52w_low_flag         INTEGER,
                volume_momentum_score     NUMERIC,
                breakout_signal           INTEGER,
                high_volume_flag          INTEGER,
                low_volume_flag           INTEGER,
                volatility_compression    NUMERIC,
                volatility_term_structure NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                        AS isin,
       ("EMA (20D)" - "EMA (50D)") / NULLIF("EMA (50D)", 0)          AS ema_slope_20d,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" AND "EMA (50D)" > "EMA (100D)"
               AND "EMA (100D)" > "EMA (250D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" AND "EMA (50D)" < "EMA (100D)"
               AND "EMA (100D)" < "EMA (250D)" THEN -1
           ELSE 0
           END                                                       AS ema_trend_consistency,
       ("Last Price" - "EMA (100D)") / NULLIF("EMA (100D)", 0) * 100 AS price_vs_ema_100d,
       CASE
           WHEN ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_high_flag,
       CASE
           WHEN ("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_low_flag,
       "Rel. Volume" * "Price Chg. % (1M)"                           AS volume_momentum_score,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)"
               AND ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS breakout_signal,
       CASE WHEN "Rel. Volume" > 1.5 THEN 1 ELSE 0 END               AS high_volume_flag,
       CASE WHEN "Rel. Volume" < 0.5 THEN 1 ELSE 0 END               AS low_volume_flag,
       "Volatility (1Y)" - "Volatility (1M)"                         AS volatility_compression,
       "Volatility (3M)" - "Volatility (6M)"                         AS volatility_term_structure
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 3: PROFITABILITY FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_profitability_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                roe                  NUMERIC,
                roa                  NUMERIC,
                gross_margin_pct     NUMERIC,
                operating_margin_pct NUMERIC,
                net_margin_pct       NUMERIC,
                ebitda_margin_pct    NUMERIC,
                roic                 NUMERIC,
                rnd_intensity        NUMERIC,
                equity_multiplier    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                 AS isin,
       "Return On Equity % (LTM)"                                                             AS roe,
       "Return on Assets (ROA) % (LTM)"                                                       AS roa,
       "Gross Profit Margin % (LTM)"                                                          AS gross_margin_pct,
       "Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                     AS operating_margin_pct,
       "Net Income Margin % (LTM)"                                                            AS net_margin_pct,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                               AS ebitda_margin_pct,
       "Net Income - (IS) (LTM)" / NULLIF("Total Equity (LTM)" + "Total Debt (LTM)", 0) * 100 AS roic,
       "R&D Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)                               AS rnd_intensity,
       "Total Assets (LTM)" / NULLIF("Total Equity (LTM)", 0)                                 AS equity_multiplier
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_margin_trends(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                   TEXT,
                gross_margin_trend_yoy NUMERIC,
                operating_margin_trend NUMERIC,
                net_margin_trend_yoy   NUMERIC,
                ebitda_margin_trend    NUMERIC,
                margin_expansion_flag  INTEGER,
                margin_stability_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                               AS isin,
       ("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)")       AS gross_margin_trend_yoy,
       (("Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("Operating Income (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100 AS operating_margin_trend,
       ("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")           AS net_margin_trend_yoy,
       (("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100           AS ebitda_margin_trend,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               AND "Net Income Margin % (LTM)" > "Net Income Margin % (FY)"
               AND ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) >
                   ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))
               THEN 1
           ELSE 0
           END                                                              AS margin_expansion_flag,
       GREATEST(0, LEAST(100,
                         100 - (ABS("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)") +
                                ABS("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")) / 2
                   ))                                                       AS margin_stability_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 4: QUALITY & RISK FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_quality_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                has_goodwill_impairment     INTEGER,
                has_asset_writedown         INTEGER,
                has_restructuring           INTEGER,
                goodwill_to_assets_pct      NUMERIC,
                intangible_intensity        NUMERIC,
                exceptional_items_to_ebitda NUMERIC,
                altman_z_score              NUMERIC,
                altman_z_trend              NUMERIC,
                current_ratio               NUMERIC,
                quick_ratio                 NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                            AS isin,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END                                   AS has_goodwill_impairment,
       CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 1 ELSE 0 END                                          AS has_asset_writedown,
       CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 1 ELSE 0 END                                    AS has_restructuring,
       "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                                          AS goodwill_to_assets_pct,
       "Gross Intangible Assets (LTM)" / NULLIF("Total Assets (LTM)", 0)                                 AS intangible_intensity,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)                                                                    AS exceptional_items_to_ebitda,
       "Altman Z-Score (LTM)"                                                                            AS altman_z_score,
       "Altman Z-Score (FY)" - "Altman Z-Score (LTM)"                                                    AS altman_z_trend,
       "Current Ratio (LTM)"                                                                             AS current_ratio,
       ("Total Current Assets (LTM)" - "Inventory (LTM)") / NULLIF("Total Current Liabilities (LTM)", 0) AS quick_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_financial_distress_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                distress_risk_score      NUMERIC,
                liquidity_stress_score   NUMERIC,
                working_capital_trend    NUMERIC,
                cash_runway_months       NUMERIC,
                combined_distress_score  NUMERIC,
                wc_deteriorating_flag    INTEGER,
                retained_earnings_growth NUMERIC,
                accumulated_deficit_flag INTEGER,
                adequate_cash_buffer     INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                            AS isin,
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100))) AS distress_risk_score,
       CASE
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
           ELSE 0.0
           END                                                                           AS liquidity_stress_score,
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0)                                            AS working_capital_trend,
       "Cash And Equivalents (FQ)" /
       NULLIF("Total Operating Expenses (LTM)" / 12.0, 0)                                AS cash_runway_months,
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 70) +
                         (100 - CASE
                                    WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
                                    WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
                                    ELSE 0.0
                             END) * 0.30))                                               AS combined_distress_score,
       CASE
           WHEN ("Working Capital (FQ)" - "Working Capital (FY)") /
                NULLIF(ABS("Working Capital (FY)"), 0) < -0.2
               THEN 1
           ELSE 0
           END                                                                           AS wc_deteriorating_flag,
       ("Retained Earnings (FQ)" - "Retained Earnings (FY)") /
       NULLIF(ABS("Retained Earnings (FY)"), 0)                                          AS retained_earnings_growth,
       CASE WHEN "Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END                          AS accumulated_deficit_flag,
       CASE
           WHEN "Cash And Equivalents (FQ)" /
                NULLIF("Total Operating Expenses (LTM)" / 12.0, 0) > 6
               THEN 1
           ELSE 0
           END                                                                           AS adequate_cash_buffer
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_accounting_quality_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                goodwill_change_rate        NUMERIC,
                restructuring_intensity     NUMERIC,
                exceptional_items_frequency INTEGER,
                merger_impact_ratio         NUMERIC,
                non_operating_income_share  NUMERIC,
                asset_sale_boost            INTEGER,
                accounting_quality_score    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                AS isin,
       ("Goodwill (LTM)" - "Goodwill (-1FY)") / NULLIF("Goodwill (-1FY)", 0) AS goodwill_change_rate,
       "Restructuring Charges (LTM)" / NULLIF("Total Assets (LTM)", 0)       AS restructuring_intensity,
       (CASE WHEN ABS("Impairment of Goodwill (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Asset Writedown (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Restructuring Charges (FQ)") > 0 THEN 1 ELSE 0 END)   AS exceptional_items_frequency,
       "Merger & Restructuring Charges (LTM)" / NULLIF("Market Cap", 0)      AS merger_impact_ratio,
       "Interest Income On Investments (LTM)" / NULLIF(ABS("Net Income - (IS) (LTM)"), 0)
                                                                             AS non_operating_income_share,
       CASE WHEN "Gain (Loss) On Sale Of Assets (LTM)" > 0 THEN 1 ELSE 0 END AS asset_sale_boost,
       GREATEST(0, LEAST(100,
                         100 -
                         (CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 25 ELSE 0 END) -
                         (CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 10 ELSE 0 END) -
                         (CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 15 ELSE 0 END) -
                         (CASE WHEN "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.30 THEN 15 ELSE 0 END) -
                         (CASE
                              WHEN (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
                                    ABS("Restructuring Charges (LTM)")) /
                                   NULLIF(ABS("Net Income - (IS) (LTM)"), 0) > 0.10 THEN 15
                              ELSE 0 END)
                   ))                                                        AS accounting_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 5: LEVERAGE & LIQUIDITY FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_leverage_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                debt_to_equity        NUMERIC,
                debt_to_assets        NUMERIC,
                equity_ratio          NUMERIC,
                interest_coverage     NUMERIC,
                current_ratio         NUMERIC,
                cash_ratio            NUMERIC,
                working_capital_ratio NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,
       "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0)                        AS debt_to_equity,
       "Total Debt (LTM)" / NULLIF("Total Assets (LTM)", 0)                        AS debt_to_assets,
       "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS equity_ratio,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                    AS interest_coverage,
       "Current Ratio (LTM)"                                                       AS current_ratio,
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) AS cash_ratio,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0)                   AS working_capital_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_efficiency_ratios(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                asset_turnover        NUMERIC,
                inventory_turnover    NUMERIC,
                receivables_days      NUMERIC,
                working_capital_turns NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                        AS isin,
       "Total Revenues (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS asset_turnover,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)                       AS inventory_turnover,
       ("Accounts Receivable/Total (FY)" / NULLIF("Total Revenues (FY)" / 365.0, 0)) AS receivables_days,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turns
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_balance_sheet_dynamics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                cash_to_assets_pct        NUMERIC,
                cash_change_qoq           NUMERIC,
                cash_vs_5y_avg            NUMERIC,
                inventory_change_yoy      NUMERIC,
                inventory_vs_5y_avg       NUMERIC,
                receivables_change_yoy    NUMERIC,
                receivables_vs_5y_avg     NUMERIC,
                working_capital_vs_5y_avg NUMERIC,
                retained_earnings_vs_5y   NUMERIC,
                intangibles_growth_flag   INTEGER,
                asset_quality_score       NUMERIC,
                balance_sheet_strength    NUMERIC,
                debt_maturity_risk        NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                    AS isin,
       "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100      AS cash_to_assets_pct,
       ("Cash And Equivalents (FQ)" - "Cash And Equivalents (FY)") /
       NULLIF(ABS("Cash And Equivalents (FY)"), 0)                               AS cash_change_qoq,
       "Cash And Equivalents (FQ)" / NULLIF("Cash And Equivalents (5YAVGFQ)", 0) AS cash_vs_5y_avg,
       ("Inventory (FY)" - "Inventory (FQ)") / NULLIF(ABS("Inventory (FQ)"), 0)  AS inventory_change_yoy,
       "Inventory (FQ)" / NULLIF("Inventory (5YAVGFQ)", 0)                       AS inventory_vs_5y_avg,
       ("Accounts Receivable/Total (FY)" - "Accounts Receivable/Total (-1FY)") /
       NULLIF(ABS("Accounts Receivable/Total (-1FY)"), 0)                        AS receivables_change_yoy,
       "Accounts Receivable/Total (FY)" / NULLIF("Accounts Receivable/Total (5YAVGFQ)", 0)
                                                                                 AS receivables_vs_5y_avg,
       "Working Capital (FQ)" / NULLIF("Working Capital (5YAVGFY)", 0)           AS working_capital_vs_5y_avg,
       "Retained Earnings (FQ)" / NULLIF("Retained Earnings (5YAVGFQ)", 0)       AS retained_earnings_vs_5y,
       CASE
           WHEN "Gross Intangible Assets (FY)" / NULLIF("Gross Intangible Assets (5YAVGFQ)", 0) > 1.5
               THEN 1
           ELSE 0
           END                                                                   AS intangibles_growth_flag,
       GREATEST(0, LEAST(100,
                         50 + ("Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100) -
                         ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100)
                   ))                                                            AS asset_quality_score,
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.10 THEN 25
                              ELSE 0 END) +
                         (CASE WHEN "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.40 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Working Capital (LTM)" > 0 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END)
                   ))                                                            AS balance_sheet_strength,
       "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0)                            AS debt_maturity_risk
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;


-- =============================================================================
-- SECTION 6: ANALYST SENTIMENT FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_sentiment_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                analyst_bullish_pct       NUMERIC,
                analyst_bearish_pct       NUMERIC,
                analyst_neutral_pct       NUMERIC, -- NEW: Hold ratings percentage
                analyst_conviction        NUMERIC,
                upside_potential          NUMERIC,
                price_target_spread_pct   NUMERIC,
                price_target_revision_1m  NUMERIC,
                price_target_revision_3m  NUMERIC,
                eps_revision_momentum     NUMERIC,
                analyst_rating_normalized NUMERIC,
                analyst_coverage_quality  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN ("# Strong Buys Ratings" + "# Buys Ratings") /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_bullish_pct,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN ("# Sell Ratings" + "# Strong Sell Ratings") /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_bearish_pct,
       -- NEW: Neutral sentiment (Hold ratings)
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN "# Hold Ratings" /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_neutral_pct,
       ABS(
               CASE
                   WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                         "# Sell Ratings" + "# Strong Sell Ratings") > 0
                       THEN (("# Strong Buys Ratings" + "# Buys Ratings") -
                             ("# Sell Ratings" + "# Strong Sell Ratings")) /
                            NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                                   "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
                   END
       )                                                                        AS analyst_conviction,
       ("Price Target - Median" - "Last Price") / NULLIF("Last Price", 0) * 100 AS upside_potential,
       ("Price Target - High" - "Price Target - Low") / NULLIF("Price Target - Median", 0) *
       100                                                                      AS price_target_spread_pct,
       ("Price Target" - "Price Target (1M Ago)") /
       NULLIF("Price Target (1M Ago)", 0)                                       AS price_target_revision_1m,
       ("Price Target" - "Price Target (3M Ago)") /
       NULLIF("Price Target (3M Ago)", 0)                                       AS price_target_revision_3m,
       COALESCE("EPS Est Avg Rev % (FY1E - 1W)", 0) * 0.30 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1M)", 0) * 0.25 +
       COALESCE("EPS Est Avg Rev % (FY1E - 3M)", 0) * 0.20 +
       COALESCE("EPS Est Avg Rev % (FY1E - 6M)", 0) * 0.15 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1Y)", 0) *
       0.10                                                                     AS eps_revision_momentum,
       ("Analyst Rating" - 1) * 25                                              AS analyst_rating_normalized,
       "Price Target - #" / NULLIF(LN(1 + "Market Cap"), 0)                     AS analyst_coverage_quality
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_price_target_dynamics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                pt_momentum_1w             NUMERIC,
                pt_momentum_1m             NUMERIC,
                pt_momentum_3m             NUMERIC,
                pt_momentum_6m             NUMERIC,
                pt_momentum_1y             NUMERIC,
                pt_median_momentum_1m      NUMERIC,
                pt_median_momentum_3m      NUMERIC,
                pt_acceleration_short      NUMERIC,
                pt_acceleration_long       NUMERIC,
                pt_consensus_convergence   NUMERIC,
                analyst_coverage_change_1m INTEGER,
                analyst_coverage_change_3m INTEGER,
                analyst_coverage_change_1y INTEGER,
                pt_vs_price_momentum       NUMERIC,
                analyst_coverage_trend     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                            AS isin,
       ("Price Target" - "Price Target (1W Ago)") / NULLIF("Price Target (1W Ago)", 0)   AS pt_momentum_1w,
       ("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)   AS pt_momentum_1m,
       ("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)   AS pt_momentum_3m,
       ("Price Target" - "Price Target (6M Ago)") / NULLIF("Price Target (6M Ago)", 0)   AS pt_momentum_6m,
       ("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)   AS pt_momentum_1y,
       ("Price Target - Median" - "Price Target - Median (1M Ago)") /
       NULLIF("Price Target - Median (1M Ago)", 0)                                       AS pt_median_momentum_1m,
       ("Price Target - Median" - "Price Target - Median (3M Ago)") /
       NULLIF("Price Target - Median (3M Ago)", 0)                                       AS pt_median_momentum_3m,
       (("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)) -
       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) AS pt_acceleration_short,
       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) -
       (("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)) AS pt_acceleration_long,
       (("Price Target - High (3M Ago)" - "Price Target - Low (3M Ago)") /
        NULLIF("Price Target - Median (3M Ago)", 0)) -
       (("Price Target - High" - "Price Target - Low") /
        NULLIF("Price Target - Median", 0))                                              AS pt_consensus_convergence,
       ("Price Target - #" - "Price Target - # (1M Ago)")::INTEGER                       AS analyst_coverage_change_1m,
       ("Price Target - #" - "Price Target - # (3M Ago)")::INTEGER                       AS analyst_coverage_change_3m,
       ("Price Target - #" - "Price Target - # (1Y Ago)")::INTEGER                       AS analyst_coverage_change_1y,
       (("Price Target" / NULLIF("Last Price", 0)) -
        ("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0))) /
       NULLIF(("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0)), 0)                AS pt_vs_price_momentum,
       (COALESCE("Price Target - #" - "Price Target - # (1M Ago)", 0) * 0.40 +
        COALESCE("Price Target - #" - "Price Target - # (3M Ago)", 0) * 0.35 +
        COALESCE("Price Target - #" - "Price Target - # (6M Ago)", 0) * 0.25) /
       NULLIF("Price Target - #"::NUMERIC, 0)                                            AS analyst_coverage_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 7: EARNINGS FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_earnings_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                eps_surprise_pct        NUMERIC,
                revenue_surprise_pct    NUMERIC,
                eps_adjustment_ratio    NUMERIC,
                gaap_adj_eps_gap_pct    NUMERIC,
                ebitda_adjustment_ratio NUMERIC,
                eps_quarterly_trend     NUMERIC,
                eps_yoy_growth          NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                AS isin,
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS/Adj. (LTM)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS eps_surprise_pct,
       CASE
           WHEN ABS("Revenues - Est Avg (FY1E)") > 0
               THEN ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
                    NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100
           END                                               AS revenue_surprise_pct,
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0) AS eps_adjustment_ratio,
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS gaap_adj_eps_gap_pct,
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0)       AS ebitda_adjustment_ratio,
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0)
           END                                               AS eps_quarterly_trend,
       CASE
           WHEN ABS("Net EPS - Basic (-1FY)") > 0
               THEN ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
                    NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100
           END                                               AS eps_yoy_growth
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_eps_trajectory_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                eps_qoq_growth        NUMERIC,
                eps_yoy_quarterly     NUMERIC,
                eps_positive_streak   INTEGER,
                eps_cagr_3y           NUMERIC,
                eps_cagr_5y           NUMERIC,
                eps_growth_accel      NUMERIC,
                eps_vs_5y_avg         NUMERIC,
                eps_improvement_count INTEGER,
                eps_trajectory_score  NUMERIC,
                eps_stability         NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                AS isin,
       CASE
           WHEN ABS("Net EPS - Basic (-1FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-1FQFQ)"), 0) * 100
           END                                                               AS eps_qoq_growth,
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0) * 100
           END                                                               AS eps_yoy_quarterly,
       (CASE WHEN "Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_streak,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                               AS eps_cagr_3y,
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                               AS eps_cagr_5y,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (-5FY)" > 0
               AND "Net EPS - Basic (FY)" > 0
               THEN ((POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                     (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
           END                                                               AS eps_growth_accel,
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN ("Net EPS - Basic (FY)" -
                     (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                       "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0)) /
                    NULLIF(ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                                "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0), 0) * 100
           END                                                               AS eps_vs_5y_avg,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                             AS eps_improvement_count,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                     AS eps_trajectory_score,
       NULL::NUMERIC                                                         AS eps_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_gaap_adjusted_analytics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                                TEXT,
                -- EPS Adjustments
                eps_adjustment_spread_ltm           NUMERIC,
                eps_adjustment_spread_fy            NUMERIC,
                eps_adjustment_spread_1fy           NUMERIC,
                eps_adjustment_spread_fq            NUMERIC,
                eps_adjustment_spread_1fqfq         NUMERIC,
                eps_adjustment_spread_2fqfq         NUMERIC,
                eps_adjustment_spread_3fqfq         NUMERIC,
                eps_adjustment_spread_4fqfq         NUMERIC,
                eps_adjustment_spread_2fy           NUMERIC,
                eps_adjustment_spread_3fy           NUMERIC,
                eps_adjustment_spread_4fy           NUMERIC,
                eps_adjustment_pct                  NUMERIC,
                -- Net Income Adjustments
                net_income_adjustment_ratio_ltm     NUMERIC,
                net_income_adjustment_ratio_fy      NUMERIC,
                net_income_adjustment_ratio_1fy     NUMERIC,
                net_income_adjustment_ratio_fq      NUMERIC,
                net_income_adjustment_ratio_5yavgfq NUMERIC,
                net_income_adjustment_ratio_1fqfq   NUMERIC,
                net_income_adjustment_ratio_2fqfq   NUMERIC,
                net_income_adjustment_ratio_3fqfq   NUMERIC,
                net_income_adjustment_ratio_4fqfq   NUMERIC,
                net_income_adjustment_ratio_2fy     NUMERIC,
                net_income_adjustment_ratio_3fy     NUMERIC,
                net_income_adjustment_ratio_4fy     NUMERIC,
                net_income_adjustment_pct           NUMERIC,
                -- EBITDA Adjustments
                ebitda_adjustment_pct_ltm           NUMERIC,
                ebitda_adjustment_pct_fy            NUMERIC,
                ebitda_adjustment_pct_1fy           NUMERIC,
                ebitda_adjustment_pct_fq            NUMERIC,
                ebitda_adjustment_pct_1fqfq         NUMERIC,
                ebitda_adjustment_pct_2fqfq         NUMERIC,
                ebitda_adjustment_pct_3fqfq         NUMERIC,
                ebitda_adjustment_pct_4fqfq         NUMERIC,
                ebitda_adjustment_pct_2fy           NUMERIC,
                ebitda_adjustment_pct_3fy           NUMERIC,
                ebitda_adjustment_pct_4fy           NUMERIC,
                -- EBIT Adjustments
                ebit_adjustment_pct_ltm             NUMERIC,
                ebit_adjustment_pct_fy              NUMERIC,
                ebit_adjustment_pct_1fy             NUMERIC,
                ebit_adjustment_pct_fq              NUMERIC,
                ebit_adjustment_pct_1fqfq           NUMERIC,
                ebit_adjustment_pct_2fqfq           NUMERIC,
                ebit_adjustment_pct_3fqfq           NUMERIC,
                ebit_adjustment_pct_4fqfq           NUMERIC,
                ebit_adjustment_pct_2fy             NUMERIC,
                ebit_adjustment_pct_3fy             NUMERIC,
                ebit_adjustment_pct_4fy             NUMERIC,
                -- Quality Scores
                earnings_quality_score              NUMERIC,
                earnings_quality_warning            INTEGER,
                forward_eps_gaap_adj_spread         NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                       AS isin,
       -- EPS Adjustment Spreads (EPS/Adj. - Net EPS - Basic)
       "EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)"                                   AS eps_adjustment_spread_ltm,
       "EPS/Adj. (FY)" - "Net EPS - Basic (FY)"                                     AS eps_adjustment_spread_fy,
       "EPS/Adj. (-1FY)" - "Net EPS - Basic (-1FY)"                                 AS eps_adjustment_spread_1fy,
       "EPS/Adj. (FQ)" - "Net EPS - Basic (FQ)"                                     AS eps_adjustment_spread_fq,
       "EPS/Adj. (-1FQFQ)" - "Net EPS - Basic (-1FQFQ)"                             AS eps_adjustment_spread_1fqfq,
       "EPS/Adj. (-2FQFQ)" - "Net EPS - Basic (-2FQFQ)"                             AS eps_adjustment_spread_2fqfq,
       "EPS/Adj. (-3FQFQ)" - "Net EPS - Basic (-3FQFQ)"                             AS eps_adjustment_spread_3fqfq,
       "EPS/Adj. (-4FQFQ)" - "Net EPS - Basic (-4FQFQ)"                             AS eps_adjustment_spread_4fqfq,
       "EPS/Adj. (-2FY)" - "Net EPS - Basic (-2FY)"                                 AS eps_adjustment_spread_2fy,
       "EPS/Adj. (-3FY)" - "Net EPS - Basic (-3FY)"                                 AS eps_adjustment_spread_3fy,
       "EPS/Adj. (-4FY)" - "Net EPS - Basic (-4FY)"                                 AS eps_adjustment_spread_4fy,
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
       NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100                                AS eps_adjustment_pct,

       -- Net Income Adjustment Ratios (Net Income/Adj. / Net Income - (IS))
       "Net Income/Adj. (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS net_income_adjustment_ratio_ltm,
       "Net Income/Adj. (FY)" / NULLIF("Net Income - (IS) (FY)", 0)                 AS net_income_adjustment_ratio_fy,
       "Net Income/Adj. (-1FY)" / NULLIF("Net Income - (IS) (-1FY)", 0)             AS net_income_adjustment_ratio_1fy,
       "Net Income/Adj. (FQ)" / NULLIF("Net Income - (IS) (FQ)", 0)                 AS net_income_adjustment_ratio_fq,
       "Net Income/Adj. (5YAVGFQ)" / NULLIF("Net Income - (IS) (5YAVGFQ)", 0)       AS net_income_adjustment_ratio_5yavgfq,
       "Net Income/Adj. (-1FQFQ)" / NULLIF("Net Income - (IS) (-1FQFQ)", 0)         AS net_income_adjustment_ratio_1fqfq,
       "Net Income/Adj. (-2FQFQ)" / NULLIF("Net Income - (IS) (-2FQFQ)", 0)         AS net_income_adjustment_ratio_2fqfq,
       "Net Income/Adj. (-3FQFQ)" / NULLIF("Net Income - (IS) (-3FQFQ)", 0)         AS net_income_adjustment_ratio_3fqfq,
       "Net Income/Adj. (-4FQFQ)" / NULLIF("Net Income - (IS) (-4FQFQ)", 0)         AS net_income_adjustment_ratio_4fqfq,
       "Net Income/Adj. (-2FY)" / NULLIF("Net Income - (IS) (-2FY)", 0)             AS net_income_adjustment_ratio_2fy,
       "Net Income/Adj. (-3FY)" / NULLIF("Net Income - (IS) (-3FY)", 0)             AS net_income_adjustment_ratio_3fy,
       "Net Income/Adj. (-4FY)" / NULLIF("Net Income - (IS) (-4FY)", 0)             AS net_income_adjustment_ratio_4fy,
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) *
       100                                                                          AS net_income_adjustment_pct,

       -- EBITDA Adjustment Percentages (EBITDA/Adj. - EBITDA) / |EBITDA| * 100
       ("EBITDA/Adj. (LTM)" - "EBITDA (LTM)") / NULLIF(ABS("EBITDA (LTM)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_ltm,
       ("EBITDA/Adj. (FY)" - "EBITDA (FY)") / NULLIF(ABS("EBITDA (FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_fy,
       ("EBITDA/Adj. (-1FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_1fy,
       ("EBITDA/Adj. (FQ)" - "EBITDA (FQ)") / NULLIF(ABS("EBITDA (FQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_fq,
       ("EBITDA/Adj. (-1FQFQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_1fqfq,
       ("EBITDA/Adj. (-2FQFQ)" - "EBITDA (-2FQFQ)") / NULLIF(ABS("EBITDA (-2FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_2fqfq,
       ("EBITDA/Adj. (-3FQFQ)" - "EBITDA (-3FQFQ)") / NULLIF(ABS("EBITDA (-3FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_3fqfq,
       ("EBITDA/Adj. (-4FQFQ)" - "EBITDA (-4FQFQ)") / NULLIF(ABS("EBITDA (-4FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_4fqfq,
       ("EBITDA/Adj. (-2FY)" - "EBITDA (-2FY)") / NULLIF(ABS("EBITDA (-2FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_2fy,
       ("EBITDA/Adj. (-3FY)" - "EBITDA (-3FY)") / NULLIF(ABS("EBITDA (-3FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_3fy,
       ("EBITDA/Adj. (-4FY)" - "EBITDA (-4FY)") / NULLIF(ABS("EBITDA (-4FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_4fy,

       -- EBIT Adjustment Percentages (EBIT/Adj. - EBIT) / |EBIT| * 100
       ("EBIT/Adj. (LTM)" - "EBIT (LTM)") / NULLIF(ABS("EBIT (LTM)"), 0) *
       100                                                                          AS ebit_adjustment_pct_ltm,
       ("EBIT/Adj. (FY)" - "EBIT (FY)") / NULLIF(ABS("EBIT (FY)"), 0) * 100         AS ebit_adjustment_pct_fy,
       ("EBIT/Adj. (-1FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_1fy,
       ("EBIT/Adj. (FQ)" - "EBIT (FQ)") / NULLIF(ABS("EBIT (FQ)"), 0) * 100         AS ebit_adjustment_pct_fq,
       ("EBIT/Adj. (-1FQFQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_1fqfq,
       ("EBIT/Adj. (-2FQFQ)" - "EBIT (-2FQFQ)") / NULLIF(ABS("EBIT (-2FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_2fqfq,
       ("EBIT/Adj. (-3FQFQ)" - "EBIT (-3FQFQ)") / NULLIF(ABS("EBIT (-3FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_3fqfq,
       ("EBIT/Adj. (-4FQFQ)" - "EBIT (-4FQFQ)") / NULLIF(ABS("EBIT (-4FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_4fqfq,
       ("EBIT/Adj. (-2FY)" - "EBIT (-2FY)") / NULLIF(ABS("EBIT (-2FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_2fy,
       ("EBIT/Adj. (-3FY)" - "EBIT (-3FY)") / NULLIF(ABS("EBIT (-3FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_3fy,
       ("EBIT/Adj. (-4FY)" - "EBIT (-4FY)") / NULLIF(ABS("EBIT (-4FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_4fy,

       -- Quality Scores (based on LTM EPS adjustment)
       GREATEST(0, LEAST(100,
                         100 - ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                                   NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100))) AS earnings_quality_score,
       CASE
           WHEN ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                    NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100) > 15
               THEN 1
           ELSE 0
           END                                                                      AS earnings_quality_warning,
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"                    AS forward_eps_gaap_adj_spread
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_gaap_revision_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                         TEXT,
                gaap_revision_momentum       NUMERIC,
                gaap_revision_1m             NUMERIC,
                gaap_revision_3m             NUMERIC,
                gaap_revision_6m             NUMERIC,
                gaap_revision_1y             NUMERIC,
                gaap_vs_norm_revision_spread NUMERIC,
                gaap_revision_acceleration   NUMERIC,
                gaap_positive_revision_flag  INTEGER,
                revision_quality_divergence  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) * 0.15                    AS gaap_revision_momentum,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)"                                        AS gaap_revision_1m,
       "EPS GAAP Est Avg Rev % (FY1E - 3M)"                                        AS gaap_revision_3m,
       "EPS GAAP Est Avg Rev % (FY1E - 6M)"                                        AS gaap_revision_6m,
       "EPS GAAP Est Avg Rev % (FY1E - 1Y)"                                        AS gaap_revision_1y,
       "EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)"      AS gaap_vs_norm_revision_spread,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 6M)" AS gaap_revision_acceleration,
       CASE
           WHEN "EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0
               THEN 1
           ELSE 0
           END                                                                     AS gaap_positive_revision_flag,
       ABS(("EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)") -
           ("EPS Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 1M)"))
                                                                                   AS revision_quality_divergence
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 8: GROWTH FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_growth_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                revenue_growth_yoy      NUMERIC,
                ebitda_growth_yoy       NUMERIC,
                operating_income_growth NUMERIC,
                fcf_growth              NUMERIC,
                revenue_cagr_5y         NUMERIC,
                forward_revenue_growth  NUMERIC,
                revenue_vs_5y_avg       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                          AS isin,
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100
           END                                                         AS revenue_growth_yoy,
       CASE
           WHEN ABS("EBITDA (-1FY)") > 0
               THEN ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100
           END                                                         AS ebitda_growth_yoy,
       CASE
           WHEN ABS("Operating Income (FY)") > 0
               THEN ("Operating Income (LTM)" - "Operating Income (FY)") /
                    NULLIF(ABS("Operating Income (FY)"), 0) * 100
           END                                                         AS operating_income_growth,
       CASE
           WHEN ABS("FCF (FY)") > 0
               THEN ("FCF (LTM)" - "FCF (FY)") / NULLIF(ABS("FCF (FY)"), 0) * 100
           END                                                         AS fcf_growth,
       "Total Revenues/CAGR (5Y FY)"                                   AS revenue_cagr_5y,
       "Revenues - Est YoY % (FY1E)"                                   AS forward_revenue_growth,
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_revenue_forecast_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                revenue_est_spread         NUMERIC,
                revenue_beat_potential     NUMERIC,
                revenue_est_revision_trend NUMERIC,
                ebitda_est_vs_actual       NUMERIC,
                forward_revenue_multiple   NUMERIC,
                revenue_estimate_count     NUMERIC,
                revenue_guidance_gap       NUMERIC,
                consensus_revenue_growth   NUMERIC,
                ebit_estimate_spread       NUMERIC,
                forward_ebitda_margin      NUMERIC,
                revenue_acceleration       NUMERIC,
                estimate_confidence_score  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                             AS revenue_est_spread,
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_beat_potential,
       "Revenues - Est YoY % (FY1E)"                                            AS revenue_est_revision_trend,
       ("EBITDA (LTM)" - "EBITDA - Est Avg (FY1E)") /
       NULLIF(ABS("EBITDA - Est Avg (FY1E)"), 0) * 100                          AS ebitda_est_vs_actual,
       "Enterprise Value" / NULLIF("Revenues - Est Avg (FY1E)", 0)              AS forward_revenue_multiple,
       "EPS Norm - Est # (FY1E)"                                                AS revenue_estimate_count,
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_guidance_gap,
       ("Revenues - Est Avg (FY1E)" - "Total Revenues (FY)") /
       NULLIF(ABS("Total Revenues (FY)"), 0) * 100                              AS consensus_revenue_growth,
       ("EBIT - Est Med (FY1E)" - "EBIT - Est Med (NTM)") /
       NULLIF(ABS("EBIT - Est Med (NTM)"), 0) * 100                             AS ebit_estimate_spread,
       "EBITDA - Est Avg (FY1E)" / NULLIF("Revenues - Est Avg (FY1E)", 0) * 100 AS forward_ebitda_margin,
       "Revenues - Est YoY % (FY1E)" - "Total Revenues/CAGR (5Y FY)"            AS revenue_acceleration,
       GREATEST(0, LEAST(100,
                         100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                   NULLIF("Revenues - Est Med (FY1E)", 0) * 100)
                   ))                                                           AS estimate_confidence_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 9: DIVIDEND FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_dividend_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                dividend_streak             INTEGER,
                dividend_yield_ltm          NUMERIC,
                dividend_yield_ntm          NUMERIC,
                dividend_payout_ratio       NUMERIC,
                fcf_dividend_coverage       NUMERIC,
                buyback_yield               NUMERIC,
                total_shareholder_yield     NUMERIC,
                dividend_growth_expectation NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                  AS isin,
       "Dividend Streak"::INTEGER                                              AS dividend_streak,
       "Div Yield (LTM)"                                                       AS dividend_yield_ltm,
       "Div Yield (NTM)"                                                       AS dividend_yield_ntm,
       ABS("Common Dividends Paid (LTM)") / NULLIF("Net Income/Adj. (LTM)", 0) AS dividend_payout_ratio,
       CASE
           WHEN ABS("Common Dividends Paid (LTM)") > 0
               THEN "FCF (LTM)" / NULLIF(ABS("Common Dividends Paid (LTM)"), 0)
           END                                                                 AS fcf_dividend_coverage,
       "Buyback Yield (LTM)"                                                   AS buyback_yield,
       COALESCE("Buyback Yield (LTM)", 0) + COALESCE("Div Yield (LTM)", 0)     AS total_shareholder_yield,
       "Div Yield (NTM)" - "Div Yield (LTM)"                                   AS dividend_growth_expectation
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_dividend_timing(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                days_since_ex_date       INTEGER,
                days_to_payment          INTEGER,
                dividend_announced_flag  INTEGER,
                ex_date_approaching_flag INTEGER,
                dividend_frequency_score INTEGER,
                dividend_consistency     NUMERIC,
                recent_dividend_change   NUMERIC,
                dividend_yield_vs_5y_avg NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                     AS isin,
       (CURRENT_DATE - "Dividend Record (Ex Date)")::INTEGER      AS days_since_ex_date,
       ("Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER AS days_to_payment,
       CASE
           WHEN (CURRENT_DATE - "Dividend Record (Announce Date)") <= 30
               THEN 1
           ELSE 0
           END                                                    AS dividend_announced_flag,
       CASE
           WHEN ("Dividend Record (Ex Date)" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                    AS ex_date_approaching_flag,
       CASE "Dividend Record (Frequency)"
           WHEN 'Quarterly' THEN 4
           WHEN 'Semi-Annual' THEN 2
           WHEN 'Annual' THEN 1
           WHEN 'Monthly' THEN 12
           ELSE 0
           END                                                    AS dividend_frequency_score,
       LEAST(1.0, "Dividend Streak"::NUMERIC / 10.0)              AS dividend_consistency,
       CASE
           WHEN "Div Yield (-1FYInd)" > 0
               THEN ("Div Yield (Ind)" - "Div Yield (-1FYInd)") /
                    NULLIF("Div Yield (-1FYInd)", 0) * 100
           END                                                    AS recent_dividend_change,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0)      AS dividend_yield_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 10: EMPLOYMENT FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_employment_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                revenue_per_employee NUMERIC,
                profit_per_employee  NUMERIC,
                ebitda_per_employee  NUMERIC,
                assets_per_employee  NUMERIC,
                fte_growth_1y_pct    NUMERIC,
                fte_growth_3y_pct    NUMERIC,
                workforce_stability  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"  AS isin,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Revenues (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS revenue_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Normalized Net Income (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS profit_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "EBITDA (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS ebitda_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Assets (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS assets_per_employee,
       CASE
           WHEN "Full Time Employees (-1FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                    NULLIF("Full Time Employees (-1FY)", 0) * 100
           END AS fte_growth_1y_pct,
       CASE
           WHEN "Full Time Employees (-3FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-3FY)") /
                    NULLIF("Full Time Employees (-3FY)", 0) * 100
           END AS fte_growth_3y_pct,
       CASE
           WHEN "Avg Employees (5YAVGFY)" > 0
               THEN "Full Time Employees (FY)" / NULLIF("Avg Employees (5YAVGFY)", 0)
           END AS workforce_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_employment_dynamics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                fte_growth_2y_pct         NUMERIC,
                fte_acceleration          NUMERIC,
                workforce_volatility      NUMERIC,
                hiring_intensity          NUMERIC,
                productivity_trend        NUMERIC,
                headcount_vs_revenue      NUMERIC,
                workforce_efficiency_gain NUMERIC,
                layoff_risk_flag          INTEGER,
                rapid_hiring_flag         INTEGER,
                sustainable_growth_flag   INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                             AS isin,
       CASE
           WHEN "Full Time Employees (-2FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
                    NULLIF("Full Time Employees (-2FY)", 0) * 100
           END                                            AS fte_growth_2y_pct,
       CASE
           WHEN "Full Time Employees (-1FY)" > 0 AND "Full Time Employees (-3FY)" > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) -
                    (POWER("Full Time Employees (FY)" / NULLIF("Full Time Employees (-3FY)", 0), 1.0 / 3.0) - 1)
           END * 100                                      AS fte_acceleration,
       ABS(("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
           NULLIF("Full Time Employees (-1FY)", 0) -
           ("Full Time Employees (-1FY)" - "Full Time Employees (-2FY)") /
           NULLIF("Full Time Employees (-2FY)", 0)) * 100 AS workforce_volatility,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) /
                    NULLIF((("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                            NULLIF(ABS("Total Revenues (-1FY)"), 0)), 0)
           END                                            AS hiring_intensity,
       CASE
           WHEN "Full Time Employees (FY)" > 0 AND "Full Time Employees (-1FY)" > 0
               THEN (("Total Revenues (FY)" / "Full Time Employees (FY)") -
                     ("Total Revenues (-1FY)" / "Full Time Employees (-1FY)")) /
                    NULLIF(ABS("Total Revenues (-1FY)" / "Full Time Employees (-1FY)"), 0) * 100
           END                                            AS productivity_trend,
       (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
        NULLIF("Full Time Employees (-1FY)", 0) * 100) -
       (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
        NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100)    AS headcount_vs_revenue,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                     NULLIF(ABS("Total Revenues (-1FY)"), 0) -
                     ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) * 100
           ELSE 0
           END                                            AS workforce_efficiency_gain,
       CASE
           WHEN "Full Time Employees (FY)" < "Full Time Employees (-1FY)"
               AND "Total Revenues (FY)" < "Total Revenues (-1FY)"
               THEN 1
           ELSE 0
           END                                            AS layoff_risk_flag,
       CASE
           WHEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0) > 0.20
               THEN 1
           ELSE 0
           END                                            AS rapid_hiring_flag,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               AND ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") > 0
               THEN 1
           ELSE 0
           END                                            AS sustainable_growth_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 11: CASH FLOW FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_cashflow_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                cfo_to_net_income     NUMERIC,
                fcf_to_net_income     NUMERIC,
                fcf_margin            NUMERIC,
                cfo_growth_yoy        NUMERIC,
                fcf_positive_ratio    NUMERIC,
                acquisition_intensity NUMERIC,
                self_funding_ratio    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                 AS isin,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS fcf_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0)        AS fcf_margin,
       ("CFO (LTM)" - "CFO (-1FY)") / NULLIF("CFO (-1FY)", 0) AS cfo_growth_yoy,
       (CASE WHEN "FCF (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END) / 5.0 AS fcf_positive_ratio,
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))         AS acquisition_intensity,
       CASE
           WHEN ABS("CFI (LTM)") > 0
               THEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0)
           END                                                AS self_funding_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_enhanced_cashflow_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                -- Existing features
                fcf_positive_years      INTEGER,
                fcf_always_positive     INTEGER,
                capex_vs_5y_avg         NUMERIC,
                underinvestment_flag    INTEGER,
                cfo_share_of_cf         NUMERIC,
                cfi_share_of_cf         NUMERIC,
                cff_share_of_cf         NUMERIC,
                self_funding_flag       INTEGER,
                acquisition_to_fcf      NUMERIC,
                sustainable_ma_flag     INTEGER,
                fcf_4q_improvement      NUMERIC,
                cash_flow_quality_score NUMERIC,
                -- NEW: CapEx temporal analysis
                capex_yoy_growth        NUMERIC,
                capex_qoq_growth        NUMERIC,
                capex_3y_trend          NUMERIC,
                capex_volatility        NUMERIC,
                capex_acceleration      INTEGER,
                capex_cut_flag          INTEGER,
                overinvestment_flag     INTEGER,
                -- NEW: Cash Acquisitions temporal analysis
                acquisitions_yoy_growth NUMERIC,
                acquisitions_vs_5y_avg  NUMERIC,
                acquisitions_ltm_total  NUMERIC,
                ma_intensity_score      NUMERIC,
                serial_acquirer_flag    INTEGER,
                acquisition_pause_flag  INTEGER,
                -- NEW: Combined investment metrics
                total_investment_to_cfo NUMERIC,
                organic_vs_inorganic    NUMERIC,
                investment_efficiency   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                            AS isin,
       -- Existing features (unchanged)
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER           AS fcf_positive_years,
       CASE
           WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
               AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0
               THEN 1
           ELSE 0
           END                                                           AS fcf_always_positive,
       ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0)
                                                                         AS capex_vs_5y_avg,
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) < 0.7
               THEN 1
           ELSE 0
           END                                                           AS underinvestment_flag,
       ABS("CFO (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfo_share_of_cf,
       ABS("CFI (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfi_share_of_cf,
       ABS("CFF (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cff_share_of_cf,
       CASE
           WHEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0) > 1
               THEN 1
           ELSE 0
           END                                                           AS self_funding_flag,
       (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
       NULLIF(ABS("FCF (LTM)"), 0)                                       AS acquisition_to_fcf,
       CASE
           WHEN (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
                NULLIF(ABS("FCF (LTM)"), 0) < 0.5
               THEN 1
           ELSE 0
           END                                                           AS sustainable_ma_flag,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0)    AS fcf_4q_improvement,
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC           AS cash_flow_quality_score,

       -- NEW: CapEx YoY growth (FY vs -1FY)
       (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-1FY)")) /
       NULLIF(ABS("Capital Expenditure (-1FY)"), 0) * 100                AS capex_yoy_growth,

       -- NEW: CapEx QoQ growth (FQ vs -1FQFQ)
       (ABS("Capital Expenditure (FQ)") - ABS("Capital Expenditure (-1FQFQ)")) /
       NULLIF(ABS("Capital Expenditure (-1FQFQ)"), 0) * 100              AS capex_qoq_growth,

       -- NEW: CapEx 3-year trend (FY vs -3FY)
       (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-3FY)")) /
       NULLIF(ABS("Capital Expenditure (-3FY)"), 0) * 100                AS capex_3y_trend,

       -- NEW: CapEx volatility (variation across quarters)
       (ABS(ABS("Capital Expenditure (FQ)") - ABS("Capital Expenditure (-1FQFQ)")) +
        ABS(ABS("Capital Expenditure (-1FQFQ)") - ABS("Capital Expenditure (-2FQFQ)")) +
        ABS(ABS("Capital Expenditure (-2FQFQ)") - ABS("Capital Expenditure (-3FQFQ)")) +
        ABS(ABS("Capital Expenditure (-3FQFQ)") - ABS("Capital Expenditure (-4FQFQ)"))) /
       NULLIF((ABS("Capital Expenditure (FQ)") + ABS("Capital Expenditure (-1FQFQ)") +
               ABS("Capital Expenditure (-2FQFQ)") + ABS("Capital Expenditure (-3FQFQ)") +
               ABS("Capital Expenditure (-4FQFQ)")) / 5.0, 0)            AS capex_volatility,

       -- NEW: CapEx acceleration flag (increasing investment rate)
       CASE
           WHEN ABS("Capital Expenditure (FY)") > ABS("Capital Expenditure (-1FY)")
               AND ABS("Capital Expenditure (-1FY)") > ABS("Capital Expenditure (-2FY)")
               THEN 1
           ELSE 0
           END                                                           AS capex_acceleration,

       -- NEW: CapEx cut flag (significant decline may signal distress or maturity)
       CASE
           WHEN (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-1FY)")) /
                NULLIF(ABS("Capital Expenditure (-1FY)"), 0) < -0.25
               THEN 1
           ELSE 0
           END                                                           AS capex_cut_flag,

       -- NEW: Overinvestment flag (CapEx significantly above historical average)
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) > 1.5
               THEN 1
           ELSE 0
           END                                                           AS overinvestment_flag,

       -- NEW: Cash Acquisitions YoY growth
       (ABS(COALESCE("Cash Acquisitions (FY)", 0)) - ABS(COALESCE("Cash Acquisitions (-1FY)", 0))) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (-1FY)", 0)), 0) * 100     AS acquisitions_yoy_growth,

       -- NEW: Cash Acquisitions vs 5Y average
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (5YAVGFQ)", 0)), 0)        AS acquisitions_vs_5y_avg,

       -- NEW: LTM total acquisitions
       ABS(COALESCE("Cash Acquisitions (LTM)", 0))                       AS acquisitions_ltm_total,

       -- NEW: M&A intensity score (acquisitions relative to market cap proxy via total assets)
       ABS(COALESCE("Cash Acquisitions (LTM)", 0)) /
       NULLIF("Total Assets (LTM)", 0) * 100                             AS ma_intensity_score,

       -- NEW: Serial acquirer flag (significant acquisitions in 3+ of last 4 years)
       CASE
           WHEN (CASE WHEN ABS(COALESCE("Cash Acquisitions (FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-1FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-2FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-3FY)", 0)) > 0 THEN 1 ELSE 0 END) >= 3
               THEN 1
           ELSE 0
           END                                                           AS serial_acquirer_flag,

       -- NEW: Acquisition pause flag (no recent acquisitions after historical activity)
       CASE
           WHEN ABS(COALESCE("Cash Acquisitions (FY)", 0)) = 0
               AND (ABS(COALESCE("Cash Acquisitions (-1FY)", 0)) > 0
                   OR ABS(COALESCE("Cash Acquisitions (-2FY)", 0)) > 0)
               THEN 1
           ELSE 0
           END                                                           AS acquisition_pause_flag,

       -- NEW: Total investment (CapEx + Acquisitions) to CFO ratio
       (ABS(COALESCE("Capital Expenditure (LTM)", 0)) + ABS(COALESCE("Cash Acquisitions (LTM)", 0))) /
       NULLIF(ABS("CFO (LTM)"), 0)                                       AS total_investment_to_cfo,

       -- NEW: Organic vs Inorganic growth ratio (CapEx / Acquisitions)
       ABS(COALESCE("Capital Expenditure (LTM)", 0)) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (LTM)", 0)), 0)            AS organic_vs_inorganic,

       -- NEW: Investment efficiency (revenue growth per unit of total investment)
       CASE
           WHEN (ABS(COALESCE("Capital Expenditure (-1FY)", 0)) + ABS(COALESCE("Cash Acquisitions (-1FY)", 0))) > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS(COALESCE("Capital Expenditure (-1FY)", 0)) +
                           ABS(COALESCE("Cash Acquisitions (-1FY)", 0)), 0)
           END                                                           AS investment_efficiency

FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_cashflow_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                cfo_quarterly_trend   NUMERIC,
                cfo_yoy_quarterly     NUMERIC,
                cfi_quarterly_trend   NUMERIC,
                cff_quarterly_trend   NUMERIC,
                fcf_quarterly_trend   NUMERIC,
                cfo_positive_quarters INTEGER,
                cfi_negative_quarters INTEGER,
                cff_pattern_score     NUMERIC,
                cash_burn_rate        NUMERIC,
                cf_volatility_score   NUMERIC,
                operating_cf_momentum NUMERIC,
                financing_dependency  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                               AS isin,
       ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100 AS cfo_quarterly_trend,
       CASE
           WHEN ABS("CFO (-4FQFQ)") > 0
               THEN ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100
           END                                                              AS cfo_yoy_quarterly,
       ("CFI (FQ)" - "CFI (-4FQFQ)") / NULLIF(ABS("CFI (-4FQFQ)"), 0) * 100 AS cfi_quarterly_trend,
       ("CFF (FQ)" - "CFF (-4FQFQ)") / NULLIF(ABS("CFF (-4FQFQ)"), 0) * 100 AS cff_quarterly_trend,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0) * 100 AS fcf_quarterly_trend,
       (CASE WHEN "CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS cfo_positive_quarters,
       (CASE WHEN "CFI (FQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-1FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-2FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-3FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-4FQFQ)" < 0 THEN 1 ELSE 0 END)::INTEGER            AS cfi_negative_quarters,
       CASE
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") > 0
               THEN -1
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") < 0
               THEN 1
           ELSE 0
           END::NUMERIC                                                     AS cff_pattern_score,
       CASE
           WHEN "FCF (LTM)" < 0
               THEN ABS("FCF (LTM)") / NULLIF("Cash And Equivalents (FQ)", 0) / 12.0
           ELSE 0
           END                                                              AS cash_burn_rate,
       (ABS("CFO (FQ)" - "CFO (-1FQFQ)") + ABS("CFO (-1FQFQ)" - "CFO (-2FQFQ)") +
        ABS("CFO (-2FQFQ)" - "CFO (-3FQFQ)") + ABS("CFO (-3FQFQ)" - "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (FQ)" + "CFO (-1FQFQ)" + "CFO (-2FQFQ)" +
                  "CFO (-3FQFQ)" + "CFO (-4FQFQ)") / 5.0, 0)                AS cf_volatility_score,
       (("CFO (FQ)" + "CFO (-1FQFQ)") - ("CFO (-3FQFQ)" + "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (-3FQFQ)" + "CFO (-4FQFQ)"), 0) * 100                AS operating_cf_momentum,
       ABS("CFF (LTM)") / NULLIF(ABS("CFO (LTM)"), 0)                       AS financing_dependency
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 12: TEMPORAL FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                fiscal_quarter          INTEGER,
                fiscal_month            INTEGER,
                fiscal_year             INTEGER,
                days_to_earnings        INTEGER,
                earnings_report_recency INTEGER,
                reporting_lag           NUMERIC,
                fiscal_year_progress    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                          AS isin,
       "Fiscal Quarter"                                AS fiscal_quarter,
       "Fiscal Month"                                  AS fiscal_month,
       "Fiscal Year"                                   AS fiscal_year,
       ("Next Earnings" - CURRENT_DATE)                AS days_to_earnings,
       (CURRENT_DATE - "Income Statement Report Date") AS earnings_report_recency,
       "Reporting Lag"                                 AS reporting_lag,
       "Fiscal Month" / 12.0                           AS fiscal_year_progress
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_fiscal_calendar_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                days_since_last_report    INTEGER,
                days_to_fy_end            INTEGER,
                is_quarter_end_month      INTEGER,
                is_fy_end_month           INTEGER,
                earnings_season_flag      INTEGER,
                pre_earnings_window       INTEGER,
                post_earnings_window      INTEGER,
                reporting_freshness_score NUMERIC,
                fiscal_quarter_progress   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                   AS isin,
       (CURRENT_DATE - "Income Statement Report Date")::INTEGER AS days_since_last_report,
       ("FY End Date" - CURRENT_DATE)::INTEGER                  AS days_to_fy_end,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12)
               THEN 1
           ELSE 0
           END                                                  AS is_quarter_end_month,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM "FY End Date")
               THEN 1
           ELSE 0
           END                                                  AS is_fy_end_month,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 2, 4, 5, 7, 8, 10, 11)
               THEN 1
           ELSE 0
           END                                                  AS earnings_season_flag,
       CASE
           WHEN ("Next Earnings" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                  AS pre_earnings_window,
       CASE
           WHEN (CURRENT_DATE - "Income Statement Report Date") BETWEEN 0 AND 7
               THEN 1
           ELSE 0
           END                                                  AS post_earnings_window,
       GREATEST(0, LEAST(100,
                         100 - ((CURRENT_DATE - "Income Statement Report Date")::NUMERIC / 90.0 * 100)
                   ))                                           AS reporting_freshness_score,
       CASE
           WHEN "Fiscal Month" IS NOT NULL
               THEN (("Fiscal Month" - 1) % 3 + 1) / 3.0
           END                                                  AS fiscal_quarter_progress
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 13: COMPOSITE SCORES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_composite_scores(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                   TEXT,
                piotroski_f_score      INTEGER,
                eps_trajectory_score   NUMERIC,
                dilution_score         NUMERIC,
                quality_momentum_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"            AS isin,
       (CASE WHEN "Return on Assets (ROA) % (LTM)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (LTM)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Return on Assets (ROA) % (LTM)" > "Return on Assets (ROA) % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "CFO (LTM)" > "Net Income - (IS) (LTM)" THEN 1 ELSE 0 END +
        CASE
            WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) <
                 "Total Debt (FY)" / NULLIF("Total Equity (FY)", 0) THEN 1
            ELSE 0 END +
        CASE WHEN "Current Ratio (LTM)" > "Current Ratio (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Shrs Out" <= "Shrs Out (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Asset Turnover (LTM)" > "Asset Turnover (FY)" THEN 1 ELSE 0 END
           )::INTEGER    AS piotroski_f_score,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100 AS eps_trajectory_score,
       GREATEST(0, LEAST(100,
                         50 - (("Shrs Out" - "Shrs Out (-1FY)") / NULLIF("Shrs Out (-1FY)", 0)) * 100
                   ))    AS dilution_score,
       (((CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
          CASE WHEN "Return On Equity % (LTM)" > 15 THEN 25 ELSE 0 END +
          CASE WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) < 1 THEN 25 ELSE 0 END +
          CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END) * 0.40) +
        (LEAST(100, GREATEST(0,
                             (("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100 + 50))) * 0.30) +
        (CASE
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 20 THEN 100
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 10 THEN 75
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 0 THEN 50
             ELSE 25
             END * 0.30)
           )             AS quality_momentum_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 14: COMPREHENSIVE FUNCTIONS (OPTIMIZED WITH MATERIALIZED VIEWS)
-- =============================================================================

-- Note: Due to size, comprehensive functions are provided as materialized views
-- for better performance. The functions remain available for ad-hoc queries.

CREATE OR REPLACE FUNCTION calc_ebit_ebitda_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                -- Current values (existing)
                ebit_fq               NUMERIC,
                ebit_ltm              NUMERIC,
                ebit_fy               NUMERIC,
                ebit_1fy              NUMERIC,
                ebitda_fq             NUMERIC,
                ebitda_ltm            NUMERIC,
                ebitda_fy             NUMERIC,
                ebitda_1fy            NUMERIC,
                -- NEW: Extended historical FY
                ebit_2fy              NUMERIC,
                ebit_3fy              NUMERIC,
                ebit_4fy              NUMERIC,
                ebitda_2fy            NUMERIC,
                ebitda_3fy            NUMERIC,
                ebitda_4fy            NUMERIC,
                -- NEW: Quarterly historical
                ebit_1fqfq            NUMERIC,
                ebit_2fqfq            NUMERIC,
                ebit_3fqfq            NUMERIC,
                ebit_4fqfq            NUMERIC,
                ebitda_1fqfq          NUMERIC,
                ebitda_2fqfq          NUMERIC,
                ebitda_3fqfq          NUMERIC,
                ebitda_4fqfq          NUMERIC,
                -- NEW: 5-year averages
                ebit_5yavgfq          NUMERIC,
                ebit_5yavgltm         NUMERIC,
                ebitda_5yavgfq        NUMERIC,
                ebitda_5yavgltm       NUMERIC,
                -- NEW: Adjusted variants
                ebit_adj_fq           NUMERIC,
                ebit_adj_ltm          NUMERIC,
                ebit_adj_fy           NUMERIC,
                ebitda_adj_fq         NUMERIC,
                ebitda_adj_ltm        NUMERIC,
                ebitda_adj_fy         NUMERIC,
                -- Derived metrics (existing + enhanced)
                ebit_growth_yoy       NUMERIC,
                ebitda_growth_yoy     NUMERIC,
                ebit_margin_ltm       NUMERIC,
                ebitda_margin_ltm     NUMERIC,
                ebit_positive_years   INTEGER,
                ebitda_positive_years INTEGER,
                -- NEW: Quarterly momentum
                ebit_qoq_growth       NUMERIC,
                ebitda_qoq_growth     NUMERIC,
                -- NEW: Multi-year CAGR
                ebit_cagr_3y          NUMERIC,
                ebitda_cagr_3y        NUMERIC,
                -- NEW: vs 5Y average
                ebit_vs_5y_avg        NUMERIC,
                ebitda_vs_5y_avg      NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                    AS isin,
       -- Current values
       "EBIT (FQ)"                                                               AS ebit_fq,
       "EBIT (LTM)"                                                              AS ebit_ltm,
       "EBIT (FY)"                                                               AS ebit_fy,
       "EBIT (-1FY)"                                                             AS ebit_1fy,
       "EBITDA (FQ)"                                                             AS ebitda_fq,
       "EBITDA (LTM)"                                                            AS ebitda_ltm,
       "EBITDA (FY)"                                                             AS ebitda_fy,
       "EBITDA (-1FY)"                                                           AS ebitda_1fy,
       -- Extended historical FY
       "EBIT (-2FY)"                                                             AS ebit_2fy,
       "EBIT (-3FY)"                                                             AS ebit_3fy,
       "EBIT (-4FY)"                                                             AS ebit_4fy,
       "EBITDA (-2FY)"                                                           AS ebitda_2fy,
       "EBITDA (-3FY)"                                                           AS ebitda_3fy,
       "EBITDA (-4FY)"                                                           AS ebitda_4fy,
       -- Quarterly historical
       "EBIT (-1FQFQ)"                                                           AS ebit_1fqfq,
       "EBIT (-2FQFQ)"                                                           AS ebit_2fqfq,
       "EBIT (-3FQFQ)"                                                           AS ebit_3fqfq,
       "EBIT (-4FQFQ)"                                                           AS ebit_4fqfq,
       "EBITDA (-1FQFQ)"                                                         AS ebitda_1fqfq,
       "EBITDA (-2FQFQ)"                                                         AS ebitda_2fqfq,
       "EBITDA (-3FQFQ)"                                                         AS ebitda_3fqfq,
       "EBITDA (-4FQFQ)"                                                         AS ebitda_4fqfq,
       -- 5-year averages
       "EBIT (5YAVGFQ)"                                                          AS ebit_5yavgfq,
       "EBIT (5YAVGLTM)"                                                         AS ebit_5yavgltm,
       "EBITDA (5YAVGFQ)"                                                        AS ebitda_5yavgfq,
       "EBITDA (5YAVGLTM)"                                                       AS ebitda_5yavgltm,
       -- Adjusted variants
       "EBIT/Adj. (FQ)"                                                          AS ebit_adj_fq,
       "EBIT/Adj. (LTM)"                                                         AS ebit_adj_ltm,
       "EBIT/Adj. (FY)"                                                          AS ebit_adj_fy,
       "EBITDA/Adj. (FQ)"                                                        AS ebitda_adj_fq,
       "EBITDA/Adj. (LTM)"                                                       AS ebitda_adj_ltm,
       "EBITDA/Adj. (FY)"                                                        AS ebitda_adj_fy,
       -- Derived metrics
       ("EBIT (FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) * 100       AS ebit_growth_yoy,
       ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100 AS ebitda_growth_yoy,
       "EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                    AS ebit_margin_ltm,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                  AS ebitda_margin_ltm,
       (CASE WHEN "EBIT (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                  AS ebit_positive_years,
       (CASE WHEN "EBITDA (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                AS ebitda_positive_years,
       -- NEW: Quarterly momentum
       ("EBIT (FQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) * 100   AS ebit_qoq_growth,
       ("EBITDA (FQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) * 100
                                                                                 AS ebitda_qoq_growth,
       -- NEW: Multi-year CAGR (3-year)
       CASE
           WHEN "EBIT (-3FY)" > 0 AND "EBIT (FY)" > 0
               THEN (POWER("EBIT (FY)" / NULLIF("EBIT (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebit_cagr_3y,
       CASE
           WHEN "EBITDA (-3FY)" > 0 AND "EBITDA (FY)" > 0
               THEN (POWER("EBITDA (FY)" / NULLIF("EBITDA (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebitda_cagr_3y,
       -- NEW: vs 5Y average
       "EBIT (LTM)" / NULLIF("EBIT (5YAVGLTM)", 0)                               AS ebit_vs_5y_avg,
       "EBITDA (LTM)" / NULLIF("EBITDA (5YAVGLTM)", 0)                           AS ebitda_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_net_income_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                -- Base values (existing)
                net_income_is_fq           NUMERIC,
                net_income_is_ltm          NUMERIC,
                net_income_is_fy           NUMERIC,
                net_income_adj_ltm         NUMERIC,
                normalized_ni_ltm          NUMERIC,
                -- NEW: Extended quarterly historical
                net_income_is_1fqfq        NUMERIC,
                net_income_is_2fqfq        NUMERIC,
                net_income_is_3fqfq        NUMERIC,
                net_income_is_4fqfq        NUMERIC,
                -- NEW: Extended yearly historical
                net_income_is_1fy          NUMERIC,
                net_income_is_2fy          NUMERIC,
                net_income_is_3fy          NUMERIC,
                net_income_is_4fy          NUMERIC,
                -- NEW: 5-year averages
                net_income_is_5yavgfq      NUMERIC,
                net_income_is_5yavgltm     NUMERIC,
                normalized_ni_5yavgfq      NUMERIC,
                normalized_ni_5yavgltm     NUMERIC,
                -- Derived metrics (existing + enhanced)
                net_income_growth_yoy      NUMERIC,
                net_income_margin_ltm      NUMERIC,
                ni_adjustment_ratio        NUMERIC,
                net_income_positive_years  INTEGER,
                earnings_quality_composite NUMERIC,
                -- NEW: Quarterly trends
                net_income_qoq_growth      NUMERIC,
                net_income_yoy_quarterly   NUMERIC,
                -- NEW: vs 5Y averages
                net_income_vs_5y_avg       NUMERIC,
                normalized_ni_vs_5y_avg    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                 AS isin,
       -- Base values
       "Net Income - (IS) (FQ)"                                               AS net_income_is_fq,
       "Net Income - (IS) (LTM)"                                              AS net_income_is_ltm,
       "Net Income - (IS) (FY)"                                               AS net_income_is_fy,
       "Net Income/Adj. (LTM)"                                                AS net_income_adj_ltm,
       "Normalized Net Income (LTM)"                                          AS normalized_ni_ltm,
       -- Extended quarterly historical
       "Net Income - (IS) (-1FQFQ)"                                           AS net_income_is_1fqfq,
       "Net Income - (IS) (-2FQFQ)"                                           AS net_income_is_2fqfq,
       "Net Income - (IS) (-3FQFQ)"                                           AS net_income_is_3fqfq,
       "Net Income - (IS) (-4FQFQ)"                                           AS net_income_is_4fqfq,
       -- Extended yearly historical
       "Net Income - (IS) (-1FY)"                                             AS net_income_is_1fy,
       "Net Income - (IS) (-2FY)"                                             AS net_income_is_2fy,
       "Net Income - (IS) (-3FY)"                                             AS net_income_is_3fy,
       "Net Income - (IS) (-4FY)"                                             AS net_income_is_4fy,
       -- 5-year averages
       "Net Income - (IS) (5YAVGFQ)"                                          AS net_income_is_5yavgfq,
       "Net Income - (IS) (5YAVGLTM)"                                         AS net_income_is_5yavgltm,
       "Normalized Net Income (5YAVGFQ)"                                      AS normalized_ni_5yavgfq,
       "Normalized Net Income (5YAVGLTM)"                                     AS normalized_ni_5yavgltm,
       -- Derived metrics
       pct_change("Net Income - (IS) (FY)", "Net Income - (IS) (-1FY)")       AS net_income_growth_yoy,
       "Net Income Margin % (LTM)"                                            AS net_income_margin_ltm,
       safe_divide("Net Income/Adj. (LTM)", "Net Income - (IS) (LTM)")        AS ni_adjustment_ratio,
       (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER  AS net_income_positive_years,
       clamp_score(
               50 +
               (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 10 ELSE -10 END) +
               (CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 5 ELSE -5 END) +
               (CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 5 ELSE -5 END) +
               (CASE
                    WHEN ABS(safe_divide("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)",
                                         "Net Income - (IS) (LTM)")) < 0.10 THEN 15
                    ELSE -15 END) +
               (CASE WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 10 ELSE -5 END) +
               (CASE WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 5 ELSE -5 END)
       )                                                                      AS earnings_quality_composite,
       -- Quarterly trends
       pct_change("Net Income - (IS) (FQ)", "Net Income - (IS) (-1FQFQ)")     AS net_income_qoq_growth,
       pct_change("Net Income - (IS) (FQ)", "Net Income - (IS) (-4FQFQ)")     AS net_income_yoy_quarterly,
       -- vs 5Y averages
       safe_divide("Net Income - (IS) (LTM)", "Net Income - (IS) (5YAVGLTM)") AS net_income_vs_5y_avg,
       safe_divide("Normalized Net Income (LTM)", "Normalized Net Income (5YAVGLTM)")
                                                                              AS normalized_ni_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_total_revenues_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                -- Base values
                revenue_fq            NUMERIC,
                revenue_ltm           NUMERIC,
                revenue_fy            NUMERIC,
                revenue_1fy           NUMERIC,
                -- 5-year averages
                revenue_5yavgfq       NUMERIC,
                revenue_5yavgltm      NUMERIC,
                -- Growth metrics
                revenue_growth_yoy    NUMERIC,
                revenue_vs_5y_avg_fq  NUMERIC,
                revenue_vs_5y_avg_ltm NUMERIC,
                -- Trend indicators
                revenue_fq_vs_avg     NUMERIC,
                revenue_momentum      NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       "Total Revenues (FQ)"                                                    AS revenue_fq,
       "Total Revenues (LTM)"                                                   AS revenue_ltm,
       "Total Revenues (FY)"                                                    AS revenue_fy,
       "Total Revenues (-1FY)"                                                  AS revenue_1fy,
       "Total Revenues (5YAVGFQ)"                                               AS revenue_5yavgfq,
       "Total Revenues (5YAVGLTM)"                                              AS revenue_5yavgltm,
       pct_change("Total Revenues (FY)", "Total Revenues (-1FY)")               AS revenue_growth_yoy,
       safe_divide("Total Revenues (FQ)", "Total Revenues (5YAVGFQ)")           AS revenue_vs_5y_avg_fq,
       safe_divide("Total Revenues (LTM)", "Total Revenues (5YAVGLTM)")         AS revenue_vs_5y_avg_ltm,
       safe_divide("Total Revenues (FQ)" - "Total Revenues (5YAVGFQ)", "Total Revenues (5YAVGFQ)") * 100
                                                                                AS revenue_fq_vs_avg,
       calc_change_ratio("Total Revenues (LTM)", "Total Revenues (-1FY)") * 100 AS revenue_momentum
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_working_capital_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                -- Current values
                wc_fq                NUMERIC,
                wc_fy                NUMERIC,
                wc_ltm               NUMERIC,
                wc_5yavgfy           NUMERIC,
                -- Quarterly historical (FQ style)
                wc_1fq               NUMERIC,
                wc_2fq               NUMERIC,
                wc_3fq               NUMERIC,
                wc_4fq               NUMERIC,
                -- Yearly historical
                wc_1fy               NUMERIC,
                wc_2fy               NUMERIC,
                wc_3fy               NUMERIC,
                wc_4fy               NUMERIC,
                -- Trend metrics
                wc_qoq_change        NUMERIC,
                wc_yoy_change        NUMERIC,
                wc_4q_trend          NUMERIC,
                wc_vs_5y_avg         NUMERIC,
                wc_positive_quarters INTEGER,
                wc_improving_flag    INTEGER,
                wc_volatility        NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                              AS isin,
       -- Current values
       "Working Capital (FQ)"                                              AS wc_fq,
       "Working Capital (FY)"                                              AS wc_fy,
       "Working Capital (LTM)"                                             AS wc_ltm,
       "Working Capital (5YAVGFY)"                                         AS wc_5yavgfy,
       -- Quarterly historical
       "Working Capital (-1FQ)"                                            AS wc_1fq,
       "Working Capital (-2FQ)"                                            AS wc_2fq,
       "Working Capital (-3FQ)"                                            AS wc_3fq,
       "Working Capital (-4FQ)"                                            AS wc_4fq,
       -- Yearly historical
       "Working Capital (-1FY)"                                            AS wc_1fy,
       "Working Capital (-2FY)"                                            AS wc_2fy,
       "Working Capital (-3FY)"                                            AS wc_3fy,
       "Working Capital (-4FY)"                                            AS wc_4fy,
       -- Trend metrics
       pct_change("Working Capital (FQ)", "Working Capital (-1FQ)")        AS wc_qoq_change,
       pct_change("Working Capital (FY)", "Working Capital (-1FY)")        AS wc_yoy_change,
       pct_change("Working Capital (FQ)", "Working Capital (-4FQ)")        AS wc_4q_trend,
       safe_divide("Working Capital (FQ)", "Working Capital (5YAVGFY)")    AS wc_vs_5y_avg,
       (CASE WHEN "Working Capital (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-1FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-2FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-3FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-4FQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS wc_positive_quarters,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (-1FQ)"
               AND "Working Capital (-1FQ)" > "Working Capital (-2FQ)"
               THEN 1
           ELSE 0 END                                                      AS wc_improving_flag,
       -- Volatility: coefficient of variation across quarters
       (ABS("Working Capital (FQ)" - "Working Capital (-1FQ)") +
        ABS("Working Capital (-1FQ)" - "Working Capital (-2FQ)") +
        ABS("Working Capital (-2FQ)" - "Working Capital (-3FQ)") +
        ABS("Working Capital (-3FQ)" - "Working Capital (-4FQ)")) /
       NULLIF(ABS(("Working Capital (FQ)" + "Working Capital (-1FQ)" +
                   "Working Capital (-2FQ)" + "Working Capital (-3FQ)" +
                   "Working Capital (-4FQ)") / 5.0), 0)                    AS wc_volatility
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_total_debt_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                -- Current values
                debt_fq              NUMERIC,
                debt_fy              NUMERIC,
                debt_ltm             NUMERIC,
                -- Quarterly historical
                debt_1fq             NUMERIC,
                debt_2fq             NUMERIC,
                debt_3fq             NUMERIC,
                debt_4fq             NUMERIC,
                -- Yearly historical
                debt_1fy             NUMERIC,
                debt_2fy             NUMERIC,
                debt_3fy             NUMERIC,
                debt_4fy             NUMERIC,
                -- Trend metrics
                debt_qoq_change      NUMERIC,
                debt_yoy_change      NUMERIC,
                debt_4q_trend        NUMERIC,
                debt_3y_cagr         NUMERIC,
                debt_deleveraging    INTEGER,
                debt_to_equity_trend NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                           AS isin,
       -- Current values
       "Total Debt (FQ)"                                                AS debt_fq,
       "Total Debt (FY)"                                                AS debt_fy,
       "Total Debt (LTM)"                                               AS debt_ltm,
       -- Quarterly historical
       "Total Debt (-1FQ)"                                              AS debt_1fq,
       "Total Debt (-2FQ)"                                              AS debt_2fq,
       "Total Debt (-3FQ)"                                              AS debt_3fq,
       "Total Debt (-4FQ)"                                              AS debt_4fq,
       -- Yearly historical
       "Total Debt (-1FY)"                                              AS debt_1fy,
       "Total Debt (-2FY)"                                              AS debt_2fy,
       "Total Debt (-3FY)"                                              AS debt_3fy,
       "Total Debt (-4FY)"                                              AS debt_4fy,
       -- Trend metrics
       pct_change("Total Debt (FQ)", "Total Debt (-1FQ)")               AS debt_qoq_change,
       pct_change("Total Debt (FY)", "Total Debt (-1FY)")               AS debt_yoy_change,
       pct_change("Total Debt (FQ)", "Total Debt (-4FQ)")               AS debt_4q_trend,
       CASE
           WHEN "Total Debt (-3FY)" > 0
               THEN (POWER(safe_divide("Total Debt (FY)", "Total Debt (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                          AS debt_3y_cagr,
       CASE
           WHEN "Total Debt (FQ)" < "Total Debt (-1FQ)"
               AND "Total Debt (-1FQ)" < "Total Debt (-2FQ)"
               THEN 1
           ELSE 0 END                                                   AS debt_deleveraging,
       safe_divide("Total Debt (FY)", "Total Equity (FY)") -
       safe_divide("Total Debt (-1FY)", NULLIF("Total Equity (FY)", 0)) AS debt_to_equity_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_total_assets_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin               TEXT,
                -- Current values
                assets_fq          NUMERIC,
                assets_fy          NUMERIC,
                assets_ltm         NUMERIC,
                -- Quarterly historical
                assets_1fq         NUMERIC,
                assets_2fq         NUMERIC,
                assets_3fq         NUMERIC,
                assets_4fq         NUMERIC,
                -- Yearly historical
                assets_1fy         NUMERIC,
                assets_2fy         NUMERIC,
                assets_3fy         NUMERIC,
                assets_4fy         NUMERIC,
                -- Trend metrics
                assets_qoq_growth  NUMERIC,
                assets_yoy_growth  NUMERIC,
                assets_3y_cagr     NUMERIC,
                asset_growth_accel NUMERIC,
                asset_base_stable  INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                   AS isin,
       "Total Assets (FQ)"                                      AS assets_fq,
       "Total Assets (FY)"                                      AS assets_fy,
       "Total Assets (LTM)"                                     AS assets_ltm,
       "Total Assets (-1FQ)"                                    AS assets_1fq,
       "Total Assets (-2FQ)"                                    AS assets_2fq,
       "Total Assets (-3FQ)"                                    AS assets_3fq,
       "Total Assets (-4FQ)"                                    AS assets_4fq,
       "Total Assets (-1FY)"                                    AS assets_1fy,
       "Total Assets (-2FY)"                                    AS assets_2fy,
       "Total Assets (-3FY)"                                    AS assets_3fy,
       "Total Assets (-4FY)"                                    AS assets_4fy,
       pct_change("Total Assets (FQ)", "Total Assets (-1FQ)")   AS assets_qoq_growth,
       pct_change("Total Assets (FY)", "Total Assets (-1FY)")   AS assets_yoy_growth,
       CASE
           WHEN "Total Assets (-3FY)" > 0
               THEN (POWER(safe_divide("Total Assets (FY)", "Total Assets (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                  AS assets_3y_cagr,
       -- Growth acceleration: recent growth vs historical
       pct_change("Total Assets (FY)", "Total Assets (-1FY)") -
       pct_change("Total Assets (-1FY)", "Total Assets (-2FY)") AS asset_growth_accel,
       -- Stability flag: growing consistently
       CASE
           WHEN "Total Assets (FY)" >= "Total Assets (-1FY)"
               AND "Total Assets (-1FY)" >= "Total Assets (-2FY)"
               AND "Total Assets (-2FY)" >= "Total Assets (-3FY)"
               THEN 1
           ELSE 0 END                                           AS asset_base_stable
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_gross_profit_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                -- Current values
                gp_fq                NUMERIC,
                gp_fy                NUMERIC,
                gp_ltm               NUMERIC,
                -- Quarterly historical (FQFQ style)
                gp_1fqfq             NUMERIC,
                gp_2fqfq             NUMERIC,
                gp_3fqfq             NUMERIC,
                gp_4fqfq             NUMERIC,
                -- Yearly historical
                gp_1fy               NUMERIC,
                gp_2fy               NUMERIC,
                gp_3fy               NUMERIC,
                gp_4fy               NUMERIC,
                -- Derived metrics
                gp_qoq_growth        NUMERIC,
                gp_yoy_growth        NUMERIC,
                gp_margin_fq         NUMERIC,
                gp_margin_trend      NUMERIC,
                gp_positive_quarters INTEGER,
                gp_margin_expansion  INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       "Gross Profit (FQ)"                                                      AS gp_fq,
       "Gross Profit (FY)"                                                      AS gp_fy,
       "Gross Profit (LTM)"                                                     AS gp_ltm,
       "Gross Profit (-1FQFQ)"                                                  AS gp_1fqfq,
       "Gross Profit (-2FQFQ)"                                                  AS gp_2fqfq,
       "Gross Profit (-3FQFQ)"                                                  AS gp_3fqfq,
       "Gross Profit (-4FQFQ)"                                                  AS gp_4fqfq,
       "Gross Profit (-1FY)"                                                    AS gp_1fy,
       "Gross Profit (-2FY)"                                                    AS gp_2fy,
       "Gross Profit (-3FY)"                                                    AS gp_3fy,
       "Gross Profit (-4FY)"                                                    AS gp_4fy,
       pct_change("Gross Profit (FQ)", "Gross Profit (-1FQFQ)")                 AS gp_qoq_growth,
       pct_change("Gross Profit (FY)", "Gross Profit (-1FY)")                   AS gp_yoy_growth,
       safe_divide("Gross Profit (FQ)", "Total Revenues (FQ)") * 100            AS gp_margin_fq,
       (safe_divide("Gross Profit (FQ)", "Total Revenues (FQ)") -
        safe_divide("Gross Profit (-4FQFQ)", "Total Revenues (5YAVGFQ)")) * 100 AS gp_margin_trend,
       (CASE WHEN "Gross Profit (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER       AS gp_positive_quarters,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               THEN 1
           ELSE 0 END                                                           AS gp_margin_expansion
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_quality_features_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                          TEXT,
                goodwill_impairment_ltm       NUMERIC,
                asset_writedown_ltm           NUMERIC,
                restructuring_ltm             NUMERIC,
                has_goodwill_impairment_ltm   INTEGER,
                goodwill_impairment_frequency INTEGER,
                asset_writedown_frequency     INTEGER,
                restructuring_frequency       INTEGER,
                exceptional_items_total_ltm   NUMERIC,
                exceptional_items_to_ebitda   NUMERIC,
                quality_issues_count_5y       INTEGER,
                accounting_quality_score      NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                       AS isin,
       "Impairment of Goodwill (LTM)"                                               AS goodwill_impairment_ltm,
       "Asset Writedown (LTM)"                                                      AS asset_writedown_ltm,
       "Restructuring Charges (LTM)"                                                AS restructuring_ltm,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END              AS has_goodwill_impairment_ltm,
       (CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER  AS goodwill_impairment_frequency,
       (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER         AS asset_writedown_frequency,
       (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER   AS restructuring_frequency,
       ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
       ABS("Restructuring Charges (LTM)")                                           AS exceptional_items_total_ltm,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF(ABS("EBITDA (LTM)"), 0)        AS exceptional_items_to_ebitda,
       ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END))::INTEGER AS quality_issues_count_5y,
       GREATEST(0, LEAST(100,
                         100 -
                         ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) * 8) -
                         ((CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4) -
                         ((CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4)
                   ))                                                               AS accounting_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_eps_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                eps_basic_fq         NUMERIC,
                eps_basic_ltm        NUMERIC,
                eps_basic_fy         NUMERIC,
                eps_adj_ltm          NUMERIC,
                eps_norm_est_fy1e    NUMERIC,
                eps_growth_yoy       NUMERIC,
                eps_cagr_3y          NUMERIC,
                eps_adjustment_ratio NUMERIC,
                eps_positive_years   INTEGER,
                eps_trajectory_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                              AS isin,
       "Net EPS - Basic (FQ)"                                              AS eps_basic_fq,
       "Net EPS - Basic (LTM)"                                             AS eps_basic_ltm,
       "Net EPS - Basic (FY)"                                              AS eps_basic_fy,
       "EPS/Adj. (LTM)"                                                    AS eps_adj_ltm,
       "EPS Norm - Est Avg (FY1E)"                                         AS eps_norm_est_fy1e,
       ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
       NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100                      AS eps_growth_yoy,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                             AS eps_cagr_3y,
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0)               AS eps_adjustment_ratio,
       (CASE WHEN "Net EPS - Basic (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_years,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                   AS eps_trajectory_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- EPS Continuing Operations Features (NEW)
-- Uses Basic EPS - Cont columns for core operations earnings quality
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_eps_continuing_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                -- Current period values
                eps_cont_ltm              NUMERIC,
                eps_cont_fq               NUMERIC,
                eps_cont_fy               NUMERIC,
                -- Historical FQ
                eps_cont_1fqfq            NUMERIC,
                eps_cont_2fqfq            NUMERIC,
                eps_cont_3fqfq            NUMERIC,
                eps_cont_4fqfq            NUMERIC,
                -- Historical FY
                eps_cont_1fy              NUMERIC,
                eps_cont_2fy              NUMERIC,
                eps_cont_3fy              NUMERIC,
                eps_cont_4fy              NUMERIC,
                -- Derived analytics
                eps_cont_qoq_growth       NUMERIC,
                eps_cont_yoy_growth       NUMERIC,
                eps_cont_cagr_3y          NUMERIC,
                eps_cont_vs_total_eps     NUMERIC,
                eps_cont_positive_streak  INTEGER,
                eps_cont_trajectory_score NUMERIC,
                -- Quality flags
                discontinued_ops_impact   NUMERIC,
                core_earnings_stability   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                           AS isin,
       -- Current period values
       "Basic EPS - Cont (LTM)"                                         AS eps_cont_ltm,
       "Basic EPS - Cont (FQ)"                                          AS eps_cont_fq,
       "Basic EPS - Cont (FY)"                                          AS eps_cont_fy,
       -- Historical FQ
       "Basic EPS - Cont (-1FQFQ)"                                      AS eps_cont_1fqfq,
       "Basic EPS - Cont (-2FQFQ)"                                      AS eps_cont_2fqfq,
       "Basic EPS - Cont (-3FQFQ)"                                      AS eps_cont_3fqfq,
       "Basic EPS - Cont (-4FQFQ)"                                      AS eps_cont_4fqfq,
       -- Historical FY
       "Basic EPS - Cont (-1FY)"                                        AS eps_cont_1fy,
       "Basic EPS - Cont (-2FY)"                                        AS eps_cont_2fy,
       "Basic EPS - Cont (-3FY)"                                        AS eps_cont_3fy,
       "Basic EPS - Cont (-4FY)"                                        AS eps_cont_4fy,
       -- QoQ growth
       pct_change("Basic EPS - Cont (FQ)", "Basic EPS - Cont (-1FQFQ)") AS eps_cont_qoq_growth,
       -- YoY growth
       pct_change("Basic EPS - Cont (FY)", "Basic EPS - Cont (-1FY)")   AS eps_cont_yoy_growth,
       -- 3-year CAGR
       CASE
           WHEN "Basic EPS - Cont (-3FY)" > 0 AND "Basic EPS - Cont (FY)" > 0
               THEN (POWER("Basic EPS - Cont (FY)" / NULLIF("Basic EPS - Cont (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                          AS eps_cont_cagr_3y,
       -- Continuing vs Total EPS ratio (how much comes from continuing ops)
       safe_divide("Basic EPS - Cont (LTM)", "Net EPS - Basic (LTM)")   AS eps_cont_vs_total_eps,
       -- Positive streak count
       (CASE WHEN "Basic EPS - Cont (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER
                                                                        AS eps_cont_positive_streak,
       -- Trajectory score (improving trend = higher score)
       (CASE WHEN "Basic EPS - Cont (FY)" > "Basic EPS - Cont (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-1FY)" > "Basic EPS - Cont (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-2FY)" > "Basic EPS - Cont (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-3FY)" > "Basic EPS - Cont (-4FY)" THEN 1 ELSE 0 END
           ) / 4.0 * 100                                                AS eps_cont_trajectory_score,
       -- Discontinued operations impact (difference between total and continuing)
       (("Net EPS - Basic (LTM)" - "Basic EPS - Cont (LTM)") /
        NULLIF(ABS("Net EPS - Basic (LTM)"), 0)) * 100                  AS discontinued_ops_impact,
       -- Core earnings stability score
       clamp_score(
               100 - ABS(pct_change("Basic EPS - Cont (FQ)", "Basic EPS - Cont (-4FQFQ)") -
                         pct_change("Basic EPS - Cont (-1FQFQ)", "Basic EPS - Cont (-4FQFQ)")) * 0.5
       )                                                                AS core_earnings_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_cashflow_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                cfo_fq                  NUMERIC,
                cfo_ltm                 NUMERIC,
                cfo_fy                  NUMERIC,
                fcf_fq                  NUMERIC,
                fcf_ltm                 NUMERIC,
                fcf_fy                  NUMERIC,
                cfo_growth_yoy          NUMERIC,
                fcf_growth_yoy          NUMERIC,
                cfo_to_net_income       NUMERIC,
                fcf_margin              NUMERIC,
                fcf_yield               NUMERIC,
                cfo_positive_years      INTEGER,
                fcf_positive_years      INTEGER,
                cash_flow_quality_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                           AS isin,
       "CFO (FQ)"                                                       AS cfo_fq,
       "CFO (LTM)"                                                      AS cfo_ltm,
       "CFO (FY)"                                                       AS cfo_fy,
       "FCF (FQ)"                                                       AS fcf_fq,
       "FCF (LTM)"                                                      AS fcf_ltm,
       "FCF (FY)"                                                       AS fcf_fy,
       ("CFO (FY)" - "CFO (-1FY)") / NULLIF(ABS("CFO (-1FY)"), 0) * 100 AS cfo_growth_yoy,
       ("FCF (FY)" - "FCF (-1FY)") / NULLIF(ABS("FCF (-1FY)"), 0) * 100 AS fcf_growth_yoy,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100            AS fcf_margin,
       "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                      AS fcf_yield,
       (CASE WHEN "CFO (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER          AS cfo_positive_years,
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER          AS fcf_positive_years,
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC          AS cash_flow_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 17: MISSING FUNCTIONS (OPTIMIZED)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Beta Risk Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_beta_risk_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                beta_1y              NUMERIC,
                beta_5y              NUMERIC,
                beta_spread          NUMERIC,
                beta_trend           NUMERIC,
                high_beta_flag       INTEGER,
                low_beta_flag        INTEGER,
                beta_stability_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                        AS isin,
       "Beta (1Y)"                                   AS beta_1y,
       "Beta (5Y)"                                   AS beta_5y,
       "Beta (1Y)" - "Beta (5Y)"                     AS beta_spread,
       ("Beta (1Y)" - "Beta (5Y)") / NULLIF(ABS("Beta (5Y)"), 0) * 100
                                                     AS beta_trend,
       CASE WHEN "Beta (1Y)" > 1.5 THEN 1 ELSE 0 END AS high_beta_flag,
       CASE WHEN "Beta (1Y)" < 0.5 THEN 1 ELSE 0 END AS low_beta_flag,
       GREATEST(0, LEAST(100,
                         100 - ABS("Beta (1Y)" - "Beta (5Y)") * 50
                   ))                                AS beta_stability_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Cost Structure Features (REFACTORED)
-- Uses helper functions and corrected column references
-- Enhanced with Marketing efficiency and SG&A 5Y comparison metrics
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_cost_structure_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                cogs_to_revenue          NUMERIC,
                opex_to_revenue          NUMERIC,
                sga_to_revenue           NUMERIC,
                rnd_to_revenue           NUMERIC,
                interest_to_revenue      NUMERIC,
                sga_trend_yoy            NUMERIC,
                operating_leverage_proxy NUMERIC,
                cost_efficiency_score    NUMERIC,
                -- NEW: Marketing efficiency metrics
                marketing_to_revenue     NUMERIC,
                marketing_trend_yoy      NUMERIC,
                marketing_vs_5y_avg      NUMERIC,
                -- NEW: SG&A efficiency
                sga_vs_5y_avg            NUMERIC,
                sga_efficiency_trend     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,
       safe_divide("Cost Of Revenues (LTM)", "Total Revenues (LTM)") * 100         AS cogs_to_revenue,
       safe_divide("Total Operating Expenses (LTM)", "Total Revenues (LTM)") * 100 AS opex_to_revenue,
       safe_divide("Selling General & Admin Expenses/Total (FY)", "Total Revenues (FY)") * 100
                                                                                   AS sga_to_revenue,
       safe_divide("R&D Expenses (LTM)", "Total Revenues (LTM)") * 100             AS rnd_to_revenue,
       safe_divide("Interest Expense/Total (LTM)", "Total Revenues (LTM)") * 100   AS interest_to_revenue,
       -- SG&A trend using available FY columns
       (safe_divide("Selling General & Admin Expenses/Total (FY)", "Total Revenues (FY)") -
        safe_divide("Selling General & Admin Expenses/Total (-1FY)", "Total Revenues (-1FY)")) * 100
                                                                                   AS sga_trend_yoy,
       CASE
           WHEN calc_change_ratio("Total Revenues (FY)", "Total Revenues (-1FY)") > 0
               THEN safe_divide(
                   calc_change_ratio("Operating Income (FY)", "Operating Income (-1FY)"),
                   calc_change_ratio("Total Revenues (FY)", "Total Revenues (-1FY)")
                    )
           END                                                                     AS operating_leverage_proxy,
       clamp_score(
               100 - safe_divide("Cost Of Revenues (LTM)", "Total Revenues (LTM)") * 100 * 0.5 -
               safe_divide("Total Operating Expenses (LTM)", "Total Revenues (LTM)") * 100 * 0.3
       )                                                                           AS cost_efficiency_score,
       -- NEW: Marketing efficiency metrics using schema columns
       safe_divide("Marketing Expenses (FY)", "Total Revenues (FY)") * 100         AS marketing_to_revenue,
       pct_change("Marketing Expenses (FY)", "Marketing Expenses (-1FY)")          AS marketing_trend_yoy,
       safe_divide("Marketing Expenses (FY)", "Marketing Expenses (5YAVGLTM)")     AS marketing_vs_5y_avg,
       -- NEW: SG&A vs 5Y average
       safe_divide("Selling General & Admin Expenses/Total (FQ)",
                   "Selling General & Admin Expenses/Total (5YAVGFQ)")             AS sga_vs_5y_avg,
       -- NEW: SG&A efficiency trend (lower ratio = better efficiency)
       (safe_divide("Selling General & Admin Expenses/Total (-1FY)", "Total Revenues (-1FY)") -
        safe_divide("Selling General & Admin Expenses/Total (FY)", "Total Revenues (FY)")) * 100
                                                                                   AS sga_efficiency_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Dividend Yield Comprehensive
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_dividend_yield_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                div_yield_ltm             NUMERIC,
                div_yield_ntm             NUMERIC,
                div_yield_ind             NUMERIC,
                div_yield_1fy_ind         NUMERIC,
                div_yield_5y_avg          NUMERIC,
                div_yield_vs_5y_avg       NUMERIC,
                div_yield_growth_expected NUMERIC,
                dividend_streak           INTEGER,
                high_yield_flag           INTEGER,
                sustainable_dividend_flag INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                AS isin,
       "Div Yield (LTM)"                                     AS div_yield_ltm,
       "Div Yield (NTM)"                                     AS div_yield_ntm,
       "Div Yield (Ind)"                                     AS div_yield_ind,
       "Div Yield (-1FYInd)"                                 AS div_yield_1fy_ind,
       "Div Yield (5YAVGLTM)"                                AS div_yield_5y_avg,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0) AS div_yield_vs_5y_avg,
       ("Div Yield (NTM)" - "Div Yield (LTM)") / NULLIF("Div Yield (LTM)", 0) * 100
                                                             AS div_yield_growth_expected,
       "Dividend Streak"::INTEGER                            AS dividend_streak,
       CASE WHEN "Div Yield (LTM)" > 4 THEN 1 ELSE 0 END     AS high_yield_flag,
       CASE
           WHEN "Div Yield (LTM)" > 0
               AND "FCF (LTM)" > ABS(COALESCE("Common Dividends Paid (LTM)", 0))
               AND "Dividend Streak" >= 5
               THEN 1
           ELSE 0
           END                                               AS sustainable_dividend_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Interest Income Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_interest_income_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                interest_income_ltm         NUMERIC,
                interest_expense_ltm        NUMERIC,
                net_interest_income         NUMERIC,
                interest_coverage_ratio     NUMERIC,
                interest_income_to_revenue  NUMERIC,
                interest_expense_to_revenue NUMERIC,
                net_interest_margin_proxy   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       "Interest Income On Investments (LTM)"                                   AS interest_income_ltm,
       "Interest Expense/Total (LTM)"                                           AS interest_expense_ltm,
       COALESCE("Interest Income On Investments (LTM)", 0) -
       COALESCE("Interest Expense/Total (LTM)", 0)                              AS net_interest_income,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                 AS interest_coverage_ratio,
       "Interest Income On Investments (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100
                                                                                AS interest_income_to_revenue,
       "Interest Expense/Total (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100 AS interest_expense_to_revenue,
       (COALESCE("Interest Income On Investments (LTM)", 0) -
        COALESCE("Interest Expense/Total (LTM)", 0)) /
       NULLIF("Total Assets (LTM)", 0) * 100                                    AS net_interest_margin_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Long Term Momentum Features (REFACTORED)
-- Uses available price columns (1Y, 3Y, 5Y) and helper functions
-- Note: Price (2Y Ago) not available in schema, adjusted weights accordingly
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_long_term_momentum_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                price_momentum_1y     NUMERIC,
                price_momentum_3y     NUMERIC,
                price_momentum_5y     NUMERIC,
                long_term_trend_score NUMERIC,
                price_vs_ema_250d     NUMERIC,
                multi_year_high_flag  INTEGER,
                secular_trend_flag    INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                     AS isin,
       pct_change("Last Price", "Price (1Y Ago)") AS price_momentum_1y,
       pct_change("Last Price", "Price (3Y Ago)") AS price_momentum_3y,
       pct_change("Last Price", "Price (5Y Ago)") AS price_momentum_5y,
       -- Weighted trend score using available periods (1Y: 50%, 3Y: 30%, 5Y: 20%)
       (COALESCE(pct_change("Last Price", "Price (1Y Ago)"), 0) * 0.50 +
        COALESCE(pct_change("Last Price", "Price (3Y Ago)"), 0) * 0.30 +
        COALESCE(pct_change("Last Price", "Price (5Y Ago)"), 0) * 0.20) / 100
                                                  AS long_term_trend_score,
       pct_change("Last Price", "EMA (250D)")     AS price_vs_ema_250d,
       CASE
           WHEN calc_change_ratio("52W High/Adj" - "Last Price", "52W High/Adj") <= 0.10
               AND calc_change_ratio("Last Price", "Price (3Y Ago)") > 0.5
               THEN 1
           ELSE 0
           END                                    AS multi_year_high_flag,
       CASE
           WHEN calc_change_ratio("Last Price", "Price (3Y Ago)") > 0.20
               AND calc_change_ratio("Last Price", "Price (1Y Ago)") > 0
               AND "EMA (50D)" > "EMA (250D)"
               THEN 1
           ELSE 0
           END                                    AS secular_trend_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Revenue Estimate Consensus (REFACTORED)
-- Uses available estimate columns (Avg and Med only)
-- Note: High, Low, # columns not available in schema
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_revenue_estimate_consensus(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                revenue_est_avg_fy1e       NUMERIC,
                revenue_est_med_fy1e       NUMERIC,
                revenue_est_avg_ntm        NUMERIC,
                revenue_est_med_ntm        NUMERIC,
                revenue_avg_med_diff_pct   NUMERIC,
                revenue_consensus_strength NUMERIC,
                revenue_revision_trend     NUMERIC,
                revenue_vs_current         NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                           AS isin,
       "Revenues - Est Avg (FY1E)"                                      AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"                                      AS revenue_est_med_fy1e,
       "Revenues - Est Avg (NTM)"                                       AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"                                       AS revenue_est_med_ntm,
       -- Difference between avg and median as proxy for estimate dispersion
       safe_divide("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)",
                   "Revenues - Est Med (FY1E)") * 100                   AS revenue_avg_med_diff_pct,
       -- Consensus strength: closer avg to median = stronger consensus
       clamp_score(
               100 - ABS(safe_divide("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)",
                                     "Revenues - Est Med (FY1E)") * 100) * 2
       )                                                                AS revenue_consensus_strength,
       "Revenues - Est YoY % (FY1E)"                                    AS revenue_revision_trend,
       -- Compare estimate to current revenue
       safe_divide("Revenues - Est Avg (FY1E)", "Total Revenues (LTM)") AS revenue_vs_current
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Revenue Features (REFACTORED)
-- Uses available revenue columns (FQ, FY, -1FY, LTM, 5YAVG)
-- Note: Quarterly historical columns (-1FQFQ to -4FQFQ) not available in schema
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_revenue_quarterly_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                -- Base revenue values
                revenue_fq                  NUMERIC,
                revenue_fy                  NUMERIC,
                revenue_ltm                 NUMERIC,
                revenue_5y_avg              NUMERIC,
                -- NEW: Quarterly historical values
                revenue_1fqfq               NUMERIC,
                revenue_2fqfq               NUMERIC,
                revenue_3fqfq               NUMERIC,
                revenue_4fqfq               NUMERIC,
                -- NEW: Extended yearly historical
                revenue_1fy                 NUMERIC,
                revenue_2fy                 NUMERIC,
                revenue_3fy                 NUMERIC,
                revenue_4fy                 NUMERIC,
                -- Growth metrics
                revenue_yoy_growth          NUMERIC,
                revenue_vs_5y_avg           NUMERIC,
                revenue_ltm_vs_fy           NUMERIC,
                revenue_fq_vs_5y_avg_fq     NUMERIC,
                -- NEW: Quarterly momentum metrics
                revenue_qoq_growth          NUMERIC,
                revenue_qoq_2q              NUMERIC,
                revenue_qoq_3q              NUMERIC,
                revenue_qoq_4q              NUMERIC,
                -- NEW: YoY quarterly comparison
                revenue_yoy_quarterly       NUMERIC,
                -- NEW: Multi-year growth
                revenue_2y_growth           NUMERIC,
                revenue_3y_growth           NUMERIC,
                revenue_4y_growth           NUMERIC,
                -- NEW: CAGR calculations
                revenue_cagr_3y             NUMERIC,
                revenue_cagr_4y             NUMERIC,
                -- NEW: Quarterly trend analysis
                revenue_4q_trend            NUMERIC,
                revenue_4q_avg              NUMERIC,
                revenue_fq_vs_4q_avg        NUMERIC,
                -- Flags and scores
                revenue_growth_flag         INTEGER,
                revenue_stability_score     NUMERIC,
                -- NEW: Additional flags
                revenue_accelerating_flag   INTEGER,
                revenue_positive_qoq_streak INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                     AS isin,
       -- Base revenue values
       "Total Revenues (FQ)"                                                      AS revenue_fq,
       "Total Revenues (FY)"                                                      AS revenue_fy,
       "Total Revenues (LTM)"                                                     AS revenue_ltm,
       "Total Revenues (5YAVGLTM)"                                                AS revenue_5y_avg,
       -- Quarterly historical values
       "Total Revenues (-1FQFQ)"                                                  AS revenue_1fqfq,
       "Total Revenues (-2FQFQ)"                                                  AS revenue_2fqfq,
       "Total Revenues (-3FQFQ)"                                                  AS revenue_3fqfq,
       "Total Revenues (-4FQFQ)"                                                  AS revenue_4fqfq,
       -- Extended yearly historical
       "Total Revenues (-1FY)"                                                    AS revenue_1fy,
       "Total Revenues (-2FY)"                                                    AS revenue_2fy,
       "Total Revenues (-3FY)"                                                    AS revenue_3fy,
       "Total Revenues (-4FY)"                                                    AS revenue_4fy,
       -- Year-over-year growth using FY data
       pct_change("Total Revenues (FY)", "Total Revenues (-1FY)")                 AS revenue_yoy_growth,
       -- Current vs 5-year average
       safe_divide("Total Revenues (LTM)", "Total Revenues (5YAVGLTM)")           AS revenue_vs_5y_avg,
       -- LTM vs FY comparison
       safe_divide("Total Revenues (LTM)", "Total Revenues (FY)")                 AS revenue_ltm_vs_fy,
       -- FQ vs 5-year average FQ
       safe_divide("Total Revenues (FQ)", "Total Revenues (5YAVGFQ)")             AS revenue_fq_vs_5y_avg_fq,
       -- Quarterly momentum: QoQ growth rates
       pct_change("Total Revenues (FQ)", "Total Revenues (-1FQFQ)")               AS revenue_qoq_growth,
       pct_change("Total Revenues (-1FQFQ)", "Total Revenues (-2FQFQ)")           AS revenue_qoq_2q,
       pct_change("Total Revenues (-2FQFQ)", "Total Revenues (-3FQFQ)")           AS revenue_qoq_3q,
       pct_change("Total Revenues (-3FQFQ)", "Total Revenues (-4FQFQ)")           AS revenue_qoq_4q,
       -- YoY quarterly comparison (current FQ vs same quarter last year)
       pct_change("Total Revenues (FQ)", "Total Revenues (-4FQFQ)")               AS revenue_yoy_quarterly,
       -- Multi-year growth rates
       pct_change("Total Revenues (FY)", "Total Revenues (-2FY)")                 AS revenue_2y_growth,
       pct_change("Total Revenues (FY)", "Total Revenues (-3FY)")                 AS revenue_3y_growth,
       pct_change("Total Revenues (FY)", "Total Revenues (-4FY)")                 AS revenue_4y_growth,
       -- CAGR calculations
       CASE
           WHEN "Total Revenues (-3FY)" > 0 AND "Total Revenues (FY)" > 0
               THEN (POWER(safe_divide("Total Revenues (FY)", "Total Revenues (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                                    AS revenue_cagr_3y,
       CASE
           WHEN "Total Revenues (-4FY)" > 0 AND "Total Revenues (FY)" > 0
               THEN (POWER(safe_divide("Total Revenues (FY)", "Total Revenues (-4FY)"), 1.0 / 4.0) - 1) * 100
           END                                                                    AS revenue_cagr_4y,
       -- Quarterly trend: FQ vs 4 quarters ago
       pct_change("Total Revenues (FQ)", "Total Revenues (-4FQFQ)")               AS revenue_4q_trend,
       -- Trailing 4-quarter average
       ("Total Revenues (FQ)" + "Total Revenues (-1FQFQ)" +
        "Total Revenues (-2FQFQ)" + "Total Revenues (-3FQFQ)") / 4.0              AS revenue_4q_avg,
       -- FQ vs trailing 4Q average
       safe_divide("Total Revenues (FQ)",
                   ("Total Revenues (FQ)" + "Total Revenues (-1FQFQ)" +
                    "Total Revenues (-2FQFQ)" + "Total Revenues (-3FQFQ)") / 4.0) AS revenue_fq_vs_4q_avg,
       -- Growth flag: 1 if growing YoY
       CASE
           WHEN "Total Revenues (FY)" > "Total Revenues (-1FY)" THEN 1
           ELSE 0
           END                                                                    AS revenue_growth_flag,
       -- Revenue stability: how close LTM is to 5Y average
       clamp_score(
               100 - ABS(safe_divide("Total Revenues (LTM)" - "Total Revenues (5YAVGLTM)",
                                     "Total Revenues (5YAVGLTM)")) * 100
       )                                                                          AS revenue_stability_score,
       -- Accelerating growth flag: recent growth > historical growth
       CASE
           WHEN pct_change("Total Revenues (FY)", "Total Revenues (-1FY)") >
                pct_change("Total Revenues (-1FY)", "Total Revenues (-2FY)")
               THEN 1
           ELSE 0
           END                                                                    AS revenue_accelerating_flag,
       -- Positive QoQ streak count
       (CASE WHEN "Total Revenues (FQ)" > "Total Revenues (-1FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-1FQFQ)" > "Total Revenues (-2FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-2FQFQ)" > "Total Revenues (-3FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-3FQFQ)" > "Total Revenues (-4FQFQ)" THEN 1 ELSE 0 END)::INTEGER
                                                                                  AS revenue_positive_qoq_streak
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Tangible Book Features (REFACTORED)
-- Uses native TBV columns from schema for accuracy
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_tangible_book_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                -- Use native schema columns directly
                tangible_book_value_fy  NUMERIC,
                tangible_book_value_ltm NUMERIC,
                -- Calculated metrics
                tangible_book_per_share NUMERIC,
                price_to_tangible_book  NUMERIC,
                tangible_equity_ratio   NUMERIC,
                intangibles_to_equity   NUMERIC,
                goodwill_to_equity      NUMERIC,
                tangible_asset_quality  NUMERIC,
                -- NEW: TBV growth metrics
                tbv_yoy_growth          NUMERIC,
                tbv_vs_calculated       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                AS isin,
       -- Use native TBV columns from schema (more accurate than calculation)
       "TBV (FY)"                                                            AS tangible_book_value_fy,
       "TBV (LTM)"                                                           AS tangible_book_value_ltm,
       -- Per share using native TBV
       "TBV (LTM)" / NULLIF("Shrs Out", 0)                                   AS tangible_book_per_share,
       -- P/TBV using native column (already in schema as P/TBV (LTM))
       "P/TBV (LTM)"                                                         AS price_to_tangible_book,
       -- Tangible equity ratio using native TBV
       "TBV (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                   AS tangible_equity_ratio,
       COALESCE("Gross Intangible Assets (LTM)", 0) / NULLIF("Total Equity (LTM)", 0) * 100
                                                                             AS intangibles_to_equity,
       COALESCE("Goodwill (LTM)", 0) / NULLIF("Total Equity (LTM)", 0) * 100 AS goodwill_to_equity,
       GREATEST(0, LEAST(100,
                         100 - (COALESCE("Goodwill (LTM)", 0) + COALESCE("Gross Intangible Assets (LTM)", 0)) /
                               NULLIF("Total Assets (LTM)", 0) * 100
                   ))                                                        AS tangible_asset_quality,
       -- NEW: TBV growth (FY to LTM)
       pct_change("TBV (LTM)", "TBV (FY)")                                   AS tbv_yoy_growth,
       -- Validation: compare native TBV to calculated (should be ~1.0)
       safe_divide("TBV (LTM)",
                   "Total Equity (LTM)" - COALESCE("Goodwill (LTM)", 0) -
                   COALESCE("Gross Intangible Assets (LTM)", 0))             AS tbv_vs_calculated
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Inventory Temporal Features (NEW)
-- Full historical coverage for inventory trend analysis
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_inventory_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                -- Current values
                inventory_ltm            NUMERIC,
                inventory_fq             NUMERIC,
                inventory_fy             NUMERIC,
                -- Quarterly historical
                inventory_1fq            NUMERIC,
                inventory_2fq            NUMERIC,
                inventory_3fq            NUMERIC,
                inventory_4fq            NUMERIC,
                -- Yearly historical
                inventory_1fy            NUMERIC,
                inventory_2fy            NUMERIC,
                inventory_3fy            NUMERIC,
                inventory_4fy            NUMERIC,
                -- Trend metrics
                inventory_qoq_change     NUMERIC,
                inventory_yoy_change     NUMERIC,
                inventory_4q_trend       NUMERIC,
                inventory_vs_5y_avg      NUMERIC,
                -- Efficiency metrics
                inventory_days           NUMERIC,
                inventory_turnover       NUMERIC,
                inventory_to_revenue     NUMERIC,
                inventory_to_assets      NUMERIC,
                -- Quality flags
                inventory_buildup_flag   INTEGER,
                inventory_reduction_flag INTEGER,
                inventory_volatility     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                          AS isin,
       -- Current values
       "Inventory (LTM)"                                               AS inventory_ltm,
       "Inventory (FQ)"                                                AS inventory_fq,
       "Inventory (FY)"                                                AS inventory_fy,
       -- Quarterly historical
       "Inventory (-1FQ)"                                              AS inventory_1fq,
       "Inventory (-2FQ)"                                              AS inventory_2fq,
       "Inventory (-3FQ)"                                              AS inventory_3fq,
       "Inventory (-4FQ)"                                              AS inventory_4fq,
       -- Yearly historical
       "Inventory (-1FY)"                                              AS inventory_1fy,
       "Inventory (-2FY)"                                              AS inventory_2fy,
       "Inventory (-3FY)"                                              AS inventory_3fy,
       "Inventory (-4FY)"                                              AS inventory_4fy,
       -- Trend metrics
       pct_change("Inventory (FQ)", "Inventory (-1FQ)")                AS inventory_qoq_change,
       pct_change("Inventory (FY)", "Inventory (-1FY)")                AS inventory_yoy_change,
       pct_change("Inventory (FQ)", "Inventory (-4FQ)")                AS inventory_4q_trend,
       safe_divide("Inventory (FQ)", "Inventory (5YAVGFQ)")            AS inventory_vs_5y_avg,
       -- Efficiency metrics
       "Inventory (LTM)" / NULLIF("Cost Of Revenues (LTM)" / 365.0, 0) AS inventory_days,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)         AS inventory_turnover,
       safe_divide("Inventory (LTM)", "Total Revenues (LTM)") * 100    AS inventory_to_revenue,
       safe_divide("Inventory (LTM)", "Total Assets (LTM)") * 100      AS inventory_to_assets,
       -- Inventory buildup flag (rising faster than revenue)
       CASE
           WHEN pct_change("Inventory (FQ)", "Inventory (-4FQ)") >
                pct_change("Total Revenues (FQ)", "Total Revenues (-4FQFQ)") + 10
               THEN 1
           ELSE 0 END                                                  AS inventory_buildup_flag,
       -- Inventory reduction flag (declining)
       CASE
           WHEN "Inventory (FQ)" < "Inventory (-1FQ)"
               AND "Inventory (-1FQ)" < "Inventory (-2FQ)"
               THEN 1
           ELSE 0 END                                                  AS inventory_reduction_flag,
       -- Volatility (coefficient of variation)
       (ABS("Inventory (FQ)" - "Inventory (-1FQ)") +
        ABS("Inventory (-1FQ)" - "Inventory (-2FQ)") +
        ABS("Inventory (-2FQ)" - "Inventory (-3FQ)") +
        ABS("Inventory (-3FQ)" - "Inventory (-4FQ)")) /
       NULLIF(ABS(("Inventory (FQ)" + "Inventory (-1FQ)" + "Inventory (-2FQ)" +
                   "Inventory (-3FQ)" + "Inventory (-4FQ)") / 5.0), 0) AS inventory_volatility
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Goodwill Temporal Features (NEW)
-- M&A activity tracking through goodwill changes
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_goodwill_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                -- Current values
                goodwill_fq                NUMERIC,
                goodwill_ltm               NUMERIC,
                goodwill_fy                NUMERIC,
                -- Quarterly historical
                goodwill_1fq               NUMERIC,
                goodwill_2fq               NUMERIC,
                goodwill_3fq               NUMERIC,
                goodwill_4fq               NUMERIC,
                -- Yearly historical
                goodwill_1fy               NUMERIC,
                goodwill_2fy               NUMERIC,
                goodwill_3fy               NUMERIC,
                goodwill_4fy               NUMERIC,
                -- Trend metrics
                goodwill_qoq_change        NUMERIC,
                goodwill_yoy_change        NUMERIC,
                goodwill_3y_growth         NUMERIC,
                goodwill_vs_5y_avg         NUMERIC,
                -- M&A activity indicators
                recent_acquisition_flag    INTEGER,
                goodwill_accumulation_rate NUMERIC,
                goodwill_to_assets_trend   NUMERIC,
                -- Risk metrics
                impairment_risk_score      NUMERIC,
                goodwill_concentration     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                        AS isin,
       -- Current values
       "Goodwill (FQ)"                                               AS goodwill_fq,
       "Goodwill (LTM)"                                              AS goodwill_ltm,
       "Goodwill (FY)"                                               AS goodwill_fy,
       -- Quarterly historical
       "Goodwill (-1FQ)"                                             AS goodwill_1fq,
       "Goodwill (-2FQ)"                                             AS goodwill_2fq,
       "Goodwill (-3FQ)"                                             AS goodwill_3fq,
       "Goodwill (-4FQ)"                                             AS goodwill_4fq,
       -- Yearly historical
       "Goodwill (-1FY)"                                             AS goodwill_1fy,
       "Goodwill (-2FY)"                                             AS goodwill_2fy,
       "Goodwill (-3FY)"                                             AS goodwill_3fy,
       "Goodwill (-4FY)"                                             AS goodwill_4fy,
       -- Trend metrics
       pct_change("Goodwill (FQ)", "Goodwill (-1FQ)")                AS goodwill_qoq_change,
       pct_change("Goodwill (FY)", "Goodwill (-1FY)")                AS goodwill_yoy_change,
       pct_change("Goodwill (FY)", "Goodwill (-3FY)")                AS goodwill_3y_growth,
       safe_divide("Goodwill (FQ)", "Goodwill (5YAVGFQ)")            AS goodwill_vs_5y_avg,
       -- Recent acquisition flag (goodwill increased significantly)
       CASE
           WHEN pct_change("Goodwill (FQ)", "Goodwill (-1FQ)") > 20
               THEN 1
           ELSE 0 END                                                AS recent_acquisition_flag,
       -- Goodwill accumulation rate (avg annual increase)
       CASE
           WHEN "Goodwill (-3FY)" > 0
               THEN (POWER(safe_divide("Goodwill (FY)", "Goodwill (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                       AS goodwill_accumulation_rate,
       -- Goodwill to assets trend (increasing concentration risk)
       (safe_divide("Goodwill (FY)", "Total Assets (FY)") -
        safe_divide("Goodwill (-1FY)", "Total Assets (-1FY)")) * 100 AS goodwill_to_assets_trend,
       -- Impairment risk score (high goodwill + declining earnings = risk)
       CASE
           WHEN "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.25
               AND "Net Income - (IS) (FY)" < "Net Income - (IS) (-1FY)"
               THEN clamp_score(
                   ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0)) * 200 +
                   ABS(pct_change("Net Income - (IS) (FY)", "Net Income - (IS) (-1FY)")) * 0.5
                    )
           ELSE clamp_score(
                   ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0)) * 100
                )
           END                                                       AS impairment_risk_score,
       -- Goodwill concentration (relative to equity)
       safe_divide("Goodwill (LTM)", "Total Equity (LTM)") * 100     AS goodwill_concentration
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- R&D Investment Temporal Features (NEW)
-- Innovation investment trends and efficiency
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_rnd_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                -- Current values
                rnd_ltm                 NUMERIC,
                rnd_fq                  NUMERIC,
                rnd_fy                  NUMERIC,
                -- Quarterly historical
                rnd_1fqfq               NUMERIC,
                rnd_2fqfq               NUMERIC,
                rnd_3fqfq               NUMERIC,
                rnd_4fqfq               NUMERIC,
                -- Yearly historical
                rnd_1fy                 NUMERIC,
                rnd_2fy                 NUMERIC,
                rnd_3fy                 NUMERIC,
                rnd_4fy                 NUMERIC,
                -- Intensity metrics
                rnd_intensity_ltm       NUMERIC,
                rnd_intensity_fy        NUMERIC,
                rnd_intensity_trend     NUMERIC,
                -- Growth metrics
                rnd_qoq_growth          NUMERIC,
                rnd_yoy_growth          NUMERIC,
                rnd_cagr_3y             NUMERIC,
                -- Efficiency metrics
                rnd_per_employee        NUMERIC,
                rnd_to_gross_profit     NUMERIC,
                rnd_roi_proxy           NUMERIC,
                -- Investment flags
                rnd_increasing_flag     INTEGER,
                rnd_cut_flag            INTEGER,
                high_rnd_intensity_flag INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                              AS isin,
       -- Current values
       "R&D Expenses (LTM)"                                                AS rnd_ltm,
       "R&D Expenses (FQ)"                                                 AS rnd_fq,
       "R&D Expenses (FY)"                                                 AS rnd_fy,
       -- Quarterly historical
       "R&D Expenses (-1FQFQ)"                                             AS rnd_1fqfq,
       "R&D Expenses (-2FQFQ)"                                             AS rnd_2fqfq,
       "R&D Expenses (-3FQFQ)"                                             AS rnd_3fqfq,
       "R&D Expenses (-4FQFQ)"                                             AS rnd_4fqfq,
       -- Yearly historical
       "R&D Expenses (-1FY)"                                               AS rnd_1fy,
       "R&D Expenses (-2FY)"                                               AS rnd_2fy,
       "R&D Expenses (-3FY)"                                               AS rnd_3fy,
       "R&D Expenses (-4FY)"                                               AS rnd_4fy,
       -- Intensity metrics (R&D / Revenue)
       safe_divide("R&D Expenses (LTM)", "Total Revenues (LTM)") * 100     AS rnd_intensity_ltm,
       safe_divide("R&D Expenses (FY)", "Total Revenues (FY)") * 100       AS rnd_intensity_fy,
       -- Intensity trend (increasing R&D commitment)
       (safe_divide("R&D Expenses (FY)", "Total Revenues (FY)") -
        safe_divide("R&D Expenses (-1FY)", "Total Revenues (-1FY)")) * 100 AS rnd_intensity_trend,
       -- Growth metrics
       pct_change("R&D Expenses (FQ)", "R&D Expenses (-1FQFQ)")            AS rnd_qoq_growth,
       pct_change("R&D Expenses (FY)", "R&D Expenses (-1FY)")              AS rnd_yoy_growth,
       CASE
           WHEN "R&D Expenses (-3FY)" > 0 AND "R&D Expenses (FY)" > 0
               THEN (POWER(safe_divide("R&D Expenses (FY)", "R&D Expenses (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                             AS rnd_cagr_3y,
       -- Efficiency metrics
       safe_divide("R&D Expenses (FY)", "Full Time Employees (FY)")        AS rnd_per_employee,
       safe_divide("R&D Expenses (LTM)", "Gross Profit (LTM)") * 100       AS rnd_to_gross_profit,
       -- R&D ROI proxy: revenue growth relative to R&D spend
       CASE
           WHEN "R&D Expenses (-1FY)" > 0
               THEN safe_divide(
                   pct_change("Total Revenues (FY)", "Total Revenues (-1FY)"),
                   safe_divide("R&D Expenses (-1FY)", "Total Revenues (-1FY)") * 100
                    )
           END                                                             AS rnd_roi_proxy,
       -- R&D increasing flag (4 consecutive quarterly increases)
       CASE
           WHEN "R&D Expenses (FQ)" > "R&D Expenses (-1FQFQ)"
               AND "R&D Expenses (-1FQFQ)" > "R&D Expenses (-2FQFQ)"
               AND "R&D Expenses (-2FQFQ)" > "R&D Expenses (-3FQFQ)"
               THEN 1
           ELSE 0 END                                                      AS rnd_increasing_flag,
       -- R&D cut flag (significant decline may signal distress)
       CASE
           WHEN pct_change("R&D Expenses (FY)", "R&D Expenses (-1FY)") < -15
               THEN 1
           ELSE 0 END                                                      AS rnd_cut_flag,
       -- High R&D intensity flag (tech/pharma typical >10%)
       CASE
           WHEN safe_divide("R&D Expenses (LTM)", "Total Revenues (LTM)") > 0.10
               THEN 1
           ELSE 0 END                                                      AS high_rnd_intensity_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Unusual Items Features (REFACTORED)
-- Uses "Other Unusual Items/Total (LTM)" - only LTM period available
-- Combines with other non-recurring items for comprehensive view
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_unusual_items_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                other_unusual_items_ltm   NUMERIC,
                impairment_goodwill_ltm   NUMERIC,
                asset_writedown_ltm       NUMERIC,
                restructuring_charges_ltm NUMERIC,
                total_unusual_items       NUMERIC,
                unusual_items_to_revenue  NUMERIC,
                unusual_items_to_ebitda   NUMERIC,
                has_unusual_items_flag    INTEGER,
                earnings_quality_impact   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                     AS isin,
       "Other Unusual Items/Total (LTM)"          AS other_unusual_items_ltm,
       "Impairment of Goodwill (LTM)"             AS impairment_goodwill_ltm,
       "Asset Writedown (LTM)"                    AS asset_writedown_ltm,
       "Restructuring Charges (LTM)"              AS restructuring_charges_ltm,
       -- Total unusual/non-recurring items
       COALESCE("Other Unusual Items/Total (LTM)", 0) +
       COALESCE("Impairment of Goodwill (LTM)", 0) +
       COALESCE("Asset Writedown (LTM)", 0) +
       COALESCE("Restructuring Charges (LTM)", 0) AS total_unusual_items,
       -- Unusual items as % of revenue
       safe_divide(
               ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                   COALESCE("Impairment of Goodwill (LTM)", 0) +
                   COALESCE("Asset Writedown (LTM)", 0) +
                   COALESCE("Restructuring Charges (LTM)", 0)),
               "Total Revenues (LTM)"
       ) * 100                                    AS unusual_items_to_revenue,
       -- Unusual items as % of EBITDA
       safe_divide(
               ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                   COALESCE("Impairment of Goodwill (LTM)", 0) +
                   COALESCE("Asset Writedown (LTM)", 0) +
                   COALESCE("Restructuring Charges (LTM)", 0)),
               ABS("EBITDA (LTM)")
       ) * 100                                    AS unusual_items_to_ebitda,
       -- Flag if any unusual items present
       CASE
           WHEN ABS(COALESCE("Other Unusual Items/Total (LTM)", 0)) +
                ABS(COALESCE("Impairment of Goodwill (LTM)", 0)) +
                ABS(COALESCE("Asset Writedown (LTM)", 0)) +
                ABS(COALESCE("Restructuring Charges (LTM)", 0)) > 0
               THEN 1
           ELSE 0 END                             AS has_unusual_items_flag,
       -- Earnings quality impact (higher = better quality, less impacted by unusual items)
       clamp_score(
               100 - safe_divide(
                             ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                                 COALESCE("Impairment of Goodwill (LTM)", 0) +
                                 COALESCE("Asset Writedown (LTM)", 0) +
                                 COALESCE("Restructuring Charges (LTM)", 0)),
                             ABS("Net Income - (IS) (LTM)")
                     ) * 100
       )                                          AS earnings_quality_impact
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Working Capital Deep Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_working_capital_deep_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                working_capital_ltm  NUMERIC,
                working_capital_fq   NUMERIC,
                working_capital_fy   NUMERIC,
                wc_to_revenue        NUMERIC,
                wc_to_assets         NUMERIC,
                wc_change_qoq        NUMERIC,
                wc_change_yoy        NUMERIC,
                days_working_capital NUMERIC,
                wc_efficiency_score  NUMERIC,
                negative_wc_flag     INTEGER,
                wc_improvement_flag  INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                              AS isin,
       "Working Capital (LTM)"                                             AS working_capital_ltm,
       "Working Capital (FQ)"                                              AS working_capital_fq,
       "Working Capital (FY)"                                              AS working_capital_fy,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100   AS wc_to_revenue,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100     AS wc_to_assets,
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0) * 100                        AS wc_change_qoq,
       ("Working Capital (FY)" - "Working Capital (-1FY)") /
       NULLIF(ABS("Working Capital (-1FY)"), 0) * 100                      AS wc_change_yoy,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)" / 365.0, 0) AS days_working_capital,
       GREATEST(0, LEAST(100,
                         50 + (CASE WHEN "Working Capital (LTM)" > 0 THEN 25 ELSE -25 END) +
                         (CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 15 ELSE 0 END) +
                         (CASE WHEN ("Working Capital (FQ)" - "Working Capital (FY)") > 0 THEN 10 ELSE -10 END)
                   ))                                                      AS wc_efficiency_score,
       CASE WHEN "Working Capital (LTM)" < 0 THEN 1 ELSE 0 END             AS negative_wc_flag,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (FY)"
               AND "Working Capital (FY)" > "Working Capital (-1FY)"
               THEN 1
           ELSE 0
           END                                                             AS wc_improvement_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- All Enhanced Features (Aggregation Function)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_all_enhanced_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin           TEXT,
                feature_count  INTEGER,
                reference_date TIMESTAMP
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"            AS isin,
       (SELECT COUNT(*)::INTEGER
        FROM information_schema.routines
        WHERE routine_name LIKE 'calc_%'
          AND routine_schema = 'public')
                         AS feature_count,
       CURRENT_TIMESTAMP AS reference_date
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- UNIFIED MATERIALIZED VIEW AND FEATURE REGISTRY
-- Integrates all feature calculation functions from CalcFinancialFeaturesSql.sql
-- =============================================================================

-- =============================================================================
-- SECTION 1: UNIFIED MATERIALIZED VIEW - ALL FEATURES
-- =============================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_all_stock_features CASCADE;

CREATE MATERIALIZED VIEW mv_all_stock_features AS
SELECT
    -- Identifier
    e."ISIN"                                                                                           AS isin,
    e."Ticker"                                                                                         AS ticker,
    e."Name"                                                                                           AS name,
    e."Region"                                                                                         AS region,
    e."Country"                                                                                        AS country,
    e."Trading Country"                                                                                AS trading_country,
    e."Exchange"                                                                                       AS exchange,
    e."Sector"                                                                                         AS sector,
    e."Industry"                                                                                       AS industry,
    e."Next Earnings"                                                                                  as next_earnings,
    e."Next Earnings (When)"                                                                           AS next_earnings_when,
    e."Next Earnings (Status)"                                                                         AS next_earnings_status,
    e."Current Fiscal Quarter"                                                                         AS current_fiscal_quarter,
    e."Next Fiscal Quarter"                                                                            AS next_fiscal_quarter,
    e."Next Earnings (Report)"                                                                         AS next_earnings_report,
    e."Earnings Report (Frequency)"                                                                    AS earnings_report_frequency,
    e."Income Statement Report Date"                                                                   AS income_statement_report_date,
    e."Next Income Statement Report Date"                                                              AS next_income_statement_report_date,
    e."Dividend Record (Currency)"                                                                     AS dividend_record_currency,
    e."Dividend Record (Amount)"                                                                       AS dividend_record_amount,
    e."Dividend Record (Frequency)"                                                                    AS dividend_record_frequency,
    e."Market Cap"                                                                                     AS market_cap,
    e."Enterprise Value"                                                                               AS enterprise_value,
    e."Last Price"                                                                                     AS last_price,
    e."Price Target"                                                                                   AS price_target,

    -- =========================================================================
    -- VALUATION FEATURES (calc_valuation_features)
    -- =========================================================================
    e."P/E (LTM)"                                                                                      AS p_e_ratio,
    e."P/B (LTM)"                                                                                      AS p_b_ratio,
    e."EV/EBITDA (LTM)"                                                                                AS ev_ebitda_ratio,
    e."EV/Sales (LTM)"                                                                                 AS ev_sales_ratio,
    e."Div Yield (LTM)"                                                                                AS dividend_yield,
    CASE
        WHEN e."Total Revenues/CAGR (5Y FY)" > 0
            THEN safe_divide(e."P/E (LTM)", e."Total Revenues/CAGR (5Y FY)")
        END                                                                                            AS peg_ratio,

    -- =========================================================================
    -- VALUATION TIMESERIES FEATURES (calc_valuation_timeseries_features)
    -- =========================================================================
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

    -- =========================================================================
    -- EXTENDED VALUATION TIMESERIES (calc_extended_valuation_timeseries)
    -- =========================================================================
    (e."EV/Sales (LTM)" - e."EV/Sales (-1FQLTM)") /
    NULLIF(e."EV/Sales (-1FQLTM)", 0)                                                                  AS ev_sales_qoq_1q,
    (e."P/E (LTM)" - e."P/E (5YAVGLTM)") / NULLIF(e."P/E (5YAVGLTM)", 0)                               AS p_e_vs_5y_avg_ext,
    (e."P/B (LTM)" - e."P/B (-1FY)") / NULLIF(e."P/B (-1FY)", 0)                                       AS p_b_momentum_yoy,
    (e."P/E (EST FY1)" - e."P/E (LTM)") / NULLIF(ABS(e."P/E (LTM)"), 0) *
    100                                                                                                AS forward_pe_premium,

    -- =========================================================================
    -- MOMENTUM FEATURES (calc_momentum_features)
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

    -- =========================================================================
    -- TECHNICAL ANALYSIS FEATURES (calc_technical_analysis_features)
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
    -- PROFITABILITY FEATURES (calc_profitability_features)
    -- =========================================================================
    e."Return On Equity % (LTM)"                                                                       AS roe,
    e."Return on Assets (ROA) % (LTM)"                                                                 AS roa,
    e."Gross Profit Margin % (LTM)"                                                                    AS gross_margin_pct,
    e."Operating Income (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                             AS operating_margin_pct,
    e."Net Income Margin % (LTM)"                                                                      AS net_margin_pct,
    e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                       AS ebitda_margin_pct,
    e."Net Income - (IS) (LTM)" / NULLIF(e."Total Equity (LTM)" + e."Total Debt (LTM)", 0) *
    100                                                                                                AS roic,
    e."R&D Expenses (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)                                       AS rnd_intensity,
    e."Total Assets (LTM)" / NULLIF(e."Total Equity (LTM)", 0)                                         AS equity_multiplier,

    -- =========================================================================
    -- MARGIN TRENDS (calc_margin_trends)
    -- =========================================================================
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

    -- =========================================================================
    -- QUALITY & RISK FEATURES (calc_quality_features)
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

    -- =========================================================================
    -- FINANCIAL DISTRESS FEATURES (calc_financial_distress_features)
    -- =========================================================================
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
    CASE WHEN e."Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END                                         AS accumulated_deficit_flag,
    CASE
        WHEN e."Cash And Equivalents (FQ)" / NULLIF(e."Total Operating Expenses (LTM)" / 12.0, 0) > 6 THEN 1
        ELSE 0
        END                                                                                            AS adequate_cash_buffer,

    -- =========================================================================
    -- ACCOUNTING QUALITY FEATURES (calc_accounting_quality_features)
    -- =========================================================================
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

    -- =========================================================================
    -- LEVERAGE & LIQUIDITY FEATURES (calc_leverage_features)
    -- =========================================================================
    e."Total Debt (LTM)" / NULLIF(e."Total Equity (LTM)", 0)                                           AS debt_to_equity,
    e."Total Debt (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                           AS debt_to_assets,
    e."Total Equity (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                         AS equity_ratio,
    e."EBIT (LTM)" / NULLIF(e."Interest Expense/Total (LTM)", 0)                                       AS interest_coverage,
    e."Cash And Equivalents (LTM)" /
    NULLIF(e."Total Current Liabilities (LTM)", 0)                                                     AS cash_ratio,
    e."Working Capital (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                      AS working_capital_ratio,

    -- =========================================================================
    -- EFFICIENCY RATIOS (calc_efficiency_ratios)
    -- =========================================================================
    e."Total Revenues (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                       AS asset_turnover,
    e."Cost Of Revenues (LTM)" / NULLIF(e."Inventory (LTM)", 0)                                        AS inventory_turnover,
    (e."Accounts Receivable/Total (FY)" /
     NULLIF(e."Total Revenues (FY)" / 365.0, 0))                                                       AS receivables_days,
    e."Total Revenues (LTM)" / NULLIF(e."Working Capital (LTM)", 0)                                    AS working_capital_turns,

    -- =========================================================================
    -- BALANCE SHEET DYNAMICS (calc_balance_sheet_dynamics)
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

    -- =========================================================================
    -- ANALYST SENTIMENT FEATURES (calc_sentiment_features)
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
    -- NEW: Neutral sentiment (Hold ratings percentage)
    CASE
        WHEN (e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
              e."# Strong Sell Ratings") > 0
            THEN e."# Hold Ratings" /
                 NULLIF(e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" + e."# Sell Ratings" +
                        e."# Strong Sell Ratings", 0) * 100
        END                                                                                            AS analyst_neutral_pct,
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

    -- =========================================================================
    -- PRICE TARGET DYNAMICS (calc_price_target_dynamics)
    -- =========================================================================
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
    (e."Price Target - #" - e."Price Target - # (1M Ago)")::INTEGER                                    AS analyst_coverage_change_1m,
    (e."Price Target - #" - e."Price Target - # (3M Ago)")::INTEGER                                    AS analyst_coverage_change_3m,
    (e."Price Target - #" - e."Price Target - # (1Y Ago)")::INTEGER                                    AS analyst_coverage_change_1y,

    -- =========================================================================
    -- EARNINGS FEATURES (calc_earnings_features)
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

    -- =========================================================================
    -- EPS TRAJECTORY FEATURES (calc_eps_trajectory_features)
    -- =========================================================================
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


    -- =========================================================================
    -- GAAP ADJUSTED ANALYTICS (calc_gaap_adjusted_analytics) - ENHANCED
    -- =========================================================================
    -- EPS Adjustment Spreads
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

    -- Net Income Adjustment Ratios
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
    (e."Net Income/Adj. (LTM)" - e."Net Income - (IS) (LTM)") / NULLIF(ABS(e."Net Income - (IS) (LTM)"), 0) *
    100                                                                                                AS net_income_adjustment_pct,

    -- EBITDA Adjustment Percentages
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

    -- EBIT Adjustment Percentages
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

    -- Quality Scores
    GREATEST(0, LEAST(100, 100 - ABS((e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") /
                                     NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) *
                                     100)))                                                            AS earnings_quality_score,
    CASE
        WHEN ABS((e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") / NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) * 100) >
             15 THEN 1
        ELSE 0
        END                                                                                            AS earnings_quality_warning,
    e."EPS Norm - Est Avg (FY1E)" - e."EPS GAAP - Est Avg (FY1E)"                                      AS forward_eps_gaap_adj_spread,

    -- =========================================================================
    -- GAAP REVISION FEATURES (calc_gaap_revision_features)
    -- =========================================================================
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
        ELSE 0
        END                                                                                            AS gaap_positive_revision_flag,

    -- =========================================================================
    -- GROWTH FEATURES (calc_growth_features)
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

    -- =========================================================================
    -- REVENUE FORECAST FEATURES (calc_revenue_forecast_features)
    -- =========================================================================
    (e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)") / NULLIF(e."Revenues - Est Med (FY1E)", 0) *
    100                                                                                                AS revenue_est_spread,
    (e."Total Revenues (LTM)" - e."Revenues - Est Avg (FY1E)") / NULLIF(ABS(e."Revenues - Est Avg (FY1E)"), 0) *
    100                                                                                                AS revenue_beat_potential,
    e."Revenues - Est YoY % (FY1E)"                                                                    AS revenue_est_revision_trend,
    (e."EBITDA (LTM)" - e."EBITDA - Est Avg (FY1E)") / NULLIF(ABS(e."EBITDA - Est Avg (FY1E)"), 0) *
    100                                                                                                AS ebitda_est_vs_actual,
    e."Enterprise Value" / NULLIF(e."Revenues - Est Avg (FY1E)", 0)                                    AS forward_revenue_multiple,
    e."EPS Norm - Est # (FY1E)"                                                                        AS revenue_estimate_count,
    (e."Revenues - Est Avg (NTM)" - e."Revenues - Est Avg (FY1E)") / NULLIF(ABS(e."Revenues - Est Avg (FY1E)"), 0) *
    100                                                                                                AS revenue_guidance_gap,
    (e."Revenues - Est Avg (FY1E)" - e."Total Revenues (FY)") / NULLIF(ABS(e."Total Revenues (FY)"), 0) *
    100                                                                                                AS consensus_revenue_growth,
    e."EBITDA - Est Avg (FY1E)" / NULLIF(e."Revenues - Est Avg (FY1E)", 0) *
    100                                                                                                AS forward_ebitda_margin,
    e."Revenues - Est YoY % (FY1E)" - e."Total Revenues/CAGR (5Y FY)"                                  AS revenue_acceleration,
    GREATEST(0, LEAST(100, 100 - ABS((e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)") /
                                     NULLIF(e."Revenues - Est Med (FY1E)", 0) *
                                     100)))                                                            AS estimate_confidence_score,

    -- =========================================================================
    -- DIVIDEND FEATURES (calc_dividend_features)
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

    -- =========================================================================
    -- DIVIDEND TIMING (calc_dividend_timing)
    -- =========================================================================
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
    e."Div Yield (LTM)" / NULLIF(e."Div Yield (5YAVGLTM)", 0)                                          AS dividend_yield_vs_5y_avg,

    -- =========================================================================
    -- EMPLOYMENT FEATURES (calc_employment_features)
    -- =========================================================================
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."Total Revenues (FY)" / NULLIF(e."Full Time Employees (FY)", 0) END                 AS revenue_per_employee,
    CASE
        WHEN e."Full Time Employees (FY)" > 0 THEN e."Normalized Net Income (FY)" /
                                                   NULLIF(e."Full Time Employees (FY)", 0) END         AS profit_per_employee,
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."EBITDA (FY)" / NULLIF(e."Full Time Employees (FY)", 0) END                         AS ebitda_per_employee,
    CASE
        WHEN e."Full Time Employees (FY)" > 0
            THEN e."Total Assets (FY)" / NULLIF(e."Full Time Employees (FY)", 0) END                   AS assets_per_employee,
    CASE
        WHEN e."Full Time Employees (-1FY)" > 0 THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                                                     NULLIF(e."Full Time Employees (-1FY)", 0) *
                                                     100 END                                           AS fte_growth_1y_pct,
    CASE
        WHEN e."Full Time Employees (-3FY)" > 0 THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-3FY)") /
                                                     NULLIF(e."Full Time Employees (-3FY)", 0) *
                                                     100 END                                           AS fte_growth_3y_pct,
    CASE
        WHEN e."Avg Employees (5YAVGFY)" > 0
            THEN e."Full Time Employees (FY)" / NULLIF(e."Avg Employees (5YAVGFY)", 0) END             AS workforce_stability,

    -- =========================================================================
    -- EMPLOYMENT DYNAMICS (calc_employment_dynamics)
    -- =========================================================================
    CASE
        WHEN e."Full Time Employees (-2FY)" > 0 THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-2FY)") /
                                                     NULLIF(e."Full Time Employees (-2FY)", 0) *
                                                     100 END                                           AS fte_growth_2y_pct,
    CASE
        WHEN e."Full Time Employees (FY)" < e."Full Time Employees (-1FY)" AND
             e."Total Revenues (FY)" < e."Total Revenues (-1FY)" THEN 1
        ELSE 0
        END                                                                                            AS layoff_risk_flag,
    CASE
        WHEN (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
             NULLIF(e."Full Time Employees (-1FY)", 0) > 0.20 THEN 1
        ELSE 0
        END                                                                                            AS rapid_hiring_flag,
    CASE
        WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") / NULLIF(ABS(e."Total Revenues (-1FY)"), 0) >
             (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") / NULLIF(e."Full Time Employees (-1FY)", 0)
            AND (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") > 0 THEN 1
        ELSE 0
        END                                                                                            AS sustainable_growth_flag,

    -- =========================================================================
    -- CASH FLOW FEATURES (calc_cashflow_features)
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
    ABS(COALESCE(e."Cash Acquisitions (-2FQFQ)", 0)) +
    ABS(COALESCE(e."Cash Acquisitions (-3FQFQ)", 0))                                                   AS acquisition_intensity,
    CASE
        WHEN ABS(e."CFI (LTM)") > 0
            THEN e."CFO (LTM)" / NULLIF(ABS(e."CFI (LTM)"), 0) END                                     AS self_funding_ratio,

    -- =========================================================================
    -- ENHANCED CASHFLOW FEATURES (calc_enhanced_cashflow_features)
    -- =========================================================================
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
        ELSE 0
        END                                                                                            AS fcf_always_positive,
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
    (ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) +
     ABS(COALESCE(e."Cash Acquisitions (-1FQFQ)", 0)) +
     ABS(COALESCE(e."Cash Acquisitions (-2FQFQ)", 0)) +
     ABS(COALESCE(e."Cash Acquisitions (-3FQFQ)", 0))) /
    NULLIF(ABS(e."FCF (LTM)"), 0)                                                                      AS acquisition_to_fcf,
    CASE
        WHEN (ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) +
              ABS(COALESCE(e."Cash Acquisitions (-1FQFQ)", 0)) +
              ABS(COALESCE(e."Cash Acquisitions (-2FQFQ)", 0)) +
              ABS(COALESCE(e."Cash Acquisitions (-3FQFQ)", 0))) /
             NULLIF(ABS(e."FCF (LTM)"), 0) < 0.5
            THEN 1
        ELSE 0
        END                                                                                            AS sustainable_ma_flag,
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
            AND ABS(e."Capital Expenditure (-1FY)") > ABS(e."Capital Expenditure (-2FY)")
            THEN 1
        ELSE 0
        END                                                                                            AS capex_acceleration,
    CASE
        WHEN (ABS(e."Capital Expenditure (FY)") - ABS(e."Capital Expenditure (-1FY)")) /
             NULLIF(ABS(e."Capital Expenditure (-1FY)"), 0) < -0.25
            THEN 1
        ELSE 0
        END                                                                                            AS capex_cut_flag,
    CASE
        WHEN ABS(e."Capital Expenditure (FQ)") / NULLIF(ABS(e."Capital Expenditure (5YAVGFQ)"), 0) > 1.5
            THEN 1
        ELSE 0
        END                                                                                            AS overinvestment_flag,
    (ABS(COALESCE(e."Cash Acquisitions (FY)", 0)) - ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0))) /
    NULLIF(ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0)), 0) *
    100                                                                                                AS acquisitions_yoy_growth,
    ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) /
    NULLIF(ABS(COALESCE(e."Cash Acquisitions (5YAVGFQ)", 0)), 0)                                       AS acquisitions_vs_5y_avg,
    ABS(COALESCE(e."Cash Acquisitions (LTM)", 0))                                                      AS acquisitions_ltm_total,
    ABS(COALESCE(e."Cash Acquisitions (LTM)", 0)) /
    NULLIF(e."Total Assets (LTM)", 0) *
    100                                                                                                AS ma_intensity_score,
    CASE
        WHEN (CASE WHEN ABS(COALESCE(e."Cash Acquisitions (FY)", 0)) > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0)) > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS(COALESCE(e."Cash Acquisitions (-2FY)", 0)) > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS(COALESCE(e."Cash Acquisitions (-3FY)", 0)) > 0 THEN 1 ELSE 0 END) >= 3
            THEN 1
        ELSE 0
        END                                                                                            AS serial_acquirer_flag,
    CASE
        WHEN ABS(COALESCE(e."Cash Acquisitions (FY)", 0)) = 0
            AND (ABS(COALESCE(e."Cash Acquisitions (-1FY)", 0)) > 0
                OR ABS(COALESCE(e."Cash Acquisitions (-2FY)", 0)) > 0)
            THEN 1
        ELSE 0
        END                                                                                            AS acquisition_pause_flag,
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

    -- =========================================================================
    -- CASHFLOW TEMPORAL FEATURES (calc_cashflow_temporal_features)
    -- =========================================================================
    (e."CFO (FQ)" - e."CFO (-4FQFQ)") / NULLIF(ABS(e."CFO (-4FQFQ)"), 0) *
    100                                                                                                AS cfo_quarterly_trend,
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
    CASE
        WHEN e."FCF (LTM)" < 0 THEN ABS(e."FCF (LTM)") / NULLIF(e."Cash And Equivalents (FQ)", 0) / 12.0
        ELSE 0 END                                                                                     AS cash_burn_rate,
    ABS(e."CFF (LTM)") / NULLIF(ABS(e."CFO (LTM)"), 0)                                                 AS financing_dependency,

    -- =========================================================================
    -- TEMPORAL FEATURES (calc_temporal_features)
    -- =========================================================================
    e."Fiscal Quarter"                                                                                 AS fiscal_quarter,
    e."Fiscal Month"                                                                                   AS fiscal_month,
    e."Fiscal Year"                                                                                    AS fiscal_year,
    (e."Next Earnings" - CURRENT_DATE)::INTEGER                                                        AS days_to_earnings,
    (CURRENT_DATE - e."Income Statement Report Date")::INTEGER                                         AS earnings_report_recency,
    e."Reporting Lag"                                                                                  AS reporting_lag,
    e."Fiscal Month" / 12.0                                                                            AS fiscal_year_progress,

    -- =========================================================================
    -- FISCAL CALENDAR FEATURES (calc_fiscal_calendar_features)
    -- =========================================================================
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

    -- =========================================================================
    -- COMPOSITE SCORES (calc_composite_scores)
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

    -- =========================================================================
    -- EBIT/EBITDA COMPREHENSIVE (calc_ebit_ebitda_comprehensive)
    -- =========================================================================
    e."EBIT (FQ)"                                                                                      AS ebit_fq,
    e."EBIT (LTM)"                                                                                     AS ebit_ltm,
    e."EBIT (FY)"                                                                                      AS ebit_fy,
    e."EBIT (-1FY)"                                                                                    AS ebit_1fy,
    e."EBITDA (FQ)"                                                                                    AS ebitda_fq,
    e."EBITDA (LTM)"                                                                                   AS ebitda_ltm,
    e."EBITDA (FY)"                                                                                    AS ebitda_fy,
    e."EBITDA (-1FY)"                                                                                  AS ebitda_1fy,
    -- NEW: Extended historical FY
    e."EBIT (-2FY)"                                                                                    AS ebit_2fy,
    e."EBIT (-3FY)"                                                                                    AS ebit_3fy,
    e."EBIT (-4FY)"                                                                                    AS ebit_4fy,
    e."EBITDA (-2FY)"                                                                                  AS ebitda_2fy,
    e."EBITDA (-3FY)"                                                                                  AS ebitda_3fy,
    e."EBITDA (-4FY)"                                                                                  AS ebitda_4fy,
    -- NEW: Quarterly historical
    e."EBIT (-1FQFQ)"                                                                                  AS ebit_1fqfq,
    e."EBIT (-2FQFQ)"                                                                                  AS ebit_2fqfq,
    e."EBIT (-3FQFQ)"                                                                                  AS ebit_3fqfq,
    e."EBIT (-4FQFQ)"                                                                                  AS ebit_4fqfq,
    e."EBITDA (-1FQFQ)"                                                                                AS ebitda_1fqfq,
    e."EBITDA (-2FQFQ)"                                                                                AS ebitda_2fqfq,
    e."EBITDA (-3FQFQ)"                                                                                AS ebitda_3fqfq,
    e."EBITDA (-4FQFQ)"                                                                                AS ebitda_4fqfq,
    -- NEW: 5-year averages
    e."EBIT (5YAVGFQ)"                                                                                 AS ebit_5yavgfq,
    e."EBIT (5YAVGLTM)"                                                                                AS ebit_5yavgltm,
    e."EBITDA (5YAVGFQ)"                                                                               AS ebitda_5yavgfq,
    e."EBITDA (5YAVGLTM)"                                                                              AS ebitda_5yavgltm,
    -- NEW: Adjusted variants
    e."EBIT/Adj. (FQ)"                                                                                 AS ebit_adj_fq,
    e."EBIT/Adj. (LTM)"                                                                                AS ebit_adj_ltm,
    e."EBIT/Adj. (FY)"                                                                                 AS ebit_adj_fy,
    e."EBITDA/Adj. (FQ)"                                                                               AS ebitda_adj_fq,
    e."EBITDA/Adj. (LTM)"                                                                              AS ebitda_adj_ltm,
    e."EBITDA/Adj. (FY)"                                                                               AS ebitda_adj_fy,
    -- Derived metrics
    (e."EBIT (FY)" - e."EBIT (-1FY)") / NULLIF(ABS(e."EBIT (-1FY)"), 0) *
    100                                                                                                AS ebit_growth_yoy,
    (e."EBITDA (FY)" - e."EBITDA (-1FY)") / NULLIF(ABS(e."EBITDA (-1FY)"), 0) *
    100                                                                                                AS ebitda_growth_yoy_comp,
    e."EBIT (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                         AS ebit_margin_ltm,
    e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                       AS ebitda_margin_ltm,
    (CASE WHEN e."EBIT (FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBIT (-1FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBIT (-2FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBIT (-3FY)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."EBIT (-4FY)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS ebit_positive_years,
    (CASE WHEN e."EBITDA (FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBITDA (-1FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBITDA (-2FY)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."EBITDA (-3FY)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."EBITDA (-4FY)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS ebitda_positive_years,
    -- NEW: Quarterly momentum
    (e."EBIT (FQ)" - e."EBIT (-1FQFQ)") / NULLIF(ABS(e."EBIT (-1FQFQ)"), 0) *
    100                                                                                                AS ebit_qoq_growth,
    (e."EBITDA (FQ)" - e."EBITDA (-1FQFQ)") / NULLIF(ABS(e."EBITDA (-1FQFQ)"), 0) * 100
                                                                                                       AS ebitda_qoq_growth,
    -- NEW: Multi-year CAGR (3-year)
    CASE
        WHEN e."EBIT (-3FY)" > 0 AND e."EBIT (FY)" > 0
            THEN (POWER(e."EBIT (FY)" / NULLIF(e."EBIT (-3FY)", 0), 1.0 / 3.0) - 1) * 100
        END                                                                                            AS ebit_cagr_3y,
    CASE
        WHEN e."EBITDA (-3FY)" > 0 AND e."EBITDA (FY)" > 0
            THEN (POWER(e."EBITDA (FY)" / NULLIF(e."EBITDA (-3FY)", 0), 1.0 / 3.0) - 1) * 100
        END                                                                                            AS ebitda_cagr_3y,
    -- NEW: vs 5Y average
    e."EBIT (LTM)" / NULLIF(e."EBIT (5YAVGLTM)", 0)                                                    AS ebit_vs_5y_avg,
    e."EBITDA (LTM)" / NULLIF(e."EBITDA (5YAVGLTM)", 0)                                                AS ebitda_vs_5y_avg,

    -- =========================================================================
    -- NET INCOME COMPREHENSIVE (calc_net_income_comprehensive)
    -- =========================================================================
    e."Net Income - (IS) (FQ)"                                                                         AS net_income_is_fq,
    e."Net Income - (IS) (LTM)"                                                                        AS net_income_is_ltm,
    e."Net Income - (IS) (FY)"                                                                         AS net_income_is_fy,
    e."Net Income/Adj. (LTM)"                                                                          AS net_income_adj_ltm,
    e."Normalized Net Income (LTM)"                                                                    AS normalized_ni_ltm,
    -- NEW: Extended quarterly historical
    e."Net Income - (IS) (-1FQFQ)"                                                                     AS net_income_is_1fqfq,
    e."Net Income - (IS) (-2FQFQ)"                                                                     AS net_income_is_2fqfq,
    e."Net Income - (IS) (-3FQFQ)"                                                                     AS net_income_is_3fqfq,
    e."Net Income - (IS) (-4FQFQ)"                                                                     AS net_income_is_4fqfq,
    -- NEW: Extended yearly historical
    e."Net Income - (IS) (-1FY)"                                                                       AS net_income_is_1fy,
    e."Net Income - (IS) (-2FY)"                                                                       AS net_income_is_2fy,
    e."Net Income - (IS) (-3FY)"                                                                       AS net_income_is_3fy,
    e."Net Income - (IS) (-4FY)"                                                                       AS net_income_is_4fy,
    -- NEW: 5-year averages
    e."Net Income - (IS) (5YAVGFQ)"                                                                    AS net_income_is_5yavgfq,
    e."Net Income - (IS) (5YAVGLTM)"                                                                   AS net_income_is_5yavgltm,
    e."Normalized Net Income (5YAVGFQ)"                                                                AS normalized_ni_5yavgfq,
    e."Normalized Net Income (5YAVGLTM)"                                                               AS normalized_ni_5yavgltm,
    -- Derived metrics
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
    )                                                                                                  AS earnings_quality_composite_comp,
    -- NEW: Quarterly trends
    pct_change(e."Net Income - (IS) (FQ)",
               e."Net Income - (IS) (-1FQFQ)")                                                         AS net_income_qoq_growth,
    pct_change(e."Net Income - (IS) (FQ)",
               e."Net Income - (IS) (-4FQFQ)")                                                         AS net_income_yoy_quarterly,
    -- vs 5Y averages
    safe_divide(e."Net Income - (IS) (LTM)",
                e."Net Income - (IS) (5YAVGLTM)")                                                      AS net_income_vs_5y_avg,
    safe_divide(e."Normalized Net Income (LTM)",
                e."Normalized Net Income (5YAVGLTM)")                                                  AS normalized_ni_vs_5y_avg,

    -- =========================================================================
    -- CASHFLOW COMPREHENSIVE (calc_cashflow_comprehensive)
    -- =========================================================================
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
    -- BETA RISK FEATURES (calc_beta_risk_features)
    -- =========================================================================
    e."Beta (1Y)"                                                                                      AS beta_1y,
    e."Beta (5Y)"                                                                                      AS beta_5y,
    e."Beta (1Y)" - e."Beta (5Y)"                                                                      AS beta_spread,
    (e."Beta (1Y)" - e."Beta (5Y)") / NULLIF(ABS(e."Beta (5Y)"), 0) *
    100                                                                                                AS beta_trend,
    CASE WHEN e."Beta (1Y)" > 1.5 THEN 1 ELSE 0 END                                                    AS high_beta_flag,
    CASE WHEN e."Beta (1Y)" < 0.5 THEN 1 ELSE 0 END                                                    AS low_beta_flag,
    GREATEST(0,
             LEAST(100, 100 - ABS(e."Beta (1Y)" - e."Beta (5Y)") * 50))                                AS beta_stability_score,

    -- =========================================================================
    -- COST STRUCTURE FEATURES (calc_cost_structure_features)
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

    -- =========================================================================
    -- INTEREST INCOME FEATURES (calc_interest_income_features)
    -- =========================================================================
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
    -- LONG TERM MOMENTUM FEATURES (calc_long_term_momentum_features)
    -- =========================================================================
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
    -- TANGIBLE BOOK FEATURES (calc_tangible_book_features)
    -- =========================================================================
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
    -- WORKING CAPITAL DEEP FEATURES (calc_working_capital_deep_features)
    -- =========================================================================
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

    -- =========================================================================
    -- UNUSUAL ITEMS FEATURES (calc_unusual_items_features)
    -- =========================================================================
    e."Other Unusual Items/Total (LTM)"                                                                AS other_unusual_items_ltm,
    COALESCE(e."Other Unusual Items/Total (LTM)", 0) + COALESCE(e."Impairment of Goodwill (LTM)", 0) +
    COALESCE(e."Asset Writedown (LTM)", 0) +
    COALESCE(e."Restructuring Charges (LTM)", 0)                                                       AS total_unusual_items,
    safe_divide(
            ABS(COALESCE(e."Other Unusual Items/Total (LTM)", 0) + COALESCE(e."Impairment of Goodwill (LTM)", 0) +
                COALESCE(e."Asset Writedown (LTM)", 0) + COALESCE(e."Restructuring Charges (LTM)", 0)),
            e."Total Revenues (LTM)"
    ) *
    100                                                                                                AS unusual_items_to_revenue,
    safe_divide(
            ABS(COALESCE(e."Other Unusual Items/Total (LTM)", 0) + COALESCE(e."Impairment of Goodwill (LTM)", 0) +
                COALESCE(e."Asset Writedown (LTM)", 0) + COALESCE(e."Restructuring Charges (LTM)", 0)),
            ABS(e."EBITDA (LTM)")
    ) *
    100                                                                                                AS unusual_items_to_ebitda,
    CASE
        WHEN
            ABS(COALESCE(e."Other Unusual Items/Total (LTM)", 0)) + ABS(COALESCE(e."Impairment of Goodwill (LTM)", 0)) +
            ABS(COALESCE(e."Asset Writedown (LTM)", 0)) + ABS(COALESCE(e."Restructuring Charges (LTM)", 0)) > 0 THEN 1
        ELSE 0
        END                                                                                            AS has_unusual_items_flag,

    -- =========================================================================
    -- REVENUE ESTIMATE CONSENSUS (calc_revenue_estimate_consensus)
    -- =========================================================================
    e."Revenues - Est Avg (FY1E)"                                                                      AS revenue_est_avg_fy1e,
    e."Revenues - Est Med (FY1E)"                                                                      AS revenue_est_med_fy1e,
    e."Revenues - Est Avg (NTM)"                                                                       AS revenue_est_avg_ntm,
    e."Revenues - Est Med (NTM)"                                                                       AS revenue_est_med_ntm,
    safe_divide(e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)", e."Revenues - Est Med (FY1E)") *
    100                                                                                                AS revenue_avg_med_diff_pct,
    clamp_score(100 - ABS(safe_divide(e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)",
                                      e."Revenues - Est Med (FY1E)") * 100) *
                      2)                                                                               AS revenue_consensus_strength,
    safe_divide(e."Revenues - Est Avg (FY1E)", e."Total Revenues (LTM)")                               AS revenue_vs_current,

    -- =========================================================================
    -- REVENUE QUARTERLY FEATURES (calc_revenue_quarterly_features)
    -- =========================================================================
    e."Total Revenues (FQ)"                                                                            AS revenue_fq,
    e."Total Revenues (-1FQFQ)"                                                                        AS revenue_1fq,
    e."Total Revenues (-2FQFQ)"                                                                        AS revenue_2fq,
    e."Total Revenues (-3FQFQ)"                                                                        as revenue_3q,
    e."Total Revenues (-4FQFQ)"                                                                        as revenue_4q,
    e."Total Revenues (FY)"                                                                            AS revenue_fy,
    e."Total Revenues (-1FY)"                                                                          AS revenue_1fy,
    e."Total Revenues (-2FY)"                                                                          AS revenue_2fy,
    e."Total Revenues (-3FY)"                                                                          AS revenue_3fy,
    e."Total Revenues (-4FY)"                                                                          AS revenue_4fy,
    e."Total Revenues (LTM)"                                                                           AS revenue_ltm,
    e."Total Revenues (5YAVGLTM)"                                                                      AS revenue_5y_avg,
    pct_change(e."Total Revenues (FY)", e."Total Revenues (-1FY)")                                     AS revenue_yoy_growth,
    pct_change(e."Total Revenues (-1FY)", e."Total Revenues (-2FY)")                                   AS revenue_1fy_vs_2fy,
    pct_change(e."Total Revenues (-2FY)", e."Total Revenues (-3FY)")                                   AS revenue_2fy_vs_3fy,
    pct_change(e."Total Revenues (-3FY)", e."Total Revenues (-4FY)")                                   AS revenue_3fy_vs_4fy,
    CASE
        WHEN e."Total Revenues (FY)" > e."Total Revenues (-1FY)" THEN 1
        ELSE 0 END                                                                                     AS revenue_growth_flag,

    -- =========================================================================
    -- REVENUE TEMPORAL (calc_total_revenues_temporal)
    -- =========================================================================
    e."Total Revenues (5YAVGFQ)"                                                                       AS revenue_5yavgfq,
    e."Total Revenues (5YAVGLTM)"                                                                      AS revenue_5yavgltm,
    pct_change(e."Total Revenues (FY)", e."Total Revenues (-1FY)")                                     AS revenue_growth_yoy_temp,
    safe_divide(e."Total Revenues (FQ)", e."Total Revenues (5YAVGFQ)")                                 AS revenue_vs_5y_avg_fq_temp,
    safe_divide(e."Total Revenues (LTM)", e."Total Revenues (5YAVGLTM)")                               AS revenue_vs_5y_avg_ltm_temp,
    safe_divide(e."Total Revenues (FQ)" - e."Total Revenues (5YAVGFQ)", e."Total Revenues (5YAVGFQ)") *
    100                                                                                                AS revenue_fq_vs_avg_temp,
    calc_change_ratio(e."Total Revenues (LTM)", e."Total Revenues (-1FY)") *
    100                                                                                                AS revenue_momentum_temp,

    -- =========================================================================
    -- WORKING CAPITAL TEMPORAL (calc_working_capital_temporal)
    -- =========================================================================
    e."Working Capital (FQ)"                                                                           AS wc_fq_temp,
    e."Working Capital (FY)"                                                                           AS wc_fy_temp,
    e."Working Capital (LTM)"                                                                          AS wc_ltm_temp,
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
    (CASE WHEN e."Working Capital (FQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Working Capital (-1FQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Working Capital (-2FQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Working Capital (-3FQ)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."Working Capital (-4FQ)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS wc_positive_quarters,
    CASE
        WHEN e."Working Capital (FQ)" > e."Working Capital (-1FQ)"
            AND e."Working Capital (-1FQ)" > e."Working Capital (-2FQ)"
            THEN 1
        ELSE 0 END                                                                                     AS wc_improving_flag_temp,
    (ABS(e."Working Capital (FQ)" - e."Working Capital (-1FQ)") +
     ABS(e."Working Capital (-1FQ)" - e."Working Capital (-2FQ)") +
     ABS(e."Working Capital (-2FQ)" - e."Working Capital (-3FQ)") +
     ABS(e."Working Capital (-3FQ)" - e."Working Capital (-4FQ)")) /
    NULLIF(ABS((e."Working Capital (FQ)" + e."Working Capital (-1FQ)" +
                e."Working Capital (-2FQ)" + e."Working Capital (-3FQ)" +
                e."Working Capital (-4FQ)") / 5.0), 0)                                                 AS wc_volatility,

    -- =========================================================================
    -- TOTAL DEBT TEMPORAL (calc_total_debt_temporal)
    -- =========================================================================
    e."Total Debt (FQ)"                                                                                AS debt_fq,
    e."Total Debt (FY)"                                                                                AS debt_fy,
    e."Total Debt (LTM)"                                                                               AS debt_ltm,
    e."Total Debt (-1FQ)"                                                                              AS debt_1fq,
    e."Total Debt (-2FQ)"                                                                              AS debt_2fq,
    e."Total Debt (-3FQ)"                                                                              AS debt_3fq,
    e."Total Debt (-4FQ)"                                                                              AS debt_4fq,
    e."Total Debt (-1FY)"                                                                              AS debt_1fy_temp,
    e."Total Debt (-2FY)"                                                                              AS debt_2fy,
    e."Total Debt (-3FY)"                                                                              AS debt_3fy,
    e."Total Debt (-4FY)"                                                                              AS debt_4fy,
    pct_change(e."Total Debt (FQ)", e."Total Debt (-1FQ)")                                             AS debt_qoq_change,
    pct_change(e."Total Debt (FY)", e."Total Debt (-1FY)")                                             AS debt_yoy_change,
    pct_change(e."Total Debt (FQ)", e."Total Debt (-4FQ)")                                             AS debt_4q_trend,
    CASE
        WHEN e."Total Debt (-3FY)" > 0
            THEN (POWER(safe_divide(e."Total Debt (FY)", e."Total Debt (-3FY)"), 1.0 / 3.0) - 1) * 100
        END                                                                                            AS debt_3y_cagr,
    CASE
        WHEN e."Total Debt (FQ)" < e."Total Debt (-1FQ)"
            AND e."Total Debt (-1FQ)" < e."Total Debt (-2FQ)"
            THEN 1
        ELSE 0 END                                                                                     AS debt_deleveraging,
    safe_divide(e."Total Debt (FY)", e."Total Equity (FY)") -
    safe_divide(e."Total Debt (-1FY)",
                NULLIF(e."Total Equity (FY)", 0))                                                      AS debt_to_equity_trend,

    -- =========================================================================
    -- TOTAL ASSETS TEMPORAL (calc_total_assets_temporal)
    -- =========================================================================
    e."Total Assets (FQ)"                                                                              AS assets_fq,
    e."Total Assets (FY)"                                                                              AS assets_fy,
    e."Total Assets (LTM)"                                                                             AS assets_ltm_temp,
    e."Total Assets (-1FQ)"                                                                            AS assets_1fq,
    e."Total Assets (-2FQ)"                                                                            AS assets_2fq,
    e."Total Assets (-3FQ)"                                                                            AS assets_3fq,
    e."Total Assets (-4FQ)"                                                                            AS assets_4fq,
    e."Total Assets (-1FY)"                                                                            AS assets_1fy,
    e."Total Assets (-2FY)"                                                                            AS assets_2fy,
    e."Total Assets (-3FY)"                                                                            AS assets_3fy,
    e."Total Assets (-4FY)"                                                                            AS assets_4fy,
    pct_change(e."Total Assets (FQ)", e."Total Assets (-1FQ)")                                         AS assets_qoq_growth,
    pct_change(e."Total Assets (FY)", e."Total Assets (-1FY)")                                         AS assets_yoy_growth,
    CASE
        WHEN e."Total Assets (-3FY)" > 0
            THEN (POWER(safe_divide(e."Total Assets (FY)", e."Total Assets (-3FY)"), 1.0 / 3.0) - 1) * 100
        END                                                                                            AS assets_3y_cagr,
    pct_change(e."Total Assets (FY)", e."Total Assets (-1FY)") -
    pct_change(e."Total Assets (-1FY)", e."Total Assets (-2FY)")                                       AS asset_growth_accel,
    CASE
        WHEN e."Total Assets (FY)" >= e."Total Assets (-1FY)"
            AND e."Total Assets (-1FY)" >= e."Total Assets (-2FY)"
            AND e."Total Assets (-2FY)" >= e."Total Assets (-3FY)"
            THEN 1
        ELSE 0 END                                                                                     AS asset_base_stable,

    -- =========================================================================
    -- GROSS PROFIT TEMPORAL (calc_gross_profit_temporal)
    -- =========================================================================
    e."Gross Profit (FQ)"                                                                              AS gp_fq,
    e."Gross Profit (FY)"                                                                              AS gp_fy,
    e."Gross Profit (LTM)"                                                                             AS gp_ltm_temp,
    e."Gross Profit (-1FQFQ)"                                                                          AS gp_1fqfq,
    e."Gross Profit (-2FQFQ)"                                                                          AS gp_2fqfq,
    e."Gross Profit (-3FQFQ)"                                                                          AS gp_3fqfq,
    e."Gross Profit (-4FQFQ)"                                                                          AS gp_4fqfq,
    e."Gross Profit (-1FY)"                                                                            AS gp_1fy,
    e."Gross Profit (-2FY)"                                                                            AS gp_2fy,
    e."Gross Profit (-3FY)"                                                                            AS gp_3fy,
    e."Gross Profit (-4FY)"                                                                            AS gp_4fy,
    pct_change(e."Gross Profit (FQ)", e."Gross Profit (-1FQFQ)")                                       AS gp_qoq_growth,
    pct_change(e."Gross Profit (FY)", e."Gross Profit (-1FY)")                                         AS gp_yoy_growth,
    safe_divide(e."Gross Profit (FQ)", e."Total Revenues (FQ)") * 100                                  AS gp_margin_fq,
    (safe_divide(e."Gross Profit (FQ)", e."Total Revenues (FQ)") -
     safe_divide(e."Gross Profit (-4FQFQ)", e."Total Revenues (5YAVGFQ)")) *
    100                                                                                                AS gp_margin_trend,
    (CASE WHEN e."Gross Profit (FQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Gross Profit (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Gross Profit (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE WHEN e."Gross Profit (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
     CASE
         WHEN e."Gross Profit (-4FQFQ)" > 0 THEN 1
         ELSE 0 END)::INTEGER                                                                          AS gp_positive_quarters,
    CASE
        WHEN e."Gross Profit Margin % (LTM)" > e."Gross Profit Margin % (FY)"
            THEN 1
        ELSE 0 END                                                                                     AS gp_margin_expansion_temp,

    -- =========================================================================
    -- DIVIDEND YIELD COMPREHENSIVE (calc_dividend_yield_comprehensive)
    -- =========================================================================
    e."Div Yield (Ind)"                                                                                AS div_yield_ind,
    e."Div Yield (-1FYInd)"                                                                            AS div_yield_1fy_ind,
    e."Div Yield (5YAVGLTM)"                                                                           AS div_yield_5y_avg,
    e."Div Yield (LTM)" / NULLIF(e."Div Yield (5YAVGLTM)", 0)                                          AS div_yield_vs_5y_avg,
    (e."Div Yield (NTM)" - e."Div Yield (LTM)") / NULLIF(e."Div Yield (LTM)", 0) *
    100                                                                                                AS div_yield_growth_expected,
    CASE WHEN e."Div Yield (LTM)" > 4 THEN 1 ELSE 0 END                                                AS high_yield_flag,
    CASE
        WHEN e."Div Yield (LTM)" > 0
            AND e."FCF (LTM)" > ABS(COALESCE(e."Common Dividends Paid (LTM)", 0))
            AND e."Dividend Streak" >= 5 THEN 1
        ELSE 0
        END                                                                                            AS sustainable_dividend_flag,

    -- =========================================================================
    -- EPS CONTINUING FEATURES (calc_eps_continuing_features) - NEW
    -- =========================================================================
    e."Basic EPS - Cont (LTM)"                                                                         AS eps_cont_ltm,
    e."Basic EPS - Cont (FQ)"                                                                          AS eps_cont_fq,
    e."Basic EPS - Cont (FY)"                                                                          AS eps_cont_fy,
    e."Basic EPS - Cont (-1FY)"                                                                        AS eps_cont_1fy,
    e."Basic EPS - Cont (-2FY)"                                                                        AS eps_cont_2fy,
    e."Basic EPS - Cont (-3FY)"                                                                        AS eps_cont_3fy,
    e."Basic EPS - Cont (-4FY)"                                                                        AS eps_cont_4fy,
    pct_change(e."Basic EPS - Cont (FQ)", e."Basic EPS - Cont (-1FQFQ)")                               AS eps_cont_qoq_growth,
    pct_change(e."Basic EPS - Cont (FY)", e."Basic EPS - Cont (-1FY)")                                 AS eps_cont_yoy_growth,
    safe_divide(e."Basic EPS - Cont (LTM)", e."Net EPS - Basic (LTM)")                                 AS eps_cont_vs_total_eps,
    ((e."Net EPS - Basic (LTM)" - e."Basic EPS - Cont (LTM)") /
     NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0)) *
    100                                                                                                AS discontinued_ops_impact,

    -- =========================================================================
    -- R&D TEMPORAL FEATURES (calc_rnd_temporal_features) - NEW
    -- =========================================================================
    e."R&D Expenses (FQ)"                                                                              AS rnd_fq,
    e."R&D Expenses (FY)"                                                                              AS rnd_fy,
    e."R&D Expenses (-1FY)"                                                                            AS rnd_1fy,
    e."R&D Expenses (-2FY)"                                                                            AS rnd_2fy,
    e."R&D Expenses (-3FY)"                                                                            AS rnd_3fy,
    e."R&D Expenses (-4FY)"                                                                            AS rnd_4fy,
    safe_divide(e."R&D Expenses (LTM)", e."Total Revenues (LTM)") * 100                                AS rnd_intensity_ltm,
    pct_change(e."R&D Expenses (FY)", e."R&D Expenses (-1FY)")                                         AS rnd_yoy_growth,
    safe_divide(e."R&D Expenses (FY)", e."Full Time Employees (FY)")                                   AS rnd_per_employee,
    CASE
        WHEN safe_divide(e."R&D Expenses (LTM)", e."Total Revenues (LTM)") > 0.10
            THEN 1
        ELSE 0 END                                                                                     AS high_rnd_intensity_flag,

    -- =========================================================================
    -- INVENTORY TEMPORAL FEATURES (calc_inventory_temporal_features) - NEW
    -- =========================================================================
    e."Inventory (-1FQ)"                                                                               AS inventory_1fq,
    e."Inventory (-2FQ)"                                                                               AS inventory_2fq,
    e."Inventory (-3FQ)"                                                                               AS inventory_3fq,
    e."Inventory (-4FQ)"                                                                               AS inventory_4fq,
    e."Inventory (-1FY)"                                                                               AS inventory_1fy,
    e."Inventory (-2FY)"                                                                               AS inventory_2fy,
    e."Inventory (-3FY)"                                                                               AS inventory_3fy,
    e."Inventory (-4FY)"                                                                               AS inventory_4fy,
    pct_change(e."Inventory (FQ)", e."Inventory (-1FQ)")                                               AS inventory_qoq_change,
    pct_change(e."Inventory (FY)", e."Inventory (-1FY)")                                               AS inventory_yoy_change,
    e."Inventory (LTM)" / NULLIF(e."Cost Of Revenues (LTM)" / 365.0, 0)                                AS inventory_days,
    e."Cost Of Revenues (LTM)" / NULLIF(e."Inventory (LTM)", 0)                                        AS inventory_turnover_mv,

    -- =========================================================================
    -- GOODWILL TEMPORAL FEATURES (calc_goodwill_temporal_features) - NEW
    -- =========================================================================
    e."Goodwill (-1FQ)"                                                                                AS goodwill_1fq,
    e."Goodwill (-2FQ)"                                                                                AS goodwill_2fq,
    e."Goodwill (-3FQ)"                                                                                AS goodwill_3fq,
    e."Goodwill (-4FQ)"                                                                                AS goodwill_4fq,
    e."Goodwill (-1FY)"                                                                                AS goodwill_1fy,
    e."Goodwill (-2FY)"                                                                                AS goodwill_2fy,
    e."Goodwill (-3FY)"                                                                                AS goodwill_3fy,
    e."Goodwill (-4FY)"                                                                                AS goodwill_4fy,
    pct_change(e."Goodwill (FQ)", e."Goodwill (-1FQ)")                                                 AS goodwill_qoq_change,
    pct_change(e."Goodwill (FY)", e."Goodwill (-1FY)")                                                 AS goodwill_yoy_change,
    pct_change(e."Goodwill (FY)", e."Goodwill (-3FY)")                                                 AS goodwill_3y_growth,
    safe_divide(e."Goodwill (LTM)", e."Total Equity (LTM)") * 100                                      AS goodwill_concentration,
    CASE
        WHEN pct_change(e."Goodwill (FQ)", e."Goodwill (-1FQ)") > 20
            THEN 1
        ELSE 0 END                                                                                     AS recent_acquisition_flag,

    -- =========================================================================
    -- MARKETING EXPENSES FEATURES (calc_cost_structure_features) - ENHANCED
    -- =========================================================================
    safe_divide(e."Marketing Expenses (FY)", e."Total Revenues (FY)") *
    100                                                                                                AS marketing_to_revenue,
    pct_change(e."Marketing Expenses (FY)",
               e."Marketing Expenses (-1FY)")                                                          AS marketing_trend_yoy,
    safe_divide(e."Marketing Expenses (FY)",
                e."Marketing Expenses (5YAVGLTM)")                                                     AS marketing_vs_5y_avg,
    safe_divide(e."Selling General & Admin Expenses/Total (FQ)",
                e."Selling General & Admin Expenses/Total (5YAVGFQ)")                                  AS sga_vs_5y_avg,

    -- =========================================================================
    -- TBV ENHANCED FEATURES (calc_tangible_book_features) - ENHANCED
    -- =========================================================================
    e."TBV (FY)"                                                                                       AS tangible_book_value_fy,
    e."TBV (LTM)"                                                                                      AS tangible_book_value_ltm,
    pct_change(e."TBV (LTM)", e."TBV (FY)")                                                            AS tbv_yoy_growth,

    -- =========================================================================
    -- METADATA
    -- =========================================================================
    CURRENT_TIMESTAMP                                                                                  AS calculated_at

FROM postgres.public.equities e;

-- Create indexes on the materialized view for common query patterns
CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_isin ON mv_all_stock_features (isin);
CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_ticker ON mv_all_stock_features (ticker);
CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_piotroski ON mv_all_stock_features (piotroski_f_score);
CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_quality ON mv_all_stock_features (accounting_quality_score);
CREATE INDEX IF NOT EXISTS idx_mv_all_stock_features_momentum ON mv_all_stock_features (price_momentum_1y);

-- =============================================================================
-- SECTION 2: FEATURE REGISTRY METADATA TABLE (ENHANCED)
-- =============================================================================

-- Create a metadata table documenting available SQL feature functions
CREATE TABLE IF NOT EXISTS feature_registry_metadata
(
    function_name     TEXT PRIMARY KEY,
    category          TEXT NOT NULL,
    feature_count     INTEGER,
    description       TEXT,
    python_equivalent TEXT,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add index for category-based queries
CREATE INDEX IF NOT EXISTS idx_feature_registry_category
    ON feature_registry_metadata (category);

-- Add index for python_equivalent lookups
CREATE INDEX IF NOT EXISTS idx_feature_registry_python_equiv
    ON feature_registry_metadata (python_equivalent);

-- Wrap upsert in transaction for atomicity
BEGIN;

-- Upsert all function metadata
INSERT INTO feature_registry_metadata (function_name, category, feature_count, description, python_equivalent,
                                       updated_at)
VALUES
    -- Helper Functions
    ('safe_divide', 'Helper Functions', 1, 'Safe division avoiding division by zero', NULL, CURRENT_TIMESTAMP),
    ('pct_change', 'Helper Functions', 1, 'Percentage change calculation', NULL, CURRENT_TIMESTAMP),
    ('calc_change_ratio', 'Helper Functions', 1, 'Change ratio without percentage multiplier', NULL, CURRENT_TIMESTAMP),
    ('clamp_score', 'Helper Functions', 1, 'Score clamping between min and max values', NULL, CURRENT_TIMESTAMP),
    ('ema_crossover_signal', 'Helper Functions', 1, 'EMA crossover signal detection', NULL, CURRENT_TIMESTAMP),

    -- Valuation Functions
    ('calc_valuation_features', 'Valuation Ratios', 6, 'P/E, P/B, EV/EBITDA, EV/Sales, dividend yield, PEG ratio',
     'engineer_valuation_ratios', CURRENT_TIMESTAMP),
    ('calc_valuation_timeseries_features', 'Valuation Timeseries', 11,
     'Valuation momentum, mean reversion, forward discount', 'engineer_valuation_timeseries_features',
     CURRENT_TIMESTAMP),
    ('calc_extended_valuation_timeseries', 'Valuation Timeseries', 11,
     'QoQ multiple trends, mean reversion, P/B momentum', 'engineer_valuation_timeseries_features', CURRENT_TIMESTAMP),

    -- Momentum & Technical Functions
    ('calc_momentum_features', 'Momentum & Technical', 14,
     'Price momentum, EMA crossovers, 52W range, beta, volatility', 'engineer_momentum_features', CURRENT_TIMESTAMP),
    ('calc_technical_analysis_features', 'Technical Analysis', 11,
     'EMA trends, breakout signals, volume momentum, volatility compression', 'engineer_technical_analysis_features',
     CURRENT_TIMESTAMP),
    ('calc_long_term_momentum_features', 'Momentum & Technical', 7,
     '3Y/5Y momentum, weighted trend score, secular trend flag', 'engineer_long_term_momentum_features',
     CURRENT_TIMESTAMP),

    -- Profitability Functions
    ('calc_profitability_features', 'Profitability', 9, 'ROE, ROA, margins, ROIC, DuPont components',
     'engineer_profitability_ratios', CURRENT_TIMESTAMP),
    ('calc_margin_trends', 'Profitability', 6, 'Gross, operating, net, EBITDA margin trends, expansion flag',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_ebit_ebitda_comprehensive', 'Profitability', 42, 'EBIT/EBITDA for all periods, growth and margins',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_total_revenues_temporal', 'Growth Metrics', 12,
     'Comprehensive revenue trends across FQ, LTM, FY and 5Y averages',
     'engineer_growth_metrics', CURRENT_TIMESTAMP),

    -- Quality & Risk Functions
    ('calc_quality_features', 'Quality & Risk', 10, 'Impairments, goodwill, Z-score, liquidity ratios',
     'engineer_accounting_quality_features', CURRENT_TIMESTAMP),
    ('calc_financial_distress_features', 'Financial Distress', 9, 'Distress risk score, liquidity stress, cash runway',
     'engineer_financial_distress_features', CURRENT_TIMESTAMP),
    ('calc_accounting_quality_features', 'Accounting Quality', 7,
     'Goodwill changes, restructuring, exceptional items, quality score', 'engineer_accounting_quality_features',
     CURRENT_TIMESTAMP),
    ('calc_quality_features_comprehensive', 'Accounting Quality', 11,
     'Detailed impairments, writedowns, restructuring across periods', 'engineer_accounting_quality_features',
     CURRENT_TIMESTAMP),
    ('calc_beta_risk_features', 'Quality & Risk', 7, 'Multi-period betas, trend analysis, stability score',
     'engineer_beta_risk_features', CURRENT_TIMESTAMP),
    ('calc_working_capital_temporal', 'Leverage & Liquidity', 21, 'Full historical coverage of working capital trends',
     'engineer_working_capital_deep_features', CURRENT_TIMESTAMP),
    ('calc_total_debt_temporal', 'Leverage & Liquidity', 18,
     'Leverage trend analysis with quarterly and yearly historical data',
     'engineer_leverage_ratios', CURRENT_TIMESTAMP),
    ('calc_total_assets_temporal', 'Balance Sheet', 17, 'Balance sheet dynamics with full historical asset coverage',
     'engineer_balance_sheet_trends', CURRENT_TIMESTAMP),

    -- Leverage & Liquidity Functions
    ('calc_leverage_features', 'Leverage & Liquidity', 7, 'Debt ratios, coverage, working capital ratio',
     'engineer_leverage_ratios', CURRENT_TIMESTAMP),
    ('calc_efficiency_ratios', 'Efficiency', 4, 'Asset and inventory turnover, receivables days',
     'engineer_efficiency_ratios', CURRENT_TIMESTAMP),
    ('calc_balance_sheet_dynamics', 'Balance Sheet', 13, 'Cash trends, inventory vs 5Y avg, asset quality, BS strength',
     'engineer_balance_sheet_trends', CURRENT_TIMESTAMP),
    ('calc_working_capital_deep_features', 'Leverage & Liquidity', 11,
     'Working capital ratios, trends, efficiency score', 'engineer_working_capital_deep_features', CURRENT_TIMESTAMP),

    -- Analyst Sentiment Functions
    ('calc_sentiment_features', 'Analyst Sentiment', 10, 'Ratings, price targets, revisions, coverage quality',
     'engineer_analyst_quality_features', CURRENT_TIMESTAMP),
    ('calc_price_target_dynamics', 'Price Target Dynamics', 15,
     'PT momentum (1W-1Y), consensus convergence, coverage changes', 'engineer_price_target_dynamics',
     CURRENT_TIMESTAMP),

-- ... existing code ...
    -- Earnings Functions
    ('calc_earnings_features', 'Earnings Quality', 7, 'Surprises, adjustments, GAAP vs non-GAAP',
     'engineer_estimated_vs_actual_analytics', CURRENT_TIMESTAMP),
    ('calc_eps_trajectory_features', 'EPS Trajectory', 10, 'EPS growth rates, CAGR, positive streak, trajectory score',
     'engineer_eps_trajectory_features', CURRENT_TIMESTAMP),
    ('calc_gaap_adjusted_analytics', 'GAAP vs Adjusted', 48,
     'Comprehensive GAAP/Adjusted analytics for EPS, Net Income, EBITDA, EBIT across all periods (LTM, FY, FQ, -1FY to -4FY, -1FQFQ to -4FQFQ, 5YAVGFQ) with quality scores',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_gaap_revision_features', 'Earnings Quality', 9, 'GAAP EPS revision momentum, spread vs normalized',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    -- ... existing code ...
    ('calc_eps_comprehensive', 'Earnings Quality', 10, 'Basic, Continuing, Adjusted EPS, growth, CAGR, trajectory',
     'engineer_eps_trajectory_features', CURRENT_TIMESTAMP),
    ('calc_net_income_comprehensive', 'Earnings Quality', 30, 'GAAP, Adjusted, Normalized NI for all periods',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_gross_profit_temporal', 'Profitability', 20,
     'Gross profit margin trend analysis with full historical coverage',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_unusual_items_features', 'Earnings Quality', 9, 'Unusual items totals, ratios, earnings quality impact',
     'engineer_unusual_items_features', CURRENT_TIMESTAMP),

    -- Growth Functions
    ('calc_growth_features', 'Growth Metrics', 7, 'Revenue, EBITDA, FCF growth rates', 'engineer_growth_metrics',
     CURRENT_TIMESTAMP),
    ('calc_revenue_forecast_features', 'Revenue Forecasting', 12, 'Estimate spread, beat potential, forward multiples',
     'engineer_revenue_forecast_features', CURRENT_TIMESTAMP),
    ('calc_revenue_estimate_consensus', 'Revenue Forecasting', 8,
     'Estimate skew, consensus confidence, upside to consensus', 'engineer_revenue_estimate_consensus',
     CURRENT_TIMESTAMP),
    ('calc_revenue_quarterly_features', 'Revenue Forecasting', 32,
     'Quarterly revenue trends with full historical coverage (-1FQFQ to -4FQFQ, -1FY to -4FY), QoQ/YoY momentum, CAGR, trend analysis, and growth flags',
     'engineer_revenue_quarterly_features', CURRENT_TIMESTAMP),

    -- Dividend Functions
    ('calc_dividend_features', 'Dividend Reliability', 8, 'Streak, yield, payout, coverage, shareholder yield',
     'engineer_dividend_reliability_features', CURRENT_TIMESTAMP),
    ('calc_dividend_timing', 'Dividend Reliability', 8, 'Days to ex-date/payment, consistency, yield vs 5Y avg',
     'engineer_dividend_timing_features', CURRENT_TIMESTAMP),
    ('calc_dividend_yield_comprehensive', 'Dividend Reliability', 10,
     'Comprehensive dividend yields, sustainability flag', 'engineer_dividend_reliability_features', CURRENT_TIMESTAMP),

    -- Employment Functions
    ('calc_employment_features', 'Employee Productivity', 7, 'Per-employee metrics, FTE growth',
     'engineer_employee_productivity_features', CURRENT_TIMESTAMP),
    ('calc_employment_dynamics', 'Employment Dynamics', 10, 'FTE growth, acceleration, hiring intensity',
     'engineer_employment_dynamics_features', CURRENT_TIMESTAMP),

    -- Cash Flow Functions
    ('calc_cashflow_features', 'Cash Flow', 7, 'CFO/NI, FCF margin, self-funding ratio',
     'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),
    ('calc_enhanced_cashflow_features', 'Enhanced Cash Flow', 12,
     'FCF consistency, CapEx efficiency, M&A sustainability', 'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),
    ('calc_cashflow_temporal_features', 'Cash Flow', 12, 'Quarterly CF trends, burn rate, volatility, momentum',
     'engineer_cashflow_temporal_features', CURRENT_TIMESTAMP),
    ('calc_cashflow_comprehensive', 'Cash Flow', 14, 'CFO, CFI, CFF, FCF for all periods, quality score',
     'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),

    -- Temporal Functions
    ('calc_temporal_features', 'Temporal Patterns', 7, 'Fiscal calendar, earnings timing', 'engineer_temporal_features',
     CURRENT_TIMESTAMP),
    ('calc_fiscal_calendar_features', 'Temporal Patterns', 9, 'Days since report, quarter/FY flags, freshness score',
     'engineer_fiscal_calendar_features', CURRENT_TIMESTAMP),

    -- Composite Functions
    ('calc_composite_scores', 'Composite Scores', 4, 'Piotroski F-Score, EPS trajectory, dilution, quality-momentum',
     'engineer_composite_scores', CURRENT_TIMESTAMP),
    ('calc_all_enhanced_features', 'Composite', 3, 'Aggregation metadata of all enhanced features',
     'engineer_all_enhanced_features', CURRENT_TIMESTAMP),

    -- Cost Structure Functions (ENHANCED with Marketing and SG&A 5Y metrics)
    ('calc_cost_structure_features', 'Efficiency Ratios', 13,
     'COGS, SG&A, R&D ratios, operating leverage, marketing efficiency, SG&A vs 5Y avg',
     'engineer_cost_structure_features', CURRENT_TIMESTAMP),

    -- Interest Income Functions
    ('calc_interest_income_features', 'Interest Income', 7, 'Net interest income, coverage ratios, income quality',
     'engineer_interest_income_features', CURRENT_TIMESTAMP),

    -- Tangible Book Functions (ENHANCED with native TBV columns)
    ('calc_tangible_book_features', 'Valuation Ratios', 10,
     'Native TBV (FY/LTM), Price-to-TBV, TBV per share, tangible equity ratio, TBV growth, validation',
     'engineer_tangible_book_features', CURRENT_TIMESTAMP),

    -- NEW Functions
    ('calc_eps_continuing_features', 'Earnings Quality', 18,
     'EPS from continuing operations analysis with discontinued ops impact, trajectory, stability',
     'engineer_eps_continuing_features', CURRENT_TIMESTAMP),

    ('calc_inventory_temporal_features', 'Balance Sheet', 21,
     'Full inventory temporal coverage with efficiency and quality metrics, buildup/reduction flags',
     'engineer_inventory_temporal_features', CURRENT_TIMESTAMP),

    ('calc_goodwill_temporal_features', 'Accounting Quality', 19,
     'M&A activity tracking through goodwill changes, impairment risk, accumulation rate',
     'engineer_goodwill_temporal_features', CURRENT_TIMESTAMP),

    ('calc_rnd_temporal_features', 'Efficiency Ratios', 21,
     'R&D investment trends, intensity, efficiency metrics, ROI proxy, investment flags',
     'engineer_rnd_temporal_features', CURRENT_TIMESTAMP),

    -- Materialized View (UPDATED feature count)
    ('mv_all_stock_features', 'Materialized View', 475,
     'Unified materialized view containing all calculated features including new temporal analytics',
     'all_features_combined', CURRENT_TIMESTAMP)

ON CONFLICT (function_name) DO UPDATE SET category          = EXCLUDED.category,
                                          feature_count     = EXCLUDED.feature_count,
                                          description       = EXCLUDED.description,
                                          python_equivalent = EXCLUDED.python_equivalent,
                                          updated_at        = CURRENT_TIMESTAMP;

COMMIT;

-- Refresh table statistics for optimal query planning
ANALYZE feature_registry_metadata;

-- =============================================================================
-- SECTION 3: UTILITY FUNCTIONS FOR MATERIALIZED VIEW
-- =============================================================================

-- Function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_all_stock_features()
    RETURNS VOID AS
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;
END;
$$ LANGUAGE plpgsql;

-- Function to get feature count summary by category
CREATE OR REPLACE FUNCTION get_feature_registry_summary()
    RETURNS TABLE
            (
                category       TEXT,
                function_count INTEGER,
                total_features INTEGER
            )
AS
$$
SELECT category,
       COUNT(*)::INTEGER                        AS function_count,
       SUM(COALESCE(feature_count, 0))::INTEGER AS total_features
FROM feature_registry_metadata
GROUP BY category
ORDER BY total_features DESC;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- SECTION 4: REFRESH MATERIALIZED VIEW
-- =============================================================================

-- Initial refresh of the materialized view
REFRESH MATERIALIZED VIEW mv_all_stock_features;

-- Add a unique index for concurrent refresh capability
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_all_stock_features_isin_unique
    ON mv_all_stock_features (isin);




