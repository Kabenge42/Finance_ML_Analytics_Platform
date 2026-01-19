-- =============================================================================
-- SQL Feature Registry for Finance ML Analytics Platform
-- Phase 9.3 Feature Engineering - PostgreSQL Implementation (Enhanced)
-- =============================================================================
-- This file provides SQL subquery functions that calculate features from the
-- postgres.public.equities table, mirroring the Python feature registry in
-- finance_ml/features/advanced/__init__.py
--
-- Version: 2.0 - Enhanced with 50+ additional features
-- Usage: Include relevant CTEs or subqueries in your analysis queries
-- =============================================================================

-- =============================================================================
-- SECTION 1: VALUATION FEATURES
-- =============================================================================

-- Valuation Ratios (engineer_valuation_ratios)
-- Returns: p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio, dividend_yield
CREATE OR REPLACE FUNCTION calc_valuation_features()
    RETURNS TABLE
            (
                ticker          TEXT,
                p_e_ratio       NUMERIC,
                p_b_ratio       NUMERIC,
                ev_ebitda_ratio NUMERIC,
                ev_sales_ratio  NUMERIC,
                dividend_yield  NUMERIC,
                peg_ratio       NUMERIC
            )
AS
$$
SELECT "Ticker"          AS ticker,
       "P/E (LTM)"       AS p_e_ratio,
       "P/B (LTM)"       AS p_b_ratio,
       "EV/EBITDA (LTM)" AS ev_ebitda_ratio,
       "EV/Sales (LTM)"  AS ev_sales_ratio,
       "Div Yield (LTM)" AS dividend_yield,
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0
               THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0)
           END           AS peg_ratio
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Valuation Timeseries Features (engineer_valuation_timeseries_features)
CREATE OR REPLACE FUNCTION calc_valuation_timeseries_features()
    RETURNS TABLE
            (
                ticker                     TEXT,
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
AS
$$
SELECT "Ticker"                                                                         AS ticker,
       -- EV/Sales 1Y Trend (NULLIF handles zero division, returning NULL)
       ("EV/Sales (LTM)" - "EV/Sales (-1FYLTM)") / NULLIF("EV/Sales (-1FYLTM)", 0)      AS ev_sales_trend_1y,
       -- EV/EBITDA Momentum
       ("EV/EBITDA (LTM)" - "EV/EBITDA (-1FYLTM)") / NULLIF("EV/EBITDA (-1FYLTM)", 0)   AS ev_ebitda_momentum,
       -- P/E Momentum YoY
       ("P/E (LTM)" - "P/E (-1FYLTM)") / NULLIF("P/E (-1FYLTM)", 0)                     AS p_e_momentum_yoy,
       -- P/E Momentum QoQ
       ("P/E (LTM)" - "P/E (-1FQLTM)") / NULLIF("P/E (-1FQLTM)", 0)                     AS p_e_momentum_qoq,
       -- Mean Reversion Features
       ("EV/Sales (LTM)" - "EV/Sales (3YAVGLTM)") / NULLIF("EV/Sales (3YAVGLTM)", 0)    AS ev_sales_vs_3y_avg,
       ("EV/EBITDA (LTM)" - "EV/EBITDA (3YAVGLTM)") / NULLIF("EV/EBITDA (3YAVGLTM)", 0) AS ev_ebitda_vs_3y_avg,
       ("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF("P/E (3YAVGLTM)", 0)                   AS p_e_vs_3y_avg,
       -- Forward vs Trailing Discount
       ("EV/Sales (NTM)" - "EV/Sales (LTM)") / NULLIF("EV/Sales (LTM)", 0)              AS ev_sales_forward_discount,
       ("EV/EBITDA (NTM)" - "EV/EBITDA (LTM)") / NULLIF("EV/EBITDA (LTM)", 0)           AS ev_ebitda_forward_discount,
       ("P/E (EST FY1)" - "P/E (LTM)") / NULLIF("P/E (LTM)", 0)                         AS p_e_forward_discount,
       -- P/B vs 5Y Average
       "P/B (LTM)" / NULLIF("P/B (5YAVG)", 0)                                           AS p_b_vs_5y_avg
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Extended Valuation Timeseries (engineer_valuation_timeseries_features)
CREATE OR REPLACE FUNCTION calc_extended_valuation_timeseries()
    RETURNS TABLE
            (
                ticker                   TEXT,
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
AS
$$
SELECT "Ticker"                                                                    AS ticker,
       -- EV/Sales QoQ Trend (multiple quarters)
       ("EV/Sales (LTM)" - "EV/Sales (-1FQLTM)") / NULLIF("EV/Sales (-1FQLTM)", 0) AS ev_sales_qoq_1q,
       ("EV/Sales (-1FQLTM)" - "EV/Sales (-2FQLTM)") / NULLIF("EV/Sales (-2FQLTM)", 0)
                                                                                   AS ev_sales_qoq_2q,
       ("EV/Sales (-2FQLTM)" - "EV/Sales (-3FQLTM)") / NULLIF("EV/Sales (-3FQLTM)", 0)
                                                                                   AS ev_sales_qoq_3q,
       ("EV/Sales (-3FQLTM)" - "EV/Sales (-4FQLTM)") / NULLIF("EV/Sales (-4FQLTM)", 0)
                                                                                   AS ev_sales_qoq_4q,

       -- P/E vs 5Y Average
       ("P/E (LTM)" - "P/E (5YAVGLTM)") / NULLIF("P/E (5YAVGLTM)", 0)              AS p_e_vs_5y_avg,

       -- P/E Percentile Proxy (vs 3Y range)
       CASE
           WHEN "P/E (LTM)" IS NOT NULL AND "P/E (3YAVGLTM)" IS NOT NULL
               THEN ("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF(ABS("P/E (3YAVGLTM)") * 0.5, 0)
           END                                                                     AS p_e_percentile_proxy,

       -- Valuation Mean Reversion Signal (composite)
       (("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF("P/E (3YAVGLTM)", 0) +
        ("EV/Sales (LTM)" - "EV/Sales (3YAVGLTM)") / NULLIF("EV/Sales (3YAVGLTM)", 0) +
        ("EV/EBITDA (LTM)" - "EV/EBITDA (3YAVGLTM)") / NULLIF("EV/EBITDA (3YAVGLTM)", 0)) / 3.0
                                                                                   AS valuation_mean_reversion,

       -- EV/EBITDA QoQ Trend
       ("EV/EBITDA (LTM)" - "EV/EBITDA (-1FQLTM)") / NULLIF("EV/EBITDA (-1FQLTM)", 0)
                                                                                   AS ev_ebitda_qoq_trend,

       -- P/B Momentum YoY
       ("P/B (LTM)" - "P/B (-1FY)") / NULLIF("P/B (-1FY)", 0)                      AS p_b_momentum_yoy,

       -- Valuation Compression (current vs historical)
       (("P/E (LTM)" / NULLIF("P/E (3YAVGLTM)", 0)) +
        ("EV/EBITDA (LTM)" / NULLIF("EV/EBITDA (3YAVGLTM)", 0))) / 2.0 - 1.0       AS valuation_compression,

       -- Forward P/E Premium over Trailing
       ("P/E (EST FY1)" - "P/E (LTM)") / NULLIF(ABS("P/E (LTM)"), 0) * 100         AS forward_pe_premium

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 2: MOMENTUM & TECHNICAL FEATURES
-- =============================================================================

-- Momentum Features (engineer_momentum_features)
CREATE OR REPLACE FUNCTION calc_momentum_features()
    RETURNS TABLE
            (
                ticker               TEXT,
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
AS
$$
SELECT "Ticker"                                                              AS ticker,
       -- Price Momentum (NULLIF handles zero division, returning NULL)
       ("Last Price" - "Price (1M Ago)") / NULLIF("Price (1M Ago)", 0) * 100 AS price_momentum_1m,
       ("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100 AS price_momentum_3m,
       ("Last Price" - "Price (6M Ago)") / NULLIF("Price (6M Ago)", 0) * 100 AS price_momentum_6m,
       ("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 100 AS price_momentum_1y,
       ("Last Price" - "Price (5D Ago)") / NULLIF("Price (5D Ago)", 0) * 100 AS price_momentum_5d,
       -- EMA Crossovers
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" THEN -1
           ELSE 0
           END                                                               AS ema_crossover_20_50,
       CASE
           WHEN "EMA (50D)" > "EMA (250D)" THEN 1
           WHEN "EMA (50D)" < "EMA (250D)" THEN -1
           ELSE 0
           END                                                               AS ema_crossover_50_250,
       -- Price vs EMA
       ("Last Price" - "EMA (20D)") / NULLIF("EMA (20D)", 0)                 AS price_vs_ema_20d,
       ("Last Price" - "EMA (250D)") / NULLIF("EMA (250D)", 0)               AS price_vs_ema_250d,
       -- 52W Range Position
       ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0)           AS pct_off_52w_high,
       ("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0)             AS pct_above_52w_low,
       LEAST(1, GREATEST(0, ("Last Price" - "52W Low/Adj") /
                            NULLIF("52W High/Adj" - "52W Low/Adj", 0)))      AS range_52w_position,
       -- Beta Momentum
       "Beta (1Y)" - "Beta (5Y)"                                             AS beta_momentum,
       -- Volatility Regime
       "Volatility (1M)" / NULLIF("Volatility (1Y)", 0)                      AS volatility_regime
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 2A: TECHNICAL ANALYSIS FEATURES (NEW)
-- =============================================================================

-- Technical Analysis Features (engineer_technical_analysis_features)
CREATE OR REPLACE FUNCTION calc_technical_analysis_features()
    RETURNS TABLE
            (
                ticker                    TEXT,
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
AS
$$
SELECT "Ticker"                                                      AS ticker,
       -- EMA Slope (20D vs 50D as proxy for short-term trend direction)
       ("EMA (20D)" - "EMA (50D)") / NULLIF("EMA (50D)", 0)          AS ema_slope_20d,

       -- EMA Trend Consistency (all EMAs aligned: bullish=1, bearish=-1, mixed=0)
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" AND "EMA (50D)" > "EMA (100D)"
               AND "EMA (100D)" > "EMA (250D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" AND "EMA (50D)" < "EMA (100D)"
               AND "EMA (100D)" < "EMA (250D)" THEN -1
           ELSE 0
           END                                                       AS ema_trend_consistency,

       -- Price vs EMA 100D (medium-term deviation)
       ("Last Price" - "EMA (100D)") / NULLIF("EMA (100D)", 0) * 100 AS price_vs_ema_100d,

       -- Near 52W High Flag (within 5% of high)
       CASE
           WHEN ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_high_flag,

       -- Near 52W Low Flag (within 5% of low)
       CASE
           WHEN ("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_low_flag,

       -- Volume Momentum Score (Relative Volume × 1M Price Change)
       "Rel. Volume" * "Price Chg. % (1M)"                           AS volume_momentum_score,

       -- Breakout Signal (EMA bullish crossover + near 52W high)
       CASE
           WHEN "EMA (20D)" > "EMA (50D)"
               AND ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS breakout_signal,

       -- High Volume Flag (Relative Volume > 1.5x average)
       CASE WHEN "Rel. Volume" > 1.5 THEN 1 ELSE 0 END               AS high_volume_flag,

       -- Low Volume Flag (Relative Volume < 0.5x average)
       CASE WHEN "Rel. Volume" < 0.5 THEN 1 ELSE 0 END               AS low_volume_flag,

       -- Volatility Compression (1Y - 1M, positive = vol decreasing)
       "Volatility (1Y)" - "Volatility (1M)"                         AS volatility_compression,

       -- Volatility Term Structure (3M vs 6M)
       "Volatility (3M)" - "Volatility (6M)"                         AS volatility_term_structure

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 3: PROFITABILITY FEATURES
-- =============================================================================

-- Profitability Ratios (engineer_profitability_ratios)
CREATE OR REPLACE FUNCTION calc_profitability_features()
    RETURNS TABLE
            (
                ticker               TEXT,
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
AS
$$
SELECT "Ticker"                                                                               AS ticker,
       "Return On Equity % (LTM)"                                                             AS roe,
       "Return on Assets (ROA) % (LTM)"                                                       AS roa,
       "Gross Profit Margin % (LTM)"                                                          AS gross_margin_pct,
       -- Operating Margin (NULLIF handles zero division)
       "Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                     AS operating_margin_pct,
       "Net Income Margin % (LTM)"                                                            AS net_margin_pct,
       -- EBITDA Margin
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                               AS ebitda_margin_pct,
       -- ROIC: Net Income / (Total Equity + Total Debt)
       "Net Income - (IS) (LTM)" / NULLIF("Total Equity (LTM)" + "Total Debt (LTM)", 0) * 100 AS roic,
       -- R&D Intensity
       "R&D Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)                               AS rnd_intensity,
       -- Equity Multiplier (DuPont)
       "Total Assets (LTM)" / NULLIF("Total Equity (LTM)", 0)                                 AS equity_multiplier
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Comprehensive EBIT/EBITDA Features with ALL temporal variants
CREATE OR REPLACE FUNCTION calc_ebit_ebitda_comprehensive()
    RETURNS TABLE
            (
                ticker                   TEXT,
                -- EBIT Raw Values (ALL periods)
                ebit_fq                  NUMERIC,
                ebit_ltm                 NUMERIC,
                ebit_fy                  NUMERIC,
                ebit_1fy                 NUMERIC,
                ebit_2fy                 NUMERIC,
                ebit_3fy                 NUMERIC,
                ebit_4fy                 NUMERIC,
                ebit_1fqfq               NUMERIC,
                ebit_2fqfq               NUMERIC,
                ebit_3fqfq               NUMERIC,
                ebit_4fqfq               NUMERIC,
                ebit_5yavg               NUMERIC,
                -- EBIT Adjusted Values (ALL periods)
                ebit_adj_fq              NUMERIC,
                ebit_adj_ltm             NUMERIC,
                ebit_adj_fy              NUMERIC,
                ebit_adj_1fy             NUMERIC,
                ebit_adj_2fy             NUMERIC,
                ebit_adj_3fy             NUMERIC,
                ebit_adj_4fy             NUMERIC,
                ebit_adj_1fqfq           NUMERIC,
                ebit_adj_2fqfq           NUMERIC,
                ebit_adj_3fqfq           NUMERIC,
                ebit_adj_4fqfq           NUMERIC,
                -- EBITDA Raw Values (ALL periods)
                ebitda_fq                NUMERIC,
                ebitda_ltm               NUMERIC,
                ebitda_fy                NUMERIC,
                ebitda_1fy               NUMERIC,
                ebitda_2fy               NUMERIC,
                ebitda_3fy               NUMERIC,
                ebitda_4fy               NUMERIC,
                ebitda_1fqfq             NUMERIC,
                ebitda_2fqfq             NUMERIC,
                ebitda_3fqfq             NUMERIC,
                ebitda_4fqfq             NUMERIC,
                ebitda_5yavg_fq          NUMERIC,
                ebitda_5yavg_ltm         NUMERIC,
                -- EBITDA Adjusted Values (ALL periods)
                ebitda_adj_fq            NUMERIC,
                ebitda_adj_ltm           NUMERIC,
                ebitda_adj_fy            NUMERIC,
                ebitda_adj_1fy           NUMERIC,
                ebitda_adj_2fy           NUMERIC,
                ebitda_adj_3fy           NUMERIC,
                ebitda_adj_4fy           NUMERIC,
                ebitda_adj_1fqfq         NUMERIC,
                ebitda_adj_2fqfq         NUMERIC,
                ebitda_adj_3fqfq         NUMERIC,
                ebitda_adj_4fqfq         NUMERIC,
                -- Growth & Trends
                ebit_growth_yoy          NUMERIC,
                ebit_growth_qoq          NUMERIC,
                ebitda_growth_yoy        NUMERIC,
                ebitda_growth_qoq        NUMERIC,
                ebit_cagr_3y             NUMERIC,
                ebitda_cagr_3y           NUMERIC,
                -- Margins
                ebit_margin_ltm          NUMERIC,
                ebit_margin_fy           NUMERIC,
                ebitda_margin_ltm        NUMERIC,
                ebitda_margin_fy         NUMERIC,
                ebit_margin_trend        NUMERIC,
                ebitda_margin_trend      NUMERIC,
                -- Adjustment Analytics
                ebit_adjustment_ratio    NUMERIC,
                ebitda_adjustment_ratio  NUMERIC,
                -- Consistency Metrics
                ebit_positive_years      INTEGER,
                ebitda_positive_years    INTEGER,
                ebit_improvement_count   INTEGER,
                ebitda_improvement_count INTEGER
            )
AS
$$
SELECT "Ticker"                                                                  AS ticker,
       -- EBIT Raw Values
       "EBIT (FQ)"                                                               AS ebit_fq,
       "EBIT (LTM)"                                                              AS ebit_ltm,
       "EBIT (FY)"                                                               AS ebit_fy,
       "EBIT (-1FY)"                                                             AS ebit_1fy,
       "EBIT (-2FY)"                                                             AS ebit_2fy,
       "EBIT (-3FY)"                                                             AS ebit_3fy,
       "EBIT (-4FY)"                                                             AS ebit_4fy,
       "EBIT (-1FQFQ)"                                                           AS ebit_1fqfq,
       "EBIT (-2FQFQ)"                                                           AS ebit_2fqfq,
       "EBIT (-3FQFQ)"                                                           AS ebit_3fqfq,
       "EBIT (-4FQFQ)"                                                           AS ebit_4fqfq,
       "EBIT (5YAVGFQ)"                                                          AS ebit_5yavg,
       -- EBIT Adjusted Values
       "EBIT/Adj. (FQ)"                                                          AS ebit_adj_fq,
       "EBIT/Adj. (LTM)"                                                         AS ebit_adj_ltm,
       "EBIT/Adj. (FY)"                                                          AS ebit_adj_fy,
       "EBIT/Adj. (-1FY)"                                                        AS ebit_adj_1fy,
       "EBIT/Adj. (-2FY)"                                                        AS ebit_adj_2fy,
       "EBIT/Adj. (-3FY)"                                                        AS ebit_adj_3fy,
       "EBIT/Adj. (-4FY)"                                                        AS ebit_adj_4fy,
       "EBIT/Adj. (-1FQFQ)"                                                      AS ebit_adj_1fqfq,
       "EBIT/Adj. (-2FQFQ)"                                                      AS ebit_adj_2fqfq,
       "EBIT/Adj. (-3FQFQ)"                                                      AS ebit_adj_3fqfq,
       "EBIT/Adj. (-4FQFQ)"                                                      AS ebit_adj_4fqfq,
       -- EBITDA Raw Values
       "EBITDA (FQ)"                                                             AS ebitda_fq,
       "EBITDA (LTM)"                                                            AS ebitda_ltm,
       "EBITDA (FY)"                                                             AS ebitda_fy,
       "EBITDA (-1FY)"                                                           AS ebitda_1fy,
       "EBITDA (-2FY)"                                                           AS ebitda_2fy,
       "EBITDA (-3FY)"                                                           AS ebitda_3fy,
       "EBITDA (-4FY)"                                                           AS ebitda_4fy,
       "EBITDA (-1FQFQ)"                                                         AS ebitda_1fqfq,
       "EBITDA (-2FQFQ)"                                                         AS ebitda_2fqfq,
       "EBITDA (-3FQFQ)"                                                         AS ebitda_3fqfq,
       "EBITDA (-4FQFQ)"                                                         AS ebitda_4fqfq,
       "EBITDA (5YAVGFQ)"                                                        AS ebitda_5yavg_fq,
       "EBITDA (5YAVGLTM)"                                                       AS ebitda_5yavg_ltm,
       -- EBITDA Adjusted Values
       "EBITDA/Adj. (FQ)"                                                        AS ebitda_adj_fq,
       "EBITDA/Adj. (LTM)"                                                       AS ebitda_adj_ltm,
       "EBITDA/Adj. (FY)"                                                        AS ebitda_adj_fy,
       "EBITDA/Adj. (-1FY)"                                                      AS ebitda_adj_1fy,
       "EBITDA/Adj. (-2FY)"                                                      AS ebitda_adj_2fy,
       "EBITDA/Adj. (-3FY)"                                                      AS ebitda_adj_3fy,
       "EBITDA/Adj. (-4FY)"                                                      AS ebitda_adj_4fy,
       "EBITDA/Adj. (-1FQFQ)"                                                    AS ebitda_adj_1fqfq,
       "EBITDA/Adj. (-2FQFQ)"                                                    AS ebitda_adj_2fqfq,
       "EBITDA/Adj. (-3FQFQ)"                                                    AS ebitda_adj_3fqfq,
       "EBITDA/Adj. (-4FQFQ)"                                                    AS ebitda_adj_4fqfq,
       -- Growth Trends
       ("EBIT (FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) * 100       AS ebit_growth_yoy,
       ("EBIT (FQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) * 100   AS ebit_growth_qoq,
       ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100 AS ebitda_growth_yoy,
       ("EBITDA (FQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) *
       100                                                                       AS ebitda_growth_qoq,
       -- CAGR (3Y)
       CASE
           WHEN "EBIT (-3FY)" > 0 AND "EBIT (FY)" > 0
               THEN (POWER("EBIT (FY)" / NULLIF("EBIT (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebit_cagr_3y,
       CASE
           WHEN "EBITDA (-3FY)" > 0 AND "EBITDA (FY)" > 0
               THEN (POWER("EBITDA (FY)" / NULLIF("EBITDA (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebitda_cagr_3y,
       -- Margins
       "EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                    AS ebit_margin_ltm,
       "EBIT (FY)" / NULLIF("Total Revenues (FY)", 0) * 100                      AS ebit_margin_fy,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                  AS ebitda_margin_ltm,
       "EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0) * 100                    AS ebitda_margin_fy,
       ("EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
       ("EBIT (FY)" / NULLIF("Total Revenues (FY)", 0))                          AS ebit_margin_trend,
       ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
       ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))                        AS ebitda_margin_trend,
       -- Adjustment Analytics
       "EBIT/Adj. (LTM)" / NULLIF("EBIT (LTM)", 0)                               AS ebit_adjustment_ratio,
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0)                           AS ebitda_adjustment_ratio,
       -- Consistency Metrics
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
       (CASE WHEN "EBIT (FY)" > "EBIT (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-1FY)" > "EBIT (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-2FY)" > "EBIT (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-3FY)" > "EBIT (-4FY)" THEN 1 ELSE 0 END)::INTEGER      AS ebit_improvement_count,
       (CASE WHEN "EBITDA (FY)" > "EBITDA (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-1FY)" > "EBITDA (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-2FY)" > "EBITDA (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-3FY)" > "EBITDA (-4FY)" THEN 1 ELSE 0 END)::INTEGER  AS ebitda_improvement_count
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Comprehensive Net Income Features (ALL variants)
CREATE OR REPLACE FUNCTION calc_net_income_comprehensive()
    RETURNS TABLE
            (
                ticker                       TEXT,
                -- Net Income (IS) - ALL periods
                net_income_is_fq             NUMERIC,
                net_income_is_ltm            NUMERIC,
                net_income_is_fy             NUMERIC,
                net_income_is_1fy            NUMERIC,
                net_income_is_2fy            NUMERIC,
                net_income_is_3fy            NUMERIC,
                net_income_is_4fy            NUMERIC,
                net_income_is_1fqfq          NUMERIC,
                net_income_is_2fqfq          NUMERIC,
                net_income_is_3fqfq          NUMERIC,
                net_income_is_4fqfq          NUMERIC,
                net_income_is_5yavg_fq       NUMERIC,
                net_income_is_5yavg_ltm      NUMERIC,
                -- Net Income Adjusted - ALL periods
                net_income_adj_fq            NUMERIC,
                net_income_adj_ltm           NUMERIC,
                net_income_adj_fy            NUMERIC,
                net_income_adj_1fy           NUMERIC,
                net_income_adj_2fy           NUMERIC,
                net_income_adj_3fy           NUMERIC,
                net_income_adj_4fy           NUMERIC,
                net_income_adj_1fqfq         NUMERIC,
                net_income_adj_2fqfq         NUMERIC,
                net_income_adj_3fqfq         NUMERIC,
                net_income_adj_4fqfq         NUMERIC,
                net_income_adj_5yavg         NUMERIC,
                -- Normalized Net Income - ALL periods
                normalized_ni_fq             NUMERIC,
                normalized_ni_ltm            NUMERIC,
                normalized_ni_fy             NUMERIC,
                normalized_ni_1fy            NUMERIC,
                normalized_ni_2fy            NUMERIC,
                normalized_ni_3fy            NUMERIC,
                normalized_ni_4fy            NUMERIC,
                normalized_ni_1fqfq          NUMERIC,
                normalized_ni_2fqfq          NUMERIC,
                normalized_ni_3fqfq          NUMERIC,
                normalized_ni_4fqfq          NUMERIC,
                normalized_ni_5yavg_fq       NUMERIC,
                normalized_ni_5yavg_ltm      NUMERIC,
                -- Growth Analytics
                net_income_growth_yoy        NUMERIC,
                net_income_growth_qoq        NUMERIC,
                normalized_ni_growth_yoy     NUMERIC,
                net_income_cagr_3y           NUMERIC,
                -- Margins
                net_income_margin_ltm        NUMERIC,
                net_income_margin_fy         NUMERIC,
                net_income_margin_trend      NUMERIC,
                -- Adjustment Analytics
                ni_adjustment_ratio          NUMERIC,
                ni_normalization_ratio       NUMERIC,
                gaap_vs_adj_spread_pct       NUMERIC,
                -- Consistency Metrics
                net_income_positive_years    INTEGER,
                net_income_improvement_count INTEGER,
                net_income_positive_quarters INTEGER,
                earnings_quality_composite   NUMERIC
            )
AS
$$
SELECT "Ticker"                                                                AS ticker,
       -- Net Income (IS) Values
       "Net Income - (IS) (FQ)"                                                AS net_income_is_fq,
       "Net Income - (IS) (LTM)"                                               AS net_income_is_ltm,
       "Net Income - (IS) (FY)"                                                AS net_income_is_fy,
       "Net Income - (IS) (-1FY)"                                              AS net_income_is_1fy,
       "Net Income - (IS) (-2FY)"                                              AS net_income_is_2fy,
       "Net Income - (IS) (-3FY)"                                              AS net_income_is_3fy,
       "Net Income - (IS) (-4FY)"                                              AS net_income_is_4fy,
       "Net Income - (IS) (-1FQFQ)"                                            AS net_income_is_1fqfq,
       "Net Income - (IS) (-2FQFQ)"                                            AS net_income_is_2fqfq,
       "Net Income - (IS) (-3FQFQ)"                                            AS net_income_is_3fqfq,
       "Net Income - (IS) (-4FQFQ)"                                            AS net_income_is_4fqfq,
       "Net Income - (IS) (5YAVGFQ)"                                           AS net_income_is_5yavg_fq,
       "Net Income - (IS) (5YAVGLTM)"                                          AS net_income_is_5yavg_ltm,
       -- Net Income Adjusted Values
       "Net Income/Adj. (FQ)"                                                  AS net_income_adj_fq,
       "Net Income/Adj. (LTM)"                                                 AS net_income_adj_ltm,
       "Net Income/Adj. (FY)"                                                  AS net_income_adj_fy,
       "Net Income/Adj. (-1FY)"                                                AS net_income_adj_1fy,
       "Net Income/Adj. (-2FY)"                                                AS net_income_adj_2fy,
       "Net Income/Adj. (-3FY)"                                                AS net_income_adj_3fy,
       "Net Income/Adj. (-4FY)"                                                AS net_income_adj_4fy,
       "Net Income/Adj. (-1FQFQ)"                                              AS net_income_adj_1fqfq,
       "Net Income/Adj. (-2FQFQ)"                                              AS net_income_adj_2fqfq,
       "Net Income/Adj. (-3FQFQ)"                                              AS net_income_adj_3fqfq,
       "Net Income/Adj. (-4FQFQ)"                                              AS net_income_adj_4fqfq,
       "Net Income/Adj. (5YAVGFQ)"                                             AS net_income_adj_5yavg,
       -- Normalized Net Income Values
       "Normalized Net Income (FQ)"                                            AS normalized_ni_fq,
       "Normalized Net Income (LTM)"                                           AS normalized_ni_ltm,
       "Normalized Net Income (FY)"                                            AS normalized_ni_fy,
       "Normalized Net Income (-1FY)"                                          AS normalized_ni_1fy,
       "Normalized Net Income (-2FY)"                                          AS normalized_ni_2fy,
       "Normalized Net Income (-3FY)"                                          AS normalized_ni_3fy,
       "Normalized Net Income (-4FY)"                                          AS normalized_ni_4fy,
       "Normalized Net Income (-1FQFQ)"                                        AS normalized_ni_1fqfq,
       "Normalized Net Income (-2FQFQ)"                                        AS normalized_ni_2fqfq,
       "Normalized Net Income (-3FQFQ)"                                        AS normalized_ni_3fqfq,
       "Normalized Net Income (-4FQFQ)"                                        AS normalized_ni_4fqfq,
       "Normalized Net Income (5YAVGFQ)"                                       AS normalized_ni_5yavg_fq,
       "Normalized Net Income (5YAVGLTM)"                                      AS normalized_ni_5yavg_ltm,
       -- Growth Analytics
       ("Net Income - (IS) (FY)" - "Net Income - (IS) (-1FY)") /
       NULLIF(ABS("Net Income - (IS) (-1FY)"), 0) * 100                        AS net_income_growth_yoy,
       ("Net Income - (IS) (FQ)" - "Net Income - (IS) (-1FQFQ)") /
       NULLIF(ABS("Net Income - (IS) (-1FQFQ)"), 0) * 100                      AS net_income_growth_qoq,
       ("Normalized Net Income (FY)" - "Normalized Net Income (-1FY)") /
       NULLIF(ABS("Normalized Net Income (-1FY)"), 0) * 100                    AS normalized_ni_growth_yoy,
       CASE
           WHEN "Net Income - (IS) (-3FY)" > 0 AND "Net Income - (IS) (FY)" > 0
               THEN (POWER("Net Income - (IS) (FY)" / NULLIF("Net Income - (IS) (-3FY)", 0), 1.0 / 3.0) - 1) *
                    100
           END                                                                 AS net_income_cagr_3y,
       -- Margins
       "Net Income Margin % (LTM)"                                             AS net_income_margin_ltm,
       "Net Income Margin % (FY)"                                              AS net_income_margin_fy,
       "Net Income Margin % (LTM)" - "Net Income Margin % (FY)"                AS net_income_margin_trend,
       -- Adjustment Analytics
       "Net Income/Adj. (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)          AS ni_adjustment_ratio,
       "Normalized Net Income (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)    AS ni_normalization_ratio,
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100                         AS gaap_vs_adj_spread_pct,
       -- Consistency Metrics
       (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER   AS net_income_positive_years,
       (CASE WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FY)" > "Net Income - (IS) (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FY)" > "Net Income - (IS) (-4FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                               AS net_income_improvement_count,
       (CASE WHEN "Net Income - (IS) (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS net_income_positive_quarters,
       -- Earnings Quality Composite (100 = best)
       GREATEST(0, LEAST(100,
                         50 +
                         (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 10 ELSE -10 END) +
                         (CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 5 ELSE -5 END) +
                         (CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 5 ELSE -5 END) +
                         (CASE
                              WHEN ABS(("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
                                       NULLIF(ABS("Net Income - (IS) (LTM)"), 0)) < 0.10 THEN 15
                              ELSE -15 END) +
                         (CASE WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 10 ELSE -5 END) +
                         (CASE WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 5 ELSE -5 END)
                   ))                                                          AS earnings_quality_composite
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Margin Trends Features (engineer_margin_trends)
CREATE OR REPLACE FUNCTION calc_margin_trends()
    RETURNS TABLE
            (
                ticker                 TEXT,
                gross_margin_trend_yoy NUMERIC,
                operating_margin_trend NUMERIC,
                net_margin_trend_yoy   NUMERIC,
                ebitda_margin_trend    NUMERIC,
                margin_expansion_flag  INTEGER,
                margin_stability_score NUMERIC
            )
AS
$$
SELECT "Ticker"                                                             AS ticker,
       -- Gross Margin Trend YoY
       ("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)")       AS gross_margin_trend_yoy,

       -- Operating Margin Trend (LTM vs FY)
       (("Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("Operating Income (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100 AS operating_margin_trend,

       -- Net Margin Trend YoY
       ("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")           AS net_margin_trend_yoy,

       -- EBITDA Margin Trend
       (("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100           AS ebitda_margin_trend,

       -- Margin Expansion Flag (all margins improving)
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               AND "Net Income Margin % (LTM)" > "Net Income Margin % (FY)"
               AND ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) >
                   ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))
               THEN 1
           ELSE 0
           END                                                              AS margin_expansion_flag,

       -- Margin Stability Score (inverse of margin volatility, 0-100)
       GREATEST(0, LEAST(100,
                         100 - (ABS("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)") +
                                ABS("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")) / 2
                   ))                                                       AS margin_stability_score

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 4: QUALITY & RISK FEATURES
-- =============================================================================

-- Quality Features (engineer_accounting_quality_features)
CREATE OR REPLACE FUNCTION calc_quality_features()
    RETURNS TABLE
            (
                ticker                      TEXT,
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
AS
$$
SELECT "Ticker"                                                                                          AS ticker,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END                                   AS has_goodwill_impairment,
       CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 1 ELSE 0 END                                          AS has_asset_writedown,
       CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 1 ELSE 0 END                                    AS has_restructuring,
       -- Goodwill to Assets (NULLIF handles zero division)
       "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                                          AS goodwill_to_assets_pct,
       -- Intangible Intensity
       "Gross Intangible Assets (LTM)" / NULLIF("Total Assets (LTM)", 0)                                 AS intangible_intensity,
       -- Exceptional Items to EBITDA
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)                                                                    AS exceptional_items_to_ebitda,
       -- Altman Z-Score
       "Altman Z-Score (LTM)"                                                                            AS altman_z_score,
       -- Altman Z Trend
       "Altman Z-Score (FY)" - "Altman Z-Score (LTM)"                                                    AS altman_z_trend,
       -- Liquidity Ratios
       "Current Ratio (LTM)"                                                                             AS current_ratio,
       -- Quick Ratio
       ("Total Current Assets (LTM)" - "Inventory (LTM)") / NULLIF("Total Current Liabilities (LTM)", 0) AS quick_ratio
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 4A: FINANCIAL DISTRESS FEATURES (NEW)
-- =============================================================================

-- Financial Distress Features (engineer_financial_distress_features)
CREATE OR REPLACE FUNCTION calc_financial_distress_features()
    RETURNS TABLE
            (
                ticker                   TEXT,
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
AS
$$
SELECT "Ticker"                                                                          AS ticker,
       -- Distress Risk Score (map Z-score to 0-100: z<=1.8 → 0, z>=3.0 → 100)
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100))) AS distress_risk_score,

       -- Liquidity Stress Score (higher = more stress)
       CASE
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
           ELSE 0.0
           END                                                                           AS liquidity_stress_score,

       -- Working Capital Trend (QoQ change)
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0)                                            AS working_capital_trend,

       -- Cash Runway (months of OpEx coverage)
       "Cash And Equivalents (FQ)" /
       NULLIF("Total Operating Expenses (LTM)" / 12.0, 0)                                AS cash_runway_months,

       -- Combined Distress Score (70% Z-score + 30% liquidity)
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 70) +
                         (100 - CASE
                                    WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
                                    WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
                                    ELSE 0.0
                             END) * 0.30))                                               AS combined_distress_score,

       -- Working Capital Deteriorating Flag
       CASE
           WHEN ("Working Capital (FQ)" - "Working Capital (FY)") /
                NULLIF(ABS("Working Capital (FY)"), 0) < -0.2
               THEN 1
           ELSE 0
           END                                                                           AS wc_deteriorating_flag,

       -- Retained Earnings Growth (FQ vs FY)
       ("Retained Earnings (FQ)" - "Retained Earnings (FY)") /
       NULLIF(ABS("Retained Earnings (FY)"), 0)                                          AS retained_earnings_growth,

       -- Accumulated Deficit Flag (negative retained earnings)
       CASE WHEN "Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END                          AS accumulated_deficit_flag,

       -- Adequate Cash Buffer (> 6 months runway)
       CASE
           WHEN "Cash And Equivalents (FQ)" /
                NULLIF("Total Operating Expenses (LTM)" / 12.0, 0) > 6
               THEN 1
           ELSE 0
           END                                                                           AS adequate_cash_buffer

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 4B: ACCOUNTING QUALITY FEATURES (NEW)
-- =============================================================================

-- Enhanced Accounting Quality Features
CREATE OR REPLACE FUNCTION calc_accounting_quality_features()
    RETURNS TABLE
            (
                ticker                      TEXT,
                goodwill_change_rate        NUMERIC,
                restructuring_intensity     NUMERIC,
                exceptional_items_frequency INTEGER,
                merger_impact_ratio         NUMERIC,
                non_operating_income_share  NUMERIC,
                asset_sale_boost            INTEGER,
                accounting_quality_score    NUMERIC
            )
AS
$$
SELECT "Ticker"                                                              AS ticker,
       -- Goodwill Change Rate (YoY)
       ("Goodwill (LTM)" - "Goodwill (-1FY)") / NULLIF("Goodwill (-1FY)", 0) AS goodwill_change_rate,

       -- Restructuring Intensity (to Total Assets)
       "Restructuring Charges (LTM)" / NULLIF("Total Assets (LTM)", 0)       AS restructuring_intensity,

       -- Exceptional Items Frequency (count of non-zero exceptional items)
       (CASE WHEN ABS("Impairment of Goodwill (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Asset Writedown (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Restructuring Charges (FQ)") > 0 THEN 1 ELSE 0 END)   AS exceptional_items_frequency,

       -- Merger Impact Ratio (Merger Charges / Market Cap)
       "Merger & Restructuring Charges (LTM)" / NULLIF("Market Cap", 0)      AS merger_impact_ratio,

       -- Non-Operating Income Share (Interest Income / Net Income)
       "Interest Income On Investments (LTM)" / NULLIF(ABS("Net Income - (IS) (LTM)"), 0)
                                                                             AS non_operating_income_share,

       -- Asset Sale Boost Flag (gain on sale of assets > 0)
       CASE WHEN "Gain (Loss) On Sale Of Assets (LTM)" > 0 THEN 1 ELSE 0 END AS asset_sale_boost,

       -- Composite Accounting Quality Score (100 = highest quality)
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

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Comprehensive Quality Features with ALL period variants
CREATE OR REPLACE FUNCTION calc_quality_features_comprehensive()
    RETURNS TABLE
            (
                ticker                        TEXT,
                -- Goodwill Impairment (ALL periods)
                goodwill_impairment_fq        NUMERIC,
                goodwill_impairment_ltm       NUMERIC,
                goodwill_impairment_fy        NUMERIC,
                goodwill_impairment_1fy       NUMERIC,
                goodwill_impairment_2fy       NUMERIC,
                goodwill_impairment_3fy       NUMERIC,
                goodwill_impairment_4fy       NUMERIC,
                goodwill_impairment_1fqfq     NUMERIC,
                goodwill_impairment_2fqfq     NUMERIC,
                goodwill_impairment_3fqfq     NUMERIC,
                goodwill_impairment_4fqfq     NUMERIC,
                goodwill_impairment_5yavg     NUMERIC,
                -- Goodwill Impairment Flags
                has_goodwill_impairment_ltm   INTEGER,
                has_goodwill_impairment_fy    INTEGER,
                has_goodwill_impairment_1fy   INTEGER,
                has_goodwill_impairment_2fy   INTEGER,
                has_goodwill_impairment_3fy   INTEGER,
                has_goodwill_impairment_4fy   INTEGER,
                goodwill_impairment_frequency INTEGER,
                -- Asset Writedown (ALL periods)
                asset_writedown_fq            NUMERIC,
                asset_writedown_ltm           NUMERIC,
                asset_writedown_fy            NUMERIC,
                asset_writedown_1fy           NUMERIC,
                asset_writedown_2fy           NUMERIC,
                asset_writedown_3fy           NUMERIC,
                asset_writedown_4fy           NUMERIC,
                asset_writedown_5fy           NUMERIC,
                asset_writedown_1fqfq         NUMERIC,
                asset_writedown_2fqfq         NUMERIC,
                asset_writedown_3fqfq         NUMERIC,
                asset_writedown_4fqfq         NUMERIC,
                asset_writedown_5yavg         NUMERIC,
                asset_writedown_frequency     INTEGER,
                -- Restructuring Charges (ALL periods)
                restructuring_fq              NUMERIC,
                restructuring_ltm             NUMERIC,
                restructuring_fy              NUMERIC,
                restructuring_1fy             NUMERIC,
                restructuring_2fy             NUMERIC,
                restructuring_3fy             NUMERIC,
                restructuring_4fy             NUMERIC,
                restructuring_1fqfq           NUMERIC,
                restructuring_2fqfq           NUMERIC,
                restructuring_3fqfq           NUMERIC,
                restructuring_4fqfq           NUMERIC,
                restructuring_5yavg           NUMERIC,
                restructuring_frequency       INTEGER,
                -- Merger & Restructuring (ALL periods)
                merger_restructuring_fq       NUMERIC,
                merger_restructuring_ltm      NUMERIC,
                merger_restructuring_fy       NUMERIC,
                merger_restructuring_5yavg    NUMERIC,
                -- Gain/Loss on Asset Sales (ALL periods)
                asset_sale_gain_fq            NUMERIC,
                asset_sale_gain_ltm           NUMERIC,
                asset_sale_gain_fy            NUMERIC,
                asset_sale_gain_1fy           NUMERIC,
                asset_sale_gain_2fy           NUMERIC,
                asset_sale_gain_3fy           NUMERIC,
                asset_sale_gain_4fy           NUMERIC,
                asset_sale_gain_1fqfq         NUMERIC,
                asset_sale_gain_2fqfq         NUMERIC,
                asset_sale_gain_3fqfq         NUMERIC,
                asset_sale_gain_4fqfq         NUMERIC,
                -- Trends & Analytics
                goodwill_impairment_trend_yoy NUMERIC,
                goodwill_impairment_trend_qoq NUMERIC,
                asset_writedown_trend_yoy     NUMERIC,
                restructuring_trend_yoy       NUMERIC,
                restructuring_trend_qoq       NUMERIC,
                exceptional_items_total_ltm   NUMERIC,
                exceptional_items_to_revenue  NUMERIC,
                exceptional_items_to_ebitda   NUMERIC,
                quality_issues_count_5y       INTEGER,
                accounting_quality_score      NUMERIC
            )
AS
$$
SELECT "Ticker"                                                                     AS ticker,
       -- Goodwill Impairment Values (ALL periods)
       "Impairment of Goodwill (FQ)"                                                AS goodwill_impairment_fq,
       "Impairment of Goodwill (LTM)"                                               AS goodwill_impairment_ltm,
       "Impairment of Goodwill (FY)"                                                AS goodwill_impairment_fy,
       "Impairment of Goodwill (-1FY)"                                              AS goodwill_impairment_1fy,
       "Impairment of Goodwill (-2FY)"                                              AS goodwill_impairment_2fy,
       "Impairment of Goodwill (-3FY)"                                              AS goodwill_impairment_3fy,
       "Impairment of Goodwill (-4FY)"                                              AS goodwill_impairment_4fy,
       "Impairment of Goodwill (-1FQFQ)"                                            AS goodwill_impairment_1fqfq,
       "Impairment of Goodwill (-2FQFQ)"                                            AS goodwill_impairment_2fqfq,
       "Impairment of Goodwill (-3FQFQ)"                                            AS goodwill_impairment_3fqfq,
       "Impairment of Goodwill (-4FQFQ)"                                            AS goodwill_impairment_4fqfq,
       "Impairment of Goodwill (5YAVGFQ)"                                           AS goodwill_impairment_5yavg,
       -- Goodwill Impairment Flags
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END              AS has_goodwill_impairment_ltm,
       CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END               AS has_goodwill_impairment_fy,
       CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_1fy,
       CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_2fy,
       CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_3fy,
       CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END             AS has_goodwill_impairment_4fy,
       (CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER  AS goodwill_impairment_frequency,
       -- Asset Writedown Values (ALL periods)
       "Asset Writedown (FQ)"                                                       AS asset_writedown_fq,
       "Asset Writedown (LTM)"                                                      AS asset_writedown_ltm,
       "Asset Writedown (FY)"                                                       AS asset_writedown_fy,
       "Asset Writedown (-1FY)"                                                     AS asset_writedown_1fy,
       "Asset Writedown (-2FY)"                                                     AS asset_writedown_2fy,
       "Asset Writedown (-3FY)"                                                     AS asset_writedown_3fy,
       "Asset Writedown (-4FY)"                                                     AS asset_writedown_4fy,
       "Asset Writedown (-5FY)"                                                     AS asset_writedown_5fy,
       "Asset Writedown (-1FQFQ)"                                                   AS asset_writedown_1fqfq,
       "Asset Writedown (-2FQFQ)"                                                   AS asset_writedown_2fqfq,
       "Asset Writedown (-3FQFQ)"                                                   AS asset_writedown_3fqfq,
       "Asset Writedown (-4FQFQ)"                                                   AS asset_writedown_4fqfq,
       "Asset Writedown (5YAVGFQ)"                                                  AS asset_writedown_5yavg,
       (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER         AS asset_writedown_frequency,
       -- Restructuring Charges Values (ALL periods)
       "Restructuring Charges (FQ)"                                                 AS restructuring_fq,
       "Restructuring Charges (LTM)"                                                AS restructuring_ltm,
       "Restructuring Charges (FY)"                                                 AS restructuring_fy,
       "Restructuring Charges (-1FY)"                                               AS restructuring_1fy,
       "Restructuring Charges (-2FY)"                                               AS restructuring_2fy,
       "Restructuring Charges (-3FY)"                                               AS restructuring_3fy,
       "Restructuring Charges (-4FY)"                                               AS restructuring_4fy,
       "Restructuring Charges (-1FQFQ)"                                             AS restructuring_1fqfq,
       "Restructuring Charges (-2FQFQ)"                                             AS restructuring_2fqfq,
       "Restructuring Charges (-3FQFQ)"                                             AS restructuring_3fqfq,
       "Restructuring Charges (-4FQFQ)"                                             AS restructuring_4fqfq,
       "Restructuring Charges (5YAVGFQ)"                                            AS restructuring_5yavg,
       (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER   AS restructuring_frequency,
       -- Merger & Restructuring
       "Merger & Restructuring Charges (FQ)"                                        AS merger_restructuring_fq,
       "Merger & Restructuring Charges (LTM)"                                       AS merger_restructuring_ltm,
       "Merger & Restructuring Charges (FY)"                                        AS merger_restructuring_fy,
       "Merger & Restructuring Charges (5YAVGFQ)"                                   AS merger_restructuring_5yavg,
       -- Gain/Loss on Asset Sales
       "Gain (Loss) On Sale Of Assets (FQ)"                                         AS asset_sale_gain_fq,
       "Gain (Loss) On Sale Of Assets (LTM)"                                        AS asset_sale_gain_ltm,
       "Gain (Loss) On Sale Of Assets (FY)"                                         AS asset_sale_gain_fy,
       "Gain (Loss) On Sale Of Assets (-1FY)"                                       AS asset_sale_gain_1fy,
       "Gain (Loss) On Sale Of Assets (-2FY)"                                       AS asset_sale_gain_2fy,
       "Gain (Loss) On Sale Of Assets (-3FY)"                                       AS asset_sale_gain_3fy,
       "Gain (Loss) On Sale Of Assets (-4FY)"                                       AS asset_sale_gain_4fy,
       "Gain (Loss) On Sale Of Assets (-1FQFQ)"                                     AS asset_sale_gain_1fqfq,
       "Gain (Loss) On Sale Of Assets (-2FQFQ)"                                     AS asset_sale_gain_2fqfq,
       "Gain (Loss) On Sale Of Assets (-3FQFQ)"                                     AS asset_sale_gain_3fqfq,
       "Gain (Loss) On Sale Of Assets (-4FQFQ)"                                     AS asset_sale_gain_4fqfq,
       -- Trends
       ("Impairment of Goodwill (FY)" - "Impairment of Goodwill (-1FY)") /
       NULLIF(ABS("Impairment of Goodwill (-1FY)"), 0)                              AS goodwill_impairment_trend_yoy,
       ("Impairment of Goodwill (FQ)" - "Impairment of Goodwill (-1FQFQ)") /
       NULLIF(ABS("Impairment of Goodwill (-1FQFQ)"), 0)                            AS goodwill_impairment_trend_qoq,
       ("Asset Writedown (FY)" - "Asset Writedown (-1FY)") /
       NULLIF(ABS("Asset Writedown (-1FY)"), 0)                                     AS asset_writedown_trend_yoy,
       ("Restructuring Charges (FY)" - "Restructuring Charges (-1FY)") /
       NULLIF(ABS("Restructuring Charges (-1FY)"), 0)                               AS restructuring_trend_yoy,
       ("Restructuring Charges (FQ)" - "Restructuring Charges (-1FQFQ)") /
       NULLIF(ABS("Restructuring Charges (-1FQFQ)"), 0)                             AS restructuring_trend_qoq,
       -- Aggregate Metrics
       ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
       ABS("Restructuring Charges (LTM)")                                           AS exceptional_items_total_ltm,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF("Total Revenues (LTM)", 0)     AS exceptional_items_to_revenue,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF(ABS("EBITDA (LTM)"), 0)        AS exceptional_items_to_ebitda,
       -- Combined 5Y Quality Issues Count
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
       -- Comprehensive Quality Score (100 = best)
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
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 5: LEVERAGE & LIQUIDITY FEATURES
-- =============================================================================

-- Leverage Features (engineer_leverage_ratios, engineer_liquidity_ratios)
CREATE OR REPLACE FUNCTION calc_leverage_features()
    RETURNS TABLE
            (
                ticker                TEXT,
                debt_to_equity        NUMERIC,
                debt_to_assets        NUMERIC,
                equity_ratio          NUMERIC,
                interest_coverage     NUMERIC,
                current_ratio         NUMERIC,
                cash_ratio            NUMERIC,
                working_capital_ratio NUMERIC
            )
AS
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
$$ LANGUAGE SQL;

-- Efficiency Ratios (engineer_efficiency_ratios)
CREATE OR REPLACE FUNCTION calc_efficiency_ratios()
    RETURNS TABLE
            (
                ticker                TEXT,
                asset_turnover        NUMERIC,
                inventory_turnover    NUMERIC,
                receivables_days      NUMERIC,
                working_capital_turns NUMERIC
            )
AS
$$
SELECT "Ticker"                                                                      AS ticker,
       -- Asset Turnover (Revenue / Total Assets)
       "Total Revenues (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS asset_turnover,

       -- Inventory Turnover (COGS / Average Inventory)
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)                       AS inventory_turnover,

       -- Receivables Days (Accounts Receivable / Daily Revenue * 365)
       ("Accounts Receivable/Total (FY)" / NULLIF("Total Revenues (FY)" / 365.0, 0)) AS receivables_days,

       -- Working Capital Turnover (Revenue / Working Capital)
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turns

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Balance Sheet Dynamics (engineer_balance_sheet_trends)
CREATE OR REPLACE FUNCTION calc_balance_sheet_dynamics()
    RETURNS TABLE
            (
                ticker                    TEXT,
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
AS
$$
SELECT "Ticker"                                                                  AS ticker,
       -- Cash to Total Assets %
       "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100      AS cash_to_assets_pct,

       -- Cash Change QoQ
       ("Cash And Equivalents (FQ)" - "Cash And Equivalents (FY)") /
       NULLIF(ABS("Cash And Equivalents (FY)"), 0)                               AS cash_change_qoq,

       -- Cash vs 5Y Average
       "Cash And Equivalents (FQ)" / NULLIF("Cash And Equivalents (5YAVGFQ)", 0) AS cash_vs_5y_avg,

       -- Inventory Change YoY
       ("Inventory (FY)" - "Inventory (FQ)") / NULLIF(ABS("Inventory (FQ)"), 0)  AS inventory_change_yoy,

       -- Inventory vs 5Y Average
       "Inventory (FQ)" / NULLIF("Inventory (5YAVGFQ)", 0)                       AS inventory_vs_5y_avg,

       -- Receivables Change YoY
       ("Accounts Receivable/Total (FY)" - "Accounts Receivable/Total (-1FY)") /
       NULLIF(ABS("Accounts Receivable/Total (-1FY)"), 0)                        AS receivables_change_yoy,

       -- Receivables vs 5Y Average
       "Accounts Receivable/Total (FY)" / NULLIF("Accounts Receivable/Total (5YAVGFQ)", 0)
                                                                                 AS receivables_vs_5y_avg,

       -- Working Capital vs 5Y Average
       "Working Capital (FQ)" / NULLIF("Working Capital (5YAVGFY)", 0)           AS working_capital_vs_5y_avg,

       -- Retained Earnings vs 5Y Average
       "Retained Earnings (FQ)" / NULLIF("Retained Earnings (5YAVGFQ)", 0)       AS retained_earnings_vs_5y,

       -- Intangibles Growth Flag (rapid growth = potential acquisition spree)
       CASE
           WHEN "Gross Intangible Assets (FY)" / NULLIF("Gross Intangible Assets (5YAVGFQ)", 0) > 1.5
               THEN 1
           ELSE 0
           END                                                                   AS intangibles_growth_flag,

       -- Asset Quality Score (lower intangibles ratio + higher cash ratio = better)
       GREATEST(0, LEAST(100,
                         50 + ("Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100) -
                         ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100)
                   ))                                                            AS asset_quality_score,

       -- Balance Sheet Strength (composite: cash, equity, working capital)
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.10 THEN 25
                              ELSE 0 END) +
                         (CASE WHEN "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.40 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Working Capital (LTM)" > 0 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END)
                   ))                                                            AS balance_sheet_strength,

       -- Debt Maturity Risk (Total Debt / EBITDA, higher = more risk)
       "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0)                            AS debt_maturity_risk

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 6: ANALYST SENTIMENT FEATURES
-- =============================================================================

-- Analyst Sentiment Features (engineer_analyst_quality_features, engineer_price_target_dynamics)
CREATE OR REPLACE FUNCTION calc_sentiment_features()
    RETURNS TABLE
            (
                ticker                    TEXT,
                analyst_bullish_pct       NUMERIC,
                analyst_bearish_pct       NUMERIC,
                analyst_conviction        NUMERIC,
                upside_potential          NUMERIC,
                price_target_spread_pct   NUMERIC,
                price_target_revision_1m  NUMERIC,
                price_target_revision_3m  NUMERIC,
                eps_revision_momentum     NUMERIC,
                analyst_rating_normalized NUMERIC,
                analyst_coverage_quality  NUMERIC
            )
AS
$$
SELECT "Ticker"                                                                 AS ticker,
       -- Analyst Distribution
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
       -- Analyst Conviction (difference between bullish and bearish)
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
       -- Upside Potential (NULLIF handles zero division)
       ("Price Target - Median" - "Last Price") / NULLIF("Last Price", 0) * 100 AS upside_potential,
       -- Price Target Spread
       ("Price Target - High" - "Price Target - Low") / NULLIF("Price Target - Median", 0) *
       100                                                                      AS price_target_spread_pct,
       -- Price Target Revisions
       ("Price Target" - "Price Target (1M Ago)") /
       NULLIF("Price Target (1M Ago)", 0)                                       AS price_target_revision_1m,
       ("Price Target" - "Price Target (3M Ago)") /
       NULLIF("Price Target (3M Ago)", 0)                                       AS price_target_revision_3m,
       -- EPS Revision Momentum (weighted average)
       COALESCE("EPS Est Avg Rev % (FY1E - 1W)", 0) * 0.30 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1M)", 0) * 0.25 +
       COALESCE("EPS Est Avg Rev % (FY1E - 3M)", 0) * 0.20 +
       COALESCE("EPS Est Avg Rev % (FY1E - 6M)", 0) * 0.15 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1Y)", 0) *
       0.10                                                                     AS eps_revision_momentum,
       -- Analyst Rating Normalized (1-5 to 0-100)
       ("Analyst Rating" - 1) * 25                                              AS analyst_rating_normalized,
       -- Coverage Quality (LN(1 + x) is always > 0 for positive Market Cap)
       "Price Target - #" / NULLIF(LN(1 + "Market Cap"), 0)                     AS analyst_coverage_quality
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 6A: PRICE TARGET DYNAMICS FEATURES (NEW)
-- =============================================================================

-- Price Target Dynamics Features (engineer_price_target_dynamics)
CREATE OR REPLACE FUNCTION calc_price_target_dynamics()
    RETURNS TABLE
            (
                ticker                     TEXT,
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
AS
$$
SELECT "Ticker"                                                                          AS ticker,
       -- Price Target Momentum (Mean/Average)
       ("Price Target" - "Price Target (1W Ago)") / NULLIF("Price Target (1W Ago)", 0)   AS pt_momentum_1w,
       ("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)   AS pt_momentum_1m,
       ("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)   AS pt_momentum_3m,
       ("Price Target" - "Price Target (6M Ago)") / NULLIF("Price Target (6M Ago)", 0)   AS pt_momentum_6m,
       ("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)   AS pt_momentum_1y,

       -- Price Target Median Momentum (more robust to outliers)
       ("Price Target - Median" - "Price Target - Median (1M Ago)") /
       NULLIF("Price Target - Median (1M Ago)", 0)                                       AS pt_median_momentum_1m,
       ("Price Target - Median" - "Price Target - Median (3M Ago)") /
       NULLIF("Price Target - Median (3M Ago)", 0)                                       AS pt_median_momentum_3m,

       -- Momentum Acceleration (short-term vs medium-term)
       (("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)) -
       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) AS pt_acceleration_short,

       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) -
       (("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)) AS pt_acceleration_long,

       -- Consensus Convergence (spread narrowing = higher analyst agreement)
       (("Price Target - High (3M Ago)" - "Price Target - Low (3M Ago)") /
        NULLIF("Price Target - Median (3M Ago)", 0)) -
       (("Price Target - High" - "Price Target - Low") /
        NULLIF("Price Target - Median", 0))                                              AS pt_consensus_convergence,

       -- Analyst Coverage Changes (absolute change in number of analysts)
       ("Price Target - #" - "Price Target - # (1M Ago)")::INTEGER                       AS analyst_coverage_change_1m,
       ("Price Target - #" - "Price Target - # (3M Ago)")::INTEGER                       AS analyst_coverage_change_3m,
       ("Price Target - #" - "Price Target - # (1Y Ago)")::INTEGER                       AS analyst_coverage_change_1y,

       -- Target vs Price Momentum Divergence
       (("Price Target" / NULLIF("Last Price", 0)) -
        ("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0))) /
       NULLIF(("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0)), 0)                AS pt_vs_price_momentum,

       -- Analyst Coverage Trend (weighted: 1M×0.4 + 3M×0.35 + 6M×0.25) / current count
       (COALESCE("Price Target - #" - "Price Target - # (1M Ago)", 0) * 0.40 +
        COALESCE("Price Target - #" - "Price Target - # (3M Ago)", 0) * 0.35 +
        COALESCE("Price Target - #" - "Price Target - # (6M Ago)", 0) * 0.25) /
       NULLIF("Price Target - #"::NUMERIC, 0)                                            AS analyst_coverage_trend

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 7: EARNINGS FEATURES
-- =============================================================================

-- Earnings Features (engineer_estimated_vs_actual_analytics, engineer_gaap_vs_adjusted_analytics)
CREATE OR REPLACE FUNCTION calc_earnings_features()
    RETURNS TABLE
            (
                ticker                  TEXT,
                eps_surprise_pct        NUMERIC,
                revenue_surprise_pct    NUMERIC,
                eps_adjustment_ratio    NUMERIC,
                gaap_adj_eps_gap_pct    NUMERIC,
                ebitda_adjustment_ratio NUMERIC,
                eps_quarterly_trend     NUMERIC,
                eps_yoy_growth          NUMERIC
            )
AS
$$
SELECT "Ticker"                                              AS ticker,
       -- EPS Surprise (Actual vs Estimate)
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS/Adj. (LTM)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS eps_surprise_pct,
       -- Revenue Surprise
       CASE
           WHEN ABS("Revenues - Est Avg (FY1E)") > 0
               THEN ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
                    NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100
           END                                               AS revenue_surprise_pct,
       -- EPS Adjustment Ratio (Adjusted / GAAP) - NULLIF handles zero division
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0) AS eps_adjustment_ratio,
       -- GAAP vs Adjusted EPS Gap
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS gaap_adj_eps_gap_pct,
       -- EBITDA Adjustment Ratio
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0)       AS ebitda_adjustment_ratio,
       -- EPS Quarterly Trend (FQ vs -4FQFQ for YoY)
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0)
           END                                               AS eps_quarterly_trend,
       -- EPS YoY Growth
       CASE
           WHEN ABS("Net EPS - Basic (-1FY)") > 0
               THEN ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
                    NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100
           END                                               AS eps_yoy_growth
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 7A: EPS TRAJECTORY FEATURES (NEW)
-- =============================================================================

-- EPS Trajectory Features (engineer_eps_trajectory_features)
CREATE OR REPLACE FUNCTION calc_eps_trajectory_features()
    RETURNS TABLE
            (
                ticker                TEXT,
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
AS
$$
SELECT "Ticker"                                                              AS ticker,
       -- Quarter-over-Quarter EPS Growth
       CASE
           WHEN ABS("Net EPS - Basic (-1FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-1FQFQ)"), 0) * 100
           END                                                               AS eps_qoq_growth,

       -- Year-over-Year Quarterly EPS Growth
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0) * 100
           END                                                               AS eps_yoy_quarterly,

       -- EPS Positive Streak (count of positive quarters out of last 5)
       (CASE WHEN "Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_streak,

       -- EPS CAGR 3Y (compound annual growth rate)
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                               AS eps_cagr_3y,

       -- EPS CAGR 5Y
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                               AS eps_cagr_5y,

       -- Growth Acceleration (3Y CAGR - 5Y CAGR, positive = accelerating growth)
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (-5FY)" > 0
               AND "Net EPS - Basic (FY)" > 0
               THEN ((POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                     (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
           END                                                               AS eps_growth_accel,

       -- EPS vs 5-Year Average (current EPS as percentage deviation from 5Y average)
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN ("Net EPS - Basic (FY)" -
                     (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                       "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0)) /
                    NULLIF(ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                                "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0), 0) * 100
           END                                                               AS eps_vs_5y_avg,

       -- EPS Improvement Count (years with YoY improvement out of last 5)
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                             AS eps_improvement_count,

       -- EPS Trajectory Score (% of improving years, scaled to 0-100)
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                     AS eps_trajectory_score,

       -- EPS Stability (inverse of coefficient of variation, higher = more stable)
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN 1 - (STDDEV(val) / NULLIF(ABS(AVG(val)), 0))
           END                                                               AS eps_stability

FROM postgres.public.equities,
     LATERAL (VALUES ("Net EPS - Basic (FY)"),
                     ("Net EPS - Basic (-1FY)"),
                     ("Net EPS - Basic (-2FY)"),
                     ("Net EPS - Basic (-3FY)"),
                     ("Net EPS - Basic (-4FY)")) AS t(val)
GROUP BY "Ticker", "Net EPS - Basic (FQ)", "Net EPS - Basic (-1FQFQ)", "Net EPS - Basic (-2FQFQ)",
         "Net EPS - Basic (-3FQFQ)", "Net EPS - Basic (-4FQFQ)", "Net EPS - Basic (FY)",
         "Net EPS - Basic (-1FY)", "Net EPS - Basic (-2FY)", "Net EPS - Basic (-3FY)",
         "Net EPS - Basic (-4FY)", "Net EPS - Basic (-5FY)";
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 7B: GAAP VS ADJUSTED ANALYTICS (NEW)
-- =============================================================================

-- GAAP vs Adjusted Analytics (engineer_gaap_vs_adjusted_analytics)
CREATE OR REPLACE FUNCTION calc_gaap_adjusted_analytics()
    RETURNS TABLE
            (
                ticker                      TEXT,
                eps_adjustment_spread       NUMERIC,
                eps_adjustment_pct          NUMERIC,
                net_income_adjustment_ratio NUMERIC,
                net_income_adjustment_pct   NUMERIC,
                ebitda_adjustment_pct       NUMERIC,
                earnings_quality_score      NUMERIC,
                earnings_quality_warning    INTEGER,
                forward_eps_gaap_adj_spread NUMERIC
            )
AS
$$
SELECT "Ticker"                                                                     AS ticker,
       -- EPS Adjustment Spread (Adjusted - GAAP, dollar difference)
       "EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)"                                   AS eps_adjustment_spread,

       -- EPS Adjustment Percentage
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
       NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100                                AS eps_adjustment_pct,

       -- Net Income Adjustment Ratio (Adjusted / GAAP)
       "Net Income/Adj. (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS net_income_adjustment_ratio,

       -- Net Income Adjustment Percentage
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100                              AS net_income_adjustment_pct,

       -- EBITDA Adjustment Percentage
       ("EBITDA/Adj. (LTM)" - "EBITDA (LTM)") /
       NULLIF(ABS("EBITDA (LTM)"), 0) * 100                                         AS ebitda_adjustment_pct,

       -- Earnings Quality Score (100 - adjustment %, capped at 0-100, higher = better)
       GREATEST(0, LEAST(100,
                         100 - ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                                   NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100))) AS earnings_quality_score,

       -- Warning flag if adjustment exceeds 15%
       CASE
           WHEN ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                    NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100) > 15
               THEN 1
           ELSE 0
           END                                                                      AS earnings_quality_warning,

       -- Forward EPS GAAP vs Adjusted Spread
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"                    AS forward_eps_gaap_adj_spread

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- GAAP EPS Revision Features (engineer_gaap_vs_adjusted_analytics)
CREATE OR REPLACE FUNCTION calc_gaap_revision_features()
    RETURNS TABLE
            (
                ticker                       TEXT,
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
AS
$$
SELECT "Ticker"                                                                    AS ticker,
       -- GAAP EPS Revision Momentum (weighted average)
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) * 0.15                    AS gaap_revision_momentum,

       -- Individual GAAP Revisions
       "EPS GAAP Est Avg Rev % (FY1E - 1M)"                                        AS gaap_revision_1m,
       "EPS GAAP Est Avg Rev % (FY1E - 3M)"                                        AS gaap_revision_3m,
       "EPS GAAP Est Avg Rev % (FY1E - 6M)"                                        AS gaap_revision_6m,
       "EPS GAAP Est Avg Rev % (FY1E - 1Y)"                                        AS gaap_revision_1y,

       -- GAAP vs Normalized Revision Spread (quality signal)
       "EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)"      AS gaap_vs_norm_revision_spread,

       -- GAAP Revision Acceleration (1M vs 6M)
       "EPS GAAP Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 6M)" AS gaap_revision_acceleration,

       -- GAAP Positive Revision Flag (all periods positive)
       CASE
           WHEN "EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0
               THEN 1
           ELSE 0
           END                                                                     AS gaap_positive_revision_flag,

       -- Revision Quality Divergence (GAAP vs Adjusted moving differently)
       ABS(("EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)") -
           ("EPS Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 1M)"))
                                                                                   AS revision_quality_divergence

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Comprehensive EPS Features with ALL temporal variants
CREATE OR REPLACE FUNCTION calc_eps_comprehensive()
    RETURNS TABLE
            (
                ticker                  TEXT,
                -- Net EPS Basic (ALL periods)
                eps_basic_fq            NUMERIC,
                eps_basic_ltm           NUMERIC,
                eps_basic_fy            NUMERIC,
                eps_basic_1fy           NUMERIC,
                eps_basic_2fy           NUMERIC,
                eps_basic_3fy           NUMERIC,
                eps_basic_4fy           NUMERIC,
                eps_basic_5fy           NUMERIC,
                eps_basic_1fqfq         NUMERIC,
                eps_basic_2fqfq         NUMERIC,
                eps_basic_3fqfq         NUMERIC,
                eps_basic_4fqfq         NUMERIC,
                -- Basic EPS Continuing (ALL periods)
                eps_cont_fq             NUMERIC,
                eps_cont_ltm            NUMERIC,
                eps_cont_fy             NUMERIC,
                eps_cont_1fy            NUMERIC,
                eps_cont_2fy            NUMERIC,
                eps_cont_3fy            NUMERIC,
                eps_cont_4fy            NUMERIC,
                eps_cont_1fqfq          NUMERIC,
                eps_cont_2fqfq          NUMERIC,
                eps_cont_3fqfq          NUMERIC,
                eps_cont_4fqfq          NUMERIC,
                -- EPS Adjusted (ALL periods)
                eps_adj_fq              NUMERIC,
                eps_adj_ltm             NUMERIC,
                eps_adj_fy              NUMERIC,
                eps_adj_1fy             NUMERIC,
                eps_adj_2fy             NUMERIC,
                eps_adj_3fy             NUMERIC,
                eps_adj_4fy             NUMERIC,
                eps_adj_1fqfq           NUMERIC,
                eps_adj_2fqfq           NUMERIC,
                eps_adj_3fqfq           NUMERIC,
                eps_adj_4fqfq           NUMERIC,
                -- Estimates
                eps_norm_est_ntm        NUMERIC,
                eps_norm_est_fy1e       NUMERIC,
                eps_gaap_est_ntm        NUMERIC,
                eps_gaap_est_fy1e       NUMERIC,
                eps_estimate_count      NUMERIC,
                -- Growth Analytics
                eps_growth_yoy          NUMERIC,
                eps_growth_qoq          NUMERIC,
                eps_growth_2y           NUMERIC,
                eps_cagr_3y             NUMERIC,
                eps_cagr_5y             NUMERIC,
                eps_growth_acceleration NUMERIC,
                -- Adjustment Analytics
                eps_adjustment_ratio    NUMERIC,
                eps_adjustment_pct      NUMERIC,
                gaap_vs_norm_est_spread NUMERIC,
                -- Consistency & Quality
                eps_positive_years      INTEGER,
                eps_positive_quarters   INTEGER,
                eps_improvement_years   INTEGER,
                eps_trajectory_score    NUMERIC,
                eps_stability_score     NUMERIC
            )
AS
$$
SELECT "Ticker"                                                              AS ticker,
       -- Net EPS Basic Values
       "Net EPS - Basic (FQ)"                                                AS eps_basic_fq,
       "Net EPS - Basic (LTM)"                                               AS eps_basic_ltm,
       "Net EPS - Basic (FY)"                                                AS eps_basic_fy,
       "Net EPS - Basic (-1FY)"                                              AS eps_basic_1fy,
       "Net EPS - Basic (-2FY)"                                              AS eps_basic_2fy,
       "Net EPS - Basic (-3FY)"                                              AS eps_basic_3fy,
       "Net EPS - Basic (-4FY)"                                              AS eps_basic_4fy,
       "Net EPS - Basic (-5FY)"                                              AS eps_basic_5fy,
       "Net EPS - Basic (-1FQFQ)"                                            AS eps_basic_1fqfq,
       "Net EPS - Basic (-2FQFQ)"                                            AS eps_basic_2fqfq,
       "Net EPS - Basic (-3FQFQ)"                                            AS eps_basic_3fqfq,
       "Net EPS - Basic (-4FQFQ)"                                            AS eps_basic_4fqfq,
       -- Basic EPS Continuing Values
       "Basic EPS - Cont (FQ)"                                               AS eps_cont_fq,
       "Basic EPS - Cont (LTM)"                                              AS eps_cont_ltm,
       "Basic EPS - Cont (FY)"                                               AS eps_cont_fy,
       "Basic EPS - Cont (-1FY)"                                             AS eps_cont_1fy,
       "Basic EPS - Cont (-2FY)"                                             AS eps_cont_2fy,
       "Basic EPS - Cont (-3FY)"                                             AS eps_cont_3fy,
       "Basic EPS - Cont (-4FY)"                                             AS eps_cont_4fy,
       "Basic EPS - Cont (-1FQFQ)"                                           AS eps_cont_1fqfq,
       "Basic EPS - Cont (-2FQFQ)"                                           AS eps_cont_2fqfq,
       "Basic EPS - Cont (-3FQFQ)"                                           AS eps_cont_3fqfq,
       "Basic EPS - Cont (-4FQFQ)"                                           AS eps_cont_4fqfq,
       -- EPS Adjusted Values
       "EPS/Adj. (FQ)"                                                       AS eps_adj_fq,
       "EPS/Adj. (LTM)"                                                      AS eps_adj_ltm,
       "EPS/Adj. (FY)"                                                       AS eps_adj_fy,
       "EPS/Adj. (-1FY)"                                                     AS eps_adj_1fy,
       "EPS/Adj. (-2FY)"                                                     AS eps_adj_2fy,
       "EPS/Adj. (-3FY)"                                                     AS eps_adj_3fy,
       "EPS/Adj. (-4FY)"                                                     AS eps_adj_4fy,
       "EPS/Adj. (-1FQFQ)"                                                   AS eps_adj_1fqfq,
       "EPS/Adj. (-2FQFQ)"                                                   AS eps_adj_2fqfq,
       "EPS/Adj. (-3FQFQ)"                                                   AS eps_adj_3fqfq,
       "EPS/Adj. (-4FQFQ)"                                                   AS eps_adj_4fqfq,
       -- Estimates
       "EPS Norm - Est Avg (NTM)"                                            AS eps_norm_est_ntm,
       "EPS Norm - Est Avg (FY1E)"                                           AS eps_norm_est_fy1e,
       "EPS GAAP - Est Avg (NTM)"                                            AS eps_gaap_est_ntm,
       "EPS GAAP - Est Avg (FY1E)"                                           AS eps_gaap_est_fy1e,
       "EPS Norm - Est # (FY1E)"                                             AS eps_estimate_count,
       -- Growth Analytics
       ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
       NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100                        AS eps_growth_yoy,
       ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
       NULLIF(ABS("Net EPS - Basic (-1FQFQ)"), 0) * 100                      AS eps_growth_qoq,
       ("Net EPS - Basic (FY)" - "Net EPS - Basic (-2FY)") /
       NULLIF(ABS("Net EPS - Basic (-2FY)"), 0) * 100                        AS eps_growth_2y,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                               AS eps_cagr_3y,
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                               AS eps_cagr_5y,
       -- Growth Acceleration (3Y CAGR - 5Y CAGR)
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN ((POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                     (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
           END                                                               AS eps_growth_acceleration,
       -- Adjustment Analytics
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0)                 AS eps_adjustment_ratio,
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
       NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100                         AS eps_adjustment_pct,
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"             AS gaap_vs_norm_est_spread,
       -- Consistency Metrics
       (CASE WHEN "Net EPS - Basic (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER   AS eps_positive_years,
       (CASE WHEN "Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_quarters,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                             AS eps_improvement_years,
       -- Trajectory Score (0-100)
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                     AS eps_trajectory_score,
       -- Stability Score (based on coefficient of variation proxy)
       GREATEST(0, LEAST(100,
                         100 - ABS("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
                               NULLIF(GREATEST(ABS("Net EPS - Basic (FY)"), 0.01), 0) * 20
                   ))                                                        AS eps_stability_score
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 8: GROWTH FEATURES
-- =============================================================================

-- Growth Features (engineer_growth_metrics)
CREATE OR REPLACE FUNCTION calc_growth_features()
    RETURNS TABLE
            (
                ticker                  TEXT,
                revenue_growth_yoy      NUMERIC,
                ebitda_growth_yoy       NUMERIC,
                operating_income_growth NUMERIC,
                fcf_growth              NUMERIC,
                revenue_cagr_5y         NUMERIC,
                forward_revenue_growth  NUMERIC,
                revenue_vs_5y_avg       NUMERIC
            )
AS
$$
SELECT "Ticker"                                                        AS ticker,
       -- Revenue Growth YoY
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100
           END                                                         AS revenue_growth_yoy,
       -- EBITDA Growth YoY
       CASE
           WHEN ABS("EBITDA (-1FY)") > 0
               THEN ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100
           END                                                         AS ebitda_growth_yoy,
       -- Operating Income Growth
       CASE
           WHEN ABS("Operating Income (FY)") > 0
               THEN ("Operating Income (LTM)" - "Operating Income (FY)") /
                    NULLIF(ABS("Operating Income (FY)"), 0) * 100
           END                                                         AS operating_income_growth,
       -- FCF Growth
       CASE
           WHEN ABS("FCF (FY)") > 0
               THEN ("FCF (LTM)" - "FCF (FY)") / NULLIF(ABS("FCF (FY)"), 0) * 100
           END                                                         AS fcf_growth,
       -- Revenue CAGR 5Y
       "Total Revenues/CAGR (5Y FY)"                                   AS revenue_cagr_5y,
       -- Forward Revenue Growth (Estimate)
       "Revenues - Est YoY % (FY1E)"                                   AS forward_revenue_growth,
       -- Revenue vs 5Y Average (NULLIF handles zero division)
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_vs_5y_avg
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Revenue Forecasting Features (engineer_revenue_forecast_features)
CREATE OR REPLACE FUNCTION calc_revenue_forecast_features()
    RETURNS TABLE
            (
                ticker                     TEXT,
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
AS
$$
SELECT "Ticker"                                                                 AS ticker,
       -- Revenue Estimate Spread (High - Low) / Median
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                             AS revenue_est_spread,

       -- Revenue Beat Potential (Current vs Estimate)
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_beat_potential,

       -- Revenue Estimate Revision Trend (use forward growth as proxy)
       "Revenues - Est YoY % (FY1E)"                                            AS revenue_est_revision_trend,

       -- EBITDA Estimate vs Actual
       ("EBITDA (LTM)" - "EBITDA - Est Avg (FY1E)") /
       NULLIF(ABS("EBITDA - Est Avg (FY1E)"), 0) * 100                          AS ebitda_est_vs_actual,

       -- Forward Revenue Multiple (EV / Forward Revenue)
       "Enterprise Value" / NULLIF("Revenues - Est Avg (FY1E)", 0)              AS forward_revenue_multiple,

       -- Revenue Estimate Count (proxy: use EPS estimate count)
       "EPS Norm - Est # (FY1E)"                                                AS revenue_estimate_count,

       -- Revenue Guidance Gap (NTM vs FY1E difference)
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_guidance_gap,

       -- Consensus Revenue Growth Expectation
       ("Revenues - Est Avg (FY1E)" - "Total Revenues (FY)") /
       NULLIF(ABS("Total Revenues (FY)"), 0) * 100                              AS consensus_revenue_growth,

       -- EBIT Estimate Spread
       ("EBIT - Est Med (FY1E)" - "EBIT - Est Med (NTM)") /
       NULLIF(ABS("EBIT - Est Med (NTM)"), 0) * 100                             AS ebit_estimate_spread,

       -- Forward EBITDA Margin (Estimated EBITDA / Estimated Revenue)
       "EBITDA - Est Avg (FY1E)" / NULLIF("Revenues - Est Avg (FY1E)", 0) * 100 AS forward_ebitda_margin,

       -- Revenue Acceleration (Forward growth vs historical growth)
       "Revenues - Est YoY % (FY1E)" - "Total Revenues/CAGR (5Y FY)"            AS revenue_acceleration,

       -- Estimate Confidence Score (narrower spread = higher confidence)
       GREATEST(0, LEAST(100,
                         100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                   NULLIF("Revenues - Est Med (FY1E)", 0) * 100)
                   ))                                                           AS estimate_confidence_score

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 9: DIVIDEND FEATURES
-- =============================================================================

-- Dividend Features (engineer_dividend_reliability_features)
CREATE OR REPLACE FUNCTION calc_dividend_features()
    RETURNS TABLE
            (
                ticker                      TEXT,
                dividend_streak             INTEGER,
                dividend_yield_ltm          NUMERIC,
                dividend_yield_ntm          NUMERIC,
                dividend_payout_ratio       NUMERIC,
                fcf_dividend_coverage       NUMERIC,
                buyback_yield               NUMERIC,
                total_shareholder_yield     NUMERIC,
                dividend_growth_expectation NUMERIC
            )
AS
$$
SELECT "Ticker"                                                                AS ticker,
       "Dividend Streak"::INTEGER                                              AS dividend_streak,
       "Div Yield (LTM)"                                                       AS dividend_yield_ltm,
       "Div Yield (NTM)"                                                       AS dividend_yield_ntm,
       -- Dividend Payout Ratio (NULLIF handles zero division)
       ABS("Common Dividends Paid (LTM)") / NULLIF("Net Income/Adj. (LTM)", 0) AS dividend_payout_ratio,
       -- FCF Dividend Coverage
       CASE
           WHEN ABS("Common Dividends Paid (LTM)") > 0
               THEN "FCF (LTM)" / NULLIF(ABS("Common Dividends Paid (LTM)"), 0)
           END                                                                 AS fcf_dividend_coverage,
       -- Buyback Yield
       "Buyback Yield (LTM)"                                                   AS buyback_yield,
       -- Total Shareholder Yield
       COALESCE("Buyback Yield (LTM)", 0) + COALESCE("Div Yield (LTM)", 0)     AS total_shareholder_yield,
       -- Dividend Growth Expectation
       "Div Yield (NTM)" - "Div Yield (LTM)"                                   AS dividend_growth_expectation
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Comprehensive Dividend Yield Features with ALL historical variants
CREATE OR REPLACE FUNCTION calc_dividend_yield_comprehensive()
    RETURNS TABLE
            (
                ticker                 TEXT,
                -- Current Yields
                div_yield_ind          NUMERIC,
                div_yield_ltm          NUMERIC,
                div_yield_ttm          NUMERIC,
                div_yield_ntm          NUMERIC,
                div_yield_5yavg        NUMERIC,
                -- Historical Indicated Yields (ALL periods)
                div_yield_ind_1fy      NUMERIC,
                div_yield_ind_2fy      NUMERIC,
                div_yield_ind_3fy      NUMERIC,
                div_yield_ind_4fy      NUMERIC,
                div_yield_ind_5fy      NUMERIC,
                -- Dividend Analytics
                dividend_streak        INTEGER,
                dividend_per_share_ltm NUMERIC,
                dividend_amount        NUMERIC,
                common_dividends_ltm   NUMERIC,
                common_dividends_fy    NUMERIC,
                buyback_yield          NUMERIC,
                -- Yield Trends
                div_yield_change_1y    NUMERIC,
                div_yield_change_2y    NUMERIC,
                div_yield_change_3y    NUMERIC,
                div_yield_change_5y    NUMERIC,
                dividend_growth_ntm    NUMERIC
            )
AS
$$
SELECT "Ticker"                                  AS ticker,
       -- Current Yields
       "Div Yield (Ind)"                         AS div_yield_ind,
       "Div Yield (LTM)"                         AS div_yield_ltm,
       "Div Yield (TTM)"                         AS div_yield_ttm,
       "Div Yield (NTM)"                         AS div_yield_ntm,
       "Div Yield (5YAVGLTM)"                    AS div_yield_5yavg,
       -- Historical Indicated Yields
       "Div Yield (-1FYInd)"                     AS div_yield_ind_1fy,
       "Div Yield (-2FYInd)"                     AS div_yield_ind_2fy,
       "Div Yield (-3FYInd)"                     AS div_yield_ind_3fy,
       "Div Yield (-4FYInd)"                     AS div_yield_ind_4fy,
       "Div Yield (-5FYInd)"                     AS div_yield_ind_5fy,
       -- Dividend Analytics
       "Dividend Streak"::INTEGER                AS dividend_streak,
       "Dividend Per Share (LTM)"                AS dividend_per_share_ltm,
       "Dividend Record (Amount)"                AS dividend_amount,
       "Common Dividends Paid (LTM)"             AS common_dividends_ltm,
       "Common Dividends Paid (FY)"              AS common_dividends_fy,
       "Buyback Yield (LTM)"                     AS buyback_yield,
       -- Yield Trends
       "Div Yield (Ind)" - "Div Yield (-1FYInd)" AS div_yield_change_1y,
       "Div Yield (Ind)" - "Div Yield (-2FYInd)" AS div_yield_change_2y,
       "Div Yield (Ind)" - "Div Yield (-3FYInd)" AS div_yield_change_3y,
       "Div Yield (Ind)" - "Div Yield (-5FYInd)" AS div_yield_change_5y,
       "Div Yield (NTM)" - "Div Yield (LTM)"     AS dividend_growth_ntm
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Dividend Timing Features (engineer_dividend_timing_features)
CREATE OR REPLACE FUNCTION calc_dividend_timing()
    RETURNS TABLE
            (
                ticker                   TEXT,
                days_since_ex_date       INTEGER,
                days_to_payment          INTEGER,
                dividend_announced_flag  INTEGER,
                ex_date_approaching_flag INTEGER,
                dividend_frequency_score INTEGER,
                dividend_consistency     NUMERIC,
                recent_dividend_change   NUMERIC,
                dividend_yield_vs_5y_avg NUMERIC
            )
AS
$$
SELECT "Ticker"                                                   AS ticker,
       -- Days Since Ex-Dividend Date
       (CURRENT_DATE - "Dividend Record (Ex Date)")::INTEGER      AS days_since_ex_date,

       -- Days to Next Payment
       ("Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER AS days_to_payment,

       -- Dividend Announced Flag (recent announcement within 30 days)
       CASE
           WHEN (CURRENT_DATE - "Dividend Record (Announce Date)") <= 30
               THEN 1
           ELSE 0
           END                                                    AS dividend_announced_flag,

       -- Ex-Date Approaching Flag (within next 14 days)
       CASE
           WHEN ("Dividend Record (Ex Date)" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                    AS ex_date_approaching_flag,

       -- Dividend Frequency Score (Quarterly=4, Semi-Annual=2, Annual=1, etc.)
       CASE "Dividend Record (Frequency)"
           WHEN 'Quarterly' THEN 4
           WHEN 'Semi-Annual' THEN 2
           WHEN 'Annual' THEN 1
           WHEN 'Monthly' THEN 12
           ELSE 0
           END                                                    AS dividend_frequency_score,

       -- Dividend Consistency (streak / 10, capped at 1.0)
       LEAST(1.0, "Dividend Streak"::NUMERIC / 10.0)              AS dividend_consistency,

       -- Recent Dividend Change (current vs previous year indicated yield)
       CASE
           WHEN "Div Yield (-1FYInd)" > 0
               THEN ("Div Yield (Ind)" - "Div Yield (-1FYInd)") /
                    NULLIF("Div Yield (-1FYInd)", 0) * 100
           END                                                    AS recent_dividend_change,

       -- Dividend Yield vs 5Y Average
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0)      AS dividend_yield_vs_5y_avg

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 10: EMPLOYMENT FEATURES
-- =============================================================================

-- Employment Features (engineer_employee_productivity_features)
CREATE OR REPLACE FUNCTION calc_employment_features()
    RETURNS TABLE
            (
                ticker               TEXT,
                revenue_per_employee NUMERIC,
                profit_per_employee  NUMERIC,
                ebitda_per_employee  NUMERIC,
                assets_per_employee  NUMERIC,
                fte_growth_1y_pct    NUMERIC,
                fte_growth_3y_pct    NUMERIC,
                workforce_stability  NUMERIC
            )
AS
$$
SELECT "Ticker" AS ticker,
       -- Revenue per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Revenues (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS revenue_per_employee,
       -- Profit per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Normalized Net Income (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS profit_per_employee,
       -- EBITDA per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "EBITDA (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS ebitda_per_employee,
       -- Assets per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Assets (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS assets_per_employee,
       -- FTE Growth 1Y
       CASE
           WHEN "Full Time Employees (-1FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                    NULLIF("Full Time Employees (-1FY)", 0) * 100
           END  AS fte_growth_1y_pct,
       -- FTE Growth 3Y
       CASE
           WHEN "Full Time Employees (-3FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-3FY)") /
                    NULLIF("Full Time Employees (-3FY)", 0) * 100
           END  AS fte_growth_3y_pct,
       -- Workforce Stability (vs 5Y avg)
       CASE
           WHEN "Avg Employees (5YAVGFY)" > 0
               THEN "Full Time Employees (FY)" / NULLIF("Avg Employees (5YAVGFY)", 0)
           END  AS workforce_stability
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Employment Dynamics Features (engineer_employment_dynamics_features)
CREATE OR REPLACE FUNCTION calc_employment_dynamics()
    RETURNS TABLE
            (
                ticker                    TEXT,
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
AS
$$
SELECT "Ticker"                                           AS ticker,
       -- FTE Growth 2Y %
       CASE
           WHEN "Full Time Employees (-2FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
                    NULLIF("Full Time Employees (-2FY)", 0) * 100
           END                                            AS fte_growth_2y_pct,

       -- FTE Acceleration (1Y growth vs 3Y CAGR)
       CASE
           WHEN "Full Time Employees (-1FY)" > 0 AND "Full Time Employees (-3FY)" > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) -
                    (POWER("Full Time Employees (FY)" / NULLIF("Full Time Employees (-3FY)", 0), 1.0 / 3.0) - 1)
           END * 100                                      AS fte_acceleration,

       -- Workforce Volatility (std dev of growth rates, simplified)
       ABS(("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
           NULLIF("Full Time Employees (-1FY)", 0) -
           ("Full Time Employees (-1FY)" - "Full Time Employees (-2FY)") /
           NULLIF("Full Time Employees (-2FY)", 0)) * 100 AS workforce_volatility,

       -- Hiring Intensity (FTE growth / Revenue growth)
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) /
                    NULLIF((("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                            NULLIF(ABS("Total Revenues (-1FY)"), 0)), 0)
           END                                            AS hiring_intensity,

       -- Productivity Trend (Revenue per employee growth)
       CASE
           WHEN "Full Time Employees (FY)" > 0 AND "Full Time Employees (-1FY)" > 0
               THEN (("Total Revenues (FY)" / "Full Time Employees (FY)") -
                     ("Total Revenues (-1FY)" / "Full Time Employees (-1FY)")) /
                    NULLIF(ABS("Total Revenues (-1FY)" / "Full Time Employees (-1FY)"), 0) * 100
           END                                            AS productivity_trend,

       -- Headcount vs Revenue Growth Alignment
       (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
        NULLIF("Full Time Employees (-1FY)", 0) * 100) -
       (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
        NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100)    AS headcount_vs_revenue,

       -- Workforce Efficiency Gain (revenue growth > headcount growth)
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

       -- Layoff Risk Flag (declining headcount + declining revenue)
       CASE
           WHEN "Full Time Employees (FY)" < "Full Time Employees (-1FY)"
               AND "Total Revenues (FY)" < "Total Revenues (-1FY)"
               THEN 1
           ELSE 0
           END                                            AS layoff_risk_flag,

       -- Rapid Hiring Flag (headcount growth > 20%)
       CASE
           WHEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0) > 0.20
               THEN 1
           ELSE 0
           END                                            AS rapid_hiring_flag,

       -- Sustainable Growth Flag (revenue growth > headcount growth > 0)
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               AND ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") > 0
               THEN 1
           ELSE 0
           END                                            AS sustainable_growth_flag

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 11: CASH FLOW FEATURES
-- =============================================================================

-- Cash Flow Features (engineer_cash_flow_quality_features, engineer_cashflow_temporal_features)
CREATE OR REPLACE FUNCTION calc_cashflow_features()
    RETURNS TABLE
            (
                ticker                TEXT,
                cfo_to_net_income     NUMERIC,
                fcf_to_net_income     NUMERIC,
                fcf_margin            NUMERIC,
                cfo_growth_yoy        NUMERIC,
                fcf_positive_ratio    NUMERIC,
                acquisition_intensity NUMERIC,
                self_funding_ratio    NUMERIC
            )
AS
$$
SELECT "Ticker"                                               AS ticker,
       -- CFO to Net Income (Accruals Quality) - NULLIF handles zero division
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS cfo_to_net_income,
       -- FCF to Net Income
       "FCF (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS fcf_to_net_income,
       -- FCF Margin
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0)        AS fcf_margin,
       -- CFO Growth YoY
       ("CFO (LTM)" - "CFO (-1FY)") / NULLIF("CFO (-1FY)", 0) AS cfo_growth_yoy,
       -- FCF Positive Quarters Ratio (from quarterly data)
       (CASE WHEN "FCF (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END) / 5.0 AS fcf_positive_ratio,
       -- Acquisition Intensity (4Q sum)
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))         AS acquisition_intensity,
       -- Self Funding Ratio (CFO / CFI)
       CASE
           WHEN ABS("CFI (LTM)") > 0
               THEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0)
           END                                                AS self_funding_ratio
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 11A: ENHANCED CASH FLOW QUALITY FEATURES (NEW)
-- =============================================================================

-- Enhanced Cash Flow Quality Features
CREATE OR REPLACE FUNCTION calc_enhanced_cashflow_features()
    RETURNS TABLE
            (
                ticker                  TEXT,
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
                cash_flow_quality_score NUMERIC
            )
AS
$$
SELECT "Ticker"                                                          AS ticker,
       -- FCF Positive Years (count over 5 years)
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER           AS fcf_positive_years,

       -- FCF Always Positive Flag (all 5 years positive)
       CASE
           WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
               AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0
               THEN 1
           ELSE 0
           END                                                           AS fcf_always_positive,

       -- CapEx vs 5Y Average (investment consistency)
       ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0)
                                                                         AS capex_vs_5y_avg,

       -- Underinvestment Flag (CapEx < 70% of historical average)
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) < 0.7
               THEN 1
           ELSE 0
           END                                                           AS underinvestment_flag,

       -- CFO Share of Total Cash Flow
       ABS("CFO (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfo_share_of_cf,

       -- CFI Share of Total Cash Flow
       ABS("CFI (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfi_share_of_cf,

       -- CFF Share of Total Cash Flow
       ABS("CFF (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cff_share_of_cf,

       -- Self-Funding Flag (CFO covers CFI needs)
       CASE
           WHEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0) > 1
               THEN 1
           ELSE 0
           END                                                           AS self_funding_flag,

       -- Acquisition to FCF Ratio (M&A sustainability)
       (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
       NULLIF(ABS("FCF (LTM)"), 0)                                       AS acquisition_to_fcf,

       -- Sustainable M&A Flag (acquisitions < 50% of FCF)
       CASE
           WHEN (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
                NULLIF(ABS("FCF (LTM)"), 0) < 0.5
               THEN 1
           ELSE 0
           END                                                           AS sustainable_ma_flag,

       -- FCF 4Q Improvement (most recent vs oldest quarter)
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0)    AS fcf_4q_improvement,

       -- Composite Cash Flow Quality Score (0-100)
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC           AS cash_flow_quality_score

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Cashflow Temporal Features (engineer_cashflow_temporal_features)
CREATE OR REPLACE FUNCTION calc_cashflow_temporal_features()
    RETURNS TABLE
            (
                ticker                TEXT,
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
AS
$$
SELECT "Ticker"                                                             AS ticker,
       -- CFO Quarterly Trend (FQ vs -4FQFQ YoY)
       ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100 AS cfo_quarterly_trend,

       -- CFO YoY Quarterly Growth
       CASE
           WHEN ABS("CFO (-4FQFQ)") > 0
               THEN ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100
           END                                                              AS cfo_yoy_quarterly,

       -- CFI Quarterly Trend
       ("CFI (FQ)" - "CFI (-4FQFQ)") / NULLIF(ABS("CFI (-4FQFQ)"), 0) * 100 AS cfi_quarterly_trend,

       -- CFF Quarterly Trend
       ("CFF (FQ)" - "CFF (-4FQFQ)") / NULLIF(ABS("CFF (-4FQFQ)"), 0) * 100 AS cff_quarterly_trend,

       -- FCF Quarterly Trend
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0) * 100 AS fcf_quarterly_trend,

       -- CFO Positive Quarters (count of last 5)
       (CASE WHEN "CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS cfo_positive_quarters,

       -- CFI Negative Quarters (normal for investing companies)
       (CASE WHEN "CFI (FQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-1FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-2FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-3FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-4FQFQ)" < 0 THEN 1 ELSE 0 END)::INTEGER            AS cfi_negative_quarters,

       -- CFF Pattern Score (positive = raising capital, negative = returning capital)
       CASE
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") > 0
               THEN -1 -- Capital raising (potentially dilutive)
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") < 0
               THEN 1 -- Capital return (buybacks/dividends)
           ELSE 0
           END::NUMERIC                                                     AS cff_pattern_score,

       -- Cash Burn Rate (negative FCF / Cash, monthly)
       CASE
           WHEN "FCF (LTM)" < 0
               THEN ABS("FCF (LTM)") / NULLIF("Cash And Equivalents (FQ)", 0) / 12.0
           ELSE 0
           END                                                              AS cash_burn_rate,

       -- CF Volatility Score (std dev proxy across quarters)
       (ABS("CFO (FQ)" - "CFO (-1FQFQ)") + ABS("CFO (-1FQFQ)" - "CFO (-2FQFQ)") +
        ABS("CFO (-2FQFQ)" - "CFO (-3FQFQ)") + ABS("CFO (-3FQFQ)" - "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (FQ)" + "CFO (-1FQFQ)" + "CFO (-2FQFQ)" +
                  "CFO (-3FQFQ)" + "CFO (-4FQFQ)") / 5.0, 0)                AS cf_volatility_score,

       -- Operating CF Momentum (recent 2Q vs older 2Q)
       (("CFO (FQ)" + "CFO (-1FQFQ)") - ("CFO (-3FQFQ)" + "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (-3FQFQ)" + "CFO (-4FQFQ)"), 0) * 100                AS operating_cf_momentum,

       -- Financing Dependency (CFF / CFO, higher = more dependent)
       ABS("CFF (LTM)") / NULLIF(ABS("CFO (LTM)"), 0)                       AS financing_dependency

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 12: TEMPORAL FEATURES
-- =============================================================================

-- Temporal Features (engineer_temporal_features, engineer_fiscal_calendar_features)
CREATE OR REPLACE FUNCTION calc_temporal_features()
    RETURNS TABLE
            (
                ticker                  TEXT,
                fiscal_quarter          INTEGER,
                fiscal_month            INTEGER,
                fiscal_year             INTEGER,
                days_to_earnings        INTEGER,
                earnings_report_recency INTEGER,
                reporting_lag           NUMERIC,
                fiscal_year_progress    NUMERIC
            )
AS
$$
SELECT "Ticker"                                        AS ticker,
       "Fiscal Quarter"                                AS fiscal_quarter,
       "Fiscal Month"                                  AS fiscal_month,
       "Fiscal Year"                                   AS fiscal_year,
       -- Direct date subtraction returns INTEGER (no EXTRACT needed)
       ("Next Earnings" - CURRENT_DATE)                AS days_to_earnings,
       (CURRENT_DATE - "Income Statement Report Date") AS earnings_report_recency,
       "Reporting Lag"                                 AS reporting_lag,
       "Fiscal Month" / 12.0                           AS fiscal_year_progress
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Extended Fiscal Calendar Features (engineer_fiscal_calendar_features)
CREATE OR REPLACE FUNCTION calc_fiscal_calendar_features()
    RETURNS TABLE
            (
                ticker                    TEXT,
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
AS
$$
SELECT "Ticker"                                                 AS ticker,
       -- Days Since Last Financial Report
       (CURRENT_DATE - "Income Statement Report Date")::INTEGER AS days_since_last_report,

       -- Days to Fiscal Year End
       ("FY End Date" - CURRENT_DATE)::INTEGER                  AS days_to_fy_end,

       -- Is Quarter End Month (current month is 3, 6, 9, or 12)
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12)
               THEN 1
           ELSE 0
           END                                                  AS is_quarter_end_month,

       -- Is Fiscal Year End Month
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM "FY End Date")
               THEN 1
           ELSE 0
           END                                                  AS is_fy_end_month,

       -- Earnings Season Flag (Jan/Feb, Apr/May, Jul/Aug, Oct/Nov)
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 2, 4, 5, 7, 8, 10, 11)
               THEN 1
           ELSE 0
           END                                                  AS earnings_season_flag,

       -- Pre-Earnings Window (within 14 days before earnings)
       CASE
           WHEN ("Next Earnings" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                  AS pre_earnings_window,

       -- Post-Earnings Window (within 7 days after report)
       CASE
           WHEN (CURRENT_DATE - "Income Statement Report Date") BETWEEN 0 AND 7
               THEN 1
           ELSE 0
           END                                                  AS post_earnings_window,

       -- Reporting Freshness Score (100 = just reported, decays over 90 days)
       GREATEST(0, LEAST(100,
                         100 - ((CURRENT_DATE - "Income Statement Report Date")::NUMERIC / 90.0 * 100)
                   ))                                           AS reporting_freshness_score,

       -- Fiscal Quarter Progress (0-1 based on fiscal month)
       CASE
           WHEN "Fiscal Month" IS NOT NULL
               THEN (("Fiscal Month" - 1) % 3 + 1) / 3.0
           END                                                  AS fiscal_quarter_progress

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 12A: COMPOSITE QUALITY SCORES (NEW)
-- =============================================================================

-- Composite Quality Scores (engineer_composite_scores)
CREATE OR REPLACE FUNCTION calc_composite_scores()
    RETURNS TABLE
            (
                ticker                 TEXT,
                piotroski_f_score      INTEGER,
                eps_trajectory_score   NUMERIC,
                dilution_score         NUMERIC,
                quality_momentum_score NUMERIC
            )
AS
$$
SELECT "Ticker"          AS ticker,
       -- Piotroski F-Score (9-point fundamental strength score)
       (
           -- F1: Positive ROA
           CASE WHEN "Return on Assets (ROA) % (LTM)" > 0 THEN 1 ELSE 0 END +
               -- F2: Positive CFO
           CASE WHEN "CFO (LTM)" > 0 THEN 1 ELSE 0 END +
               -- F3: ROA Improvement (FY vs -1FY)
           CASE WHEN "Return on Assets (ROA) % (LTM)" > "Return on Assets (ROA) % (FY)" THEN 1 ELSE 0 END +
               -- F4: Accrual Quality (CFO > Net Income)
           CASE WHEN "CFO (LTM)" > "Net Income - (IS) (LTM)" THEN 1 ELSE 0 END +
               -- F5: Leverage Decrease
           CASE
               WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) <
                    "Total Debt (FY)" / NULLIF("Total Equity (FY)", 0) THEN 1
               ELSE 0 END +
               -- F6: Liquidity Improvement
           CASE WHEN "Current Ratio (LTM)" > "Current Ratio (FY)" THEN 1 ELSE 0 END +
               -- F7: No Dilution (shares unchanged or decreased)
           CASE WHEN "Shrs Out" <= "Shrs Out (-1FY)" THEN 1 ELSE 0 END +
               -- F8: Gross Margin Improvement
           CASE WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" THEN 1 ELSE 0 END +
               -- F9: Asset Turnover Improvement
           CASE WHEN "Asset Turnover (LTM)" > "Asset Turnover (FY)" THEN 1 ELSE 0 END
           )::INTEGER    AS piotroski_f_score,

       -- EPS Trajectory Score (% of improving years, scaled to 0-100)
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100 AS eps_trajectory_score,

       -- Dilution Score (100 = buyback, 0 = heavy dilution)
       GREATEST(0, LEAST(100,
                         50 - (("Shrs Out" - "Shrs Out (-1FY)") / NULLIF("Shrs Out (-1FY)", 0)) * 100
                   ))    AS dilution_score,

       -- Quality-Momentum Composite (combines quality factors with momentum)
       (
           -- Quality component (40% weight)
           ((CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
             CASE WHEN "Return On Equity % (LTM)" > 15 THEN 25 ELSE 0 END +
             CASE WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) < 1 THEN 25 ELSE 0 END +
             CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END) * 0.40) +
               -- Momentum component (30% weight)
           (LEAST(100, GREATEST(0,
                                (("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100 + 50))) * 0.30) +
               -- Growth component (30% weight)
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

FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- Comprehensive Cash Flow Features with ALL temporal variants
CREATE OR REPLACE FUNCTION calc_cashflow_comprehensive()
    RETURNS TABLE
            (
                ticker                   TEXT,
                -- CFO (ALL periods)
                cfo_fq                   NUMERIC,
                cfo_ltm                  NUMERIC,
                cfo_fy                   NUMERIC,
                cfo_1fy                  NUMERIC,
                cfo_2fy                  NUMERIC,
                cfo_3fy                  NUMERIC,
                cfo_4fy                  NUMERIC,
                cfo_1fqfq                NUMERIC,
                cfo_2fqfq                NUMERIC,
                cfo_3fqfq                NUMERIC,
                cfo_4fqfq                NUMERIC,
                -- CFI (ALL periods)
                cfi_fq                   NUMERIC,
                cfi_ltm                  NUMERIC,
                cfi_fy                   NUMERIC,
                cfi_1fy                  NUMERIC,
                cfi_2fy                  NUMERIC,
                cfi_3fy                  NUMERIC,
                cfi_4fy                  NUMERIC,
                cfi_1fqfq                NUMERIC,
                cfi_2fqfq                NUMERIC,
                cfi_3fqfq                NUMERIC,
                cfi_4fqfq                NUMERIC,
                -- CFF (ALL periods)
                cff_fq                   NUMERIC,
                cff_ltm                  NUMERIC,
                cff_fy                   NUMERIC,
                cff_1fy                  NUMERIC,
                cff_2fy                  NUMERIC,
                cff_3fy                  NUMERIC,
                cff_4fy                  NUMERIC,
                cff_1fqfq                NUMERIC,
                cff_2fqfq                NUMERIC,
                cff_3fqfq                NUMERIC,
                cff_4fqfq                NUMERIC,
                -- FCF (ALL periods)
                fcf_fq                   NUMERIC,
                fcf_ltm                  NUMERIC,
                fcf_fy                   NUMERIC,
                fcf_1fy                  NUMERIC,
                fcf_2fy                  NUMERIC,
                fcf_3fy                  NUMERIC,
                fcf_4fy                  NUMERIC,
                fcf_1fqfq                NUMERIC,
                fcf_2fqfq                NUMERIC,
                fcf_3fqfq                NUMERIC,
                fcf_4fqfq                NUMERIC,
                fcf_5yavg                NUMERIC,
                -- Cash Acquisitions (ALL periods)
                acquisitions_fq          NUMERIC,
                acquisitions_ltm         NUMERIC,
                acquisitions_fy          NUMERIC,
                acquisitions_1fy         NUMERIC,
                acquisitions_1fqfq       NUMERIC,
                acquisitions_2fqfq       NUMERIC,
                acquisitions_3fqfq       NUMERIC,
                acquisitions_4fqfq       NUMERIC,
                acquisitions_5yavg       NUMERIC,
                -- CapEx (ALL periods)
                capex_fq                 NUMERIC,
                capex_ltm                NUMERIC,
                capex_fy                 NUMERIC,
                capex_1fy                NUMERIC,
                capex_5yavg              NUMERIC,
                -- Growth Trends
                cfo_growth_yoy           NUMERIC,
                cfo_growth_qoq           NUMERIC,
                fcf_growth_yoy           NUMERIC,
                fcf_growth_qoq           NUMERIC,
                cfo_cagr_3y              NUMERIC,
                fcf_cagr_3y              NUMERIC,
                -- Quality Metrics
                cfo_to_net_income        NUMERIC,
                fcf_to_net_income        NUMERIC,
                fcf_margin               NUMERIC,
                fcf_yield                NUMERIC,
                -- Consistency Metrics
                cfo_positive_years       INTEGER,
                cfo_positive_quarters    INTEGER,
                fcf_positive_years       INTEGER,
                fcf_positive_quarters    INTEGER,
                cfi_negative_years       INTEGER,
                -- Pattern Analysis
                cff_capital_return_flag  INTEGER,
                self_funding_ratio       NUMERIC,
                acquisition_intensity_4q NUMERIC,
                capex_vs_5y_avg          NUMERIC,
                cash_flow_quality_score  NUMERIC
            )
AS
$$
SELECT "Ticker"                                                             AS ticker,
       -- CFO Values
       "CFO (FQ)"                                                           AS cfo_fq,
       "CFO (LTM)"                                                          AS cfo_ltm,
       "CFO (FY)"                                                           AS cfo_fy,
       "CFO (-1FY)"                                                         AS cfo_1fy,
       "CFO (-2FY)"                                                         AS cfo_2fy,
       "CFO (-3FY)"                                                         AS cfo_3fy,
       "CFO (-4FY)"                                                         AS cfo_4fy,
       "CFO (-1FQFQ)"                                                       AS cfo_1fqfq,
       "CFO (-2FQFQ)"                                                       AS cfo_2fqfq,
       "CFO (-3FQFQ)"                                                       AS cfo_3fqfq,
       "CFO (-4FQFQ)"                                                       AS cfo_4fqfq,
       -- CFI Values
       "CFI (FQ)"                                                           AS cfi_fq,
       "CFI (LTM)"                                                          AS cfi_ltm,
       "CFI (FY)"                                                           AS cfi_fy,
       "CFI (-1FY)"                                                         AS cfi_1fy,
       "CFI (-2FY)"                                                         AS cfi_2fy,
       "CFI (-3FY)"                                                         AS cfi_3fy,
       "CFI (-4FY)"                                                         AS cfi_4fy,
       "CFI (-1FQFQ)"                                                       AS cfi_1fqfq,
       "CFI (-2FQFQ)"                                                       AS cfi_2fqfq,
       "CFI (-3FQFQ)"                                                       AS cfi_3fqfq,
       "CFI (-4FQFQ)"                                                       AS cfi_4fqfq,
       -- CFF Values
       "CFF (FQ)"                                                           AS cff_fq,
       "CFF (LTM)"                                                          AS cff_ltm,
       "CFF (FY)"                                                           AS cff_fy,
       "CFF (-1FY)"                                                         AS cff_1fy,
       "CFF (-2FY)"                                                         AS cff_2fy,
       "CFF (-3FY)"                                                         AS cff_3fy,
       "CFF (-4FY)"                                                         AS cff_4fy,
       "CFF (-1FQFQ)"                                                       AS cff_1fqfq,
       "CFF (-2FQFQ)"                                                       AS cff_2fqfq,
       "CFF (-3FQFQ)"                                                       AS cff_3fqfq,
       "CFF (-4FQFQ)"                                                       AS cff_4fqfq,
       -- FCF Values
       "FCF (FQ)"                                                           AS fcf_fq,
       "FCF (LTM)"                                                          AS fcf_ltm,
       "FCF (FY)"                                                           AS fcf_fy,
       "FCF (-1FY)"                                                         AS fcf_1fy,
       "FCF (-2FY)"                                                         AS fcf_2fy,
       "FCF (-3FY)"                                                         AS fcf_3fy,
       "FCF (-4FY)"                                                         AS fcf_4fy,
       "FCF (-1FQFQ)"                                                       AS fcf_1fqfq,
       "FCF (-2FQFQ)"                                                       AS fcf_2fqfq,
       "FCF (-3FQFQ)"                                                       AS fcf_3fqfq,
       "FCF (-4FQFQ)"                                                       AS fcf_4fqfq,
       "FCF (5YAVGFQ)"                                                      AS fcf_5yavg,
       -- Cash Acquisitions
       "Cash Acquisitions (FQ)"                                             AS acquisitions_fq,
       "Cash Acquisitions (LTM)"                                            AS acquisitions_ltm,
       "Cash Acquisitions (FY)"                                             AS acquisitions_fy,
       "Cash Acquisitions (-1FY)"                                           AS acquisitions_1fy,
       "Cash Acquisitions (-1FQFQ)"                                         AS acquisitions_1fqfq,
       "Cash Acquisitions (-2FQFQ)"                                         AS acquisitions_2fqfq,
       "Cash Acquisitions (-3FQFQ)"                                         AS acquisitions_3fqfq,
       "Cash Acquisitions (-4FQFQ)"                                         AS acquisitions_4fqfq,
       "Cash Acquisitions (5YAVGFQ)"                                        AS acquisitions_5yavg,
       -- CapEx
       "Capital Expenditure (FQ)"                                           AS capex_fq,
       "Capital Expenditure (LTM)"                                          AS capex_ltm,
       "Capital Expenditure (FY)"                                           AS capex_fy,
       "Capital Expenditure (-1FY)"                                         AS capex_1fy,
       "Capital Expenditure (5YAVGFQ)"                                      AS capex_5yavg,
       -- Growth Trends
       ("CFO (FY)" - "CFO (-1FY)") / NULLIF(ABS("CFO (-1FY)"), 0) * 100     AS cfo_growth_yoy,
       ("CFO (FQ)" - "CFO (-1FQFQ)") / NULLIF(ABS("CFO (-1FQFQ)"), 0) * 100 AS cfo_growth_qoq,
       ("FCF (FY)" - "FCF (-1FY)") / NULLIF(ABS("FCF (-1FY)"), 0) * 100     AS fcf_growth_yoy,
       ("FCF (FQ)" - "FCF (-1FQFQ)") / NULLIF(ABS("FCF (-1FQFQ)"), 0) * 100 AS fcf_growth_qoq,
       CASE
           WHEN "CFO (-3FY)" > 0 AND "CFO (FY)" > 0
               THEN (POWER("CFO (FY)" / NULLIF("CFO (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                              AS cfo_cagr_3y,
       CASE
           WHEN "FCF (-3FY)" > 0 AND "FCF (FY)" > 0
               THEN (POWER("FCF (FY)" / NULLIF("FCF (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                              AS fcf_cagr_3y,
       -- Quality Metrics
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)                   AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)                   AS fcf_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0)                      AS fcf_margin,
       "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                          AS fcf_yield,
       -- Consistency Metrics
       (CASE WHEN "CFO (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER              AS cfo_positive_years,
       (CASE WHEN "CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS cfo_positive_quarters,
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER              AS fcf_positive_years,
       (CASE WHEN "FCF (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS fcf_positive_quarters,
       (CASE WHEN "CFI (FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-1FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-2FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-3FY)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-4FY)" < 0 THEN 1 ELSE 0 END)::INTEGER              AS cfi_negative_years,
       -- Pattern Analysis
       CASE
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") < 0 THEN 1
           ELSE 0
           END                                                              AS cff_capital_return_flag,
       "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0)                            AS self_funding_ratio,
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) + ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))                       AS acquisition_intensity_4q,
       ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0)
                                                                            AS capex_vs_5y_avg,
       -- Cash Flow Quality Score (0-100)
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0 AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0
                THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC              AS cash_flow_quality_score
FROM postgres.public.equities;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 13: COMPREHENSIVE FEATURE VIEW (ENHANCED)
-- =============================================================================

-- First, drop the existing view
DROP VIEW IF EXISTS vw_feature_registry CASCADE;

-- Then recreate it with ALL features from the entire registry
CREATE VIEW vw_feature_registry AS
SELECT e."Ticker",
       e."ISIN",
       e."Name",
       e."Region",
       e."Country",
       e."Sector",
       e."Industry",
       e."Last Price",
       e."Market Cap",
       e."Enterprise Value",

       -- ==========================================================================
       -- VALUATION FEATURES (from calc_valuation_features)
       -- ==========================================================================
       e."P/E (LTM)"                                                                                AS p_e_ratio,
       e."P/B (LTM)"                                                                                AS p_b_ratio,
       e."EV/EBITDA (LTM)"                                                                          AS ev_ebitda_ratio,
       e."EV/Sales (LTM)"                                                                           AS ev_sales_ratio,
       e."Div Yield (LTM)"                                                                          AS dividend_yield,
       CASE
           WHEN e."Total Revenues/CAGR (5Y FY)" > 0
               THEN e."P/E (LTM)" / NULLIF(e."Total Revenues/CAGR (5Y FY)", 0)
           END                                                                                      AS peg_ratio,

       -- ==========================================================================
       -- VALUATION TIMESERIES FEATURES (from calc_valuation_timeseries_features)
       -- ==========================================================================
       (e."EV/Sales (LTM)" - e."EV/Sales (-1FYLTM)") /
       NULLIF(e."EV/Sales (-1FYLTM)", 0)                                                            AS ev_sales_trend_1y,
       (e."EV/EBITDA (LTM)" - e."EV/EBITDA (-1FYLTM)") /
       NULLIF(e."EV/EBITDA (-1FYLTM)", 0)                                                           AS ev_ebitda_momentum,
       (e."P/E (LTM)" - e."P/E (-1FYLTM)") / NULLIF(e."P/E (-1FYLTM)", 0)                           AS p_e_momentum_yoy,
       (e."P/E (LTM)" - e."P/E (-1FQLTM)") / NULLIF(e."P/E (-1FQLTM)", 0)                           AS p_e_momentum_qoq,
       (e."EV/Sales (LTM)" - e."EV/Sales (3YAVGLTM)") /
       NULLIF(e."EV/Sales (3YAVGLTM)", 0)                                                           AS ev_sales_vs_3y_avg,
       (e."EV/EBITDA (LTM)" - e."EV/EBITDA (3YAVGLTM)") /
       NULLIF(e."EV/EBITDA (3YAVGLTM)", 0)                                                          AS ev_ebitda_vs_3y_avg,
       (e."P/E (LTM)" - e."P/E (3YAVGLTM)") / NULLIF(e."P/E (3YAVGLTM)", 0)                         AS p_e_vs_3y_avg,
       (e."EV/Sales (NTM)" - e."EV/Sales (LTM)") /
       NULLIF(e."EV/Sales (LTM)", 0)                                                                AS ev_sales_forward_discount,
       (e."EV/EBITDA (NTM)" - e."EV/EBITDA (LTM)") /
       NULLIF(e."EV/EBITDA (LTM)", 0)                                                               AS ev_ebitda_forward_discount,
       (e."P/E (EST FY1)" - e."P/E (LTM)") / NULLIF(e."P/E (LTM)", 0)                               AS p_e_forward_discount,
       e."P/B (LTM)" / NULLIF(e."P/B (5YAVG)", 0)                                                   AS p_b_vs_5y_avg,

       -- Extended Valuation Timeseries
       (e."EV/Sales (LTM)" - e."EV/Sales (-1FQLTM)") /
       NULLIF(e."EV/Sales (-1FQLTM)", 0)                                                            AS ev_sales_qoq_1q,
       (e."P/E (LTM)" - e."P/E (5YAVGLTM)") / NULLIF(e."P/E (5YAVGLTM)", 0)                         AS p_e_vs_5y_avg,
       (("P/E (LTM)" / NULLIF(e."P/E (3YAVGLTM)", 0)) +
        (e."EV/EBITDA (LTM)" / NULLIF(e."EV/EBITDA (3YAVGLTM)", 0))) / 2.0 -
       1.0                                                                                          AS valuation_compression,
       (e."P/E (EST FY1)" - e."P/E (LTM)") / NULLIF(ABS(e."P/E (LTM)"), 0) *
       100                                                                                          AS forward_pe_premium,
       (e."P/B (LTM)" - e."P/B (-1FY)") / NULLIF(e."P/B (-1FY)", 0)                                 AS p_b_momentum_yoy,

       -- ==========================================================================
       -- MOMENTUM FEATURES (from calc_momentum_features)
       -- ==========================================================================
       (e."Last Price" - e."Price (1M Ago)") / NULLIF(e."Price (1M Ago)", 0) *
       100                                                                                          AS price_momentum_1m,
       (e."Last Price" - e."Price (3M Ago)") / NULLIF(e."Price (3M Ago)", 0) *
       100                                                                                          AS price_momentum_3m,
       (e."Last Price" - e."Price (6M Ago)") / NULLIF(e."Price (6M Ago)", 0) *
       100                                                                                          AS price_momentum_6m,
       (e."Last Price" - e."Price (1Y Ago)") / NULLIF(e."Price (1Y Ago)", 0) *
       100                                                                                          AS price_momentum_1y,
       (e."Last Price" - e."Price (5D Ago)") / NULLIF(e."Price (5D Ago)", 0) *
       100                                                                                          AS price_momentum_5d,
       CASE
           WHEN e."EMA (20D)" > e."EMA (50D)" THEN 1
           WHEN e."EMA (20D)" < e."EMA (50D)" THEN -1
           ELSE 0 END                                                                               AS ema_crossover_20_50,
       CASE
           WHEN e."EMA (50D)" > e."EMA (250D)" THEN 1
           WHEN e."EMA (50D)" < e."EMA (250D)" THEN -1
           ELSE 0 END                                                                               AS ema_crossover_50_250,
       (e."Last Price" - e."EMA (20D)") / NULLIF(e."EMA (20D)", 0)                                  AS price_vs_ema_20d,
       (e."Last Price" - e."EMA (250D)") / NULLIF(e."EMA (250D)", 0)                                AS price_vs_ema_250d,
       (e."52W High/Adj" - e."Last Price") / NULLIF(e."52W High/Adj", 0)                            AS pct_off_52w_high,
       (e."Last Price" - e."52W Low/Adj") / NULLIF(e."52W Low/Adj", 0)                              AS pct_above_52w_low,
       LEAST(1, GREATEST(0, (e."Last Price" - e."52W Low/Adj") /
                            NULLIF(e."52W High/Adj" - e."52W Low/Adj", 0)))                         AS range_52w_position,
       e."Beta (1Y)" - e."Beta (5Y)"                                                                AS beta_momentum,
       e."Volatility (1M)" / NULLIF(e."Volatility (1Y)", 0)                                         AS volatility_regime,

       -- ==========================================================================
       -- TECHNICAL ANALYSIS FEATURES (from calc_technical_analysis_features)
       -- ==========================================================================
       (e."EMA (20D)" - e."EMA (50D)") / NULLIF(e."EMA (50D)", 0)                                   AS ema_slope_20d,
       CASE
           WHEN e."EMA (20D)" > e."EMA (50D)" AND e."EMA (50D)" > e."EMA (100D)"
               AND e."EMA (100D)" > e."EMA (250D)" THEN 1
           WHEN e."EMA (20D)" < e."EMA (50D)" AND e."EMA (50D)" < e."EMA (100D)"
               AND e."EMA (100D)" < e."EMA (250D)" THEN -1
           ELSE 0 END                                                                               AS ema_trend_consistency,
       (e."Last Price" - e."EMA (100D)") / NULLIF(e."EMA (100D)", 0) * 100                          AS price_vs_ema_100d,
       CASE
           WHEN (e."52W High/Adj" - e."Last Price") / NULLIF(e."52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0 END                                                                               AS near_52w_high_flag,
       CASE
           WHEN (e."Last Price" - e."52W Low/Adj") / NULLIF(e."52W Low/Adj", 0) <= 0.05
               THEN 1
           ELSE 0 END                                                                               AS near_52w_low_flag,
       e."Rel. Volume" * e."Price Chg. % (1M)"                                                      AS volume_momentum_score,
       CASE
           WHEN e."EMA (20D)" > e."EMA (50D)"
               AND (e."52W High/Adj" - e."Last Price") / NULLIF(e."52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0 END                                                                               AS breakout_signal,
       CASE WHEN e."Rel. Volume" > 1.5 THEN 1 ELSE 0 END                                            AS high_volume_flag,
       CASE WHEN e."Rel. Volume" < 0.5 THEN 1 ELSE 0 END                                            AS low_volume_flag,
       e."Volatility (1Y)" - e."Volatility (1M)"                                                    AS volatility_compression,
       e."Volatility (3M)" - e."Volatility (6M)"                                                    AS volatility_term_structure,

       -- ==========================================================================
       -- PROFITABILITY FEATURES (from calc_profitability_features)
       -- ==========================================================================
       e."Return On Equity % (LTM)"                                                                 AS roe,
       e."Return on Assets (ROA) % (LTM)"                                                           AS roa,
       e."Gross Profit Margin % (LTM)"                                                              AS gross_margin_pct,
       e."Operating Income (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                       AS operating_margin_pct,
       e."Net Income Margin % (LTM)"                                                                AS net_margin_pct,
       e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0) * 100                                 AS ebitda_margin_pct,
       e."Net Income - (IS) (LTM)" / NULLIF(e."Total Equity (LTM)" + e."Total Debt (LTM)", 0) * 100 AS roic,
       e."R&D Expenses (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)                                 AS rnd_intensity,
       e."Total Assets (LTM)" / NULLIF(e."Total Equity (LTM)", 0)                                   AS equity_multiplier,

       -- ==========================================================================
       -- MARGIN TRENDS (from calc_margin_trends)
       -- ==========================================================================
       (e."Gross Profit Margin % (LTM)" - e."Gross Profit Margin % (FY)")                           AS gross_margin_trend_yoy,
       ((e."Operating Income (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)) -
        (e."Operating Income (FY)" / NULLIF(e."Total Revenues (FY)", 0))) *
       100                                                                                          AS operating_margin_trend,
       (e."Net Income Margin % (LTM)" - e."Net Income Margin % (FY)")                               AS net_margin_trend_yoy,
       ((e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)) -
        (e."EBITDA (FY)" / NULLIF(e."Total Revenues (FY)", 0))) *
       100                                                                                          AS ebitda_margin_trend,
       CASE
           WHEN e."Gross Profit Margin % (LTM)" > e."Gross Profit Margin % (FY)"
               AND e."Net Income Margin % (LTM)" > e."Net Income Margin % (FY)"
               AND (e."EBITDA (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)) >
                   (e."EBITDA (FY)" / NULLIF(e."Total Revenues (FY)", 0))
               THEN 1
           ELSE 0 END                                                                               AS margin_expansion_flag,
       GREATEST(0, LEAST(100,
                         100 - (ABS(e."Gross Profit Margin % (LTM)" - e."Gross Profit Margin % (FY)") +
                                ABS(e."Net Income Margin % (LTM)" - e."Net Income Margin % (FY)")) / 2
                   ))                                                                               AS margin_stability_score,

       -- ==========================================================================
       -- QUALITY & RISK FEATURES (from calc_quality_features)
       -- ==========================================================================
       CASE WHEN e."Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END                            AS has_goodwill_impairment,
       CASE WHEN e."Asset Writedown (LTM)" <> 0 THEN 1 ELSE 0 END                                   AS has_asset_writedown,
       CASE WHEN e."Restructuring Charges (LTM)" <> 0 THEN 1 ELSE 0 END                             AS has_restructuring,
       e."Goodwill (LTM)" / NULLIF(e."Total Assets (LTM)", 0) * 100                                 AS goodwill_to_assets_pct,
       e."Gross Intangible Assets (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                        AS intangible_intensity,
       (ABS(e."Impairment of Goodwill (LTM)") + ABS(e."Asset Writedown (LTM)") +
        ABS(e."Restructuring Charges (LTM)")) /
       NULLIF(ABS(e."EBITDA (LTM)"), 0)                                                             AS exceptional_items_to_ebitda,
       e."Altman Z-Score (LTM)"                                                                     AS altman_z_score,
       e."Altman Z-Score (FY)" - e."Altman Z-Score (LTM)"                                           AS altman_z_trend,
       e."Current Ratio (LTM)"                                                                      AS current_ratio,
       (e."Total Current Assets (LTM)" - e."Inventory (LTM)") /
       NULLIF(e."Total Current Liabilities (LTM)", 0)                                               AS quick_ratio,

       -- ==========================================================================
       -- FINANCIAL DISTRESS FEATURES (from calc_financial_distress_features)
       -- ==========================================================================
       GREATEST(0, LEAST(100,
                         ((e."Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100)))          AS distress_risk_score,
       CASE
           WHEN e."Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN e."Current Ratio (LTM)" < 1.5 THEN 15.0
           ELSE 0.0 END                                                                             AS liquidity_stress_score,
       (e."Working Capital (FQ)" - e."Working Capital (FY)") /
       NULLIF(ABS(e."Working Capital (FY)"), 0)                                                     AS working_capital_trend,
       e."Cash And Equivalents (FQ)" /
       NULLIF(e."Total Operating Expenses (LTM)" / 12.0, 0)                                         AS cash_runway_months,
       CASE
           WHEN (e."Working Capital (FQ)" - e."Working Capital (FY)") /
                NULLIF(ABS(e."Working Capital (FY)"), 0) < -0.2 THEN 1
           ELSE 0 END                                                                               AS wc_deteriorating_flag,
       (e."Retained Earnings (FQ)" - e."Retained Earnings (FY)") /
       NULLIF(ABS(e."Retained Earnings (FY)"), 0)                                                   AS retained_earnings_growth,
       CASE WHEN e."Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END                                   AS accumulated_deficit_flag,
       CASE
           WHEN e."Cash And Equivalents (FQ)" / NULLIF(e."Total Operating Expenses (LTM)" / 12.0, 0) > 6
               THEN 1
           ELSE 0 END                                                                               AS adequate_cash_buffer,

       -- ==========================================================================
       -- ACCOUNTING QUALITY FEATURES (from calc_accounting_quality_features)
       -- ==========================================================================
       (e."Goodwill (LTM)" - e."Goodwill (-1FY)") /
       NULLIF(e."Goodwill (-1FY)", 0)                                                               AS goodwill_change_rate,
       e."Restructuring Charges (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                          AS restructuring_intensity,
       (CASE WHEN ABS(e."Impairment of Goodwill (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(e."Asset Writedown (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN ABS(e."Restructuring Charges (FQ)") > 0 THEN 1
            ELSE 0 END)                                                                             AS exceptional_items_frequency,
       e."Merger & Restructuring Charges (LTM)" / NULLIF(e."Market Cap", 0)                         AS merger_impact_ratio,
       e."Interest Income On Investments (LTM)" /
       NULLIF(ABS(e."Net Income - (IS) (LTM)"), 0)                                                  AS non_operating_income_share,
       CASE WHEN e."Gain (Loss) On Sale Of Assets (LTM)" > 0 THEN 1 ELSE 0 END                      AS asset_sale_boost,
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
                   ))                                                                               AS accounting_quality_score,

       -- ==========================================================================
       -- LEVERAGE FEATURES (from calc_leverage_features)
       -- ==========================================================================
       e."Total Debt (LTM)" / NULLIF(e."Total Equity (LTM)", 0)                                     AS debt_to_equity,
       e."Total Debt (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                     AS debt_to_assets,
       e."Total Equity (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                   AS equity_ratio,
       e."EBIT (LTM)" / NULLIF(e."Interest Expense/Total (LTM)", 0)                                 AS interest_coverage,
       e."Cash And Equivalents (LTM)" / NULLIF(e."Total Current Liabilities (LTM)", 0)              AS cash_ratio,
       e."Working Capital (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                AS working_capital_ratio,

       -- ==========================================================================
       -- EFFICIENCY RATIOS (from calc_efficiency_ratios)
       -- ==========================================================================
       e."Total Revenues (LTM)" / NULLIF(e."Total Assets (LTM)", 0)                                 AS asset_turnover,
       e."Cost Of Revenues (LTM)" / NULLIF(e."Inventory (LTM)", 0)                                  AS inventory_turnover,
       (e."Accounts Receivable/Total (FY)" /
        NULLIF(e."Total Revenues (FY)" / 365.0, 0))                                                 AS receivables_days,
       e."Total Revenues (LTM)" / NULLIF(e."Working Capital (LTM)", 0)                              AS working_capital_turns,

       -- ==========================================================================
       -- BALANCE SHEET DYNAMICS (from calc_balance_sheet_dynamics)
       -- ==========================================================================
       e."Cash And Equivalents (LTM)" / NULLIF(e."Total Assets (LTM)", 0) *
       100                                                                                          AS cash_to_assets_pct,
       (e."Cash And Equivalents (FQ)" - e."Cash And Equivalents (FY)") /
       NULLIF(ABS(e."Cash And Equivalents (FY)"), 0)                                                AS cash_change_qoq,
       e."Cash And Equivalents (FQ)" /
       NULLIF(e."Cash And Equivalents (5YAVGFQ)", 0)                                                AS cash_vs_5y_avg,
       (e."Inventory (FY)" - e."Inventory (FQ)") /
       NULLIF(ABS(e."Inventory (FQ)"), 0)                                                           AS inventory_change_yoy,
       e."Inventory (FQ)" / NULLIF(e."Inventory (5YAVGFQ)", 0)                                      AS inventory_vs_5y_avg,
       (e."Accounts Receivable/Total (FY)" - e."Accounts Receivable/Total (-1FY)") /
       NULLIF(ABS(e."Accounts Receivable/Total (-1FY)"), 0)                                         AS receivables_change_yoy,
       e."Working Capital (FQ)" / NULLIF(e."Working Capital (5YAVGFY)", 0)                          AS working_capital_vs_5y_avg,
       e."Retained Earnings (FQ)" /
       NULLIF(e."Retained Earnings (5YAVGFQ)", 0)                                                   AS retained_earnings_vs_5y,
       CASE
           WHEN e."Gross Intangible Assets (FY)" / NULLIF(e."Gross Intangible Assets (5YAVGFQ)", 0) > 1.5
               THEN 1
           ELSE 0 END                                                                               AS intangibles_growth_flag,
       GREATEST(0, LEAST(100,
                         50 + (e."Cash And Equivalents (LTM)" / NULLIF(e."Total Assets (LTM)", 0) * 100) -
                         (e."Goodwill (LTM)" / NULLIF(e."Total Assets (LTM)", 0) * 100)
                   ))                                                                               AS asset_quality_score,
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN e."Cash And Equivalents (LTM)" / NULLIF(e."Total Assets (LTM)", 0) > 0.10 THEN 25
                              ELSE 0 END) +
                         (CASE
                              WHEN e."Total Equity (LTM)" / NULLIF(e."Total Assets (LTM)", 0) > 0.40 THEN 25
                              ELSE 0 END) +
                         (CASE WHEN e."Working Capital (LTM)" > 0 THEN 25 ELSE 0 END) +
                         (CASE WHEN e."Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END)
                   ))                                                                               AS balance_sheet_strength,
       e."Total Debt (LTM)" / NULLIF(e."EBITDA (LTM)", 0)                                           AS debt_maturity_risk,

       -- ==========================================================================
       -- ANALYST SENTIMENT FEATURES (from calc_sentiment_features)
       -- ==========================================================================
       CASE
           WHEN (e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" +
                 e."# Sell Ratings" + e."# Strong Sell Ratings") > 0
               THEN (e."# Strong Buys Ratings" + e."# Buys Ratings") /
                    NULLIF(e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" +
                           e."# Sell Ratings" + e."# Strong Sell Ratings", 0) * 100
           END                                                                                      AS analyst_bullish_pct,
       CASE
           WHEN (e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" +
                 e."# Sell Ratings" + e."# Strong Sell Ratings") > 0
               THEN (e."# Sell Ratings" + e."# Strong Sell Ratings") /
                    NULLIF(e."# Strong Buys Ratings" + e."# Buys Ratings" + e."# Hold Ratings" +
                           e."# Sell Ratings" + e."# Strong Sell Ratings", 0) * 100
           END                                                                                      AS analyst_bearish_pct,
       (e."Price Target - Median" - e."Last Price") / NULLIF(e."Last Price", 0) *
       100                                                                                          AS upside_potential,
       (e."Price Target - High" - e."Price Target - Low") / NULLIF(e."Price Target - Median", 0) *
       100                                                                                          AS price_target_spread_pct,
       (e."Price Target" - e."Price Target (1M Ago)") /
       NULLIF(e."Price Target (1M Ago)", 0)                                                         AS price_target_revision_1m,
       (e."Price Target" - e."Price Target (3M Ago)") /
       NULLIF(e."Price Target (3M Ago)", 0)                                                         AS price_target_revision_3m,
       COALESCE(e."EPS Est Avg Rev % (FY1E - 1W)", 0) * 0.30 +
       COALESCE(e."EPS Est Avg Rev % (FY1E - 1M)", 0) * 0.25 +
       COALESCE(e."EPS Est Avg Rev % (FY1E - 3M)", 0) * 0.20 +
       COALESCE(e."EPS Est Avg Rev % (FY1E - 6M)", 0) * 0.15 +
       COALESCE(e."EPS Est Avg Rev % (FY1E - 1Y)", 0) *
       0.10                                                                                         AS eps_revision_momentum,
       (e."Analyst Rating" - 1) * 25                                                                AS analyst_rating_normalized,
       e."Price Target - #" / NULLIF(LN(1 + e."Market Cap"), 0)                                     AS analyst_coverage_quality,

       -- ==========================================================================
       -- PRICE TARGET DYNAMICS (from calc_price_target_dynamics)
       -- ==========================================================================
       (e."Price Target" - e."Price Target (1W Ago)") /
       NULLIF(e."Price Target (1W Ago)", 0)                                                         AS pt_momentum_1w,
       (e."Price Target" - e."Price Target (1M Ago)") /
       NULLIF(e."Price Target (1M Ago)", 0)                                                         AS pt_momentum_1m,
       (e."Price Target" - e."Price Target (3M Ago)") /
       NULLIF(e."Price Target (3M Ago)", 0)                                                         AS pt_momentum_3m,
       (e."Price Target" - e."Price Target (6M Ago)") /
       NULLIF(e."Price Target (6M Ago)", 0)                                                         AS pt_momentum_6m,
       (e."Price Target" - e."Price Target (1Y Ago)") /
       NULLIF(e."Price Target (1Y Ago)", 0)                                                         AS pt_momentum_1y,
       (e."Price Target - Median" - e."Price Target - Median (1M Ago)") /
       NULLIF(e."Price Target - Median (1M Ago)", 0)                                                AS pt_median_momentum_1m,
       (e."Price Target - Median" - e."Price Target - Median (3M Ago)") /
       NULLIF(e."Price Target - Median (3M Ago)", 0)                                                AS pt_median_momentum_3m,
       ((e."Price Target - High (3M Ago)" - e."Price Target - Low (3M Ago)") /
        NULLIF(e."Price Target - Median (3M Ago)", 0)) -
       ((e."Price Target - High" - e."Price Target - Low") /
        NULLIF(e."Price Target - Median", 0))                                                       AS pt_consensus_convergence,
       (e."Price Target - #" - e."Price Target - # (1M Ago)")::INTEGER                              AS analyst_coverage_change_1m,
       (e."Price Target - #" - e."Price Target - # (3M Ago)")::INTEGER                              AS analyst_coverage_change_3m,
       (e."Price Target - #" - e."Price Target - # (1Y Ago)")::INTEGER                              AS analyst_coverage_change_1y,

       -- ==========================================================================
       -- EARNINGS FEATURES (from calc_earnings_features)
       -- ==========================================================================
       CASE
           WHEN ABS(e."EPS Norm - Est Avg (FY1E)") > 0
               THEN (e."EPS/Adj. (LTM)" - e."EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS(e."EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                                                                      AS eps_surprise_pct,
       CASE
           WHEN ABS(e."Revenues - Est Avg (FY1E)") > 0
               THEN (e."Total Revenues (LTM)" - e."Revenues - Est Avg (FY1E)") /
                    NULLIF(ABS(e."Revenues - Est Avg (FY1E)"), 0) * 100
           END                                                                                      AS revenue_surprise_pct,
       e."EPS/Adj. (LTM)" / NULLIF(e."Net EPS - Basic (LTM)", 0)                                    AS eps_adjustment_ratio,
       e."EBITDA/Adj. (LTM)" / NULLIF(e."EBITDA (LTM)", 0)                                          AS ebitda_adjustment_ratio,
       CASE
           WHEN ABS(e."Net EPS - Basic (-4FQFQ)") > 0
               THEN (e."Net EPS - Basic (FQ)" - e."Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS(e."Net EPS - Basic (-4FQFQ)"), 0)
           END                                                                                      AS eps_quarterly_trend,
       CASE
           WHEN ABS(e."Net EPS - Basic (-1FY)") > 0
               THEN (e."Net EPS - Basic (FY)" - e."Net EPS - Basic (-1FY)") /
                    NULLIF(ABS(e."Net EPS - Basic (-1FY)"), 0) * 100
           END                                                                                      AS eps_yoy_growth,

       -- ==========================================================================
       -- EPS TRAJECTORY FEATURES (from calc_eps_trajectory_features)
       -- ==========================================================================
       (CASE WHEN e."Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN e."Net EPS - Basic (-4FQFQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                                                    AS eps_positive_streak,
       CASE
           WHEN e."Net EPS - Basic (-3FY)" > 0 AND e."Net EPS - Basic (FY)" > 0
               THEN (POWER(e."Net EPS - Basic (FY)" / NULLIF(e."Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                                      AS eps_cagr_3y,
       CASE
           WHEN e."Net EPS - Basic (-5FY)" > 0 AND e."Net EPS - Basic (FY)" > 0
               THEN (POWER(e."Net EPS - Basic (FY)" / NULLIF(e."Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                                                      AS eps_cagr_5y,
       (CASE WHEN e."Net EPS - Basic (FY)" > e."Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN e."Net EPS - Basic (-1FY)" > e."Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN e."Net EPS - Basic (-2FY)" > e."Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN e."Net EPS - Basic (-3FY)" > e."Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN e."Net EPS - Basic (-4FY)" > e."Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 *
       100                                                                                          AS eps_trajectory_score,

       -- ==========================================================================
       -- GAAP VS ADJUSTED (from calc_gaap_adjusted_analytics)
       -- ==========================================================================
       e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)"                                               AS eps_adjustment_spread,
       (e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") /
       NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) *
       100                                                                                          AS eps_adjustment_pct,
       e."Net Income/Adj. (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0)                           AS net_income_adjustment_ratio,
       (e."EBITDA/Adj. (LTM)" - e."EBITDA (LTM)") / NULLIF(ABS(e."EBITDA (LTM)"), 0) *
       100                                                                                          AS ebitda_adjustment_pct,
       GREATEST(0, LEAST(100,
                         100 - ABS((e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") /
                                   NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) *
                                   100)))                                                           AS earnings_quality_score,
       CASE
           WHEN ABS((e."EPS/Adj. (LTM)" - e."Net EPS - Basic (LTM)") /
                    NULLIF(ABS(e."Net EPS - Basic (LTM)"), 0) * 100) > 15
               THEN 1
           ELSE 0 END                                                                               AS earnings_quality_warning,

       -- ==========================================================================
       -- GAAP REVISION FEATURES (from calc_gaap_revision_features)
       -- ==========================================================================
       COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
       COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
       COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
       COALESCE(e."EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) *
       0.15                                                                                         AS gaap_revision_momentum,
       e."EPS Est Avg Rev % (FY1E - 3M)" -
       e."EPS GAAP Est Avg Rev % (FY1E - 3M)"                                                       AS gaap_vs_norm_revision_spread,
       CASE
           WHEN e."EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0
               AND e."EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0
               AND e."EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0
               THEN 1
           ELSE 0 END                                                                               AS gaap_positive_revision_flag,

       -- ==========================================================================
       -- GROWTH FEATURES (from calc_growth_features)
       -- ==========================================================================
       CASE
           WHEN ABS(e."Total Revenues (-1FY)") > 0
               THEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
                    NULLIF(ABS(e."Total Revenues (-1FY)"), 0) * 100
           END                                                                                      AS revenue_growth_yoy,
       CASE
           WHEN ABS(e."EBITDA (-1FY)") > 0
               THEN (e."EBITDA (FY)" - e."EBITDA (-1FY)") / NULLIF(ABS(e."EBITDA (-1FY)"), 0) * 100
           END                                                                                      AS ebitda_growth_yoy,
       CASE
           WHEN ABS(e."Operating Income (FY)") > 0
               THEN (e."Operating Income (LTM)" - e."Operating Income (FY)") /
                    NULLIF(ABS(e."Operating Income (FY)"), 0) * 100
           END                                                                                      AS operating_income_growth,
       CASE
           WHEN ABS(e."FCF (FY)") > 0
               THEN (e."FCF (LTM)" - e."FCF (FY)") / NULLIF(ABS(e."FCF (FY)"), 0) * 100
           END                                                                                      AS fcf_growth,
       e."Total Revenues/CAGR (5Y FY)"                                                              AS revenue_cagr_5y,
       e."Revenues - Est YoY % (FY1E)"                                                              AS forward_revenue_growth,
       e."Total Revenues (LTM)" / NULLIF(e."Total Revenues (5YAVGLTM)", 0)                          AS revenue_vs_5y_avg,

       -- ==========================================================================
       -- REVENUE FORECAST FEATURES (from calc_revenue_forecast_features)
       -- ==========================================================================
       (e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)") /
       NULLIF(e."Revenues - Est Med (FY1E)", 0) *
       100                                                                                          AS revenue_est_spread,
       (e."Total Revenues (LTM)" - e."Revenues - Est Avg (FY1E)") /
       NULLIF(ABS(e."Revenues - Est Avg (FY1E)"), 0) *
       100                                                                                          AS revenue_beat_potential,
       e."Enterprise Value" / NULLIF(e."Revenues - Est Avg (FY1E)", 0)                              AS forward_revenue_multiple,
       (e."Revenues - Est Avg (FY1E)" - e."Total Revenues (FY)") /
       NULLIF(ABS(e."Total Revenues (FY)"), 0) *
       100                                                                                          AS consensus_revenue_growth,
       e."EBITDA - Est Avg (FY1E)" / NULLIF(e."Revenues - Est Avg (FY1E)", 0) *
       100                                                                                          AS forward_ebitda_margin,
       e."Revenues - Est YoY % (FY1E)" - e."Total Revenues/CAGR (5Y FY)"                            AS revenue_acceleration,
       GREATEST(0, LEAST(100,
                         100 - ABS((e."Revenues - Est Avg (FY1E)" - e."Revenues - Est Med (FY1E)") /
                                   NULLIF(e."Revenues - Est Med (FY1E)", 0) * 100)
                   ))                                                                               AS estimate_confidence_score,

       -- ==========================================================================
       -- DIVIDEND FEATURES (from calc_dividend_features)
       -- ==========================================================================
       e."Dividend Streak"::INTEGER                                                                 AS dividend_streak,
       e."Div Yield (LTM)"                                                                          AS dividend_yield_ltm,
       e."Div Yield (NTM)"                                                                          AS dividend_yield_ntm,
       ABS(e."Common Dividends Paid (LTM)") /
       NULLIF(e."Net Income/Adj. (LTM)", 0)                                                         AS dividend_payout_ratio,
       CASE
           WHEN ABS(e."Common Dividends Paid (LTM)") > 0
               THEN e."FCF (LTM)" / NULLIF(ABS(e."Common Dividends Paid (LTM)"), 0)
           END                                                                                      AS fcf_dividend_coverage,
       e."Buyback Yield (LTM)"                                                                      AS buyback_yield,
       COALESCE(e."Buyback Yield (LTM)", 0) +
       COALESCE(e."Div Yield (LTM)", 0)                                                             AS total_shareholder_yield,
       e."Div Yield (NTM)" - e."Div Yield (LTM)"                                                    AS dividend_growth_expectation,

       -- ==========================================================================
       -- DIVIDEND TIMING (from calc_dividend_timing)
       -- ==========================================================================
       (CURRENT_DATE - e."Dividend Record (Ex Date)")::INTEGER                                      AS days_since_ex_date,
       (e."Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER                                 AS days_to_payment,
       CASE
           WHEN (CURRENT_DATE - e."Dividend Record (Announce Date)") <= 30
               THEN 1
           ELSE 0 END                                                                               AS dividend_announced_flag,
       CASE
           WHEN (e."Dividend Record (Ex Date)" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0 END                                                                               AS ex_date_approaching_flag,
       LEAST(1.0, e."Dividend Streak"::NUMERIC / 10.0)                                              AS dividend_consistency,
       e."Div Yield (LTM)" / NULLIF(e."Div Yield (5YAVGLTM)", 0)                                    AS dividend_yield_vs_5y_avg,

       -- ==========================================================================
       -- EMPLOYMENT FEATURES (from calc_employment_features)
       -- ==========================================================================
       CASE
           WHEN e."Full Time Employees (FY)" > 0
               THEN e."Total Revenues (FY)" / NULLIF(e."Full Time Employees (FY)", 0)
           END                                                                                      AS revenue_per_employee,
       CASE
           WHEN e."Full Time Employees (FY)" > 0
               THEN e."Normalized Net Income (FY)" / NULLIF(e."Full Time Employees (FY)", 0)
           END                                                                                      AS profit_per_employee,
       CASE
           WHEN e."Full Time Employees (FY)" > 0
               THEN e."EBITDA (FY)" / NULLIF(e."Full Time Employees (FY)", 0)
           END                                                                                      AS ebitda_per_employee,
       CASE
           WHEN e."Full Time Employees (-1FY)" > 0
               THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                    NULLIF(e."Full Time Employees (-1FY)", 0) * 100
           END                                                                                      AS fte_growth_1y_pct,
       CASE
           WHEN e."Full Time Employees (-3FY)" > 0
               THEN (e."Full Time Employees (FY)" - e."Full Time Employees (-3FY)") /
                    NULLIF(e."Full Time Employees (-3FY)", 0) * 100
           END                                                                                      AS fte_growth_3y_pct,
       CASE
           WHEN e."Avg Employees (5YAVGFY)" > 0
               THEN e."Full Time Employees (FY)" / NULLIF(e."Avg Employees (5YAVGFY)", 0)
           END                                                                                      AS workforce_stability,

       -- ==========================================================================
       -- EMPLOYMENT DYNAMICS (from calc_employment_dynamics)
       -- ==========================================================================
       CASE
           WHEN e."Full Time Employees (FY)" < e."Full Time Employees (-1FY)"
               AND e."Total Revenues (FY)" < e."Total Revenues (-1FY)"
               THEN 1
           ELSE 0 END                                                                               AS layoff_risk_flag,
       CASE
           WHEN (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                NULLIF(e."Full Time Employees (-1FY)", 0) > 0.20
               THEN 1
           ELSE 0 END                                                                               AS rapid_hiring_flag,
       CASE
           WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
                NULLIF(ABS(e."Total Revenues (-1FY)"), 0) >
                (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") /
                NULLIF(e."Full Time Employees (-1FY)", 0)
               AND (e."Full Time Employees (FY)" - e."Full Time Employees (-1FY)") > 0
               THEN 1
           ELSE 0 END                                                                               AS sustainable_growth_flag,

       -- ==========================================================================
       -- CASH FLOW FEATURES (from calc_cashflow_features)
       -- ==========================================================================
       e."CFO (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0)                                       AS cfo_to_net_income,
       e."FCF (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0)                                       AS fcf_to_net_income,
       e."FCF (LTM)" / NULLIF(e."Total Revenues (LTM)", 0)                                          AS fcf_margin,
       (e."CFO (LTM)" - e."CFO (-1FY)") / NULLIF(e."CFO (-1FY)", 0)                                 AS cfo_growth_yoy,
       (CASE WHEN e."FCF (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END) /
       5.0                                                                                          AS fcf_positive_ratio,
       ABS(COALESCE(e."Cash Acquisitions (FQ)", 0)) +
       ABS(COALESCE(e."Cash Acquisitions (-1FQFQ)", 0)) +
       ABS(COALESCE(e."Cash Acquisitions (-2FQFQ)", 0)) +
       ABS(COALESCE(e."Cash Acquisitions (-3FQFQ)", 0))                                             AS acquisition_intensity,
       CASE
           WHEN ABS(e."CFI (LTM)") > 0
               THEN e."CFO (LTM)" / NULLIF(ABS(e."CFI (LTM)"), 0)
           END                                                                                      AS self_funding_ratio,

       -- ==========================================================================
       -- ENHANCED CASH FLOW (from calc_enhanced_cashflow_features)
       -- ==========================================================================
       (CASE WHEN e."FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN e."FCF (-4FY)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                                                    AS fcf_positive_years,
       CASE
           WHEN e."FCF (FY)" > 0 AND e."FCF (-1FY)" > 0 AND e."FCF (-2FY)" > 0
               AND e."FCF (-3FY)" > 0 AND e."FCF (-4FY)" > 0
               THEN 1
           ELSE 0 END                                                                               AS fcf_always_positive,
       ABS(e."Capital Expenditure (FQ)") /
       NULLIF(ABS(e."Capital Expenditure (5YAVGFQ)"), 0)                                            AS capex_vs_5y_avg,
       CASE
           WHEN ABS(e."Capital Expenditure (FQ)") / NULLIF(ABS(e."Capital Expenditure (5YAVGFQ)"), 0) < 0.7
               THEN 1
           ELSE 0 END                                                                               AS underinvestment_flag,
       CASE
           WHEN e."CFO (LTM)" / NULLIF(ABS(e."CFI (LTM)"), 0) > 1 THEN 1
           ELSE 0 END                                                                               AS self_funding_flag,
       (CASE WHEN e."CFO (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN e."FCF (FY)" > 0 AND e."FCF (-1FY)" > 0 AND e."FCF (-2FY)" > 0
                AND e."FCF (-3FY)" > 0 AND e."FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN e."CFO (LTM)" > ABS(e."CFI (LTM)") THEN 25 ELSE 0 END +
        CASE
            WHEN e."FCF (LTM)" > 0 THEN 25
            ELSE 0 END)::NUMERIC                                                                    AS cash_flow_quality_score,

       -- ==========================================================================
       -- CASHFLOW TEMPORAL (from calc_cashflow_temporal_features)
       -- ==========================================================================
       (e."CFO (FQ)" - e."CFO (-4FQFQ)") / NULLIF(ABS(e."CFO (-4FQFQ)"), 0) *
       100                                                                                          AS cfo_quarterly_trend,
       (e."FCF (FQ)" - e."FCF (-4FQFQ)") / NULLIF(ABS(e."FCF (-4FQFQ)"), 0) *
       100                                                                                          AS fcf_quarterly_trend,
       (CASE WHEN e."CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN e."CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN e."CFO (-4FQFQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                                                    AS cfo_positive_quarters,
       CASE
           WHEN e."FCF (LTM)" < 0
               THEN ABS(e."FCF (LTM)") / NULLIF(e."Cash And Equivalents (FQ)", 0) / 12.0
           ELSE 0 END                                                                               AS cash_burn_rate,
       ABS(e."CFF (LTM)") / NULLIF(ABS(e."CFO (LTM)"), 0)                                           AS financing_dependency,

       -- ==========================================================================
       -- TEMPORAL FEATURES (from calc_temporal_features)
       -- ==========================================================================
       e."Fiscal Quarter",
       e."Fiscal Month",
       e."Fiscal Year",
       (e."Next Earnings" - CURRENT_DATE)                                                           AS days_to_earnings,
       (CURRENT_DATE - e."Income Statement Report Date")                                            AS earnings_report_recency,
       e."Reporting Lag",
       e."Fiscal Month" / 12.0                                                                      AS fiscal_year_progress,

       -- ==========================================================================
       -- FISCAL CALENDAR FEATURES (from calc_fiscal_calendar_features)
       -- ==========================================================================
       (CURRENT_DATE - e."Income Statement Report Date")::INTEGER                                   AS days_since_last_report,
       (e."FY End Date" - CURRENT_DATE)::INTEGER                                                    AS days_to_fy_end,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12) THEN 1
           ELSE 0 END                                                                               AS is_quarter_end_month,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 2, 4, 5, 7, 8, 10, 11) THEN 1
           ELSE 0 END                                                                               AS earnings_season_flag,
       CASE
           WHEN (e."Next Earnings" - CURRENT_DATE) BETWEEN 0 AND 14 THEN 1
           ELSE 0 END                                                                               AS pre_earnings_window,
       CASE
           WHEN (CURRENT_DATE - e."Income Statement Report Date") BETWEEN 0 AND 7 THEN 1
           ELSE 0 END                                                                               AS post_earnings_window,
       GREATEST(0, LEAST(100,
                         100 - ((CURRENT_DATE - e."Income Statement Report Date")::NUMERIC / 90.0 * 100)
                   ))                                                                               AS reporting_freshness_score,

       -- ==========================================================================
       -- COMPOSITE SCORES (from calc_composite_scores)
       -- ==========================================================================
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
           )::INTEGER                                                                               AS piotroski_f_score,
       GREATEST(0, LEAST(100,
                         50 - ((e."Shrs Out" - e."Shrs Out (-1FY)") / NULLIF(e."Shrs Out (-1FY)", 0)) * 100
                   ))                                                                               AS dilution_score,
       (((CASE WHEN e."CFO (LTM)" / NULLIF(e."Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
          CASE WHEN e."Return On Equity % (LTM)" > 15 THEN 25 ELSE 0 END +
          CASE WHEN e."Total Debt (LTM)" / NULLIF(e."Total Equity (LTM)", 0) < 1 THEN 25 ELSE 0 END +
          CASE WHEN e."Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END) * 0.40) +
        (LEAST(100, GREATEST(0,
                             ((e."Last Price" - e."Price (3M Ago)") / NULLIF(e."Price (3M Ago)", 0) * 100 + 50))) *
         0.30) +
        (CASE
             WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
                  NULLIF(ABS(e."Total Revenues (-1FY)"), 0) * 100 > 20 THEN 100
             WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
                  NULLIF(ABS(e."Total Revenues (-1FY)"), 0) * 100 > 10 THEN 75
             WHEN (e."Total Revenues (FY)" - e."Total Revenues (-1FY)") /
                  NULLIF(ABS(e."Total Revenues (-1FY)"), 0) * 100 > 0 THEN 50
             ELSE 25 END * 0.30)
           )                                                                                        AS quality_momentum_score

FROM postgres.public.equities e;

-- =============================================================================
-- SECTION 13A: PERFORMANCE OPTIMIZATION - TABLE INDEXES
-- =============================================================================
-- Essential indexes for the equities table to improve query performance
-- These indexes support filtering operations commonly used in feature queries

-- ===========================================
-- ID columns (primary lookups)
-- ===========================================
CREATE INDEX IF NOT EXISTS idx_equities_ticker ON public.equities ("Ticker");
CREATE INDEX IF NOT EXISTS idx_equities_isin ON public.equities ("ISIN");
CREATE INDEX IF NOT EXISTS idx_equities_name ON public.equities ("Name");

-- ===========================================
-- CATEGORICAL columns (filtering/grouping)
-- ===========================================
CREATE INDEX IF NOT EXISTS idx_equities_region ON public.equities ("Region");
CREATE INDEX IF NOT EXISTS idx_equities_country ON public.equities ("Country");
CREATE INDEX IF NOT EXISTS idx_equities_trading_country ON public.equities ("Trading Country");
CREATE INDEX IF NOT EXISTS idx_equities_exchange ON public.equities ("Exchange");
CREATE INDEX IF NOT EXISTS idx_equities_unit ON public.equities ("Unit");
CREATE INDEX IF NOT EXISTS idx_equities_sector ON public.equities ("Sector");
CREATE INDEX IF NOT EXISTS idx_equities_industry ON public.equities ("Industry");
CREATE INDEX IF NOT EXISTS idx_equities_style_class ON public.equities ("Style Class");
CREATE INDEX IF NOT EXISTS idx_equities_size_class ON public.equities ("Size Class");

-- ===========================================
-- DATE columns (temporal queries)
-- ===========================================
CREATE INDEX IF NOT EXISTS idx_equities_report_date ON public.equities ("Income Statement Report Date");
CREATE INDEX IF NOT EXISTS idx_equities_next_earnings ON public.equities ("Next Earnings");
CREATE INDEX IF NOT EXISTS idx_equities_last_updated ON public.equities ("Last Updated");
CREATE INDEX IF NOT EXISTS idx_equities_fy_end_date ON public.equities ("FY End Date");

-- ===========================================
-- MARKET columns (sorting/filtering by size)
-- ===========================================
CREATE INDEX IF NOT EXISTS idx_equities_market_cap ON public.equities ("Market Cap");
CREATE INDEX IF NOT EXISTS idx_equities_enterprise_value ON public.equities ("Enterprise Value");
CREATE INDEX IF NOT EXISTS idx_equities_last_price ON public.equities ("Last Price");

-- ===========================================
-- Composite indexes for common query patterns
-- ===========================================
CREATE INDEX IF NOT EXISTS idx_equities_sector_region ON public.equities ("Sector", "Region");
CREATE INDEX IF NOT EXISTS idx_equities_country_sector ON public.equities ("Country", "Sector");
CREATE INDEX IF NOT EXISTS idx_equities_industry_size ON public.equities ("Industry", "Size Class");
CREATE INDEX IF NOT EXISTS idx_equities_region_market_cap ON public.equities ("Region", "Market Cap" DESC);

-- =============================================================================
-- SECTION 13B: NEW VIEWS FOR ENHANCED FEATURES
-- =============================================================================

CREATE OR REPLACE VIEW v_technical_analysis_features AS
SELECT "Ticker"                                                      AS ticker,
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
       CASE WHEN "Rel. Volume" > 1.5 THEN 1 ELSE 0 END               AS high_volume_flag
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_financial_distress_features AS
SELECT "Ticker"                                                                          AS ticker,
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
       NULLIF("Total Operating Expenses (LTM)" / 12.0, 0)                                AS cash_runway_months
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_price_target_dynamics AS
SELECT "Ticker"                                                    AS ticker,
       ("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)
                                                                   AS pt_momentum_1m,
       ("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)
                                                                   AS pt_momentum_3m,
       ("Price Target - Median" - "Price Target - Median (3M Ago)") /
       NULLIF("Price Target - Median (3M Ago)", 0)                 AS pt_median_momentum_3m,
       (("Price Target - High (3M Ago)" - "Price Target - Low (3M Ago)") /
        NULLIF("Price Target - Median (3M Ago)", 0)) -
       (("Price Target - High" - "Price Target - Low") /
        NULLIF("Price Target - Median", 0))                        AS pt_consensus_convergence,
       ("Price Target - #" - "Price Target - # (3M Ago)")::INTEGER AS analyst_coverage_change_3m
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_composite_scores AS
SELECT "Ticker"          AS ticker,
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
                   ))    AS dilution_score
FROM postgres.public.equities;
-- =============================================================================
-- SECTION 13C: MATERIALIZED VIEW FOR PERFORMANCE
-- =============================================================================
-- Materialized view for better performance when data doesn't change frequently
-- Use this instead of vw_feature_registry for read-heavy workloads

DROP MATERIALIZED VIEW IF EXISTS mv_feature_registry CASCADE;

CREATE MATERIALIZED VIEW mv_feature_registry AS
SELECT *
FROM vw_feature_registry;

-- Indexes on the materialized view for common query patterns
CREATE INDEX IF NOT EXISTS idx_mv_feature_ticker ON mv_feature_registry ("Ticker");
CREATE INDEX IF NOT EXISTS idx_mv_feature_sector ON mv_feature_registry ("Sector");
CREATE INDEX IF NOT EXISTS idx_mv_feature_region ON mv_feature_registry ("Region");
CREATE INDEX IF NOT EXISTS idx_mv_feature_market_cap ON mv_feature_registry ("Market Cap");
CREATE INDEX IF NOT EXISTS idx_mv_feature_country ON mv_feature_registry ("Country");
CREATE INDEX IF NOT EXISTS idx_mv_feature_industry ON mv_feature_registry ("Industry");

-- Composite indexes for common multi-column filters
CREATE INDEX IF NOT EXISTS idx_mv_feature_sector_region ON mv_feature_registry ("Sector", "Region");
CREATE INDEX IF NOT EXISTS idx_mv_feature_piotroski ON mv_feature_registry (piotroski_f_score DESC);
CREATE INDEX IF NOT EXISTS idx_mv_feature_momentum ON mv_feature_registry (price_momentum_3m DESC);
CREATE INDEX IF NOT EXISTS idx_mv_feature_quality ON mv_feature_registry (earnings_quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_mv_feature_distress ON mv_feature_registry (distress_risk_score);

-- To refresh the materialized view (run periodically or after data updates):
-- REFRESH MATERIALIZED VIEW CONCURRENTLY mv_feature_registry;

-- Additional materialized views for specific feature categories (performance optimization)
DROP MATERIALIZED VIEW IF EXISTS mv_technical_features CASCADE;

CREATE MATERIALIZED VIEW mv_technical_features AS
SELECT *
FROM v_technical_analysis_features;;

CREATE INDEX IF NOT EXISTS idx_mv_tech_ticker ON mv_technical_features (ticker);
CREATE INDEX IF NOT EXISTS idx_mv_tech_breakout ON mv_technical_features (breakout_signal) WHERE breakout_signal = 1;
CREATE INDEX IF NOT EXISTS idx_mv_tech_trend ON mv_technical_features (ema_trend_consistency);

DROP MATERIALIZED VIEW IF EXISTS mv_distress_features CASCADE;

CREATE MATERIALIZED VIEW mv_distress_features AS
SELECT *
FROM v_financial_distress_features;

CREATE INDEX IF NOT EXISTS idx_mv_distress_ticker ON mv_distress_features (ticker);
CREATE INDEX IF NOT EXISTS idx_mv_distress_risk ON mv_distress_features (distress_risk_score);
CREATE INDEX IF NOT EXISTS idx_mv_distress_liquidity ON mv_distress_features (liquidity_stress_score);

DROP MATERIALIZED VIEW IF EXISTS mv_composite_scores CASCADE;

CREATE MATERIALIZED VIEW mv_composite_scores AS
SELECT *
FROM v_composite_scores;

CREATE INDEX IF NOT EXISTS idx_mv_composite_ticker ON mv_composite_scores (ticker);
CREATE INDEX IF NOT EXISTS idx_mv_composite_piotroski ON mv_composite_scores (piotroski_f_score DESC);
CREATE INDEX IF NOT EXISTS idx_mv_composite_eps_traj ON mv_composite_scores (eps_trajectory_score DESC);

DROP MATERIALIZED VIEW IF EXISTS mv_price_target_dynamics CASCADE;

CREATE MATERIALIZED VIEW mv_price_target_dynamics AS
SELECT *
FROM v_price_target_dynamics;

CREATE INDEX IF NOT EXISTS idx_mv_pt_ticker ON mv_price_target_dynamics (ticker);
CREATE INDEX IF NOT EXISTS idx_mv_pt_momentum ON mv_price_target_dynamics (pt_momentum_3m DESC);

-- Refresh all materialized views function
CREATE OR REPLACE FUNCTION refresh_all_feature_materialized_views()
    RETURNS void
    LANGUAGE plpgsql
AS
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_feature_registry;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_technical_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_distress_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_composite_scores;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_price_target_dynamics;
END;
$$;

-- To refresh all materialized views at once:
-- SELECT refresh_all_feature_materialized_views();

-- =============================================================================
-- SECTION 13D: ALTERNATIVE VIEWS FOR FUNCTION INLINING
-- =============================================================================
-- Views as alternatives to functions for better query optimization
-- SQL-language functions with RETURNS TABLE may not inline well

CREATE OR REPLACE VIEW v_valuation_features AS
SELECT "Ticker"          AS ticker,
       "P/E (LTM)"       AS p_e_ratio,
       "P/B (LTM)"       AS p_b_ratio,
       "EV/EBITDA (LTM)" AS ev_ebitda_ratio,
       "EV/Sales (LTM)"  AS ev_sales_ratio,
       "Div Yield (LTM)" AS dividend_yield,
       CASE
           WHEN "Total Revenues/CAGR (5Y FY)" > 0
               THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0)
           END           AS peg_ratio
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_momentum_features AS
SELECT "Ticker"                                                                   AS ticker,
       ("Last Price" - "Price (1M Ago)") / NULLIF("Price (1M Ago)", 0) * 100      AS price_momentum_1m,
       ("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100      AS price_momentum_3m,
       ("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 100      AS price_momentum_1y,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" THEN -1
           ELSE 0
           END                                                                    AS ema_crossover_20_50,
       ("Last Price" - "52W Low/Adj") / NULLIF("52W High/Adj" - "52W Low/Adj", 0) AS range_52w_position,
       "Beta (1Y)" - "Beta (5Y)"                                                  AS beta_momentum
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_profitability_features AS
SELECT "Ticker"                                                 AS ticker,
       "Return On Equity % (LTM)"                               AS roe,
       "Return on Assets (ROA) % (LTM)"                         AS roa,
       "Gross Profit Margin % (LTM)"                            AS gross_margin_pct,
       "Net Income Margin % (LTM)"                              AS net_margin_pct,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100 AS ebitda_margin_pct
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_leverage_features AS
SELECT "Ticker"                                                 AS ticker,
       "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0)     AS debt_to_equity,
       "Total Debt (LTM)" / NULLIF("Total Assets (LTM)", 0)     AS debt_to_assets,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0) AS interest_coverage,
       "Current Ratio (LTM)"                                    AS current_ratio
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_margin_trends AS
SELECT "Ticker"                                                             AS ticker,
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
           END                                                              AS margin_expansion_flag
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_efficiency_ratios AS
SELECT "Ticker"                                                                      AS ticker,
       "Total Revenues (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS asset_turnover,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)                       AS inventory_turnover,
       ("Accounts Receivable/Total (FY)" / NULLIF("Total Revenues (FY)" / 365.0, 0)) AS receivables_days,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turns
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_balance_sheet_dynamics AS
SELECT "Ticker"                                                             AS ticker,
       "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100 AS cash_to_assets_pct,
       ("Cash And Equivalents (FQ)" - "Cash And Equivalents (FY)") /
       NULLIF(ABS("Cash And Equivalents (FY)"), 0)                          AS cash_change_qoq,
       "Inventory (FY)" - "Inventory (FQ)"                                  AS inventory_change_yoy,
       "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0)                       AS debt_maturity_risk
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_revenue_forecast_features AS
SELECT "Ticker"                                                    AS ticker,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                AS revenue_est_spread,
       "Revenues - Est YoY % (FY1E)"                               AS revenue_est_revision_trend,
       "Enterprise Value" / NULLIF("Revenues - Est Avg (FY1E)", 0) AS forward_revenue_multiple,
       ("Revenues - Est Avg (FY1E)" - "Total Revenues (FY)") /
       NULLIF(ABS("Total Revenues (FY)"), 0) * 100                 AS consensus_revenue_growth
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_employment_dynamics AS
SELECT "Ticker"                                        AS ticker,
       CASE
           WHEN "Full Time Employees (-2FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
                    NULLIF("Full Time Employees (-2FY)", 0) * 100
           END                                         AS fte_growth_2y_pct,
       (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
        NULLIF("Full Time Employees (-1FY)", 0) * 100) -
       (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
        NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100) AS headcount_vs_revenue,
       CASE
           WHEN "Full Time Employees (FY)" < "Full Time Employees (-1FY)"
               AND "Total Revenues (FY)" < "Total Revenues (-1FY)"
               THEN 1
           ELSE 0
           END                                         AS layoff_risk_flag
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_dividend_timing AS
SELECT "Ticker"                                                   AS ticker,
       (CURRENT_DATE - "Dividend Record (Ex Date)")::INTEGER      AS days_since_ex_date,
       ("Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER AS days_to_payment,
       CASE
           WHEN (CURRENT_DATE - "Dividend Record (Announce Date)") <= 30
               THEN 1
           ELSE 0
           END                                                    AS dividend_announced_flag,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0)      AS dividend_yield_vs_5y_avg
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_fiscal_calendar_features AS
SELECT "Ticker"                                                 AS ticker,
       (CURRENT_DATE - "Income Statement Report Date")::INTEGER AS days_since_last_report,
       ("FY End Date" - CURRENT_DATE)::INTEGER                  AS days_to_fy_end,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12)
               THEN 1
           ELSE 0
           END                                                  AS is_quarter_end_month,
       GREATEST(0, LEAST(100,
                         100 - ((CURRENT_DATE - "Income Statement Report Date")::NUMERIC / 90.0 * 100)
                   ))                                           AS reporting_freshness_score
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_cashflow_temporal_features AS
SELECT "Ticker"                                                             AS ticker,
       ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100 AS cfo_quarterly_trend,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0) * 100 AS fcf_quarterly_trend,
       CASE
           WHEN "FCF (LTM)" < 0
               THEN ABS("FCF (LTM)") / NULLIF("Cash And Equivalents (FQ)", 0) / 12.0
           ELSE 0
           END                                                              AS cash_burn_rate,
       ABS("CFF (LTM)") / NULLIF(ABS("CFO (LTM)"), 0)                       AS financing_dependency
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_gaap_revision_features AS
SELECT "Ticker"                                                               AS ticker,
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) * 0.15               AS gaap_revision_momentum,
       "EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)" AS gaap_vs_norm_revision_spread,
       CASE
           WHEN "EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0
               THEN 1
           ELSE 0
           END                                                                AS gaap_positive_revision_flag
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_extended_valuation_timeseries AS
SELECT "Ticker"                                                                    AS ticker,
       ("EV/Sales (LTM)" - "EV/Sales (-1FQLTM)") / NULLIF("EV/Sales (-1FQLTM)", 0) AS ev_sales_qoq_1q,
       ("P/E (LTM)" - "P/E (5YAVGLTM)") / NULLIF("P/E (5YAVGLTM)", 0)              AS p_e_vs_5y_avg,
       (("P/E (LTM)" / NULLIF("P/E (3YAVGLTM)", 0)) +
        ("EV/EBITDA (LTM)" / NULLIF("EV/EBITDA (3YAVGLTM)", 0))) / 2.0 - 1.0       AS valuation_compression
FROM postgres.public.equities;

-- =============================================================================
-- SECTION 13E: ENHANCED FEATURE FUNCTIONS (NEW)
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
SELECT "Ticker"                                                                AS ticker,
       ("Last Price" - "Price (QTD Ago)") / NULLIF("Price (QTD Ago)", 0) * 100 AS price_momentum_qtd,
       ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 100   AS price_momentum_3y,
       ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 100   AS price_momentum_5y,
       (POWER("Last Price" / NULLIF("Price (1Y Ago)", 0), 1.0) - 1) -
       (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1)      AS momentum_acceleration_1y,
       (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1) -
       (POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1)      AS momentum_acceleration_3y,
       (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 0.5 +
        ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 0.3 +
        ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 0.2) *
       100                                                                     AS long_term_trend_score,
       "Last Price" / NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)") / 2, 0)     AS price_vs_3y_avg,
       "Last Price" /
       NULLIF(("Price (1Y Ago)" + "Price (3Y Ago)" + "Price (5Y Ago)") / 3, 0) AS price_vs_5y_avg,
       CASE
           WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
                ("Last Price" > "Price (1Y Ago)") AND ("Last Price" > "Price (3Y Ago)") THEN 1.0
           WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
                ("Last Price" > "Price (1Y Ago)") THEN 0.75
           WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") THEN 0.5
           WHEN ("Last Price" > "Price (1M Ago)") THEN 0.25
           ELSE 0
           END                                                                 AS momentum_consistency,
       CASE
           WHEN POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1 > 0.10 THEN 1
           ELSE 0 END                                                          AS secular_trend_flag
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
SELECT "Ticker"                                                            AS ticker,
       "Beta (1Y)"                                                         AS beta_1y,
       "Beta (2Y)"                                                         AS beta_2y,
       "Beta (5Y)"                                                         AS beta_5y,
       "Beta (1Y)" - "Beta (2Y)"                                           AS beta_trend_short,
       "Beta (2Y)" - "Beta (5Y)"                                           AS beta_trend_long,
       CASE
           WHEN GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") - LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") > 0
               THEN 1.0 /
                    (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") - LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)"))
           ELSE 1.0
           END                                                             AS beta_stability,
       CASE WHEN ABS("Beta (1Y)" - "Beta (5Y)") > 0.3 THEN 1 ELSE 0 END    AS beta_regime_change,
       "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2           AS systematic_risk_score,
       CASE
           WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8 THEN 1
           ELSE 0 END                                                      AS defensive_stock_flag,
       CASE WHEN "Beta (1Y)" > 1.3 AND "Beta (2Y)" > 1.3 THEN 1 ELSE 0 END AS high_beta_flag
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
       "Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)" AS net_interest_income,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                AS interest_coverage_ebit,
       "EBITDA (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)              AS interest_coverage_ebitda,
       "Interest Income On Investments (LTM)" / NULLIF("Total Revenues (LTM)", 0) *
       100                                                                     AS interest_income_to_revenue,
       ("Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)") / NULLIF("Total Assets (LTM)", 0) *
       100                                                                     AS net_interest_margin,
       ("Interest Income On Investments (LTM)" + "Gain (Loss) On Sale Of Assets (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0)                               AS non_operating_income_ratio,
       "Operating Income (LTM)" / NULLIF("Operating Income (LTM)" + "Interest Income On Investments (LTM)",
                                         0)                                    AS financial_income_quality,
       "Interest Expense/Total (LTM)" / NULLIF("EBIT (LTM)", 0)                AS interest_burden_ratio
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
SELECT "Ticker"                                                                      AS ticker,
       "TBV (FY)"                                                                    AS tbv_fy,
       "TBV (LTM)"                                                                   AS tbv_ltm,
       "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0), 0)                 AS price_to_tbv,
       "TBV (LTM)" / NULLIF("Shrs Out", 0)                                           AS tbv_per_share,
       ("TBV (LTM)" - "TBV (FY)") / NULLIF(ABS("TBV (FY)"), 0) * 100                 AS tbv_growth_yoy,
       "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0)                                 AS tangible_equity_ratio,
       ("Goodwill (LTM)" + "Gross Intangible Assets (LTM)") / NULLIF("TBV (LTM)", 0) AS intangible_to_tbv_ratio,
       "TBV (LTM)" / NULLIF("Market Cap", 0)                                         AS tbv_vs_market_cap,
       "TBV (LTM)" - "Total Debt (LTM)"                                              AS net_tangible_assets,
       ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0) * 100                   AS tbv_margin_of_safety
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
SELECT "Ticker"                                                                                          AS ticker,
       "Total Current Assets (LTM)"                                                                      AS current_assets_ltm,
       "Total Current Liabilities (LTM)"                                                                 AS current_liabilities_ltm,
       "Total Current Assets (LTM)" - "Total Current Liabilities (LTM)"                                  AS net_working_capital,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0)                                       AS working_capital_to_revenue,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0)                                         AS working_capital_to_assets,
       "Total Current Assets (LTM)" /
       NULLIF("Total Current Liabilities (LTM)", 0)                                                      AS current_ratio,
       ("Total Current Assets (LTM)" - "Inventory (LTM)") / NULLIF("Total Current Liabilities (LTM)", 0) AS quick_ratio,
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0)                       AS cash_ratio,
       ("Total Current Assets (LTM)" /
        NULLIF("Total Operating Expenses (LTM)" / 365, 0))                                               AS defensive_interval,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                                       AS working_capital_turnover,
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
                              ELSE 0 END)
                   ))                                                                                    AS liquidity_score,
       "Total Revenues (LTM)" / NULLIF(ABS("Working Capital (LTM)"), 0)                                  AS working_capital_efficiency
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
SELECT "Ticker"                                  AS ticker,
       "Other Unusual Items/Total (LTM)"         AS other_unusual_items_ltm,
       ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
       ABS("Merger & Restructuring Charges (LTM)") + ABS("Gain (Loss) On Sale Of Assets (LTM)") +
       ABS("Other Unusual Items/Total (LTM)")    AS total_unusual_items,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) / NULLIF("Total Revenues (LTM)", 0) *
       100                                       AS unusual_to_revenue_ratio,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)            AS unusual_to_ebitda_ratio,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
        ABS("Other Unusual Items/Total (LTM)")) /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) AS unusual_to_net_income_ratio,
       CASE
           WHEN ABS("Impairment of Goodwill (LTM)") < 1 AND ABS("Asset Writedown (LTM)") < 1 AND
                ABS("Restructuring Charges (LTM)") < 1 AND ABS("Other Unusual Items/Total (LTM)") < 1 THEN 1
           ELSE 0 END                            AS clean_earnings_flag,
       CASE
           WHEN (CASE WHEN ABS("Impairment of Goodwill (FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Impairment of Goodwill (-1FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Restructuring Charges (FY)") > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS("Restructuring Charges (-1FY)") > 0 THEN 1 ELSE 0 END) >= 3 THEN 1
           ELSE 0 END                            AS recurring_unusual_flag,
       LEAST(100,
             (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
              ABS("Other Unusual Items/Total (LTM)")) / NULLIF(ABS("Net Income - (IS) (LTM)"), 0) *
             100)                                AS earnings_noise_score,
       "Net Income - (IS) (LTM)" + "Impairment of Goodwill (LTM)" + "Asset Writedown (LTM)" +
       "Restructuring Charges (LTM)"             AS quality_adjusted_ni,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)")) /
       NULLIF("Shrs Out", 0)                     AS exceptional_items_impact
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
SELECT "Ticker"                                AS ticker,
       "Revenues - Est Avg (NTM)"              AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"              AS revenue_est_med_ntm,
       "Revenues - Est Avg (FY1E)"             AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"             AS revenue_est_med_fy1e,
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Med (NTM)") / NULLIF("Revenues - Est Med (NTM)", 0) *
       100                                     AS estimate_skew_ntm,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") / NULLIF("Revenues - Est Med (FY1E)", 0) *
       100                                     AS estimate_skew_fy1e,
       GREATEST(0, LEAST(100, 100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                        NULLIF("Revenues - Est Med (FY1E)", 0) *
                                        100))) AS consensus_confidence,
       ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0) *
       100                                     AS upside_to_consensus,
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") / NULLIF("Revenues - Est Avg (FY1E)", 0) *
       100                                     AS estimate_vs_actual_ltm,
       ("Revenues - Est Med (NTM)" - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0) *
       100                                     AS forward_revenue_growth,
       CASE
           WHEN "Total Revenues (LTM)" > "Revenues - Est Avg (FY1E)" THEN 1.0
           ELSE 0.0 END                        AS revenue_beat_history
FROM postgres.public.equities;
$$ LANGUAGE SQL;



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
SELECT "Ticker"                                                             AS ticker,
       "Selling General & Admin Expenses/Total (FQ)" / NULLIF("Total Revenues (FQ)", 0) *
       100                                                                  AS sga_to_revenue_fq,
       "Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0) *
       100                                                                  AS sga_to_revenue_fy,
       ("Selling General & Admin Expenses/Total (FY)" - "Selling General & Admin Expenses/Total (-1FY)") /
       NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0) *
       100                                                                  AS sga_trend_yoy,
       "Selling General & Admin Expenses/Total (FQ)" /
       NULLIF("Selling General & Admin Expenses/Total (5YAVGFQ)", 0)        AS sga_vs_5yavg,
       "Marketing Expenses (FQ)" / NULLIF("Total Revenues (FQ)", 0) * 100   AS marketing_to_revenue_fq,
       "Marketing Expenses (FY)" / NULLIF("Total Revenues (FY)", 0) * 100   AS marketing_to_revenue_fy,
       ("Marketing Expenses (FY)" - "Marketing Expenses (-1FY)") / NULLIF(ABS("Marketing Expenses (-1FY)"), 0) *
       100                                                                  AS marketing_trend_yoy,
       ("Marketing Expenses (FY)" + "Marketing Expenses (-1FY)") / 2 /
       NULLIF("Marketing Expenses (5YAVGLTM)", 0)                           AS marketing_vs_5yavg,
       "Total Operating Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0) AS operating_expense_ratio,
       "Cost Of Revenues (LTM)" / NULLIF("Total Revenues (LTM)", 0)         AS cost_of_revenue_ratio,
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0 AND ABS("Total Operating Expenses (LTM)") > 0
               THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(ABS("Total Revenues (-1FY)"), 0)) /
                    NULLIF((("Selling General & Admin Expenses/Total (FY)" -
                             "Selling General & Admin Expenses/Total (-1FY)") /
                            NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0)), 0)
           END                                                              AS operating_leverage_score,
       ("Selling General & Admin Expenses/Total (-1FY)" / NULLIF("Total Revenues (-1FY)", 0)) -
       ("Selling General & Admin Expenses/Total (FY)" /
        NULLIF("Total Revenues (FY)", 0))                                   AS cost_efficiency_trend
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
       "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0)   AS revenue_fq_vs_5yavg,
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_ltm_vs_5yavg,
       ("Total Revenues (FQ)" * 4 - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0) *
       100                                                             AS revenue_qoq_growth,
       ("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(ABS("Total Revenues (-1FY)"), 0) *
       100                                                             AS revenue_yoy_growth,
       "Total Revenues (FQ)" * 4                                       AS revenue_quarterly_run_rate,
       "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4, 0)   AS revenue_seasonality_factor
FROM postgres.public.equities;
$$ LANGUAGE SQL;

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
    "Ticker"                                                                AS ticker,
    "Name"                                                                  AS name,
    "Sector"                                                                AS sector,
    "Industry"                                                              AS industry,

    -- Revenue Quarterly Features
    "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0)           AS revenue_fq_vs_5yavg,
    ("Total Revenues (FQ)" * 4 - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0) *
    100                                                                     AS revenue_qoq_growth,
    "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4, 0)           AS revenue_seasonality_factor,

    -- Cost Structure Features
    "Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0) *
    100                                                                     AS sga_to_revenue_fy,
    ("Selling General & Admin Expenses/Total (FY)" - "Selling General & Admin Expenses/Total (-1FY)") /
    NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0) *
    100                                                                     AS sga_trend_yoy,
    "Marketing Expenses (FY)" / NULLIF("Total Revenues (FY)", 0) * 100      AS marketing_to_revenue_fy,
    CASE
        WHEN ABS("Total Revenues (-1FY)") > 0 AND ABS("Total Operating Expenses (LTM)") > 0
            THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") / NULLIF(ABS("Total Revenues (-1FY)"), 0)) /
                 NULLIF((("Selling General & Admin Expenses/Total (FY)" -
                          "Selling General & Admin Expenses/Total (-1FY)") /
                         NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0)), 0)
        END                                                                 AS operating_leverage_score,
    ("Selling General & Admin Expenses/Total (-1FY)" / NULLIF("Total Revenues (-1FY)", 0)) -
    ("Selling General & Admin Expenses/Total (FY)" /
     NULLIF("Total Revenues (FY)", 0))                                      AS cost_efficiency_trend,

    -- Tangible Book Features
    "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0), 0)           AS price_to_tbv,
    "TBV (LTM)" / NULLIF("Shrs Out", 0)                                     AS tbv_per_share,
    "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0)                           AS tangible_equity_ratio,
    ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0) * 100             AS tbv_margin_of_safety,

    -- Interest Income Features
    "Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)" AS net_interest_income,
    "EBITDA (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)              AS interest_coverage_ebitda,
    "Operating Income (LTM)" / NULLIF("Operating Income (LTM)" + "Interest Income On Investments (LTM)",
                                      0)                                    AS financial_income_quality,
    "Interest Expense/Total (LTM)" / NULLIF("EBIT (LTM)", 0)                AS interest_burden_ratio,

    -- Long-Term Momentum Features
    ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 100   AS price_momentum_3y,
    ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 100   AS price_momentum_5y,
    (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1) -
    (POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1)      AS momentum_acceleration_3y,
    (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 0.5 +
     ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 0.3 +
     ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 0.2) *
    100                                                                     AS long_term_trend_score,
    CASE
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
             ("Last Price" > "Price (1Y Ago)") AND ("Last Price" > "Price (3Y Ago)") THEN 1.0
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
             ("Last Price" > "Price (1Y Ago)") THEN 0.75
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") THEN 0.5
        WHEN ("Last Price" > "Price (1M Ago)") THEN 0.25
        ELSE 0
        END                                                                 AS momentum_consistency,
    CASE
        WHEN POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1 > 0.10 THEN 1
        ELSE 0 END                                                          AS secular_trend_flag,

    -- Beta Risk Features
    "Beta (2Y)"                                                             AS beta_2y,
    "Beta (1Y)" - "Beta (2Y)"                                               AS beta_trend_short,
    "Beta (2Y)" - "Beta (5Y)"                                               AS beta_trend_long,
    CASE
        WHEN GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") - LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") > 0
            THEN 1.0 / (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") - LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)"))
        ELSE 1.0
        END                                                                 AS beta_stability,
    CASE WHEN ABS("Beta (1Y)" - "Beta (5Y)") > 0.3 THEN 1 ELSE 0 END        AS beta_regime_change,
    "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2               AS systematic_risk_score,
    CASE
        WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8 THEN 1
        ELSE 0 END                                                          AS defensive_stock_flag,
    CASE WHEN "Beta (1Y)" > 1.3 AND "Beta (2Y)" > 1.3 THEN 1 ELSE 0 END     AS high_beta_flag,

    -- Revenue Estimate Consensus
    ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") / NULLIF("Revenues - Est Med (FY1E)", 0) *
    100                                                                     AS estimate_skew_fy1e,
    GREATEST(0, LEAST(100, 100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                     NULLIF("Revenues - Est Med (FY1E)", 0) *
                                     100)))                                 AS revenue_consensus_confidence,
    ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0) *
    100                                                                     AS upside_to_consensus,

    -- Unusual Items Features
    ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
    ABS("Merger & Restructuring Charges (LTM)") + ABS("Gain (Loss) On Sale Of Assets (LTM)") +
    ABS("Other Unusual Items/Total (LTM)")                                  AS total_unusual_items,
    (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
     ABS("Other Unusual Items/Total (LTM)")) /
    NULLIF(ABS("Net Income - (IS) (LTM)"), 0)                               AS unusual_to_net_income_ratio,
    CASE
        WHEN ABS("Impairment of Goodwill (LTM)") < 1 AND ABS("Asset Writedown (LTM)") < 1 AND
             ABS("Restructuring Charges (LTM)") < 1 AND ABS("Other Unusual Items/Total (LTM)") < 1 THEN 1
        ELSE 0 END                                                          AS clean_earnings_flag,
    CASE
        WHEN (CASE WHEN ABS("Impairment of Goodwill (FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Impairment of Goodwill (-1FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Restructuring Charges (FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Restructuring Charges (-1FY)") > 0 THEN 1 ELSE 0 END) >= 3 THEN 1
        ELSE 0 END                                                          AS recurring_unusual_flag,
    LEAST(100,
          (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
           ABS("Other Unusual Items/Total (LTM)")) / NULLIF(ABS("Net Income - (IS) (LTM)"), 0) *
          100)                                                              AS earnings_noise_score,
    "Net Income - (IS) (LTM)" + "Impairment of Goodwill (LTM)" + "Asset Writedown (LTM)" +
    "Restructuring Charges (LTM)"                                           AS quality_adjusted_ni,


    -- Working Capital Deep Features
    ("Total Current Assets (LTM)" /
     NULLIF("Total Operating Expenses (LTM)" / 365, 0))                     AS defensive_interval,
    "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)             AS working_capital_turnover,
    GREATEST(0, LEAST(100,
                      (CASE
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 2 THEN 40
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1.5
                               THEN 30
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1 THEN 20
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
                           ELSE 0 END)
                ))                                                          AS liquidity_score,
    "Total Revenues (LTM)" / NULLIF(ABS("Working Capital (LTM)"), 0)        AS working_capital_efficiency

FROM postgres.public.equities;
$$;

-- =============================================================================
-- SECTION 13F: VIEWS FOR NEW ENHANCED FUNCTIONS
-- =============================================================================

CREATE OR REPLACE VIEW v_long_term_momentum_features AS
SELECT "Ticker"                                                                AS ticker,
       ("Last Price" - "Price (QTD Ago)") / NULLIF("Price (QTD Ago)", 0) * 100 AS price_momentum_qtd,
       ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 100   AS price_momentum_3y,
       ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 100   AS price_momentum_5y,
       (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 0.5 +
        ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 0.3 +
        ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 0.2) *
       100                                                                     AS long_term_trend_score,
       CASE
           WHEN POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1 > 0.10 THEN 1
           ELSE 0 END                                                          AS secular_trend_flag
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_beta_risk_features AS
SELECT "Ticker"                                                                                  AS ticker,
       "Beta (1Y)"                                                                               AS beta_1y,
       "Beta (2Y)"                                                                               AS beta_2y,
       "Beta (5Y)"                                                                               AS beta_5y,
       "Beta (1Y)" - "Beta (2Y)"                                                                 AS beta_trend_short,
       "Beta (2Y)" - "Beta (5Y)"                                                                 AS beta_trend_long,
       "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2                                 AS systematic_risk_score,
       CASE WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8 THEN 1 ELSE 0 END AS defensive_stock_flag
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_interest_income_features AS
SELECT "Ticker"                                                                AS ticker,
       "Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)" AS net_interest_income,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                AS interest_coverage_ebit,
       "EBITDA (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)              AS interest_coverage_ebitda,
       "Operating Income (LTM)" / NULLIF("Operating Income (LTM)" + "Interest Income On Investments (LTM)",
                                         0)                                    AS financial_income_quality
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_tangible_book_features AS
SELECT "Ticker"                                                      AS ticker,
       "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0), 0) AS price_to_tbv,
       "TBV (LTM)" / NULLIF("Shrs Out", 0)                           AS tbv_per_share,
       "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0)                 AS tangible_equity_ratio,
       ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0) * 100   AS tbv_margin_of_safety
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_working_capital_deep_features AS
SELECT "Ticker"                                                                                          AS ticker,
       "Total Current Assets (LTM)" - "Total Current Liabilities (LTM)"                                  AS net_working_capital,
       "Total Current Assets (LTM)" /
       NULLIF("Total Current Liabilities (LTM)", 0)                                                      AS current_ratio,
       ("Total Current Assets (LTM)" - "Inventory (LTM)") / NULLIF("Total Current Liabilities (LTM)", 0) AS quick_ratio,
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0)                       AS cash_ratio,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                                       AS working_capital_turnover
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_unusual_items_features AS
SELECT "Ticker"                               AS ticker,
       ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)") +
       ABS("Merger & Restructuring Charges (LTM)") + ABS("Gain (Loss) On Sale Of Assets (LTM)") +
       ABS("Other Unusual Items/Total (LTM)") AS total_unusual_items,
       CASE
           WHEN ABS("Impairment of Goodwill (LTM)") < 1 AND ABS("Asset Writedown (LTM)") < 1 AND
                ABS("Restructuring Charges (LTM)") < 1 AND ABS("Other Unusual Items/Total (LTM)") < 1 THEN 1
           ELSE 0 END                         AS clean_earnings_flag
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_revenue_estimate_consensus AS
SELECT "Ticker"                                AS ticker,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") / NULLIF("Revenues - Est Med (FY1E)", 0) *
       100                                     AS estimate_skew_fy1e,
       GREATEST(0, LEAST(100, 100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                        NULLIF("Revenues - Est Med (FY1E)", 0) *
                                        100))) AS consensus_confidence,
       ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0) *
       100                                     AS upside_to_consensus
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_cost_structure_features AS
SELECT "Ticker"                                                                               AS ticker,
       "Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0) * 100 AS sga_to_revenue_fy,
       "Total Operating Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)                   AS operating_expense_ratio,
       "Cost Of Revenues (LTM)" / NULLIF("Total Revenues (LTM)", 0)                           AS cost_of_revenue_ratio
FROM postgres.public.equities;

CREATE OR REPLACE VIEW v_revenue_quarterly_features AS
SELECT "Ticker"                                                      AS ticker,
       "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0) AS revenue_fq_vs_5yavg,
       ("Total Revenues (FQ)" * 4 - "Total Revenues (LTM)") / NULLIF("Total Revenues (LTM)", 0) *
       100                                                           AS revenue_qoq_growth,
       "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4, 0) AS revenue_seasonality_factor
FROM postgres.public.equities;

-- =============================================================================
-- SECTION 14: FEATURE REGISTRY METADATA TABLE (ENHANCED)
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

-- Upsert all function metadata (original + new)
INSERT INTO feature_registry_metadata (function_name, category, feature_count, description, python_equivalent,
                                       updated_at)
VALUES
    -- Original functions
    ('calc_valuation_features', 'Valuation Ratios', 7, 'P/E, P/B, EV/EBITDA, EV/Sales, PEG ratios',
     'engineer_valuation_ratios', CURRENT_TIMESTAMP),
    ('calc_valuation_timeseries_features', 'Valuation Timeseries', 11,
     'Valuation momentum, mean reversion, forward discount', 'engineer_valuation_timeseries_features',
     CURRENT_TIMESTAMP),
    ('calc_momentum_features', 'Momentum & Technical', 14, 'Price momentum, EMA crossovers, 52W range, beta',
     'engineer_momentum_features', CURRENT_TIMESTAMP),
    ('calc_profitability_features', 'Profitability', 9, 'ROE, ROA, margins, ROIC, DuPont',
     'engineer_profitability_ratios', CURRENT_TIMESTAMP),
    ('calc_quality_features', 'Quality & Risk', 10, 'Impairments, goodwill, Z-score, liquidity',
     'engineer_accounting_quality_features', CURRENT_TIMESTAMP),
    ('calc_leverage_features', 'Leverage & Liquidity', 7, 'Debt ratios, coverage, working capital',
     'engineer_leverage_ratios', CURRENT_TIMESTAMP),
    ('calc_sentiment_features', 'Analyst Sentiment', 10, 'Ratings, price targets, revisions',
     'engineer_analyst_quality_features', CURRENT_TIMESTAMP),
    ('calc_earnings_features', 'Earnings Quality', 7, 'Surprises, adjustments, GAAP vs non-GAAP',
     'engineer_estimated_vs_actual_analytics', CURRENT_TIMESTAMP),
    ('calc_growth_features', 'Growth Metrics', 7, 'Revenue, EBITDA, FCF growth', 'engineer_growth_metrics',
     CURRENT_TIMESTAMP),
    ('calc_dividend_features', 'Dividend Reliability', 8, 'Streak, yield, payout, coverage',
     'engineer_dividend_reliability_features', CURRENT_TIMESTAMP),
    ('calc_employment_features', 'Employee Productivity', 7, 'Per-employee metrics, FTE growth',
     'engineer_employee_productivity_features', CURRENT_TIMESTAMP),
    ('calc_cashflow_features', 'Cash Flow', 7, 'CFO/NI, FCF margin, self-funding',
     'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),
    ('calc_temporal_features', 'Temporal Patterns', 7, 'Fiscal calendar, earnings timing',
     'engineer_temporal_features', CURRENT_TIMESTAMP),
    -- NEW functions
    ('calc_technical_analysis_features', 'Technical Analysis', 11,
     'EMA trends, breakout signals, volume momentum, volatility compression',
     'engineer_technical_analysis_features', CURRENT_TIMESTAMP),
    ('calc_financial_distress_features', 'Financial Distress', 9,
     'Distress risk score, liquidity stress, working capital trend, cash runway',
     'engineer_financial_distress_features', CURRENT_TIMESTAMP),
    ('calc_accounting_quality_features', 'Accounting Quality', 7,
     'Goodwill changes, restructuring intensity, exceptional items, quality score',
     'engineer_accounting_quality_features', CURRENT_TIMESTAMP),
    ('calc_price_target_dynamics', 'Price Target Dynamics', 15,
     'PT momentum (1W-1Y), consensus convergence, analyst coverage changes',
     'engineer_price_target_dynamics', CURRENT_TIMESTAMP),
    ('calc_eps_trajectory_features', 'EPS Trajectory', 10,
     'EPS growth rates, CAGR, positive streak, trajectory score, stability',
     'engineer_eps_trajectory_features', CURRENT_TIMESTAMP),
    ('calc_gaap_adjusted_analytics', 'GAAP vs Adjusted', 8,
     'EPS/NI adjustment spreads and ratios, earnings quality score',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_enhanced_cashflow_features', 'Enhanced Cash Flow', 12,
     'FCF consistency, CapEx efficiency, M&A sustainability, self-funding',
     'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),
    ('calc_composite_scores', 'Composite Scores', 4,
     'Piotroski F-Score, EPS trajectory, dilution score, quality-momentum',
     'engineer_composite_scores', CURRENT_TIMESTAMP),
    -- NEW (Phase 9.3 parity)
    ('calc_margin_trends', 'Profitability', 6,
     'Gross, operating, net, and EBITDA margin trends (LTM vs FY), expansion flag',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_efficiency_ratios', 'Efficiency', 4,
     'Asset and inventory turnover, receivables days, working capital turns',
     'engineer_efficiency_ratios', CURRENT_TIMESTAMP),
    ('calc_balance_sheet_dynamics', 'Balance Sheet', 13,
     'Cash trends, inventory/receivables vs 5Y avg, asset quality, BS strength',
     'engineer_balance_sheet_trends', CURRENT_TIMESTAMP),
    ('calc_revenue_forecast_features', 'Revenue Forecasting', 12,
     'Estimate spread, beat potential, revision trend, forward multiples',
     'engineer_revenue_forecast_features', CURRENT_TIMESTAMP),
    ('calc_employment_dynamics', 'Employment Dynamics', 10,
     'FTE growth, acceleration, hiring intensity, productivity trends',
     'engineer_employment_dynamics_features', CURRENT_TIMESTAMP),
    ('calc_dividend_timing', 'Dividend Reliability', 8,
     'Days to ex-date/payment, announcement flag, consistency, yield vs 5Y avg',
     'engineer_dividend_timing_features', CURRENT_TIMESTAMP),
    ('calc_fiscal_calendar_features', 'Temporal Patterns', 9,
     'Days since report, quarter/FY end flags, earnings season, freshness score',
     'engineer_fiscal_calendar_features', CURRENT_TIMESTAMP),
    ('calc_cashflow_temporal_features', 'Cash Flow', 12,
     'Quarterly trends (CFO/CFI/CFF/FCF), burn rate, volatility, momentum',
     'engineer_cashflow_temporal_features', CURRENT_TIMESTAMP),
    ('calc_gaap_revision_features', 'Earnings Quality', 9,
     'GAAP EPS revision momentum (1M-1Y), spread vs normalized, acceleration',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_extended_valuation_timeseries', 'Valuation Timeseries', 11,
     'QoQ multiple trends (EV/Sales, EV/EBITDA), mean reversion, P/B momentum',
     'engineer_valuation_timeseries_features', CURRENT_TIMESTAMP),
    -- Comprehensive functions (Phase 9.3+)
    ('calc_ebit_ebitda_comprehensive', 'Profitability', 74,
     'Raw and adjusted EBIT/EBITDA for all periods (FQ, LTM, FY, historical), growth and margins',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_net_income_comprehensive', 'Earnings Quality', 58,
     'GAAP, Adjusted, and Normalized Net Income for all periods, growth and quality metrics',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_quality_features_comprehensive', 'Accounting Quality', 79,
     'Detailed impairments, writedowns, restructuring, and gain/loss across all periods',
     'engineer_accounting_quality_features', CURRENT_TIMESTAMP),
    ('calc_eps_comprehensive', 'Earnings Quality', 61,
     'Basic, Continuing, and Adjusted EPS across all periods, growth, CAGR, and trajectory',
     'engineer_eps_trajectory_features', CURRENT_TIMESTAMP),
    ('calc_cashflow_comprehensive', 'Cash Flow', 78,
     'CFO, CFI, CFF, and FCF for all periods, acquisitions, CapEx, and quality metrics',
     'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),
    -- NEW Enhanced Feature Functions (feature_registry_enhancements)
    ('calc_interest_income_features', 'Interest Income', 10,
     'Net interest income, coverage ratios (EBIT/EBITDA), income quality, burden ratio',
     'engineer_interest_income_features', CURRENT_TIMESTAMP),
    ('calc_long_term_momentum_features', 'Momentum & Technical', 10,
     '3Y/5Y momentum, CAGR acceleration, weighted trend score, consistency flag',
     'engineer_long_term_momentum_features', CURRENT_TIMESTAMP),
    ('calc_tangible_book_features', 'Valuation Ratios', 10,
     'Price-to-TBV, TBV per share, tangible equity ratio, margin of safety',
     'engineer_tangible_book_features', CURRENT_TIMESTAMP),
    ('calc_beta_risk_features', 'Quality & Risk', 10,
     'Multi-period betas, trend analysis, stability, regime change detection',
     'engineer_beta_risk_features', CURRENT_TIMESTAMP),
    ('calc_working_capital_deep_features', 'Leverage & Liquidity', 12,
     'Current/quick/cash ratios, defensive interval, liquidity score (0-100)',
     'engineer_working_capital_deep_features', CURRENT_TIMESTAMP),
    ('calc_unusual_items_features', 'Earnings Quality', 10,
     'Total unusual items, noise score, clean earnings flag, quality-adjusted NI',
     'engineer_unusual_items_features', CURRENT_TIMESTAMP),
    ('calc_revenue_estimate_consensus', 'Revenue Forecasting', 11,
     'Estimate skew, consensus confidence, upside to consensus, beat history',
     'engineer_revenue_estimate_consensus', CURRENT_TIMESTAMP),
    ('calc_enhanced_valuation_ratios', 'Valuation Ratios', 12,
     'PEG variants (adjusted/forward), yields (earnings/FCF/shareholder), composite score',
     'engineer_enhanced_valuation_ratios', CURRENT_TIMESTAMP),
    ('calc_cost_structure_features', 'Efficiency Ratios', 12,
     'SG&A trends, marketing intensity, operating leverage, cost efficiency',
     'engineer_cost_structure_features', CURRENT_TIMESTAMP),
    ('calc_revenue_quarterly_features', 'Revenue Forecasting', 12,
     'Quarterly revenue trends, seasonality factor, run rate, vs 5Y average',
     'engineer_revenue_quarterly_features', CURRENT_TIMESTAMP),
    ('calc_all_enhanced_features', 'Composite', 53,
     'Comprehensive aggregation of all enhanced features across categories',
     'engineer_all_enhanced_features', CURRENT_TIMESTAMP)
ON CONFLICT (function_name) DO UPDATE SET category          = EXCLUDED.category,
                                          feature_count     = EXCLUDED.feature_count,
                                          description       = EXCLUDED.description,
                                          python_equivalent = EXCLUDED.python_equivalent,
                                          updated_at        = CURRENT_TIMESTAMP;

COMMIT;

-- Refresh table statistics for optimal query planning
ANALYZE feature_registry_metadata;

-- =============================================================================
-- SECTION 15: USAGE EXAMPLES
-- =============================================================================

-- Example 1: Get valuation features for all stocks
-- SELECT * FROM calc_valuation_features() WHERE p_e_ratio > 0 ORDER BY p_e_ratio;

-- Example 2: Screen for undervalued momentum stocks
-- SELECT v.ticker, v.p_e_ratio, v.ev_ebitda_ratio, m.price_momentum_3m, m.ema_crossover_20_50
-- FROM calc_valuation_features() v
-- JOIN calc_momentum_features() m ON v.ticker = m.ticker
-- WHERE v.p_e_ratio < 15 AND m.price_momentum_3m > 5 AND m.ema_crossover_20_50 = 1;

-- Example 3: Use the comprehensive view
-- SELECT * FROM vw_feature_registry WHERE "Sector" = 'Technology' ORDER BY revenue_growth_yoy DESC;

-- Example 4: Quality screen with multiple factors
-- SELECT ticker, altman_z_score, debt_to_equity, cfo_to_net_income
-- FROM calc_quality_features() q
-- JOIN calc_leverage_features() l USING (ticker)
-- JOIN calc_cashflow_features() c USING (ticker)
-- WHERE altman_z_score > 3 AND debt_to_equity < 1 AND cfo_to_net_income > 1;

-- Example 5: NEW - High-quality momentum breakout candidates
-- SELECT t.ticker, t.breakout_signal, t.ema_trend_consistency, t.volume_momentum_score,
--        c.piotroski_f_score, p.pt_momentum_3m
-- FROM calc_technical_analysis_features() t
-- JOIN calc_composite_scores() c USING (ticker)
-- JOIN calc_price_target_dynamics() p USING (ticker)
-- WHERE t.breakout_signal = 1 AND c.piotroski_f_score >= 7 AND p.pt_momentum_3m > 0;

-- Example 6: NEW - Distress screening (avoid troubled companies)
-- SELECT d.ticker, d.distress_risk_score, d.liquidity_stress_score, d.cash_runway_months,
--        q.accounting_quality_score
-- FROM calc_financial_distress_features() d
-- JOIN calc_accounting_quality_features() q USING (ticker)
-- WHERE d.distress_risk_score < 50 OR d.liquidity_stress_score > 15
-- ORDER BY d.distress_risk_score;

-- Example 7: NEW - Earnings quality analysis
-- SELECT g.ticker, g.eps_adjustment_pct, g.earnings_quality_score, g.earnings_quality_warning,
--        e.eps_trajectory_score, e.eps_positive_streak
-- FROM calc_gaap_adjusted_analytics() g
-- JOIN calc_eps_trajectory_features() e USING (ticker)
-- WHERE g.earnings_quality_warning = 0 AND e.eps_trajectory_score > 60;

-- Example 8: NEW - Multi-factor quality screen
-- SELECT v.ticker, 
--        c.piotroski_f_score,
--        cf.cash_flow_quality_score,
--        g.earnings_quality_score,
--        d.distress_risk_score
-- FROM v_valuation_features v
-- JOIN calc_composite_scores() c USING (ticker)
-- JOIN calc_enhanced_cashflow_features() cf USING (ticker)
-- JOIN calc_gaap_adjusted_analytics() g USING (ticker)
-- JOIN calc_financial_distress_features() d USING (ticker)
-- WHERE c.piotroski_f_score >= 7
--   AND cf.cash_flow_quality_score >= 75
--   AND g.earnings_quality_score >= 80
--   AND d.distress_risk_score >= 60
-- ORDER BY c.piotroski_f_score DESC, cf.cash_flow_quality_score DESC;