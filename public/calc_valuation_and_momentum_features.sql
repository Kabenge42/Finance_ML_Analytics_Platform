-- =============================================================================
-- HELPER FUNCTIONS: Reusable calculation patterns
-- =============================================================================

-- Safe percentage change calculation (handles NULL and zero division)
CREATE OR REPLACE FUNCTION safe_pct_change(current_val NUMERIC, previous_val NUMERIC)
    RETURNS NUMERIC
    LANGUAGE SQL
    IMMUTABLE
AS
$$
SELECT (current_val - previous_val) / NULLIF(ABS(previous_val), 0) * 100;
$$;

-- Safe ratio calculation (handles NULL and zero division)
CREATE OR REPLACE FUNCTION safe_ratio(numerator NUMERIC, denominator NUMERIC)
    RETURNS NUMERIC
    LANGUAGE SQL
    IMMUTABLE
AS
$$
SELECT numerator / NULLIF(denominator, 0);
$$;

-- Boolean flag as integer (1 for true, 0 for false)
CREATE OR REPLACE FUNCTION bool_to_flag(condition BOOLEAN)
    RETURNS INTEGER
    LANGUAGE SQL
    IMMUTABLE
AS
$$
SELECT CASE WHEN condition THEN 1 ELSE 0 END;
$$;

-- Count positive values in a series (for consistency metrics)
CREATE OR REPLACE FUNCTION count_positive(VARIADIC vals NUMERIC[])
    RETURNS INTEGER
    LANGUAGE SQL
    IMMUTABLE
AS
$$
SELECT COUNT(*)::INTEGER
FROM UNNEST(vals) AS v
WHERE v > 0;
$$;

-- Count improvements (value > previous value) in a series
CREATE OR REPLACE FUNCTION count_improvements(VARIADIC vals NUMERIC[])
    RETURNS INTEGER
    LANGUAGE SQL
    IMMUTABLE
AS
$$
SELECT COUNT(*)::INTEGER
FROM (SELECT v, LAG(v) OVER (ORDER BY ordinality) AS prev_v
      FROM UNNEST(vals) WITH ORDINALITY AS t(v, ordinality)) sub
WHERE v > prev_v;
$$;

-- Clamp value to range [min_val, max_val]
CREATE OR REPLACE FUNCTION clamp(value NUMERIC, min_val NUMERIC, max_val NUMERIC)
    RETURNS NUMERIC
    LANGUAGE SQL
    IMMUTABLE
AS
$$
SELECT GREATEST(min_val, LEAST(max_val, value));
$$;

-- =============================================================================
-- SECTION 1: VALUATION FEATURES (Refactored)
-- =============================================================================

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       "P/E (LTM)",
       "P/B (LTM)",
       "EV/EBITDA (LTM)",
       "EV/Sales (LTM)",
       "Div Yield (LTM)",
       safe_ratio("P/E (LTM)", "Total Revenues/CAGR (5Y FY)")
FROM postgres.public.equities;
$$;

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       safe_pct_change("EV/Sales (LTM)", "EV/Sales (-1FYLTM)") / 100,
       safe_pct_change("EV/EBITDA (LTM)", "EV/EBITDA (-1FYLTM)") / 100,
       safe_pct_change("P/E (LTM)", "P/E (-1FYLTM)") / 100,
       safe_pct_change("P/E (LTM)", "P/E (-1FQLTM)") / 100,
       safe_pct_change("EV/Sales (LTM)", "EV/Sales (3YAVGLTM)") / 100,
       safe_pct_change("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)") / 100,
       safe_pct_change("P/E (LTM)", "P/E (3YAVGLTM)") / 100,
       safe_pct_change("EV/Sales (NTM)", "EV/Sales (LTM)") / 100,
       safe_pct_change("EV/EBITDA (NTM)", "EV/EBITDA (LTM)") / 100,
       safe_pct_change("P/E (EST FY1)", "P/E (LTM)") / 100,
       safe_ratio("P/B (LTM)", "P/B (5YAVG)")
FROM postgres.public.equities;
$$;

-- =============================================================================
-- SECTION 2: MOMENTUM & TECHNICAL FEATURES (Refactored)
-- =============================================================================

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       safe_pct_change("Last Price", "Price (1M Ago)"),
       safe_pct_change("Last Price", "Price (3M Ago)"),
       safe_pct_change("Last Price", "Price (6M Ago)"),
       safe_pct_change("Last Price", "Price (1Y Ago)"),
       safe_pct_change("Last Price", "Price (5D Ago)"),
       -- EMA Crossovers
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" THEN -1
           ELSE 0
           END,
       CASE
           WHEN "EMA (50D)" > "EMA (250D)" THEN 1
           WHEN "EMA (50D)" < "EMA (250D)" THEN -1
           ELSE 0
           END,
       safe_ratio("Last Price" - "EMA (20D)", "EMA (20D)"),
       safe_ratio("Last Price" - "EMA (250D)", "EMA (250D)"),
       safe_ratio("52W High/Adj" - "Last Price", "52W High/Adj"),
       safe_ratio("Last Price" - "52W Low/Adj", "52W Low/Adj"),
       clamp(safe_ratio("Last Price" - "52W Low/Adj", "52W High/Adj" - "52W Low/Adj"), 0, 1),
       "Beta (1Y)" - "Beta (5Y)",
       safe_ratio("Volatility (1M)", "Volatility (1Y)")
FROM postgres.public.equities;
$$;

-- =============================================================================
-- SECTION 3: PROFITABILITY FEATURES (Refactored)
-- =============================================================================

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       "Return On Equity % (LTM)",
       "Return on Assets (ROA) % (LTM)",
       "Gross Profit Margin % (LTM)",
       safe_ratio("Operating Income (LTM)", "Total Revenues (LTM)") * 100,
       "Net Income Margin % (LTM)",
       safe_ratio("EBITDA (LTM)", "Total Revenues (LTM)") * 100,
       safe_ratio("Net Income - (IS) (LTM)", "Total Equity (LTM)" + "Total Debt (LTM)") * 100,
       safe_ratio("R&D Expenses (LTM)", "Total Revenues (LTM)"),
       safe_ratio("Total Assets (LTM)", "Total Equity (LTM)")
FROM postgres.public.equities;
$$;

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       "Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)",
       (safe_ratio("Operating Income (LTM)", "Total Revenues (LTM)") -
        safe_ratio("Operating Income (FY)", "Total Revenues (FY)")) * 100,
       "Net Income Margin % (LTM)" - "Net Income Margin % (FY)",
       (safe_ratio("EBITDA (LTM)", "Total Revenues (LTM)") -
        safe_ratio("EBITDA (FY)", "Total Revenues (FY)")) * 100,
       bool_to_flag(
               "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
                   AND "Net Income Margin % (LTM)" > "Net Income Margin % (FY)"
                   AND safe_ratio("EBITDA (LTM)", "Total Revenues (LTM)") >
                       safe_ratio("EBITDA (FY)", "Total Revenues (FY)")
       ),
       clamp(
               100 - (ABS("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)") +
                      ABS("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")) / 2,
               0, 100
       )
FROM postgres.public.equities;
$$;

-- =============================================================================
-- SECTION 4: QUALITY & RISK FEATURES (Refactored)
-- =============================================================================

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       bool_to_flag("Impairment of Goodwill (LTM)" <> 0),
       bool_to_flag("Asset Writedown (LTM)" <> 0),
       bool_to_flag("Restructuring Charges (LTM)" <> 0),
       safe_ratio("Goodwill (LTM)", "Total Assets (LTM)") * 100,
       safe_ratio("Gross Intangible Assets (LTM)", "Total Assets (LTM)"),
       safe_ratio(
               ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") + ABS("Restructuring Charges (LTM)"),
               ABS("EBITDA (LTM)")
       ),
       "Altman Z-Score (LTM)",
       "Altman Z-Score (FY)" - "Altman Z-Score (LTM)",
       "Current Ratio (LTM)",
       safe_ratio("Total Current Assets (LTM)" - "Inventory (LTM)", "Total Current Liabilities (LTM)")
FROM postgres.public.equities;
$$;

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       clamp(safe_ratio("Altman Z-Score (LTM)" - 1.8, 3.0 - 1.8) * 100, 0, 100),
       CASE
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
           ELSE 0.0
           END,
       safe_pct_change("Working Capital (FQ)", "Working Capital (FY)") / 100,
       safe_ratio("Cash And Equivalents (FQ)", "Total Operating Expenses (LTM)" / 12.0),
       clamp(
               safe_ratio("Altman Z-Score (LTM)" - 1.8, 3.0 - 1.8) * 70 +
               (100 - CASE
                          WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
                          WHEN "Current Ratio (LTM)" < 1.5 THEN 15.0
                          ELSE 0.0
                   END) * 0.30,
               0, 100
       ),
       bool_to_flag(safe_pct_change("Working Capital (FQ)", "Working Capital (FY)") / 100 < -0.2),
       safe_pct_change("Retained Earnings (FQ)", "Retained Earnings (FY)") / 100,
       bool_to_flag("Retained Earnings (FQ)" < 0),
       bool_to_flag(safe_ratio("Cash And Equivalents (FQ)", "Total Operating Expenses (LTM)" / 12.0) > 6)
FROM postgres.public.equities;
$$;

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
    LANGUAGE SQL
AS
$$
SELECT "Ticker",
       safe_pct_change("Goodwill (LTM)", "Goodwill (-1FY)") / 100,
       safe_ratio("Restructuring Charges (LTM)", "Total Assets (LTM)"),
       (bool_to_flag(ABS("Impairment of Goodwill (FQ)") > 0) +
        bool_to_flag(ABS("Asset Writedown (FQ)") > 0) +
        bool_to_flag(ABS("Restructuring Charges (FQ)") > 0)),
       safe_ratio("Merger & Restructuring Charges (LTM)", "Market Cap"),
       safe_ratio("Interest Income On Investments (LTM)", ABS("Net Income - (IS) (LTM)")),
       bool_to_flag("Gain (Loss) On Sale Of Assets (LTM)" > 0),
       clamp(
               100 -
               bool_to_flag("Impairment of Goodwill (LTM)" <> 0) * 25 -
               bool_to_flag("Asset Writedown (LTM)" <> 0) * 10 -
               bool_to_flag("Restructuring Charges (LTM)" <> 0) * 15 -
               bool_to_flag(safe_ratio("Goodwill (LTM)", "Total Assets (LTM)") > 0.30) * 15 -
               bool_to_flag(
                       safe_ratio(
                               ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
                               ABS("Restructuring Charges (LTM)"),
                               ABS("Net Income - (IS) (LTM)")
                       ) > 0.10
               ) * 15,
               0, 100
       )
FROM postgres.public.equities;
$$;
