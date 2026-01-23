-- ===================================================================
-- Equities Data Import Script
-- ===================================================================
-- Documentation: See docs/column_mapping_reference.md for column aliases
-- Usage: psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
\echo 'Starting equities data import...'

-- ===================================================================
-- SESSION-LEVEL TUNING FOR BULK IMPORT
-- ===================================================================
-- These settings optimize PostgreSQL for bulk data import operations.
-- They will be reset when the session ends.
-- WARNING: synchronous_commit = OFF should only be used for imports, not production!

SET work_mem = '256MB'; -- Increase memory for sorting/hashing operations
SET maintenance_work_mem = '512MB'; -- Increase memory for maintenance operations
SET synchronous_commit = OFF; -- Defer WAL writes (faster, but less durable during import)
SET checkpoint_completion_target = 0.9; -- Spread checkpoint I/O over longer period

\echo 'Session tuning applied for bulk import optimization.'

DO
$$
    BEGIN
        RAISE NOTICE 'Import started at %', NOW();
    END
$$;

-- Show current table status
SELECT 'Current equities table row count:' AS status, COUNT(*) AS row_count
FROM equities;

-- ===================================================================
-- HELPER FUNCTIONS
-- ===================================================================

-- Converts TEXT to NUMERIC, treating common non-numeric patterns as NULL
CREATE OR REPLACE FUNCTION text_to_numeric_safe(input_text TEXT)
    RETURNS NUMERIC AS
$$
SELECT CASE
           WHEN input_text IS NULL
               OR TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE', 'n/a', 'na', 'null', 'none')
               THEN NULL
           WHEN TRIM(input_text) ~ '^-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
               THEN TRIM(input_text)::NUMERIC
           END AS result
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

-- Converts TEXT to DATE safely, returns NULL for invalid input
CREATE OR REPLACE FUNCTION text_to_date_safe(input_text TEXT, date_format TEXT DEFAULT 'YYYY-MM-DD')
    RETURNS DATE AS
$$
BEGIN
    IF input_text IS NULL OR TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE') THEN
        RETURN NULL;
    END IF;
    RETURN TO_DATE(TRIM(input_text), date_format);
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Parse FY End to Date
-- ===================================================================
-- Converts "FY End" text (e.g., "Dec 2024", "Mar 2025") to a DATE
-- Returns the last day of the specified month/year

CREATE OR REPLACE FUNCTION parse_fiscal_year_end_date(fy_end_text TEXT)
    RETURNS DATE AS
$$
DECLARE
    month_name TEXT;
    year_text  TEXT;
    month_num  INTEGER;
BEGIN
    IF fy_end_text IS NULL OR TRIM(fy_end_text) = '' THEN
        RETURN NULL;
    END IF;

    -- Handle various input formats
    fy_end_text := TRIM(fy_end_text);

    -- Try "Mon YYYY" format first
    month_name := SPLIT_PART(fy_end_text, ' ', 1);
    year_text := SPLIT_PART(fy_end_text, ' ', 2);

    -- Validate year is numeric and reasonable
    IF year_text !~ '^\d{4}$' OR year_text::INTEGER < 1900 OR year_text::INTEGER > 2100 THEN
        RETURN NULL;
    END IF;

    month_num := CASE UPPER(LEFT(month_name, 3))
                     WHEN 'JAN' THEN 1
                     WHEN 'FEB' THEN 2
                     WHEN 'MAR' THEN 3
                     WHEN 'APR' THEN 4
                     WHEN 'MAY' THEN 5
                     WHEN 'JUN' THEN 6
                     WHEN 'JUL' THEN 7
                     WHEN 'AUG' THEN 8
                     WHEN 'SEP' THEN 9
                     WHEN 'OCT' THEN 10
                     WHEN 'NOV' THEN 11
                     WHEN 'DEC' THEN 12
        END;

    IF month_num IS NULL THEN
        RETURN NULL;
    END IF;

    -- Last day of month using cleaner syntax
    RETURN (MAKE_DATE(year_text::INTEGER, month_num, 1) + INTERVAL '1 month - 1 day')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Convert Frequency to Interval Months
-- ===================================================================
-- Centralizes the mapping from frequency text to number of months
-- Quarterly = 3, Semi-Annual = 6, Annual = 12

CREATE OR REPLACE FUNCTION frequency_to_months(earnings_report_frequency TEXT)
    RETURNS INTEGER AS
$$
BEGIN
    RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'Quarterly')))
               WHEN 'QUARTERLY' THEN 3
               WHEN 'SEMI-ANNUALLY' THEN 6
               WHEN 'SEMI-ANNUAL' THEN 6
               WHEN 'ANNUALLY' THEN 12
               WHEN 'ANNUAL' THEN 12
               ELSE 3 -- Default to quarterly
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Convert Interval Months to Frequency Text
-- ===================================================================
-- Converts number of months to frequency description

CREATE OR REPLACE FUNCTION months_to_frequency(interval_months INTEGER)
    RETURNS TEXT AS
$$
BEGIN
    RETURN CASE
               WHEN interval_months <= 3 THEN 'Quarterly'
               WHEN interval_months <= 6 THEN 'Semi-Annually'
               WHEN interval_months <= 12 THEN 'Annually'
               ELSE 'Quarterly' -- Default
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Reporting Interval
-- ===================================================================
-- Returns the reporting interval in months based on frequency
-- Quarterly = 3, Semi-Annual = 6, Annual = 12

CREATE OR REPLACE FUNCTION calculate_reporting_interval(earnings_report_frequency TEXT)
    RETURNS INTEGER AS
$$
BEGIN
    RETURN frequency_to_months(earnings_report_frequency);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Derive Earnings Report Frequency
-- ===================================================================
-- Derives frequency from the interval between two report dates
-- or from the reporting interval value

CREATE OR REPLACE FUNCTION derive_earnings_report_frequency(
    income_statement_report_date DATE,
    fy_end_date DATE
)
    RETURNS TEXT AS
$$
DECLARE
    months_diff INTEGER;
BEGIN
    IF income_statement_report_date IS NULL OR fy_end_date IS NULL THEN
        RETURN 'Quarterly'; -- Default
    END IF;

    -- Calculate months between report date and FY end
    months_diff := ABS(
            (EXTRACT(YEAR FROM income_statement_report_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
                + (EXTRACT(MONTH FROM income_statement_report_date) - EXTRACT(MONTH FROM fy_end_date))
                   );

    -- Normalize to reporting period (mod 12, then check pattern)
    months_diff := months_diff % 12;
    IF months_diff = 0 THEN
        months_diff := 12;
    END IF;

    -- Determine frequency based on the reporting pattern
    -- If months align with 3-month intervals: Quarterly
    -- If months align with 6-month intervals: Semi-Annually
    -- If months align with 12-month intervals: Annually
    RETURN CASE
               WHEN months_diff IN (3, 6, 9, 12) AND months_diff % 3 = 0 AND months_diff % 6 != 0 THEN 'Quarterly'
               WHEN months_diff IN (6, 12) AND months_diff % 6 = 0 AND months_diff != 12 THEN 'Semi-Annually'
               ELSE 'Quarterly' -- Default for edge cases
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- Unified Fiscal Date Calculator
-- ===================================================================
-- Returns all fiscal metrics including frequency-aware calculations
-- earnings_report_frequency is now a primary input that drives calculations

CREATE OR REPLACE FUNCTION calculate_fiscal_info(
    reference_date DATE,
    fy_end_date DATE,
    input_earnings_frequency TEXT DEFAULT NULL,
    OUT fiscal_month INTEGER,
    OUT fiscal_quarter INTEGER,
    OUT fiscal_year INTEGER,
    OUT next_quarter INTEGER,
    OUT next_quarter_year INTEGER,
    OUT reporting_interval INTEGER,
    OUT earnings_report_frequency TEXT,
    OUT next_earnings_report_type TEXT
) AS
$$
DECLARE
    months_since_fy_end INTEGER;
    interval_months   INTEGER;
    quarter_increment INTEGER;
BEGIN
    IF reference_date IS NULL OR fy_end_date IS NULL THEN
        RETURN;
    END IF;

    -- Determine earnings frequency (use input if provided, otherwise derive)
    IF input_earnings_frequency IS NOT NULL AND TRIM(input_earnings_frequency) != '' THEN
        earnings_report_frequency := input_earnings_frequency;
    ELSE
        earnings_report_frequency := derive_earnings_report_frequency(reference_date, fy_end_date);
    END IF;

    -- Get interval months from frequency
    interval_months := frequency_to_months(earnings_report_frequency);
    reporting_interval := interval_months;

    -- Calculate months since fiscal year end
    months_since_fy_end := (EXTRACT(YEAR FROM reference_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
        + (EXTRACT(MONTH FROM reference_date) - EXTRACT(MONTH FROM fy_end_date));

    -- Fiscal month (1-12)
    fiscal_month := ((months_since_fy_end - 1) % 12) + 1;
    IF fiscal_month <= 0 THEN
        fiscal_month := fiscal_month + 12;
    END IF;

    -- Current fiscal quarter
    fiscal_quarter := CEIL(fiscal_month / 3.0)::INTEGER;

    -- Calculate next quarter based on frequency
    quarter_increment := CASE
                             WHEN interval_months = 3 THEN 1 -- Quarterly: next quarter
                             WHEN interval_months = 6 THEN 2 -- Semi-Annual: skip a quarter
                             WHEN interval_months = 12 THEN 4 -- Annual: same quarter next year
                             ELSE 1
        END;

    next_quarter := fiscal_quarter + quarter_increment;

    -- Handle wrap-around for quarters > 4
    IF next_quarter > 4 THEN
        next_quarter := ((next_quarter - 1) % 4) + 1;
    END IF;

    -- Fiscal year calculations
    fiscal_year := EXTRACT(YEAR FROM fy_end_date)::INTEGER + 1 + ((months_since_fy_end - 1) / 12);

    -- Next quarter year (increments if we wrap past Q4)
    IF fiscal_quarter + quarter_increment > 4 THEN
        next_quarter_year := fiscal_year + ((fiscal_quarter + quarter_increment - 1) / 4);
    ELSE
        next_quarter_year := fiscal_year;
    END IF;

    -- Determine report type based on next quarter
    next_earnings_report_type := CASE
                                     WHEN next_quarter = 4 THEN 'Full Year'
                                     WHEN interval_months = 6 AND next_quarter IN (2, 4) THEN 'Half Year'
                                     WHEN interval_months = 12 THEN 'Full Year'
                                     ELSE 'Interim'
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Income Statement Report Date
-- ===================================================================
-- Calculates the next expected income statement report date based on
-- the earnings report frequency
-- Example: 2025-09-30 + Quarterly = 2025-12-30 (+ 3 months)
-- Example: 2025-11-30 + Semi-Annual = 2026-05-30 (+ 6 months)

CREATE OR REPLACE FUNCTION calculate_next_income_statement_report_date(
    income_statement_report_date DATE,
    earnings_report_frequency TEXT
)
    RETURNS DATE AS
$$
DECLARE
    interval_months INTEGER;
BEGIN
    IF income_statement_report_date IS NULL THEN
        RETURN NULL;
    END IF;

    -- Get interval months from frequency
    interval_months := frequency_to_months(earnings_report_frequency);

    -- Add the interval to the report date
    RETURN (income_statement_report_date + (interval_months || ' months')::INTERVAL)::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Year End Date
-- ===================================================================

CREATE OR REPLACE FUNCTION calculate_next_fy_end_date(fy_end_date DATE)
    RETURNS DATE AS
$$
BEGIN
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;

    -- Handle Feb 29 → Feb 28 transition properly
    -- Adding '1 year' interval handles this automatically in PostgreSQL
    RETURN (fy_end_date + INTERVAL '1 year')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Quarter
-- ===================================================================
-- Returns the next fiscal quarter based on current quarter and frequency

CREATE OR REPLACE FUNCTION calculate_next_fiscal_quarter(
    current_fiscal_quarter INTEGER,
    earnings_report_frequency TEXT
)
    RETURNS INTEGER AS
$$
DECLARE
    quarter_increment INTEGER;
    next_quarter      INTEGER;
    interval_months   INTEGER;
BEGIN
    IF current_fiscal_quarter IS NULL THEN
        RETURN NULL;
    END IF;

    -- Get interval months and calculate quarter increment
    interval_months := frequency_to_months(earnings_report_frequency);
    quarter_increment := interval_months / 3;

    -- Calculate next quarter with wrap-around
    next_quarter := current_fiscal_quarter + quarter_increment;

    -- Handle wrap-around (quarters 1-4)
    IF next_quarter > 4 THEN
        next_quarter := ((next_quarter - 1) % 4) + 1;
    END IF;

    RETURN next_quarter;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Reporting Lag
-- ===================================================================
-- Returns the difference in days between "Next Earnings" and
-- "Income Statement Report Date"
-- Also considers typical expected lags by frequency for validation

CREATE OR REPLACE FUNCTION calculate_reporting_lag(
    next_earnings DATE,
    income_statement_report_date DATE,
    earnings_report_frequency TEXT DEFAULT 'Quarterly'
)
    RETURNS INTEGER AS
$$
BEGIN
    IF next_earnings IS NULL OR income_statement_report_date IS NULL THEN
        RETURN NULL;
    END IF;

    -- Return actual lag in days
    -- Typical expected lags by frequency (for reference/validation):
    --   Quarterly: ~45 days
    --   Semi-Annual: ~60 days
    --   Annual: ~90 days
    RETURN next_earnings - income_statement_report_date;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Expected Report Date
-- ===================================================================
-- Given a fiscal period end date and frequency, estimates when the
-- earnings report should be released (period end + typical lag)

CREATE OR REPLACE FUNCTION calculate_expected_report_date(
    period_end_date DATE,
    earnings_report_frequency TEXT
)
    RETURNS DATE AS
$$
DECLARE
    expected_lag_days INTEGER;
BEGIN
    IF period_end_date IS NULL THEN
        RETURN NULL;
    END IF;

    -- Typical reporting lags by frequency
    expected_lag_days := CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'Quarterly')))
                             WHEN 'QUARTERLY' THEN 45
                             WHEN 'SEMI-ANNUALLY' THEN 60
                             WHEN 'SEMI-ANNUAL' THEN 60
                             WHEN 'ANNUALLY' THEN 90
                             WHEN 'ANNUAL' THEN 90
                             ELSE 45
        END;

    RETURN period_end_date + (expected_lag_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Validate Fiscal Dates
-- ===================================================================

CREATE OR REPLACE FUNCTION validate_fiscal_dates(
    fy_end_date DATE,
    report_date DATE,
    reference_date DATE DEFAULT CURRENT_DATE
)
    RETURNS TABLE
            (
                issue    TEXT,
                severity TEXT
            )
AS
$$
BEGIN
    -- FY End in future relative to data
    IF fy_end_date > reference_date THEN
        RETURN QUERY SELECT 'FY End Date is in the future'::TEXT AS issue, 'WARNING'::TEXT AS severity;
    END IF;

    -- Report date before FY End (impossible)
    IF report_date IS NOT NULL AND report_date < fy_end_date - INTERVAL '1 year' THEN
        RETURN QUERY SELECT 'Report date predates fiscal year'::TEXT AS issue, 'ERROR'::TEXT AS severity;
    END IF;

    -- Report date too far in future
    IF report_date > reference_date + INTERVAL '1 day' THEN
        RETURN QUERY SELECT 'Report date is in the future'::TEXT AS issue, 'WARNING'::TEXT AS severity;
    END IF;

    -- FY End not on month boundary
    IF fy_end_date != (DATE_TRUNC('month', fy_end_date) + INTERVAL '1 month - 1 day')::DATE THEN
        RETURN QUERY SELECT 'FY End is not last day of month'::TEXT AS issue, 'INFO'::TEXT AS severity;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- Importing US Region Data...
-- ===================================================================
\echo 'Importing regional data (US, EU, APAC, ROTW)...'

-- ===================================================================
-- STAGING TABLE CREATION
-- Columns organized by logical category for maintainability
-- ===================================================================
DROP TABLE IF EXISTS screening_staging;
CREATE TEMP TABLE screening_staging
(
    "Ticker"                                           TEXT,
    "ISIN"                                             TEXT,
    "Name"                                             TEXT,
    "Description"                                      TEXT,
    "Region"                                           TEXT,
    "Country"                                          TEXT,
    "Trading Country"                                  TEXT,
    "Exchange"                                         TEXT,
    "Unit"                                             TEXT,
    "Sector"                                           TEXT,
    "Industry"                                         TEXT,
    "Style Class"                                      TEXT,
    "Size Class"                                       TEXT,
    "Last Updated"                                     TEXT,
    "Income Statement Report Date"                     TEXT,
    "FY End"                                           TEXT,
    "Next Earnings"                                    TEXT,
    "Next Earnings (When)"                             TEXT,
    "Next Earnings (Status)"                           TEXT,
    "Dividend Record (Currency)"                       TEXT,
    "Dividend Record (Amount)"                         TEXT,
    "Dividend Record (Frequency)"                      TEXT,
    "Dividend Streak"                                  TEXT,
    "Dividend Record (Announce Date)"                  TEXT,
    "Dividend Record (Payable Date)"                   TEXT,
    "Dividend Record (Record Date)"                    TEXT,
    "Dividend Record (Ex Date)"                        TEXT,
    "Market Cap"                                       TEXT,
    "Enterprise Value"                                 TEXT,
    "Last Price"                                       TEXT,
    "Price Target (YTD Ago)"                           TEXT,
    "Total Return (YTD)"                               TEXT,
    "Price Target"                                     TEXT,
    "Price Target - Low"                               TEXT,
    "Price Target - Median"                            TEXT,
    "Price Target - High"                              TEXT,
    "Price Target - #"                                 TEXT,
    "P/E (NTM)"                                        TEXT,
    "P/E (LTM)"                                        TEXT,
    "Altman Z-Score (FY)"                              TEXT,
    "Altman Z-Score (FQ)"                              TEXT,
    "Altman Z-Score (LTM)"                             TEXT,
    "Beta (1Y)"                                        TEXT,
    "Beta (2Y)"                                        TEXT,
    "Beta (5Y)"                                        TEXT,
    "Analyst Rating"                                   TEXT,
    "# Strong Sell Ratings"                            TEXT,
    "# Strong Buys Ratings"                            TEXT,
    "# Hold Ratings"                                   TEXT,
    "# Buys Ratings"                                   TEXT,
    "# Sell Ratings"                                   TEXT,
    "Total Revenues/CAGR (5Y FY)"                      TEXT,
    "Total Revenues (FQ)"                              TEXT,
    "Total Revenues (-1FY)"                            TEXT,
    "Total Revenues (FY)"                              TEXT,
    "Total Revenues (LTM)"                             TEXT,
    "Total Operating Expenses (LTM)"                   TEXT,
    "P/TBV (LTM)"                                      TEXT,
    "TBV (FY)"                                         TEXT,
    "TBV (LTM)"                                        TEXT,
    "Market Cap (Country R)"                           TEXT,
    "Tot. Return %/CAGR (3Y)"                          TEXT,
    "Tot. Return %/CAGR (10Y)"                         TEXT,
    "Total Return (5Y)"                                TEXT,
    "Total Return (10Y)"                               TEXT,
    "Net Income/Adj. (-1FY)"                           TEXT,
    "CFF (LTM)"                                        TEXT,
    "CFI (LTM)"                                        TEXT,
    "FCF (LTM)"                                        TEXT,
    "CFO (LTM)"                                        TEXT,
    "EBITDA (FQ)"                                      TEXT,
    "EBITDA (LTM)"                                     TEXT,
    "EBITDA (FY)"                                      TEXT,
    "EBITDA (-1FY)"                                    TEXT,
    "EBITDA/Adj. (LTM)"                                TEXT,
    "EBITDA/Adj. (FY)"                                 TEXT,
    "EBITDA/Adj. (-1FY)"                               TEXT,
    "EBIT (FQ)"                                        TEXT,
    "EBIT (LTM)"                                       TEXT,
    "EBIT (FY)"                                        TEXT,
    "EBIT (-1FY)"                                      TEXT,
    "EBIT/Adj. (-1FY)"                                 TEXT,
    "EBIT/Adj. (FY)"                                   TEXT,
    "EBIT/Adj. (LTM)"                                  TEXT,
    "EBIT - Est Med (FY1E)"                            TEXT,
    "EBIT - Est Med (NTM)"                             TEXT,
    "Return On Equity % (LTM)"                         TEXT,
    "Return On Equity % (FY)"                          TEXT,
    "Net Income - (IS) (FY)"                           TEXT,
    "Net Income - (IS) (LTM)"                          TEXT,
    "Normalized Net Income (FY)"                       TEXT,
    "Normalized Net Income (LTM)"                      TEXT,
    "Net Income/Adj. (FY)"                             TEXT,
    "Net Income/Adj. (LTM)"                            TEXT,
    "Net Income Margin % (FY)"                         TEXT,
    "Net Income Margin % (LTM)"                        TEXT,
    "Volatility (1M)"                                  TEXT,
    "Volatility (3M)"                                  TEXT,
    "Volatility (6M)"                                  TEXT,
    "Volatility (1Y)"                                  TEXT,
    "Volume (Shrs)"                                    TEXT,
    "Dividend Per Share (LTM)"                         TEXT,
    "Div Yield (Ind)"                                  TEXT,
    "Div Yield (LTM)"                                  TEXT,
    "Total Debt (FY)"                                  TEXT,
    "Total Equity (FY)"                                TEXT,
    "Total Equity (LTM)"                               TEXT,
    "Total Debt (LTM)"                                 TEXT,
    "Total Assets (LTM)"                               TEXT,
    "Total Assets (FY)"                                TEXT,
    "Current Ratio (FY)"                               TEXT,
    "Current Ratio (LTM)"                              TEXT,
    "Gross Profit Margin % (FY)"                       TEXT,
    "Gross Profit Margin % (LTM)"                      TEXT,
    "Asset Turnover (FY)"                              TEXT,
    "Asset Turnover (LTM)"                             TEXT,
    "Gross Profit (LTM)"                               TEXT,
    "Gross Profit (FY)"                                TEXT,
    "EPS Norm - Est Avg (NTM)"                         TEXT,
    "EPS/Adj. (-1FY)"                                  TEXT,
    "EPS/Adj. (FY)"                                    TEXT,
    "EPS/Adj. (LTM)"                                   TEXT,
    "EPS Norm - Est Avg (FY1E)"                        TEXT,
    "Gain (Loss) On Sale Of Assets (LTM)"              TEXT,
    "Cost Of Revenues (LTM)"                           TEXT,
    "Cash Acquisitions (LTM)"                          TEXT,
    "Cash Acquisitions (FY)"                           TEXT,
    "Cash Acquisitions (-1FY)"                         TEXT,
    "Inventory (LTM)"                                  TEXT,
    "Goodwill (FQ)"                                    TEXT,
    "Goodwill (LTM)"                                   TEXT,
    "Goodwill (FY)"                                    TEXT,
    "Goodwill (-1FY)"                                  TEXT,
    "Impairment of Goodwill (FQ)"                      TEXT,
    "Impairment of Goodwill (LTM)"                     TEXT,
    "Impairment of Goodwill (-1FY)"                    TEXT,
    "Impairment of Goodwill (FY)"                      TEXT,
    "Operating Income (LTM)"                           TEXT,
    "Asset Writedown (LTM)"                            TEXT,
    "Asset Writedown (FY)"                             TEXT,
    "Asset Writedown (-1FY)"                           TEXT,
    "Operating Income (FY)"                            TEXT,
    "Capital Expenditure (LTM)"                        TEXT,
    "Capital Expenditure (-1FY)"                       TEXT,
    "Capital Expenditure (FY)"                         TEXT,
    "Retained Earnings (LTM)"                          TEXT,
    "Total Current Assets (LTM)"                       TEXT,
    "Total Current Liabilities (LTM)"                  TEXT,
    "R&D Expenses (LTM)"                               TEXT,
    "Restructuring Charges (LTM)"                      TEXT,
    "Restructuring Charges (FQ)"                       TEXT,
    "Restructuring Charges (-1FY)"                     TEXT,
    "Restructuring Charges (FY)"                       TEXT,
    "Interest Expense/Total (LTM)"                     TEXT,
    "Merger & Restructuring Charges (LTM)"             TEXT,
    "Working Capital (LTM)"                            TEXT,
    "Other Unusual Items/Total (LTM)"                  TEXT,
    "Interest Income On Investments (LTM)"             TEXT,
    "Buyback Yield (LTM)"                              TEXT,
    "Return on Assets (ROA) % (LTM)"                   TEXT,
    "Return on Assets (ROA) % (FY)"                    TEXT,
    "Net Income - (IS) (-1FY)"                         TEXT,
    "Normalized Net Income (-1FY)"                     TEXT,
    "CFF (FY)"                                         TEXT,
    "CFF (-1FY)"                                       TEXT,
    "CFI (FY)"                                         TEXT,
    "CFI (-1FY)"                                       TEXT,
    "CFO (FY)"                                         TEXT,
    "CFO (-1FY)"                                       TEXT,
    "Div Yield (-1FYInd)"                              TEXT,
    "FCF (FY)"                                         TEXT,
    "FCF (-1FY)"                              TEXT,
    "Capital Expenditure (FQ)"                         TEXT,
    "Capital Expenditure (5YAVGFQ)"                    TEXT,
    "CFF (FQ)"                                         TEXT,
    "CFI (FQ)"                                         TEXT,
    "CFO (FQ)"                                         TEXT,
    "FCF (FQ)"                                         TEXT,
    "Total Revenues (5YAVGFQ)"                         TEXT,
    "EBITDA (5YAVGFQ)"                                 TEXT,
    "EBIT (5YAVGFQ)"                                   TEXT,
    "FCF (5YAVGFQ)"                                    TEXT,
    "Cash Acquisitions (FQ)"                           TEXT,
    "Cash Acquisitions (5YAVGFQ)"                      TEXT,
    "Asset Writedown (FQ)"                             TEXT,
    "Asset Writedown (5YAVGFQ)"                        TEXT,
    "Impairment of Goodwill (5YAVGFQ)"                 TEXT,
    "Operating Income (FQ)"                            TEXT,
    "Operating Income (5YAVGFQ)"                       TEXT,
    "P/B (LTM)"                                        TEXT,
    "P/B (-1FY)"                                       TEXT,
    "P/B (5YAVG)"                                      TEXT,
    "Cash And Equivalents (LTM)"                       TEXT,
    "Cash And Equivalents (FQ)"                        TEXT,
    "Cash And Equivalents (FY)"                        TEXT,
    "Cash And Equivalents (5YAVGFQ)"                   TEXT,
    "Inventory (FQ)"                                   TEXT,
    "Inventory (FY)"                                   TEXT,
    "Goodwill (5YAVGFQ)"                               TEXT,
    "Inventory (5YAVGFQ)"                              TEXT,
    "Retained Earnings (FQ)"                           TEXT,
    "Retained Earnings (FY)"                           TEXT,
    "Retained Earnings (5YAVGFQ)"                      TEXT,
    "Working Capital (FQ)"                             TEXT,
    "Working Capital (FY)"                             TEXT,
    "Working Capital (5YAVGFY)"                        TEXT,
    "Div Yield (TTM)"                                  TEXT,
    "Div Yield (NTM)"                                  TEXT,
    "Div Yield (5YAVGLTM)"                             TEXT,
    "Gross Intangible Assets (LTM)"                    TEXT,
    "Gross Intangible Assets (FY)"                     TEXT,
    "Gross Intangible Assets (5YAVGFQ)"                TEXT,
    "Restructuring Charges (5YAVGFQ)"                  TEXT,
    "Merger & Restructuring Charges (FQ)"              TEXT,
    "Merger & Restructuring Charges (FY)"              TEXT,
    "Merger & Restructuring Charges (5YAVGFQ)"         TEXT,
    "Normalized Net Income (FQ)"                       TEXT,
    "Normalized Net Income (5YAVGFQ)"                  TEXT,
    "Net Income/Adj. (FQ)"                             TEXT,
    "Net Income/Adj. (5YAVGFQ)"                        TEXT,
    "Net Income - (IS) (FQ)"                           TEXT,
    "Net Income - (IS) (5YAVGFQ)"                      TEXT,
    "Net Income - (IS) (5YAVGLTM)"                     TEXT,
    "Normalized Net Income (5YAVGLTM)"                 TEXT,
    "EBITDA (5YAVGLTM)"                                TEXT,
    "EBIT (5YAVGLTM)"                                  TEXT,
    "Total Revenues (5YAVGLTM)"                        TEXT,
    "Revenues - Est YoY % (FY1E)"                      TEXT,
    "Price Chg. % (1M)"                                TEXT,
    "Price Chg. % (3M)"                                TEXT,
    "1-Day %"                                          TEXT,
    "Price (5D Ago)"                                   TEXT,
    "Price (1W Ago)"                                   TEXT,
    "Price (1M Ago)"                                   TEXT,
    "Price (3M Ago)"                                   TEXT,
    "Price (6M Ago)"                                   TEXT,
    "Price (1Y Ago)"                                   TEXT,
    "Price (3Y Ago)"                                   TEXT,
    "Price (5Y Ago)"                                   TEXT,
    "Price (QTD Ago)"                                  TEXT,
    "Rel. Volume"                                      TEXT,
    "Shrs Out"                                         TEXT,
    "Shrs Out (-1FY)"                                  TEXT,
    "Common Dividends Paid (LTM)"                      TEXT,
    "Common Dividends Paid (FY)"                       TEXT,
    "Selling General & Admin Expenses/Total (FQ)"      TEXT,
    "Selling General & Admin Expenses/Total (FY)"      TEXT,
    "Selling General & Admin Expenses/Total (-1FY)"    TEXT,
    "Selling General & Admin Expenses/Total (5YAVGFQ)" TEXT,
    "Accounts Receivable/Total (FY)"                   TEXT,
    "Accounts Receivable/Total (-1FY)"                 TEXT,
    "Accounts Receivable/Total (5YAVGFQ)"              TEXT,
    "Marketing Expenses (FQ)"                          TEXT,
    "Marketing Expenses (FY)"                          TEXT,
    "Marketing Expenses (-1FY)"                        TEXT,
    "Marketing Expenses (5YAVGLTM)"                    TEXT,
    "Revenues - Est Avg (NTM)"                         TEXT,
    "Revenues - Est Avg (FY1E)"                        TEXT,
    "Revenues - Est Med (NTM)"                         TEXT,
    "Revenues - Est Med (FY1E)"                        TEXT,
    "EV/Sales (EST FY1)"                               TEXT,
    "EV/Sales (LTM)"                                   TEXT,
    "EV/Sales (NTM)"                                   TEXT,
    "EV/Sales (-1FYLTM)"                               TEXT,
    "EV/Sales (-2FYLTM)"                               TEXT,
    "EV/Sales (-3FYLTM)"                               TEXT,
    "EV/Sales (3YAVGLTM)"                              TEXT,
    "EV/Sales (-1FQLTM)"                               TEXT,
    "EV/Sales (-2FQLTM)"                               TEXT,
    "EV/Sales (-3FQLTM)"                               TEXT,
    "EV/Sales (-4FQLTM)"                               TEXT,
    "52W High/Adj"                                     TEXT,
    "52W Low/Adj"                                      TEXT,
    "EMA (20D)"                                        TEXT,
    "EMA (50D)"                                        TEXT,
    "EMA (100D)"                                       TEXT,
    "EMA (250D)"                                       TEXT,
    "EV/EBITDA (LTM)"                                  TEXT,
    "EV/EBITDA (NTM)"                                  TEXT,
    "EV/EBITDA (-1FYLTM)"                              TEXT,
    "EV/EBITDA (-1FQLTM)"                              TEXT,
    "EV/EBITDA (3YAVGLTM)"                             TEXT,
    "EV/EBITDA (EST FY1)"                              TEXT,
    "P/E (EST FY1)"                                    TEXT,
    "P/E (-1FYLTM)"                                    TEXT,
    "P/E (-2FYLTM)"                                    TEXT,
    "P/E (-3FYLTM)"                                    TEXT,
    "P/E (3YAVGLTM)"                                   TEXT,
    "P/E (-1FQLTM)"                                    TEXT,
    "P/E (-2FQLTM)"                                    TEXT,
    "P/E (-3FQLTM)"                                    TEXT,
    "P/E (5YAVGLTM)"                                   TEXT,
    "P/E (-0FQQoQLTM)"                                 TEXT,
    "P/E (-0FYYoYLTM)"                                 TEXT,
    "P/E (-1FYYoYLTM)"                                 TEXT,
    "P/E (-0FQYoYLTM)"                                 TEXT,
    "Full Time Employees (FQ)"                         TEXT,
    "Full Time Employees (FY)"                         TEXT,
    "Full Time Employees (-1FY)"                       TEXT,
    "Full Time Employees (-2FY)"                       TEXT,
    "Full Time Employees (-3FY)"                       TEXT,
    "Avg Employees (5YAVGFY)"                          TEXT,
    "Net EPS - Basic (LTM)"                            TEXT,
    "Net EPS - Basic (FQ)"                             TEXT,
    "Net EPS - Basic (FY)"                             TEXT,
    "Net EPS - Basic (-1FQFQ)"                         TEXT,
    "Net EPS - Basic (-2FQFQ)"                         TEXT,
    "Net EPS - Basic (-3FQFQ)"                         TEXT,
    "Net EPS - Basic (-4FQFQ)"                         TEXT,
    "Net EPS - Basic (-1FY)"                           TEXT,
    "Net EPS - Basic (-2FY)"                           TEXT,
    "Net EPS - Basic (-3FY)"                           TEXT,
    "Net EPS - Basic (-4FY)"                           TEXT,
    "Net EPS - Basic (-5FY)"                           TEXT,
    "EPS Est Avg Rev % (FY1E - 1W)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 1M)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 3M)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 6M)"                    TEXT,
    "EPS Est Avg Rev % (FY1E - 1Y)"                    TEXT,
    "Div Yield (-2FYInd)"                              TEXT,
    "Div Yield (-3FYInd)"                              TEXT,
    "Div Yield (-4FYInd)"                              TEXT,
    "Div Yield (-5FYInd)"                              TEXT,
    "EBITDA - Est Avg (NTM)"                           TEXT,
    "EBITDA - Est Avg (FY1E)"                          TEXT,
    "EPS GAAP - Est Avg (NTM)"                         TEXT,
    "EPS GAAP - Est Avg (FY1E)"                        TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 1M)"               TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 3M)"               TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 6M)"               TEXT,
    "EPS GAAP Est Avg Rev % (FY1E - 1Y)"               TEXT,
    "EPS Norm - Est # (FY1E)"                 TEXT,
    "CFO (-1FQFQ)"                            TEXT,
    "CFO (-2FQFQ)"                            TEXT,
    "CFO (-3FQFQ)"                            TEXT,
    "CFO (-4FQFQ)"                            TEXT,
    "CFI (-1FQFQ)"                            TEXT,
    "CFI (-2FQFQ)"                            TEXT,
    "CFI (-3FQFQ)"                            TEXT,
    "CFI (-4FQFQ)"                            TEXT,
    "CFI (-2FY)"                              TEXT,
    "CFI (-3FY)"                              TEXT,
    "CFI (-4FY)"                              TEXT,
    "FCF (-1FQFQ)"                            TEXT,
    "FCF (-2FQFQ)"                            TEXT,
    "FCF (-3FQFQ)"                            TEXT,
    "FCF (-4FQFQ)"                            TEXT,
    "CFF (-2FY)"                              TEXT,
    "CFF (-3FY)"                              TEXT,
    "CFF (-4FY)"                              TEXT,
    "CFF (-1FQFQ)"                            TEXT,
    "CFF (-2FQFQ)"                            TEXT,
    "CFF (-3FQFQ)"                            TEXT,
    "CFF (-4FQFQ)"                            TEXT,
    "CFO (-2FY)"                              TEXT,
    "CFO (-3FY)"                              TEXT,
    "CFO (-4FY)"                              TEXT,
    "Cash Acquisitions (-1FQFQ)"              TEXT,
    "Cash Acquisitions (-2FQFQ)"              TEXT,
    "Cash Acquisitions (-3FQFQ)"              TEXT,
    "Cash Acquisitions (-4FQFQ)"              TEXT,
    "FCF (-2FY)"                              TEXT,
    "FCF (-3FY)"                              TEXT,
    "FCF (-4FY)"                              TEXT,
    "Price Target (1W Ago)"                   TEXT,
    "Price Target (1M Ago)"                   TEXT,
    "Price Target (3M Ago)"                   TEXT,
    "Price Target (6M Ago)"                   TEXT,
    "Price Target (MTD Ago)"                  TEXT,
    "Price Target (QTD Ago)"                  TEXT,
    "Price Target (1Y Ago)"                   TEXT,
    "Price Target - # (3M Ago)"               TEXT,
    "Price Target - # (6M Ago)"               TEXT,
    "Price Target - # (YTD Ago)"              TEXT,
    "Price Target - # (1Y Ago)"               TEXT,
    "Price Target - # (1W Ago)"               TEXT,
    "Price Target - # (1M Ago)"               TEXT,
    "Price Target - # (MTD Ago)"              TEXT,
    "Price Target - # (QTD Ago)"              TEXT,
    "Price Target - High (1W Ago)"            TEXT,
    "Price Target - High (1M Ago)"            TEXT,
    "Price Target - High (6M Ago)"            TEXT,
    "Price Target - High (MTD Ago)"           TEXT,
    "Price Target - High (3M Ago)"            TEXT,
    "Price Target - High (QTD Ago)"           TEXT,
    "Price Target - High (1Y Ago)"            TEXT,
    "Price Target - High (YTD Ago)"           TEXT,
    "Price Target - Low (1W Ago)"             TEXT,
    "Price Target - Low (1M Ago)"             TEXT,
    "Price Target - Low (3M Ago)"             TEXT,
    "Price Target - Low (6M Ago)"             TEXT,
    "Price Target - Low (MTD Ago)"            TEXT,
    "Price Target - Low (QTD Ago)"            TEXT,
    "Price Target - Low (YTD Ago)"            TEXT,
    "Price Target - Low (1Y Ago)"             TEXT,
    "Price Target - Median (1W Ago)"          TEXT,
    "Price Target - Median (1M Ago)"          TEXT,
    "Price Target - Median (3M Ago)"          TEXT,
    "Price Target - Median (6M Ago)"          TEXT,
    "Price Target - Median (MTD Ago)"         TEXT,
    "Price Target - Median (QTD Ago)"         TEXT,
    "Price Target - Median (YTD Ago)"         TEXT,
    "Price Target - Median (1Y Ago)"          TEXT,
    "Impairment of Goodwill (-1FQFQ)"         TEXT,
    "Impairment of Goodwill (-2FQFQ)"         TEXT,
    "Impairment of Goodwill (-3FQFQ)"         TEXT,
    "Impairment of Goodwill (-4FQFQ)"         TEXT,
    "Impairment of Goodwill (-2FY)"           TEXT,
    "Impairment of Goodwill (-3FY)"           TEXT,
    "Impairment of Goodwill (-4FY)"           TEXT,
    "Asset Writedown (-1FQFQ)"                TEXT,
    "Asset Writedown (-2FQFQ)"                TEXT,
    "Asset Writedown (-3FQFQ)"                TEXT,
    "Asset Writedown (-4FQFQ)"                TEXT,
    "Asset Writedown (-2FY)"                  TEXT,
    "Asset Writedown (-3FY)"                  TEXT,
    "Asset Writedown (-4FY)"                  TEXT,
    "Asset Writedown (-5FY)"                  TEXT,
    "Gain (Loss) On Sale Of Assets (FQ)"      TEXT,
    "Gain (Loss) On Sale Of Assets (FY)"      TEXT,
    "Gain (Loss) On Sale Of Assets (-1FQFQ)"  TEXT,
    "Gain (Loss) On Sale Of Assets (-2FQFQ)"  TEXT,
    "Gain (Loss) On Sale Of Assets (-3FQFQ)"  TEXT,
    "Gain (Loss) On Sale Of Assets (-4FQFQ)"  TEXT,
    "Gain (Loss) On Sale Of Assets (-1FY)"    TEXT,
    "Gain (Loss) On Sale Of Assets (-2FY)"    TEXT,
    "Gain (Loss) On Sale Of Assets (-3FY)"    TEXT,
    "Gain (Loss) On Sale Of Assets (-4FY)"    TEXT,
    "Restructuring Charges (-1FQFQ)"          TEXT,
    "Restructuring Charges (-2FQFQ)"          TEXT,
    "Restructuring Charges (-3FQFQ)"          TEXT,
    "Restructuring Charges (-4FQFQ)"          TEXT,
    "Restructuring Charges (-2FY)"            TEXT,
    "Restructuring Charges (-3FY)"            TEXT,
    "Restructuring Charges (-4FY)"            TEXT,
    "Net Income - (IS) (-1FQFQ)"              TEXT,
    "Net Income - (IS) (-2FQFQ)"              TEXT,
    "Net Income - (IS) (-3FQFQ)"              TEXT,
    "Net Income - (IS) (-4FQFQ)"              TEXT,
    "Net Income - (IS) (-2FY)"                TEXT,
    "Net Income - (IS) (-3FY)"                TEXT,
    "Net Income - (IS) (-4FY)"                TEXT,
    "Normalized Net Income (-1FQFQ)"          TEXT,
    "Normalized Net Income (-2FQFQ)"          TEXT,
    "Normalized Net Income (-3FQFQ)"          TEXT,
    "Normalized Net Income (-4FQFQ)"          TEXT,
    "Normalized Net Income (-2FY)"            TEXT,
    "Normalized Net Income (-3FY)"            TEXT,
    "Normalized Net Income (-4FY)"            TEXT,
    "Net Income/Adj. (-1FQFQ)"                TEXT,
    "Net Income/Adj. (-2FQFQ)"                TEXT,
    "Net Income/Adj. (-3FQFQ)"                TEXT,
    "Net Income/Adj. (-4FQFQ)"                TEXT,
    "Net Income/Adj. (-2FY)"                  TEXT,
    "Net Income/Adj. (-3FY)"                  TEXT,
    "Net Income/Adj. (-4FY)"                  TEXT,
    "EBIT (-1FQFQ)"                           TEXT,
    "EBIT (-2FQFQ)"                           TEXT,
    "EBIT (-3FQFQ)"                           TEXT,
    "EBIT (-4FQFQ)"                           TEXT,
    "EBIT (-2FY)"                             TEXT,
    "EBIT (-3FY)"                             TEXT,
    "EBIT (-4FY)"                             TEXT,
    "EBIT/Adj. (FQ)"                          TEXT,
    "EBIT/Adj. (-1FQFQ)"                      TEXT,
    "EBIT/Adj. (-2FQFQ)"                      TEXT,
    "EBIT/Adj. (-3FQFQ)"                      TEXT,
    "EBIT/Adj. (-4FQFQ)"                      TEXT,
    "EBIT/Adj. (-2FY)"                        TEXT,
    "EBIT/Adj. (-3FY)"                        TEXT,
    "EBIT/Adj. (-4FY)"                        TEXT,
    "EBITDA (-1FQFQ)"                         TEXT,
    "EBITDA (-2FQFQ)"                         TEXT,
    "EBITDA (-3FQFQ)"                         TEXT,
    "EBITDA (-4FQFQ)"                         TEXT,
    "EBITDA (-2FY)"                           TEXT,
    "EBITDA (-4FY)"                           TEXT,
    "EBITDA (-3FY)"                           TEXT,
    "EBITDA/Adj. (FQ)"                        TEXT,
    "EBITDA/Adj. (-1FQFQ)"                    TEXT,
    "EBITDA/Adj. (-2FQFQ)"                    TEXT,
    "EBITDA/Adj. (-3FQFQ)"                    TEXT,
    "EBITDA/Adj. (-4FQFQ)"                    TEXT,
    "EBITDA/Adj. (-2FY)"                      TEXT,
    "EBITDA/Adj. (-3FY)"                      TEXT,
    "EBITDA/Adj. (-4FY)"                      TEXT,
    "Basic EPS - Cont (LTM)"                  TEXT,
    "Basic EPS - Cont (FQ)"                   TEXT,
    "Basic EPS - Cont (FY)"                   TEXT,
    "Basic EPS - Cont (-1FQFQ)"               TEXT,
    "Basic EPS - Cont (-2FQFQ)"               TEXT,
    "Basic EPS - Cont (-4FQFQ)"               TEXT,
    "Basic EPS - Cont (-3FQFQ)"               TEXT,
    "Basic EPS - Cont (-1FY)"                 TEXT,
    "Basic EPS - Cont (-2FY)"                 TEXT,
    "Basic EPS - Cont (-3FY)"                 TEXT,
    "Basic EPS - Cont (-4FY)"                 TEXT,
    "EPS/Adj. (FQ)"                           TEXT,
    "EPS/Adj. (-1FQFQ)"                       TEXT,
    "EPS/Adj. (-3FQFQ)"                       TEXT,
    "EPS/Adj. (-4FQFQ)"                       TEXT,
    "EPS/Adj. (-2FQFQ)"                       TEXT,
    "EPS/Adj. (-2FY)"                         TEXT,
    "EPS/Adj. (-3FY)"                         TEXT,
    "EPS/Adj. (-4FY)"                         TEXT,
    "Cash Acquisitions (-2FY)"                TEXT,
    "Cash Acquisitions (-3FY)"                TEXT,
    "Cash Acquisitions (-4FY)"                TEXT,
    "Capital Expenditure (-1FQFQ)"            TEXT,
    "Capital Expenditure (-3FQFQ)"            TEXT,
    "Capital Expenditure (-4FQFQ)"            TEXT,
    "Capital Expenditure (-2FQFQ)"            TEXT,
    "Capital Expenditure (-2FY)"              TEXT,
    "Capital Expenditure (-3FY)"              TEXT,
    "Capital Expenditure (-4FY)"              TEXT,
    "Working Capital (-1FQ)"                  TEXT,
    "Working Capital (-2FQ)"                  TEXT,
    "Working Capital (-3FQ)"                  TEXT,
    "Working Capital (-4FQ)"                  TEXT,
    "Working Capital (-1FY)"                  TEXT,
    "Working Capital (-2FY)"                  TEXT,
    "Working Capital (-3FY)"                  TEXT,
    "Working Capital (-4FY)"                  TEXT,
    "Total Debt (FQ)"                         TEXT,
    "Total Debt (-1FQ)"                       TEXT,
    "Total Debt (-2FQ)"                       TEXT,
    "Total Debt (-3FQ)"                       TEXT,
    "Total Debt (-4FQ)"                       TEXT,
    "Total Debt (-1FY)"                       TEXT,
    "Total Debt (-2FY)"                       TEXT,
    "Total Debt (-3FY)"                       TEXT,
    "Total Debt (-4FY)"                       TEXT,
    "Total Assets (FQ)"                       TEXT,
    "Total Assets (-1FQ)"                     TEXT,
    "Total Assets (-2FQ)"                     TEXT,
    "Total Assets (-3FQ)"                     TEXT,
    "Total Assets (-4FQ)"                     TEXT,
    "Total Assets (-1FY)"                     TEXT,
    "Total Assets (-2FY)"                     TEXT,
    "Total Assets (-3FY)"                     TEXT,
    "Total Assets (-4FY)"                     TEXT,
    "Gross Profit (FQ)"                       TEXT,
    "Gross Profit (-1FQFQ)"                   TEXT,
    "Gross Profit (-3FQFQ)"                   TEXT,
    "Gross Profit (-4FQFQ)"                   TEXT,
    "Gross Profit (-2FQFQ)"                   TEXT,
    "Gross Profit (-1FY)"                     TEXT,
    "Gross Profit (-2FY)"                     TEXT,
    "Gross Profit (-3FY)"                     TEXT,
    "Gross Profit (-4FY)"                     TEXT,
    "Inventory (-1FQ)"                        TEXT,
    "Inventory (-3FQ)"                        TEXT,
    "Inventory (-4FQ)"                        TEXT,
    "Inventory (-2FQ)"                        TEXT,
    "Inventory (-1FY)"                        TEXT,
    "Inventory (-2FY)"                        TEXT,
    "Inventory (-4FY)"                        TEXT,
    "Inventory (-3FY)"                        TEXT,
    "Goodwill (-1FQ)"                         TEXT,
    "Goodwill (-4FQ)"                         TEXT,
    "Goodwill (-2FQ)"                         TEXT,
    "Goodwill (-3FQ)"                         TEXT,
    "Goodwill (-2FY)"                         TEXT,
    "Goodwill (-3FY)"                         TEXT,
    "Goodwill (-4FY)"                         TEXT,
    "Operating Income (-1FQFQ)"               TEXT,
    "Operating Income (-3FQFQ)"               TEXT,
    "Operating Income (-4FQFQ)"               TEXT,
    "Operating Income (-2FQFQ)"               TEXT,
    "Operating Income (-1FY)"                 TEXT,
    "Operating Income (-2FY)"                 TEXT,
    "Operating Income (-4FY)"                 TEXT,
    "Operating Income (-3FY)"                 TEXT,
    "Retained Earnings (-1FQ)"                TEXT,
    "Retained Earnings (-2FQ)"                TEXT,
    "Retained Earnings (-3FQ)"                TEXT,
    "Retained Earnings (-4FQ)"                TEXT,
    "Retained Earnings (-1FY)"                TEXT,
    "Retained Earnings (-2FY)"                TEXT,
    "Retained Earnings (-3FY)"                TEXT,
    "Retained Earnings (-4FY)"                TEXT,
    "R&D Expenses (FQ)"                       TEXT,
    "R&D Expenses (FY)"                       TEXT,
    "R&D Expenses (-1FQFQ)"                   TEXT,
    "R&D Expenses (-2FQFQ)"                   TEXT,
    "R&D Expenses (-3FQFQ)"                   TEXT,
    "R&D Expenses (-4FQFQ)"                   TEXT,
    "R&D Expenses (-1FY)"                     TEXT,
    "R&D Expenses (-2FY)"                     TEXT,
    "R&D Expenses (-4FY)"                     TEXT,
    "R&D Expenses (-3FY)"                     TEXT,
    "Merger & Restructuring Charges (-1FQFQ)" TEXT,
    "Merger & Restructuring Charges (-3FQFQ)" TEXT,
    "Merger & Restructuring Charges (-4FQFQ)" TEXT,
    "Merger & Restructuring Charges (-2FQFQ)" TEXT,
    "Merger & Restructuring Charges (-1FY)"   TEXT,
    "Merger & Restructuring Charges (-3FY)"   TEXT,
    "Merger & Restructuring Charges (-4FY)"   TEXT,
    "Merger & Restructuring Charges (-2FY)"   TEXT,
    "Cash And Equivalents (-1FQ)"             TEXT,
    "Cash And Equivalents (-3FQ)"             TEXT,
    "Cash And Equivalents (-4FQ)"             TEXT,
    "Cash And Equivalents (-2FQ)"             TEXT,
    "Cash And Equivalents (-1FY)"             TEXT,
    "Cash And Equivalents (-2FY)"             TEXT,
    "Cash And Equivalents (-3FY)"             TEXT,
    "Cash And Equivalents (-4FY)"             TEXT,
    "Gross Intangible Assets (FQ)"            TEXT,
    "Gross Intangible Assets (-1FQ)"          TEXT,
    "Gross Intangible Assets (-3FQ)"          TEXT,
    "Gross Intangible Assets (-4FQ)"          TEXT,
    "Gross Intangible Assets (-2FQ)"          TEXT,
    "Gross Intangible Assets (-1FY)"          TEXT,
    "Gross Intangible Assets (-2FY)"          TEXT,
    "Gross Intangible Assets (-3FY)"          TEXT,
    "Gross Intangible Assets (-4FY)"          TEXT,
    "Total Revenues (-1FQFQ)"                 TEXT,
    "Total Revenues (-2FQFQ)"                 TEXT,
    "Total Revenues (-3FQFQ)"                 TEXT,
    "Total Revenues (-4FQFQ)"                 TEXT,
    "Total Revenues (-2FY)"                   TEXT,
    "Total Revenues (-3FY)"                   TEXT,
    "Total Revenues (-4FY)"                   TEXT
);
-- ===================================================================
-- DATA IMPORT EXECUTION
-- ===================================================================

-- US Region
\echo 'Importing US data...'
\copy screening_staging FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- EU Region
\echo 'Importing EU data...'
\copy screening_staging FROM 'data/screening_eu.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- APAC Region
\echo 'Importing APAC data...'
\copy screening_staging FROM 'data/screening_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- ROTW Region
\echo 'Importing ROTW data...'
\copy screening_staging FROM 'data/screening_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- ===================================================================
-- DATA VALIDATION (PRE-INSERT)
-- ===================================================================
\echo 'Validating imported data...'
SELECT 'Total rows in staging:' AS info, COUNT(*) AS count
FROM screening_staging;

TRUNCATE TABLE equities;
INSERT INTO equities ("Ticker", "ISIN", "Name", "Region", "Country", "Trading Country", "Exchange", "Unit", "Sector",
                      "Industry", "Style Class", "Size Class", "FY End", "Next Earnings (When)",
                      "Next Earnings (Status)", "Dividend Record (Currency)", "Dividend Record (Frequency)",
                      "Current Fiscal Quarter", "Next Fiscal Quarter", "Next Earnings (Report)", "Last Updated",
                      "Income Statement Report Date", "Next Earnings", "Dividend Record (Announce Date)",
                      "Dividend Record (Payable Date)", "Dividend Record (Record Date)", "Dividend Record (Ex Date)",
                      "Reference Date", "FY End Date", "Next FY End Date", "Price Target", "Price Target - Median",
                      "Dividend Record (Amount)", "Market Cap", "Enterprise Value", "Last Price",
                      "Price Target (YTD Ago)", "Price Target - Low", "Price Target - High", "Market Cap (Country R)",
                      "Volume (Shrs)", "Dividend Per Share (LTM)", "Price (5D Ago)", "Price (1W Ago)", "Price (1M Ago)",
                      "Price (3M Ago)", "Price (6M Ago)", "Price (1Y Ago)", "Price (3Y Ago)", "Price (5Y Ago)",
                      "Price (QTD Ago)", "Rel. Volume", "52W High/Adj", "52W Low/Adj", "EMA (20D)", "EMA (50D)",
                      "EMA (100D)", "EMA (250D)", "Price Target (1W Ago)", "Price Target (1M Ago)",
                      "Price Target (3M Ago)", "Price Target (6M Ago)", "Price Target (MTD Ago)",
                      "Price Target (QTD Ago)", "Price Target (1Y Ago)", "Price Target - High (1W Ago)",
                      "Price Target - High (1M Ago)", "Price Target - High (6M Ago)", "Price Target - High (MTD Ago)",
                      "Price Target - High (3M Ago)", "Price Target - High (QTD Ago)", "Price Target - High (1Y Ago)",
                      "Price Target - High (YTD Ago)", "Price Target - Low (1W Ago)", "Price Target - Low (1M Ago)",
                      "Price Target - Low (3M Ago)", "Price Target - Low (6M Ago)", "Price Target - Low (MTD Ago)",
                      "Price Target - Low (QTD Ago)", "Price Target - Low (YTD Ago)", "Price Target - Low (1Y Ago)",
                      "Price Target - Median (1W Ago)", "Price Target - Median (1M Ago)",
                      "Price Target - Median (3M Ago)", "Price Target - Median (6M Ago)",
                      "Price Target - Median (MTD Ago)", "Price Target - Median (QTD Ago)",
                      "Price Target - Median (YTD Ago)", "Price Target - Median (1Y Ago)", "Total Revenues (FQ)",
                      "Total Revenues (-1FY)", "Total Revenues (FY)", "Total Revenues (LTM)",
                      "Total Operating Expenses (LTM)", "Net Income/Adj. (-1FY)", "EBITDA (FQ)", "EBITDA (LTM)",
                      "EBITDA (FY)", "EBITDA (-1FY)", "EBITDA/Adj. (LTM)", "EBITDA/Adj. (FY)", "EBITDA/Adj. (-1FY)",
                      "EBIT (FQ)", "EBIT (LTM)", "EBIT (FY)", "EBIT (-1FY)", "EBIT/Adj. (-1FY)", "EBIT/Adj. (FY)",
                      "EBIT/Adj. (LTM)", "EBIT - Est Med (FY1E)", "EBIT - Est Med (NTM)", "Net Income - (IS) (FY)",
                      "Net Income - (IS) (LTM)", "Normalized Net Income (FY)", "Normalized Net Income (LTM)",
                      "Net Income/Adj. (FY)", "Net Income/Adj. (LTM)", "Gross Profit (LTM)", "Gross Profit (FY)",
                      "Cost Of Revenues (LTM)", "Operating Income (LTM)", "Operating Income (FY)", "R&D Expenses (LTM)",
                      "Interest Expense/Total (LTM)", "Interest Income On Investments (LTM)",
                      "Net Income - (IS) (-1FY)", "Normalized Net Income (-1FY)", "Total Revenues (5YAVGFQ)",
                      "EBITDA (5YAVGFQ)", "EBIT (5YAVGFQ)", "Operating Income (FQ)", "Operating Income (5YAVGFQ)",
                      "Normalized Net Income (FQ)", "Normalized Net Income (5YAVGFQ)", "Net Income/Adj. (FQ)",
                      "Net Income/Adj. (5YAVGFQ)", "Net Income - (IS) (FQ)", "Net Income - (IS) (5YAVGFQ)",
                      "Net Income - (IS) (5YAVGLTM)", "Normalized Net Income (5YAVGLTM)", "EBITDA (5YAVGLTM)",
                      "EBIT (5YAVGLTM)", "Total Revenues (5YAVGLTM)", "Selling General & Admin Expenses/Total (FQ)",
                      "Selling General & Admin Expenses/Total (FY)", "Selling General & Admin Expenses/Total (-1FY)",
                      "Selling General & Admin Expenses/Total (5YAVGFQ)",
                      "Marketing Expenses (FQ)",
                      "Marketing Expenses (FY)", "Marketing Expenses (-1FY)", "Marketing Expenses (5YAVGLTM)",
                      "Revenues - Est Avg (NTM)", "Revenues - Est Avg (FY1E)", "Revenues - Est Med (NTM)",
                      "Revenues - Est Med (FY1E)", "EBITDA - Est Avg (NTM)", "EBITDA - Est Avg (FY1E)", "TBV (FY)",
                      "TBV (LTM)", "Total Debt (FY)", "Total Equity (FY)", "Total Equity (LTM)", "Total Debt (LTM)",
                      "Total Assets (LTM)", "Total Assets (FY)", "Inventory (LTM)", "Goodwill (FQ)", "Goodwill (LTM)",
                      "Goodwill (FY)", "Goodwill (-1FY)", "Retained Earnings (LTM)", "Total Current Assets (LTM)",
                      "Total Current Liabilities (LTM)", "Working Capital (LTM)", "Cash And Equivalents (LTM)",
                      "Cash And Equivalents (FQ)", "Cash And Equivalents (FY)", "Cash And Equivalents (5YAVGFQ)",
                      "Inventory (FQ)", "Inventory (FY)", "Goodwill (5YAVGFQ)", "Inventory (5YAVGFQ)",
                      "Retained Earnings (FQ)", "Retained Earnings (FY)", "Retained Earnings (5YAVGFQ)",
                      "Working Capital (FQ)", "Working Capital (FY)", "Working Capital (5YAVGFY)",
                      "Gross Intangible Assets (LTM)", "Gross Intangible Assets (FY)",
                      "Gross Intangible Assets (5YAVGFQ)", "Accounts Receivable/Total (FY)",
                      "Accounts Receivable/Total (-1FY)", "Accounts Receivable/Total (5YAVGFQ)", "CFF (LTM)",
                      "CFI (LTM)", "FCF (LTM)", "CFO (LTM)", "Cash Acquisitions (LTM)", "Cash Acquisitions (FY)",
                      "Cash Acquisitions (-1FY)", "Capital Expenditure (LTM)", "Capital Expenditure (-1FY)",
                      "Capital Expenditure (FY)", "CFF (FY)", "CFF (-1FY)", "CFI (FY)", "CFI (-1FY)", "CFO (FY)",
                      "CFO (-1FY)", "FCF (FY)", "FCF (-1FY)", "Capital Expenditure (FQ)",
                      "Capital Expenditure (5YAVGFQ)", "CFF (FQ)", "CFI (FQ)", "CFO (FQ)", "FCF (FQ)", "FCF (5YAVGFQ)",
                      "Cash Acquisitions (FQ)", "Cash Acquisitions (5YAVGFQ)", "Common Dividends Paid (LTM)",
                      "Common Dividends Paid (FY)", "CFO (-1FQFQ)", "CFO (-2FQFQ)", "CFO (-3FQFQ)", "CFO (-4FQFQ)",
                      "CFI (-1FQFQ)", "CFI (-2FQFQ)", "CFI (-3FQFQ)", "CFI (-4FQFQ)", "CFI (-2FY)", "CFI (-3FY)",
                      "CFI (-4FY)", "FCF (-1FQFQ)", "FCF (-2FQFQ)", "FCF (-3FQFQ)", "FCF (-4FQFQ)", "CFF (-2FY)",
                      "CFF (-3FY)", "CFF (-4FY)", "CFF (-1FQFQ)", "CFF (-2FQFQ)", "CFF (-3FQFQ)", "CFF (-4FQFQ)",
                      "CFO (-2FY)", "CFO (-3FY)", "CFO (-4FY)", "Cash Acquisitions (-1FQFQ)",
                      "Cash Acquisitions (-2FQFQ)", "Cash Acquisitions (-3FQFQ)", "Cash Acquisitions (-4FQFQ)",
                      "FCF (-2FY)", "FCF (-3FY)", "FCF (-4FY)",
    -- NEW: Cash Acquisitions Historical (FY)
                      "Cash Acquisitions (-2FY)", "Cash Acquisitions (-3FY)", "Cash Acquisitions (-4FY)",
    -- NEW: Capital Expenditure Historical
                      "Capital Expenditure (-1FQFQ)", "Capital Expenditure (-2FQFQ)", "Capital Expenditure (-3FQFQ)",
                      "Capital Expenditure (-4FQFQ)", "Capital Expenditure (-2FY)", "Capital Expenditure (-3FY)",
                      "Capital Expenditure (-4FY)",
    -- NEW: Working Capital Historical
                      "Working Capital (-1FQ)", "Working Capital (-2FQ)", "Working Capital (-3FQ)",
                      "Working Capital (-4FQ)", "Working Capital (-1FY)", "Working Capital (-2FY)",
                      "Working Capital (-3FY)", "Working Capital (-4FY)",
    -- NEW: Total Debt Historical
                      "Total Debt (FQ)", "Total Debt (-1FQ)", "Total Debt (-2FQ)", "Total Debt (-3FQ)",
                      "Total Debt (-4FQ)", "Total Debt (-1FY)", "Total Debt (-2FY)", "Total Debt (-3FY)",
                      "Total Debt (-4FY)",
    -- NEW: Total Assets Historical
                      "Total Assets (FQ)", "Total Assets (-1FQ)", "Total Assets (-2FQ)", "Total Assets (-3FQ)",
                      "Total Assets (-4FQ)", "Total Assets (-1FY)", "Total Assets (-2FY)", "Total Assets (-3FY)",
                      "Total Assets (-4FY)",
    -- NEW: Gross Profit Historical
                      "Gross Profit (FQ)", "Gross Profit (-1FQFQ)", "Gross Profit (-2FQFQ)", "Gross Profit (-3FQFQ)",
                      "Gross Profit (-4FQFQ)", "Gross Profit (-1FY)", "Gross Profit (-2FY)", "Gross Profit (-3FY)",
                      "Gross Profit (-4FY)",
    -- NEW: Inventory Historical
                      "Inventory (-1FQ)", "Inventory (-2FQ)", "Inventory (-3FQ)", "Inventory (-4FQ)",
                      "Inventory (-1FY)", "Inventory (-2FY)", "Inventory (-3FY)", "Inventory (-4FY)",
    -- NEW: Goodwill Historical
                      "Goodwill (-1FQ)", "Goodwill (-2FQ)", "Goodwill (-3FQ)", "Goodwill (-4FQ)",
                      "Goodwill (-2FY)", "Goodwill (-3FY)", "Goodwill (-4FY)",
    -- NEW: Operating Income Historical
                      "Operating Income (-1FQFQ)", "Operating Income (-2FQFQ)", "Operating Income (-3FQFQ)",
                      "Operating Income (-4FQFQ)", "Operating Income (-1FY)", "Operating Income (-2FY)",
                      "Operating Income (-3FY)", "Operating Income (-4FY)",
    -- NEW: Retained Earnings Historical
                      "Retained Earnings (-1FQ)", "Retained Earnings (-2FQ)", "Retained Earnings (-3FQ)",
                      "Retained Earnings (-4FQ)", "Retained Earnings (-1FY)", "Retained Earnings (-2FY)",
                      "Retained Earnings (-3FY)", "Retained Earnings (-4FY)",
    -- NEW: R&D Expenses Historical
                      "R&D Expenses (FQ)", "R&D Expenses (FY)", "R&D Expenses (-1FQFQ)", "R&D Expenses (-2FQFQ)",
                      "R&D Expenses (-3FQFQ)", "R&D Expenses (-4FQFQ)", "R&D Expenses (-1FY)", "R&D Expenses (-2FY)",
                      "R&D Expenses (-3FY)", "R&D Expenses (-4FY)",
    -- NEW: Merger & Restructuring Charges Historical
                      "Merger & Restructuring Charges (-1FQFQ)", "Merger & Restructuring Charges (-2FQFQ)",
                      "Merger & Restructuring Charges (-3FQFQ)", "Merger & Restructuring Charges (-4FQFQ)",
                      "Merger & Restructuring Charges (-1FY)", "Merger & Restructuring Charges (-2FY)",
                      "Merger & Restructuring Charges (-3FY)", "Merger & Restructuring Charges (-4FY)",
    -- NEW: Cash And Equivalents Historical
                      "Cash And Equivalents (-1FQ)", "Cash And Equivalents (-2FQ)", "Cash And Equivalents (-3FQ)",
                      "Cash And Equivalents (-4FQ)", "Cash And Equivalents (-1FY)", "Cash And Equivalents (-2FY)",
                      "Cash And Equivalents (-3FY)", "Cash And Equivalents (-4FY)",
    -- NEW: Gross Intangible Assets Historical
                      "Gross Intangible Assets (FQ)", "Gross Intangible Assets (-1FQ)",
                      "Gross Intangible Assets (-2FQ)",
                      "Gross Intangible Assets (-3FQ)", "Gross Intangible Assets (-4FQ)",
                      "Gross Intangible Assets (-1FY)",
                      "Gross Intangible Assets (-2FY)", "Gross Intangible Assets (-3FY)",
                      "Gross Intangible Assets (-4FY)",
    -- Continue with existing columns
                      "P/E (NTM)", "P/E (LTM)",
                      "Altman Z-Score (FY)",
                      "Altman Z-Score (FQ)", "Altman Z-Score (LTM)", "P/TBV (LTM)", "Return On Equity % (LTM)",
                      "Return On Equity % (FY)", "Current Ratio (FY)", "Current Ratio (LTM)", "Asset Turnover (FY)",
                      "Asset Turnover (LTM)", "EPS Norm - Est Avg (NTM)", "EPS/Adj. (-1FY)", "EPS/Adj. (FY)",
                      "EPS/Adj. (LTM)", "EPS Norm - Est Avg (FY1E)", "Return on Assets (ROA) % (LTM)",
                      "Return on Assets (ROA) % (FY)", "P/B (LTM)", "P/B (-1FY)", "P/B (5YAVG)", "EV/Sales (EST FY1)",
                      "EV/Sales (LTM)", "EV/Sales (NTM)", "EV/Sales (-1FYLTM)", "EV/Sales (-2FYLTM)",
                      "EV/Sales (-3FYLTM)", "EV/Sales (3YAVGLTM)", "EV/Sales (-1FQLTM)", "EV/Sales (-2FQLTM)",
                      "EV/Sales (-3FQLTM)", "EV/Sales (-4FQLTM)", "EV/EBITDA (LTM)", "EV/EBITDA (NTM)",
                      "EV/EBITDA (-1FYLTM)", "EV/EBITDA (-1FQLTM)", "EV/EBITDA (3YAVGLTM)", "EV/EBITDA (EST FY1)",
                      "P/E (EST FY1)", "P/E (-1FYLTM)", "P/E (-2FYLTM)", "P/E (-3FYLTM)", "P/E (3YAVGLTM)",
                      "P/E (-1FQLTM)", "P/E (-2FQLTM)", "P/E (-3FQLTM)", "P/E (5YAVGLTM)", "P/E (-0FQQoQLTM)",
                      "P/E (-0FYYoYLTM)", "P/E (-1FYYoYLTM)", "P/E (-0FQYoYLTM)", "Net EPS - Basic (LTM)",
                      "Net EPS - Basic (FQ)", "Net EPS - Basic (FY)", "Net EPS - Basic (-1FQFQ)",
                      "Net EPS - Basic (-2FQFQ)", "Net EPS - Basic (-3FQFQ)", "Net EPS - Basic (-4FQFQ)",
                      "Net EPS - Basic (-1FY)", "Net EPS - Basic (-2FY)", "Net EPS - Basic (-3FY)",
                      "Net EPS - Basic (-4FY)", "Net EPS - Basic (-5FY)", "EPS GAAP - Est Avg (NTM)",
                      "EPS GAAP - Est Avg (FY1E)", "Total Return (YTD)", "Beta (1Y)", "Beta (2Y)", "Beta (5Y)",
                      "Total Revenues/CAGR (5Y FY)", "Tot. Return %/CAGR (3Y)", "Tot. Return %/CAGR (10Y)",
                      "Total Return (5Y)", "Total Return (10Y)", "Net Income Margin % (FY)",
                      "Net Income Margin % (LTM)", "Volatility (1M)", "Volatility (3M)", "Volatility (6M)",
                      "Volatility (1Y)", "Div Yield (Ind)", "Div Yield (LTM)",
                      "Gross Profit Margin % (FY)",
                      "Gross Profit Margin % (LTM)", "Buyback Yield (LTM)", "Div Yield (-1FYInd)", "Div Yield (TTM)",
                      "Div Yield (NTM)", "Div Yield (5YAVGLTM)", "Revenues - Est YoY % (FY1E)", "Price Chg. % (1M)",
                      "Price Chg. % (3M)", "1-Day %", "EPS Est Avg Rev % (FY1E - 1W)", "EPS Est Avg Rev % (FY1E - 1M)",
                      "EPS Est Avg Rev % (FY1E - 3M)", "EPS Est Avg Rev % (FY1E - 6M)", "EPS Est Avg Rev % (FY1E - 1Y)",
                      "Div Yield (-2FYInd)", "Div Yield (-3FYInd)", "Div Yield (-4FYInd)", "Div Yield (-5FYInd)",
                      "EPS GAAP Est Avg Rev % (FY1E - 1M)", "EPS GAAP Est Avg Rev % (FY1E - 3M)",
                      "EPS GAAP Est Avg Rev % (FY1E - 6M)", "EPS GAAP Est Avg Rev % (FY1E - 1Y)", "Dividend Streak",
                      "Price Target - #", "Analyst Rating", "# Strong Sell Ratings", "# Strong Buys Ratings",
                      "# Hold Ratings", "# Buys Ratings", "# Sell Ratings", "Shrs Out", "Shrs Out (-1FY)",
                      "Full Time Employees (FQ)", "Full Time Employees (FY)", "Full Time Employees (-1FY)",
                      "Full Time Employees (-2FY)", "Full Time Employees (-3FY)", "Avg Employees (5YAVGFY)",
                      "EPS Norm - Est # (FY1E)", "Price Target - # (3M Ago)", "Price Target - # (6M Ago)",
                      "Price Target - # (YTD Ago)", "Price Target - # (1Y Ago)", "Price Target - # (1W Ago)",
                      "Price Target - # (1M Ago)", "Price Target - # (MTD Ago)", "Price Target - # (QTD Ago)",
                      "Gain (Loss) On Sale Of Assets (LTM)", "Impairment of Goodwill (FQ)",
                      "Impairment of Goodwill (LTM)", "Impairment of Goodwill (-1FY)", "Impairment of Goodwill (FY)",
                      "Asset Writedown (LTM)", "Asset Writedown (FY)", "Asset Writedown (-1FY)",
                      "Restructuring Charges (LTM)", "Restructuring Charges (FQ)", "Restructuring Charges (-1FY)",
                      "Restructuring Charges (FY)", "Merger & Restructuring Charges (LTM)",
                      "Other Unusual Items/Total (LTM)", "Asset Writedown (FQ)", "Asset Writedown (5YAVGFQ)",
                      "Impairment of Goodwill (5YAVGFQ)", "Restructuring Charges (5YAVGFQ)",
                      "Merger & Restructuring Charges (FQ)", "Merger & Restructuring Charges (FY)",
                      "Merger & Restructuring Charges (5YAVGFQ)", "Description", "Fiscal Month", "Fiscal Quarter",
                      "Fiscal Year", "Reporting Lag", "Next Income Statement Report Date", "Reporting Interval",
                      "Earnings Report (Frequency)",
    -- NEW COLUMNS
                      "Impairment of Goodwill (-1FQFQ)", "Impairment of Goodwill (-2FQFQ)",
                      "Impairment of Goodwill (-3FQFQ)", "Impairment of Goodwill (-4FQFQ)",
                      "Impairment of Goodwill (-2FY)", "Impairment of Goodwill (-3FY)", "Impairment of Goodwill (-4FY)",
                      "Asset Writedown (-1FQFQ)", "Asset Writedown (-2FQFQ)", "Asset Writedown (-3FQFQ)",
                      "Asset Writedown (-4FQFQ)", "Asset Writedown (-2FY)", "Asset Writedown (-3FY)",
                      "Asset Writedown (-4FY)", "Asset Writedown (-5FY)",
                      "Gain (Loss) On Sale Of Assets (FQ)", "Gain (Loss) On Sale Of Assets (FY)",
                      "Gain (Loss) On Sale Of Assets (-1FQFQ)", "Gain (Loss) On Sale Of Assets (-2FQFQ)",
                      "Gain (Loss) On Sale Of Assets (-3FQFQ)", "Gain (Loss) On Sale Of Assets (-4FQFQ)",
                      "Gain (Loss) On Sale Of Assets (-1FY)", "Gain (Loss) On Sale Of Assets (-2FY)",
                      "Gain (Loss) On Sale Of Assets (-3FY)", "Gain (Loss) On Sale Of Assets (-4FY)",
                      "Restructuring Charges (-1FQFQ)", "Restructuring Charges (-2FQFQ)",
                      "Restructuring Charges (-3FQFQ)", "Restructuring Charges (-4FQFQ)",
                      "Restructuring Charges (-2FY)", "Restructuring Charges (-3FY)", "Restructuring Charges (-4FY)",
                      "Net Income - (IS) (-1FQFQ)", "Net Income - (IS) (-2FQFQ)", "Net Income - (IS) (-3FQFQ)",
                      "Net Income - (IS) (-4FQFQ)", "Net Income - (IS) (-2FY)", "Net Income - (IS) (-3FY)",
                      "Net Income - (IS) (-4FY)",
                      "Normalized Net Income (-1FQFQ)", "Normalized Net Income (-2FQFQ)",
                      "Normalized Net Income (-3FQFQ)", "Normalized Net Income (-4FQFQ)",
                      "Normalized Net Income (-2FY)", "Normalized Net Income (-3FY)", "Normalized Net Income (-4FY)",
                      "Net Income/Adj. (-1FQFQ)", "Net Income/Adj. (-2FQFQ)", "Net Income/Adj. (-3FQFQ)",
                      "Net Income/Adj. (-4FQFQ)", "Net Income/Adj. (-2FY)", "Net Income/Adj. (-3FY)",
                      "Net Income/Adj. (-4FY)",
                      "EBIT (-1FQFQ)", "EBIT (-2FQFQ)", "EBIT (-3FQFQ)", "EBIT (-4FQFQ)",
                      "EBIT (-2FY)", "EBIT (-3FY)", "EBIT (-4FY)",
                      "EBIT/Adj. (FQ)", "EBIT/Adj. (-1FQFQ)", "EBIT/Adj. (-2FQFQ)", "EBIT/Adj. (-3FQFQ)",
                      "EBIT/Adj. (-4FQFQ)", "EBIT/Adj. (-2FY)", "EBIT/Adj. (-3FY)", "EBIT/Adj. (-4FY)",
                      "EBITDA (-1FQFQ)", "EBITDA (-2FQFQ)", "EBITDA (-3FQFQ)", "EBITDA (-4FQFQ)",
                      "EBITDA (-2FY)", "EBITDA (-3FY)", "EBITDA (-4FY)",
                      "EBITDA/Adj. (FQ)", "EBITDA/Adj. (-1FQFQ)", "EBITDA/Adj. (-2FQFQ)", "EBITDA/Adj. (-3FQFQ)",
                      "EBITDA/Adj. (-4FQFQ)", "EBITDA/Adj. (-2FY)", "EBITDA/Adj. (-3FY)", "EBITDA/Adj. (-4FY)",
                      "Basic EPS - Cont (LTM)", "Basic EPS - Cont (FQ)", "Basic EPS - Cont (FY)",
                      "Basic EPS - Cont (-1FQFQ)", "Basic EPS - Cont (-2FQFQ)", "Basic EPS - Cont (-3FQFQ)",
                      "Basic EPS - Cont (-4FQFQ)", "Basic EPS - Cont (-1FY)", "Basic EPS - Cont (-2FY)",
                      "Basic EPS - Cont (-3FY)", "Basic EPS - Cont (-4FY)",
                      "EPS/Adj. (FQ)", "EPS/Adj. (-1FQFQ)", "EPS/Adj. (-2FQFQ)", "EPS/Adj. (-3FQFQ)",
                      "EPS/Adj. (-4FQFQ)", "EPS/Adj. (-2FY)", "EPS/Adj. (-3FY)", "EPS/Adj. (-4FY)",
                      "Total Revenues (-1FQFQ)", "Total Revenues (-2FQFQ)", "Total Revenues (-3FQFQ)",
                      "Total Revenues (-4FQFQ)", "Total Revenues (-2FY)", "Total Revenues (-3FY)",
                      "Total Revenues (-4FY)")
SELECT NULLIF(TRIM(s."Ticker"), '')                                              AS "Ticker",
       NULLIF(TRIM(s."ISIN"), '')                                                AS "ISIN",
       NULLIF(TRIM(s."Name"), '')                                                AS "Name",
       COALESCE(NULLIF(TRIM(s."Region"), ''), 'n/a')                             AS "Region",
       COALESCE(NULLIF(TRIM(s."Country"), ''), 'n/a')                            AS "Country",
       COALESCE(NULLIF(TRIM(s."Trading Country"), ''), 'n/a')                    AS "Trading Country",
       COALESCE(NULLIF(TRIM(s."Exchange"), ''), 'n/a')                           AS "Exchange",
       COALESCE(NULLIF(TRIM(s."Unit"), ''), 'n/a')                               AS "Unit",
       COALESCE(NULLIF(TRIM(s."Sector"), ''), 'n/a')                             AS "Sector",
       COALESCE(NULLIF(TRIM(s."Industry"), ''), 'n/a')                           AS "Industry",
       COALESCE(NULLIF(TRIM(s."Style Class"), ''), 'n/a')                        AS "Style Class",
       COALESCE(NULLIF(TRIM(s."Size Class"), ''), 'n/a')                         AS "Size Class",
       COALESCE(NULLIF(TRIM(s."FY End"), ''), 'n/a')                             AS "FY End",
       COALESCE(NULLIF(TRIM(s."Next Earnings (When)"), ''), 'n/a')               AS "Next Earnings (When)",
       COALESCE(NULLIF(TRIM(s."Next Earnings (Status)"), ''), 'n/a')             AS "Next Earnings (Status)",
       COALESCE(NULLIF(TRIM(s."Dividend Record (Currency)"), ''), 'n/a')         AS "Dividend Record (Currency)",
       COALESCE(NULLIF(TRIM(s."Dividend Record (Frequency)"), ''), 'n/a')        AS "Dividend Record (Frequency)",
       'Q' || current_fiscal.fiscal_quarter || ' ' || current_fiscal.fiscal_year AS "Current Fiscal Quarter",
       'Q' || report_fiscal.next_quarter || ' ' ||
       report_fiscal.next_quarter_year                                           AS "Next Fiscal Quarter",
       report_fiscal.next_earnings_report_type                                   AS "Next Earnings (Report)",
       NULLIF(TRIM(s."Last Updated"), '')::DATE                                  AS "Last Updated",
       NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE                  AS "Income Statement Report Date",
       NULLIF(TRIM(s."Next Earnings"), '')::DATE                                 AS "Next Earnings",
       NULLIF(TRIM(s."Dividend Record (Announce Date)"), '')::DATE               AS "Dividend Record (Announce Date)",
       NULLIF(TRIM(s."Dividend Record (Payable Date)"), '')::DATE                AS "Dividend Record (Payable Date)",
       NULLIF(TRIM(s."Dividend Record (Record Date)"), '')::DATE                 AS "Dividend Record (Record Date)",
       NULLIF(TRIM(s."Dividend Record (Ex Date)"), '')::DATE                     AS "Dividend Record (Ex Date)",
       CURRENT_DATE                                                              AS "Reference Date",
       parsed.fy_end_date                                                        AS "FY End Date",
       next_fy.next_fy_end_date                                                  AS "Next FY End Date",
       text_to_numeric_safe(s."Price Target")                                    AS "Price Target",
       text_to_numeric_safe(s."Price Target - Median")                           AS "Price Target - Median",
       COALESCE(text_to_numeric_safe(s."Dividend Record (Amount)"), 0)           AS "Dividend Record (Amount)",
       text_to_numeric_safe(s."Market Cap")                                      AS "Market Cap",
       text_to_numeric_safe(s."Enterprise Value")                                AS "Enterprise Value",
       text_to_numeric_safe(s."Last Price")                                      AS "Last Price",
       text_to_numeric_safe(s."Price Target (YTD Ago)")                          AS "Price Target (YTD Ago)",
       text_to_numeric_safe(s."Price Target - Low")                              AS "Price Target - Low",
       text_to_numeric_safe(s."Price Target - High")                             AS "Price Target - High",
       text_to_numeric_safe(s."Market Cap (Country R)")                          AS "Market Cap (Country R)",
       text_to_numeric_safe(s."Volume (Shrs)")                                   AS "Volume (Shrs)",
       COALESCE(text_to_numeric_safe(s."Dividend Per Share (LTM)"), 0)           AS "Dividend Per Share (LTM)",
       text_to_numeric_safe(s."Price (5D Ago)")                                  AS "Price (5D Ago)",
       text_to_numeric_safe(s."Price (1W Ago)")                                  AS "Price (1W Ago)",
       text_to_numeric_safe(s."Price (1M Ago)")                                  AS "Price (1M Ago)",
       text_to_numeric_safe(s."Price (3M Ago)")                                  AS "Price (3M Ago)",
       text_to_numeric_safe(s."Price (6M Ago)")                                  AS "Price (6M Ago)",
       text_to_numeric_safe(s."Price (1Y Ago)")                                  AS "Price (1Y Ago)",
       text_to_numeric_safe(s."Price (3Y Ago)")                                  AS "Price (3Y Ago)",
       text_to_numeric_safe(s."Price (5Y Ago)")                                  AS "Price (5Y Ago)",
       text_to_numeric_safe(s."Price (QTD Ago)")                                 AS "Price (QTD Ago)",
       text_to_numeric_safe(s."Rel. Volume")                                     AS "Rel. Volume",
       text_to_numeric_safe(s."52W High/Adj")                                    AS "52W High/Adj",
       text_to_numeric_safe(s."52W Low/Adj")                                     AS "52W Low/Adj",
       text_to_numeric_safe(s."EMA (20D)")                                       AS "EMA (20D)",
       text_to_numeric_safe(s."EMA (50D)")                                       AS "EMA (50D)",
       text_to_numeric_safe(s."EMA (100D)")                                      AS "EMA (100D)",
       text_to_numeric_safe(s."EMA (250D)")                                      AS "EMA (250D)",
       text_to_numeric_safe(s."Price Target (1W Ago)")                           AS "Price Target (1W Ago)",
       text_to_numeric_safe(s."Price Target (1M Ago)")                           AS "Price Target (1M Ago)",
       text_to_numeric_safe(s."Price Target (3M Ago)")                           AS "Price Target (3M Ago)",
       text_to_numeric_safe(s."Price Target (6M Ago)")                           AS "Price Target (6M Ago)",
       text_to_numeric_safe(s."Price Target (MTD Ago)")                          AS "Price Target (MTD Ago)",
       text_to_numeric_safe(s."Price Target (QTD Ago)")                          AS "Price Target (QTD Ago)",
       text_to_numeric_safe(s."Price Target (1Y Ago)")                           AS "Price Target (1Y Ago)",
       text_to_numeric_safe(s."Price Target - High (1W Ago)")                    AS "Price Target - High (1W Ago)",
       text_to_numeric_safe(s."Price Target - High (1M Ago)")                    AS "Price Target - High (1M Ago)",
       text_to_numeric_safe(s."Price Target - High (6M Ago)")                    AS "Price Target - High (6M Ago)",
       text_to_numeric_safe(s."Price Target - High (MTD Ago)")                   AS "Price Target - High (MTD Ago)",
       text_to_numeric_safe(s."Price Target - High (3M Ago)")                    AS "Price Target - High (3M Ago)",
       text_to_numeric_safe(s."Price Target - High (QTD Ago)")                   AS "Price Target - High (QTD Ago)",
       text_to_numeric_safe(s."Price Target - High (1Y Ago)")                    AS "Price Target - High (1Y Ago)",
       text_to_numeric_safe(s."Price Target - High (YTD Ago)")                   AS "Price Target - High (YTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (1W Ago)")                     AS "Price Target - Low (1W Ago)",
       text_to_numeric_safe(s."Price Target - Low (1M Ago)")                     AS "Price Target - Low (1M Ago)",
       text_to_numeric_safe(s."Price Target - Low (3M Ago)")                     AS "Price Target - Low (3M Ago)",
       text_to_numeric_safe(s."Price Target - Low (6M Ago)")                     AS "Price Target - Low (6M Ago)",
       text_to_numeric_safe(s."Price Target - Low (MTD Ago)")                    AS "Price Target - Low (MTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (QTD Ago)")                    AS "Price Target - Low (QTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (YTD Ago)")                    AS "Price Target - Low (YTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (1Y Ago)")                     AS "Price Target - Low (1Y Ago)",
       text_to_numeric_safe(s."Price Target - Median (1W Ago)")                  AS "Price Target - Median (1W Ago)",
       text_to_numeric_safe(s."Price Target - Median (1M Ago)")                  AS "Price Target - Median (1M Ago)",
       text_to_numeric_safe(s."Price Target - Median (3M Ago)")                  AS "Price Target - Median (3M Ago)",
       text_to_numeric_safe(s."Price Target - Median (6M Ago)")                  AS "Price Target - Median (6M Ago)",
       text_to_numeric_safe(s."Price Target - Median (MTD Ago)")                 AS "Price Target - Median (MTD Ago)",
       text_to_numeric_safe(s."Price Target - Median (QTD Ago)")                 AS "Price Target - Median (QTD Ago)",
       text_to_numeric_safe(s."Price Target - Median (YTD Ago)")                 AS "Price Target - Median (YTD Ago)",
       text_to_numeric_safe(s."Price Target - Median (1Y Ago)")                  AS "Price Target - Median (1Y Ago)",
       COALESCE(text_to_numeric_safe(s."Total Revenues (FQ)"), 0)                AS "Total Revenues (FQ)",
       COALESCE(text_to_numeric_safe(s."Total Revenues (-1FY)"), 0)              AS "Total Revenues (-1FY)",
       COALESCE(text_to_numeric_safe(s."Total Revenues (FY)"), 0)                AS "Total Revenues (FY)",
       COALESCE(text_to_numeric_safe(s."Total Revenues (LTM)"), 0)               AS "Total Revenues (LTM)",
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (LTM)"),
                0)                                                               AS "Total Operating Expenses (LTM)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-1FY)"), 0)             AS "Net Income/Adj. (-1FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA (FQ)"), 0)                        AS "EBITDA (FQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA (LTM)"), 0)                       AS "EBITDA (LTM)",
       COALESCE(text_to_numeric_safe(s."EBITDA (FY)"), 0)                        AS "EBITDA (FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA (-1FY)"), 0)                      AS "EBITDA (-1FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (LTM)"), 0)                  AS "EBITDA/Adj. (LTM)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (FY)"), 0)                   AS "EBITDA/Adj. (FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-1FY)"), 0)                 AS "EBITDA/Adj. (-1FY)",
       COALESCE(text_to_numeric_safe(s."EBIT (FQ)"), 0)                          AS "EBIT (FQ)",
       COALESCE(text_to_numeric_safe(s."EBIT (LTM)"), 0)                         AS "EBIT (LTM)",
       COALESCE(text_to_numeric_safe(s."EBIT (FY)"), 0)                          AS "EBIT (FY)",
       COALESCE(text_to_numeric_safe(s."EBIT (-1FY)"), 0)                        AS "EBIT (-1FY)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-1FY)"), 0)                   AS "EBIT/Adj. (-1FY)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (FY)"), 0)                     AS "EBIT/Adj. (FY)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (LTM)"), 0)                    AS "EBIT/Adj. (LTM)",
       COALESCE(text_to_numeric_safe(s."EBIT - Est Med (FY1E)"), 0)              AS "EBIT - Est Med (FY1E)",
       COALESCE(text_to_numeric_safe(s."EBIT - Est Med (NTM)"), 0)               AS "EBIT - Est Med (NTM)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (FY)"), 0)             AS "Net Income - (IS) (FY)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (LTM)"), 0)            AS "Net Income - (IS) (LTM)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (FY)"), 0)         AS "Normalized Net Income (FY)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (LTM)"), 0)        AS "Normalized Net Income (LTM)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (FY)"), 0)               AS "Net Income/Adj. (FY)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (LTM)"), 0)              AS "Net Income/Adj. (LTM)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (LTM)"), 0)                 AS "Gross Profit (LTM)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (FY)"), 0)                  AS "Gross Profit (FY)",
       COALESCE(text_to_numeric_safe(s."Cost Of Revenues (LTM)"), 0)             AS "Cost Of Revenues (LTM)",
       COALESCE(text_to_numeric_safe(s."Operating Income (LTM)"), 0)             AS "Operating Income (LTM)",
       COALESCE(text_to_numeric_safe(s."Operating Income (FY)"), 0)              AS "Operating Income (FY)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (LTM)"), 0)                 AS "R&D Expenses (LTM)",
       COALESCE(text_to_numeric_safe(s."Interest Expense/Total (LTM)"), 0)       AS "Interest Expense/Total (LTM)",
       COALESCE(text_to_numeric_safe(s."Interest Income On Investments (LTM)"),
                0)                                                               AS "Interest Income On Investments (LTM)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-1FY)"), 0)           AS "Net Income - (IS) (-1FY)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-1FY)"), 0)       AS "Normalized Net Income (-1FY)",
       COALESCE(text_to_numeric_safe(s."Total Revenues (5YAVGFQ)"), 0)           AS "Total Revenues (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA (5YAVGFQ)"), 0)                   AS "EBITDA (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT (5YAVGFQ)"), 0)                     AS "EBIT (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Operating Income (FQ)"), 0)              AS "Operating Income (FQ)",
       COALESCE(text_to_numeric_safe(s."Operating Income (5YAVGFQ)"), 0)         AS "Operating Income (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (FQ)"), 0)         AS "Normalized Net Income (FQ)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (5YAVGFQ)"),
                0)                                                               AS "Normalized Net Income (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (FQ)"), 0)               AS "Net Income/Adj. (FQ)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (5YAVGFQ)"), 0)          AS "Net Income/Adj. (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (FQ)"), 0)             AS "Net Income - (IS) (FQ)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (5YAVGFQ)"), 0)        AS "Net Income - (IS) (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (5YAVGLTM)"), 0)       AS "Net Income - (IS) (5YAVGLTM)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (5YAVGLTM)"),
                0)                                                               AS "Normalized Net Income (5YAVGLTM)",
       COALESCE(text_to_numeric_safe(s."EBITDA (5YAVGLTM)"), 0)                  AS "EBITDA (5YAVGLTM)",
       COALESCE(text_to_numeric_safe(s."EBIT (5YAVGLTM)"), 0)                    AS "EBIT (5YAVGLTM)",
       COALESCE(text_to_numeric_safe(s."Total Revenues (5YAVGLTM)"), 0)          AS "Total Revenues (5YAVGLTM)",
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (FQ)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (FQ)",
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (FY)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (FY)",
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (-1FY)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (-1FY)",
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (5YAVGFQ)"),
                0)                                                               AS "Selling General & Admin Expenses/Total (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (FQ)"), 0)            AS "Marketing Expenses (FQ)",
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (FY)"), 0)            AS "Marketing Expenses (FY)",
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (-1FY)"), 0)          AS "Marketing Expenses (-1FY)",
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (5YAVGLTM)"),
                0)                                                               AS "Marketing Expenses (5YAVGLTM)",
       COALESCE(text_to_numeric_safe(s."Revenues - Est Avg (NTM)"), 0)           AS "Revenues - Est Avg (NTM)",
       COALESCE(text_to_numeric_safe(s."Revenues - Est Avg (FY1E)"), 0)          AS "Revenues - Est Avg (FY1E)",
       COALESCE(text_to_numeric_safe(s."Revenues - Est Med (NTM)"), 0)           AS "Revenues - Est Med (NTM)",
       COALESCE(text_to_numeric_safe(s."Revenues - Est Med (FY1E)"), 0)          AS "Revenues - Est Med (FY1E)",
       COALESCE(text_to_numeric_safe(s."EBITDA - Est Avg (NTM)"), 0)             AS "EBITDA - Est Avg (NTM)",
       COALESCE(text_to_numeric_safe(s."EBITDA - Est Avg (FY1E)"), 0)            AS "EBITDA - Est Avg (FY1E)",
       COALESCE(text_to_numeric_safe(s."TBV (FY)"), 0)                           AS "TBV (FY)",
       COALESCE(text_to_numeric_safe(s."TBV (LTM)"), 0)                          AS "TBV (LTM)",
       COALESCE(text_to_numeric_safe(s."Total Debt (FY)"), 0)                    AS "Total Debt (FY)",
       COALESCE(text_to_numeric_safe(s."Total Equity (FY)"), 0)                  AS "Total Equity (FY)",
       COALESCE(text_to_numeric_safe(s."Total Equity (LTM)"), 0)                 AS "Total Equity (LTM)",
       COALESCE(text_to_numeric_safe(s."Total Debt (LTM)"), 0)                   AS "Total Debt (LTM)",
       COALESCE(text_to_numeric_safe(s."Total Assets (LTM)"), 0)                 AS "Total Assets (LTM)",
       COALESCE(text_to_numeric_safe(s."Total Assets (FY)"), 0)                  AS "Total Assets (FY)",
       COALESCE(text_to_numeric_safe(s."Inventory (LTM)"), 0)                    AS "Inventory (LTM)",
       COALESCE(text_to_numeric_safe(s."Goodwill (FQ)"), 0)                      AS "Goodwill (FQ)",
       COALESCE(text_to_numeric_safe(s."Goodwill (LTM)"), 0)                     AS "Goodwill (LTM)",
       COALESCE(text_to_numeric_safe(s."Goodwill (FY)"), 0)                      AS "Goodwill (FY)",
       COALESCE(text_to_numeric_safe(s."Goodwill (-1FY)"), 0)                    AS "Goodwill (-1FY)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (LTM)"), 0)            AS "Retained Earnings (LTM)",
       COALESCE(text_to_numeric_safe(s."Total Current Assets (LTM)"), 0)         AS "Total Current Assets (LTM)",
       COALESCE(text_to_numeric_safe(s."Total Current Liabilities (LTM)"),
                0)                                                               AS "Total Current Liabilities (LTM)",
       COALESCE(text_to_numeric_safe(s."Working Capital (LTM)"), 0)              AS "Working Capital (LTM)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (LTM)"), 0)         AS "Cash And Equivalents (LTM)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (FQ)"), 0)          AS "Cash And Equivalents (FQ)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (FY)"), 0)          AS "Cash And Equivalents (FY)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (5YAVGFQ)"),
                0)                                                               AS "Cash And Equivalents (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Inventory (FQ)"), 0)                     AS "Inventory (FQ)",
       COALESCE(text_to_numeric_safe(s."Inventory (FY)"), 0)                     AS "Inventory (FY)",
       COALESCE(text_to_numeric_safe(s."Goodwill (5YAVGFQ)"), 0)                 AS "Goodwill (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Inventory (5YAVGFQ)"), 0)                AS "Inventory (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (FQ)"), 0)             AS "Retained Earnings (FQ)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (FY)"), 0)             AS "Retained Earnings (FY)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (5YAVGFQ)"), 0)        AS "Retained Earnings (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Working Capital (FQ)"), 0)               AS "Working Capital (FQ)",
       COALESCE(text_to_numeric_safe(s."Working Capital (FY)"), 0)               AS "Working Capital (FY)",
       COALESCE(text_to_numeric_safe(s."Working Capital (5YAVGFY)"), 0)          AS "Working Capital (5YAVGFY)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (LTM)"),
                0)                                                               AS "Gross Intangible Assets (LTM)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (FY)"), 0)       AS "Gross Intangible Assets (FY)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (5YAVGFQ)"),
                0)                                                               AS "Gross Intangible Assets (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Accounts Receivable/Total (FY)"),
                0)                                                               AS "Accounts Receivable/Total (FY)",
       COALESCE(text_to_numeric_safe(s."Accounts Receivable/Total (-1FY)"),
                0)                                                               AS "Accounts Receivable/Total (-1FY)",
       COALESCE(text_to_numeric_safe(s."Accounts Receivable/Total (5YAVGFQ)"),
                0)                                                               AS "Accounts Receivable/Total (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."CFF (LTM)"), 0)                          AS "CFF (LTM)",
       COALESCE(text_to_numeric_safe(s."CFI (LTM)"), 0)                          AS "CFI (LTM)",
       COALESCE(text_to_numeric_safe(s."FCF (LTM)"), 0)                          AS "FCF (LTM)",
       COALESCE(text_to_numeric_safe(s."CFO (LTM)"), 0)                          AS "CFO (LTM)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (LTM)"), 0)            AS "Cash Acquisitions (LTM)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (FY)"), 0)             AS "Cash Acquisitions (FY)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-1FY)"), 0)           AS "Cash Acquisitions (-1FY)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (LTM)"), 0)          AS "Capital Expenditure (LTM)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-1FY)"), 0)         AS "Capital Expenditure (-1FY)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (FY)"), 0)           AS "Capital Expenditure (FY)",
       COALESCE(text_to_numeric_safe(s."CFF (FY)"), 0)                           AS "CFF (FY)",
       COALESCE(text_to_numeric_safe(s."CFF (-1FY)"), 0)                         AS "CFF (-1FY)",
       COALESCE(text_to_numeric_safe(s."CFI (FY)"), 0)                           AS "CFI (FY)",
       COALESCE(text_to_numeric_safe(s."CFI (-1FY)"), 0)                         AS "CFI (-1FY)",
       COALESCE(text_to_numeric_safe(s."CFO (FY)"), 0)                           AS "CFO (FY)",
       COALESCE(text_to_numeric_safe(s."CFO (-1FY)"), 0)                         AS "CFO (-1FY)",
       COALESCE(text_to_numeric_safe(s."FCF (FY)"), 0)                           AS "FCF (FY)",
       COALESCE(text_to_numeric_safe(s."FCF (-1FY)"), 0)                         AS "FCF (-1FY)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (FQ)"), 0)           AS "Capital Expenditure (FQ)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (5YAVGFQ)"),
                0)                                                               AS "Capital Expenditure (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."CFF (FQ)"), 0)                           AS "CFF (FQ)",
       COALESCE(text_to_numeric_safe(s."CFI (FQ)"), 0)                           AS "CFI (FQ)",
       COALESCE(text_to_numeric_safe(s."CFO (FQ)"), 0)                           AS "CFO (FQ)",
       COALESCE(text_to_numeric_safe(s."FCF (FQ)"), 0)                           AS "FCF (FQ)",
       COALESCE(text_to_numeric_safe(s."FCF (5YAVGFQ)"), 0)                      AS "FCF (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (FQ)"), 0)             AS "Cash Acquisitions (FQ)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (5YAVGFQ)"), 0)        AS "Cash Acquisitions (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Common Dividends Paid (LTM)"), 0)        AS "Common Dividends Paid (LTM)",
       COALESCE(text_to_numeric_safe(s."Common Dividends Paid (FY)"), 0)         AS "Common Dividends Paid (FY)",
       COALESCE(text_to_numeric_safe(s."CFO (-1FQFQ)"), 0)                       AS "CFO (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFO (-2FQFQ)"), 0)                       AS "CFO (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFO (-3FQFQ)"), 0)                       AS "CFO (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFO (-4FQFQ)"), 0)                       AS "CFO (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFI (-1FQFQ)"), 0)                       AS "CFI (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFI (-2FQFQ)"), 0)                       AS "CFI (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFI (-3FQFQ)"), 0)                       AS "CFI (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFI (-4FQFQ)"), 0)                       AS "CFI (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFI (-2FY)"), 0)                         AS "CFI (-2FY)",
       COALESCE(text_to_numeric_safe(s."CFI (-3FY)"), 0)                         AS "CFI (-3FY)",
       COALESCE(text_to_numeric_safe(s."CFI (-4FY)"), 0)                         AS "CFI (-4FY)",
       COALESCE(text_to_numeric_safe(s."FCF (-1FQFQ)"), 0)                       AS "FCF (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."FCF (-2FQFQ)"), 0)                       AS "FCF (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."FCF (-3FQFQ)"), 0)                       AS "FCF (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."FCF (-4FQFQ)"), 0)                       AS "FCF (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFF (-2FY)"), 0)                         AS "CFF (-2FY)",
       COALESCE(text_to_numeric_safe(s."CFF (-3FY)"), 0)                         AS "CFF (-3FY)",
       COALESCE(text_to_numeric_safe(s."CFF (-4FY)"), 0)                         AS "CFF (-4FY)",
       COALESCE(text_to_numeric_safe(s."CFF (-1FQFQ)"), 0)                       AS "CFF (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFF (-2FQFQ)"), 0)                       AS "CFF (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFF (-3FQFQ)"), 0)                       AS "CFF (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFF (-4FQFQ)"), 0)                       AS "CFF (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."CFO (-2FY)"), 0)                         AS "CFO (-2FY)",
       COALESCE(text_to_numeric_safe(s."CFO (-3FY)"), 0)                         AS "CFO (-3FY)",
       COALESCE(text_to_numeric_safe(s."CFO (-4FY)"), 0)                         AS "CFO (-4FY)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-1FQFQ)"), 0)         AS "Cash Acquisitions (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-2FQFQ)"), 0)         AS "Cash Acquisitions (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-3FQFQ)"), 0)         AS "Cash Acquisitions (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-4FQFQ)"), 0)         AS "Cash Acquisitions (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."FCF (-2FY)"), 0)                         AS "FCF (-2FY)",
       COALESCE(text_to_numeric_safe(s."FCF (-3FY)"), 0)                         AS "FCF (-3FY)",
       COALESCE(text_to_numeric_safe(s."FCF (-4FY)"), 0)                         AS "FCF (-4FY)",
       -- NEW: Cash Acquisitions Historical (FY)
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-2FY)"), 0)           AS "Cash Acquisitions (-2FY)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-3FY)"), 0)           AS "Cash Acquisitions (-3FY)",
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-4FY)"), 0)           AS "Cash Acquisitions (-4FY)",
       -- NEW: Capital Expenditure Historical
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-1FQFQ)"), 0)       AS "Capital Expenditure (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-2FQFQ)"), 0)       AS "Capital Expenditure (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-3FQFQ)"), 0)       AS "Capital Expenditure (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-4FQFQ)"), 0)       AS "Capital Expenditure (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-2FY)"), 0)         AS "Capital Expenditure (-2FY)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-3FY)"), 0)         AS "Capital Expenditure (-3FY)",
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-4FY)"), 0)         AS "Capital Expenditure (-4FY)",
       -- NEW: Working Capital Historical
       COALESCE(text_to_numeric_safe(s."Working Capital (-1FQ)"), 0)             AS "Working Capital (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Working Capital (-2FQ)"), 0)             AS "Working Capital (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Working Capital (-3FQ)"), 0)             AS "Working Capital (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Working Capital (-4FQ)"), 0)             AS "Working Capital (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Working Capital (-1FY)"), 0)             AS "Working Capital (-1FY)",
       COALESCE(text_to_numeric_safe(s."Working Capital (-2FY)"), 0)             AS "Working Capital (-2FY)",
       COALESCE(text_to_numeric_safe(s."Working Capital (-3FY)"), 0)             AS "Working Capital (-3FY)",
       COALESCE(text_to_numeric_safe(s."Working Capital (-4FY)"), 0)             AS "Working Capital (-4FY)",
       -- NEW: Total Debt Historical
       COALESCE(text_to_numeric_safe(s."Total Debt (FQ)"), 0)                    AS "Total Debt (FQ)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-1FQ)"), 0)                  AS "Total Debt (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-2FQ)"), 0)                  AS "Total Debt (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-3FQ)"), 0)                  AS "Total Debt (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-4FQ)"), 0)                  AS "Total Debt (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-1FY)"), 0)                  AS "Total Debt (-1FY)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-2FY)"), 0)                  AS "Total Debt (-2FY)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-3FY)"), 0)                  AS "Total Debt (-3FY)",
       COALESCE(text_to_numeric_safe(s."Total Debt (-4FY)"), 0)                  AS "Total Debt (-4FY)",
       -- NEW: Total Assets Historical
       COALESCE(text_to_numeric_safe(s."Total Assets (FQ)"), 0)                  AS "Total Assets (FQ)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-1FQ)"), 0)                AS "Total Assets (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-2FQ)"), 0)                AS "Total Assets (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-3FQ)"), 0)                AS "Total Assets (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-4FQ)"), 0)                AS "Total Assets (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-1FY)"), 0)                AS "Total Assets (-1FY)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-2FY)"), 0)                AS "Total Assets (-2FY)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-3FY)"), 0)                AS "Total Assets (-3FY)",
       COALESCE(text_to_numeric_safe(s."Total Assets (-4FY)"), 0)                AS "Total Assets (-4FY)",
       -- NEW: Gross Profit Historical
       COALESCE(text_to_numeric_safe(s."Gross Profit (FQ)"), 0)                  AS "Gross Profit (FQ)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-1FQFQ)"), 0)              AS "Gross Profit (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-2FQFQ)"), 0)              AS "Gross Profit (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-3FQFQ)"), 0)              AS "Gross Profit (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-4FQFQ)"), 0)              AS "Gross Profit (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-1FY)"), 0)                AS "Gross Profit (-1FY)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-2FY)"), 0)                AS "Gross Profit (-2FY)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-3FY)"), 0)                AS "Gross Profit (-3FY)",
       COALESCE(text_to_numeric_safe(s."Gross Profit (-4FY)"), 0)                AS "Gross Profit (-4FY)",
       -- NEW: Inventory Historical
       COALESCE(text_to_numeric_safe(s."Inventory (-1FQ)"), 0)                   AS "Inventory (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Inventory (-2FQ)"), 0)                   AS "Inventory (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Inventory (-3FQ)"), 0)                   AS "Inventory (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Inventory (-4FQ)"), 0)                   AS "Inventory (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Inventory (-1FY)"), 0)                   AS "Inventory (-1FY)",
       COALESCE(text_to_numeric_safe(s."Inventory (-2FY)"), 0)                   AS "Inventory (-2FY)",
       COALESCE(text_to_numeric_safe(s."Inventory (-3FY)"), 0)                   AS "Inventory (-3FY)",
       COALESCE(text_to_numeric_safe(s."Inventory (-4FY)"), 0)                   AS "Inventory (-4FY)",
       -- NEW: Goodwill Historical
       COALESCE(text_to_numeric_safe(s."Goodwill (-1FQ)"), 0)                    AS "Goodwill (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Goodwill (-2FQ)"), 0)                    AS "Goodwill (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Goodwill (-3FQ)"), 0)                    AS "Goodwill (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Goodwill (-4FQ)"), 0)                    AS "Goodwill (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Goodwill (-2FY)"), 0)                    AS "Goodwill (-2FY)",
       COALESCE(text_to_numeric_safe(s."Goodwill (-3FY)"), 0)                    AS "Goodwill (-3FY)",
       COALESCE(text_to_numeric_safe(s."Goodwill (-4FY)"), 0)                    AS "Goodwill (-4FY)",
       -- NEW: Operating Income Historical
       COALESCE(text_to_numeric_safe(s."Operating Income (-1FQFQ)"), 0)          AS "Operating Income (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Operating Income (-2FQFQ)"), 0)          AS "Operating Income (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Operating Income (-3FQFQ)"), 0)          AS "Operating Income (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Operating Income (-4FQFQ)"), 0)          AS "Operating Income (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Operating Income (-1FY)"), 0)            AS "Operating Income (-1FY)",
       COALESCE(text_to_numeric_safe(s."Operating Income (-2FY)"), 0)            AS "Operating Income (-2FY)",
       COALESCE(text_to_numeric_safe(s."Operating Income (-3FY)"), 0)            AS "Operating Income (-3FY)",
       COALESCE(text_to_numeric_safe(s."Operating Income (-4FY)"), 0)            AS "Operating Income (-4FY)",
       -- NEW: Retained Earnings Historical
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-1FQ)"), 0)           AS "Retained Earnings (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-2FQ)"), 0)           AS "Retained Earnings (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-3FQ)"), 0)           AS "Retained Earnings (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-4FQ)"), 0)           AS "Retained Earnings (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-1FY)"), 0)           AS "Retained Earnings (-1FY)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-2FY)"), 0)           AS "Retained Earnings (-2FY)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-3FY)"), 0)           AS "Retained Earnings (-3FY)",
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-4FY)"), 0)           AS "Retained Earnings (-4FY)",
       -- NEW: R&D Expenses Historical
       COALESCE(text_to_numeric_safe(s."R&D Expenses (FQ)"), 0)                  AS "R&D Expenses (FQ)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (FY)"), 0)                  AS "R&D Expenses (FY)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-1FQFQ)"), 0)              AS "R&D Expenses (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-2FQFQ)"), 0)              AS "R&D Expenses (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-3FQFQ)"), 0)              AS "R&D Expenses (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-4FQFQ)"), 0)              AS "R&D Expenses (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-1FY)"), 0)                AS "R&D Expenses (-1FY)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-2FY)"), 0)                AS "R&D Expenses (-2FY)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-3FY)"), 0)                AS "R&D Expenses (-3FY)",
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-4FY)"), 0)                AS "R&D Expenses (-4FY)",
       -- NEW: Merger & Restructuring Charges Historical
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-1FQFQ)"),
                0)                                                               AS "Merger & Restructuring Charges (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-2FQFQ)"),
                0)                                                               AS "Merger & Restructuring Charges (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-3FQFQ)"),
                0)                                                               AS "Merger & Restructuring Charges (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-4FQFQ)"),
                0)                                                               AS "Merger & Restructuring Charges (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-1FY)"),
                0)                                                               AS "Merger & Restructuring Charges (-1FY)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-2FY)"),
                0)                                                               AS "Merger & Restructuring Charges (-2FY)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-3FY)"),
                0)                                                               AS "Merger & Restructuring Charges (-3FY)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-4FY)"),
                0)                                                               AS "Merger & Restructuring Charges (-4FY)",
       -- NEW: Cash And Equivalents Historical
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-1FQ)"), 0)        AS "Cash And Equivalents (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-2FQ)"), 0)        AS "Cash And Equivalents (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-3FQ)"), 0)        AS "Cash And Equivalents (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-4FQ)"), 0)        AS "Cash And Equivalents (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-1FY)"), 0)        AS "Cash And Equivalents (-1FY)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-2FY)"), 0)        AS "Cash And Equivalents (-2FY)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-3FY)"), 0)        AS "Cash And Equivalents (-3FY)",
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-4FY)"), 0)        AS "Cash And Equivalents (-4FY)",
       -- NEW: Gross Intangible Assets Historical
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (FQ)"), 0)       AS "Gross Intangible Assets (FQ)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-1FQ)"),
                0)                                                               AS "Gross Intangible Assets (-1FQ)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-2FQ)"),
                0)                                                               AS "Gross Intangible Assets (-2FQ)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-3FQ)"),
                0)                                                               AS "Gross Intangible Assets (-3FQ)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-4FQ)"),
                0)                                                               AS "Gross Intangible Assets (-4FQ)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-1FY)"),
                0)                                                               AS "Gross Intangible Assets (-1FY)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-2FY)"),
                0)                                                               AS "Gross Intangible Assets (-2FY)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-3FY)"),
                0)                                                               AS "Gross Intangible Assets (-3FY)",
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-4FY)"),
                0)                                                               AS "Gross Intangible Assets (-4FY)",
       -- Continue with existing P/E columns
       text_to_numeric_safe(s."P/E (NTM)")                                       AS "P/E (NTM)",
       text_to_numeric_safe(s."P/E (LTM)")                                       AS "P/E (LTM)",
       text_to_numeric_safe(s."Altman Z-Score (FY)")                             AS "Altman Z-Score (FY)",
       text_to_numeric_safe(s."Altman Z-Score (FQ)")                             AS "Altman Z-Score (FQ)",
       text_to_numeric_safe(s."Altman Z-Score (LTM)")                            AS "Altman Z-Score (LTM)",
       text_to_numeric_safe(s."P/TBV (LTM)")                                     AS "P/TBV (LTM)",
       text_to_numeric_safe(s."Return On Equity % (LTM)")                        AS "Return On Equity % (LTM)",
       text_to_numeric_safe(s."Return On Equity % (FY)")                         AS "Return On Equity % (FY)",
       text_to_numeric_safe(s."Current Ratio (FY)")                              AS "Current Ratio (FY)",
       text_to_numeric_safe(s."Current Ratio (LTM)")                             AS "Current Ratio (LTM)",
       text_to_numeric_safe(s."Asset Turnover (FY)")                             AS "Asset Turnover (FY)",
       text_to_numeric_safe(s."Asset Turnover (LTM)")                            AS "Asset Turnover (LTM)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (NTM)")                        AS "EPS Norm - Est Avg (NTM)",
       text_to_numeric_safe(s."EPS/Adj. (-1FY)")                                 AS "EPS/Adj. (-1FY)",
       text_to_numeric_safe(s."EPS/Adj. (FY)")                                   AS "EPS/Adj. (FY)",
       text_to_numeric_safe(s."EPS/Adj. (LTM)")                                  AS "EPS/Adj. (LTM)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY1E)")                       AS "EPS Norm - Est Avg (FY1E)",
       text_to_numeric_safe(s."Return on Assets (ROA) % (LTM)")                  AS "Return on Assets (ROA) % (LTM)",
       text_to_numeric_safe(s."Return on Assets (ROA) % (FY)")                   AS "Return on Assets (ROA) % (FY)",
       text_to_numeric_safe(s."P/B (LTM)")                                       AS "P/B (LTM)",
       text_to_numeric_safe(s."P/B (-1FY)")                                      AS "P/B (-1FY)",
       text_to_numeric_safe(s."P/B (5YAVG)")                                     AS "P/B (5YAVG)",
       text_to_numeric_safe(s."EV/Sales (EST FY1)")                              AS "EV/Sales (EST FY1)",
       text_to_numeric_safe(s."EV/Sales (LTM)")                                  AS "EV/Sales (LTM)",
       text_to_numeric_safe(s."EV/Sales (NTM)")                                  AS "EV/Sales (NTM)",
       text_to_numeric_safe(s."EV/Sales (-1FYLTM)")                              AS "EV/Sales (-1FYLTM)",
       text_to_numeric_safe(s."EV/Sales (-2FYLTM)")                              AS "EV/Sales (-2FYLTM)",
       text_to_numeric_safe(s."EV/Sales (-3FYLTM)")                              AS "EV/Sales (-3FYLTM)",
       text_to_numeric_safe(s."EV/Sales (3YAVGLTM)")                             AS "EV/Sales (3YAVGLTM)",
       text_to_numeric_safe(s."EV/Sales (-1FQLTM)")                              AS "EV/Sales (-1FQLTM)",
       text_to_numeric_safe(s."EV/Sales (-2FQLTM)")                              AS "EV/Sales (-2FQLTM)",
       text_to_numeric_safe(s."EV/Sales (-3FQLTM)")                              AS "EV/Sales (-3FQLTM)",
       text_to_numeric_safe(s."EV/Sales (-4FQLTM)")                              AS "EV/Sales (-4FQLTM)",
       text_to_numeric_safe(s."EV/EBITDA (LTM)")                                 AS "EV/EBITDA (LTM)",
       text_to_numeric_safe(s."EV/EBITDA (NTM)")                                 AS "EV/EBITDA (NTM)",
       text_to_numeric_safe(s."EV/EBITDA (-1FYLTM)")                             AS "EV/EBITDA (-1FYLTM)",
       text_to_numeric_safe(s."EV/EBITDA (-1FQLTM)")                             AS "EV/EBITDA (-1FQLTM)",
       text_to_numeric_safe(s."EV/EBITDA (3YAVGLTM)")                            AS "EV/EBITDA (3YAVGLTM)",
       text_to_numeric_safe(s."EV/EBITDA (EST FY1)")                             AS "EV/EBITDA (EST FY1)",
       text_to_numeric_safe(s."P/E (EST FY1)")                                   AS "P/E (EST FY1)",
       text_to_numeric_safe(s."P/E (-1FYLTM)")                                   AS "P/E (-1FYLTM)",
       text_to_numeric_safe(s."P/E (-2FYLTM)")                                   AS "P/E (-2FYLTM)",
       text_to_numeric_safe(s."P/E (-3FYLTM)")                                   AS "P/E (-3FYLTM)",
       text_to_numeric_safe(s."P/E (3YAVGLTM)")                                  AS "P/E (3YAVGLTM)",
       text_to_numeric_safe(s."P/E (-1FQLTM)")                                   AS "P/E (-1FQLTM)",
       text_to_numeric_safe(s."P/E (-2FQLTM)")                                   AS "P/E (-2FQLTM)",
       text_to_numeric_safe(s."P/E (-3FQLTM)")                                   AS "P/E (-3FQLTM)",
       text_to_numeric_safe(s."P/E (5YAVGLTM)")                                  AS "P/E (5YAVGLTM)",
       text_to_numeric_safe(s."P/E (-0FQQoQLTM)")                                AS "P/E (-0FQQoQLTM)",
       text_to_numeric_safe(s."P/E (-0FYYoYLTM)")                                AS "P/E (-0FYYoYLTM)",
       text_to_numeric_safe(s."P/E (-1FYYoYLTM)")                                AS "P/E (-1FYYoYLTM)",
       text_to_numeric_safe(s."P/E (-0FQYoYLTM)")                                AS "P/E (-0FQYoYLTM)",
       text_to_numeric_safe(s."Net EPS - Basic (LTM)")                           AS "Net EPS - Basic (LTM)",
       text_to_numeric_safe(s."Net EPS - Basic (FQ)")                            AS "Net EPS - Basic (FQ)",
       text_to_numeric_safe(s."Net EPS - Basic (FY)")                            AS "Net EPS - Basic (FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-1FQFQ)")                        AS "Net EPS - Basic (-1FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-2FQFQ)")                        AS "Net EPS - Basic (-2FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-3FQFQ)")                        AS "Net EPS - Basic (-3FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-4FQFQ)")                        AS "Net EPS - Basic (-4FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-1FY)")                          AS "Net EPS - Basic (-1FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-2FY)")                          AS "Net EPS - Basic (-2FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-3FY)")                          AS "Net EPS - Basic (-3FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-4FY)")                          AS "Net EPS - Basic (-4FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-5FY)")                          AS "Net EPS - Basic (-5FY)",
       text_to_numeric_safe(s."EPS GAAP - Est Avg (NTM)")                        AS "EPS GAAP - Est Avg (NTM)",
       text_to_numeric_safe(s."EPS GAAP - Est Avg (FY1E)")                       AS "EPS GAAP - Est Avg (FY1E)",
       text_to_numeric_safe(s."Total Return (YTD)")                              AS "Total Return (YTD)",
       text_to_numeric_safe(s."Beta (1Y)")                                       AS "Beta (1Y)",
       text_to_numeric_safe(s."Beta (2Y)")                                       AS "Beta (2Y)",
       text_to_numeric_safe(s."Beta (5Y)")                                       AS "Beta (5Y)",
       text_to_numeric_safe(s."Total Revenues/CAGR (5Y FY)")                     AS "Total Revenues/CAGR (5Y FY)",
       text_to_numeric_safe(s."Tot. Return %/CAGR (3Y)")                         AS "Tot. Return %/CAGR (3Y)",
       text_to_numeric_safe(s."Tot. Return %/CAGR (10Y)")                        AS "Tot. Return %/CAGR (10Y)",
       text_to_numeric_safe(s."Total Return (5Y)")                               AS "Total Return (5Y)",
       text_to_numeric_safe(s."Total Return (10Y)")                              AS "Total Return (10Y)",
       text_to_numeric_safe(s."Net Income Margin % (FY)")                        AS "Net Income Margin % (FY)",
       text_to_numeric_safe(s."Net Income Margin % (LTM)")                       AS "Net Income Margin % (LTM)",
       text_to_numeric_safe(s."Volatility (1M)")                                 AS "Volatility (1M)",
       text_to_numeric_safe(s."Volatility (3M)")                                 AS "Volatility (3M)",
       text_to_numeric_safe(s."Volatility (6M)")                                 AS "Volatility (6M)",
       text_to_numeric_safe(s."Volatility (1Y)")                                 AS "Volatility (1Y)",
       text_to_numeric_safe(s."Div Yield (Ind)")                                 AS "Div Yield (Ind)",
       text_to_numeric_safe(s."Div Yield (LTM)")                                 AS "Div Yield (LTM)",
       text_to_numeric_safe(s."Gross Profit Margin % (FY)")                      AS "Gross Profit Margin % (FY)",
       text_to_numeric_safe(s."Gross Profit Margin % (LTM)")                     AS "Gross Profit Margin % (LTM)",
       text_to_numeric_safe(s."Buyback Yield (LTM)")                             AS "Buyback Yield (LTM)",
       text_to_numeric_safe(s."Div Yield (-1FYInd)")                             AS "Div Yield (-1FYInd)",
       text_to_numeric_safe(s."Div Yield (TTM)")                                 AS "Div Yield (TTM)",
       text_to_numeric_safe(s."Div Yield (NTM)")                                 AS "Div Yield (NTM)",
       text_to_numeric_safe(s."Div Yield (5YAVGLTM)")                            AS "Div Yield (5YAVGLTM)",
       text_to_numeric_safe(s."Revenues - Est YoY % (FY1E)")                     AS "Revenues - Est YoY % (FY1E)",
       text_to_numeric_safe(s."Price Chg. % (1M)")                               AS "Price Chg. % (1M)",
       text_to_numeric_safe(s."Price Chg. % (3M)")                               AS "Price Chg. % (3M)",
       text_to_numeric_safe(s."1-Day %")                                         AS "1-Day %",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1W)")                   AS "EPS Est Avg Rev % (FY1E - 1W)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1M)")                   AS "EPS Est Avg Rev % (FY1E - 1M)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 3M)")                   AS "EPS Est Avg Rev % (FY1E - 3M)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 6M)")                   AS "EPS Est Avg Rev % (FY1E - 6M)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1Y)")                   AS "EPS Est Avg Rev % (FY1E - 1Y)",
       text_to_numeric_safe(s."Div Yield (-2FYInd)")                             AS "Div Yield (-2FYInd)",
       text_to_numeric_safe(s."Div Yield (-3FYInd)")                             AS "Div Yield (-3FYInd)",
       text_to_numeric_safe(s."Div Yield (-4FYInd)")                             AS "Div Yield (-4FYInd)",
       text_to_numeric_safe(s."Div Yield (-5FYInd)")                             AS "Div Yield (-5FYInd)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1M)")              AS "EPS GAAP Est Avg Rev % (FY1E - 1M)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 3M)")              AS "EPS GAAP Est Avg Rev % (FY1E - 3M)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 6M)")              AS "EPS GAAP Est Avg Rev % (FY1E - 6M)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1Y)")              AS "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
       COALESCE(text_to_numeric_safe(s."Dividend Streak"), 0)                    AS "Dividend Streak",
       COALESCE(text_to_numeric_safe(s."Price Target - #"), 0)                   AS "Price Target - #",
       COALESCE(text_to_numeric_safe(s."Analyst Rating"), 0)                     AS "Analyst Rating",
       COALESCE(text_to_numeric_safe(s."# Strong Sell Ratings"), 0)              AS "# Strong Sell Ratings",
       COALESCE(text_to_numeric_safe(s."# Strong Buys Ratings"), 0)              AS "# Strong Buys Ratings",
       COALESCE(text_to_numeric_safe(s."# Hold Ratings"), 0)                     AS "# Hold Ratings",
       COALESCE(text_to_numeric_safe(s."# Buys Ratings"), 0)                     AS "# Buys Ratings",
       COALESCE(text_to_numeric_safe(s."# Sell Ratings"), 0)                     AS "# Sell Ratings",
       COALESCE(text_to_numeric_safe(s."Shrs Out"), 0)                           AS "Shrs Out",
       COALESCE(text_to_numeric_safe(s."Shrs Out (-1FY)"), 0)                    AS "Shrs Out (-1FY)",
       COALESCE(text_to_numeric_safe(s."Full Time Employees (FQ)"), 0)           AS "Full Time Employees (FQ)",
       COALESCE(text_to_numeric_safe(s."Full Time Employees (FY)"), 0)           AS "Full Time Employees (FY)",
       COALESCE(text_to_numeric_safe(s."Full Time Employees (-1FY)"), 0)         AS "Full Time Employees (-1FY)",
       COALESCE(text_to_numeric_safe(s."Full Time Employees (-2FY)"), 0)         AS "Full Time Employees (-2FY)",
       COALESCE(text_to_numeric_safe(s."Full Time Employees (-3FY)"), 0)         AS "Full Time Employees (-3FY)",
       COALESCE(text_to_numeric_safe(s."Avg Employees (5YAVGFY)"), 0)            AS "Avg Employees (5YAVGFY)",
       COALESCE(text_to_numeric_safe(s."EPS Norm - Est # (FY1E)"), 0)            AS "EPS Norm - Est # (FY1E)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (3M Ago)"), 0)          AS "Price Target - # (3M Ago)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (6M Ago)"), 0)          AS "Price Target - # (6M Ago)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (YTD Ago)"), 0)         AS "Price Target - # (YTD Ago)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (1Y Ago)"), 0)          AS "Price Target - # (1Y Ago)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (1W Ago)"), 0)          AS "Price Target - # (1W Ago)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (1M Ago)"), 0)          AS "Price Target - # (1M Ago)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (MTD Ago)"), 0)         AS "Price Target - # (MTD Ago)",
       COALESCE(text_to_numeric_safe(s."Price Target - # (QTD Ago)"), 0)         AS "Price Target - # (QTD Ago)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (LTM)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (LTM)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (FQ)"), 0)        AS "Impairment of Goodwill (FQ)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (LTM)"), 0)       AS "Impairment of Goodwill (LTM)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-1FY)"),
                0)                                                               AS "Impairment of Goodwill (-1FY)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (FY)"), 0)        AS "Impairment of Goodwill (FY)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (LTM)"), 0)              AS "Asset Writedown (LTM)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (FY)"), 0)               AS "Asset Writedown (FY)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-1FY)"), 0)             AS "Asset Writedown (-1FY)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (LTM)"), 0)        AS "Restructuring Charges (LTM)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (FQ)"), 0)         AS "Restructuring Charges (FQ)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-1FY)"), 0)       AS "Restructuring Charges (-1FY)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (FY)"), 0)         AS "Restructuring Charges (FY)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (LTM)"),
                0)                                                               AS "Merger & Restructuring Charges (LTM)",
       COALESCE(text_to_numeric_safe(s."Other Unusual Items/Total (LTM)"),
                0)                                                               AS "Other Unusual Items/Total (LTM)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (FQ)"), 0)               AS "Asset Writedown (FQ)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (5YAVGFQ)"), 0)          AS "Asset Writedown (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (5YAVGFQ)"),
                0)                                                               AS "Impairment of Goodwill (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (5YAVGFQ)"),
                0)                                                               AS "Restructuring Charges (5YAVGFQ)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (FQ)"),
                0)                                                               AS "Merger & Restructuring Charges (FQ)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (FY)"),
                0)                                                               AS "Merger & Restructuring Charges (FY)",
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (5YAVGFQ)"),
                0)                                                               AS "Merger & Restructuring Charges (5YAVGFQ)",
       NULLIF(TRIM(s."Description"), '')                                         AS "Description",
       report_fiscal.fiscal_month                                                AS "Fiscal Month",
       report_fiscal.fiscal_quarter                                              AS "Fiscal Quarter",
       report_fiscal.fiscal_year                                                 AS "Fiscal Year",
       calculate_reporting_lag(
               NULLIF(TRIM(s."Next Earnings"), '')::DATE,
               NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE,
               report_fiscal.earnings_report_frequency
       )                                                                         AS "Reporting Lag",
       calculate_next_income_statement_report_date(
               NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE,
               report_fiscal.earnings_report_frequency
       )                                                                         AS "Next Income Statement Report Date",
       report_fiscal.reporting_interval                                          AS "Reporting Interval",
       report_fiscal.earnings_report_frequency                                   AS "Earnings Report (Frequency)",
       -- NEW: Impairment of Goodwill Historical
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-1FQFQ)"),
                0)                                                               AS "Impairment of Goodwill (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-2FQFQ)"),
                0)                                                               AS "Impairment of Goodwill (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-3FQFQ)"),
                0)                                                               AS "Impairment of Goodwill (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-4FQFQ)"),
                0)                                                               AS "Impairment of Goodwill (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-2FY)"),
                0)                                                               AS "Impairment of Goodwill (-2FY)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-3FY)"),
                0)                                                               AS "Impairment of Goodwill (-3FY)",
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-4FY)"),
                0)                                                               AS "Impairment of Goodwill (-4FY)",
       -- NEW: Asset Writedown Historical
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-1FQFQ)"), 0)           AS "Asset Writedown (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-2FQFQ)"), 0)           AS "Asset Writedown (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-3FQFQ)"), 0)           AS "Asset Writedown (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-4FQFQ)"), 0)           AS "Asset Writedown (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-2FY)"), 0)             AS "Asset Writedown (-2FY)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-3FY)"), 0)             AS "Asset Writedown (-3FY)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-4FY)"), 0)             AS "Asset Writedown (-4FY)",
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-5FY)"), 0)             AS "Asset Writedown (-5FY)",
       -- NEW: Gain (Loss) On Sale Of Assets Historical
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (FQ)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (FQ)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (FY)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (FY)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-1FQFQ)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-2FQFQ)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-3FQFQ)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-4FQFQ)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-1FY)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-1FY)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-2FY)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-2FY)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-3FY)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-3FY)",
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-4FY)"),
                0)                                                               AS "Gain (Loss) On Sale Of Assets (-4FY)",
       -- NEW: Restructuring Charges Historical
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-1FQFQ)"),
                0)                                                               AS "Restructuring Charges (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-2FQFQ)"),
                0)                                                               AS "Restructuring Charges (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-3FQFQ)"),
                0)                                                               AS "Restructuring Charges (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-4FQFQ)"),
                0)                                                               AS "Restructuring Charges (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-2FY)"), 0)       AS "Restructuring Charges (-2FY)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-3FY)"), 0)       AS "Restructuring Charges (-3FY)",
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-4FY)"), 0)       AS "Restructuring Charges (-4FY)",
       -- NEW: Net Income - (IS) Historical
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-1FQFQ)"), 0)         AS "Net Income - (IS) (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-2FQFQ)"), 0)         AS "Net Income - (IS) (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-3FQFQ)"), 0)         AS "Net Income - (IS) (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-4FQFQ)"), 0)         AS "Net Income - (IS) (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-2FY)"), 0)           AS "Net Income - (IS) (-2FY)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-3FY)"), 0)           AS "Net Income - (IS) (-3FY)",
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-4FY)"), 0)           AS "Net Income - (IS) (-4FY)",
       -- NEW: Normalized Net Income Historical
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-1FQFQ)"),
                0)                                                               AS "Normalized Net Income (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-2FQFQ)"),
                0)                                                               AS "Normalized Net Income (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-3FQFQ)"),
                0)                                                               AS "Normalized Net Income (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-4FQFQ)"),
                0)                                                               AS "Normalized Net Income (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-2FY)"), 0)       AS "Normalized Net Income (-2FY)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-3FY)"), 0)       AS "Normalized Net Income (-3FY)",
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-4FY)"), 0)       AS "Normalized Net Income (-4FY)",
       -- NEW: Net Income/Adj. Historical
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-1FQFQ)"), 0)           AS "Net Income/Adj. (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-2FQFQ)"), 0)           AS "Net Income/Adj. (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-3FQFQ)"), 0)           AS "Net Income/Adj. (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-4FQFQ)"), 0)           AS "Net Income/Adj. (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-2FY)"), 0)             AS "Net Income/Adj. (-2FY)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-3FY)"), 0)             AS "Net Income/Adj. (-3FY)",
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-4FY)"), 0)             AS "Net Income/Adj. (-4FY)",
       -- NEW: EBIT Historical
       COALESCE(text_to_numeric_safe(s."EBIT (-1FQFQ)"), 0)                      AS "EBIT (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT (-2FQFQ)"), 0)                      AS "EBIT (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT (-3FQFQ)"), 0)                      AS "EBIT (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT (-4FQFQ)"), 0)                      AS "EBIT (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT (-2FY)"), 0)                        AS "EBIT (-2FY)",
       COALESCE(text_to_numeric_safe(s."EBIT (-3FY)"), 0)                        AS "EBIT (-3FY)",
       COALESCE(text_to_numeric_safe(s."EBIT (-4FY)"), 0)                        AS "EBIT (-4FY)",
       -- NEW: EBIT/Adj. Historical
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (FQ)"), 0)                     AS "EBIT/Adj. (FQ)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-1FQFQ)"), 0)                 AS "EBIT/Adj. (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-2FQFQ)"), 0)                 AS "EBIT/Adj. (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-3FQFQ)"), 0)                 AS "EBIT/Adj. (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-4FQFQ)"), 0)                 AS "EBIT/Adj. (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-2FY)"), 0)                   AS "EBIT/Adj. (-2FY)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-3FY)"), 0)                   AS "EBIT/Adj. (-3FY)",
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-4FY)"), 0)                   AS "EBIT/Adj. (-4FY)",
       -- NEW: EBITDA Historical
       COALESCE(text_to_numeric_safe(s."EBITDA (-1FQFQ)"), 0)                    AS "EBITDA (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA (-2FQFQ)"), 0)                    AS "EBITDA (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA (-3FQFQ)"), 0)                    AS "EBITDA (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA (-4FQFQ)"), 0)                    AS "EBITDA (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA (-2FY)"), 0)                      AS "EBITDA (-2FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA (-3FY)"), 0)                      AS "EBITDA (-3FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA (-4FY)"), 0)                      AS "EBITDA (-4FY)",
       -- NEW: EBITDA/Adj. Historical
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (FQ)"), 0)                   AS "EBITDA/Adj. (FQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-1FQFQ)"), 0)               AS "EBITDA/Adj. (-1FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-2FQFQ)"), 0)               AS "EBITDA/Adj. (-2FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-3FQFQ)"), 0)               AS "EBITDA/Adj. (-3FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-4FQFQ)"), 0)               AS "EBITDA/Adj. (-4FQFQ)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-2FY)"), 0)                 AS "EBITDA/Adj. (-2FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-3FY)"), 0)                 AS "EBITDA/Adj. (-3FY)",
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-4FY)"), 0)                 AS "EBITDA/Adj. (-4FY)",
       -- NEW: Basic EPS - Cont Historical
       text_to_numeric_safe(s."Basic EPS - Cont (LTM)")                          AS "Basic EPS - Cont (LTM)",
       text_to_numeric_safe(s."Basic EPS - Cont (FQ)")                           AS "Basic EPS - Cont (FQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (FY)")                           AS "Basic EPS - Cont (FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-1FQFQ)")                       AS "Basic EPS - Cont (-1FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-2FQFQ)")                       AS "Basic EPS - Cont (-2FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-3FQFQ)")                       AS "Basic EPS - Cont (-3FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-4FQFQ)")                       AS "Basic EPS - Cont (-4FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-1FY)")                         AS "Basic EPS - Cont (-1FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-2FY)")                         AS "Basic EPS - Cont (-2FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-3FY)")                         AS "Basic EPS - Cont (-3FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-4FY)")                         AS "Basic EPS - Cont (-4FY)",
       -- NEW: EPS/Adj. Historical
       text_to_numeric_safe(s."EPS/Adj. (FQ)")                                   AS "EPS/Adj. (FQ)",
       text_to_numeric_safe(s."EPS/Adj. (-1FQFQ)")                               AS "EPS/Adj. (-1FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-2FQFQ)")                               AS "EPS/Adj. (-2FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-3FQFQ)")                               AS "EPS/Adj. (-3FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-4FQFQ)")                               AS "EPS/Adj. (-4FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-2FY)")                                 AS "EPS/Adj. (-2FY)",
       text_to_numeric_safe(s."EPS/Adj. (-3FY)")                                 AS "EPS/Adj. (-3FY)",
       text_to_numeric_safe(s."EPS/Adj. (-4FY)")                                 AS "EPS/Adj. (-4FY)",
       -- NEW: Total Revenues Historical
       text_to_numeric_safe(s."Total Revenues (-1FQFQ)")                         AS "Total Revenues (-1FQFQ)",
       text_to_numeric_safe(s."Total Revenues (-2FQFQ)")                         AS "Total Revenues (-2FQFQ)",
       text_to_numeric_safe(s."Total Revenues (-3FQFQ)")                         AS "Total Revenues (-3FQFQ)",
       text_to_numeric_safe(s."Total Revenues (-4FQFQ)")                         AS "Total Revenues (-4FQFQ)",
       text_to_numeric_safe(s."Total Revenues (-2FY)")                           AS "Total Revenues (-2FY)",
       text_to_numeric_safe(s."Total Revenues (-3FY)")                           AS "Total Revenues (-3FY)",
       text_to_numeric_safe(s."Total Revenues (-4FY)")                           AS "Total Revenues (-4FY)"
FROM screening_staging s,
     LATERAL (
         SELECT parse_fiscal_year_end_date(NULLIF(TRIM(s."FY End"), '')) AS fy_end_date
         ) parsed,
     LATERAL (
         SELECT calculate_next_fy_end_date(parsed.fy_end_date) AS next_fy_end_date
         ) next_fy,
     LATERAL (
         SELECT * FROM calculate_fiscal_info(CURRENT_DATE::DATE, parsed.fy_end_date, NULL::TEXT)
         ) current_fiscal,
     LATERAL (
         SELECT *
         FROM calculate_fiscal_info(NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE, parsed.fy_end_date,
                                    NULL::TEXT)
         ) report_fiscal
ON CONFLICT DO NOTHING;

-- FINAL VALIDATION
-- ===================================================================
\echo 'Final validation...'
SELECT 'Total rows in equities:' AS info, COUNT(*) AS count
FROM equities;
SELECT 'Rows by Region:' AS info, "Region", COUNT(*) AS count
FROM equities
GROUP BY "Region"
ORDER BY "Region";
SELECT 'Rows by Sector (top 10):' AS info, "Sector", COUNT(*) AS count
FROM equities
GROUP BY "Sector"
ORDER BY COUNT(*) DESC
LIMIT 10;

-- ===================================================================
-- CLEANUP
-- ===================================================================
DROP TABLE IF EXISTS screening_staging;
DROP FUNCTION IF EXISTS text_to_numeric_safe(TEXT);
DROP FUNCTION IF EXISTS text_to_date_safe(TEXT, TEXT);
DROP FUNCTION IF EXISTS parse_fiscal_year_end_date(TEXT);
DROP FUNCTION IF EXISTS frequency_to_months(TEXT);
DROP FUNCTION IF EXISTS months_to_frequency(INTEGER);
DROP FUNCTION IF EXISTS calculate_reporting_interval(TEXT);
DROP FUNCTION IF EXISTS derive_earnings_report_frequency(DATE, DATE);
DROP FUNCTION IF EXISTS calculate_fiscal_info(DATE, DATE, TEXT);
DROP FUNCTION IF EXISTS calculate_next_income_statement_report_date(DATE, TEXT);
DROP FUNCTION IF EXISTS calculate_next_fy_end_date(DATE);
DROP FUNCTION IF EXISTS calculate_next_fiscal_quarter(INTEGER, TEXT);
DROP FUNCTION IF EXISTS calculate_reporting_lag(DATE, DATE, TEXT);
DROP FUNCTION IF EXISTS calculate_expected_report_date(DATE, TEXT);
DROP FUNCTION IF EXISTS validate_fiscal_dates(DATE, DATE, DATE);

\echo 'Import complete!'
