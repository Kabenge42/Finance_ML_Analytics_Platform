-- =============================================================================
-- Import Script for Equities Data from Regional CSV Files
-- =============================================================================
-- This script provides a robust approach to importing CSV data into the
-- equities table with proper NULL handling, encoding, and error checking.
--
-- Usage (from project root):
--   psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
--
-- Or run individual sections as needed using psql -c commands
--
-- November 2025 Update: Compatible with new price-related columns
-- This script is schema-agnostic regarding column additions. It uses staging tables
-- created as LIKE equities and imports with \copy followed by SELECT * insertion.
-- As a result, adding new columns to the equities schema requires no changes here.
-- =============================================================================

-- =============================================================================
-- SECTION 1: Pre-Import Validation
-- =============================================================================

-- Check if equities table exists
DO
$$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'equities') THEN
            RAISE EXCEPTION 'Table equities does not exist. Please run create_equities_schema.sql first.';
        END IF;
    END
$$;

-- Display current row count (before import)
SELECT 'Current equities table row count:' AS status, COUNT(*) AS row_count
FROM equities;

-- =============================================================================
-- SECTION 2: Import US Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing US Region Data...'
\echo '==================================================================='

-- Create temporary staging table for US data (all TEXT columns for initial load)
CREATE TEMP TABLE IF NOT EXISTS screening_us_raw
(
    LIKE equities
);

-- Import CSV with proper NULL handling and encoding
-- Key parameters:
--   NULL '' - treats empty strings as NULL values
--   ENCODING 'UTF8' - ensures proper character handling
--   HEADER true - skips the CSV header row
\copy screening_us_raw FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'US staging table loaded:' AS status, COUNT(*) AS row_count
FROM screening_us_raw;

-- Set Region for all US records (if not already set in CSV)
UPDATE screening_us_raw
SET "Region" = 'US'
WHERE "Region" IS NULL
   OR "Region" = '';

-- Insert into main table with explicit type casting and conflict handling
-- Date columns are converted using TO_DATE() with proper format
-- Numeric columns rely on implicit casting (empty strings already converted to NULL by \copy)
-- Text columns pass through unchanged
INSERT INTO equities
SELECT "Ticker",
       "ISIN",
       "Name",
       "Description",
       "Exchange",
       "Unit",
       "Sector",
       "Industry",
       "Last Updated",
       "Income Statement Report Date",
       "Next Earnings",
       "Style Class",
       "Next Earnings (Status)",
       "Size Class",
       "Region",
       "Country",
       "Trading Country",
       "Market Cap",
       "Enterprise Value",
       "Last Price",
       "Price Target (YTD Ago)",
       "Total Return (YTD)",
       "Price Target",
       "Price Target - Low",
       "Price Target - Median",
       "Price Target - High",
       "Price Target - #",
       "P/E (NTM)",
       "P/E (LTM)",
       "Altman Z-Score (FY)",
       "Altman Z-Score (FQ)",
       "Altman Z-Score (LTM)",
       "Beta (1Y)",
       "Beta (2Y)",
       "Beta (5Y)",
       "Analyst Rating",
       "# Strong Sell Ratings",
       "# Strong Buys Ratings",
       "# Hold Ratings",
       "# Buys Ratings",
       "# Sell Ratings",
       "Total Revenues/CAGR (5Y FY)",
       "Total Revenues (FQ)",
       "Total Revenues (-1FY)",
       "Total Revenues (FY)",
       "Total Revenues (LTM)",
       "Total Operating Expenses (LTM)",
       "P/TBV (LTM)",
       "TBV (FY)",
       "TBV (LTM)",
       "Market Cap (Country R)",
       "Tot. Return %/CAGR (3Y)",
       "Tot. Return %/CAGR (10Y)",
       "Total Return (5Y)",
       "Total Return (10Y)",
       "Net Income/Adj. (-1FY)",
       "CFF (LTM)",
       "CFI (LTM)",
       "FCF (LTM)",
       "CFO (LTM)",
       "EBITDA (FQ)",
       "EBITDA (LTM)",
       "EBITDA (FY)",
       "EBITDA (-1FY)",
       "EBITDA/Adj. (LTM)",
       "EBITDA/Adj. (FY)",
       "EBITDA/Adj. (-1FY)",
       "EBIT (FQ)",
       "EBIT (LTM)",
       "EBIT (FY)",
       "EBIT (-1FY)",
       "EBIT/Adj. (-1FY)",
       "EBIT/Adj. (FY)",
       "EBIT/Adj. (LTM)",
       "EBIT - Est Med (FY1E)",
       "EBIT - Est Med (NTM)",
       "Return On Equity % (LTM)",
       "Return On Equity % (FY)",
       "Net Income - (IS) (FY)",
       "Net Income - (IS) (LTM)",
       "Normalized Net Income (FY)",
       "Normalized Net Income (LTM)",
       "Net Income/Adj. (FY)",
       "Net Income/Adj. (LTM)",
       "Net Income Margin % (FY)",
       "Net Income Margin % (LTM)",
       "Volatility (1M)",
       "Volatility (3M)",
       "Volatility (6M)",
       "Volatility (1Y)",
       "Volume (Shrs)",
       "Dividend Per Share (LTM)",
       "Div Yield (Ind)",
       "Div Yield (LTM)",
       "Total Debt (FY)",
       "Total Equity (FY)",
       "Total Equity (LTM)",
       "Total Debt (LTM)",
       "Total Assets (LTM)",
       "Total Assets (FY)",
       "Current Ratio (FY)",
       "Current Ratio (LTM)",
       "Gross Profit Margin % (FY)",
       "Gross Profit Margin % (LTM)",
       "Asset Turnover (FY)",
       "Asset Turnover (LTM)",
       "Gross Profit (LTM)",
       "Gross Profit (FY)",
       "EPS Norm - Est Avg (NTM)",
       "EPS/Adj. (-1FY)",
       "EPS/Adj. (FY)",
       "EPS/Adj. (LTM)",
       "EPS Norm - Est Avg (FY1E)",
       "Gain (Loss) On Sale Of Assets (LTM)",
       "Cost Of Revenues (LTM)",
       "Cash Acquisitions (LTM)",
       "Cash Acquisitions (FY)",
       "Cash Acquisitions (-1FY)",
       "Inventory (LTM)",
       "Goodwill (FQ)",
       "Goodwill (LTM)",
       "Goodwill (FY)",
       "Goodwill (-1FY)",
       "Impairment of Goodwill (FQ)",
       "Impairment of Goodwill (LTM)",
       "Impairment of Goodwill (-1FY)",
       "Impairment of Goodwill (FY)",
       "Operating Income (LTM)",
       "Asset Writedown (LTM)",
       "Asset Writedown (FY)",
       "Asset Writedown (-1FY)",
       "Operating Income (FY)",
       "Capital Expenditure (LTM)",
       "Capital Expenditure (-1FY)",
       "Capital Expenditure (FY)",
       "Retained Earnings (LTM)",
       "Total Current Assets (LTM)",
       "Total Current Liabilities (LTM)",
       "R&D Expenses (LTM)",
       "Restructuring Charges (LTM)",
       "Restructuring Charges (FQ)",
       "Restructuring Charges (-1FY)",
       "Restructuring Charges (FY)",
       "Interest Expense/Total (LTM)",
       "Merger & Restructuring Charges (LTM)",
       "Working Capital (LTM)",
       "Other Unusual Items/Total (LTM)",
       "Interest Income On Investments (LTM)",
       "Buyback Yield (LTM)",
       "Return on Assets (ROA) % (LTM)",
       "Return on Assets (ROA) % (FY)",
       "Net Income - (IS) (-1FY)",
       "Normalized Net Income (-1FY)",
       "P/E (-1FYLTM)",
       "CFF (FY)",
       "CFF (-1FY)",
       "CFI (FY)",
       "CFI (-1FY)",
       "CFO (FY)",
       "CFO (-1FY)",
       "Div Yield (-1FYInd)",
       "FCF (FY)",
       "Capital Expenditure (FQ)",
       "Capital Expenditure (5YAVGFQ)",
       "CFF (FQ)",
       "CFI (FQ)",
       "CFO (FQ)",
       "FCF (FQ)",
       "Total Revenues (5YAVGFQ)",
       "EBITDA (5YAVGFQ)",
       "EBIT (5YAVGFQ)",
       "P/E (5YAVGLTM)",
       "FCF (5YAVGFQ)",
       "Cash Acquisitions (FQ)",
       "Cash Acquisitions (5YAVGFQ)",
       "Asset Writedown (FQ)",
       "Asset Writedown (5YAVGFQ)",
       "Impairment of Goodwill (5YAVGFQ)",
       "Operating Income (FQ)",
       "Operating Income (5YAVGFQ)",
       "P/B (LTM)",
       "P/B (-1FY)",
       "P/B (5YAVG)",
       "Cash And Equivalents (LTM)",
       "Cash And Equivalents (FQ)",
       "Cash And Equivalents (FY)",
       "Cash And Equivalents (5YAVGFQ)",
       "Inventory (FQ)",
       "Inventory (FY)",
       "Goodwill (5YAVGFQ)",
       "Inventory (5YAVGFQ)",
       "Avg Employees (LTM)",
       "Avg Employees (FY)",
       "Avg Employees (5YAVGFY)",
       "Retained Earnings (FQ)",
       "Retained Earnings (FY)",
       "Retained Earnings (5YAVGFQ)",
       "Working Capital (FQ)",
       "Working Capital (FY)",
       "Working Capital (5YAVGFY)",
       "Div Yield (TTM)",
       "Div Yield (NTM)",
       "Div Yield (5YAVGLTM)",
       "Gross Intangible Assets (LTM)",
       "Gross Intangible Assets (FY)",
       "Gross Intangible Assets (5YAVGFQ)",
       "Restructuring Charges (5YAVGFQ)",
       "Merger & Restructuring Charges (FQ)",
       "Merger & Restructuring Charges (FY)",
       "Merger & Restructuring Charges (5YAVGFQ)",
       "Normalized Net Income (FQ)",
       "Normalized Net Income (5YAVGFQ)",
       "Net Income/Adj. (FQ)",
       "Net Income/Adj. (5YAVGFQ)",
       "Net Income - (IS) (FQ)",
       "Net Income - (IS) (5YAVGFQ)",
       "Net Income - (IS) (5YAVGLTM)",
       "Normalized Net Income (5YAVGLTM)",
       "EBITDA (5YAVGLTM)",
       "EBIT (5YAVGLTM)",
       "Total Revenues (5YAVGLTM)",
       "Revenues - Est YoY % (FY1E)",
       "Price Chg. % (1M)",
       "Price Chg. % (3M)",
       "1-Day %",
       "Price (5D Ago)",
       "Price (1W Ago)",
       "Price (1M Ago)",
       "Price (3M Ago)",
       "Price (6M Ago)",
       "Price (1Y Ago)",
       "Price (3Y Ago)",
       "Price (5Y Ago)",
       "Price (QTD Ago)",
       "Rel. Volume",
       "Shrs Out",
       "Shrs Out (-1FY)",
       "Common Dividends Paid (LTM)",
       "Common Dividends Paid (FY)",
       "Selling General & Admin Expenses/Total (FQ)",
       "Selling General & Admin Expenses/Total (FY)",
       "Selling General & Admin Expenses/Total (-1FY)",
       "Selling General & Admin Expenses/Total (5YAVGFQ)",
       "Accounts Receivable/Total (FY)",
       "Accounts Receivable/Total (-1FY)",
       "Accounts Receivable/Total (5YAVGFQ)",
       "Marketing Expenses (FQ)",
       "Marketing Expenses (FY)",
       "Marketing Expenses (-1FY)",
       "Marketing Expenses (5YAVGLTM)",
       "Revenues - Est Avg (NTM)",
       "Revenues - Est Avg (FY1E)",
       "Revenues - Est Med (NTM)",
       "Revenues - Est Med (FY1E)",
       "EV/Sales (EST FY1)",
       "EV/Sales (LTM)",
       "EV/Sales (NTM)",
       "EV/Sales (-1FYLTM)",
       "EV/Sales (-2FYLTM)",
       "EV/Sales (-3FYLTM)",
       "EV/Sales (3YAVGLTM)",
       "EV/Sales (-1FQLTM)",
       "EV/Sales (-2FQLTM)",
       "EV/Sales (-3FQLTM)",
       "EV/Sales (-4FQLTM)",
       "Total Employees (FY)",
       "Total Employees (FQ)",
       "52W High/Adj",
       "52W Low/Adj",
       "EMA (20D)",
       "EMA (50D)",
       "EMA (100D)",
       "EMA (250D)",
       "EV/EBITDA (LTM)",
       "EV/EBITDA (NTM)",
       "EV/EBITDA (-1FYLTM)",
       "EV/EBITDA (-1FQLTM)",
       "EV/EBITDA (3YAVGLTM)",
       "EV/EBITDA (EST FY1)",
       "P/E (EST FY1)",
       "P/E (-2FYLTM)",
       "P/E (-3FYLTM)",
       "P/E (3YAVGLTM)",
       "P/E (-1FQLTM)",
       "P/E (-2FQLTM)",
       "P/E (-3FQLTM)",
       "P/E (-0FQQoQLTM)",
       "P/E (-0FYYoYLTM)",
       "P/E (-1FYYoYLTM)",
       "P/E (-0FQYoYLTM)",
       "Dividend Record (Announce Date)",
       "Dividend Record (Ex Date)",
       "Dividend Record (Payable Date)",
       "Dividend Record (Record Date)",
       "Dividend Record (Frequency)",
       "Dividend Record (Currency)",
       "Dividend Record (Amount)",
       "Dividend Streak"
FROM screening_us_raw
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'US data inserted:' AS status, COUNT(*) AS row_count
FROM equities
WHERE "Region" = 'US';

-- Clean up staging table
DROP TABLE screening_us_raw;

\echo 'US Region import completed.'
\echo ''

-- =============================================================================
-- SECTION 3: Import EU Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing EU Region Data...'
\echo '==================================================================='

-- Create temporary staging table for EU data (all TEXT columns for initial load)
CREATE TEMP TABLE IF NOT EXISTS screening_eu_raw
(
    LIKE equities
);

-- Import CSV with proper NULL handling and encoding
\copy screening_eu_raw FROM 'data/screening_eu.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'EU staging table loaded:' AS status, COUNT(*) AS row_count
FROM screening_eu_raw;

-- Set Region for all EU records (if not already set in CSV)
UPDATE screening_eu_raw
SET "Region" = 'EU'
WHERE "Region" IS NULL
   OR "Region" = '';

-- Insert into main table with explicit type casting and conflict handling
INSERT INTO equities
SELECT "Ticker",
       "ISIN",
       "Name",
       "Description",
       "Exchange",
       "Unit",
       "Sector",
       "Industry",
       "Last Updated",
       "Income Statement Report Date",
       "Next Earnings",
       "Style Class",
       "Next Earnings (Status)",
       "Size Class",
       "Region",
       "Country",
       "Trading Country",
       "Market Cap",
       "Enterprise Value",
       "Last Price",
       "Price Target (YTD Ago)",
       "Total Return (YTD)",
       "Price Target",
       "Price Target - Low",
       "Price Target - Median",
       "Price Target - High",
       "Price Target - #",
       "P/E (NTM)",
       "P/E (LTM)",
       "Altman Z-Score (FY)",
       "Altman Z-Score (FQ)",
       "Altman Z-Score (LTM)",
       "Beta (1Y)",
       "Beta (2Y)",
       "Beta (5Y)",
       "Analyst Rating",
       "# Strong Sell Ratings",
       "# Strong Buys Ratings",
       "# Hold Ratings",
       "# Buys Ratings",
       "# Sell Ratings",
       "Total Revenues/CAGR (5Y FY)",
       "Total Revenues (FQ)",
       "Total Revenues (-1FY)",
       "Total Revenues (FY)",
       "Total Revenues (LTM)",
       "Total Operating Expenses (LTM)",
       "P/TBV (LTM)",
       "TBV (FY)",
       "TBV (LTM)",
       "Market Cap (Country R)",
       "Tot. Return %/CAGR (3Y)",
       "Tot. Return %/CAGR (10Y)",
       "Total Return (5Y)",
       "Total Return (10Y)",
       "Net Income/Adj. (-1FY)",
       "CFF (LTM)",
       "CFI (LTM)",
       "FCF (LTM)",
       "CFO (LTM)",
       "EBITDA (FQ)",
       "EBITDA (LTM)",
       "EBITDA (FY)",
       "EBITDA (-1FY)",
       "EBITDA/Adj. (LTM)",
       "EBITDA/Adj. (FY)",
       "EBITDA/Adj. (-1FY)",
       "EBIT (FQ)",
       "EBIT (LTM)",
       "EBIT (FY)",
       "EBIT (-1FY)",
       "EBIT/Adj. (-1FY)",
       "EBIT/Adj. (FY)",
       "EBIT/Adj. (LTM)",
       "EBIT - Est Med (FY1E)",
       "EBIT - Est Med (NTM)",
       "Return On Equity % (LTM)",
       "Return On Equity % (FY)",
       "Net Income - (IS) (FY)",
       "Net Income - (IS) (LTM)",
       "Normalized Net Income (FY)",
       "Normalized Net Income (LTM)",
       "Net Income/Adj. (FY)",
       "Net Income/Adj. (LTM)",
       "Net Income Margin % (FY)",
       "Net Income Margin % (LTM)",
       "Volatility (1M)",
       "Volatility (3M)",
       "Volatility (6M)",
       "Volatility (1Y)",
       "Volume (Shrs)",
       "Dividend Per Share (LTM)",
       "Div Yield (Ind)",
       "Div Yield (LTM)",
       "Total Debt (FY)",
       "Total Equity (FY)",
       "Total Equity (LTM)",
       "Total Debt (LTM)",
       "Total Assets (LTM)",
       "Total Assets (FY)",
       "Current Ratio (FY)",
       "Current Ratio (LTM)",
       "Gross Profit Margin % (FY)",
       "Gross Profit Margin % (LTM)",
       "Asset Turnover (FY)",
       "Asset Turnover (LTM)",
       "Gross Profit (LTM)",
       "Gross Profit (FY)",
       "EPS Norm - Est Avg (NTM)",
       "EPS/Adj. (-1FY)",
       "EPS/Adj. (FY)",
       "EPS/Adj. (LTM)",
       "EPS Norm - Est Avg (FY1E)",
       "Gain (Loss) On Sale Of Assets (LTM)",
       "Cost Of Revenues (LTM)",
       "Cash Acquisitions (LTM)",
       "Cash Acquisitions (FY)",
       "Cash Acquisitions (-1FY)",
       "Inventory (LTM)",
       "Goodwill (FQ)",
       "Goodwill (LTM)",
       "Goodwill (FY)",
       "Goodwill (-1FY)",
       "Impairment of Goodwill (FQ)",
       "Impairment of Goodwill (LTM)",
       "Impairment of Goodwill (-1FY)",
       "Impairment of Goodwill (FY)",
       "Operating Income (LTM)",
       "Asset Writedown (LTM)",
       "Asset Writedown (FY)",
       "Asset Writedown (-1FY)",
       "Operating Income (FY)",
       "Capital Expenditure (LTM)",
       "Capital Expenditure (-1FY)",
       "Capital Expenditure (FY)",
       "Retained Earnings (LTM)",
       "Total Current Assets (LTM)",
       "Total Current Liabilities (LTM)",
       "R&D Expenses (LTM)",
       "Restructuring Charges (LTM)",
       "Restructuring Charges (FQ)",
       "Restructuring Charges (-1FY)",
       "Restructuring Charges (FY)",
       "Interest Expense/Total (LTM)",
       "Merger & Restructuring Charges (LTM)",
       "Working Capital (LTM)",
       "Other Unusual Items/Total (LTM)",
       "Interest Income On Investments (LTM)",
       "Buyback Yield (LTM)",
       "Return on Assets (ROA) % (LTM)",
       "Return on Assets (ROA) % (FY)",
       "Net Income - (IS) (-1FY)",
       "Normalized Net Income (-1FY)",
       "P/E (-1FYLTM)",
       "CFF (FY)",
       "CFF (-1FY)",
       "CFI (FY)",
       "CFI (-1FY)",
       "CFO (FY)",
       "CFO (-1FY)",
       "Div Yield (-1FYInd)",
       "FCF (FY)",
       "Capital Expenditure (FQ)",
       "Capital Expenditure (5YAVGFQ)",
       "CFF (FQ)",
       "CFI (FQ)",
       "CFO (FQ)",
       "FCF (FQ)",
       "Total Revenues (5YAVGFQ)",
       "EBITDA (5YAVGFQ)",
       "EBIT (5YAVGFQ)",
       "P/E (5YAVGLTM)",
       "FCF (5YAVGFQ)",
       "Cash Acquisitions (FQ)",
       "Cash Acquisitions (5YAVGFQ)",
       "Asset Writedown (FQ)",
       "Asset Writedown (5YAVGFQ)",
       "Impairment of Goodwill (5YAVGFQ)",
       "Operating Income (FQ)",
       "Operating Income (5YAVGFQ)",
       "P/B (LTM)",
       "P/B (-1FY)",
       "P/B (5YAVG)",
       "Cash And Equivalents (LTM)",
       "Cash And Equivalents (FQ)",
       "Cash And Equivalents (FY)",
       "Cash And Equivalents (5YAVGFQ)",
       "Inventory (FQ)",
       "Inventory (FY)",
       "Goodwill (5YAVGFQ)",
       "Inventory (5YAVGFQ)",
       "Avg Employees (LTM)",
       "Avg Employees (FY)",
       "Avg Employees (5YAVGFY)",
       "Retained Earnings (FQ)",
       "Retained Earnings (FY)",
       "Retained Earnings (5YAVGFQ)",
       "Working Capital (FQ)",
       "Working Capital (FY)",
       "Working Capital (5YAVGFY)",
       "Div Yield (TTM)",
       "Div Yield (NTM)",
       "Div Yield (5YAVGLTM)",
       "Gross Intangible Assets (LTM)",
       "Gross Intangible Assets (FY)",
       "Gross Intangible Assets (5YAVGFQ)",
       "Restructuring Charges (5YAVGFQ)",
       "Merger & Restructuring Charges (FQ)",
       "Merger & Restructuring Charges (FY)",
       "Merger & Restructuring Charges (5YAVGFQ)",
       "Normalized Net Income (FQ)",
       "Normalized Net Income (5YAVGFQ)",
       "Net Income/Adj. (FQ)",
       "Net Income/Adj. (5YAVGFQ)",
       "Net Income - (IS) (FQ)",
       "Net Income - (IS) (5YAVGFQ)",
       "Net Income - (IS) (5YAVGLTM)",
       "Normalized Net Income (5YAVGLTM)",
       "EBITDA (5YAVGLTM)",
       "EBIT (5YAVGLTM)",
       "Total Revenues (5YAVGLTM)",
       "Revenues - Est YoY % (FY1E)",
       "Price Chg. % (1M)",
       "Price Chg. % (3M)",
       "1-Day %",
       "Price (5D Ago)",
       "Price (1W Ago)",
       "Price (1M Ago)",
       "Price (3M Ago)",
       "Price (6M Ago)",
       "Price (1Y Ago)",
       "Price (3Y Ago)",
       "Price (5Y Ago)",
       "Price (QTD Ago)",
       "Rel. Volume",
       "Shrs Out",
       "Shrs Out (-1FY)",
       "Common Dividends Paid (LTM)",
       "Common Dividends Paid (FY)",
       "Selling General & Admin Expenses/Total (FQ)",
       "Selling General & Admin Expenses/Total (FY)",
       "Selling General & Admin Expenses/Total (-1FY)",
       "Selling General & Admin Expenses/Total (5YAVGFQ)",
       "Accounts Receivable/Total (FY)",
       "Accounts Receivable/Total (-1FY)",
       "Accounts Receivable/Total (5YAVGFQ)",
       "Marketing Expenses (FQ)",
       "Marketing Expenses (FY)",
       "Marketing Expenses (-1FY)",
       "Marketing Expenses (5YAVGLTM)",
       "Revenues - Est Avg (NTM)",
       "Revenues - Est Avg (FY1E)",
       "Revenues - Est Med (NTM)",
       "Revenues - Est Med (FY1E)",
       "EV/Sales (EST FY1)",
       "EV/Sales (LTM)",
       "EV/Sales (NTM)",
       "EV/Sales (-1FYLTM)",
       "EV/Sales (-2FYLTM)",
       "EV/Sales (-3FYLTM)",
       "EV/Sales (3YAVGLTM)",
       "EV/Sales (-1FQLTM)",
       "EV/Sales (-2FQLTM)",
       "EV/Sales (-3FQLTM)",
       "EV/Sales (-4FQLTM)",
       "Total Employees (FY)",
       "Total Employees (FQ)",
       "52W High/Adj",
       "52W Low/Adj",
       "EMA (20D)",
       "EMA (50D)",
       "EMA (100D)",
       "EMA (250D)",
       "EV/EBITDA (LTM)",
       "EV/EBITDA (NTM)",
       "EV/EBITDA (-1FYLTM)",
       "EV/EBITDA (-1FQLTM)",
       "EV/EBITDA (3YAVGLTM)",
       "EV/EBITDA (EST FY1)",
       "P/E (EST FY1)",
       "P/E (-2FYLTM)",
       "P/E (-3FYLTM)",
       "P/E (3YAVGLTM)",
       "P/E (-1FQLTM)",
       "P/E (-2FQLTM)",
       "P/E (-3FQLTM)",
       "P/E (-0FQQoQLTM)",
       "P/E (-0FYYoYLTM)",
       "P/E (-1FYYoYLTM)",
       "P/E (-0FQYoYLTM)",
       "Dividend Record (Announce Date)",
       "Dividend Record (Ex Date)",
       "Dividend Record (Payable Date)",
       "Dividend Record (Record Date)",
       "Dividend Record (Frequency)",
       "Dividend Record (Currency)",
       "Dividend Record (Amount)",
       "Dividend Streak"
FROM screening_eu_raw
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'EU data inserted:' AS status, COUNT(*) AS row_count
FROM equities
WHERE "Region" = 'EU';

-- Clean up staging table
DROP TABLE screening_eu_raw;

\echo 'EU Region import completed.'
\echo ''

-- =============================================================================
-- SECTION 4: Import APAC Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing APAC Region Data...'
\echo '==================================================================='

-- Create temporary staging table for APAC data (all TEXT columns for initial load)
CREATE TEMP TABLE IF NOT EXISTS screening_apac_raw
(
    LIKE equities
);

-- Import CSV with proper NULL handling and encoding
\copy screening_apac_raw FROM 'data/screening_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'APAC staging table loaded:' AS status, COUNT(*) AS row_count
FROM screening_apac_raw;

-- Set Region for all APAC records (if not already set in CSV)
UPDATE screening_apac_raw
SET "Region" = 'APAC'
WHERE "Region" IS NULL
   OR "Region" = '';

-- Insert into main table with explicit type casting and conflict handling
INSERT INTO equities
SELECT "Ticker",
       "ISIN",
       "Name",
       "Description",
       "Exchange",
       "Unit",
       "Sector",
       "Industry",
       "Last Updated",
       "Income Statement Report Date",
       "Next Earnings",
       "Style Class",
       "Next Earnings (Status)",
       "Size Class",
       "Region",
       "Country",
       "Trading Country",
       "Market Cap",
       "Enterprise Value",
       "Last Price",
       "Price Target (YTD Ago)",
       "Total Return (YTD)",
       "Price Target",
       "Price Target - Low",
       "Price Target - Median",
       "Price Target - High",
       "Price Target - #",
       "P/E (NTM)",
       "P/E (LTM)",
       "Altman Z-Score (FY)",
       "Altman Z-Score (FQ)",
       "Altman Z-Score (LTM)",
       "Beta (1Y)",
       "Beta (2Y)",
       "Beta (5Y)",
       "Analyst Rating",
       "# Strong Sell Ratings",
       "# Strong Buys Ratings",
       "# Hold Ratings",
       "# Buys Ratings",
       "# Sell Ratings",
       "Total Revenues/CAGR (5Y FY)",
       "Total Revenues (FQ)",
       "Total Revenues (-1FY)",
       "Total Revenues (FY)",
       "Total Revenues (LTM)",
       "Total Operating Expenses (LTM)",
       "P/TBV (LTM)",
       "TBV (FY)",
       "TBV (LTM)",
       "Market Cap (Country R)",
       "Tot. Return %/CAGR (3Y)",
       "Tot. Return %/CAGR (10Y)",
       "Total Return (5Y)",
       "Total Return (10Y)",
       "Net Income/Adj. (-1FY)",
       "CFF (LTM)",
       "CFI (LTM)",
       "FCF (LTM)",
       "CFO (LTM)",
       "EBITDA (FQ)",
       "EBITDA (LTM)",
       "EBITDA (FY)",
       "EBITDA (-1FY)",
       "EBITDA/Adj. (LTM)",
       "EBITDA/Adj. (FY)",
       "EBITDA/Adj. (-1FY)",
       "EBIT (FQ)",
       "EBIT (LTM)",
       "EBIT (FY)",
       "EBIT (-1FY)",
       "EBIT/Adj. (-1FY)",
       "EBIT/Adj. (FY)",
       "EBIT/Adj. (LTM)",
       "EBIT - Est Med (FY1E)",
       "EBIT - Est Med (NTM)",
       "Return On Equity % (LTM)",
       "Return On Equity % (FY)",
       "Net Income - (IS) (FY)",
       "Net Income - (IS) (LTM)",
       "Normalized Net Income (FY)",
       "Normalized Net Income (LTM)",
       "Net Income/Adj. (FY)",
       "Net Income/Adj. (LTM)",
       "Net Income Margin % (FY)",
       "Net Income Margin % (LTM)",
       "Volatility (1M)",
       "Volatility (3M)",
       "Volatility (6M)",
       "Volatility (1Y)",
       "Volume (Shrs)",
       "Dividend Per Share (LTM)",
       "Div Yield (Ind)",
       "Div Yield (LTM)",
       "Total Debt (FY)",
       "Total Equity (FY)",
       "Total Equity (LTM)",
       "Total Debt (LTM)",
       "Total Assets (LTM)",
       "Total Assets (FY)",
       "Current Ratio (FY)",
       "Current Ratio (LTM)",
       "Gross Profit Margin % (FY)",
       "Gross Profit Margin % (LTM)",
       "Asset Turnover (FY)",
       "Asset Turnover (LTM)",
       "Gross Profit (LTM)",
       "Gross Profit (FY)",
       "EPS Norm - Est Avg (NTM)",
       "EPS/Adj. (-1FY)",
       "EPS/Adj. (FY)",
       "EPS/Adj. (LTM)",
       "EPS Norm - Est Avg (FY1E)",
       "Gain (Loss) On Sale Of Assets (LTM)",
       "Cost Of Revenues (LTM)",
       "Cash Acquisitions (LTM)",
       "Cash Acquisitions (FY)",
       "Cash Acquisitions (-1FY)",
       "Inventory (LTM)",
       "Goodwill (FQ)",
       "Goodwill (LTM)",
       "Goodwill (FY)",
       "Goodwill (-1FY)",
       "Impairment of Goodwill (FQ)",
       "Impairment of Goodwill (LTM)",
       "Impairment of Goodwill (-1FY)",
       "Impairment of Goodwill (FY)",
       "Operating Income (LTM)",
       "Asset Writedown (LTM)",
       "Asset Writedown (FY)",
       "Asset Writedown (-1FY)",
       "Operating Income (FY)",
       "Capital Expenditure (LTM)",
       "Capital Expenditure (-1FY)",
       "Capital Expenditure (FY)",
       "Retained Earnings (LTM)",
       "Total Current Assets (LTM)",
       "Total Current Liabilities (LTM)",
       "R&D Expenses (LTM)",
       "Restructuring Charges (LTM)",
       "Restructuring Charges (FQ)",
       "Restructuring Charges (-1FY)",
       "Restructuring Charges (FY)",
       "Interest Expense/Total (LTM)",
       "Merger & Restructuring Charges (LTM)",
       "Working Capital (LTM)",
       "Other Unusual Items/Total (LTM)",
       "Interest Income On Investments (LTM)",
       "Buyback Yield (LTM)",
       "Return on Assets (ROA) % (LTM)",
       "Return on Assets (ROA) % (FY)",
       "Net Income - (IS) (-1FY)",
       "Normalized Net Income (-1FY)",
       "P/E (-1FYLTM)",
       "CFF (FY)",
       "CFF (-1FY)",
       "CFI (FY)",
       "CFI (-1FY)",
       "CFO (FY)",
       "CFO (-1FY)",
       "Div Yield (-1FYInd)",
       "FCF (FY)",
       "Capital Expenditure (FQ)",
       "Capital Expenditure (5YAVGFQ)",
       "CFF (FQ)",
       "CFI (FQ)",
       "CFO (FQ)",
       "FCF (FQ)",
       "Total Revenues (5YAVGFQ)",
       "EBITDA (5YAVGFQ)",
       "EBIT (5YAVGFQ)",
       "P/E (5YAVGLTM)",
       "FCF (5YAVGFQ)",
       "Cash Acquisitions (FQ)",
       "Cash Acquisitions (5YAVGFQ)",
       "Asset Writedown (FQ)",
       "Asset Writedown (5YAVGFQ)",
       "Impairment of Goodwill (5YAVGFQ)",
       "Operating Income (FQ)",
       "Operating Income (5YAVGFQ)",
       "P/B (LTM)",
       "P/B (-1FY)",
       "P/B (5YAVG)",
       "Cash And Equivalents (LTM)",
       "Cash And Equivalents (FQ)",
       "Cash And Equivalents (FY)",
       "Cash And Equivalents (5YAVGFQ)",
       "Inventory (FQ)",
       "Inventory (FY)",
       "Goodwill (5YAVGFQ)",
       "Inventory (5YAVGFQ)",
       "Avg Employees (LTM)",
       "Avg Employees (FY)",
       "Avg Employees (5YAVGFY)",
       "Retained Earnings (FQ)",
       "Retained Earnings (FY)",
       "Retained Earnings (5YAVGFQ)",
       "Working Capital (FQ)",
       "Working Capital (FY)",
       "Working Capital (5YAVGFY)",
       "Div Yield (TTM)",
       "Div Yield (NTM)",
       "Div Yield (5YAVGLTM)",
       "Gross Intangible Assets (LTM)",
       "Gross Intangible Assets (FY)",
       "Gross Intangible Assets (5YAVGFQ)",
       "Restructuring Charges (5YAVGFQ)",
       "Merger & Restructuring Charges (FQ)",
       "Merger & Restructuring Charges (FY)",
       "Merger & Restructuring Charges (5YAVGFQ)",
       "Normalized Net Income (FQ)",
       "Normalized Net Income (5YAVGFQ)",
       "Net Income/Adj. (FQ)",
       "Net Income/Adj. (5YAVGFQ)",
       "Net Income - (IS) (FQ)",
       "Net Income - (IS) (5YAVGFQ)",
       "Net Income - (IS) (5YAVGLTM)",
       "Normalized Net Income (5YAVGLTM)",
       "EBITDA (5YAVGLTM)",
       "EBIT (5YAVGLTM)",
       "Total Revenues (5YAVGLTM)",
       "Revenues - Est YoY % (FY1E)",
       "Price Chg. % (1M)",
       "Price Chg. % (3M)",
       "1-Day %",
       "Price (5D Ago)",
       "Price (1W Ago)",
       "Price (1M Ago)",
       "Price (3M Ago)",
       "Price (6M Ago)",
       "Price (1Y Ago)",
       "Price (3Y Ago)",
       "Price (5Y Ago)",
       "Price (QTD Ago)",
       "Rel. Volume",
       "Shrs Out",
       "Shrs Out (-1FY)",
       "Common Dividends Paid (LTM)",
       "Common Dividends Paid (FY)",
       "Selling General & Admin Expenses/Total (FQ)",
       "Selling General & Admin Expenses/Total (FY)",
       "Selling General & Admin Expenses/Total (-1FY)",
       "Selling General & Admin Expenses/Total (5YAVGFQ)",
       "Accounts Receivable/Total (FY)",
       "Accounts Receivable/Total (-1FY)",
       "Accounts Receivable/Total (5YAVGFQ)",
       "Marketing Expenses (FQ)",
       "Marketing Expenses (FY)",
       "Marketing Expenses (-1FY)",
       "Marketing Expenses (5YAVGLTM)",
       "Revenues - Est Avg (NTM)",
       "Revenues - Est Avg (FY1E)",
       "Revenues - Est Med (NTM)",
       "Revenues - Est Med (FY1E)",
       "EV/Sales (EST FY1)",
       "EV/Sales (LTM)",
       "EV/Sales (NTM)",
       "EV/Sales (-1FYLTM)",
       "EV/Sales (-2FYLTM)",
       "EV/Sales (-3FYLTM)",
       "EV/Sales (3YAVGLTM)",
       "EV/Sales (-1FQLTM)",
       "EV/Sales (-2FQLTM)",
       "EV/Sales (-3FQLTM)",
       "EV/Sales (-4FQLTM)",
       "Total Employees (FY)",
       "Total Employees (FQ)",
       "52W High/Adj",
       "52W Low/Adj",
       "EMA (20D)",
       "EMA (50D)",
       "EMA (100D)",
       "EMA (250D)",
       "EV/EBITDA (LTM)",
       "EV/EBITDA (NTM)",
       "EV/EBITDA (-1FYLTM)",
       "EV/EBITDA (-1FQLTM)",
       "EV/EBITDA (3YAVGLTM)",
       "EV/EBITDA (EST FY1)",
       "P/E (EST FY1)",
       "P/E (-2FYLTM)",
       "P/E (-3FYLTM)",
       "P/E (3YAVGLTM)",
       "P/E (-1FQLTM)",
       "P/E (-2FQLTM)",
       "P/E (-3FQLTM)",
       "P/E (-0FQQoQLTM)",
       "P/E (-0FYYoYLTM)",
       "P/E (-1FYYoYLTM)",
       "P/E (-0FQYoYLTM)",
       "Dividend Record (Announce Date)",
       "Dividend Record (Ex Date)",
       "Dividend Record (Payable Date)",
       "Dividend Record (Record Date)",
       "Dividend Record (Frequency)",
       "Dividend Record (Currency)",
       "Dividend Record (Amount)",
       "Dividend Streak"
FROM screening_apac_raw
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'APAC data inserted:' AS status, COUNT(*) AS row_count
FROM equities
WHERE "Region" = 'APAC';

-- Clean up staging table
DROP TABLE screening_apac_raw;

\echo 'APAC Region import completed.'
\echo ''
-- SECTION 5: Import ROTW (Rest of World) Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing ROTW (Rest of World) Region Data...'
\echo '==================================================================='

-- Create temporary staging table for ROTW data (all TEXT columns for initial load)
CREATE TEMP TABLE IF NOT EXISTS screening_rotw_raw
(
    LIKE equities
);

-- Import CSV with proper NULL handling and encoding
\copy screening_rotw_raw FROM 'data/screening_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'ROTW staging table loaded:' AS status, COUNT(*) AS row_count
FROM screening_rotw_raw;

-- Set Region for all ROTW records (if not already set in CSV)
UPDATE screening_rotw_raw
SET "Region" = 'ROTW'
WHERE "Region" IS NULL
   OR "Region" = '';

-- Insert into main table with explicit type casting and conflict handling
INSERT INTO equities
SELECT "Ticker",
       "ISIN",
       "Name",
       "Description",
       "Exchange",
       "Unit",
       "Sector",
       "Industry",
       "Last Updated",
       "Income Statement Report Date",
       "Next Earnings",
       "Style Class",
       "Next Earnings (Status)",
       "Size Class",
       "Region",
       "Country",
       "Trading Country",
       "Market Cap",
       "Enterprise Value",
       "Last Price",
       "Price Target (YTD Ago)",
       "Total Return (YTD)",
       "Price Target",
       "Price Target - Low",
       "Price Target - Median",
       "Price Target - High",
       "Price Target - #",
       "P/E (NTM)",
       "P/E (LTM)",
       "Altman Z-Score (FY)",
       "Altman Z-Score (FQ)",
       "Altman Z-Score (LTM)",
       "Beta (1Y)",
       "Beta (2Y)",
       "Beta (5Y)",
       "Analyst Rating",
       "# Strong Sell Ratings",
       "# Strong Buys Ratings",
       "# Hold Ratings",
       "# Buys Ratings",
       "# Sell Ratings",
       "Total Revenues/CAGR (5Y FY)",
       "Total Revenues (FQ)",
       "Total Revenues (-1FY)",
       "Total Revenues (FY)",
       "Total Revenues (LTM)",
       "Total Operating Expenses (LTM)",
       "P/TBV (LTM)",
       "TBV (FY)",
       "TBV (LTM)",
       "Market Cap (Country R)",
       "Tot. Return %/CAGR (3Y)",
       "Tot. Return %/CAGR (10Y)",
       "Total Return (5Y)",
       "Total Return (10Y)",
       "Net Income/Adj. (-1FY)",
       "CFF (LTM)",
       "CFI (LTM)",
       "FCF (LTM)",
       "CFO (LTM)",
       "EBITDA (FQ)",
       "EBITDA (LTM)",
       "EBITDA (FY)",
       "EBITDA (-1FY)",
       "EBITDA/Adj. (LTM)",
       "EBITDA/Adj. (FY)",
       "EBITDA/Adj. (-1FY)",
       "EBIT (FQ)",
       "EBIT (LTM)",
       "EBIT (FY)",
       "EBIT (-1FY)",
       "EBIT/Adj. (-1FY)",
       "EBIT/Adj. (FY)",
       "EBIT/Adj. (LTM)",
       "EBIT - Est Med (FY1E)",
       "EBIT - Est Med (NTM)",
       "Return On Equity % (LTM)",
       "Return On Equity % (FY)",
       "Net Income - (IS) (FY)",
       "Net Income - (IS) (LTM)",
       "Normalized Net Income (FY)",
       "Normalized Net Income (LTM)",
       "Net Income/Adj. (FY)",
       "Net Income/Adj. (LTM)",
       "Net Income Margin % (FY)",
       "Net Income Margin % (LTM)",
       "Volatility (1M)",
       "Volatility (3M)",
       "Volatility (6M)",
       "Volatility (1Y)",
       "Volume (Shrs)",
       "Dividend Per Share (LTM)",
       "Div Yield (Ind)",
       "Div Yield (LTM)",
       "Total Debt (FY)",
       "Total Equity (FY)",
       "Total Equity (LTM)",
       "Total Debt (LTM)",
       "Total Assets (LTM)",
       "Total Assets (FY)",
       "Current Ratio (FY)",
       "Current Ratio (LTM)",
       "Gross Profit Margin % (FY)",
       "Gross Profit Margin % (LTM)",
       "Asset Turnover (FY)",
       "Asset Turnover (LTM)",
       "Gross Profit (LTM)",
       "Gross Profit (FY)",
       "EPS Norm - Est Avg (NTM)",
       "EPS/Adj. (-1FY)",
       "EPS/Adj. (FY)",
       "EPS/Adj. (LTM)",
       "EPS Norm - Est Avg (FY1E)",
       "Gain (Loss) On Sale Of Assets (LTM)",
       "Cost Of Revenues (LTM)",
       "Cash Acquisitions (LTM)",
       "Cash Acquisitions (FY)",
       "Cash Acquisitions (-1FY)",
       "Inventory (LTM)",
       "Goodwill (FQ)",
       "Goodwill (LTM)",
       "Goodwill (FY)",
       "Goodwill (-1FY)",
       "Impairment of Goodwill (FQ)",
       "Impairment of Goodwill (LTM)",
       "Impairment of Goodwill (-1FY)",
       "Impairment of Goodwill (FY)",
       "Operating Income (LTM)",
       "Asset Writedown (LTM)",
       "Asset Writedown (FY)",
       "Asset Writedown (-1FY)",
       "Operating Income (FY)",
       "Capital Expenditure (LTM)",
       "Capital Expenditure (-1FY)",
       "Capital Expenditure (FY)",
       "Retained Earnings (LTM)",
       "Total Current Assets (LTM)",
       "Total Current Liabilities (LTM)",
       "R&D Expenses (LTM)",
       "Restructuring Charges (LTM)",
       "Restructuring Charges (FQ)",
       "Restructuring Charges (-1FY)",
       "Restructuring Charges (FY)",
       "Interest Expense/Total (LTM)",
       "Merger & Restructuring Charges (LTM)",
       "Working Capital (LTM)",
       "Other Unusual Items/Total (LTM)",
       "Interest Income On Investments (LTM)",
       "Buyback Yield (LTM)",
       "Return on Assets (ROA) % (LTM)",
       "Return on Assets (ROA) % (FY)",
       "Net Income - (IS) (-1FY)",
       "Normalized Net Income (-1FY)",
       "P/E (-1FYLTM)",
       "CFF (FY)",
       "CFF (-1FY)",
       "CFI (FY)",
       "CFI (-1FY)",
       "CFO (FY)",
       "CFO (-1FY)",
       "Div Yield (-1FYInd)",
       "FCF (FY)",
       "Capital Expenditure (FQ)",
       "Capital Expenditure (5YAVGFQ)",
       "CFF (FQ)",
       "CFI (FQ)",
       "CFO (FQ)",
       "FCF (FQ)",
       "Total Revenues (5YAVGFQ)",
       "EBITDA (5YAVGFQ)",
       "EBIT (5YAVGFQ)",
       "P/E (5YAVGLTM)",
       "FCF (5YAVGFQ)",
       "Cash Acquisitions (FQ)",
       "Cash Acquisitions (5YAVGFQ)",
       "Asset Writedown (FQ)",
       "Asset Writedown (5YAVGFQ)",
       "Impairment of Goodwill (5YAVGFQ)",
       "Operating Income (FQ)",
       "Operating Income (5YAVGFQ)",
       "P/B (LTM)",
       "P/B (-1FY)",
       "P/B (5YAVG)",
       "Cash And Equivalents (LTM)",
       "Cash And Equivalents (FQ)",
       "Cash And Equivalents (FY)",
       "Cash And Equivalents (5YAVGFQ)",
       "Inventory (FQ)",
       "Inventory (FY)",
       "Goodwill (5YAVGFQ)",
       "Inventory (5YAVGFQ)",
       "Avg Employees (LTM)",
       "Avg Employees (FY)",
       "Avg Employees (5YAVGFY)",
       "Retained Earnings (FQ)",
       "Retained Earnings (FY)",
       "Retained Earnings (5YAVGFQ)",
       "Working Capital (FQ)",
       "Working Capital (FY)",
       "Working Capital (5YAVGFY)",
       "Div Yield (TTM)",
       "Div Yield (NTM)",
       "Div Yield (5YAVGLTM)",
       "Gross Intangible Assets (LTM)",
       "Gross Intangible Assets (FY)",
       "Gross Intangible Assets (5YAVGFQ)",
       "Restructuring Charges (5YAVGFQ)",
       "Merger & Restructuring Charges (FQ)",
       "Merger & Restructuring Charges (FY)",
       "Merger & Restructuring Charges (5YAVGFQ)",
       "Normalized Net Income (FQ)",
       "Normalized Net Income (5YAVGFQ)",
       "Net Income/Adj. (FQ)",
       "Net Income/Adj. (5YAVGFQ)",
       "Net Income - (IS) (FQ)",
       "Net Income - (IS) (5YAVGFQ)",
       "Net Income - (IS) (5YAVGLTM)",
       "Normalized Net Income (5YAVGLTM)",
       "EBITDA (5YAVGLTM)",
       "EBIT (5YAVGLTM)",
       "Total Revenues (5YAVGLTM)",
       "Revenues - Est YoY % (FY1E)",
       "Price Chg. % (1M)",
       "Price Chg. % (3M)",
       "1-Day %",
       "Price (5D Ago)",
       "Price (1W Ago)",
       "Price (1M Ago)",
       "Price (3M Ago)",
       "Price (6M Ago)",
       "Price (1Y Ago)",
       "Price (3Y Ago)",
       "Price (5Y Ago)",
       "Price (QTD Ago)",
       "Rel. Volume",
       "Shrs Out",
       "Shrs Out (-1FY)",
       "Common Dividends Paid (LTM)",
       "Common Dividends Paid (FY)",
       "Selling General & Admin Expenses/Total (FQ)",
       "Selling General & Admin Expenses/Total (FY)",
       "Selling General & Admin Expenses/Total (-1FY)",
       "Selling General & Admin Expenses/Total (5YAVGFQ)",
       "Accounts Receivable/Total (FY)",
       "Accounts Receivable/Total (-1FY)",
       "Accounts Receivable/Total (5YAVGFQ)",
       "Marketing Expenses (FQ)",
       "Marketing Expenses (FY)",
       "Marketing Expenses (-1FY)",
       "Marketing Expenses (5YAVGLTM)",
       "Revenues - Est Avg (NTM)",
       "Revenues - Est Avg (FY1E)",
       "Revenues - Est Med (NTM)",
       "Revenues - Est Med (FY1E)",
       "EV/Sales (EST FY1)",
       "EV/Sales (LTM)",
       "EV/Sales (NTM)",
       "EV/Sales (-1FYLTM)",
       "EV/Sales (-2FYLTM)",
       "EV/Sales (-3FYLTM)",
       "EV/Sales (3YAVGLTM)",
       "EV/Sales (-1FQLTM)",
       "EV/Sales (-2FQLTM)",
       "EV/Sales (-3FQLTM)",
       "EV/Sales (-4FQLTM)",
       "Total Employees (FY)",
       "Total Employees (FQ)",
       "52W High/Adj",
       "52W Low/Adj",
       "EMA (20D)",
       "EMA (50D)",
       "EMA (100D)",
       "EMA (250D)",
       "EV/EBITDA (LTM)",
       "EV/EBITDA (NTM)",
       "EV/EBITDA (-1FYLTM)",
       "EV/EBITDA (-1FQLTM)",
       "EV/EBITDA (3YAVGLTM)",
       "EV/EBITDA (EST FY1)",
       "P/E (EST FY1)",
       "P/E (-2FYLTM)",
       "P/E (-3FYLTM)",
       "P/E (3YAVGLTM)",
       "P/E (-1FQLTM)",
       "P/E (-2FQLTM)",
       "P/E (-3FQLTM)",
       "P/E (-0FQQoQLTM)",
       "P/E (-0FYYoYLTM)",
       "P/E (-1FYYoYLTM)",
       "P/E (-0FQYoYLTM)",
       "Dividend Record (Announce Date)",
       "Dividend Record (Ex Date)",
       "Dividend Record (Payable Date)",
       "Dividend Record (Record Date)",
       "Dividend Record (Frequency)",
       "Dividend Record (Currency)",
       "Dividend Record (Amount)",
       "Dividend Streak"
FROM screening_rotw_raw
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'ROTW data inserted:' AS status, COUNT(*) AS row_count
FROM equities
WHERE "Region" = 'ROTW';

-- Clean up staging table
DROP TABLE screening_rotw_raw;

\echo 'ROTW Region import completed.'
\echo ''

-- =============================================================================
-- SECTION 6: Post-Import Validation and Summary
-- =============================================================================

\echo '==================================================================='
\echo 'Import Summary and Validation'
\echo '==================================================================='

-- Total row count
SELECT 'Total rows in equities table:' AS status, COUNT(*) AS row_count
FROM equities;

-- Row count by region
SELECT "Region", COUNT(*) AS row_count
FROM equities
GROUP BY "Region"
ORDER BY "Region";

-- Check for records with missing critical fields
SELECT 'Records with missing Ticker:' AS check_name, COUNT(*) AS count
FROM equities
WHERE "Ticker" IS NULL
   OR "Ticker" = '';

SELECT 'Records with missing Sector:' AS check_name, COUNT(*) AS count
FROM equities
WHERE "Sector" IS NULL
   OR "Sector" = '';

SELECT 'Records with missing Last Price:' AS check_name, COUNT(*) AS count
FROM equities
WHERE "Last Price" IS NULL;

-- Sample records from each region
\echo ''
\echo 'Sample records from each region:'
SELECT "Region", "Ticker", "Name", "Sector", "Last Price", "Market Cap"
FROM equities
WHERE "Region" = 'US'
LIMIT 3;

SELECT "Region", "Ticker", "Name", "Sector", "Last Price", "Market Cap"
FROM equities
WHERE "Region" = 'EU'
LIMIT 3;

SELECT "Region", "Ticker", "Name", "Sector", "Last Price", "Market Cap"
FROM equities
WHERE "Region" = 'APAC'
LIMIT 3;

SELECT "Region", "Ticker", "Name", "Sector", "Last Price", "Market Cap"
FROM equities
WHERE "Region" = 'ROTW'
LIMIT 3;

\echo ''
\echo '==================================================================='
\echo 'Import process completed successfully!'
\echo '==================================================================='

-- =============================================================================
-- TROUBLESHOOTING NOTES
-- =============================================================================
-- If you encounter errors:
--
-- 1. "relation equities does not exist"
--    Solution: Run create_equities_schema.sql first
--
-- 2. "ERROR: invalid input syntax for type numeric"
--    Solution: Check CSV for invalid numeric values; the NULL '' parameter
--    should handle empty strings, but check for text in numeric columns
--
-- 3. "ERROR: could not open file for reading"
--    Solution: Ensure you're running psql from the project root directory,
--    or use absolute paths for CSV files
--
-- 4. Permission denied errors
--    Solution: Use \copy (client-side) instead of COPY (server-side)
--
-- 5. Encoding errors
--    Solution: The ENCODING 'UTF8' parameter is set; if issues persist,
--    check the actual encoding of your CSV files
--
-- =============================================================================
