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
-- =============================================================================

-- =============================================================================
-- SECTION 1: Pre-Import Validation
-- =============================================================================

-- Check if equities table exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'equities') THEN
        RAISE EXCEPTION 'Table equities does not exist. Please run create_equities_schema.sql first.';
    END IF;
END $$;

-- Display current row count (before import)
SELECT 'Current equities table row count:' AS status, COUNT(*) AS row_count FROM equities;

-- =============================================================================
-- SECTION 2: Import US Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing US Region Data...'
\echo '==================================================================='

-- Create temporary staging table for US data
CREATE TEMP TABLE IF NOT EXISTS equities_staging_us (LIKE equities);

-- Import CSV with proper NULL handling and encoding
-- Key parameters:
--   NULL '' - treats empty strings as NULL values
--   ENCODING 'UTF8' - ensures proper character handling
--   HEADER true - skips the CSV header row
\copy equities_staging_us FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'US staging table loaded:' AS status, COUNT(*) AS row_count FROM equities_staging_us;

-- Set Region for all US records (if not already set in CSV)
UPDATE equities_staging_us SET "Region" = 'US' WHERE "Region" IS NULL OR "Region" = '';

-- Insert into main table with conflict handling
INSERT INTO equities 
SELECT * FROM equities_staging_us
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'US data inserted:' AS status, COUNT(*) AS row_count FROM equities WHERE "Region" = 'US';

-- Clean up staging table
DROP TABLE equities_staging_us;

\echo 'US Region import completed.'
\echo ''

-- =============================================================================
-- SECTION 3: Import EU Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing EU Region Data...'
\echo '==================================================================='

-- Create temporary staging table for EU data
CREATE TEMP TABLE IF NOT EXISTS equities_staging_eu (LIKE equities);

-- Import CSV with proper NULL handling and encoding
\copy equities_staging_eu FROM 'data/screening_eu.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'EU staging table loaded:' AS status, COUNT(*) AS row_count FROM equities_staging_eu;

-- Set Region for all EU records (if not already set in CSV)
UPDATE equities_staging_eu SET "Region" = 'EU' WHERE "Region" IS NULL OR "Region" = '';

-- Insert into main table with conflict handling
INSERT INTO equities 
SELECT * FROM equities_staging_eu
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'EU data inserted:' AS status, COUNT(*) AS row_count FROM equities WHERE "Region" = 'EU';

-- Clean up staging table
DROP TABLE equities_staging_eu;

\echo 'EU Region import completed.'
\echo ''

-- =============================================================================
-- SECTION 4: Import APAC Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing APAC Region Data...'
\echo '==================================================================='

-- Create temporary staging table for APAC data
CREATE TEMP TABLE IF NOT EXISTS equities_staging_apac (LIKE equities);

-- Import CSV with proper NULL handling and encoding
\copy equities_staging_apac FROM 'data/screening_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'APAC staging table loaded:' AS status, COUNT(*) AS row_count FROM equities_staging_apac;

-- Set Region for all APAC records (if not already set in CSV)
UPDATE equities_staging_apac SET "Region" = 'APAC' WHERE "Region" IS NULL OR "Region" = '';

-- Insert into main table with conflict handling
INSERT INTO equities 
SELECT * FROM equities_staging_apac
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'APAC data inserted:' AS status, COUNT(*) AS row_count FROM equities WHERE "Region" = 'APAC';

-- Clean up staging table
DROP TABLE equities_staging_apac;

\echo 'APAC Region import completed.'
\echo ''

-- =============================================================================
-- SECTION 5: Import ROTW (Rest of World) Region Data
-- =============================================================================

\echo '==================================================================='
\echo 'Importing ROTW (Rest of World) Region Data...'
\echo '==================================================================='

-- Create temporary staging table for ROTW data
CREATE TEMP TABLE IF NOT EXISTS equities_staging_rotw (LIKE equities);

-- Import CSV with proper NULL handling and encoding
\copy equities_staging_rotw FROM 'data/screening_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8')

-- Display staging table statistics
SELECT 'ROTW staging table loaded:' AS status, COUNT(*) AS row_count FROM equities_staging_rotw;

-- Set Region for all ROTW records (if not already set in CSV)
UPDATE equities_staging_rotw SET "Region" = 'ROTW' WHERE "Region" IS NULL OR "Region" = '';

-- Insert into main table with conflict handling
INSERT INTO equities 
SELECT * FROM equities_staging_rotw
ON CONFLICT DO NOTHING;

-- Display post-import statistics
SELECT 'ROTW data inserted:' AS status, COUNT(*) AS row_count FROM equities WHERE "Region" = 'ROTW';

-- Clean up staging table
DROP TABLE equities_staging_rotw;

\echo 'ROTW Region import completed.'
\echo ''

-- =============================================================================
-- SECTION 6: Post-Import Validation and Summary
-- =============================================================================

\echo '==================================================================='
\echo 'Import Summary and Validation'
\echo '==================================================================='

-- Total row count
SELECT 'Total rows in equities table:' AS status, COUNT(*) AS row_count FROM equities;

-- Row count by region
SELECT "Region", COUNT(*) AS row_count
FROM equities
GROUP BY "Region"
ORDER BY "Region";

-- Check for records with missing critical fields
SELECT 'Records with missing Ticker:' AS check_name, COUNT(*) AS count 
FROM equities 
WHERE "Ticker" IS NULL OR "Ticker" = '';

SELECT 'Records with missing Sector:' AS check_name, COUNT(*) AS count 
FROM equities 
WHERE "Sector" IS NULL OR "Sector" = '';

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
