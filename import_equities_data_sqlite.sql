-- =============================================================================
-- SQLite Import Script for Equities Data from Regional CSV Files
-- =============================================================================
-- This script imports the four regional CSVs into a local SQLite database.
-- It uses staging tables cloned from the equities schema, sets the Region for
-- each staging table, then inserts into the main equities table with
-- de-duplication via a UNIQUE index on ("Ticker","Region").
--
-- Requirements:
--   1) Run create_equities_schema_sqlite.sql first to create the equities table
--      and the UNIQUE index.
--   2) Ensure the CSV files exist under the data/ directory:
--      - data/screening_us.csv
--      - data/screening_eu.csv
--      - data/screening_apac.csv
--      - data/screening_rotw.csv
--
-- Usage (from project root):
--   sqlite3 equities.db ".read import_equities_data_sqlite.sql"
--
-- November 2025 Update: Compatible with new price-related columns
-- This script clones the schema from 'equities' into temp staging tables (SELECT * WHERE 0)
-- and imports CSVs with .import followed by INSERT OR IGNORE SELECT *. No changes are needed
-- when new columns are appended to the equities schema.
-- =============================================================================

-- Stop on first error
.bail on

-- Configure CSV import mode
.mode csv
.headers on
.separator ,

-- Optional: If your sqlite3 supports it, ensure empty fields are imported as NULL
-- .import --nullvalue "" <file> <table>
-- Not all sqlite3 builds support --nullvalue for import; post-process steps
-- below can be used if needed to convert empty strings to NULL for selected
-- columns.

-- =============================================================================
-- Begin Transaction
-- =============================================================================
BEGIN TRANSACTION;

-- =============================================================================
-- Pre-Import Validation
-- =============================================================================
.print 'Checking for equities table...'
SELECT CASE
           WHEN EXISTS (SELECT 1
                        FROM sqlite_master
                        WHERE type = 'table'
                          AND name = 'equities') THEN 'OK: equities table exists'
           ELSE 'ERROR: equities table not found. Run create_equities_schema_sqlite.sql' END AS validation_status;

-- =============================================================================
-- Import US Region Data
-- =============================================================================
.print '==============================================='
.print 'Importing US Region Data...'
.print '==============================================='

DROP TABLE IF EXISTS equities_staging_us;
CREATE TEMP TABLE equities_staging_us AS
SELECT *
FROM equities
WHERE 0;

.import data/screening_us.csv equities_staging_us

-- Remove potential header row if imported as data
DELETE
FROM equities_staging_us
WHERE "Ticker" = 'Ticker';

-- Set Region for all US records (if not already set in CSV)
UPDATE equities_staging_us
SET "Region"='US'
WHERE "Region" IS NULL
   OR "Region" = '';

-- Insert into main table with de-duplication (requires unique index on Ticker+Region)
INSERT OR IGNORE INTO equities
SELECT *
FROM equities_staging_us;

DROP TABLE IF EXISTS equities_staging_us;

-- =============================================================================
-- Import EU Region Data
-- =============================================================================
.print '==============================================='
.print 'Importing EU Region Data...'
.print '==============================================='

DROP TABLE IF EXISTS equities_staging_eu;
CREATE TEMP TABLE equities_staging_eu AS
SELECT *
FROM equities
WHERE 0;

.import data/screening_eu.csv equities_staging_eu

-- Remove potential header row if imported as data
DELETE
FROM equities_staging_eu
WHERE "Ticker" = 'Ticker';

UPDATE equities_staging_eu
SET "Region"='EU'
WHERE "Region" IS NULL
   OR "Region" = '';
INSERT OR IGNORE INTO equities
SELECT *
FROM equities_staging_eu;
DROP TABLE IF EXISTS equities_staging_eu;

-- =============================================================================
-- Import APAC Region Data
-- =============================================================================
.print '==============================================='
.print 'Importing APAC Region Data...'
.print '==============================================='

DROP TABLE IF EXISTS equities_staging_apac;
CREATE TEMP TABLE equities_staging_apac AS
SELECT *
FROM equities
WHERE 0;

.import data/screening_apac.csv equities_staging_apac

-- Remove potential header row if imported as data
DELETE
FROM equities_staging_apac
WHERE "Ticker" = 'Ticker';

UPDATE equities_staging_apac
SET "Region"='APAC'
WHERE "Region" IS NULL
   OR "Region" = '';
INSERT OR IGNORE INTO equities
SELECT *
FROM equities_staging_apac;
DROP TABLE IF EXISTS equities_staging_apac;

-- =============================================================================
-- Import ROTW Region Data
-- =============================================================================
.print '==============================================='
.print 'Importing ROTW Region Data...'
.print '==============================================='

DROP TABLE IF EXISTS equities_staging_rotw;
CREATE TEMP TABLE equities_staging_rotw AS
SELECT *
FROM equities
WHERE 0;

.import data/screening_rotw.csv equities_staging_rotw

-- Remove potential header row if imported as data
DELETE
FROM equities_staging_rotw
WHERE "Ticker" = 'Ticker';

UPDATE equities_staging_rotw
SET "Region"='ROTW'
WHERE "Region" IS NULL
   OR "Region" = '';
INSERT OR IGNORE INTO equities
SELECT *
FROM equities_staging_rotw;
DROP TABLE IF EXISTS equities_staging_rotw;

-- =============================================================================
-- Post-Import Summary
-- =============================================================================
.print '==============================================='
.print 'Import Summary and Validation'
.print '==============================================='

.print 'Total rows in equities table:'
SELECT COUNT(*) AS total_rows
FROM equities;

.print 'Row count by region:'
SELECT "Region", COUNT(*) AS region_row_count
FROM equities
GROUP BY "Region"
ORDER BY "Region";

.print 'Records with missing Ticker:'
SELECT COUNT(*) AS missing_ticker
FROM equities
WHERE "Ticker" IS NULL
   OR TRIM("Ticker") = '';

.print 'Records with missing Sector:'
SELECT COUNT(*) AS missing_sector
FROM equities
WHERE "Sector" IS NULL
   OR TRIM("Sector") = '';

.print 'Records with missing Last Price:'
SELECT COUNT(*) AS missing_last_price
FROM equities
WHERE "Last Price" IS NULL;

.print 'Sample records (US):'
SELECT "Region"     AS region,
       "Ticker"     AS ticker,
       "Name"       AS name,
       "Sector"     AS sector,
       "Last Price" AS last_price,
       "Market Cap" AS market_cap
FROM equities
WHERE "Region" = 'US'
LIMIT 3;

.print 'Sample records (EU):'
SELECT "Region"     AS region,
       "Ticker"     AS ticker,
       "Name"       AS name,
       "Sector"     AS sector,
       "Last Price" AS last_price,
       "Market Cap" AS market_cap
FROM equities
WHERE "Region" = 'EU'
LIMIT 3;

.print 'Sample records (APAC):'
SELECT "Region"     AS region,
       "Ticker"     AS ticker,
       "Name"       AS name,
       "Sector"     AS sector,
       "Last Price" AS last_price,
       "Market Cap" AS market_cap
FROM equities
WHERE "Region" = 'APAC'
LIMIT 3;

.print 'Sample records (ROTW):'
SELECT "Region"     AS region,
       "Ticker"     AS ticker,
       "Name"       AS name,
       "Sector"     AS sector,
       "Last Price" AS last_price,
       "Market Cap" AS market_cap
FROM equities
WHERE "Region" = 'ROTW'
LIMIT 3;

.print '==============================================='
.print 'Committing transaction...'
COMMIT;
.print 'Import process completed successfully!'
.print '==============================================='
