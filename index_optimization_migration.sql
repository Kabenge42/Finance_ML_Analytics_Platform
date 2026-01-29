-- ============================================================
-- Index Optimization Migration for postgres.public schema
-- ============================================================

BEGIN;

-- 1. EQUITIES: Add primary key and remove redundant index
ALTER TABLE equities
    ADD CONSTRAINT equities_pkey PRIMARY KEY ("ISIN");
DROP INDEX IF EXISTS idx_equities_isin;

-- 2. EQUITIES: Consolidate geographic indexes
DROP INDEX IF EXISTS idx_equities_region;
DROP INDEX IF EXISTS idx_equities_country;
DROP INDEX IF EXISTS idx_equities_trading_country;
DROP INDEX IF EXISTS idx_equities_exchange;
CREATE INDEX idx_equities_geography ON equities ("Region", "Country", "Exchange");

-- 3. EQUITIES: Consolidate classification indexes
DROP INDEX IF EXISTS idx_equities_sector;
DROP INDEX IF EXISTS idx_equities_industry;
DROP INDEX IF EXISTS idx_equities_style_class;
DROP INDEX IF EXISTS idx_equities_size_class;
CREATE INDEX idx_equities_classification
    ON equities ("Sector", "Industry", "Size Class", "Style Class");

-- 4. EQUITIES: Optimize name index
DROP INDEX IF EXISTS idx_equities_name;
CREATE INDEX idx_equities_name ON equities ("Name" text_pattern_ops);

-- 5. EQUITIES: Add analytical indexes
CREATE INDEX idx_equities_fiscal
    ON equities ("Fiscal Year", "Fiscal Quarter", "Income Statement Report Date");
CREATE INDEX idx_equities_market_cap ON equities ("Market Cap" DESC NULLS LAST);

-- 6. MV_ALL_STOCK_FEATURES: Remove redundant ISIN index
DROP INDEX IF EXISTS idx_mv_all_stock_features_isin;

-- 7. EQUITIES_SCHEMA_METADATA: Consider dropping DDL index (verify usage first)
-- DROP INDEX IF EXISTS idx_equities_schema_metadata_ddl;

COMMIT;

-- Post-migration: Update statistics
ANALYZE equities;
ANALYZE mv_all_stock_features;
ANALYZE equities_schema_metadata;
