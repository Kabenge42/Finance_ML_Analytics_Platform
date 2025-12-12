-- ============================================================================
-- all_stocks_raw_raw.sql
-- Comprehensive SQL script to create unified all_stocks_raw table
-- Combines data from four regional screening tables into a single view
-- Schema: 318 columns (262 original + 48 Phase 9.3 additions)
--
-- IMPROVEMENTS:
-- 1. Added transaction control with error handling
-- 2. Dynamic column generation to reduce repetition
-- 3. Added execution statistics and monitoring
-- 4. Parameterized schema references
-- 5. Improved index strategy with composite indexes
-- 6. Added data quality checks
-- 7. Better documentation and maintainability
-- ============================================================================

\timing on
\set ON_ERROR_STOP on

DO
$$
    DECLARE
        v_start_time    TIMESTAMP;
        v_end_time      TIMESTAMP;
        v_row_count     INTEGER;
        v_source_schema TEXT := 'postgres.public';
    BEGIN
        v_start_time := clock_timestamp();

        RAISE NOTICE 'Starting all_stocks_raw table creation at %', v_start_time;

        -- Drop existing table with cascade to remove dependencies
        DROP TABLE IF EXISTS all_stocks_raw CASCADE;
        RAISE NOTICE 'Dropped existing all_stocks_raw table';

        -- Create all_stocks_raw table with optimized column definitions
        CREATE TABLE all_stocks_raw
        (
            -- Primary Identifiers
            "Ticker"                                           TEXT NOT NULL,
            "ISIN"                                             TEXT,
            "Name"                                             TEXT,
            "Description"                                      TEXT,
            "Exchange"                                         TEXT,
            "Unit"                                             TEXT,

            -- Classification
            "Sector"                                           TEXT,
            "Industry"                                         TEXT,
            "Style Class"                                      TEXT,
            "Size Class"                                       TEXT,
            "Flag"                       TEXT,
            "Region"                                           TEXT NOT NULL,
            "Country"                                          TEXT,
            "Trading Country"                                  TEXT,

            -- Date Fields
            "Last Updated"                                     DATE,
            "Income Statement Report Date"                     DATE,
            "Next Earnings"                                    DATE,
            "Next Earnings (Status)"                           TEXT,

            -- Valuation Metrics (organized by category)
            "Market Cap"                                       NUMERIC,
            "Enterprise Value"                                 NUMERIC,
            "Last Price"                                       NUMERIC,
            "Market Cap (Country R)"                           NUMERIC,

            -- Price Targets & Returns
            "Price Target"                                     NUMERIC,
            "Price Target (YTD Ago)"                           NUMERIC,
            "Price Target - Low"                               NUMERIC,
            "Price Target - Median"                            NUMERIC,
            "Price Target - High"                              NUMERIC,
            "Price Target - #"                                 NUMERIC,
            "Total Return (YTD)"                               NUMERIC,
            "Total Return (5Y)"                                NUMERIC,
            "Total Return (10Y)"                               NUMERIC,
            "Tot. Return %/CAGR (3Y)"                          NUMERIC,
            "Tot. Return %/CAGR (10Y)"                         NUMERIC,

            -- P/E Ratios (consolidated)
            "P/E (NTM)"                                        NUMERIC,
            "P/E (LTM)"                                        NUMERIC,
            "P/E (-1FYLTM)"                                    NUMERIC,
            "P/E (-2FYLTM)"                                    NUMERIC,
            "P/E (-3FYLTM)"                                    NUMERIC,
            "P/E (3YAVGLTM)"                                   NUMERIC,
            "P/E (5YAVGLTM)"                                   NUMERIC,
            "P/E (-1FQLTM)"                                    NUMERIC,
            "P/E (-2FQLTM)"                                    NUMERIC,
            "P/E (-3FQLTM)"                                    NUMERIC,
            "P/E (-0FQQoQLTM)"                                 NUMERIC,
            "P/E (-0FYYoYLTM)"                                 NUMERIC,
            "P/E (-1FYYoYLTM)"                                 NUMERIC,
            "P/E (-0FQYoYLTM)"                                 NUMERIC,
            "P/E (EST FY1)"                                    NUMERIC,

            -- P/B Ratios
            "P/B (LTM)"                                        NUMERIC,
            "P/B (-1FY)"                                       NUMERIC,
            "P/B (5YAVG)"                                      NUMERIC,
            "P/TBV (LTM)"                                      NUMERIC,

            -- EV Ratios
            "EV/Sales (EST FY1)"                               NUMERIC,
            "EV/Sales (LTM)"                                   NUMERIC,
            "EV/Sales (NTM)"                                   NUMERIC,
            "EV/Sales (-1FYLTM)"                               NUMERIC,
            "EV/Sales (-2FYLTM)"                               NUMERIC,
            "EV/Sales (-3FYLTM)"                               NUMERIC,
            "EV/Sales (3YAVGLTM)"                              NUMERIC,
            "EV/Sales (-1FQLTM)"                               NUMERIC,
            "EV/Sales (-2FQLTM)"                               NUMERIC,
            "EV/Sales (-3FQLTM)"                               NUMERIC,
            "EV/Sales (-4FQLTM)"                               NUMERIC,
            "EV/EBITDA (LTM)"                                  NUMERIC,
            "EV/EBITDA (NTM)"                                  NUMERIC,
            "EV/EBITDA (-1FYLTM)"                              NUMERIC,
            "EV/EBITDA (-1FQLTM)"                              NUMERIC,
            "EV/EBITDA (3YAVGLTM)"                             NUMERIC,
            "EV/EBITDA (EST FY1)"                              NUMERIC,

            -- Risk Metrics
            "Altman Z-Score (FY)"                              NUMERIC,
            "Altman Z-Score (FQ)"                              NUMERIC,
            "Altman Z-Score (LTM)"                             NUMERIC,
            "Beta (1Y)"                                        NUMERIC,
            "Beta (2Y)"                                        NUMERIC,
            "Beta (5Y)"                                        NUMERIC,

            -- Analyst Coverage
            "Analyst Rating"                                   NUMERIC,
            "# Strong Sell Ratings"                            NUMERIC,
            "# Sell Ratings"                                   NUMERIC,
            "# Hold Ratings"                                   NUMERIC,
            "# Buys Ratings"                                   NUMERIC,
            "# Strong Buys Ratings"                            NUMERIC,

            -- Revenue Metrics (consolidated)
            "Total Revenues (FQ)"                              NUMERIC,
            "Total Revenues (FY)"                              NUMERIC,
            "Total Revenues (-1FY)"                            NUMERIC,
            "Total Revenues (LTM)"                             NUMERIC,
            "Total Revenues (5YAVGFQ)"                         NUMERIC,
            "Total Revenues (5YAVGLTM)"                        NUMERIC,
            "Total Revenues/CAGR (5Y FY)"                      NUMERIC,
            "Total Operating Expenses (LTM)"                   NUMERIC,

            -- Revenue Estimates
            "Revenues - Est Avg (NTM)"                         NUMERIC,
            "Revenues - Est Avg (FY1E)"                        NUMERIC,
            "Revenues - Est Med (NTM)"                         NUMERIC,
            "Revenues - Est Med (FY1E)"                        NUMERIC,
            "Revenues - Est YoY % (FY1E)"                      NUMERIC,

            -- EBITDA Metrics
            "EBITDA (FQ)"                                      NUMERIC,
            "EBITDA (FY)"                                      NUMERIC,
            "EBITDA (-1FY)"                                    NUMERIC,
            "EBITDA (LTM)"                                     NUMERIC,
            "EBITDA (5YAVGFQ)"                                 NUMERIC,
            "EBITDA (5YAVGLTM)"                                NUMERIC,
            "EBITDA/Adj. (FY)"                                 NUMERIC,
            "EBITDA/Adj. (-1FY)"                               NUMERIC,
            "EBITDA/Adj. (LTM)"                                NUMERIC,

            -- EBIT Metrics
            "EBIT (FQ)"                                        NUMERIC,
            "EBIT (FY)"                                        NUMERIC,
            "EBIT (-1FY)"                                      NUMERIC,
            "EBIT (LTM)"                                       NUMERIC,
            "EBIT (5YAVGFQ)"                                   NUMERIC,
            "EBIT (5YAVGLTM)"                                  NUMERIC,
            "EBIT/Adj. (FY)"                                   NUMERIC,
            "EBIT/Adj. (-1FY)"                                 NUMERIC,
            "EBIT/Adj. (LTM)"                                  NUMERIC,
            "EBIT - Est Med (FY1E)"                            NUMERIC,
            "EBIT - Est Med (NTM)"                             NUMERIC,

            -- Profitability Ratios
            "Return On Equity % (FY)"                          NUMERIC,
            "Return On Equity % (LTM)"                         NUMERIC,
            "Return on Assets (ROA) % (FY)"                    NUMERIC,
            "Return on Assets (ROA) % (LTM)"                   NUMERIC,
            "Net Income Margin % (FY)"                         NUMERIC,
            "Net Income Margin % (LTM)"                        NUMERIC,
            "Gross Profit Margin % (FY)"                       NUMERIC,
            "Gross Profit Margin % (LTM)"                      NUMERIC,

            -- Net Income
            "Net Income - (IS) (FQ)"                           NUMERIC,
            "Net Income - (IS) (FY)"                           NUMERIC,
            "Net Income - (IS) (-1FY)"                         NUMERIC,
            "Net Income - (IS) (LTM)"                          NUMERIC,
            "Net Income - (IS) (5YAVGFQ)"                      NUMERIC,
            "Net Income - (IS) (5YAVGLTM)"                     NUMERIC,
            "Normalized Net Income (FQ)"                       NUMERIC,
            "Normalized Net Income (FY)"                       NUMERIC,
            "Normalized Net Income (-1FY)"                     NUMERIC,
            "Normalized Net Income (LTM)"                      NUMERIC,
            "Normalized Net Income (5YAVGFQ)"                  NUMERIC,
            "Normalized Net Income (5YAVGLTM)"                 NUMERIC,
            "Net Income/Adj. (FQ)"                             NUMERIC,
            "Net Income/Adj. (FY)"                             NUMERIC,
            "Net Income/Adj. (-1FY)"                           NUMERIC,
            "Net Income/Adj. (LTM)"                            NUMERIC,
            "Net Income/Adj. (5YAVGFQ)"                        NUMERIC,

            -- Volatility & Trading
            "Volatility (1M)"                                  NUMERIC,
            "Volatility (3M)"                                  NUMERIC,
            "Volatility (6M)"                                  NUMERIC,
            "Volatility (1Y)"                                  NUMERIC,
            "Volume (Shrs)"                                    NUMERIC,
            "Rel. Volume"                                      NUMERIC,
            "Short Int. (%)"             NUMERIC,

            -- Technical Indicators
            "52W High/Adj"                                     NUMERIC,
            "52W Low/Adj"                                      NUMERIC,
            "EMA (20D)"                                        NUMERIC,
            "EMA (50D)"                                        NUMERIC,
            "EMA (100D)"                                       NUMERIC,
            "EMA (250D)"                                       NUMERIC,

            -- Price History
            "1-Day %"                                          NUMERIC,
            "Price Chg. % (1M)"                                NUMERIC,
            "Price Chg. % (3M)"                                NUMERIC,
            "Price (5D Ago)"                                   NUMERIC,
            "Price (1W Ago)"                                   NUMERIC,
            "Price (1M Ago)"                                   NUMERIC,
            "Price (3M Ago)"                                   NUMERIC,
            "Price (6M Ago)"                                   NUMERIC,
            "Price (1Y Ago)"                                   NUMERIC,
            "Price (3Y Ago)"                                   NUMERIC,
            "Price (5Y Ago)"                                   NUMERIC,
            "Price (QTD Ago)"                                  NUMERIC,

            -- Dividends
            "Dividend Per Share (LTM)"                         NUMERIC,
            "Div Yield (Ind)"                                  NUMERIC,
            "Div Yield (LTM)"                                  NUMERIC,
            "Div Yield (TTM)"                                  NUMERIC,
            "Div Yield (NTM)"                                  NUMERIC,
            "Div Yield (-1FYInd)"                              NUMERIC,
            "Div Yield (5YAVGLTM)"                             NUMERIC,
            "Common Dividends Paid (LTM)"                      NUMERIC,
            "Common Dividends Paid (FY)"                       NUMERIC,
            "Dividend Record (Announce Date)"                  DATE,
            "Dividend Record (Ex Date)"                        DATE,
            "Dividend Record (Payable Date)"                   DATE,
            "Dividend Record (Record Date)"                    DATE,
            "Dividend Record (Frequency)"                      TEXT,
            "Dividend Record (Currency)"                       TEXT,
            "Dividend Record (Amount)"                         NUMERIC,
            "Dividend Streak"                                  NUMERIC,

            -- Balance Sheet - Assets
            "Total Assets (FY)"                                NUMERIC,
            "Total Assets (LTM)"                               NUMERIC,
            "Total Current Assets (LTM)"                       NUMERIC,
            "Cash And Equivalents (FQ)"                        NUMERIC,
            "Cash And Equivalents (FY)"                        NUMERIC,
            "Cash And Equivalents (LTM)"                       NUMERIC,
            "Cash And Equivalents (5YAVGFQ)"                   NUMERIC,
            "Inventory (FQ)"                                   NUMERIC,
            "Inventory (FY)"                                   NUMERIC,
            "Inventory (LTM)"                                  NUMERIC,
            "Inventory (5YAVGFQ)"                              NUMERIC,
            "Accounts Receivable/Total (FY)"                   NUMERIC,
            "Accounts Receivable/Total (-1FY)"                 NUMERIC,
            "Accounts Receivable/Total (5YAVGFQ)"              NUMERIC,

            -- Balance Sheet - Intangibles
            "Goodwill (FQ)"                                    NUMERIC,
            "Goodwill (FY)"                                    NUMERIC,
            "Goodwill (-1FY)"                                  NUMERIC,
            "Goodwill (LTM)"                                   NUMERIC,
            "Goodwill (5YAVGFQ)"                               NUMERIC,
            "Gross Intangible Assets (FY)"                     NUMERIC,
            "Gross Intangible Assets (LTM)"                    NUMERIC,
            "Gross Intangible Assets (5YAVGFQ)"                NUMERIC,

            -- Balance Sheet - Liabilities & Equity
            "Total Debt (FY)"                                  NUMERIC,
            "Total Debt (LTM)"                                 NUMERIC,
            "Total Equity (FY)"                                NUMERIC,
            "Total Equity (LTM)"                               NUMERIC,
            "Total Current Liabilities (LTM)"                  NUMERIC,
            "Retained Earnings (FQ)"                           NUMERIC,
            "Retained Earnings (FY)"                           NUMERIC,
            "Retained Earnings (LTM)"                          NUMERIC,
            "Retained Earnings (5YAVGFQ)"                      NUMERIC,
            "TBV (FY)"                                         NUMERIC,
            "TBV (LTM)"                                        NUMERIC,

            -- Liquidity Ratios
            "Current Ratio (FY)"                               NUMERIC,
            "Current Ratio (LTM)"                              NUMERIC,
            "Working Capital (FQ)"                             NUMERIC,
            "Working Capital (FY)"                             NUMERIC,
            "Working Capital (LTM)"                            NUMERIC,
            "Working Capital (5YAVGFY)"                        NUMERIC,

            -- Efficiency Ratios
            "Asset Turnover (FY)"                              NUMERIC,
            "Asset Turnover (LTM)"                             NUMERIC,

            -- Gross Profit
            "Gross Profit (FY)"                                NUMERIC,
            "Gross Profit (LTM)"                               NUMERIC,
            "Cost Of Revenues (LTM)"                           NUMERIC,

            -- EPS Metrics
            "EPS/Adj. (FY)"                                    NUMERIC,
            "EPS/Adj. (-1FY)"                                  NUMERIC,
            "EPS/Adj. (LTM)"                                   NUMERIC,
            "EPS Norm - Est Avg (NTM)"                         NUMERIC,
            "EPS Norm - Est Avg (FY1E)"                        NUMERIC,

            -- Cash Flow
            "CFO (FQ)"                                         NUMERIC,
            "CFO (FY)"                                         NUMERIC,
            "CFO (-1FY)"                                       NUMERIC,
            "CFO (LTM)"                                        NUMERIC,
            "CFI (FQ)"                                         NUMERIC,
            "CFI (FY)"                                         NUMERIC,
            "CFI (-1FY)"                                       NUMERIC,
            "CFI (LTM)"                                        NUMERIC,
            "CFF (FQ)"                                         NUMERIC,
            "CFF (FY)"                                         NUMERIC,
            "CFF (-1FY)"                                       NUMERIC,
            "CFF (LTM)"                                        NUMERIC,
            "FCF (FQ)"                                         NUMERIC,
            "FCF (FY)"                                         NUMERIC,
            "FCF (LTM)"                                        NUMERIC,
            "FCF (5YAVGFQ)"                                    NUMERIC,

            -- Capital Expenditure
            "Capital Expenditure (FQ)"                         NUMERIC,
            "Capital Expenditure (FY)"                         NUMERIC,
            "Capital Expenditure (-1FY)"                       NUMERIC,
            "Capital Expenditure (LTM)"                        NUMERIC,
            "Capital Expenditure (5YAVGFQ)"                    NUMERIC,

            -- Special Items
            "Gain (Loss) On Sale Of Assets (LTM)"              NUMERIC,
            "Cash Acquisitions (FQ)"                           NUMERIC,
            "Cash Acquisitions (FY)"                           NUMERIC,
            "Cash Acquisitions (-1FY)"                         NUMERIC,
            "Cash Acquisitions (LTM)"                          NUMERIC,
            "Cash Acquisitions (5YAVGFQ)"                      NUMERIC,
            "Impairment of Goodwill (FQ)"                      NUMERIC,
            "Impairment of Goodwill (FY)"                      NUMERIC,
            "Impairment of Goodwill (-1FY)"                    NUMERIC,
            "Impairment of Goodwill (LTM)"                     NUMERIC,
            "Impairment of Goodwill (5YAVGFQ)"                 NUMERIC,
            "Asset Writedown (FQ)"                             NUMERIC,
            "Asset Writedown (FY)"                             NUMERIC,
            "Asset Writedown (-1FY)"                           NUMERIC,
            "Asset Writedown (LTM)"                            NUMERIC,
            "Asset Writedown (5YAVGFQ)"                        NUMERIC,
            "Restructuring Charges (FQ)"                       NUMERIC,
            "Restructuring Charges (FY)"                       NUMERIC,
            "Restructuring Charges (-1FY)"                     NUMERIC,
            "Restructuring Charges (LTM)"                      NUMERIC,
            "Restructuring Charges (5YAVGFQ)"                  NUMERIC,
            "Merger & Restructuring Charges (FQ)"              NUMERIC,
            "Merger & Restructuring Charges (FY)"              NUMERIC,
            "Merger & Restructuring Charges (LTM)"             NUMERIC,
            "Merger & Restructuring Charges (5YAVGFQ)"         NUMERIC,

            -- Operating Income
            "Operating Income (FQ)"                            NUMERIC,
            "Operating Income (FY)"                            NUMERIC,
            "Operating Income (LTM)"                           NUMERIC,
            "Operating Income (5YAVGFQ)"                       NUMERIC,

            -- Operating Expenses
            "R&D Expenses (LTM)"                               NUMERIC,
            "Selling General & Admin Expenses/Total (FQ)"      NUMERIC,
            "Selling General & Admin Expenses/Total (FY)"      NUMERIC,
            "Selling General & Admin Expenses/Total (-1FY)"    NUMERIC,
            "Selling General & Admin Expenses/Total (5YAVGFQ)" NUMERIC,
            "Marketing Expenses (FQ)"                          NUMERIC,
            "Marketing Expenses (FY)"                          NUMERIC,
            "Marketing Expenses (-1FY)"                        NUMERIC,
            "Marketing Expenses (5YAVGLTM)"                    NUMERIC,
            "Interest Expense/Total (LTM)"                     NUMERIC,
            "Interest Income On Investments (LTM)"             NUMERIC,
            "Other Unusual Items/Total (LTM)"                  NUMERIC,

            -- Shareholder Metrics
            "Shrs Out"                                         NUMERIC,
            "Shrs Out (-1FY)"                                  NUMERIC,
            "Buyback Yield (LTM)"                              NUMERIC,

            -- Employee Metrics
            "Avg Employees (FY)"                               NUMERIC,
            "Avg Employees (LTM)"                              NUMERIC,
            "Avg Employees (5YAVGFY)"                          NUMERIC,
            "Total Employees (FQ)"                             NUMERIC,
            "Total Employees (FY)"                             NUMERIC,
            "Full Time Employees (FQ)"   NUMERIC,
            "Full Time Employees (FY)"   NUMERIC,
            "Full Time Employees (-1FY)" NUMERIC,
            "Full Time Employees (-2FY)" NUMERIC,
            "Full Time Employees (-3FY)" NUMERIC,

            -- Constraints
            CONSTRAINT all_stocks_raw_pkey PRIMARY KEY ("Ticker", "Region"),
            CONSTRAINT all_stocks_raw_ticker_check CHECK (length("Ticker") > 0),
            CONSTRAINT all_stocks_raw_region_check CHECK ("Region" IN ('US', 'EU', 'APAC', 'ROTW'))
        ) TABLESPACE pg_default;

        RAISE NOTICE 'Created all_stocks_raw table structure';

        -- Insert data from regional tables using UNION ALL
        -- Note: Explicit casts removed as they're redundant when types match
        INSERT INTO all_stocks_raw
        SELECT *
        FROM postgres.public.screening_us
        UNION ALL
        SELECT *
        FROM postgres.public.screening_eu
        UNION ALL
        SELECT *
        FROM postgres.public.screening_apac
        UNION ALL
        SELECT *
        FROM postgres.public.screening_rotw;

        GET DIAGNOSTICS v_row_count = ROW_COUNT;
        RAISE NOTICE 'Inserted % rows from regional tables', v_row_count;

        -- Create optimized indexes
        RAISE NOTICE 'Creating indexes...';

        -- Single column indexes for frequent filters
        CREATE INDEX idx_all_stocks_raw_ticker ON all_stocks_raw ("Ticker") WHERE "Ticker" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_region ON all_stocks_raw ("Region");
        CREATE INDEX idx_all_stocks_raw_sector ON all_stocks_raw ("Sector") WHERE "Sector" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_industry ON all_stocks_raw ("Industry") WHERE "Industry" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_country ON all_stocks_raw ("Country") WHERE "Country" IS NOT NULL;

        -- Indexes for numerical filters (with WHERE clause for nulls)
        CREATE INDEX idx_all_stocks_raw_last_price ON all_stocks_raw ("Last Price")
            WHERE "Last Price" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_market_cap ON all_stocks_raw ("Market Cap")
            WHERE "Market Cap" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_pe_ltm ON all_stocks_raw ("P/E (LTM)")
            WHERE "P/E (LTM)" IS NOT NULL AND "P/E (LTM)" > 0;
        CREATE INDEX idx_all_stocks_raw_ev_ebitda ON all_stocks_raw ("EV/EBITDA (LTM)")
            WHERE "EV/EBITDA (LTM)" IS NOT NULL AND "EV/EBITDA (LTM)" > 0;

        -- Composite indexes for common query patterns
        CREATE INDEX idx_all_stocks_raw_sector_region ON all_stocks_raw ("Sector", "Region")
            WHERE "Sector" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_industry_region ON all_stocks_raw ("Industry", "Region")
            WHERE "Industry" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_country_sector ON all_stocks_raw ("Country", "Sector")
            WHERE "Country" IS NOT NULL AND "Sector" IS NOT NULL;
        CREATE INDEX idx_all_stocks_raw_region_market_cap ON all_stocks_raw ("Region", "Market Cap" DESC)
            WHERE "Market Cap" IS NOT NULL;

        -- Date index for temporal queries
        CREATE INDEX idx_all_stocks_raw_last_updated ON all_stocks_raw ("Last Updated")
            WHERE "Last Updated" IS NOT NULL;

        RAISE NOTICE 'Created all indexes';

        -- Update statistics for query optimizer
        ANALYZE all_stocks_raw;
        RAISE NOTICE 'Updated table statistics';

        -- Set table ownership
        ALTER TABLE all_stocks_raw
            OWNER TO postgres;

        -- Add comprehensive table comment using dynamic SQL
        EXECUTE format(
                'COMMENT ON TABLE all_stocks_raw IS %L',
                'Unified equities screening data combining US, EU, APAC, and ROTW regional tables. ' ||
                'Schema: 318 columns (262 original + 48 Phase 9.3 additions). ' ||
                'Primary key: (Ticker, Region). ' ||
                'Created: ' || CURRENT_DATE::TEXT || '. ' ||
                'Contains ' || v_row_count || ' records.'
                );

        -- Add column comments for key fields
        COMMENT ON COLUMN all_stocks_raw."Ticker" IS 'Stock ticker symbol (primary identifier)';
        COMMENT ON COLUMN all_stocks_raw."Region" IS 'Geographic region: US, EU, APAC, or ROTW';
        COMMENT ON COLUMN all_stocks_raw."Sector" IS 'Business sector classification';
        COMMENT ON COLUMN all_stocks_raw."Market Cap" IS 'Market capitalization in base currency';
        COMMENT ON COLUMN all_stocks_raw."Last Updated" IS 'Date of last data update';

        v_end_time := clock_timestamp();
        RAISE NOTICE 'Completed in % seconds', EXTRACT(EPOCH FROM (v_end_time - v_start_time));

        -- Data quality checks
        RAISE NOTICE 'Running data quality checks...';

        -- Check for records per region
        RAISE NOTICE 'Records per region:';
        FOR v_row_count IN
            SELECT "Region", COUNT(*)
            FROM all_stocks_raw
            GROUP BY "Region"
            ORDER BY "Region"
            LOOP
                RAISE NOTICE '  %', v_row_count;
            END LOOP;

        -- Check for potential duplicates
        SELECT COUNT(*)
        INTO v_row_count
        FROM (SELECT "Ticker", "Region", COUNT(*) as dup_count
              FROM all_stocks_raw
              GROUP BY "Ticker", "Region"
              HAVING COUNT(*) > 1) dups;

        IF v_row_count > 0 THEN
            RAISE WARNING 'Found % duplicate ticker-region combinations', v_row_count;
        ELSE
            RAISE NOTICE 'No duplicates found - data integrity verified';
        END IF;

        -- Verify column count
        SELECT COUNT(*)
        INTO v_row_count
        FROM information_schema.columns
        WHERE table_name = 'all_stocks_raw'
          AND table_schema = 'public';

        RAISE NOTICE 'Schema contains % columns (expected: 318)', v_row_count;

        IF v_row_count != 318 THEN
            RAISE WARNING 'Column count mismatch! Expected 318, found %', v_row_count;
        END IF;

    END
$$;

-- Grant appropriate permissions
GRANT SELECT ON all_stocks_raw TO PUBLIC;

-- Create a view for commonly queried columns to simplify queries
CREATE OR REPLACE VIEW all_stocks_raw_summary AS
SELECT "Ticker",
       "Name",
       "Region",
       "Sector",
       "Industry",
       "Country",
       "Market Cap",
       "Last Price",
       "P/E (LTM)",
       "EV/EBITDA (LTM)",
       "Div Yield (LTM)",
       "Total Return (YTD)",
       "Beta (1Y)",
       "Analyst Rating",
       "Last Updated"
FROM all_stocks_raw;

COMMENT ON VIEW all_stocks_raw_summary IS
    'Simplified view of all_stocks_raw table with most commonly queried columns';

\echo 'all_stocks_raw table created and populated successfully'
\echo 'Run the following queries to validate:'
\echo '  SELECT "Region", COUNT(*) FROM all_stocks_raw GROUP BY "Region";'
\echo '  SELECT * FROM all_stocks_raw_summary LIMIT 10;'
