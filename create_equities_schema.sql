-- Drop existing table if it exists
DROP TABLE IF EXISTS equities;

/*
 * SEMANTIC ROLE CLASSIFICATION SYSTEM
 * ====================================
 *
 * This schema uses semantic roles aligned with preprocessing requirements:
 *
 * ROLE DEFINITIONS:
 * -----------------
 * id               - Unique identifiers (ticker, isin, name, description)
 * date             - Temporal columns for time-series analysis
 * categorical      - Grouping/classification columns (sector, region, industry, exchange, etc.)
 * feature          - General engineered features from phase_93
 * market           - Market/trading data (price, volume, market cap, shares outstanding, dividends)
 * financial_statement - P&L line items (revenues, expenses, costs, operating items)
 * balance_sheet    - Balance sheet items (assets, liabilities, equity, working capital)
 * cash_flow        - Cash flow statement items (CFO, CFI, CFF, FCF, capex)
 * ratio            - Pre-normalized financial ratios (P/E, P/B, EV/EBITDA, ROE, ROA, turnover)
 * percentage       - Bounded metrics [0-100] (margins, growth rates, returns, volatility, beta)
 * count            - Discrete integers (analyst counts, employees, shares, dividend streak)
 *
 * PREPROCESSING IMPLICATIONS:
 * ---------------------------
 * - Log Transform: Apply to financial_statement, balance_sheet, cash_flow, market (excluding price)
 * - Winsorization: Apply to financial_statement, balance_sheet, cash_flow, market (excluding price)
 * - Scaling: Apply to all numeric except id and date
 * - Price Preservation: NEVER transform columns with role='market' that are actual prices
 * - Ratio Handling: Skip winsorization (already normalized)
 * - Percentage Handling: Skip percentile capping (naturally bounded)
 */

CREATE TABLE equities
(
    -- ===========================================
    -- IDENTIFIERS (id role)
    -- ===========================================
    "Ticker"                                           TEXT,                      -- id: Stock ticker symbol
    "ISIN"                                             TEXT,                      -- id: International Securities ID
    "Name"                                             TEXT,                      -- id: Company name
    "Description"                                      TEXT,                      -- id: Company description

    -- ===========================================
    -- CATEGORICAL (categorical role)
    -- ===========================================
    "Region"                                           TEXT,                      -- categorical: Geographic region
    "Country"                                          TEXT,                      -- categorical: Country of incorporation
    "Trading Country"                                  TEXT,                      -- categorical: Trading country
    "Exchange"                                         TEXT,                      -- categorical: Stock exchange
    "Unit"                                             TEXT,                      -- categorical: Currency unit
    "Sector"                                           TEXT,                      -- categorical: GICS Sector
    "Industry"                                         TEXT,                      -- categorical: GICS Industry
    "Style Class"                                      TEXT,                      -- categorical: Investment style (Value/Growth/Blend)
    "Size Class"                                       TEXT,                      -- categorical: Market cap size class
    "Next Earnings (When)"                             TEXT,                      -- categorical: Earnings timing description
    "Next Earnings (Status)"                           TEXT,                      -- categorical: Earnings announcement status
    "Next Earnings (Report)"                           TEXT,                      -- categorical: Next earnings report type (Full Year/Interim)
    "Dividend Record (Currency)"                       TEXT,                      -- categorical: Dividend currency
    "Dividend Record (Frequency)"                      TEXT,                      -- categorical: Dividend frequency (Annual/Quarterly/etc.)
    "FY End"                                           TEXT,                      -- categorical: Fiscal year end (stored as text)

    -- ===========================================
    -- TEMPORAL (date role)
    -- ===========================================
    "Last Updated"                                     DATE,                      -- date: Data snapshot date
    "Income Statement Report Date"                     DATE,                      -- date: Latest IS filing date
    "Next Earnings"                                    DATE,                      -- date: Next earnings announcement
    "Dividend Record (Announce Date)"                  DATE,                      -- date: Dividend announcement
    "Dividend Record (Payable Date)"                   DATE,                      -- date: Dividend payment date
    "Dividend Record (Record Date)"                    DATE,                      -- date: Dividend record date
    "Dividend Record (Ex Date)"                        DATE,                      -- date: Dividend ex-date
    "FY End Date"                                      DATE,                      -- date: Fiscal year end date (parsed from FY End text)
    "Next FY End Date"                                 DATE,                      -- date: Next fiscal year end date
    "Fiscal Month"                                     INTEGER,                   -- date: Months between Income Statement Report Date and FY End Date
    "Fiscal Quarter"                                   INTEGER,                   -- date: Fiscal quarter (1-4) from report date
    "Fiscal Year"                                      INTEGER,                   -- date: Fiscal year from report date
    "Current Fiscal Quarter"                           TEXT,                      -- date: Current fiscal quarter (formatted as "Q4 2025")
    "Next Fiscal Quarter"                              TEXT,                      -- date: Next fiscal quarter (formatted as "Q4 2025")

    -- ===========================================
    -- REFERENCE DATE (date role)
    -- ===========================================
    "Reference Date"                                   DATE DEFAULT CURRENT_DATE, -- date: Data snapshot reference date for temporal analysis

    -- ===========================================
    -- MARKET DATA (market role)
    -- ===========================================
    -- Current prices and targets (preserve original dollar units)
    "Last Price"                                       NUMERIC,                   -- market: Current trading price
    "Price Target"                                     NUMERIC,                   -- market: Analyst consensus target
    "Price Target (YTD Ago)"                           NUMERIC,                   -- market: Historical price target
    "Price Target - Low"                               NUMERIC,                   -- market: Low analyst target
    "Price Target - Median"                            NUMERIC,                   -- market: Median analyst target
    "Price Target - High"                              NUMERIC,                   -- market: High analyst target
    "Price (5D Ago)"                                   NUMERIC,                   -- market: Price 5 days ago
    "Price (1W Ago)"                                   NUMERIC,                   -- market: Price 1 week ago
    "Price (1M Ago)"                                   NUMERIC,                   -- market: Price 1 month ago
    "Price (3M Ago)"                                   NUMERIC,                   -- market: Price 3 months ago
    "Price (6M Ago)"                                   NUMERIC,                   -- market: Price 6 months ago
    "Price (1Y Ago)"                                   NUMERIC,                   -- market: Price 1 year ago
    "Price (3Y Ago)"                                   NUMERIC,                   -- market: Price 3 years ago
    "Price (5Y Ago)"                                   NUMERIC,                   -- market: Price 5 years ago
    "Price (QTD Ago)"                                  NUMERIC,                   -- market: Price quarter-to-date ago
    "52W High/Adj"                                     NUMERIC,                   -- market: 52-week adjusted high
    "52W Low/Adj"                                      NUMERIC,                   -- market: 52-week adjusted low
    "EMA (20D)"                                        NUMERIC,                   -- market: 20-day exponential moving average
    "EMA (50D)"                                        NUMERIC,                   -- market: 50-day exponential moving average
    "EMA (100D)"                                       NUMERIC,                   -- market: 100-day exponential moving average
    "EMA (250D)"                                       NUMERIC,                   -- market: 250-day exponential moving average

    -- Market capitalization and valuation
    "Market Cap"                                       NUMERIC,                   -- market: Market capitalization
    "Enterprise Value"                                 NUMERIC,                   -- market: Enterprise value
    "Market Cap (Country R)"                           NUMERIC,                   -- market: Market cap country ranking

    -- Volume and shares
    "Volume (Shrs)"                                    NUMERIC,                   -- market: Trading volume (shares)
    "Rel. Volume"                                      NUMERIC,                   -- market: Relative volume ratio
    "Shrs Out"                                         NUMERIC DEFAULT 0,         -- count: Shares outstanding
    "Shrs Out (-1FY)"                                  NUMERIC DEFAULT 0,         -- count: Shares outstanding (previous FY)

    -- Dividend amounts (market prices per share)
    "Dividend Record (Amount)"                         NUMERIC,                   -- market: Dividend amount per share
    "Dividend Per Share (LTM)"                         NUMERIC,                   -- market: Dividend per share (LTM)
    "Common Dividends Paid (LTM)"                      NUMERIC DEFAULT 0,         -- cash_flow: Total dividends paid (LTM)
    "Common Dividends Paid (FY)"                       NUMERIC DEFAULT 0,         -- cash_flow: Total dividends paid (FY)

    -- ===========================================
    -- FINANCIAL STATEMENT (financial_statement role)
    -- ===========================================
    -- Revenue line items
    "Total Revenues (FQ)"                              NUMERIC DEFAULT 0,         -- financial_statement: Total revenues (FQ)
    "Total Revenues (-1FY)"                            NUMERIC DEFAULT 0,         -- financial_statement: Total revenues (previous FY)
    "Total Revenues (FY)"                              NUMERIC DEFAULT 0,         -- financial_statement: Total revenues (FY)
    "Total Revenues (LTM)"                             NUMERIC DEFAULT 0,         -- financial_statement: Total revenues (LTM)
    "Total Revenues (5YAVGFQ)"                         NUMERIC DEFAULT 0,         -- financial_statement: 5Y average revenues (FQ)
    "Total Revenues (5YAVGLTM)"                        NUMERIC DEFAULT 0,         -- financial_statement: 5Y average revenues (LTM)
    "Revenues - Est Avg (NTM)"                         NUMERIC DEFAULT 0,         -- financial_statement: Revenue estimate (NTM)
    "Revenues - Est Avg (FY1E)"                        NUMERIC DEFAULT 0,         -- financial_statement: Revenue estimate (FY1E)
    "Revenues - Est Med (NTM)"                         NUMERIC DEFAULT 0,         -- financial_statement: Revenue estimate median (NTM)
    "Revenues - Est Med (FY1E)"                        NUMERIC DEFAULT 0,         -- financial_statement: Revenue estimate median (FY1E)

    -- EBITDA line items
    "EBITDA (FQ)"                                      NUMERIC DEFAULT 0,         -- financial_statement: EBITDA (FQ)
    "EBITDA (LTM)"                                     NUMERIC DEFAULT 0,         -- financial_statement: EBITDA (LTM)
    "EBITDA (FY)"                                      NUMERIC DEFAULT 0,         -- financial_statement: EBITDA (FY)
    "EBITDA (-1FY)"                                    NUMERIC DEFAULT 0,         -- financial_statement: EBITDA (previous FY)
    "EBITDA/Adj. (LTM)"                                NUMERIC DEFAULT 0,         -- financial_statement: Adjusted EBITDA (LTM)
    "EBITDA/Adj. (FY)"                                 NUMERIC DEFAULT 0,         -- financial_statement: Adjusted EBITDA (FY)
    "EBITDA/Adj. (-1FY)"                               NUMERIC DEFAULT 0,         -- financial_statement: Adjusted EBITDA (previous FY)
    "EBITDA (5YAVGFQ)"                                 NUMERIC DEFAULT 0,         -- financial_statement: 5Y average EBITDA (FQ)
    "EBITDA (5YAVGLTM)"                                NUMERIC DEFAULT 0,         -- financial_statement: 5Y average EBITDA (LTM)
    "EBITDA - Est Avg (NTM)"                           NUMERIC DEFAULT 0,         -- financial_statement: EBITDA estimate (NTM)
    "EBITDA - Est Avg (FY1E)"                          NUMERIC DEFAULT 0,         -- financial_statement: EBITDA estimate (FY1E)

    -- EBIT line items
    "EBIT (FQ)"                                        NUMERIC DEFAULT 0,         -- financial_statement: EBIT (FQ)
    "EBIT (LTM)"                                       NUMERIC DEFAULT 0,         -- financial_statement: EBIT (LTM)
    "EBIT (FY)"                                        NUMERIC DEFAULT 0,         -- financial_statement: EBIT (FY)
    "EBIT (-1FY)"                                      NUMERIC DEFAULT 0,         -- financial_statement: EBIT (previous FY)
    "EBIT/Adj. (-1FY)"                                 NUMERIC DEFAULT 0,         -- financial_statement: Adjusted EBIT (previous FY)
    "EBIT/Adj. (FY)"                                   NUMERIC DEFAULT 0,         -- financial_statement: Adjusted EBIT (FY)
    "EBIT/Adj. (LTM)"                                  NUMERIC DEFAULT 0,         -- financial_statement: Adjusted EBIT (LTM)
    "EBIT (5YAVGFQ)"                                   NUMERIC DEFAULT 0,         -- financial_statement: 5Y average EBIT (FQ)
    "EBIT (5YAVGLTM)"                                  NUMERIC DEFAULT 0,         -- financial_statement: 5Y average EBIT (LTM)
    "EBIT - Est Med (FY1E)"                            NUMERIC DEFAULT 0,         -- financial_statement: EBIT estimate median (FY1E)
    "EBIT - Est Med (NTM)"                             NUMERIC DEFAULT 0,         -- financial_statement: EBIT estimate median (NTM)

    -- Net income line items
    "Net Income - (IS) (FY)"                           NUMERIC DEFAULT 0,         -- financial_statement: Net income (FY)
    "Net Income - (IS) (LTM)"                          NUMERIC DEFAULT 0,         -- financial_statement: Net income (LTM)
    "Net Income - (IS) (FQ)"                           NUMERIC DEFAULT 0,         -- financial_statement: Net income (FQ)
    "Net Income - (IS) (-1FY)"                         NUMERIC DEFAULT 0,         -- financial_statement: Net income (previous FY)
    "Net Income - (IS) (5YAVGFQ)"                      NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg net income (FQ)
    "Net Income - (IS) (5YAVGLTM)"                     NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg net income (LTM)
    "Normalized Net Income (FY)"                       NUMERIC DEFAULT 0,         -- financial_statement: Normalized net income (FY)
    "Normalized Net Income (LTM)"                      NUMERIC DEFAULT 0,         -- financial_statement: Normalized net income (LTM)
    "Normalized Net Income (FQ)"                       NUMERIC DEFAULT 0,         -- financial_statement: Normalized net income (FQ)
    "Normalized Net Income (-1FY)"                     NUMERIC DEFAULT 0,         -- financial_statement: Normalized net income (previous FY)
    "Normalized Net Income (5YAVGFQ)"                  NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg normalized NI (FQ)
    "Normalized Net Income (5YAVGLTM)"                 NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg normalized NI (LTM)
    "Net Income/Adj. (FY)"                             NUMERIC DEFAULT 0,         -- financial_statement: Adjusted net income (FY)
    "Net Income/Adj. (LTM)"                            NUMERIC DEFAULT 0,         -- financial_statement: Adjusted net income (LTM)
    "Net Income/Adj. (FQ)"                             NUMERIC DEFAULT 0,         -- financial_statement: Adjusted net income (FQ)
    "Net Income/Adj. (-1FY)"                           NUMERIC DEFAULT 0,         -- financial_statement: Adjusted net income (previous FY)
    "Net Income/Adj. (5YAVGFQ)"                        NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg adjusted NI (FQ)

    -- Operating income and expenses
    "Operating Income (LTM)"                           NUMERIC DEFAULT 0,         -- financial_statement: Operating income (LTM)
    "Operating Income (FY)"                            NUMERIC DEFAULT 0,         -- financial_statement: Operating income (FY)
    "Operating Income (FQ)"                            NUMERIC DEFAULT 0,         -- financial_statement: Operating income (FQ)
    "Operating Income (5YAVGFQ)"                       NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg operating income (FQ)
    "Total Operating Expenses (LTM)"                   NUMERIC DEFAULT 0,         -- financial_statement: Total operating expenses (LTM)
    "Gross Profit (LTM)"                               NUMERIC DEFAULT 0,         -- financial_statement: Gross profit (LTM)
    "Gross Profit (FY)"                                NUMERIC DEFAULT 0,         -- financial_statement: Gross profit (FY)
    "Cost Of Revenues (LTM)"                           NUMERIC DEFAULT 0,         -- financial_statement: Cost of revenues (LTM)
    "R&D Expenses (LTM)"                               NUMERIC DEFAULT 0,         -- financial_statement: Research & development expenses
    "Selling General & Admin Expenses/Total (FQ)"      NUMERIC DEFAULT 0,         -- financial_statement: SG&A expenses (FQ)
    "Selling General & Admin Expenses/Total (FY)"      NUMERIC DEFAULT 0,         -- financial_statement: SG&A expenses (FY)
    "Selling General & Admin Expenses/Total (-1FY)"    NUMERIC DEFAULT 0,         -- financial_statement: SG&A expenses (previous FY)
    "Selling General & Admin Expenses/Total (5YAVGFQ)" NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg SG&A (FQ)
    "Marketing Expenses (FQ)"                          NUMERIC DEFAULT 0,         -- financial_statement: Marketing expenses (FQ)
    "Marketing Expenses (FY)"                          NUMERIC DEFAULT 0,         -- financial_statement: Marketing expenses (FY)
    "Marketing Expenses (-1FY)"                        NUMERIC DEFAULT 0,         -- financial_statement: Marketing expenses (previous FY)
    "Marketing Expenses (5YAVGLTM)"                    NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg marketing (LTM)

    -- Special/non-recurring items
    "Restructuring Charges (LTM)"                      NUMERIC DEFAULT 0,         -- financial_statement: Restructuring charges (LTM)
    "Restructuring Charges (FQ)"                       NUMERIC DEFAULT 0,         -- financial_statement: Restructuring charges (FQ)
    "Restructuring Charges (-1FY)"                     NUMERIC DEFAULT 0,         -- financial_statement: Restructuring charges (previous FY)
    "Restructuring Charges (FY)"                       NUMERIC DEFAULT 0,         -- financial_statement: Restructuring charges (FY)
    "Restructuring Charges (5YAVGFQ)"                  NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg restructuring (FQ)
    "Merger & Restructuring Charges (LTM)"             NUMERIC DEFAULT 0,         -- financial_statement: Merger & restructuring (LTM)
    "Merger & Restructuring Charges (FQ)"              NUMERIC DEFAULT 0,         -- financial_statement: Merger & restructuring (FQ)
    "Merger & Restructuring Charges (FY)"              NUMERIC DEFAULT 0,         -- financial_statement: Merger & restructuring (FY)
    "Merger & Restructuring Charges (5YAVGFQ)"         NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg merger charges (FQ)
    "Asset Writedown (LTM)"                            NUMERIC DEFAULT 0,         -- financial_statement: Asset writedown (LTM)
    "Asset Writedown (FY)"                             NUMERIC DEFAULT 0,         -- financial_statement: Asset writedown (FY)
    "Asset Writedown (FQ)"                             NUMERIC DEFAULT 0,         -- financial_statement: Asset writedown (FQ)
    "Asset Writedown (-1FY)"                           NUMERIC DEFAULT 0,         -- financial_statement: Asset writedown (previous FY)
    "Asset Writedown (5YAVGFQ)"                        NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg writedown (FQ)
    "Impairment of Goodwill (FQ)"                      NUMERIC DEFAULT 0,         -- financial_statement: Goodwill impairment (FQ)
    "Impairment of Goodwill (LTM)"                     NUMERIC DEFAULT 0,         -- financial_statement: Goodwill impairment (LTM)
    "Impairment of Goodwill (-1FY)"                    NUMERIC DEFAULT 0,         -- financial_statement: Goodwill impairment (previous FY)
    "Impairment of Goodwill (FY)"                      NUMERIC DEFAULT 0,         -- financial_statement: Goodwill impairment (FY)
    "Impairment of Goodwill (5YAVGFQ)"                 NUMERIC DEFAULT 0,         -- financial_statement: 5Y avg goodwill impairment (FQ)
    "Other Unusual Items/Total (LTM)"                  NUMERIC DEFAULT 0,         -- financial_statement: Other unusual items (LTM)
    "Gain (Loss) On Sale Of Assets (LTM)"              NUMERIC DEFAULT 0,         -- financial_statement: Gain/loss on asset sales (LTM)
    "Interest Expense/Total (LTM)"                     NUMERIC DEFAULT 0,         -- financial_statement: Interest expense (LTM)
    "Interest Income On Investments (LTM)"             NUMERIC DEFAULT 0,         -- financial_statement: Interest income (LTM)

    -- ===========================================
    -- BALANCE SHEET (balance_sheet role)
    -- ===========================================
    "Total Assets (LTM)"                               NUMERIC DEFAULT 0,         -- balance_sheet: Total assets (LTM)
    "Total Assets (FY)"                                NUMERIC DEFAULT 0,         -- balance_sheet: Total assets (FY)
    "Total Equity (FY)"                                NUMERIC DEFAULT 0,         -- balance_sheet: Total equity (FY)
    "Total Equity (LTM)"                               NUMERIC DEFAULT 0,         -- balance_sheet: Total equity (LTM)
    "Total Debt (FY)"                                  NUMERIC DEFAULT 0,         -- balance_sheet: Total debt (FY)
    "Total Debt (LTM)"                                 NUMERIC DEFAULT 0,         -- balance_sheet: Total debt (LTM)
    "Total Current Assets (LTM)"                       NUMERIC DEFAULT 0,         -- balance_sheet: Total current assets (LTM)
    "Total Current Liabilities (LTM)"                  NUMERIC DEFAULT 0,         -- balance_sheet: Total current liabilities (LTM)
    "Working Capital (LTM)"                            NUMERIC DEFAULT 0,         -- balance_sheet: Working capital (LTM)
    "Working Capital (FQ)"                             NUMERIC DEFAULT 0,         -- balance_sheet: Working capital (FQ)
    "Working Capital (FY)"                             NUMERIC DEFAULT 0,         -- balance_sheet: Working capital (FY)
    "Working Capital (5YAVGFY)"                        NUMERIC DEFAULT 0,         -- balance_sheet: 5Y avg working capital (FY)
    "TBV (FY)"                                         NUMERIC DEFAULT 0,         -- balance_sheet: Tangible book value (FY)
    "TBV (LTM)"                                        NUMERIC DEFAULT 0,         -- balance_sheet: Tangible book value (LTM)
    "Cash And Equivalents (LTM)"                       NUMERIC DEFAULT 0,         -- balance_sheet: Cash and equivalents (LTM)
    "Cash And Equivalents (FQ)"                        NUMERIC DEFAULT 0,         -- balance_sheet: Cash and equivalents (FQ)
    "Cash And Equivalents (FY)"                        NUMERIC DEFAULT 0,         -- balance_sheet: Cash and equivalents (FY)
    "Cash And Equivalents (5YAVGFQ)"                   NUMERIC DEFAULT 0,         -- balance_sheet: 5Y avg cash (FQ)
    "Retained Earnings (LTM)"                          NUMERIC DEFAULT 0,         -- balance_sheet: Retained earnings (LTM)
    "Retained Earnings (FQ)"                           NUMERIC DEFAULT 0,         -- balance_sheet: Retained earnings (FQ)
    "Retained Earnings (FY)"                           NUMERIC DEFAULT 0,         -- balance_sheet: Retained earnings (FY)
    "Retained Earnings (5YAVGFQ)"                      NUMERIC DEFAULT 0,         -- balance_sheet: 5Y avg retained earnings (FQ)
    "Inventory (LTM)"                                  NUMERIC DEFAULT 0,         -- balance_sheet: Inventory (LTM)
    "Inventory (FQ)"                                   NUMERIC DEFAULT 0,         -- balance_sheet: Inventory (FQ)
    "Inventory (FY)"                                   NUMERIC DEFAULT 0,         -- balance_sheet: Inventory (FY)
    "Inventory (5YAVGFQ)"                              NUMERIC DEFAULT 0,         -- balance_sheet: 5Y avg inventory (FQ)
    "Goodwill (FQ)"                                    NUMERIC DEFAULT 0,         -- balance_sheet: Goodwill (FQ)
    "Goodwill (LTM)"                                   NUMERIC DEFAULT 0,         -- balance_sheet: Goodwill (LTM)
    "Goodwill (FY)"                                    NUMERIC DEFAULT 0,         -- balance_sheet: Goodwill (FY)
    "Goodwill (-1FY)"                                  NUMERIC DEFAULT 0,         -- balance_sheet: Goodwill (previous FY)
    "Goodwill (5YAVGFQ)"                               NUMERIC DEFAULT 0,         -- balance_sheet: 5Y avg goodwill (FQ)
    "Gross Intangible Assets (LTM)"                    NUMERIC DEFAULT 0,         -- balance_sheet: Gross intangible assets (LTM)
    "Gross Intangible Assets (FY)"                     NUMERIC DEFAULT 0,         -- balance_sheet: Gross intangible assets (FY)
    "Gross Intangible Assets (5YAVGFQ)"                NUMERIC DEFAULT 0,         -- balance_sheet: 5Y avg intangibles (FQ)
    "Accounts Receivable/Total (FY)"                   NUMERIC DEFAULT 0,         -- balance_sheet: Accounts receivable (FY)
    "Accounts Receivable/Total (-1FY)"                 NUMERIC DEFAULT 0,         -- balance_sheet: Accounts receivable (previous FY)
    "Accounts Receivable/Total (5YAVGFQ)"              NUMERIC DEFAULT 0,         -- balance_sheet: 5Y avg receivables (FQ)

    -- ===========================================
    -- CASH FLOW (cash_flow role)
    -- ===========================================
    "CFF (LTM)"                                        NUMERIC DEFAULT 0,         -- cash_flow: Cash from financing (LTM)
    "CFF (FY)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Cash from financing (FY)
    "CFF (FQ)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Cash from financing (FQ)
    "CFF (-1FY)"                                       NUMERIC DEFAULT 0,         -- cash_flow: Cash from financing (previous FY)
    "CFI (LTM)"                                        NUMERIC DEFAULT 0,         -- cash_flow: Cash from investing (LTM)
    "CFI (FY)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Cash from investing (FY)
    "CFI (FQ)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Cash from investing (FQ)
    "CFI (-1FY)"                                       NUMERIC DEFAULT 0,         -- cash_flow: Cash from investing (previous FY)
    "FCF (LTM)"                                        NUMERIC DEFAULT 0,         -- cash_flow: Free cash flow (LTM)
    "FCF (FY)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Free cash flow (FY)
    "FCF (FQ)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Free cash flow (FQ)
    "FCF (5YAVGFQ)"                                    NUMERIC DEFAULT 0,         -- cash_flow: 5Y avg FCF (FQ)
    "CFO (LTM)"                                        NUMERIC DEFAULT 0,         -- cash_flow: Cash from operations (LTM)
    "CFO (FY)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Cash from operations (FY)
    "CFO (FQ)"                                         NUMERIC DEFAULT 0,         -- cash_flow: Cash from operations (FQ)
    "CFO (-1FY)"                                       NUMERIC DEFAULT 0,         -- cash_flow: Cash from operations (previous FY)
    "Capital Expenditure (LTM)"                        NUMERIC DEFAULT 0,         -- cash_flow: Capital expenditure (LTM)
    "Capital Expenditure (-1FY)"                       NUMERIC DEFAULT 0,         -- cash_flow: Capital expenditure (previous FY)
    "Capital Expenditure (FY)"                         NUMERIC DEFAULT 0,         -- cash_flow: Capital expenditure (FY)
    "Capital Expenditure (FQ)"                         NUMERIC DEFAULT 0,         -- cash_flow: Capital expenditure (FQ)
    "Capital Expenditure (5YAVGFQ)"                    NUMERIC DEFAULT 0,         -- cash_flow: 5Y avg capex (FQ)
    "Cash Acquisitions (LTM)"                          NUMERIC DEFAULT 0,         -- cash_flow: Cash acquisitions (LTM)
    "Cash Acquisitions (FY)"                           NUMERIC DEFAULT 0,         -- cash_flow: Cash acquisitions (FY)
    "Cash Acquisitions (FQ)"                           NUMERIC DEFAULT 0,         -- cash_flow: Cash acquisitions (FQ)
    "Cash Acquisitions (-1FY)"                         NUMERIC DEFAULT 0,         -- cash_flow: Cash acquisitions (previous FY)
    "Cash Acquisitions (5YAVGFQ)"                      NUMERIC DEFAULT 0,         -- cash_flow: 5Y avg acquisitions (FQ)

    -- ===========================================
    -- RATIOS (ratio role)
    -- ===========================================
    -- Valuation ratios
    "P/E (NTM)"                                        NUMERIC,                   -- ratio: Price-to-Earnings (Next Twelve Months)
    "P/E (LTM)"                                        NUMERIC,                   -- ratio: Price-to-Earnings (Last Twelve Months)
    "P/E (EST FY1)"                                    NUMERIC,                   -- ratio: P/E (Estimate FY1)
    "P/E (-1FYLTM)"                                    NUMERIC,                   -- ratio: P/E (1 FY ago LTM)
    "P/E (-2FYLTM)"                                    NUMERIC,                   -- ratio: P/E (2 FY ago LTM)
    "P/E (-3FYLTM)"                                    NUMERIC,                   -- ratio: P/E (3 FY ago LTM)
    "P/E (3YAVGLTM)"                                   NUMERIC,                   -- ratio: P/E (3-year average LTM)
    "P/E (-1FQLTM)"                                    NUMERIC,                   -- ratio: P/E (1 FQ ago LTM)
    "P/E (-2FQLTM)"                                    NUMERIC,                   -- ratio: P/E (2 FQ ago LTM)
    "P/E (-3FQLTM)"                                    NUMERIC,                   -- ratio: P/E (3 FQ ago LTM)
    "P/E (5YAVGLTM)"                                   NUMERIC,                   -- ratio: P/E (5-year average LTM)
    "P/E (-0FQQoQLTM)"                                 NUMERIC,                   -- ratio: P/E (Quarter-over-Quarter LTM)
    "P/E (-0FYYoYLTM)"                                 NUMERIC,                   -- ratio: P/E (Year-over-Year LTM)
    "P/E (-1FYYoYLTM)"                                 NUMERIC,                   -- ratio: P/E (1 FY YoY LTM)
    "P/E (-0FQYoYLTM)"                                 NUMERIC,                   -- ratio: P/E (Quarter YoY LTM)
    "P/B (LTM)"                                        NUMERIC,                   -- ratio: Price-to-Book (LTM)
    "P/B (-1FY)"                                       NUMERIC,                   -- ratio: Price-to-Book (previous FY)
    "P/B (5YAVG)"                                      NUMERIC,                   -- ratio: Price-to-Book (5Y average)
    "P/TBV (LTM)"                                      NUMERIC,                   -- ratio: Price-to-Tangible Book Value (LTM)
    "EV/Sales (EST FY1)"                               NUMERIC,                   -- ratio: EV/Sales (Estimate FY1)
    "EV/Sales (LTM)"                                   NUMERIC,                   -- ratio: EV/Sales (LTM)
    "EV/Sales (NTM)"                                   NUMERIC,                   -- ratio: EV/Sales (NTM)
    "EV/Sales (-1FYLTM)"                               NUMERIC,                   -- ratio: EV/Sales (1 FY ago LTM)
    "EV/Sales (-2FYLTM)"                               NUMERIC,                   -- ratio: EV/Sales (2 FY ago LTM)
    "EV/Sales (-3FYLTM)"                               NUMERIC,                   -- ratio: EV/Sales (3 FY ago LTM)
    "EV/Sales (3YAVGLTM)"                              NUMERIC,                   -- ratio: EV/Sales (3-year average LTM)
    "EV/Sales (-1FQLTM)"                               NUMERIC,                   -- ratio: EV/Sales (1 FQ ago LTM)
    "EV/Sales (-2FQLTM)"                               NUMERIC,                   -- ratio: EV/Sales (2 FQ ago LTM)
    "EV/Sales (-3FQLTM)"                               NUMERIC,                   -- ratio: EV/Sales (3 FQ ago LTM)
    "EV/Sales (-4FQLTM)"                               NUMERIC,                   -- ratio: EV/Sales (4 FQ ago LTM)
    "EV/EBITDA (LTM)"                                  NUMERIC,                   -- ratio: EV/EBITDA (LTM)
    "EV/EBITDA (NTM)"                                  NUMERIC,                   -- ratio: EV/EBITDA (NTM)
    "EV/EBITDA (-1FYLTM)"                              NUMERIC,                   -- ratio: EV/EBITDA (1 FY ago LTM)
    "EV/EBITDA (-1FQLTM)"                              NUMERIC,                   -- ratio: EV/EBITDA (1 FQ ago LTM)
    "EV/EBITDA (3YAVGLTM)"                             NUMERIC,                   -- ratio: EV/EBITDA (3-year average LTM)
    "EV/EBITDA (EST FY1)"                              NUMERIC,                   -- ratio: EV/EBITDA (Estimate FY1)

    -- Profitability/efficiency ratios
    "Return On Equity % (LTM)"                         NUMERIC,                   -- ratio: ROE (LTM)
    "Return On Equity % (FY)"                          NUMERIC,                   -- ratio: ROE (FY)
    "Return on Assets (ROA) % (LTM)"                   NUMERIC,                   -- ratio: ROA (LTM)
    "Return on Assets (ROA) % (FY)"                    NUMERIC,                   -- ratio: ROA (FY)
    "Current Ratio (FY)"                               NUMERIC,                   -- ratio: Current ratio (FY)
    "Current Ratio (LTM)"                              NUMERIC,                   -- ratio: Current ratio (LTM)
    "Asset Turnover (FY)"                              NUMERIC,                   -- ratio: Asset turnover (FY)
    "Asset Turnover (LTM)"                             NUMERIC,                   -- ratio: Asset turnover (LTM)

    -- Risk/credit scores
    "Altman Z-Score (FY)"                              NUMERIC,                   -- ratio: Altman Z-Score (FY)
    "Altman Z-Score (FQ)"                              NUMERIC,                   -- ratio: Altman Z-Score (FQ)
    "Altman Z-Score (LTM)"                             NUMERIC,                   -- ratio: Altman Z-Score (LTM)

    -- Per-share metrics (ratios)
    "EPS/Adj. (-1FY)"                                  NUMERIC,                   -- ratio: Adjusted EPS (previous FY)
    "EPS/Adj. (FY)"                                    NUMERIC,                   -- ratio: Adjusted EPS (FY)
    "EPS/Adj. (LTM)"                                   NUMERIC,                   -- ratio: Adjusted EPS (LTM)
    "EPS Norm - Est Avg (FY1E)"                        NUMERIC,                   -- ratio: EPS normalized estimate (FY1E)
    "EPS Norm - Est Avg (NTM)"                         NUMERIC,                   -- ratio: EPS normalized estimate (NTM)
    "Net EPS - Basic (LTM)"                            NUMERIC,                   -- ratio: Net EPS Basic (LTM)
    "Net EPS - Basic (FQ)"                             NUMERIC,                   -- ratio: Net EPS Basic (FQ)
    "Net EPS - Basic (FY)"                             NUMERIC,                   -- ratio: Net EPS Basic (FY)
    "Net EPS - Basic (-1FQFQ)"                         NUMERIC,                   -- ratio: Net EPS Basic (1 FQ ago)
    "Net EPS - Basic (-2FQFQ)"                         NUMERIC,                   -- ratio: Net EPS Basic (2 FQ ago)
    "Net EPS - Basic (-3FQFQ)"                         NUMERIC,                   -- ratio: Net EPS Basic (3 FQ ago)
    "Net EPS - Basic (-4FQFQ)"                         NUMERIC,                   -- ratio: Net EPS Basic (4 FQ ago)
    "Net EPS - Basic (-1FY)"                           NUMERIC,                   -- ratio: Net EPS Basic (1 FY ago)
    "Net EPS - Basic (-2FY)"                           NUMERIC,                   -- ratio: Net EPS Basic (2 FY ago)
    "Net EPS - Basic (-3FY)"                           NUMERIC,                   -- ratio: Net EPS Basic (3 FY ago)
    "Net EPS - Basic (-4FY)"                           NUMERIC,                   -- ratio: Net EPS Basic (4 FY ago)
    "Net EPS - Basic (-5FY)"                           NUMERIC,                   -- ratio: Net EPS Basic (5 FY ago)
    "EPS GAAP - Est Avg (NTM)"                         NUMERIC,                   -- ratio: EPS GAAP estimate (NTM)
    "EPS GAAP - Est Avg (FY1E)"                        NUMERIC,                   -- ratio: EPS GAAP estimate (FY1E)

    -- ===========================================
    -- PERCENTAGES (percentage role)
    -- ===========================================
    -- Returns
    "Total Return (YTD)"                               NUMERIC,                   -- percentage: YTD return
    "Total Return (5Y)"                                NUMERIC,                   -- percentage: 5-year total return
    "Total Return (10Y)"                               NUMERIC,                   -- percentage: 10-year total return
    "Tot. Return %/CAGR (3Y)"                          NUMERIC,                   -- percentage: 3-year total return CAGR
    "Tot. Return %/CAGR (10Y)"                         NUMERIC,                   -- percentage: 10-year total return CAGR
    "Price Chg. % (1M)"                                NUMERIC,                   -- percentage: 1-month price change
    "Price Chg. % (3M)"                                NUMERIC,                   -- percentage: 3-month price change
    "1-Day %"                                          NUMERIC,                   -- percentage: 1-day price change

    -- Growth rates
    "Total Revenues/CAGR (5Y FY)"                      NUMERIC,                   -- percentage: 5-year revenue CAGR
    "Revenues - Est YoY % (FY1E)"                      NUMERIC,                   -- percentage: Revenue estimate YoY change (FY1E)

    -- Margins
    "Net Income Margin % (FY)"                         NUMERIC,                   -- percentage: Net income margin (FY)
    "Net Income Margin % (LTM)"                        NUMERIC,                   -- percentage: Net income margin (LTM)
    "Gross Profit Margin % (FY)"                       NUMERIC,                   -- percentage: Gross profit margin (FY)
    "Gross Profit Margin % (LTM)"                      NUMERIC,                   -- percentage: Gross profit margin (LTM)

    -- Risk metrics
    "Volatility (1M)"                                  NUMERIC,                   -- percentage: 1-month volatility
    "Volatility (3M)"                                  NUMERIC,                   -- percentage: 3-month volatility
    "Volatility (6M)"                                  NUMERIC,                   -- percentage: 6-month volatility
    "Volatility (1Y)"                                  NUMERIC,                   -- percentage: 1-year volatility
    "Beta (1Y)"                                        NUMERIC,                   -- percentage: 1-year beta
    "Beta (2Y)"                                        NUMERIC,                   -- percentage: 2-year beta
    "Beta (5Y)"                                        NUMERIC,                   -- percentage: 5-year beta

    -- Dividend yields
    "Div Yield (Ind)"                                  NUMERIC,                   -- percentage: Dividend yield (Indicated)
    "Div Yield (LTM)"                                  NUMERIC,                   -- percentage: Dividend yield (LTM)
    "Div Yield (TTM)"                                  NUMERIC,                   -- percentage: Dividend yield (TTM)
    "Div Yield (NTM)"                                  NUMERIC,                   -- percentage: Dividend yield (NTM)
    "Div Yield (-1FYInd)"                              NUMERIC,                   -- percentage: Dividend yield (Previous FY indicated)
    "Div Yield (-2FYInd)"                              NUMERIC,                   -- percentage: Dividend yield (2 FY ago indicated)
    "Div Yield (-3FYInd)"                              NUMERIC,                   -- percentage: Dividend yield (3 FY ago indicated)
    "Div Yield (-4FYInd)"                              NUMERIC,                   -- percentage: Dividend yield (4 FY ago indicated)
    "Div Yield (-5FYInd)"                              NUMERIC,                   -- percentage: Dividend yield (5 FY ago indicated)
    "Div Yield (5YAVGLTM)"                             NUMERIC,                   -- percentage: Dividend yield (5Y avg LTM)
    "Buyback Yield (LTM)"                              NUMERIC,                   -- percentage: Buyback yield (LTM)

    -- Estimate revisions
    "EPS Est Avg Rev % (FY1E - 1W)"                    NUMERIC,                   -- percentage: EPS estimate revision (1 week)
    "EPS Est Avg Rev % (FY1E - 1M)"                    NUMERIC,                   -- percentage: EPS estimate revision (1 month)
    "EPS Est Avg Rev % (FY1E - 3M)"                    NUMERIC,                   -- percentage: EPS estimate revision (3 months)
    "EPS Est Avg Rev % (FY1E - 6M)"                    NUMERIC,                   -- percentage: EPS estimate revision (6 months)
    "EPS Est Avg Rev % (FY1E - 1Y)"                    NUMERIC,                   -- percentage: EPS estimate revision (1 year)
    "EPS GAAP Est Avg Rev % (FY1E - 1M)"               NUMERIC,                   -- percentage: EPS GAAP estimate revision (1 month)
    "EPS GAAP Est Avg Rev % (FY1E - 3M)"               NUMERIC,                   -- percentage: EPS GAAP estimate revision (3 months)
    "EPS GAAP Est Avg Rev % (FY1E - 6M)"               NUMERIC,                   -- percentage: EPS GAAP estimate revision (6 months)
    "EPS GAAP Est Avg Rev % (FY1E - 1Y)"               NUMERIC,                   -- percentage: EPS GAAP estimate revision (1 year)

    -- ===========================================
    -- COUNT (count role)
    -- ===========================================
    "Price Target - #"                                 NUMERIC DEFAULT 0,         -- count: Number of price targets
    "Analyst Rating"                                   NUMERIC DEFAULT 0,         -- count: Consensus analyst rating
    "# Strong Sell Ratings"                            NUMERIC DEFAULT 0,         -- count: Number of strong sell ratings
    "# Strong Buys Ratings"                            NUMERIC DEFAULT 0,         -- count: Number of strong buy ratings
    "# Hold Ratings"                                   NUMERIC DEFAULT 0,         -- count: Number of hold ratings
    "# Buys Ratings"                                   NUMERIC DEFAULT 0,         -- count: Number of buy ratings
    "# Sell Ratings"                                   NUMERIC DEFAULT 0,         -- count: Number of sell ratings
    "EPS Norm - Est # (FY1E)"                          NUMERIC DEFAULT 0,         -- count: Number of EPS estimates (FY1E)
    "Dividend Streak"                                  NUMERIC DEFAULT 0,         -- count: Consecutive years of dividend payments
    "Full Time Employees (FQ)"                         NUMERIC DEFAULT 0,         -- count: Full time employees (FQ)
    "Full Time Employees (FY)"                         NUMERIC DEFAULT 0,         -- count: Full time employees (FY)
    "Full Time Employees (-1FY)"                       NUMERIC DEFAULT 0,         -- count: Full time employees (previous FY)
    "Full Time Employees (-2FY)"                       NUMERIC DEFAULT 0,         -- count: Full time employees (2 years ago)
    "Full Time Employees (-3FY)"                       NUMERIC DEFAULT 0,         -- count: Full time employees (3 years ago)
    "Avg Employees (5YAVGFY)"                          NUMERIC                    -- count: Average employees (5-year average FY)

) TABLESPACE pg_default;

ALTER TABLE equities
    OWNER TO postgres;

COMMENT ON TABLE equities IS 'Equities screening data with financial metrics and company information (semantic roles aligned with preprocessing pipeline)';