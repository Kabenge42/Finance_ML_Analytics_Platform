-- Drop existing table if it exists
DROP TABLE IF EXISTS "Equities";
DROP TABLE IF EXISTS equities;
-- Drop existing table if it exists
DROP TABLE IF EXISTS "Equities";
DROP TABLE IF EXISTS equities;

/*
 * SEMANTIC CATEGORY CLASSIFICATION SYSTEM
 * =======================================
 *
 * This schema integrates semantic categories from finance_ml/ml_workflow/preprocessing/column_semantics.py
 * to enable intelligent preprocessing decisions. Each numerical column is classified into one of five
 * semantic categories that determine appropriate preprocessing transformations:
 *
 * 1. PRICE COLUMNS (21 columns)
 *    - Definition: Current prices, price targets, historical prices, 52w bounds, EMAs
 *    - Preprocessing: NEVER transform - must preserve original dollar units
 *    - Business Rationale: Required for core valuation metric (Predicted_Target - Last_Price) / Last_Price
 *    - Examples: last_price, price_target, price_5d_ago, 52w_high_adj, ema_20d
 *
 * 2. MARKET_VALUE COLUMNS (20 columns)
 *    - Definition: Market cap, enterprise value, balance sheet, income statement, cash flow items
 *    - Preprocessing: Log-transform recommended (highly skewed, often skewness > 2.0)
 *    - Business Rationale: Preserve information about valid extremes while normalizing scale
 *    - Examples: market_cap, enterprise_value, revenue, ebitda, total_assets, total_debt
 *
 * 3. RATIO COLUMNS (~70 columns)
 *    - Definition: Pre-normalized financial ratios (valuation, profitability, leverage, liquidity)
 *    - Preprocessing: Already normalized - winsorization may not be needed
 *    - Business Rationale: Ratios are relative metrics already normalized across scales
 *    - Examples: p_e, p_b, roe, roa, debt_equity, current_ratio, ev_ebitda, ev_sales
 *
 * 4. PERCENTAGE COLUMNS (~25 columns)
 *    - Definition: Margin metrics, growth rates, volatility measures (bounded [0, 100])
 *    - Preprocessing: Already bounded - percentile capping inappropriate
 *    - Business Rationale: Naturally bounded percentage metrics
 *    - Examples: gross_margin, operating_margin, revenue_growth_yoy, volatility_1y, beta
 *
 * 5. COUNT COLUMNS (8 columns)
 *    - Definition: Discrete integer counts (analyst ratings, employees)
 *    - Preprocessing: Discrete values - inappropriate for continuous scaling
 *    - Business Rationale: Integer counts require different treatment than continuous variables
 *    - Examples: num_analysts, num_strong_buy_ratings, num_employees
 *
 * CATEGORICAL COLUMNS (9 columns)
 *    - Definition: Non-numeric categories for grouping and encoding
 *    - Preprocessing: One-hot encoding, label encoding, or target encoding
 *    - Examples: sector, industry, region, country, exchange, style_class, size_class
 *
 * PREPROCESSING IMPLICATIONS:
 * - Winsorization: Apply to MARKET_VALUE and OTHER columns only
 * - Log-Transform: Apply to MARKET_VALUE columns only
 * - Scaling: Apply to all except PRICE columns
 * - Encoding: Apply to CATEGORICAL columns only
 *
 * See finance_ml/ml_workflow/preprocessing/column_semantics.py for implementation details.
 */

-- Create equities table with appropriate data types
CREATE TABLE equities
(
    -- ====================
    -- IDENTIFIER COLUMNS
    -- ====================
    "Ticker"                                   TEXT,
    "ISIN"                                     TEXT,
    "Name"                                     TEXT,
    "Description"                              TEXT,
    -- CATEGORICAL: Exchange where stock trades
    "Exchange"                                 TEXT,
    "Unit"                                     TEXT,
    -- CATEGORICAL: GICS Sector classification
    "Sector"                                   TEXT,
    -- CATEGORICAL: GICS Industry classification
    "Industry"                                 TEXT,
    "Last Updated"                             DATE,
    "Income Statement Report Date"             DATE,
    "Next Earnings"                            DATE,
    -- CATEGORICAL: Investment style (Value, Growth, Blend)
    "Style Class"                              TEXT,
    -- CATEGORICAL: Earnings announcement status
    "Next Earnings (Status)"                   TEXT,
    -- CATEGORICAL: Market cap size (Small, Mid, Large)
    "Size Class"                               TEXT,
    -- CATEGORICAL: Geographic region
    "Region"                                   TEXT,
    -- CATEGORICAL: Country of incorporation
    "Country"                                  TEXT,
    -- CATEGORICAL: Country where stock trades
    "Trading Country"                          TEXT,

    -- ====================
    -- MARKET VALUE COLUMNS - Log-transform recommended
    -- ====================
    "Market Cap"                               NUMERIC, -- MARKET_VALUE: Market capitalization
    "Enterprise Value"                         NUMERIC, -- MARKET_VALUE: Enterprise value

    -- ====================
    -- PRICE COLUMNS - NEVER transform (preserve original dollar units)
    -- ====================
    "Last Price"                               NUMERIC, -- PRICE: Current market price (critical for valuation)
    "Price Target (YTD Ago)"                   NUMERIC, -- PRICE: Historical price target (YTD)
    "Total Return (YTD)"                       NUMERIC, -- PERCENTAGE: YTD return percentage
    "Price Target"                             NUMERIC, -- PRICE: Analyst consensus target price
    "Price Target - Low"                       NUMERIC, -- PRICE: Low analyst price target
    "Price Target - Median"                    NUMERIC, -- PRICE: Median analyst price target
    "Price Target - High"                      NUMERIC, -- PRICE: High analyst price target
    "Price Target - #"                         NUMERIC, -- COUNT: Number of price targets

    -- ====================
    -- RATIO COLUMNS - Pre-normalized financial ratios
    -- ====================
    "P/E (NTM)"                                NUMERIC, -- RATIO: Price-to-Earnings (Next Twelve Months)
    "P/E (LTM)"                                NUMERIC, -- RATIO: Price-to-Earnings (Last Twelve Months)
    "Altman Z-Score (FY)"                      NUMERIC, -- RATIO: Bankruptcy prediction score (Fiscal Year)
    "Altman Z-Score (FQ)"                      NUMERIC, -- RATIO: Bankruptcy prediction score (Fiscal Quarter)
    "Altman Z-Score (LTM)"                     NUMERIC, -- RATIO: Bankruptcy prediction score (LTM)

    -- ====================
    -- PERCENTAGE COLUMNS - Volatility & Risk Metrics
    -- ====================
    "Beta (1Y)"                                NUMERIC, -- PERCENTAGE: 1-year beta
    "Beta (2Y)"                                NUMERIC, -- PERCENTAGE: 2-year beta
    "Beta (5Y)"                                NUMERIC, -- PERCENTAGE: 5-year beta

    -- ====================
    -- COUNT COLUMNS - Analyst Coverage
    -- ====================
    "Analyst Rating"                           NUMERIC, -- COUNT: Consensus analyst rating
    "# Strong Sell Ratings"                    NUMERIC, -- COUNT: Number of strong sell ratings
    "# Strong Buys Ratings"                    NUMERIC, -- COUNT: Number of strong buy ratings
    "# Hold Ratings"                           NUMERIC, -- COUNT: Number of hold ratings
    "# Buys Ratings"                           NUMERIC, -- COUNT: Number of buy ratings
    "# Sell Ratings"                           NUMERIC, -- COUNT: Number of sell ratings

    -- ====================
    -- PERCENTAGE/MARKET_VALUE - Growth & Revenue Metrics
    -- ====================
    "Total Revenues/CAGR (5Y FY)"              NUMERIC, -- PERCENTAGE: 5-year revenue CAGR
    "Total Revenues (FQ)"                      NUMERIC, -- MARKET_VALUE: Total revenues (Fiscal Quarter)
    "Total Revenues (-1FY)"                    NUMERIC, -- MARKET_VALUE: Total revenues (Previous Fiscal Year)
    "Total Revenues (FY)"                      NUMERIC, -- MARKET_VALUE: Total revenues (Fiscal Year)
    "Total Revenues (LTM)"                     NUMERIC, -- MARKET_VALUE: Total revenues (Last Twelve Months)
    "Total Operating Expenses (LTM)"           NUMERIC, -- MARKET_VALUE: Operating expenses (LTM)

    -- RATIO: Tangible Book Value Metrics
    "P/TBV (LTM)"                              NUMERIC, -- RATIO: Price-to-Tangible Book Value (LTM)
    "TBV (FY)"                                 NUMERIC, -- MARKET_VALUE: Tangible book value (FY)
    "TBV (LTM)"                                NUMERIC, -- MARKET_VALUE: Tangible book value (LTM)
    "Market Cap (Country R)"                   NUMERIC, -- MARKET_VALUE: Market cap country ranking

    -- PERCENTAGE: Total Return Metrics
    "Tot. Return %/CAGR (3Y)"                  NUMERIC, -- PERCENTAGE: 3-year total return CAGR
    "Tot. Return %/CAGR (10Y)"                 NUMERIC, -- PERCENTAGE: 10-year total return CAGR
    "Total Return (5Y)"                        NUMERIC, -- PERCENTAGE: 5-year total return
    "Total Return (10Y)"                       NUMERIC, -- PERCENTAGE: 10-year total return

    -- MARKET_VALUE: Net Income & Cash Flow Metrics
    "Net Income/Adj. (-1FY)"                   NUMERIC, -- MARKET_VALUE: Adjusted net income (Previous FY)
    "CFF (LTM)"                                NUMERIC, -- MARKET_VALUE: Cash from financing (LTM)
    "CFI (LTM)"                                NUMERIC, -- MARKET_VALUE: Cash from investing (LTM)
    "FCF (LTM)"                                NUMERIC, -- MARKET_VALUE: Free cash flow (LTM)
    "CFO (LTM)"                                NUMERIC, -- MARKET_VALUE: Cash from operations (LTM)

    -- MARKET_VALUE: EBITDA Metrics
    "EBITDA (FQ)"                              NUMERIC, -- MARKET_VALUE: EBITDA (Fiscal Quarter)
    "EBITDA (LTM)"                             NUMERIC, -- MARKET_VALUE: EBITDA (Last Twelve Months)
    "EBITDA (FY)"                              NUMERIC, -- MARKET_VALUE: EBITDA (Fiscal Year)
    "EBITDA (-1FY)"                            NUMERIC, -- MARKET_VALUE: EBITDA (Previous Fiscal Year)
    "EBITDA/Adj. (LTM)"                        NUMERIC, -- MARKET_VALUE: Adjusted EBITDA (LTM)
    "EBITDA/Adj. (FY)"                         NUMERIC, -- MARKET_VALUE: Adjusted EBITDA (FY)
    "EBITDA/Adj. (-1FY)"                       NUMERIC, -- MARKET_VALUE: Adjusted EBITDA (Previous FY)

    -- MARKET_VALUE: EBIT Metrics
    "EBIT (FQ)"                                NUMERIC, -- MARKET_VALUE: EBIT (Fiscal Quarter)
    "EBIT (LTM)"                               NUMERIC, -- MARKET_VALUE: EBIT (Last Twelve Months)
    "EBIT (FY)"                                NUMERIC, -- MARKET_VALUE: EBIT (Fiscal Year)
    "EBIT (-1FY)"                              NUMERIC, -- MARKET_VALUE: EBIT (Previous Fiscal Year)
    "EBIT/Adj. (-1FY)"                         NUMERIC, -- MARKET_VALUE: Adjusted EBIT (Previous FY)
    "EBIT/Adj. (FY)"                           NUMERIC, -- MARKET_VALUE: Adjusted EBIT (FY)
    "EBIT/Adj. (LTM)"                          NUMERIC, -- MARKET_VALUE: Adjusted EBIT (LTM)
    "EBIT - Est Med (FY1E)"                    NUMERIC, -- MARKET_VALUE: EBIT estimate median (FY1E)
    "EBIT - Est Med (NTM)"                     NUMERIC, -- MARKET_VALUE: EBIT estimate median (NTM)

    -- RATIO: Profitability Metrics
    "Return On Equity % (LTM)"                 NUMERIC, -- RATIO: ROE percentage (LTM)
    "Return On Equity % (FY)"                  NUMERIC, -- RATIO: ROE percentage (FY)

    -- MARKET_VALUE: Net Income Metrics
    "Net Income - (IS) (FY)"                   NUMERIC, -- MARKET_VALUE: Net income from income statement (FY)
    "Net Income - (IS) (LTM)"                  NUMERIC, -- MARKET_VALUE: Net income from income statement (LTM)
    "Normalized Net Income (FY)"               NUMERIC, -- MARKET_VALUE: Normalized net income (FY)
    "Normalized Net Income (LTM)"              NUMERIC, -- MARKET_VALUE: Normalized net income (LTM)
    "Net Income/Adj. (FY)"                     NUMERIC, -- MARKET_VALUE: Adjusted net income (FY)
    "Net Income/Adj. (LTM)"                    NUMERIC, -- MARKET_VALUE: Adjusted net income (LTM)

    -- PERCENTAGE: Margin Metrics
    "Net Income Margin % (FY)"                 NUMERIC, -- PERCENTAGE: Net income margin (FY)
    "Net Income Margin % (LTM)"                NUMERIC, -- PERCENTAGE: Net income margin (LTM)

    -- PERCENTAGE: Volatility Metrics
    "Volatility (1M)"                          NUMERIC, -- PERCENTAGE: 1-month volatility
    "Volatility (3M)"                          NUMERIC, -- PERCENTAGE: 3-month volatility
    "Volatility (6M)"                          NUMERIC, -- PERCENTAGE: 6-month volatility
    "Volatility (1Y)"                          NUMERIC, -- PERCENTAGE: 1-year volatility

    -- OTHER: Volume & Dividend Metrics
    "Volume (Shrs)"                            NUMERIC, -- OTHER: Trading volume in shares
    "Dividend Per Share (LTM)"                 NUMERIC, -- PRICE: Dividend per share (LTM)
    "Div Yield (Ind)"                          NUMERIC, -- PERCENTAGE: Dividend yield (Indicated)
    "Div Yield (LTM)"                          NUMERIC, -- PERCENTAGE: Dividend yield (LTM)

    -- MARKET_VALUE: Balance Sheet Metrics
    "Total Debt (FY)"                          NUMERIC, -- MARKET_VALUE: Total debt (FY)
    "Total Equity (FY)"                        NUMERIC, -- MARKET_VALUE: Total equity (FY)
    "Total Equity (LTM)"                       NUMERIC, -- MARKET_VALUE: Total equity (LTM)
    "Total Debt (LTM)"                         NUMERIC, -- MARKET_VALUE: Total debt (LTM)
    "Total Assets (LTM)"                       NUMERIC, -- MARKET_VALUE: Total assets (LTM)
    "Total Assets (FY)"                        NUMERIC, -- MARKET_VALUE: Total assets (FY)

    -- RATIO: Liquidity & Efficiency Ratios
    "Current Ratio (FY)"                       NUMERIC, -- RATIO: Current ratio (FY)
    "Current Ratio (LTM)"                      NUMERIC, -- RATIO: Current ratio (LTM)

    -- PERCENTAGE: Margin Ratios
    "Gross Profit Margin % (FY)"               NUMERIC, -- PERCENTAGE: Gross profit margin (FY)
    "Gross Profit Margin % (LTM)"              NUMERIC, -- PERCENTAGE: Gross profit margin (LTM)

    -- RATIO: Efficiency Metrics
    "Asset Turnover (FY)"                      NUMERIC, -- RATIO: Asset turnover (FY)
    "Asset Turnover (LTM)"                     NUMERIC, -- RATIO: Asset turnover (LTM)

    -- MARKET_VALUE: Gross Profit
    "Gross Profit (LTM)"                       NUMERIC, -- MARKET_VALUE: Gross profit (LTM)
    "Gross Profit (FY)"                        NUMERIC, -- MARKET_VALUE: Gross profit (FY)
    "EPS Norm - Est Avg (NTM)"                 NUMERIC,
    "EPS/Adj. (-1FY)"                          NUMERIC,
    "EPS/Adj. (FY)"                            NUMERIC,
    "EPS/Adj. (LTM)"                           NUMERIC,
    "EPS Norm - Est Avg (FY1E)"                NUMERIC,
    "Gain (Loss) On Sale Of Assets (LTM)"      NUMERIC,
    "Cost Of Revenues (LTM)"                   NUMERIC,
    "Cash Acquisitions (LTM)"                  NUMERIC,
    "Cash Acquisitions (FY)"                   NUMERIC,
    "Cash Acquisitions (-1FY)"                 NUMERIC,
    "Inventory (LTM)"                          NUMERIC,
    "Goodwill (FQ)"                            NUMERIC,
    "Goodwill (LTM)"                           NUMERIC,
    "Goodwill (FY)"                            NUMERIC,
    "Goodwill (-1FY)"                          NUMERIC,
    "Impairment of Goodwill (FQ)"              NUMERIC,
    "Impairment of Goodwill (LTM)"             NUMERIC,
    "Impairment of Goodwill (-1FY)"            NUMERIC,
    "Impairment of Goodwill (FY)"              NUMERIC,
    "Operating Income (LTM)"                   NUMERIC,
    "Asset Writedown (LTM)"                    NUMERIC,
    "Asset Writedown (FY)"                     NUMERIC,
    "Asset Writedown (-1FY)"                   NUMERIC,
    "Operating Income (FY)"                    NUMERIC,
    "Capital Expenditure (LTM)"                NUMERIC,
    "Capital Expenditure (-1FY)"               NUMERIC,
    "Capital Expenditure (FY)"                 NUMERIC,
    "Retained Earnings (LTM)"                  NUMERIC,
    "Total Current Assets (LTM)"               NUMERIC,
    "Total Current Liabilities (LTM)"          NUMERIC,
    "R&D Expenses (LTM)"                       NUMERIC,
    "Restructuring Charges (LTM)"              NUMERIC,
    "Restructuring Charges (FQ)"               NUMERIC,
    "Restructuring Charges (-1FY)"             NUMERIC,
    "Restructuring Charges (FY)"               NUMERIC,
    "Interest Expense/Total (LTM)"             NUMERIC,
    "Merger & Restructuring Charges (LTM)"     NUMERIC,
    "Working Capital (LTM)"                    NUMERIC,
    "Other Unusual Items/Total (LTM)"          NUMERIC,
    "Interest Income On Investments (LTM)"     NUMERIC,
    "Buyback Yield (LTM)"                      NUMERIC,
    "Return on Assets (ROA) % (LTM)"           NUMERIC,
    "Return on Assets (ROA) % (FY)"            NUMERIC,
    "Net Income - (IS) (-1FY)"                 NUMERIC,
    "Normalized Net Income (-1FY)"             NUMERIC,
    "P/E (-1FYLTM)"                            NUMERIC,
    "CFF (FY)"                                 NUMERIC,
    "CFF (-1FY)"                               NUMERIC,
    "CFI (FY)"                                 NUMERIC,
    "CFI (-1FY)"                               NUMERIC,
    "CFO (FY)"                                 NUMERIC,
    "CFO (-1FY)"                               NUMERIC,
    "Div Yield (-1FYInd)"                      NUMERIC,
    "FCF (FY)"                                 NUMERIC,
    "Capital Expenditure (FQ)"                 NUMERIC,
    "Capital Expenditure (5YAVGFQ)"            NUMERIC,
    "CFF (FQ)"                                 NUMERIC,
    "CFI (FQ)"                                 NUMERIC,
    "CFO (FQ)"                                 NUMERIC,
    "FCF (FQ)"                                 NUMERIC,
    "Total Revenues (5YAVGFQ)"                 NUMERIC,
    "EBITDA (5YAVGFQ)"                         NUMERIC,
    "EBIT (5YAVGFQ)"                           NUMERIC,
    "P/E (5YAVGLTM)"                           NUMERIC,
    "FCF (5YAVGFQ)"                            NUMERIC,
    "Cash Acquisitions (FQ)"                   NUMERIC,
    "Cash Acquisitions (5YAVGFQ)"              NUMERIC,
    "Asset Writedown (FQ)"                     NUMERIC,
    "Asset Writedown (5YAVGFQ)"                NUMERIC,
    "Impairment of Goodwill (5YAVGFQ)"         NUMERIC,
    "Operating Income (FQ)"                    NUMERIC,
    "Operating Income (5YAVGFQ)"               NUMERIC,
    "P/B (LTM)"                                NUMERIC,
    "P/B (-1FY)"                               NUMERIC,
    "P/B (5YAVG)"                              NUMERIC,
    "Cash And Equivalents (LTM)"               NUMERIC,
    "Cash And Equivalents (FQ)"                NUMERIC,
    "Cash And Equivalents (FY)"                NUMERIC,
    "Cash And Equivalents (5YAVGFQ)"           NUMERIC,
    "Inventory (FQ)"                           NUMERIC,
    "Inventory (FY)"                           NUMERIC,
    "Goodwill (5YAVGFQ)"                       NUMERIC,
    "Inventory (5YAVGFQ)"                      NUMERIC,
    "Retained Earnings (FQ)"                   NUMERIC,
    "Retained Earnings (FY)"                   NUMERIC,
    "Retained Earnings (5YAVGFQ)"              NUMERIC,
    "Working Capital (FQ)"                     NUMERIC,
    "Working Capital (FY)"                     NUMERIC,
    "Working Capital (5YAVGFY)"                NUMERIC,
    "Div Yield (TTM)"                          NUMERIC,
    "Div Yield (NTM)"                          NUMERIC,
    "Div Yield (5YAVGLTM)"                     NUMERIC,
    "Gross Intangible Assets (LTM)"            NUMERIC,
    "Gross Intangible Assets (FY)"             NUMERIC,
    "Gross Intangible Assets (5YAVGFQ)"        NUMERIC,
    "Restructuring Charges (5YAVGFQ)"          NUMERIC,
    "Merger & Restructuring Charges (FQ)"      NUMERIC,
    "Merger & Restructuring Charges (FY)"      NUMERIC,
    "Merger & Restructuring Charges (5YAVGFQ)" NUMERIC,
    "Normalized Net Income (FQ)"               NUMERIC,
    "Normalized Net Income (5YAVGFQ)"          NUMERIC,
    "Net Income/Adj. (FQ)"                     NUMERIC,
    "Net Income/Adj. (5YAVGFQ)"                NUMERIC,
    "Net Income - (IS) (FQ)"                   NUMERIC,
    "Net Income - (IS) (5YAVGFQ)"              NUMERIC,
    "Net Income - (IS) (5YAVGLTM)"             NUMERIC,
    "Normalized Net Income (5YAVGLTM)"         NUMERIC,
    "EBITDA (5YAVGLTM)"                        NUMERIC,
    "EBIT (5YAVGLTM)"                          NUMERIC,
    "Total Revenues (5YAVGLTM)"                NUMERIC,
    "Revenues - Est YoY % (FY1E)"                      NUMERIC,
    -- PERCENTAGE: Price Change Metrics
    "Price Chg. % (1M)"                        NUMERIC, -- PERCENTAGE: 1-month price change
    "Price Chg. % (3M)"                        NUMERIC, -- PERCENTAGE: 3-month price change
    "1-Day %"                                  NUMERIC, -- PERCENTAGE: 1-day price change

    -- PRICE: Historical Prices (for momentum calculations)
    "Price (5D Ago)"                           NUMERIC, -- PRICE: Price 5 days ago
    "Price (1W Ago)"                           NUMERIC, -- PRICE: Price 1 week ago
    "Price (1M Ago)"                           NUMERIC, -- PRICE: Price 1 month ago
    "Price (3M Ago)"                           NUMERIC, -- PRICE: Price 3 months ago
    "Price (6M Ago)"                           NUMERIC, -- PRICE: Price 6 months ago
    "Price (1Y Ago)"                           NUMERIC, -- PRICE: Price 1 year ago
    "Price (3Y Ago)"                           NUMERIC, -- PRICE: Price 3 years ago
    "Price (5Y Ago)"                           NUMERIC, -- PRICE: Price 5 years ago
    "Price (QTD Ago)"                          NUMERIC, -- PRICE: Price quarter-to-date ago
    "Rel. Volume"                                      NUMERIC,
    "Shrs Out"                                         NUMERIC,
    "Shrs Out (-1FY)"                                  NUMERIC,
    "Common Dividends Paid (LTM)"                      NUMERIC,
    "Common Dividends Paid (FY)"                       NUMERIC,
    "Selling General & Admin Expenses/Total (FQ)"      NUMERIC,
    "Selling General & Admin Expenses/Total (FY)"      NUMERIC,
    "Selling General & Admin Expenses/Total (-1FY)"    NUMERIC,
    "Selling General & Admin Expenses/Total (5YAVGFQ)" NUMERIC,
    "Accounts Receivable/Total (FY)"                   NUMERIC,
    "Accounts Receivable/Total (-1FY)"                 NUMERIC,
    "Accounts Receivable/Total (5YAVGFQ)"              NUMERIC,
    "Marketing Expenses (FQ)"                          NUMERIC,
    "Marketing Expenses (FY)"                          NUMERIC,
    "Marketing Expenses (-1FY)"                        NUMERIC,
    "Marketing Expenses (5YAVGLTM)"            NUMERIC,
    -- Phase 9.3 Schema Version 1.3 additions: 48 new columns
    -- Category 1: Revenue Forecasting Estimates (4 columns)
    "Revenues - Est Avg (NTM)"                 NUMERIC, -- MARKET_VALUE: Revenue estimate average (NTM)
    "Revenues - Est Avg (FY1E)"                NUMERIC, -- MARKET_VALUE: Revenue estimate average (FY1E)
    "Revenues - Est Med (NTM)"                 NUMERIC, -- MARKET_VALUE: Revenue estimate median (NTM)
    "Revenues - Est Med (FY1E)"                NUMERIC, -- MARKET_VALUE: Revenue estimate median (FY1E)
    -- Category 2: EV/Sales Time-Series (11 columns)
    "EV/Sales (EST FY1)"                       NUMERIC, -- RATIO: EV/Sales (Estimate FY1)
    "EV/Sales (LTM)"                           NUMERIC, -- RATIO: EV/Sales (Last Twelve Months)
    "EV/Sales (NTM)"                           NUMERIC, -- RATIO: EV/Sales (Next Twelve Months)
    "EV/Sales (-1FYLTM)"                       NUMERIC, -- RATIO: EV/Sales (1 FY ago LTM)
    "EV/Sales (-2FYLTM)"                       NUMERIC, -- RATIO: EV/Sales (2 FY ago LTM)
    "EV/Sales (-3FYLTM)"                       NUMERIC, -- RATIO: EV/Sales (3 FY ago LTM)
    "EV/Sales (3YAVGLTM)"                      NUMERIC, -- RATIO: EV/Sales (3-year average LTM)
    "EV/Sales (-1FQLTM)"                       NUMERIC, -- RATIO: EV/Sales (1 FQ ago LTM)
    "EV/Sales (-2FQLTM)"                       NUMERIC, -- RATIO: EV/Sales (2 FQ ago LTM)
    "EV/Sales (-3FQLTM)"                       NUMERIC, -- RATIO: EV/Sales (3 FQ ago LTM)
    "EV/Sales (-4FQLTM)"                       NUMERIC, -- RATIO: EV/Sales (4 FQ ago LTM)
    -- Category 3: Employment Metrics (7 columns)
    "Full Time Employees (FQ)"   NUMERIC,               -- COUNT: Full time employees (Fiscal Quarter)
    "Full Time Employees (FY)"   NUMERIC,               -- COUNT: Full time employees (Fiscal Year)
    "Full Time Employees (-1FY)" NUMERIC,               -- COUNT: Full time employees (Previous Fiscal Year)
    "Full Time Employees (-2FY)" NUMERIC,               -- COUNT: Full time employees (2 Years Ago)
    "Full Time Employees (-3FY)" NUMERIC,               -- COUNT: Full time employees (3 Years Ago)
    "Avg Employees (5YAVGFY)"    NUMERIC,               -- COUNT: Average employees (5-year average Fiscal Year)
    -- Category 4: Technical Indicators (6 columns)
    -- PRICE: 52-Week Bounds (for relative positioning)
    "52W High/Adj"                             NUMERIC, -- PRICE: 52-week adjusted high
    "52W Low/Adj"                              NUMERIC, -- PRICE: 52-week adjusted low
    -- PRICE: Exponential Moving Averages (technical indicators)
    "EMA (20D)"                                NUMERIC, -- PRICE: 20-day EMA
    "EMA (50D)"                                NUMERIC, -- PRICE: 50-day EMA
    "EMA (100D)"                               NUMERIC, -- PRICE: 100-day EMA
    "EMA (250D)"                               NUMERIC, -- PRICE: 250-day EMA (1-year trend proxy)
    -- Category 5: EV/EBITDA Extended Time-Series (6 columns)
    "EV/EBITDA (LTM)"                          NUMERIC, -- RATIO: EV/EBITDA (Last Twelve Months)
    "EV/EBITDA (NTM)"                          NUMERIC, -- RATIO: EV/EBITDA (Next Twelve Months)
    "EV/EBITDA (-1FYLTM)"                      NUMERIC, -- RATIO: EV/EBITDA (1 FY ago LTM)
    "EV/EBITDA (-1FQLTM)"                      NUMERIC, -- RATIO: EV/EBITDA (1 FQ ago LTM)
    "EV/EBITDA (3YAVGLTM)"                     NUMERIC, -- RATIO: EV/EBITDA (3-year average LTM)
    "EV/EBITDA (EST FY1)"                      NUMERIC, -- RATIO: EV/EBITDA (Estimate FY1)
    -- Category 6: P/E Extended Time-Series (11 columns)
    "P/E (EST FY1)"                            NUMERIC, -- RATIO: P/E (Estimate FY1)
    "P/E (-2FYLTM)"                            NUMERIC, -- RATIO: P/E (2 FY ago LTM)
    "P/E (-3FYLTM)"                            NUMERIC, -- RATIO: P/E (3 FY ago LTM)
    "P/E (3YAVGLTM)"                           NUMERIC, -- RATIO: P/E (3-year average LTM)
    "P/E (-1FQLTM)"                            NUMERIC, -- RATIO: P/E (1 FQ ago LTM)
    "P/E (-2FQLTM)"                            NUMERIC, -- RATIO: P/E (2 FQ ago LTM)
    "P/E (-3FQLTM)"                            NUMERIC, -- RATIO: P/E (3 FQ ago LTM)
    "P/E (-0FQQoQLTM)"                         NUMERIC, -- RATIO: P/E (Quarter-over-Quarter LTM)
    "P/E (-0FYYoYLTM)"                         NUMERIC, -- RATIO: P/E (Year-over-Year LTM)
    "P/E (-1FYYoYLTM)"                         NUMERIC, -- RATIO: P/E (1 FY YoY LTM)
    "P/E (-0FQYoYLTM)"                         NUMERIC, -- RATIO: P/E (Quarter YoY LTM)
    -- Category 7: Dividend Record Information (8 columns)
    "Dividend Record (Announce Date)"          DATE,    -- Date: Dividend announcement date
    "Dividend Record (Ex Date)"                DATE,    -- Date: Dividend ex-date
    "Dividend Record (Payable Date)"           DATE,    -- Date: Dividend payable date
    "Dividend Record (Record Date)"            DATE,    -- Date: Dividend record date
    "Dividend Record (Frequency)"              TEXT,    -- CATEGORICAL: Dividend frequency (Quarterly, Annual, etc.)
    "Dividend Record (Currency)"               TEXT,    -- CATEGORICAL: Dividend currency
    "Dividend Record (Amount)"                 NUMERIC, -- PRICE: Dividend amount
    "Dividend Streak"                          NUMERIC  -- COUNT: Consecutive years of dividend payments
) TABLESPACE pg_default;

-- Set table ownership
ALTER TABLE equities
    OWNER TO postgres;

-- Add comments
COMMENT ON TABLE equities IS 'Equities screening data with financial metrics and company information';

/*
 * =======================================
 * SEMANTIC CATEGORY SUMMARY & COLUMN INDEX
 * =======================================
 *
 * This section provides a complete mapping of columns to their semantic categories
 * as defined in finance_ml/ml_workflow/preprocessing/column_semantics.py
 *
 * PRICE COLUMNS (21 total - NEVER transform):
 * -------------------------------------------
 * Current prices and targets:
 *   - Last Price, Price Target, Price Target (YTD Ago)
 *   - Price Target - Low, Price Target - Median, Price Target - High
 *
 * Historical prices (for momentum):
 *   - Price (5D Ago), Price (1W Ago), Price (1M Ago)
 *   - Price (3M Ago), Price (6M Ago), Price (1Y Ago)
 *   - Price (3Y Ago), Price (5Y Ago), Price (QTD Ago)
 *
 * 52-week bounds (for relative positioning):
 *   - 52W High/Adj, 52W Low/Adj
 *
 * Exponential moving averages (technical indicators):
 *   - EMA (20D), EMA (50D), EMA (100D), EMA (250D)
 *
 * Other price-related:
 *   - Dividend Record (Amount)
 *
 * MARKET_VALUE COLUMNS (20+ core columns - Log-transform recommended):
 * ---------------------------------------------------------------------
 * Market cap & enterprise value:
 *   - Market Cap, Enterprise Value, Market Cap (Country R)
 *
 * Balance sheet items:
 *   - Total Assets (FY/LTM), Total Debt (FY/LTM), Total Equity (FY/LTM)
 *   - TBV (FY/LTM), Cash And Equivalents, Tangible Book Value
 *
 * Income statement items:
 *   - Total Revenues (FQ/FY/LTM/-1FY/5YAVG variants)
 *   - EBITDA (FQ/FY/LTM/-1FY/Adj/5YAVG variants)
 *   - EBIT (FQ/FY/LTM/-1FY/Adj/5YAVG variants)
 *   - Net Income variants, Operating Income, Gross Profit
 *   - Total Operating Expenses (LTM)
 *
 * Cash flow items:
 *   - CFO (Cash from Operations), CFI (Cash from Investing)
 *   - CFF (Cash from Financing), FCF (Free Cash Flow)
 *   - All with FQ/FY/LTM/-1FY/5YAVG variants
 *
 * Revenue estimates:
 *   - Revenues - Est Avg (NTM/FY1E), Revenues - Est Med (NTM/FY1E)
 *
 * RATIO COLUMNS (~70 columns - Pre-normalized):
 * ----------------------------------------------
 * Valuation ratios:
 *   - P/E (NTM/LTM/EST FY1 and extended time-series variants)
 *   - P/B (LTM/-1FY/5YAVG), P/TBV (LTM)
 *   - EV/EBITDA (LTM/NTM/EST FY1 and time-series variants)
 *   - EV/Sales (LTM/NTM/EST FY1 and time-series variants)
 *
 * Profitability ratios:
 *   - Return On Equity % (LTM/FY), Return on Assets (ROA) % (LTM/FY)
 *
 * Leverage & liquidity ratios:
 *   - Current Ratio (FY/LTM)
 *
 * Efficiency ratios:
 *   - Asset Turnover (FY/LTM)
 *
 * Risk scores:
 *   - Altman Z-Score (FY/FQ/LTM)
 *
 * PERCENTAGE COLUMNS (~25 columns - Already bounded):
 * ----------------------------------------------------
 * Margin metrics:
 *   - Gross Profit Margin % (FY/LTM)
 *   - Net Income Margin % (FY/LTM)
 *
 * Growth rates:
 *   - Total Revenues/CAGR (5Y FY)
 *   - Tot. Return %/CAGR (3Y/10Y)
 *   - Total Return (5Y/10Y/YTD)
 *   - Revenues - Est YoY % (FY1E)
 *
 * Volatility & risk metrics:
 *   - Volatility (1M/3M/6M/1Y)
 *   - Beta (1Y/2Y/5Y)
 *
 * Price change metrics:
 *   - Price Chg. % (1M/3M), 1-Day %
 *
 * Dividend yields:
 *   - Div Yield (Ind/LTM/TTM/NTM/5YAVGLTM/-1FYInd)
 *
 * COUNT COLUMNS (8 columns - Discrete integers):
 * -----------------------------------------------
 * Analyst coverage:
 *   - Price Target - #, Analyst Rating
 *   - # Strong Sell Ratings, # Strong Buys Ratings
 *   - # Hold Ratings, # Buys Ratings, # Sell Ratings
 *
 * Employment:
 *   - Total Employees (FY/FQ)
 *
 * Dividend history:
 *   - Dividend Streak
 *
 * CATEGORICAL COLUMNS (11 columns - Require encoding):
 * -----------------------------------------------------
 * Geographic & market classification:
 *   - Exchange, Sector, Industry
 *   - Region, Country, Trading Country
 *
 * Company classification:
 *   - Style Class (Value/Growth/Blend)
 *   - Size Class (Small/Mid/Large Cap)
 *
 * Status indicators:
 *   - Next Earnings (Status)
 *
 * Dividend information:
 *   - Dividend Record (Frequency), Dividend Record (Currency)
 *
 * PREPROCESSING DECISION RULES:
 * ==============================
 * Based on column_semantics.py functions:
 *
 * 1. get_winsorizable_columns():
 *    - Include: MARKET_VALUE, OTHER
 *    - Exclude: PRICE, RATIO, PERCENTAGE, COUNT
 *
 * 2. get_log_transform_columns():
 *    - Include: MARKET_VALUE (if not already log_* prefixed)
 *    - Exclude: PRICE, RATIO, PERCENTAGE, COUNT
 *
 * 3. get_scalable_columns():
 *    - Include: MARKET_VALUE, RATIO, PERCENTAGE, COUNT, OTHER
 *    - Exclude: PRICE (critical - must preserve original units)
 *
 * 4. Encoding:
 *    - Apply to: CATEGORICAL columns only
 *    - Methods: One-hot, label, or target encoding
 *
 * BUSINESS RATIONALE:
 * ===================
 * The core business metric requires price columns in original dollar units:
 *
 *   Mispricing = (Predicted_Target - Last_Price) / Last_Price
 *
 * This extends to:
 * - Historical prices: for momentum calculations like (price - price_1m_ago) / price_1m_ago
 * - 52-week bounds: for relative positioning (price - 52w_low) / (52w_high - 52w_low)
 * - EMAs: for technical analysis and trend identification
 *
 * Transforming these columns would corrupt valuation and momentum analysis.
 */
