CREATE TABLE IF NOT EXISTS equities
(
    "Ticker"          TEXT,
    "ISIN"            TEXT,
    "Name"            TEXT,
    "Description"     TEXT,
    "Sector"          TEXT,
    "Industry"        TEXT,
    "Region"          TEXT,
    "Country"         TEXT,
    "Trading Country" TEXT,
    "P/E (NTM)"       NUMERIC,
    "Last Price"      NUMERIC,
    "Price Target"    NUMERIC,
    "Market Cap"      NUMERIC
);