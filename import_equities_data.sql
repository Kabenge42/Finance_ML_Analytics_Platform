-- Auto-generated from COLUMN_SCHEMA
COPY equities ("Ticker", "ISIN", "Name", "Description", "Sector", "Industry", "Region", "Country", "Trading Country",
               "P/E (NTM)", "Last Price", "Price Target", "Market Cap")
    FROM '/path/to/data.csv'
    WITH (FORMAT csv, HEADER true, NULL '');
