create view vw_identifier_columns (isin, industry, trading_country, region, name, country, ticker, sector, exchange) as
SELECT "ISIN"            AS isin,
       "Industry"        AS industry,
       "Trading Country" AS trading_country,
       "Region"          AS region,
       "Name"            AS name,
       "Country"         AS country,
       "Ticker"          AS ticker,
       "Sector"          AS sector,
       "Exchange"        AS exchange
FROM equities e;

alter table vw_identifier_columns
    owner to postgres;

