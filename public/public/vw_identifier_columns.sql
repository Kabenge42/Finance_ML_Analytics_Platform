create view vw_identifier_columns (isin, ticker, name, region, country, trading_country, exchange, sector, industry) as
SELECT "ISIN"            AS isin,
       "Ticker"          AS ticker,
       "Name"            AS name,
       "Region"          AS region,
       "Country"         AS country,
       "Trading Country" AS trading_country,
       "Exchange"        AS exchange,
       "Sector"          AS sector,
       "Industry"        AS industry
FROM equities e;

alter table vw_identifier_columns
    owner to postgres;

