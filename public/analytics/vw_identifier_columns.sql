create view analytics.vw_identifier_columns
            (isin, ticker, name, region, country, trading_country, exchange, sector, industry,
             dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
             next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
             dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
             dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
             next_fy_end_date, next_income_statement_report_date, reference_date)
as
SELECT isin,
       ticker,
       name,
       region,
       country,
       trading_country,
       exchange,
       sector,
       industry,
       dividend_record_frequency,
       earnings_report_frequency,
       fy_end,
       next_earnings_report,
       next_earnings_status,
       next_earnings_when,
       next_fiscal_quarter,
       reporting_interval,
       size_class,
       style_class,
       unit,
       dividend_record_announce_date,
       dividend_record_ex_date,
       dividend_record_payable_date,
       dividend_record_record_date,
       fy_end_date,
       income_statement_report_date,
       last_updated,
       next_earnings,
       next_fy_end_date,
       next_income_statement_report_date,
       reference_date
FROM vw_identifier_columns vic
WHERE next_earnings IS NOT NULL;

alter table analytics.vw_identifier_columns
    owner to postgres;

