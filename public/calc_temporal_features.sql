create function calc_temporal_features()
    returns TABLE
            (
                ticker                  text,
                fiscal_quarter          integer,
                fiscal_month            integer,
                fiscal_year             integer,
                days_to_earnings        integer,
                earnings_report_recency integer,
                reporting_lag           numeric,
                fiscal_year_progress    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                        AS ticker,
       "Fiscal Quarter"                                AS fiscal_quarter,
       "Fiscal Month"                                  AS fiscal_month,
       "Fiscal Year"                                   AS fiscal_year,
       -- Direct date subtraction returns INTEGER (no EXTRACT needed)
       ("Next Earnings" - CURRENT_DATE)                AS days_to_earnings,
       (CURRENT_DATE - "Income Statement Report Date") AS earnings_report_recency,
       "Reporting Lag"                                 AS reporting_lag,
       "Fiscal Month" / 12.0                           AS fiscal_year_progress
FROM postgres.public.equities;
$$;

alter function calc_temporal_features() owner to postgres;

