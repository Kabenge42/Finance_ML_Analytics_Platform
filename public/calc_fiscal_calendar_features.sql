create function calc_fiscal_calendar_features()
    returns TABLE
            (
                ticker                    text,
                days_since_last_report    integer,
                days_to_fy_end            integer,
                is_quarter_end_month      integer,
                is_fy_end_month           integer,
                earnings_season_flag      integer,
                pre_earnings_window       integer,
                post_earnings_window      integer,
                reporting_freshness_score numeric,
                fiscal_quarter_progress   numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                 AS ticker,
       -- Days Since Last Financial Report
       (CURRENT_DATE - "Income Statement Report Date")::INTEGER AS days_since_last_report,

       -- Days to Fiscal Year End
       ("FY End Date" - CURRENT_DATE)::INTEGER                  AS days_to_fy_end,

       -- Is Quarter End Month (current month is 3, 6, 9, or 12)
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12)
               THEN 1
           ELSE 0
           END                                                  AS is_quarter_end_month,

       -- Is Fiscal Year End Month
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM "FY End Date")
               THEN 1
           ELSE 0
           END                                                  AS is_fy_end_month,

       -- Earnings Season Flag (Jan/Feb, Apr/May, Jul/Aug, Oct/Nov)
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 2, 4, 5, 7, 8, 10, 11)
               THEN 1
           ELSE 0
           END                                                  AS earnings_season_flag,

       -- Pre-Earnings Window (within 14 days before earnings)
       CASE
           WHEN ("Next Earnings" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                  AS pre_earnings_window,

       -- Post-Earnings Window (within 7 days after report)
       CASE
           WHEN (CURRENT_DATE - "Income Statement Report Date") BETWEEN 0 AND 7
               THEN 1
           ELSE 0
           END                                                  AS post_earnings_window,

       -- Reporting Freshness Score (100 = just reported, decays over 90 days)
       GREATEST(0, LEAST(100,
                         100 - ((CURRENT_DATE - "Income Statement Report Date")::NUMERIC / 90.0 * 100)
                   ))                                           AS reporting_freshness_score,

       -- Fiscal Quarter Progress (0-1 based on fiscal month)
       CASE
           WHEN "Fiscal Month" IS NOT NULL
               THEN (("Fiscal Month" - 1) % 3 + 1) / 3.0
           END                                                  AS fiscal_quarter_progress

FROM postgres.public.equities;
$$;

alter function calc_fiscal_calendar_features() owner to postgres;

