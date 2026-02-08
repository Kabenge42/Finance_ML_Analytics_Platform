create table analytics.temporal_distributions
(
    fiscal_quarter            text,
    fiscal_month              text,
    fiscal_year               text,
    days_to_earnings          text,
    earnings_report_recency   text,
    reporting_lag             text,
    fiscal_year_progress      text,
    days_since_last_report    text,
    days_to_fy_end            text,
    is_quarter_end_month      text,
    is_fy_end_month           text,
    earnings_season_flag      text,
    pre_earnings_window       text,
    post_earnings_window      text,
    reporting_freshness_score text,
    fiscal_quarter_progress   text
);

alter table analytics.temporal_distributions
    owner to postgres;

