create view vw_features_temporal
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, fiscal_quarter,
             fiscal_month, fiscal_year, days_to_earnings, earnings_report_recency, reporting_lag, fiscal_year_progress,
             days_since_last_report, days_to_fy_end, is_quarter_end_month, is_fy_end_month, earnings_season_flag,
             pre_earnings_window, post_earnings_window, reporting_freshness_score, fiscal_quarter_progress)
as
SELECT id.isin,
       id.ticker,
       id.name,
       id.industry,
       id.sector,
       id.trading_country,
       id.region,
       id.country,
       id.exchange,
       tf.fiscal_quarter,
       tf.fiscal_month,
       tf.fiscal_year,
       tf.days_to_earnings,
       tf.earnings_report_recency,
       tf.reporting_lag,
       tf.fiscal_year_progress,
       fcf.days_since_last_report,
       fcf.days_to_fy_end,
       fcf.is_quarter_end_month,
       fcf.is_fy_end_month,
       fcf.earnings_season_flag,
       fcf.pre_earnings_window,
       fcf.post_earnings_window,
       fcf.reporting_freshness_score,
       fcf.fiscal_quarter_progress
FROM vw_identifier_columns                         id
         LEFT JOIN calc_temporal_features()        tf(isin, fiscal_quarter, fiscal_month, fiscal_year, days_to_earnings,
                                                      earnings_report_recency, reporting_lag, fiscal_year_progress)
                   USING (isin)
         LEFT JOIN calc_fiscal_calendar_features() fcf(isin, days_since_last_report, days_to_fy_end,
                                                       is_quarter_end_month, is_fy_end_month, earnings_season_flag,
                                                       pre_earnings_window, post_earnings_window,
                                                       reporting_freshness_score, fiscal_quarter_progress) USING (isin);

comment on view vw_features_temporal is 'Temporal and fiscal calendar features for earnings timing and seasonality.
    Source functions: calc_temporal_features, calc_fiscal_calendar_features';

alter table vw_features_temporal
    owner to postgres;

