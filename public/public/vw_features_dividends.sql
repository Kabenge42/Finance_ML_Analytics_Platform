create view vw_features_dividends
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, dividend_streak,
             dividend_yield_ltm, dividend_yield_ntm, dividend_payout_ratio, fcf_dividend_coverage, buyback_yield,
             total_shareholder_yield, dividend_growth_expectation, days_since_ex_date, days_to_payment,
             dividend_announced_flag, ex_date_approaching_flag, dividend_frequency_score, dividend_consistency,
             recent_dividend_change, dividend_yield_vs_5y_avg, div_yield_ltm, div_yield_ntm, div_yield_ind,
             div_yield_1fy_ind, div_yield_5y_avg, div_yield_vs_5y_avg, div_yield_growth_expected, dividend_streak_comp,
             high_yield_flag, sustainable_dividend_flag)
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
       df.dividend_streak,
       df.dividend_yield_ltm,
       df.dividend_yield_ntm,
       df.dividend_payout_ratio,
       df.fcf_dividend_coverage,
       df.buyback_yield,
       df.total_shareholder_yield,
       df.dividend_growth_expectation,
       dt.days_since_ex_date,
       dt.days_to_payment,
       dt.dividend_announced_flag,
       dt.ex_date_approaching_flag,
       dt.dividend_frequency_score,
       dt.dividend_consistency,
       dt.recent_dividend_change,
       dt.dividend_yield_vs_5y_avg,
       dyc.div_yield_ltm,
       dyc.div_yield_ntm,
       dyc.div_yield_ind,
       dyc.div_yield_1fy_ind,
       dyc.div_yield_5y_avg,
       dyc.div_yield_vs_5y_avg,
       dyc.div_yield_growth_expected,
       dyc.dividend_streak AS dividend_streak_comp,
       dyc.high_yield_flag,
       dyc.sustainable_dividend_flag
FROM vw_identifier_columns                             id
         LEFT JOIN calc_dividend_features()            df(isin, dividend_streak, dividend_yield_ltm, dividend_yield_ntm,
                                                          dividend_payout_ratio, fcf_dividend_coverage, buyback_yield,
                                                          total_shareholder_yield, dividend_growth_expectation)
                   USING (isin)
         LEFT JOIN calc_dividend_timing()              dt(isin, days_since_ex_date, days_to_payment,
                                                          dividend_announced_flag, ex_date_approaching_flag,
                                                          dividend_frequency_score, dividend_consistency,
                                                          recent_dividend_change, dividend_yield_vs_5y_avg) USING (isin)
         LEFT JOIN calc_dividend_yield_comprehensive() dyc(isin, div_yield_ltm, div_yield_ntm, div_yield_ind,
                                                           div_yield_1fy_ind, div_yield_5y_avg, div_yield_vs_5y_avg,
                                                           div_yield_growth_expected, dividend_streak, high_yield_flag,
                                                           sustainable_dividend_flag) USING (isin);

comment on view vw_features_dividends is 'Dividend metrics including yield, payout ratios, timing, and sustainability.
    Source functions: calc_dividend_features, calc_dividend_timing, calc_dividend_yield_comprehensive';

alter table vw_features_dividends
    owner to postgres;

