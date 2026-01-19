create view v_unusual_items_features
            (ticker, other_unusual_items_ltm, total_unusual_items, unusual_to_revenue_ratio, unusual_to_ebitda_ratio,
             unusual_to_net_income_ratio, clean_earnings_flag, recurring_unusual_flag, earnings_noise_score,
             quality_adjusted_ni, exceptional_items_impact)
as
SELECT ticker,
       other_unusual_items_ltm,
       total_unusual_items,
       unusual_to_revenue_ratio,
       unusual_to_ebitda_ratio,
       unusual_to_net_income_ratio,
       clean_earnings_flag,
       recurring_unusual_flag,
       earnings_noise_score,
       quality_adjusted_ni,
       exceptional_items_impact
FROM calc_unusual_items_features() calc_unusual_items_features(ticker, other_unusual_items_ltm, total_unusual_items,
                                                               unusual_to_revenue_ratio, unusual_to_ebitda_ratio,
                                                               unusual_to_net_income_ratio, clean_earnings_flag,
                                                               recurring_unusual_flag, earnings_noise_score,
                                                               quality_adjusted_ni, exceptional_items_impact);

alter table v_unusual_items_features
    owner to postgres;

