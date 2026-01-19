create view v_working_capital_deep_features
            (ticker, current_assets_ltm, current_liabilities_ltm, net_working_capital, working_capital_to_revenue,
             working_capital_to_assets, current_ratio, quick_ratio, cash_ratio, defensive_interval,
             working_capital_turnover, liquidity_score, working_capital_efficiency)
as
SELECT ticker,
       current_assets_ltm,
       current_liabilities_ltm,
       net_working_capital,
       working_capital_to_revenue,
       working_capital_to_assets,
       current_ratio,
       quick_ratio,
       cash_ratio,
       defensive_interval,
       working_capital_turnover,
       liquidity_score,
       working_capital_efficiency
FROM calc_working_capital_deep_features() calc_working_capital_deep_features(ticker, current_assets_ltm,
                                                                             current_liabilities_ltm,
                                                                             net_working_capital,
                                                                             working_capital_to_revenue,
                                                                             working_capital_to_assets, current_ratio,
                                                                             quick_ratio, cash_ratio,
                                                                             defensive_interval,
                                                                             working_capital_turnover, liquidity_score,
                                                                             working_capital_efficiency);

alter table v_working_capital_deep_features
    owner to postgres;

