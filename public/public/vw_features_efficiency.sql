create view vw_features_efficiency
            (isin, asset_turnover, inventory_turnover, receivables_days, working_capital_turns) as
SELECT isin,
       asset_turnover,
       inventory_turnover,
       receivables_days,
       working_capital_turns
FROM calc_efficiency_ratios() calc_efficiency_ratios(isin, asset_turnover, inventory_turnover, receivables_days,
                                                     working_capital_turns);

alter table vw_features_efficiency
    owner to postgres;

