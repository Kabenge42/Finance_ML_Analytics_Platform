create view vw_features_price_target_dynamics
            (isin, pt_momentum_1w, pt_momentum_1m, pt_momentum_3m, pt_momentum_6m, pt_momentum_1y,
             pt_median_momentum_1m, pt_median_momentum_3m, pt_acceleration_short, pt_acceleration_long,
             pt_consensus_convergence, analyst_coverage_change_1m, analyst_coverage_change_3m,
             analyst_coverage_change_1y, pt_vs_price_momentum, analyst_coverage_trend)
as
SELECT isin,
       pt_momentum_1w,
       pt_momentum_1m,
       pt_momentum_3m,
       pt_momentum_6m,
       pt_momentum_1y,
       pt_median_momentum_1m,
       pt_median_momentum_3m,
       pt_acceleration_short,
       pt_acceleration_long,
       pt_consensus_convergence,
       analyst_coverage_change_1m,
       analyst_coverage_change_3m,
       analyst_coverage_change_1y,
       pt_vs_price_momentum,
       analyst_coverage_trend
FROM calc_price_target_dynamics() calc_price_target_dynamics(isin, pt_momentum_1w, pt_momentum_1m, pt_momentum_3m,
                                                             pt_momentum_6m, pt_momentum_1y, pt_median_momentum_1m,
                                                             pt_median_momentum_3m, pt_acceleration_short,
                                                             pt_acceleration_long, pt_consensus_convergence,
                                                             analyst_coverage_change_1m, analyst_coverage_change_3m,
                                                             analyst_coverage_change_1y, pt_vs_price_momentum,
                                                             analyst_coverage_trend);

alter table vw_features_price_target_dynamics
    owner to postgres;

