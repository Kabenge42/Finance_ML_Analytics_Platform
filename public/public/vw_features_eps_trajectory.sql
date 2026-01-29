create view vw_features_eps_trajectory
            (isin, eps_qoq_growth, eps_yoy_quarterly, eps_positive_streak, eps_cagr_3y, eps_cagr_5y, eps_growth_accel,
             eps_vs_5y_avg, eps_improvement_count, eps_trajectory_score, eps_stability)
as
SELECT isin,
       eps_qoq_growth,
       eps_yoy_quarterly,
       eps_positive_streak,
       eps_cagr_3y,
       eps_cagr_5y,
       eps_growth_accel,
       eps_vs_5y_avg,
       eps_improvement_count,
       eps_trajectory_score,
       eps_stability
FROM calc_eps_trajectory_features() calc_eps_trajectory_features(isin, eps_qoq_growth, eps_yoy_quarterly,
                                                                 eps_positive_streak, eps_cagr_3y, eps_cagr_5y,
                                                                 eps_growth_accel, eps_vs_5y_avg, eps_improvement_count,
                                                                 eps_trajectory_score, eps_stability);

alter table vw_features_eps_trajectory
    owner to postgres;

