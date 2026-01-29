create view vw_features_composite_scores
            (isin, piotroski_f_score, eps_trajectory_score, dilution_score, quality_momentum_score) as
SELECT isin,
       piotroski_f_score,
       eps_trajectory_score,
       dilution_score,
       quality_momentum_score
FROM calc_composite_scores() calc_composite_scores(isin, piotroski_f_score, eps_trajectory_score, dilution_score,
                                                   quality_momentum_score);

alter table vw_features_composite_scores
    owner to postgres;

