create view v_revenue_estimate_consensus
            (ticker, revenue_est_avg_ntm, revenue_est_med_ntm, revenue_est_avg_fy1e, revenue_est_med_fy1e,
             estimate_skew_ntm, estimate_skew_fy1e, consensus_confidence, upside_to_consensus, estimate_vs_actual_ltm,
             forward_revenue_growth, revenue_beat_history)
as
SELECT ticker,
       revenue_est_avg_ntm,
       revenue_est_med_ntm,
       revenue_est_avg_fy1e,
       revenue_est_med_fy1e,
       estimate_skew_ntm,
       estimate_skew_fy1e,
       consensus_confidence,
       upside_to_consensus,
       estimate_vs_actual_ltm,
       forward_revenue_growth,
       revenue_beat_history
FROM calc_revenue_estimate_consensus() calc_revenue_estimate_consensus(ticker, revenue_est_avg_ntm, revenue_est_med_ntm,
                                                                       revenue_est_avg_fy1e, revenue_est_med_fy1e,
                                                                       estimate_skew_ntm, estimate_skew_fy1e,
                                                                       consensus_confidence, upside_to_consensus,
                                                                       estimate_vs_actual_ltm, forward_revenue_growth,
                                                                       revenue_beat_history);

alter table v_revenue_estimate_consensus
    owner to postgres;

