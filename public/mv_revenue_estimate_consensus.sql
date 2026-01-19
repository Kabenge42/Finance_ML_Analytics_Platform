create materialized view mv_revenue_estimate_consensus as
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
FROM v_revenue_estimate_consensus;

alter materialized view mv_revenue_estimate_consensus owner to postgres;

create index idx_mv_rev_est_ticker
    on mv_revenue_estimate_consensus (ticker);

create index idx_mv_rev_est_conf
    on mv_revenue_estimate_consensus (consensus_confidence desc);

