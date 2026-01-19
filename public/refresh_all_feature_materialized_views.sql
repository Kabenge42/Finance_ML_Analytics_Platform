create function refresh_all_feature_materialized_views() returns void
    language plpgsql
as
$$
BEGIN
    -- Refresh the unified enhanced features view
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_feature_registry_enhanced;

    -- Individual feature MVs (keep for granular queries)
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_revenue_quarterly_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_cost_structure_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_tangible_book_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_interest_income_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_long_term_momentum_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_beta_risk_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_revenue_estimate_consensus;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_unusual_items_features;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_enhanced_valuation_ratios;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_working_capital_deep_features;

    -- Category views
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_revenue_analysis;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_cost_analysis;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_asset_quality;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_interest_analysis;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_momentum_analysis;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_beta_risk;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_estimate_consensus;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_unusual_items;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_enhanced_valuation;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_working_capital;
END;
$$;

alter function refresh_all_feature_materialized_views() owner to postgres;

