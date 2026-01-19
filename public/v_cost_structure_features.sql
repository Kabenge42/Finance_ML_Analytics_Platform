create view v_cost_structure_features
            (ticker, sga_to_revenue_fq, sga_to_revenue_fy, sga_trend_yoy, sga_vs_5yavg, marketing_to_revenue_fq,
             marketing_to_revenue_fy, marketing_trend_yoy, marketing_vs_5yavg, operating_expense_ratio,
             cost_of_revenue_ratio, operating_leverage_score, cost_efficiency_trend)
as
SELECT ticker,
       sga_to_revenue_fq,
       sga_to_revenue_fy,
       sga_trend_yoy,
       sga_vs_5yavg,
       marketing_to_revenue_fq,
       marketing_to_revenue_fy,
       marketing_trend_yoy,
       marketing_vs_5yavg,
       operating_expense_ratio,
       cost_of_revenue_ratio,
       operating_leverage_score,
       cost_efficiency_trend
FROM calc_cost_structure_features() calc_cost_structure_features(ticker, sga_to_revenue_fq, sga_to_revenue_fy,
                                                                 sga_trend_yoy, sga_vs_5yavg, marketing_to_revenue_fq,
                                                                 marketing_to_revenue_fy, marketing_trend_yoy,
                                                                 marketing_vs_5yavg, operating_expense_ratio,
                                                                 cost_of_revenue_ratio, operating_leverage_score,
                                                                 cost_efficiency_trend);

alter table v_cost_structure_features
    owner to postgres;

