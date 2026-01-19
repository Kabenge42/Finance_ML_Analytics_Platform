create view v_revenue_quarterly_features
            (ticker, revenue_fq, revenue_ltm, revenue_fy, revenue_1fy, revenue_5yavg_fq, revenue_5yavg_ltm,
             revenue_fq_vs_5yavg, revenue_ltm_vs_5yavg, revenue_qoq_growth, revenue_yoy_growth,
             revenue_quarterly_run_rate, revenue_seasonality_factor)
as
SELECT ticker,
       revenue_fq,
       revenue_ltm,
       revenue_fy,
       revenue_1fy,
       revenue_5yavg_fq,
       revenue_5yavg_ltm,
       revenue_fq_vs_5yavg,
       revenue_ltm_vs_5yavg,
       revenue_qoq_growth,
       revenue_yoy_growth,
       revenue_quarterly_run_rate,
       revenue_seasonality_factor
FROM calc_revenue_quarterly_features() calc_revenue_quarterly_features(ticker, revenue_fq, revenue_ltm, revenue_fy,
                                                                       revenue_1fy, revenue_5yavg_fq, revenue_5yavg_ltm,
                                                                       revenue_fq_vs_5yavg, revenue_ltm_vs_5yavg,
                                                                       revenue_qoq_growth, revenue_yoy_growth,
                                                                       revenue_quarterly_run_rate,
                                                                       revenue_seasonality_factor);

alter table v_revenue_quarterly_features
    owner to postgres;

