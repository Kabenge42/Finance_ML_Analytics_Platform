create view vw_features_valuation_timeseries
            (isin, ev_sales_trend_1y, ev_ebitda_momentum, p_e_momentum_yoy, p_e_momentum_qoq, ev_sales_vs_3y_avg,
             ev_ebitda_vs_3y_avg, p_e_vs_3y_avg, ev_sales_forward_discount, ev_ebitda_forward_discount,
             p_e_forward_discount, p_b_vs_5y_avg, ev_sales_qoq_1q, ev_sales_qoq_2q, ev_sales_qoq_3q, ev_sales_qoq_4q,
             p_e_vs_5y_avg, p_e_percentile_proxy, valuation_mean_reversion, ev_ebitda_qoq_trend, p_b_momentum_yoy,
             valuation_compression, forward_pe_premium)
as
SELECT isin,
       vts.ev_sales_trend_1y,
       vts.ev_ebitda_momentum,
       vts.p_e_momentum_yoy,
       vts.p_e_momentum_qoq,
       vts.ev_sales_vs_3y_avg,
       vts.ev_ebitda_vs_3y_avg,
       vts.p_e_vs_3y_avg,
       vts.ev_sales_forward_discount,
       vts.ev_ebitda_forward_discount,
       vts.p_e_forward_discount,
       vts.p_b_vs_5y_avg,
       evts.ev_sales_qoq_1q,
       evts.ev_sales_qoq_2q,
       evts.ev_sales_qoq_3q,
       evts.ev_sales_qoq_4q,
       evts.p_e_vs_5y_avg,
       evts.p_e_percentile_proxy,
       evts.valuation_mean_reversion,
       evts.ev_ebitda_qoq_trend,
       evts.p_b_momentum_yoy,
       evts.valuation_compression,
       evts.forward_pe_premium
FROM calc_valuation_timeseries_features()               vts(isin, ev_sales_trend_1y, ev_ebitda_momentum,
                                                            p_e_momentum_yoy, p_e_momentum_qoq, ev_sales_vs_3y_avg,
                                                            ev_ebitda_vs_3y_avg, p_e_vs_3y_avg,
                                                            ev_sales_forward_discount, ev_ebitda_forward_discount,
                                                            p_e_forward_discount, p_b_vs_5y_avg)
         FULL JOIN calc_extended_valuation_timeseries() evts(isin, ev_sales_qoq_1q, ev_sales_qoq_2q, ev_sales_qoq_3q,
                                                             ev_sales_qoq_4q, p_e_vs_5y_avg, p_e_percentile_proxy,
                                                             valuation_mean_reversion, ev_ebitda_qoq_trend,
                                                             p_b_momentum_yoy, valuation_compression,
                                                             forward_pe_premium) USING (isin);

alter table vw_features_valuation_timeseries
    owner to postgres;

