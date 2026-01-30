create view vw_features_valuation_ratios
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, p_e_ratio, p_b_ratio,
             ev_ebitda_ratio, ev_sales_ratio, dividend_yield, peg_ratio, ev_sales_trend_1y, ev_ebitda_momentum,
             p_e_momentum_yoy, p_e_momentum_qoq, ev_sales_vs_3y_avg, ev_ebitda_vs_3y_avg, p_e_vs_3y_avg,
             ev_sales_forward_discount, ev_ebitda_forward_discount, p_e_forward_discount, p_b_vs_5y_avg,
             ev_sales_qoq_1q, ev_sales_qoq_2q, ev_sales_qoq_3q, ev_sales_qoq_4q, p_e_vs_5y_avg, p_e_percentile_proxy,
             valuation_mean_reversion, ev_ebitda_qoq_trend, p_b_momentum_yoy, valuation_compression, forward_pe_premium,
             tangible_book_value_fy, tangible_book_value_ltm, tangible_book_per_share, price_to_tangible_book,
             tangible_equity_ratio, intangibles_to_equity, goodwill_to_equity, tangible_asset_quality, tbv_yoy_growth,
             tbv_vs_calculated)
as
SELECT id.isin,
       id.ticker,
       id.name,
       id.industry,
       id.sector,
       id.trading_country,
       id.region,
       id.country,
       id.exchange,
       vf.p_e_ratio,
       vf.p_b_ratio,
       vf.ev_ebitda_ratio,
       vf.ev_sales_ratio,
       vf.dividend_yield,
       vf.peg_ratio,
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
       evt.ev_sales_qoq_1q,
       evt.ev_sales_qoq_2q,
       evt.ev_sales_qoq_3q,
       evt.ev_sales_qoq_4q,
       evt.p_e_vs_5y_avg,
       evt.p_e_percentile_proxy,
       evt.valuation_mean_reversion,
       evt.ev_ebitda_qoq_trend,
       evt.p_b_momentum_yoy,
       evt.valuation_compression,
       evt.forward_pe_premium,
       tb.tangible_book_value_fy,
       tb.tangible_book_value_ltm,
       tb.tangible_book_per_share,
       tb.price_to_tangible_book,
       tb.tangible_equity_ratio,
       tb.intangibles_to_equity,
       tb.goodwill_to_equity,
       tb.tangible_asset_quality,
       tb.tbv_yoy_growth,
       tb.tbv_vs_calculated
FROM vw_identifier_columns                              id
         LEFT JOIN calc_valuation_features()            vf(isin, p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio,
                                                           dividend_yield, peg_ratio) USING (isin)
         LEFT JOIN calc_valuation_timeseries_features() vts(isin, ev_sales_trend_1y, ev_ebitda_momentum,
                                                            p_e_momentum_yoy, p_e_momentum_qoq, ev_sales_vs_3y_avg,
                                                            ev_ebitda_vs_3y_avg, p_e_vs_3y_avg,
                                                            ev_sales_forward_discount, ev_ebitda_forward_discount,
                                                            p_e_forward_discount, p_b_vs_5y_avg) USING (isin)
         LEFT JOIN calc_extended_valuation_timeseries() evt(isin, ev_sales_qoq_1q, ev_sales_qoq_2q, ev_sales_qoq_3q,
                                                            ev_sales_qoq_4q, p_e_vs_5y_avg, p_e_percentile_proxy,
                                                            valuation_mean_reversion, ev_ebitda_qoq_trend,
                                                            p_b_momentum_yoy, valuation_compression, forward_pe_premium)
                   USING (isin)
         LEFT JOIN calc_tangible_book_features()        tb(isin, tangible_book_value_fy, tangible_book_value_ltm,
                                                           tangible_book_per_share, price_to_tangible_book,
                                                           tangible_equity_ratio, intangibles_to_equity,
                                                           goodwill_to_equity, tangible_asset_quality, tbv_yoy_growth,
                                                           tbv_vs_calculated) USING (isin);

comment on view vw_features_valuation_ratios is 'Valuation metrics including P/E, P/B, EV/EBITDA, tangible book value, and timeseries analysis.
    Source functions: calc_valuation_features, calc_valuation_timeseries_features,
    calc_extended_valuation_timeseries, calc_tangible_book_features';

alter table vw_features_valuation_ratios
    owner to postgres;

