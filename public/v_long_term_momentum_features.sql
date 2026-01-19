create view v_long_term_momentum_features
            (ticker, price_momentum_qtd, price_momentum_3y, price_momentum_5y, momentum_acceleration_1y,
             momentum_acceleration_3y, long_term_trend_score, price_vs_3y_avg, price_vs_5y_avg, momentum_consistency,
             secular_trend_flag)
as
SELECT ticker,
       price_momentum_qtd,
       price_momentum_3y,
       price_momentum_5y,
       momentum_acceleration_1y,
       momentum_acceleration_3y,
       long_term_trend_score,
       price_vs_3y_avg,
       price_vs_5y_avg,
       momentum_consistency,
       secular_trend_flag
FROM calc_long_term_momentum_features() calc_long_term_momentum_features(ticker, price_momentum_qtd, price_momentum_3y,
                                                                         price_momentum_5y, momentum_acceleration_1y,
                                                                         momentum_acceleration_3y,
                                                                         long_term_trend_score, price_vs_3y_avg,
                                                                         price_vs_5y_avg, momentum_consistency,
                                                                         secular_trend_flag);

alter table v_long_term_momentum_features
    owner to postgres;

