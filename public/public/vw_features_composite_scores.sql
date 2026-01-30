create view vw_features_composite_scores
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, piotroski_f_score,
             eps_trajectory_score, dilution_score, quality_momentum_score, net_income_is_fq, net_income_is_ltm,
             net_income_is_fy, net_income_adj_ltm, normalized_ni_ltm, net_income_is_1fqfq, net_income_is_2fqfq,
             net_income_is_3fqfq, net_income_is_4fqfq, net_income_is_1fy, net_income_is_2fy, net_income_is_3fy,
             net_income_is_4fy, net_income_is_5yavgfq, net_income_is_5yavgltm, normalized_ni_5yavgfq,
             normalized_ni_5yavgltm, net_income_growth_yoy, net_income_margin_ltm, ni_adjustment_ratio,
             net_income_positive_years, earnings_quality_composite, net_income_qoq_growth, net_income_yoy_quarterly,
             net_income_vs_5y_avg, normalized_ni_vs_5y_avg)
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
       cs.piotroski_f_score,
       cs.eps_trajectory_score,
       cs.dilution_score,
       cs.quality_momentum_score,
       nic.net_income_is_fq,
       nic.net_income_is_ltm,
       nic.net_income_is_fy,
       nic.net_income_adj_ltm,
       nic.normalized_ni_ltm,
       nic.net_income_is_1fqfq,
       nic.net_income_is_2fqfq,
       nic.net_income_is_3fqfq,
       nic.net_income_is_4fqfq,
       nic.net_income_is_1fy,
       nic.net_income_is_2fy,
       nic.net_income_is_3fy,
       nic.net_income_is_4fy,
       nic.net_income_is_5yavgfq,
       nic.net_income_is_5yavgltm,
       nic.normalized_ni_5yavgfq,
       nic.normalized_ni_5yavgltm,
       nic.net_income_growth_yoy,
       nic.net_income_margin_ltm,
       nic.ni_adjustment_ratio,
       nic.net_income_positive_years,
       nic.earnings_quality_composite,
       nic.net_income_qoq_growth,
       nic.net_income_yoy_quarterly,
       nic.net_income_vs_5y_avg,
       nic.normalized_ni_vs_5y_avg
FROM vw_identifier_columns                         id
         LEFT JOIN calc_composite_scores()         cs(isin, piotroski_f_score, eps_trajectory_score, dilution_score,
                                                      quality_momentum_score) USING (isin)
         LEFT JOIN calc_net_income_comprehensive() nic(isin, net_income_is_fq, net_income_is_ltm, net_income_is_fy,
                                                       net_income_adj_ltm, normalized_ni_ltm, net_income_is_1fqfq,
                                                       net_income_is_2fqfq, net_income_is_3fqfq, net_income_is_4fqfq,
                                                       net_income_is_1fy, net_income_is_2fy, net_income_is_3fy,
                                                       net_income_is_4fy, net_income_is_5yavgfq, net_income_is_5yavgltm,
                                                       normalized_ni_5yavgfq, normalized_ni_5yavgltm,
                                                       net_income_growth_yoy, net_income_margin_ltm,
                                                       ni_adjustment_ratio, net_income_positive_years,
                                                       earnings_quality_composite, net_income_qoq_growth,
                                                       net_income_yoy_quarterly, net_income_vs_5y_avg,
                                                       normalized_ni_vs_5y_avg) USING (isin);

comment on view vw_features_composite_scores is 'Composite scoring metrics including Piotroski F-Score and earnings quality.
    Source functions: calc_composite_scores, calc_net_income_comprehensive';

alter table vw_features_composite_scores
    owner to postgres;

