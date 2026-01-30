create view vw_features_quality_risk
            (isin, ticker, name, industry, sector, trading_country, region, country, exchange, has_goodwill_impairment,
             has_asset_writedown, has_restructuring, goodwill_to_assets_pct, intangible_intensity,
             exceptional_items_to_ebitda, altman_z_score, altman_z_trend, current_ratio, quick_ratio, beta_1y, beta_5y,
             beta_spread, beta_trend, high_beta_flag, low_beta_flag, beta_stability_score, distress_risk_score,
             liquidity_stress_score, working_capital_trend, cash_runway_months, combined_distress_score,
             wc_deteriorating_flag, retained_earnings_growth, accumulated_deficit_flag, adequate_cash_buffer,
             goodwill_change_rate, restructuring_intensity, exceptional_items_frequency, merger_impact_ratio,
             non_operating_income_share, asset_sale_boost, accounting_quality_score, goodwill_impairment_ltm,
             asset_writedown_ltm, restructuring_ltm, has_goodwill_impairment_ltm, goodwill_impairment_frequency,
             asset_writedown_frequency, restructuring_frequency, exceptional_items_total_ltm,
             exceptional_items_to_ebitda_comp, quality_issues_count_5y, accounting_quality_score_comp)
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
       qf.has_goodwill_impairment,
       qf.has_asset_writedown,
       qf.has_restructuring,
       qf.goodwill_to_assets_pct,
       qf.intangible_intensity,
       qf.exceptional_items_to_ebitda,
       qf.altman_z_score,
       qf.altman_z_trend,
       qf.current_ratio,
       qf.quick_ratio,
       br.beta_1y,
       br.beta_5y,
       br.beta_spread,
       br.beta_trend,
       br.high_beta_flag,
       br.low_beta_flag,
       br.beta_stability_score,
       fdf.distress_risk_score,
       fdf.liquidity_stress_score,
       fdf.working_capital_trend,
       fdf.cash_runway_months,
       fdf.combined_distress_score,
       fdf.wc_deteriorating_flag,
       fdf.retained_earnings_growth,
       fdf.accumulated_deficit_flag,
       fdf.adequate_cash_buffer,
       aqf.goodwill_change_rate,
       aqf.restructuring_intensity,
       aqf.exceptional_items_frequency,
       aqf.merger_impact_ratio,
       aqf.non_operating_income_share,
       aqf.asset_sale_boost,
       aqf.accounting_quality_score,
       qfc.goodwill_impairment_ltm,
       qfc.asset_writedown_ltm,
       qfc.restructuring_ltm,
       qfc.has_goodwill_impairment_ltm,
       qfc.goodwill_impairment_frequency,
       qfc.asset_writedown_frequency,
       qfc.restructuring_frequency,
       qfc.exceptional_items_total_ltm,
       qfc.exceptional_items_to_ebitda AS exceptional_items_to_ebitda_comp,
       qfc.quality_issues_count_5y,
       qfc.accounting_quality_score    AS accounting_quality_score_comp
FROM vw_identifier_columns                               id
         LEFT JOIN calc_quality_features()               qf(isin, has_goodwill_impairment, has_asset_writedown,
                                                            has_restructuring, goodwill_to_assets_pct,
                                                            intangible_intensity, exceptional_items_to_ebitda,
                                                            altman_z_score, altman_z_trend, current_ratio, quick_ratio)
                   USING (isin)
         LEFT JOIN calc_beta_risk_features()             br(isin, beta_1y, beta_5y, beta_spread, beta_trend,
                                                            high_beta_flag, low_beta_flag, beta_stability_score)
                   USING (isin)
         LEFT JOIN calc_financial_distress_features()    fdf(isin, distress_risk_score, liquidity_stress_score,
                                                             working_capital_trend, cash_runway_months,
                                                             combined_distress_score, wc_deteriorating_flag,
                                                             retained_earnings_growth, accumulated_deficit_flag,
                                                             adequate_cash_buffer) USING (isin)
         LEFT JOIN calc_accounting_quality_features()    aqf(isin, goodwill_change_rate, restructuring_intensity,
                                                             exceptional_items_frequency, merger_impact_ratio,
                                                             non_operating_income_share, asset_sale_boost,
                                                             accounting_quality_score) USING (isin)
         LEFT JOIN calc_quality_features_comprehensive() qfc(isin, goodwill_impairment_ltm, asset_writedown_ltm,
                                                             restructuring_ltm, has_goodwill_impairment_ltm,
                                                             goodwill_impairment_frequency, asset_writedown_frequency,
                                                             restructuring_frequency, exceptional_items_total_ltm,
                                                             exceptional_items_to_ebitda, quality_issues_count_5y,
                                                             accounting_quality_score) USING (isin);

comment on view vw_features_quality_risk is 'Quality and risk metrics including accounting quality, financial distress, and beta analysis.
    Source functions: calc_quality_features, calc_beta_risk_features, calc_financial_distress_features,
    calc_accounting_quality_features, calc_quality_features_comprehensive';

alter table vw_features_quality_risk
    owner to postgres;

