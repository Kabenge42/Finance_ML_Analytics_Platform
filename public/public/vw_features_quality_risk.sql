create view vw_features_quality_risk
            (isin, has_goodwill_impairment, has_asset_writedown, has_restructuring, goodwill_to_assets_pct,
             intangible_intensity, exceptional_items_to_ebitda, altman_z_score, altman_z_trend, current_ratio,
             quick_ratio, beta_1y, beta_5y, beta_spread, beta_trend, high_beta_flag, low_beta_flag,
             beta_stability_score)
as
SELECT isin,
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
       br.beta_stability_score
FROM calc_quality_features()                 qf(isin, has_goodwill_impairment, has_asset_writedown, has_restructuring,
                                                goodwill_to_assets_pct, intangible_intensity,
                                                exceptional_items_to_ebitda, altman_z_score, altman_z_trend,
                                                current_ratio, quick_ratio)
         FULL JOIN calc_beta_risk_features() br(isin, beta_1y, beta_5y, beta_spread, beta_trend, high_beta_flag,
                                                low_beta_flag, beta_stability_score) USING (isin);

alter table vw_features_quality_risk
    owner to postgres;

