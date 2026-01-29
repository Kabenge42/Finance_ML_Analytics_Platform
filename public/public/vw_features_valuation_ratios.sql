create view vw_features_valuation_ratios
            (isin, p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio, dividend_yield, peg_ratio,
             tangible_book_value_fy, tangible_book_value_ltm, tangible_book_per_share, price_to_tangible_book,
             tangible_equity_ratio, intangibles_to_equity, goodwill_to_equity, tangible_asset_quality, tbv_yoy_growth,
             tbv_vs_calculated)
as
SELECT isin,
       vf.p_e_ratio,
       vf.p_b_ratio,
       vf.ev_ebitda_ratio,
       vf.ev_sales_ratio,
       vf.dividend_yield,
       vf.peg_ratio,
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
FROM calc_valuation_features()                   vf(isin, p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio,
                                                    dividend_yield, peg_ratio)
         FULL JOIN calc_tangible_book_features() tb(isin, tangible_book_value_fy, tangible_book_value_ltm,
                                                    tangible_book_per_share, price_to_tangible_book,
                                                    tangible_equity_ratio, intangibles_to_equity, goodwill_to_equity,
                                                    tangible_asset_quality, tbv_yoy_growth, tbv_vs_calculated)
                   USING (isin);

alter table vw_features_valuation_ratios
    owner to postgres;

