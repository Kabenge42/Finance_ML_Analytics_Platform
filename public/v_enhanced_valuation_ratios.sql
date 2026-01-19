create view v_enhanced_valuation_ratios
            (ticker, forward_pe, trailing_pe, pe_forward_discount, peg_ratio, peg_adjusted, peg_forward, pe_to_growth,
             ev_to_fcf, earnings_yield, fcf_yield, shareholder_yield_total, valuation_composite_score)
as
SELECT ticker,
       forward_pe,
       trailing_pe,
       pe_forward_discount,
       peg_ratio,
       peg_adjusted,
       peg_forward,
       pe_to_growth,
       ev_to_fcf,
       earnings_yield,
       fcf_yield,
       shareholder_yield_total,
       valuation_composite_score
FROM calc_enhanced_valuation_ratios() calc_enhanced_valuation_ratios(ticker, forward_pe, trailing_pe,
                                                                     pe_forward_discount, peg_ratio, peg_adjusted,
                                                                     peg_forward, pe_to_growth, ev_to_fcf,
                                                                     earnings_yield, fcf_yield, shareholder_yield_total,
                                                                     valuation_composite_score);

alter table v_enhanced_valuation_ratios
    owner to postgres;

